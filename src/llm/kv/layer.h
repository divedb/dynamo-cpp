// SPDX-License-Identifier: Apache-2.0
//
// KV cache tensor layers and block copies — Dynamo's lib/llm kv/layer.rs.
//
// A KvLayer is one attention layer's KV cache: a 5D tensor over one Storage
// slab, laid out KvFirst  [2, blocks, block_size, heads/tp, head_size] or
// BlockFirst [blocks, 2, block_size, heads/tp, head_size]. KvBlockStorage
// bundles the per-layer slabs for a whole model; CopyStream moves selected
// blocks between two such bundles (GPU↔host offload/onboard) through the
// block-copy ABI, layer by layer so transfers can chase the forward pass.
//
// Deviations from Rust v0.1.0 (commented at the sites):
//  - copy_stream_destroy is called on destruction (Rust never frees it);
//  - host↔host block copies are allowed (Rust rejects Pinned→Pinned and
//    System entirely — it always has CUDA and never runs host-only).

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "llm/kv/storage.h"

namespace dynamo::llm::kv {

enum class KvLayout {
  KvFirst,     // [kv, block, block_size, head, head_dim]
  BlockFirst,  // [block, kv, block_size, head, head_dim]
};

struct KvModelDetails {
  size_t number_of_layers = 0;
  size_t number_of_heads = 0;
  size_t head_size = 0;
  DType dtype = DType::F32;

  bool operator==(const KvModelDetails&) const = default;

  size_t elements_per_token_per_layer() const { return 2 * number_of_heads * head_size; }
  size_t bytes_per_token_per_layer() const {
    return elements_per_token_per_layer() * dtype_size(dtype);
  }
};

struct KvBlockDetails {
  KvLayout layout = KvLayout::KvFirst;
  size_t block_size = 0;
  size_t tp_rank = 0;
  size_t tp_size = 1;
  KvModelDetails model_details;

  /// Throws std::invalid_argument when tp settings are inconsistent.
  void validate() const;

  bool is_compatible(const KvBlockDetails& other) const {
    return layout == other.layout && block_size == other.block_size &&
           tp_size == other.tp_size && model_details == other.model_details;
  }

  size_t bytes_per_token_block_per_layer() const {
    return (model_details.bytes_per_token_per_layer() * block_size) / tp_size;
  }

  /// The non-contiguous dimension above the block dimension (2 for KvFirst —
  /// the kv axis; 1 for BlockFirst — blocks are outermost).
  size_t prefix_dim() const { return layout == KvLayout::KvFirst ? 2 : 1; }

  /// The contiguous span below the block dimension, in elements.
  size_t suffix_dim() const {
    size_t suffix = block_size * (model_details.number_of_heads / tp_size) *
                    model_details.head_size;
    return layout == KvLayout::KvFirst ? suffix : 2 * suffix;
  }

  size_t elem_size() const { return dtype_size(model_details.dtype); }
};

/// One attention layer's KV cache tensor.
class KvLayer {
 public:
  /// Wraps an existing slab (bring-your-own memory). Throws when the slab is
  /// too small or the tp settings are inconsistent.
  KvLayer(KvLayout layout, OwnedStorage storage, size_t number_of_blocks, size_t block_size,
          size_t number_of_heads, size_t head_size, DType dtype, size_t tp_size = 1,
          size_t tp_rank = 0);

  static KvLayer from_storage(const KvBlockDetails& details, size_t number_of_blocks,
                              OwnedStorage storage);

  std::array<size_t, 5> layer_shape() const;
  TensorView<5> view() const { return TensorView<5>(storage_, layer_shape(), dtype_); }

  KvLayout layout() const { return layout_; }
  size_t number_of_blocks() const { return number_of_blocks_; }
  size_t block_size() const { return block_size_; }
  size_t number_of_heads() const { return number_of_heads_; }
  size_t head_size() const { return head_size_; }
  DType dtype() const { return dtype_; }
  size_t tp_size() const { return tp_size_; }
  size_t tp_rank() const { return tp_rank_; }
  const OwnedStorage& storage() const { return storage_; }

  /// Copies the named blocks of this layer into another layer (blocking,
  /// one-shot; CopyStream is the staged multi-layer variant).
  void copy_blocks_to(const std::vector<size_t>& src_block_ids, KvLayer& dst,
                      const std::vector<size_t>& dst_block_ids) const;

 private:
  KvLayout layout_;
  OwnedStorage storage_;
  size_t number_of_blocks_;
  size_t block_size_;
  size_t number_of_heads_;
  size_t head_size_;
  DType dtype_;
  size_t tp_size_;
  size_t tp_rank_;
};

/// All layers of one model's KV cache on one storage tier.
class KvBlockStorage {
 public:
  /// Validates and adopts pre-built layers (must agree on kind/shape).
  static KvBlockStorage from_layers(std::vector<KvLayer> layers);

  /// Allocates number_of_layers fresh slabs on the given tier.
  static KvBlockStorage allocate(size_t number_of_blocks, const KvBlockDetails& details,
                                 StorageKind kind);

  const KvLayer& layer(size_t index) const;
  KvLayer& layer(size_t index);
  size_t layer_count() const { return layers_.size(); }
  size_t number_of_blocks() const { return number_of_blocks_; }
  StorageKind storage_kind() const { return storage_kind_; }
  const KvBlockDetails& block_details() const { return block_details_; }

 private:
  KvBlockStorage(KvBlockDetails details, StorageKind kind, size_t number_of_blocks,
                 std::vector<KvLayer> layers)
      : block_details_(std::move(details)),
        storage_kind_(kind),
        number_of_blocks_(number_of_blocks),
        layers_(std::move(layers)) {}

  KvBlockDetails block_details_;
  StorageKind storage_kind_;
  size_t number_of_blocks_;
  std::vector<KvLayer> layers_;
};

/// Precomputed copy geometry between two compatible block storages (one per
/// direction; build once, reuse across CopyStream cycles).
struct CopyStreamBlockMap {
  std::vector<uint64_t> src_layer_ptrs;
  std::vector<uint64_t> dst_layer_ptrs;
  int prefix_dim = 0;
  int src_block_dim = 0;
  int dst_block_dim = 0;
  int suffix_dim = 0;
  int elem_size = 0;

  static std::shared_ptr<const CopyStreamBlockMap> create(const KvBlockStorage& src,
                                                          const KvBlockStorage& dst);
};

/// Stateful staged block transfer: stage the block map and the block-id
/// pairs, then trigger layers individually (chasing the forward pass) or all
/// at once; sync() blocks until the backend finished.
class CopyStream {
 public:
  CopyStream(size_t max_num_layers, size_t max_num_blocks);
  ~CopyStream();
  CopyStream(CopyStream&&) noexcept;
  CopyStream& operator=(CopyStream&&) noexcept;
  CopyStream(const CopyStream&) = delete;
  CopyStream& operator=(const CopyStream&) = delete;

  void prepare_block_map(std::shared_ptr<const CopyStreamBlockMap> map);
  void prepare_block_ids(std::vector<int> src_block_ids, std::vector<int> dst_block_ids);

  /// Copies one layer's staged blocks (idempotent per cycle — a doorbell
  /// keeps double triggers harmless).
  void trigger_layer(size_t layer);
  void trigger_all_layers();
  size_t layer_count() const;

  /// Blocks until all triggered copies completed.
  void sync();

  /// TP-rescatter of one layer (src_tp_size → dst_tp_size, both powers of
  /// two, src < dst): permutes [.., scatter, dnh, hs] to the front. See the
  /// ABI header for the reference quirks preserved.
  void scatter_copy_layer(size_t layer, const std::array<size_t, 5>& dims, size_t elem_size,
                          size_t block_dim_index, size_t src_tp_size, size_t dst_tp_size);

  /// Clears the staged map/ids for the next cycle.
  void reset();

 private:
  void* handle_ = nullptr;
  size_t max_num_layers_ = 0;
  size_t max_num_blocks_ = 0;
  bool staged_layers_ = false;
  bool staged_block_ids_ = false;
  std::vector<bool> layer_doorbells_;
  std::vector<int> src_block_ids_;
  std::vector<int> dst_block_ids_;
  std::shared_ptr<const CopyStreamBlockMap> block_map_;
};

}  // namespace dynamo::llm::kv
