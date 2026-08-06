// SPDX-License-Identifier: Apache-2.0

#include "llm/kv/layer.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

#include <spdlog/spdlog.h>

#include "llm/kv/kernels/block_copy.h"

namespace dynamo::llm::kv {

void KvBlockDetails::validate() const {
  if (tp_size == 0 || model_details.number_of_heads % tp_size != 0) {
    throw std::invalid_argument("tp_size must evenly divide number_of_heads");
  }
  if (tp_rank >= tp_size) {
    throw std::invalid_argument("tp_rank must be less than tp_size");
  }
  if (tp_size > model_details.number_of_heads) {
    throw std::invalid_argument("tp_size must be at most number_of_heads");
  }
}

KvLayer::KvLayer(KvLayout layout, OwnedStorage storage, size_t number_of_blocks,
                 size_t block_size, size_t number_of_heads, size_t head_size, DType dtype,
                 size_t tp_size, size_t tp_rank)
    : layout_(layout),
      storage_(std::move(storage)),
      number_of_blocks_(number_of_blocks),
      block_size_(block_size),
      number_of_heads_(number_of_heads),
      head_size_(head_size),
      dtype_(dtype),
      tp_size_(tp_size),
      tp_rank_(tp_rank) {
  if (number_of_blocks_ == 0 || block_size_ == 0 || number_of_heads_ == 0 || head_size_ == 0) {
    throw std::invalid_argument("kv layer dimensions must be non-zero");
  }
  if (tp_size_ == 0 || number_of_heads_ % tp_size_ != 0) {
    throw std::invalid_argument("number_of_heads must be divisible by tp_size");
  }
  if (tp_rank_ >= tp_size_) {
    throw std::invalid_argument("tp_rank must be less than tp_size");
  }
  auto shape = layer_shape();
  size_t elements = 1;
  for (size_t dim : shape) elements *= dim;
  if (storage_.size() < elements * dtype_size(dtype_)) {
    throw std::invalid_argument("storage is smaller than the layer tensor");
  }
}

KvLayer KvLayer::from_storage(const KvBlockDetails& details, size_t number_of_blocks,
                              OwnedStorage storage) {
  return KvLayer(details.layout, std::move(storage), number_of_blocks, details.block_size,
                 details.model_details.number_of_heads, details.model_details.head_size,
                 details.model_details.dtype, details.tp_size, details.tp_rank);
}

std::array<size_t, 5> KvLayer::layer_shape() const {
  size_t heads = number_of_heads_ / tp_size_;
  if (layout_ == KvLayout::KvFirst) {
    return {2, number_of_blocks_, block_size_, heads, head_size_};
  }
  return {number_of_blocks_, 2, block_size_, heads, head_size_};
}

void KvLayer::copy_blocks_to(const std::vector<size_t>& src_block_ids, KvLayer& dst,
                             const std::vector<size_t>& dst_block_ids) const {
  if (src_block_ids.size() != dst_block_ids.size()) {
    throw std::invalid_argument("src and dst block id lists must have the same length");
  }
  if (layout_ != dst.layout_) {
    throw std::invalid_argument("src and dst must have the same layout");
  }
  // Deviation from Rust (which rejects Pinned→Pinned and System entirely):
  // any combination except Device→Device is allowed — cudaMemcpyDefault
  // handles host↔host and host↔device alike, and the CPU backend covers
  // host↔host in non-CUDA builds.
  if (storage_.kind() == StorageKind::Device && dst.storage_.kind() == StorageKind::Device) {
    throw std::runtime_error("device-to-device block copies are not implemented");
  }

  std::vector<int> src_ids(src_block_ids.begin(), src_block_ids.end());
  std::vector<int> dst_ids(dst_block_ids.begin(), dst_block_ids.end());

  size_t suffix = block_size_ * (number_of_heads_ / tp_size_) * head_size_;
  if (layout_ == KvLayout::BlockFirst) suffix *= 2;
  int prefix = layout_ == KvLayout::KvFirst ? 2 : 1;

  int rc = copy_blocks_3d(reinterpret_cast<const void*>(storage_.pointer()),
                          reinterpret_cast<void*>(dst.storage_.pointer()), src_ids.data(),
                          dst_ids.data(), static_cast<int>(src_ids.size()), prefix,
                          static_cast<int>(number_of_blocks_),
                          static_cast<int>(dst.number_of_blocks_), static_cast<int>(suffix),
                          static_cast<int>(dtype_size(dtype_)));
  if (rc != 0) throw std::runtime_error("failed to copy blocks");
}

KvBlockStorage KvBlockStorage::from_layers(std::vector<KvLayer> layers) {
  if (layers.empty()) throw std::invalid_argument("layers must not be empty");

  StorageKind kind = layers[0].storage().kind();
  size_t number_of_blocks = layers[0].number_of_blocks();
  for (const auto& layer : layers) {
    if (layer.storage().kind() != kind) {
      throw std::invalid_argument("all layers must have the same storage kind");
    }
    if (layer.number_of_blocks() != number_of_blocks) {
      throw std::invalid_argument("all layers must have the same number of blocks");
    }
  }

  KvBlockDetails details;
  details.layout = layers[0].layout();
  details.block_size = layers[0].block_size();
  details.tp_size = layers[0].tp_size();
  details.tp_rank = layers[0].tp_rank();
  details.model_details = KvModelDetails{layers.size(), layers[0].number_of_heads(),
                                         layers[0].head_size(), layers[0].dtype()};
  details.validate();

  size_t bytes_per_block = details.bytes_per_token_block_per_layer();
  for (const auto& layer : layers) {
    if (layer.storage().size() < bytes_per_block) {
      throw std::invalid_argument("layer storage too small for one block");
    }
  }

  return KvBlockStorage(std::move(details), kind, number_of_blocks, std::move(layers));
}

KvBlockStorage KvBlockStorage::allocate(size_t number_of_blocks, const KvBlockDetails& details,
                                        StorageKind kind) {
  details.validate();
  size_t bytes = details.bytes_per_token_block_per_layer() * number_of_blocks;

  std::vector<KvLayer> layers;
  layers.reserve(details.model_details.number_of_layers);
  for (size_t i = 0; i < details.model_details.number_of_layers; ++i) {
    layers.push_back(KvLayer::from_storage(details, number_of_blocks,
                                           OwnedStorage::create(bytes, kind)));
  }
  return from_layers(std::move(layers));
}

const KvLayer& KvBlockStorage::layer(size_t index) const {
  if (index >= layers_.size()) throw std::out_of_range("layer index out of bounds");
  return layers_[index];
}

KvLayer& KvBlockStorage::layer(size_t index) {
  if (index >= layers_.size()) throw std::out_of_range("layer index out of bounds");
  return layers_[index];
}

std::shared_ptr<const CopyStreamBlockMap> CopyStreamBlockMap::create(const KvBlockStorage& src,
                                                                     const KvBlockStorage& dst) {
  if (!src.block_details().is_compatible(dst.block_details())) {
    throw std::invalid_argument("src and dst must have compatible block details");
  }
  if (src.layer_count() != dst.layer_count()) {
    throw std::invalid_argument("src and dst must have the same number of layers");
  }

  auto map = std::make_shared<CopyStreamBlockMap>();
  for (size_t i = 0; i < src.layer_count(); ++i) {
    map->src_layer_ptrs.push_back(src.layer(i).storage().pointer());
    map->dst_layer_ptrs.push_back(dst.layer(i).storage().pointer());
  }
  map->prefix_dim = static_cast<int>(src.block_details().prefix_dim());
  map->suffix_dim = static_cast<int>(src.block_details().suffix_dim());
  map->elem_size = static_cast<int>(src.block_details().elem_size());
  map->src_block_dim = static_cast<int>(src.number_of_blocks());
  map->dst_block_dim = static_cast<int>(dst.number_of_blocks());

  if (map->elem_size < 1 || map->elem_size > 8) {
    throw std::invalid_argument("element size must be 1..8 bytes");
  }
  return map;
}

CopyStream::CopyStream(size_t max_num_layers, size_t max_num_blocks)
    : max_num_layers_(max_num_layers),
      max_num_blocks_(max_num_blocks),
      layer_doorbells_(max_num_layers, false) {
  if (copy_stream_create(&handle_, static_cast<int>(max_num_layers),
                         static_cast<int>(max_num_blocks)) != 0 ||
      handle_ == nullptr) {
    throw std::runtime_error("failed to create copy stream");
  }
}

CopyStream::~CopyStream() {
  // Deviation: Rust never calls copy_stream_destroy (the handle leaks there).
  if (handle_) copy_stream_destroy(handle_);
}

CopyStream::CopyStream(CopyStream&& other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)),
      max_num_layers_(other.max_num_layers_),
      max_num_blocks_(other.max_num_blocks_),
      staged_layers_(other.staged_layers_),
      staged_block_ids_(other.staged_block_ids_),
      layer_doorbells_(std::move(other.layer_doorbells_)),
      src_block_ids_(std::move(other.src_block_ids_)),
      dst_block_ids_(std::move(other.dst_block_ids_)),
      block_map_(std::move(other.block_map_)) {}

CopyStream& CopyStream::operator=(CopyStream&& other) noexcept {
  if (this != &other) {
    if (handle_) copy_stream_destroy(handle_);
    handle_ = std::exchange(other.handle_, nullptr);
    max_num_layers_ = other.max_num_layers_;
    max_num_blocks_ = other.max_num_blocks_;
    staged_layers_ = other.staged_layers_;
    staged_block_ids_ = other.staged_block_ids_;
    layer_doorbells_ = std::move(other.layer_doorbells_);
    src_block_ids_ = std::move(other.src_block_ids_);
    dst_block_ids_ = std::move(other.dst_block_ids_);
    block_map_ = std::move(other.block_map_);
  }
  return *this;
}

void CopyStream::prepare_block_map(std::shared_ptr<const CopyStreamBlockMap> map) {
  if (map->src_layer_ptrs.empty() ||
      map->src_layer_ptrs.size() != map->dst_layer_ptrs.size()) {
    throw std::invalid_argument("block map must carry matching non-empty layer lists");
  }
  if (map->src_layer_ptrs.size() > max_num_layers_) {
    throw std::invalid_argument("number of layers exceeds the copy stream capacity");
  }
  if (staged_layers_) throw std::logic_error("layers already staged");

  staged_layers_ = true;
  block_map_ = std::move(map);
  std::fill(layer_doorbells_.begin(), layer_doorbells_.end(), false);
}

void CopyStream::prepare_block_ids(std::vector<int> src_block_ids,
                                   std::vector<int> dst_block_ids) {
  if (src_block_ids.size() != dst_block_ids.size()) {
    throw std::invalid_argument("src and dst block id lists must have the same length");
  }
  if (src_block_ids.size() > max_num_blocks_) {
    throw std::invalid_argument("number of blocks exceeds the copy stream capacity");
  }
  if (!staged_layers_) throw std::logic_error("layers must be staged before block ids");
  if (staged_block_ids_) throw std::logic_error("block ids already staged");

  auto assert_unique = [](const std::vector<int>& ids, const char* what) {
    std::unordered_set<int> seen(ids.begin(), ids.end());
    if (seen.size() != ids.size()) {
      throw std::invalid_argument(std::string(what) + " block ids must be unique");
    }
  };
  assert_unique(src_block_ids, "src");
  assert_unique(dst_block_ids, "dst");

  src_block_ids_ = std::move(src_block_ids);
  dst_block_ids_ = std::move(dst_block_ids);

  if (copy_stream_prepare_block_ids(handle_, src_block_ids_.data(), dst_block_ids_.data(),
                                    static_cast<int>(src_block_ids_.size())) != 0) {
    throw std::runtime_error("failed to prepare block ids");
  }
  staged_block_ids_ = true;
}

void CopyStream::trigger_layer(size_t layer) {
  if (!staged_layers_) throw std::logic_error("layers must be staged before triggering");
  if (!staged_block_ids_) throw std::logic_error("block ids must be staged before triggering");
  if (layer >= block_map_->src_layer_ptrs.size()) {
    throw std::out_of_range("layer index out of bounds");
  }
  if (layer_doorbells_[layer]) {
    spdlog::trace("copy stream: layer {} already triggered; no-op", layer);
    return;
  }

  if (copy_stream_memcpy(handle_,
                         reinterpret_cast<const void*>(block_map_->src_layer_ptrs[layer]),
                         reinterpret_cast<void*>(block_map_->dst_layer_ptrs[layer]),
                         block_map_->prefix_dim, block_map_->suffix_dim, block_map_->elem_size,
                         block_map_->src_block_dim, block_map_->dst_block_dim) != 0) {
    throw std::runtime_error("failed to execute layer copy");
  }
  layer_doorbells_[layer] = true;
}

size_t CopyStream::layer_count() const {
  return block_map_ ? block_map_->src_layer_ptrs.size() : 0;
}

void CopyStream::trigger_all_layers() {
  for (size_t layer = 0; layer < layer_count(); ++layer) trigger_layer(layer);
}

void CopyStream::sync() {
  if (copy_stream_sync(handle_) != 0) {
    throw std::runtime_error("failed to synchronize copy stream");
  }
}

void CopyStream::scatter_copy_layer(size_t layer, const std::array<size_t, 5>& dims,
                                    size_t elem_size, size_t block_dim_index, size_t src_tp_size,
                                    size_t dst_tp_size) {
  if (block_dim_index > 1) throw std::invalid_argument("block_dim_index must be 0 or 1");
  if (elem_size == 0 || elem_size > 8) throw std::invalid_argument("elem_size must be 1..8");
  if (src_tp_size >= dst_tp_size) {
    throw std::invalid_argument("src_tp_size must be less than dst_tp_size");
  }
  auto power_of_two = [](size_t v) { return v != 0 && (v & (v - 1)) == 0; };
  if (!power_of_two(src_tp_size) || !power_of_two(dst_tp_size)) {
    throw std::invalid_argument("tp sizes must be powers of 2");
  }
  if (!staged_layers_ || !staged_block_ids_) {
    throw std::logic_error("layers and block ids must be staged before scattering");
  }
  if (layer >= block_map_->src_layer_ptrs.size()) {
    throw std::out_of_range("layer index out of bounds");
  }

  size_t scatter_factor = dst_tp_size / src_tp_size;
  std::array<uint32_t, 6> src_6d = {
      static_cast<uint32_t>(dims[0]),
      static_cast<uint32_t>(dims[1]),
      static_cast<uint32_t>(dims[2]),
      static_cast<uint32_t>(scatter_factor),
      static_cast<uint32_t>(dims[3] / scatter_factor),
      static_cast<uint32_t>(dims[4]),
  };

  if (copy_stream_scatter(handle_,
                          reinterpret_cast<const void*>(block_map_->src_layer_ptrs[layer]),
                          reinterpret_cast<void*>(block_map_->dst_layer_ptrs[layer]),
                          src_6d.data(), 6, static_cast<uint32_t>(elem_size),
                          static_cast<uint32_t>(block_dim_index),
                          static_cast<uint32_t>(block_map_->src_block_dim),
                          static_cast<uint32_t>(block_map_->dst_block_dim)) != 0) {
    throw std::runtime_error("failed to execute tensor scatter");
  }
}

void CopyStream::reset() {
  staged_layers_ = false;
  staged_block_ids_ = false;
}

}  // namespace dynamo::llm::kv
