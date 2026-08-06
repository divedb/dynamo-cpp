// SPDX-License-Identifier: Apache-2.0
//
// KV storage slabs and tensor views — Dynamo's lib/llm kv/storage.rs.
//
// A Storage is one large slab of bytes in one of three tiers:
//   - System: heap memory (implemented here; Rust v0.1.0 declares the kind
//     but raises "not yet supported" — extension so host-only builds work).
//   - Pinned: page-locked host memory via cudaMallocHost. Without CUDA the
//     CPU backend serves plain aligned heap memory (pinned-ness is a
//     performance property, not a semantic one).
//   - Device: GPU memory. Requires a CUDA build (DYNAMO_HAVE_CUDA); creation
//     throws otherwise.
//
// TensorView is a non-owning row-major D-dimensional view over a Storage.
// Element accessors work on host-accessible tiers only; device data moves
// through the copy ABI (h2d/d2h/copy_to_view_blocking).

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>

namespace dynamo::llm::kv {

enum class StorageKind { Device, Pinned, System };

inline const char* to_string(StorageKind kind) {
  switch (kind) {
    case StorageKind::Device: return "device";
    case StorageKind::Pinned: return "pinned";
    case StorageKind::System: return "system";
  }
  return "?";
}

inline bool is_host_accessible(StorageKind kind) { return kind != StorageKind::Device; }

enum class DType { F32, F16, BF16, FP8, U8, U16, U32, U64, I8, I16, I32, I64 };

constexpr size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::F32: return 4;
    case DType::F16: return 2;
    case DType::BF16: return 2;
    case DType::FP8: return 1;
    case DType::U8: return 1;
    case DType::U16: return 2;
    case DType::U32: return 4;
    case DType::U64: return 8;
    case DType::I8: return 1;
    case DType::I16: return 2;
    case DType::I32: return 4;
    case DType::I64: return 8;
  }
  return 0;
}

/// One slab of bytes on some tier.
class Storage {
 public:
  virtual ~Storage() = default;
  /// Base address as an integer (device pointers are not dereferenceable).
  virtual uint64_t pointer() const = 0;
  virtual size_t size() const = 0;
  virtual StorageKind kind() const = 0;
};

/// Shared-ownership handle to a Storage (Rust's OwnedStorage).
class OwnedStorage : public Storage {
 public:
  OwnedStorage() = default;
  explicit OwnedStorage(std::shared_ptr<Storage> storage) : storage_(std::move(storage)) {}

  /// Allocates a zeroed slab on the given tier. Device requires a CUDA
  /// build; all failures throw.
  static OwnedStorage create(size_t bytes, StorageKind kind);

  bool valid() const { return storage_ != nullptr; }
  uint64_t pointer() const override { return storage_->pointer(); }
  size_t size() const override { return storage_->size(); }
  StorageKind kind() const override { return storage_->kind(); }

 private:
  std::shared_ptr<Storage> storage_;
};

/// Non-owning row-major view of a D-dimensional tensor over a Storage.
template <size_t D>
class TensorView {
 public:
  TensorView(const Storage& storage, std::array<size_t, D> shape, size_t element_size)
      : storage_(&storage), shape_(shape), element_size_(element_size) {
    static_assert(D > 0, "zero-dimensional views are not supported");
    strides_[D - 1] = 1;
    for (size_t i = D - 1; i > 0; --i) strides_[i - 1] = strides_[i] * shape_[i];
    total_elements_ = std::accumulate(shape_.begin(), shape_.end(), size_t{1},
                                      std::multiplies<size_t>());
    if (total_elements_ * element_size_ > storage.size()) {
      throw std::runtime_error("tensor view of " +
                               std::to_string(total_elements_ * element_size_) +
                               " bytes exceeds storage of " + std::to_string(storage.size()));
    }
  }

  TensorView(const Storage& storage, std::array<size_t, D> shape, DType dtype)
      : TensorView(storage, shape, dtype_size(dtype)) {}

  const std::array<size_t, D>& shape() const { return shape_; }
  const std::array<size_t, D>& strides() const { return strides_; }
  size_t element_size() const { return element_size_; }
  size_t offset() const { return offset_; }
  size_t num_elements() const { return total_elements_; }
  size_t size_in_bytes() const { return total_elements_ * element_size_; }
  StorageKind storage_kind() const { return storage_->kind(); }
  uint64_t data() const { return storage_->pointer() + offset_; }

  bool in_bounds(const std::array<size_t, D>& indices) const {
    for (size_t i = 0; i < D; ++i) {
      if (indices[i] >= shape_[i]) return false;
    }
    return true;
  }

  size_t flat_index(const std::array<size_t, D>& indices) const {
    check_indices(indices);
    size_t flat = 0;
    for (size_t i = 0; i < D; ++i) flat += indices[i] * strides_[i];
    return flat;
  }

  /// Byte offset from the storage base (includes the view's own offset).
  size_t byte_offset(const std::array<size_t, D>& indices) const {
    return offset_ + flat_index(indices) * element_size_;
  }

  uint64_t address(const std::array<size_t, D>& indices) const {
    return storage_->pointer() + byte_offset(indices);
  }

  bool is_contiguous() const {
    size_t expected = 1;
    for (size_t i = D; i > 0; --i) {
      if (strides_[i - 1] != expected) return false;
      expected *= shape_[i - 1];
    }
    return true;
  }

  template <typename E>
  E get(const std::array<size_t, D>& indices) const {
    check_host_access("read");
    check_element_size(sizeof(E));
    E value;
    std::memcpy(&value, reinterpret_cast<const char*>(storage_->pointer()) + byte_offset(indices),
                sizeof(E));
    return value;
  }

  template <typename E>
  void set(const std::array<size_t, D>& indices, E value) {
    check_host_access("write");
    check_element_size(sizeof(E));
    std::memcpy(reinterpret_cast<char*>(storage_->pointer()) + byte_offset(indices), &value,
                sizeof(E));
  }

  template <typename E>
  void fill(E value) {
    check_host_access("fill");
    check_element_size(sizeof(E));
    if (is_contiguous()) {
      E* base = reinterpret_cast<E*>(reinterpret_cast<char*>(storage_->pointer()) + offset_);
      for (size_t i = 0; i < total_elements_; ++i) base[i] = value;
      return;
    }
    // Sliced views are non-contiguous in the outer dims; walk every index.
    for (size_t flat = 0; flat < total_elements_; ++flat) {
      set(unflatten(flat), value);
    }
  }

  /// Restrict dimension `dim` to [start, end) — same strides, shifted offset.
  TensorView slice(size_t dim, size_t start, std::optional<size_t> end = std::nullopt) const {
    if (dim >= D) throw std::out_of_range("slice dimension out of range");
    size_t end_idx = end.value_or(shape_[dim]);
    if (end_idx > shape_[dim] || start >= end_idx) {
      throw std::out_of_range("invalid slice range");
    }
    TensorView sliced(*this);
    sliced.shape_[dim] = end_idx - start;
    sliced.offset_ = offset_ + start * strides_[dim] * element_size_;
    sliced.total_elements_ = std::accumulate(sliced.shape_.begin(), sliced.shape_.end(), size_t{1},
                                             std::multiplies<size_t>());
    return sliced;
  }

  /// Converts a flat element index of this view into per-dimension indices.
  std::array<size_t, D> unflatten(size_t flat) const {
    std::array<size_t, D> indices{};
    for (size_t i = D; i > 0; --i) {
      indices[i - 1] = flat % shape_[i - 1];
      flat /= shape_[i - 1];
    }
    return indices;
  }

  /// Blocking copy into another same-shape contiguous view (any tier mix the
  /// copy backend supports; on CPU-only builds both sides must be host).
  void copy_to_view_blocking(TensorView& dst) const;

 private:
  void check_indices(const std::array<size_t, D>& indices) const {
    if (!in_bounds(indices)) throw std::out_of_range("tensor index out of bounds");
  }
  void check_host_access(const char* what) const {
    if (!is_host_accessible(storage_->kind())) {
      throw std::runtime_error(std::string("cannot ") + what + " device tensor directly");
    }
  }
  void check_element_size(size_t size) const {
    if (size != element_size_) {
      throw std::runtime_error("element type size " + std::to_string(size) +
                               " does not match view element size " +
                               std::to_string(element_size_));
    }
  }

  const Storage* storage_;
  std::array<size_t, D> shape_;
  std::array<size_t, D> strides_;  // in elements
  size_t element_size_;
  size_t offset_ = 0;  // bytes from storage base
  size_t total_elements_ = 0;
};

/// Blocking raw copy between storage addresses via the copy backend
/// (cudaMemcpyDefault semantics under CUDA; plain memcpy on the CPU backend).
void storage_memcpy_blocking(uint64_t dst, uint64_t src, size_t bytes);

template <size_t D>
void TensorView<D>::copy_to_view_blocking(TensorView<D>& dst) const {
  if (shape_ != dst.shape_ || strides_ != dst.strides_) {
    throw std::runtime_error("copy_to_view_blocking: shape or strides mismatch");
  }
  if (!is_contiguous() || !dst.is_contiguous()) {
    throw std::runtime_error("copy_to_view_blocking: views must be contiguous");
  }
  if (element_size_ != dst.element_size_) {
    throw std::runtime_error("copy_to_view_blocking: element size mismatch");
  }
  storage_memcpy_blocking(dst.data(), data(), size_in_bytes());
}

}  // namespace dynamo::llm::kv
