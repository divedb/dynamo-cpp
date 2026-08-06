// SPDX-License-Identifier: Apache-2.0

#include "llm/kv/storage.h"

#include <cstdlib>

#include "llm/kv/kernels/block_copy.h"

namespace dynamo::llm::kv {

namespace {

/// Heap slab (Rust v0.1.0 declares StorageType::System but raises "not yet
/// supported"; implemented here so host-only builds have a first-class tier).
class SystemStorage final : public Storage {
 public:
  explicit SystemStorage(size_t bytes) : bytes_(bytes) {
    if (bytes == 0) throw std::invalid_argument("storage size must be > 0");
    ptr_ = std::calloc(1, bytes);
    if (!ptr_) throw std::bad_alloc();
  }
  ~SystemStorage() override { std::free(ptr_); }
  SystemStorage(const SystemStorage&) = delete;
  SystemStorage& operator=(const SystemStorage&) = delete;

  uint64_t pointer() const override { return reinterpret_cast<uint64_t>(ptr_); }
  size_t size() const override { return bytes_; }
  StorageKind kind() const override { return StorageKind::System; }

 private:
  void* ptr_ = nullptr;
  size_t bytes_;
};

/// Page-locked host slab via the copy backend (plain heap memory on the CPU
/// backend — pinned-ness only affects transfer performance).
class PinnedStorage final : public Storage {
 public:
  explicit PinnedStorage(size_t bytes) : bytes_(bytes) {
    if (bytes == 0) throw std::invalid_argument("storage size must be > 0");
    if (cuda_malloc_host(&ptr_, bytes) != 0 || !ptr_) {
      throw std::runtime_error("failed to allocate pinned memory");
    }
  }
  ~PinnedStorage() override { cuda_free_host(ptr_); }
  PinnedStorage(const PinnedStorage&) = delete;
  PinnedStorage& operator=(const PinnedStorage&) = delete;

  uint64_t pointer() const override { return reinterpret_cast<uint64_t>(ptr_); }
  size_t size() const override { return bytes_; }
  StorageKind kind() const override { return StorageKind::Pinned; }

 private:
  void* ptr_ = nullptr;
  size_t bytes_;
};

/// Device slab; only creatable on CUDA builds (the CPU backend's allocator
/// always fails).
class DeviceStorage final : public Storage {
 public:
  explicit DeviceStorage(size_t bytes) : bytes_(bytes) {
    if (bytes == 0) throw std::invalid_argument("storage size must be > 0");
    if (cuda_malloc_device(&ptr_, bytes) != 0 || !ptr_) {
      throw std::runtime_error("failed to allocate device memory (CUDA build required)");
    }
  }
  ~DeviceStorage() override { cuda_free_device(ptr_); }
  DeviceStorage(const DeviceStorage&) = delete;
  DeviceStorage& operator=(const DeviceStorage&) = delete;

  uint64_t pointer() const override { return reinterpret_cast<uint64_t>(ptr_); }
  size_t size() const override { return bytes_; }
  StorageKind kind() const override { return StorageKind::Device; }

 private:
  void* ptr_ = nullptr;
  size_t bytes_;
};

}  // namespace

OwnedStorage OwnedStorage::create(size_t bytes, StorageKind kind) {
  switch (kind) {
    case StorageKind::System: return OwnedStorage(std::make_shared<SystemStorage>(bytes));
    case StorageKind::Pinned: return OwnedStorage(std::make_shared<PinnedStorage>(bytes));
    case StorageKind::Device: return OwnedStorage(std::make_shared<DeviceStorage>(bytes));
  }
  throw std::invalid_argument("unknown storage kind");
}

void storage_memcpy_blocking(uint64_t dst, uint64_t src, size_t bytes) {
  if (cuda_memcpy_sync(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src), bytes) !=
      0) {
    throw std::runtime_error("storage copy failed");
  }
}

}  // namespace dynamo::llm::kv
