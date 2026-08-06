// SPDX-License-Identifier: Apache-2.0
//
// Host-memory implementation of the block-copy ABI (linked when CUDA is not
// available). Everything is synchronous; offset arithmetic matches the CUDA
// reference exactly so the same tests validate either backend.

#include "llm/kv/kernels/block_copy.h"

#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

struct CpuCopyStream {
  std::vector<int> src_block_ids;
  std::vector<int> dst_block_ids;
  int max_num_blocks = 0;
};

int copy_blocks_impl(const void* src_data, void* dst_data, const int* src_ids, const int* dst_ids,
                     int num_block_pairs, int prefix_dim, int suffix_dim, int elem_size,
                     int src_block_dim, int dst_block_dim) {
  if (!src_data || !dst_data || !src_ids || !dst_ids) return -1;
  if (num_block_pairs <= 0 || prefix_dim <= 0 || suffix_dim <= 0 || elem_size <= 0) return -1;

  const size_t suffix_bytes = static_cast<size_t>(suffix_dim) * elem_size;
  const size_t src_prefix_stride = static_cast<size_t>(src_block_dim) * suffix_bytes;
  const size_t dst_prefix_stride = static_cast<size_t>(dst_block_dim) * suffix_bytes;

  const char* src = static_cast<const char*>(src_data);
  char* dst = static_cast<char*>(dst_data);
  for (int prefix = 0; prefix < prefix_dim; ++prefix) {
    for (int pair = 0; pair < num_block_pairs; ++pair) {
      std::memcpy(dst + prefix * dst_prefix_stride + dst_ids[pair] * suffix_bytes,
                  src + prefix * src_prefix_stride + src_ids[pair] * suffix_bytes, suffix_bytes);
    }
  }
  return 0;
}

}  // namespace

extern "C" {

int copy_blocks_3d(const void* src_data, void* dst_data, const int* h_src_block_ids,
                   const int* h_dst_block_ids, int num_block_pairs, int prefix_dim,
                   int src_blocks_dim, int dst_blocks_dim, int suffix_dim, int elem_size) {
  return copy_blocks_impl(src_data, dst_data, h_src_block_ids, h_dst_block_ids, num_block_pairs,
                          prefix_dim, suffix_dim, elem_size, src_blocks_dim, dst_blocks_dim);
}

int copy_stream_create(void** stream, int /*num_layers*/, int num_blocks) {
  auto* cs = new CpuCopyStream();
  cs->max_num_blocks = num_blocks;
  *stream = cs;
  return 0;
}

int copy_stream_destroy(void* stream) {
  delete static_cast<CpuCopyStream*>(stream);
  return 0;
}

int copy_stream_prepare_block_ids(void* stream, const int* src_block_ids,
                                  const int* dst_block_ids, int num_block_pairs) {
  auto* cs = static_cast<CpuCopyStream*>(stream);
  if (num_block_pairs > cs->max_num_blocks) return -1;
  cs->src_block_ids.assign(src_block_ids, src_block_ids + num_block_pairs);
  cs->dst_block_ids.assign(dst_block_ids, dst_block_ids + num_block_pairs);
  return 0;
}

int copy_stream_launch(void* stream, const void* src_data, void* dst_data, int prefix_dim,
                       int suffix_dim, int elem_size, int src_block_dim, int dst_block_dim) {
  // The CUDA backend launches a gather kernel here; on host memory the memcpy
  // path is the same operation.
  return copy_stream_memcpy(stream, src_data, dst_data, prefix_dim, suffix_dim, elem_size,
                            src_block_dim, dst_block_dim);
}

int copy_stream_memcpy(void* stream, const void* src_data, void* dst_data, int prefix_dim,
                       int suffix_dim, int elem_size, int src_block_dim, int dst_block_dim) {
  auto* cs = static_cast<CpuCopyStream*>(stream);
  return copy_blocks_impl(src_data, dst_data, cs->src_block_ids.data(), cs->dst_block_ids.data(),
                          static_cast<int>(cs->src_block_ids.size()), prefix_dim, suffix_dim,
                          elem_size, src_block_dim, dst_block_dim);
}

int copy_stream_sync(void* /*stream*/) {
  return 0;  // all CPU copies are synchronous
}

int copy_stream_scatter(void* stream, const void* src_data, void* dst_data, const uint32_t* dims,
                        uint32_t num_dims, uint32_t elem_size, uint32_t block_dim_index,
                        uint32_t src_block_dim, uint32_t dst_block_dim) {
  if (num_dims != 6) return -1;
  if (block_dim_index > 1) return -2;
  const uint32_t kv_dim_index = block_dim_index == 0 ? 1 : 0;
  if (dims[block_dim_index] != src_block_dim) return -3;
  if (dims[kv_dim_index] != 2) return -4;

  auto* cs = static_cast<CpuCopyStream*>(stream);

  // 5D collapse of the 6D shapes (dims[4]*dims[5] is one contiguous copy) —
  // mirrors permute_scatter_memcpy in the CUDA reference, including its
  // quirk of ignoring the staged block ids (identity block mapping over the
  // staged pair count).
  size_t src_shape[5];
  src_shape[block_dim_index] = src_block_dim;
  src_shape[kv_dim_index] = dims[kv_dim_index];
  src_shape[2] = dims[2];
  src_shape[3] = dims[3];
  src_shape[4] = static_cast<size_t>(dims[4]) * dims[5];

  size_t dst_shape[5];
  dst_shape[0] = dims[3];  // scatter factor
  dst_shape[block_dim_index + 1] = dst_block_dim;
  dst_shape[kv_dim_index + 1] = dims[kv_dim_index];
  dst_shape[3] = dims[2];  // block size
  dst_shape[4] = static_cast<size_t>(dims[4]) * dims[5];

  size_t src_strides[5];
  size_t dst_strides[5];
  src_strides[4] = elem_size;
  dst_strides[4] = elem_size;
  for (int i = 3; i >= 0; --i) {
    src_strides[i] = src_strides[i + 1] * src_shape[i + 1];
    dst_strides[i] = dst_strides[i + 1] * dst_shape[i + 1];
  }

  const size_t copy_bytes = static_cast<size_t>(dims[4]) * dims[5] * elem_size;
  const char* src = static_cast<const char*>(src_data);
  char* dst = static_cast<char*>(dst_data);

  size_t src_idx[5] = {0, 0, 0, 0, 0};
  size_t dst_idx[5] = {0, 0, 0, 0, 0};
  const size_t num_blocks = cs->src_block_ids.size();
  for (size_t block = 0; block < num_blocks; ++block) {
    src_idx[block_dim_index] = block;
    dst_idx[block_dim_index + 1] = block;
    for (size_t kv = 0; kv < src_shape[kv_dim_index]; ++kv) {
      src_idx[kv_dim_index] = kv;
      dst_idx[kv_dim_index + 1] = kv;
      for (size_t bs = 0; bs < src_shape[2]; ++bs) {
        src_idx[2] = bs;
        dst_idx[3] = bs;
        for (size_t scatter = 0; scatter < src_shape[3]; ++scatter) {
          src_idx[3] = scatter;
          dst_idx[0] = scatter;
          size_t src_offset = 0;
          size_t dst_offset = 0;
          for (int i = 0; i < 4; ++i) {
            src_offset += src_idx[i] * src_strides[i];
            dst_offset += dst_idx[i] * dst_strides[i];
          }
          std::memcpy(dst + dst_offset, src + src_offset, copy_bytes);
        }
      }
    }
  }
  return 0;
}

int cuda_malloc_host(void** ptr, size_t size) {
  *ptr = std::calloc(1, size);
  return *ptr ? 0 : -1;
}

int cuda_free_host(void* ptr) {
  std::free(ptr);
  return 0;
}

int cuda_malloc_device(void** /*ptr*/, size_t /*size*/) {
  return -1;  // no device without CUDA
}

int cuda_free_device(void* /*ptr*/) { return -1; }

int cuda_memcpy_async(void* dst, const void* src, size_t count, void* /*stream*/) {
  std::memcpy(dst, src, count);
  return 0;
}

int cuda_memcpy_sync(void* dst, const void* src, size_t count) {
  std::memcpy(dst, src, count);
  return 0;
}

}  // extern "C"
