// SPDX-License-Identifier: Apache-2.0
//
// C ABI of the KV block-copy backend — the exact FFI surface Dynamo's
// kv/layer.rs and kv/storage.rs bind against, plus device-allocation
// entry points (Rust gets those from cudarc instead).
//
// Two implementations, one linked per build:
//   - kernels/block_copy.cu   (DYNAMO_WITH_CUDA): the reference CUDA kernel,
//     vendored from third_party/dynamo lib/llm/src/kernels/block_copy.cu.
//   - kernels/block_copy_cpu.cpp: host-memory implementation — copies are
//     synchronous memcpy loops, "streams" complete immediately, and device
//     allocation fails. Semantics (offsets, orderings, quirks) mirror the
//     CUDA file so the same tests validate both.
//
// All functions return 0 on success (cudaError_t-compatible).

#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {

/// Copies num_block_pairs blocks between two [prefix, blocks, suffix] slabs:
/// for each pair p and prefix i, suffix_dim*elem_size bytes move from
/// src[i, src_block_ids[p], :] to dst[i, dst_block_ids[p], :]. Blocking.
int copy_blocks_3d(const void* src_data, void* dst_data, const int* h_src_block_ids,
                   const int* h_dst_block_ids, int num_block_pairs, int prefix_dim,
                   int src_blocks_dim, int dst_blocks_dim, int suffix_dim, int elem_size);

/// Stateful variant: a copy stream owns staged block-id lists (and, under
/// CUDA, a cudaStream_t + device id buffers).
int copy_stream_create(void** stream, int num_layers, int num_blocks);
int copy_stream_destroy(void* stream);
int copy_stream_prepare_block_ids(void* stream, const int* src_block_ids,
                                  const int* dst_block_ids, int num_block_pairs);
int copy_stream_launch(void* stream, const void* src_data, void* dst_data, int prefix_dim,
                       int suffix_dim, int elem_size, int src_block_dim, int dst_block_dim);
int copy_stream_memcpy(void* stream, const void* src_data, void* dst_data, int prefix_dim,
                       int suffix_dim, int elem_size, int src_block_dim, int dst_block_dim);
int copy_stream_sync(void* stream);

/// TP-rescatter permutation of one layer: 6D view [kv/block, block/kv, bs,
/// scatter, dnh, hs] permuted to [scatter, kv/block, block/kv, bs, dnh, hs].
/// Reference quirk preserved: the staged block ids are ignored — blocks
/// 0..num_staged_pairs-1 are copied identity-mapped.
int copy_stream_scatter(void* stream, const void* src_data, void* dst_data, const uint32_t* dims,
                        uint32_t num_dims, uint32_t elem_size, uint32_t block_dim_index,
                        uint32_t src_block_dim, uint32_t dst_block_dim);

/// Pinned host memory (plain zeroed heap memory on the CPU backend).
int cuda_malloc_host(void** ptr, size_t size);
int cuda_free_host(void* ptr);

/// Device memory (always fails on the CPU backend). Our addition to the
/// reference ABI — Rust allocates device memory through cudarc.
int cuda_malloc_device(void** ptr, size_t size);
int cuda_free_device(void* ptr);

/// cudaMemcpyDefault-style copies (plain memcpy on the CPU backend; stream
/// may be null for the default stream).
int cuda_memcpy_async(void* dst, const void* src, size_t count, void* stream);
int cuda_memcpy_sync(void* dst, const void* src, size_t count);

}  // extern "C"
