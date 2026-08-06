// SPDX-License-Identifier: Apache-2.0
//
// KV block manager (pools, reservations, prefill preparation) and the
// storage/layer/copy-stream stack. Ports of the Rust kv/ unit tests plus
// CPU-backend copy verification; the same layer tests validate the CUDA
// backend when built with one.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include <array>
#include <cstring>
#include <numeric>
#include <vector>

#include "llm/kv/layer.h"
#include "llm/kv/manager.h"
#include "llm/kv/reserved.h"
#include "llm/kv/reuse.h"
#include "llm/kv/storage.h"

using namespace dynamo::llm;
using namespace dynamo::llm::kv;

namespace {

std::vector<Token> tokens_of(std::initializer_list<Token> values) { return {values}; }

std::vector<KvBlock> create_blocks(std::vector<Token> tokens, size_t block_size) {
  auto [token_blocks, tail] = TokenSequence::split_tokens(std::move(tokens), block_size);
  std::vector<KvBlock> blocks;
  blocks.reserve(token_blocks.size());
  for (auto& tb : token_blocks) blocks.emplace_back(KvBlock(std::move(tb)));
  return blocks;
}

std::vector<SequenceHash> hashes_of(const std::vector<KvBlock>& blocks) {
  std::vector<SequenceHash> hashes;
  for (const auto& b : blocks) hashes.push_back(b.token_block.sequence_hash);
  return hashes;
}

/// Inserts blocks tail-to-root, matching the Rust tests' insertion order.
void insert_reversed(AvailableBlocks& pool, std::vector<KvBlock> blocks) {
  for (auto it = blocks.rbegin(); it != blocks.rend(); ++it) pool.insert(std::move(*it));
}

}  // namespace

// ---------------------------------------------------------------------------
// Available pool (reuse.rs tests)
// ---------------------------------------------------------------------------

TEST_CASE("available pool matches blocks in order and takes returns back", "[llm][kv][blocks]") {
  AvailableBlocks pool;

  auto blocks = create_blocks(tokens_of({1, 2, 3, 4}), 2);
  REQUIRE(blocks.size() == 2);
  auto hashes = hashes_of(blocks);

  for (auto& block : blocks) pool.insert(std::move(block));
  REQUIRE(pool.total_blocks() == 2);
  REQUIRE(pool.available_blocks() == 2);

  {
    auto matched = pool.match_blocks(hashes);
    REQUIRE(matched.size() == 2);
    REQUIRE(pool.total_blocks() == 2);
    REQUIRE(pool.available_blocks() == 0);
    REQUIRE(matched[0]->token_block.sequence_hash == hashes[0]);
    REQUIRE(matched[1]->token_block.sequence_hash == hashes[1]);
    // handles drop here (tail first), returning both blocks with state intact
  }

  REQUIRE(pool.available_blocks() == 2);
  auto rematched = pool.match_blocks(hashes);
  REQUIRE(rematched.size() == 2);
}

TEST_CASE("equal-priority blocks are taken in return order", "[llm][kv][blocks]") {
  AvailableBlocks pool;

  auto blocks1 = create_blocks(tokens_of({1, 2, 3, 4}), 2);
  auto blocks2 = create_blocks(tokens_of({5, 6, 7, 8}), 2);
  for (auto& b : blocks1) b.priority = 1;
  for (auto& b : blocks2) b.priority = 1;

  insert_reversed(pool, std::move(blocks2));
  insert_reversed(pool, std::move(blocks1));

  auto taken = pool.take_blocks(4);
  REQUIRE(taken.size() == 4);
  CHECK(taken[0]->token_block.tokens[0] == 7);
  CHECK(taken[1]->token_block.tokens[0] == 5);
  CHECK(taken[2]->token_block.tokens[0] == 3);
  CHECK(taken[3]->token_block.tokens[0] == 1);
  for (auto& block : taken) std::move(block).take();  // detach; nothing returns
  REQUIRE(pool.available_blocks() == 0);
}

TEST_CASE("lower-priority blocks are evicted first", "[llm][kv][blocks]") {
  AvailableBlocks pool;

  auto blocks1 = create_blocks(tokens_of({1, 2, 3, 4}), 2);
  auto blocks2 = create_blocks(tokens_of({5, 6, 7, 8}), 2);
  for (auto& b : blocks1) b.priority = 1;
  for (auto& b : blocks2) b.priority = 2;

  insert_reversed(pool, std::move(blocks2));
  insert_reversed(pool, std::move(blocks1));

  auto taken = pool.take_blocks(4);
  REQUIRE(taken.size() == 4);
  CHECK(taken[0]->token_block.tokens[0] == 3);
  CHECK(taken[1]->token_block.tokens[0] == 1);
  CHECK(taken[2]->token_block.tokens[0] == 7);
  CHECK(taken[3]->token_block.tokens[0] == 5);
}

TEST_CASE("priority updates reorder eviction", "[llm][kv][blocks]") {
  AvailableBlocks pool;

  auto blocks1 = create_blocks(tokens_of({1, 2, 3, 4}), 2);
  auto blocks2 = create_blocks(tokens_of({5, 6, 7, 8}), 2);
  for (auto& b : blocks1) b.priority = 1;
  for (auto& b : blocks2) b.priority = 1;
  auto block2_hashes = hashes_of(blocks2);

  insert_reversed(pool, std::move(blocks2));
  insert_reversed(pool, std::move(blocks1));

  std::vector<std::pair<SequenceHash, uint32_t>> updates;
  for (auto hash : block2_hashes) updates.emplace_back(hash, 2);
  pool.update_priorities(updates);

  auto taken = pool.take_blocks(4);
  REQUIRE(taken.size() == 4);
  CHECK(taken[0]->token_block.tokens[0] == 3);
  CHECK(taken[1]->token_block.tokens[0] == 1);
  CHECK(taken[2]->token_block.tokens[0] == 7);
  CHECK(taken[3]->token_block.tokens[0] == 5);
}

TEST_CASE("reset wipes block state selectively or entirely", "[llm][kv][blocks]") {
  AvailableBlocks pool;

  auto blocks1 = create_blocks(tokens_of({1, 2, 3, 4}), 2);
  auto blocks2 = create_blocks(tokens_of({5, 6, 7, 8}), 2);
  auto block1_hashes = hashes_of(blocks1);
  auto block2_hashes = hashes_of(blocks2);

  SECTION("reset_all") {
    insert_reversed(pool, std::move(blocks2));
    insert_reversed(pool, std::move(blocks1));
    pool.reset_all();
    REQUIRE(pool.match_blocks(block2_hashes).empty());
    REQUIRE(pool.match_blocks(block1_hashes).empty());
    // Wiped blocks remain takeable as uninitialized capacity.
    REQUIRE(pool.take_blocks(4).size() == 4);
  }

  SECTION("selective reset") {
    insert_reversed(pool, std::move(blocks2));
    insert_reversed(pool, std::move(blocks1));
    pool.reset(block2_hashes);
    REQUIRE(pool.match_blocks(block2_hashes).empty());
    REQUIRE(pool.match_blocks(block1_hashes).size() == 2);
  }
}

// ---------------------------------------------------------------------------
// Reserved blocks (reserved.rs test)
// ---------------------------------------------------------------------------

TEST_CASE("reserved blocks are shared by hash and return to the pool on last drop",
          "[llm][kv][blocks]") {
  AvailableBlocks pool;
  ReservedBlocks reserved_blocks(2);

  insert_reversed(pool, create_blocks(tokens_of({5, 6, 7, 8}), 2));
  insert_reversed(pool, create_blocks(tokens_of({1, 2, 3, 4}), 2));
  REQUIRE(pool.total_blocks() == 4);
  REQUIRE(pool.available_blocks() == 4);

  // A request for [1, 2]: one full block, empty tail.
  auto [request_blocks, tail] = TokenSequence::split_tokens(tokens_of({1, 2}), 2);
  REQUIRE(request_blocks.size() == 1);
  REQUIRE(tail.tokens().empty());

  REQUIRE(reserved_blocks.match_token_blocks(request_blocks).empty());

  auto matched = pool.match_token_blocks(request_blocks);
  REQUIRE(matched.size() == 1);

  std::vector<ReservedBlock> reserved;
  for (auto& block : matched) reserved.push_back(reserved_blocks.register_block(std::move(block)));
  REQUIRE(reserved.size() == 1);
  REQUIRE(reserved[0].inflight_count() == 1);
  REQUIRE(pool.available_blocks() == 3);

  {
    auto reserved2 = reserved_blocks.match_token_blocks(request_blocks);
    REQUIRE(reserved2.size() == 1);
    REQUIRE(reserved2[0].inflight_count() == 2);
    REQUIRE(pool.available_blocks() == 3);
  }

  REQUIRE(reserved[0].inflight_count() == 1);
  REQUIRE(pool.available_blocks() == 3);

  reserved.clear();
  REQUIRE(pool.available_blocks() == 4);
}

TEST_CASE("registering a duplicate block reuses the existing registration", "[llm][kv][blocks]") {
  AvailableBlocks pool;
  ReservedBlocks reserved_blocks(2);

  auto blocks = create_blocks(tokens_of({1, 2}), 2);
  KvBlock duplicate = blocks[0];  // same token block → same hash
  pool.insert(std::move(blocks[0]));
  pool.insert(std::move(duplicate));  // second copy parks as uninitialized
  REQUIRE(pool.available_blocks() == 2);

  auto taken = pool.take_blocks(2);
  REQUIRE(taken.size() == 2);
  // The uninitialized copy comes out first with wiped identity; give it the
  // same token block so both register attempts race on one hash.
  taken[0].update_token_block(taken[1]->token_block);

  auto first = reserved_blocks.register_block(std::move(taken[0]));
  REQUIRE(pool.available_blocks() == 0);
  auto second = reserved_blocks.register_block(std::move(taken[1]));
  // The loser's block went straight back to the pool.
  REQUIRE(pool.available_blocks() == 1);
  REQUIRE(first.inflight_count() == 2);
  REQUIRE(second.inflight_count() == 2);

  REQUIRE_THROWS(reserved_blocks.register_block(UniqueBlock{}));
}

TEST_CASE("registering a partial block is rejected", "[llm][kv][blocks]") {
  AvailableBlocks pool;
  ReservedBlocks reserved_blocks(4);  // expects 4-token blocks

  pool.insert(KvBlock{});
  auto taken = pool.take_blocks(1);
  REQUIRE(taken.size() == 1);
  auto [request_blocks, tail] = TokenSequence::split_tokens(tokens_of({1, 2}), 2);
  taken[0].update_token_block(request_blocks[0]);  // 2 tokens ≠ block size 4
  REQUIRE_THROWS_WITH(reserved_blocks.register_block(std::move(taken[0])),
                      "block size mismatch");
}

// ---------------------------------------------------------------------------
// Manager (manager.rs flow)
// ---------------------------------------------------------------------------

TEST_CASE("prefill preparation matches, allocates, and shares blocks", "[llm][kv][manager]") {
  KvStorageManager manager(2);
  for (int i = 0; i < 100; ++i) manager.available_blocks().insert(KvBlock{});

  // 9 tokens with block size 2 → 4 full blocks + 1 tail token (the
  // commented-out Rust manager test).
  std::vector<Token> tokens(9);
  std::iota(tokens.begin(), tokens.end(), 0);

  auto matched = manager.prepare_prefill_sequence(tokens);
  REQUIRE(matched.inflight_blocks.empty());
  REQUIRE(matched.remaining_blocks.size() == 4);
  REQUIRE(matched.tail_block.tokens().size() == 1);

  auto offload = manager.prepare_prefill_offload(std::move(matched));
  REQUIRE(offload.complete_prefill_blocks.size() == 4);
  REQUIRE(offload.tail_prefill_block.kv_block.valid());
  REQUIRE(manager.available_blocks().available_blocks() == 95);

  // Each pool block was stamped with its token block.
  for (size_t i = 0; i < 4; ++i) {
    REQUIRE(offload.complete_prefill_blocks[i]->token_block.tokens ==
            std::vector<Token>{static_cast<Token>(2 * i), static_cast<Token>(2 * i + 1)});
  }

  // Simulate prefill completion: dropping the prefilled blocks parks them
  // (state intact) in the available pool...
  offload.complete_prefill_blocks.clear();
  offload.tail_prefill_block.kv_block = UniqueBlock{};
  REQUIRE(manager.available_blocks().available_blocks() == 100);

  // ...so an identical request now reuses all four blocks without prefill,
  // promoted to inflight registrations.
  auto rematched = manager.prepare_prefill_sequence(tokens);
  REQUIRE(rematched.inflight_blocks.size() == 4);
  REQUIRE(rematched.remaining_blocks.empty());
  REQUIRE(manager.available_blocks().available_blocks() == 96);

  // A third concurrent request shares the same inflight blocks.
  auto shared = manager.prepare_prefill_sequence(tokens);
  REQUIRE(shared.inflight_blocks.size() == 4);
  REQUIRE(shared.inflight_blocks[0].inflight_count() == 2);
  REQUIRE(manager.available_blocks().available_blocks() == 96);
}

TEST_CASE("prefill offload fails fast when the pool is exhausted", "[llm][kv][manager]") {
  KvStorageManager manager(2);
  manager.available_blocks().insert(KvBlock{});  // 1 block; request needs 3

  std::vector<Token> tokens = {0, 1, 2, 3};
  auto matched = manager.prepare_prefill_sequence(tokens);
  REQUIRE(matched.remaining_blocks.size() == 2);
  REQUIRE_THROWS(manager.prepare_prefill_offload(std::move(matched)));
}

// ---------------------------------------------------------------------------
// Storage + TensorView (storage.rs tests)
// ---------------------------------------------------------------------------

TEST_CASE("tensor views compute strides, offsets, and bounds", "[llm][kv][storage]") {
  auto storage = OwnedStorage::create(2 * 3 * 4 * sizeof(float), StorageKind::System);
  TensorView<3> view(storage, {2, 3, 4}, DType::F32);

  CHECK(view.strides() == std::array<size_t, 3>{12, 4, 1});
  CHECK(view.num_elements() == 24);
  CHECK(view.size_in_bytes() == 96);
  CHECK(view.is_contiguous());
  CHECK(view.flat_index({1, 2, 3}) == 23);
  CHECK(view.byte_offset({0, 1, 2}) == 6 * sizeof(float));
  CHECK(view.address({0, 0, 0}) == storage.pointer());
  CHECK_FALSE(view.in_bounds({2, 0, 0}));
  CHECK_THROWS(view.flat_index({0, 3, 0}));

  // A view larger than its storage is rejected.
  CHECK_THROWS(TensorView<3>(storage, {2, 3, 5}, DType::F32));
  // System storage arrives zeroed.
  CHECK(view.get<float>({1, 1, 1}) == 0.0f);
}

TEST_CASE("tensor views read, write, fill, and slice host storage", "[llm][kv][storage]") {
  auto storage = OwnedStorage::create(4 * 6 * sizeof(float), StorageKind::System);
  TensorView<2> view(storage, {4, 6}, DType::F32);

  view.fill(1.5f);
  CHECK(view.get<float>({3, 5}) == 1.5f);
  view.set<float>({2, 4}, 42.0f);
  CHECK(view.get<float>({2, 4}) == 42.0f);
  CHECK_THROWS(view.get<double>({0, 0}));  // element size mismatch

  // Slicing rows [1, 3) keeps strides and shifts the offset.
  auto rows = view.slice(0, 1, 3);
  CHECK(rows.shape() == std::array<size_t, 2>{2, 6});
  CHECK(rows.is_contiguous());  // dim-0 slices stay contiguous
  CHECK(rows.get<float>({1, 4}) == 42.0f);

  // Slicing an inner dim goes non-contiguous; fill walks every index.
  auto cols = view.slice(1, 2, 4);
  CHECK(cols.shape() == std::array<size_t, 2>{4, 2});
  CHECK_FALSE(cols.is_contiguous());
  cols.fill(7.0f);
  CHECK(view.get<float>({0, 2}) == 7.0f);
  CHECK(view.get<float>({3, 3}) == 7.0f);
  CHECK(view.get<float>({0, 1}) == 1.5f);   // outside the slice untouched
  CHECK(view.get<float>({2, 4}) == 42.0f);  // outside the slice untouched
  CHECK_THROWS(view.slice(1, 3, 2));
  CHECK_THROWS(view.slice(2, 0, 1));
}

TEST_CASE("blocking copies move data between host storages", "[llm][kv][storage]") {
  auto src = OwnedStorage::create(16 * sizeof(float), StorageKind::System);
  auto dst = OwnedStorage::create(16 * sizeof(float), StorageKind::Pinned);
  CHECK(dst.kind() == StorageKind::Pinned);

  TensorView<2> src_view(src, {4, 4}, DType::F32);
  TensorView<2> dst_view(dst, {4, 4}, DType::F32);
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j) src_view.set<float>({i, j}, static_cast<float>(i * 4 + j));
  }

  src_view.copy_to_view_blocking(dst_view);
  CHECK(dst_view.get<float>({3, 2}) == 14.0f);
  CHECK(dst_view.get<float>({0, 0}) == 0.0f);

  TensorView<2> small(dst, {2, 2}, DType::F32);
  CHECK_THROWS(src_view.copy_to_view_blocking(small));  // shape mismatch
}

#ifndef DYNAMO_HAVE_CUDA
TEST_CASE("device storage is unavailable without CUDA", "[llm][kv][storage]") {
  CHECK_THROWS(OwnedStorage::create(64, StorageKind::Device));
}
#endif

// ---------------------------------------------------------------------------
// Layers + copy streams (layer.rs tests, CPU backend)
// ---------------------------------------------------------------------------

namespace {

KvBlockDetails test_details(KvLayout layout) {
  KvBlockDetails details;
  details.layout = layout;
  details.block_size = 2;
  details.model_details = KvModelDetails{2, 2, 4, DType::F32};  // 2 layers, 2 heads, head_size 4
  return details;
}

/// Stamps every element of every block with its block id (the Rust tests'
/// fill_layer_with_block_id helper).
void fill_with_block_ids(KvBlockStorage& storage) {
  size_t block_dim = storage.block_details().layout == KvLayout::KvFirst ? 1 : 0;
  for (size_t l = 0; l < storage.layer_count(); ++l) {
    auto view = storage.layer(l).view();
    for (size_t b = 0; b < storage.number_of_blocks(); ++b) {
      view.slice(block_dim, b, b + 1).fill(static_cast<float>(b));
    }
  }
}

float block_id_at(const KvBlockStorage& storage, size_t layer, size_t block) {
  size_t block_dim = storage.block_details().layout == KvLayout::KvFirst ? 1 : 0;
  auto view = storage.layer(layer).view();
  std::array<size_t, 5> idx{};
  idx[block_dim] = block;
  idx[2] = 1;  // sample an interior element
  idx[4] = 3;
  return view.get<float>(idx);
}

}  // namespace

TEST_CASE("kv layers allocate, validate, and expose 5D views", "[llm][kv][layer]") {
  for (KvLayout layout : {KvLayout::KvFirst, KvLayout::BlockFirst}) {
    auto storage = KvBlockStorage::allocate(4, test_details(layout), StorageKind::System);
    REQUIRE(storage.layer_count() == 2);
    REQUIRE(storage.number_of_blocks() == 4);

    auto shape = storage.layer(0).layer_shape();
    if (layout == KvLayout::KvFirst) {
      CHECK(shape == std::array<size_t, 5>{2, 4, 2, 2, 4});
    } else {
      CHECK(shape == std::array<size_t, 5>{4, 2, 2, 2, 4});
    }
    CHECK(storage.layer(0).view().num_elements() == 2 * 4 * 2 * 2 * 4);
  }

  // tp must divide heads; rank must be in range.
  auto bad = test_details(KvLayout::KvFirst);
  bad.tp_size = 3;
  CHECK_THROWS(KvBlockStorage::allocate(4, bad, StorageKind::System));
  bad.tp_size = 2;
  bad.tp_rank = 2;
  CHECK_THROWS(KvBlockStorage::allocate(4, bad, StorageKind::System));
}

TEST_CASE("copy_blocks_to moves selected blocks between layers", "[llm][kv][layer]") {
  for (KvLayout layout : {KvLayout::KvFirst, KvLayout::BlockFirst}) {
    auto src = KvBlockStorage::allocate(4, test_details(layout), StorageKind::System);
    auto dst = KvBlockStorage::allocate(4, test_details(layout), StorageKind::System);
    fill_with_block_ids(src);

    src.layer(0).copy_blocks_to({0, 1, 2, 3}, dst.layer(0), {3, 2, 1, 0});

    for (size_t b = 0; b < 4; ++b) {
      CHECK(block_id_at(dst, 0, 3 - b) == static_cast<float>(b));
      CHECK(block_id_at(dst, 1, b) == 0.0f);  // other layers untouched
    }

    CHECK_THROWS(src.layer(0).copy_blocks_to({0, 1}, dst.layer(0), {0}));
  }
}

TEST_CASE("copy streams stage a block map and trigger layer copies", "[llm][kv][layer]") {
  auto src = KvBlockStorage::allocate(4, test_details(KvLayout::KvFirst), StorageKind::System);
  auto dst = KvBlockStorage::allocate(6, test_details(KvLayout::KvFirst), StorageKind::System);
  fill_with_block_ids(src);

  auto map = CopyStreamBlockMap::create(src, dst);
  CHECK(map->src_block_dim == 4);
  CHECK(map->dst_block_dim == 6);

  CopyStream stream(2, 4);
  stream.prepare_block_map(map);
  CHECK_THROWS(stream.prepare_block_map(map));  // already staged
  CHECK_THROWS(stream.prepare_block_ids({0, 0}, {1, 2}));  // duplicate src ids
  stream.prepare_block_ids({0, 2}, {5, 1});
  REQUIRE(stream.layer_count() == 2);

  stream.trigger_all_layers();
  stream.trigger_layer(0);  // doorbell: harmless double trigger
  stream.sync();

  for (size_t layer = 0; layer < 2; ++layer) {
    CHECK(block_id_at(dst, layer, 5) == 0.0f);
    CHECK(block_id_at(dst, layer, 1) == 2.0f);
    CHECK(block_id_at(dst, layer, 0) == 0.0f);  // untouched
  }

  // reset() allows a new staging cycle on the same stream.
  stream.reset();
  stream.prepare_block_map(map);
  stream.prepare_block_ids({1}, {0});
  stream.trigger_all_layers();
  stream.sync();
  CHECK(block_id_at(dst, 0, 0) == 1.0f);

  // Incompatible bundles are rejected at map construction.
  auto other = KvBlockStorage::allocate(4, test_details(KvLayout::BlockFirst),
                                        StorageKind::System);
  CHECK_THROWS(CopyStreamBlockMap::create(src, other));
}

TEST_CASE("scatter_copy_layer reshards heads across tensor-parallel ranks", "[llm][kv][layer]") {
  // KvFirst layer: [kv=2, blocks=3, bs=2, heads=4, hs=2], f32, tp 1 → 2.
  auto details = test_details(KvLayout::KvFirst);
  details.model_details.number_of_heads = 4;
  details.model_details.head_size = 2;
  auto src = KvBlockStorage::allocate(3, details, StorageKind::System);
  auto dst = KvBlockStorage::allocate(3, details, StorageKind::System);

  // Stamp every element with its flat index.
  {
    auto view = src.layer(0).view();
    for (size_t flat = 0; flat < view.num_elements(); ++flat) {
      view.set<float>(view.unflatten(flat), static_cast<float>(flat));
    }
  }

  CopyStream stream(2, 3);
  stream.prepare_block_map(CopyStreamBlockMap::create(src, dst));
  stream.prepare_block_ids({0, 1, 2}, {0, 1, 2});

  const size_t kv = 2, blocks = 3, bs = 2, heads = 4, hs = 2;
  const size_t src_tp = 1, dst_tp = 2;
  stream.scatter_copy_layer(0, {kv, blocks, bs, heads, hs}, sizeof(float),
                            /*block_dim_index=*/1, src_tp, dst_tp);
  stream.sync();

  // Element src[k][b][p][h][e] must land at dst6d[r][k][b][p][h'][e] with
  // r = h / (heads/scatter), h' = h % (heads/scatter) — derived from the
  // documented permutation, independently of the backend's stride math.
  const size_t scatter = dst_tp / src_tp;      // 2
  const size_t dnh = heads / scatter;          // heads per dst rank
  auto src_view = src.layer(0).view();
  const auto* dst_data = reinterpret_cast<const float*>(dst.layer(0).storage().pointer());
  for (size_t k = 0; k < kv; ++k) {
    for (size_t b = 0; b < blocks; ++b) {
      for (size_t p = 0; p < bs; ++p) {
        for (size_t h = 0; h < heads; ++h) {
          for (size_t e = 0; e < hs; ++e) {
            size_t r = h / dnh;
            size_t hd = h % dnh;
            size_t dst_flat =
                ((((r * kv + k) * blocks + b) * bs + p) * dnh + hd) * hs + e;
            CHECK(dst_data[dst_flat] == src_view.get<float>({k, b, p, h, e}));
          }
        }
      }
    }
  }

  CHECK_THROWS(stream.scatter_copy_layer(0, {kv, blocks, bs, heads, hs}, 4, 1, 2, 2));  // src==dst
  CHECK_THROWS(stream.scatter_copy_layer(0, {kv, blocks, bs, heads, hs}, 4, 2, 1, 2));  // bad dim
}
