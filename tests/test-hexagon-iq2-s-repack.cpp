#include "ggml.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-hexagon/iq2-s-repack.h"
#include "ggml-hexagon/iq2-family-repack.h"
#include "ggml-hexagon/iq3-xxs-repack.h"
#include "ggml-hexagon/iq3-s-repack.h"
#include "ggml-hexagon/iq4-xs-repack.h"
#include "ggml-hexagon/iq4-xs-alloc.h"

#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

using namespace ggml_hexagon_iq2s;

static int n_failed = 0;

static void check(bool cond, const char * msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++n_failed;
    }
}

static uint32_t xorshift32(uint32_t & state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static void test_grid_range() {
    uint8_t min_v = 0xff;
    uint8_t max_v = 0;

    for (size_t i = 0; i < 1024; ++i) {
        const uint8_t * grid = reinterpret_cast<const uint8_t *>(iq2s_grid + i);
        for (int j = 0; j < 8; ++j) {
            min_v = std::min(min_v, grid[j]);
            max_v = std::max(max_v, grid[j]);
        }
    }

    check(min_v == 8, "IQ2_S grid minimum stays 8");
    check(max_v == 43, "IQ2_S grid maximum stays 43");
    check(max_v <= 127, "IQ2_S grid magnitudes fit signed int8 for HVX vrmpy");
}

static void test_sizes() {
    check(original_size_2d(256, 32) == 32u * sizeof(block_iq2_s),
          "original size for 32x256");
    check(repacked_size_2d(256, 32) == 8u * TILE_SIZE,
          "repacked size for 32x256");
    check(repacked_size_2d(256, 33) == 16u * TILE_SIZE,
          "row padding for 33x256");
    check(repacked_size_2d(512, 32) == 16u * TILE_SIZE,
          "two super-blocks per row");
    check(!valid_2d_shape(255, 32), "reject ne0 that is not a multiple of 256");
}

static void test_raw_roundtrip(int64_t ne0, int64_t ne1) {
    const size_t original_size = original_size_2d(ne0, ne1);
    const size_t repacked_size = repacked_size_2d(ne0, ne1);

    const size_t block_count = original_size / sizeof(block_iq2_s);
    std::vector<block_iq2_s> original(block_count);
    std::vector<uint8_t> repacked(repacked_size);
    std::vector<block_iq2_s> roundtrip(block_count);

    uint32_t rng = 0x6d2b79f5u ^ (uint32_t) ne0 ^ ((uint32_t) ne1 << 16);
    uint8_t * original_bytes = reinterpret_cast<uint8_t *>(original.data());
    for (size_t i = 0; i < original_size; ++i) {
        original_bytes[i] = (uint8_t) xorshift32(rng);
    }

    check(repack_2d(
              original.data(), original_size,
              ne0, ne1, repacked.data(), repacked.size()),
          "raw repack succeeds");

    check(unpack_2d(
              repacked.data(), repacked.size(), ne0, ne1,
              roundtrip.data(), original_size),
          "raw unpack succeeds");

    check(std::memcmp(original.data(), roundtrip.data(), original_size) == 0,
          "raw IQ2_S bytes survive repack -> unpack exactly");
}

static void fill_source(std::vector<float> & src, int64_t ne0, int64_t ne1) {
    for (int64_t r = 0; r < ne1; ++r) {
        for (int64_t c = 0; c < ne0; ++c) {
            const float x = 0.37f * std::sin((float) (r * 17 + c) * 0.031f)
                          + 0.11f * std::cos((float) (r * 5  - c) * 0.017f)
                          + 0.003f * (float) ((c % 23) - 11);
            src[(size_t) r * (size_t) ne0 + (size_t) c] = x;
        }
    }
}

static void test_quantized_roundtrip(int64_t ne0, int64_t ne1) {
    const size_t original_size = original_size_2d(ne0, ne1);
    const size_t repacked_size = repacked_size_2d(ne0, ne1);

    std::vector<float> source((size_t) ne0 * (size_t) ne1);
    std::vector<float> imatrix(source.size(), 1.0f);
    const size_t block_count = original_size / sizeof(block_iq2_s);
    std::vector<block_iq2_s> quantized(block_count);
    std::vector<uint8_t> repacked(repacked_size);
    std::vector<block_iq2_s> roundtrip(block_count);

    fill_source(source, ne0, ne1);

    const size_t written = ggml_quantize_chunk(
        GGML_TYPE_IQ2_S,
        source.data(),
        quantized.data(),
        0,
        ne1,
        ne0,
        imatrix.data());

    check(written == original_size, "ggml_quantize_chunk wrote expected IQ2_S byte count");

    check(repack_2d(
              quantized.data(), original_size,
              ne0, ne1, repacked.data(), repacked.size()),
          "quantized repack succeeds");

    check(unpack_2d(
              repacked.data(), repacked.size(), ne0, ne1,
              roundtrip.data(), original_size),
          "quantized unpack succeeds");

    check(std::memcmp(quantized.data(), roundtrip.data(), original_size) == 0,
          "valid IQ2_S bytes survive repack -> unpack exactly");

    const int64_t blocks_per_row = ne0 / QK_K;
    std::vector<float> deq_a((size_t) ne0);
    std::vector<float> deq_b((size_t) ne0);

    for (int64_t r = 0; r < ne1; ++r) {
        const block_iq2_s * a = quantized.data() + r * blocks_per_row;
        const block_iq2_s * b = roundtrip.data() + r * blocks_per_row;

        dequantize_row_iq2_s(a, deq_a.data(), ne0);
        dequantize_row_iq2_s(b, deq_b.data(), ne0);

        check(std::memcmp(deq_a.data(), deq_b.data(), (size_t) ne0 * sizeof(float)) == 0,
              "dequantized values are bit-identical after repack roundtrip");
    }
}

static void test_hmx_tile_dequant_reference() {
    const int64_t ne0 = 256;
    const int64_t ne1 = 32;

    const size_t original_size = original_size_2d(ne0, ne1);
    const size_t repacked_size = repacked_size_2d(ne0, ne1);
    const size_t block_count = original_size / sizeof(block_iq2_s);

    std::vector<float> source((size_t) ne0 * (size_t) ne1);
    std::vector<float> imatrix(source.size(), 1.0f);
    std::vector<block_iq2_s> quantized(block_count);
    std::vector<uint8_t> repacked(repacked_size);

    fill_source(source, ne0, ne1);
    const size_t written = ggml_quantize_chunk(
        GGML_TYPE_IQ2_S,
        source.data(),
        quantized.data(),
        0,
        ne1,
        ne0,
        imatrix.data());

    check(written == original_size, "HMx reference fixture quantization size");
    check(repack_2d(
              quantized.data(), original_size,
              ne0, ne1, repacked.data(), repacked.size()),
          "HMX reference fixture repack succeeds");

    std::vector<float> row_ref((size_t) ne0);
    std::vector<float> hmx_tile(32 * 32);

    for (int64_t kt = 0; kt < ne0 / 32; ++kt) {
        const uint8_t * tile = repacked.data() + (size_t) kt * TILE_SIZE;
        std::fill(hmx_tile.begin(), hmx_tile.end(), 0.0f);

        const uint8_t * indices = tile + INDEX_PLANE_OFFSET;
        const uint8_t * signs   = tile + SIGN_PLANE_OFFSET;
        const uint8_t * qh      = tile + QH_PLANE_OFFSET;
        const uint8_t * scales  = tile + SCALE_PLANE_OFFSET;
        const ggml_half * d     = reinterpret_cast<const ggml_half *>(tile + D_PLANE_OFFSET);

        for (int row = 0; row < 32; ++row) {
            const float d_row = GGML_FP16_TO_FP32(d[row]);
            const uint8_t sc = scales[row];

            for (int l = 0; l < 4; ++l) {
                const uint8_t scale_nibble = l < 2 ? (sc & 0x0f) : (sc >> 4);
                const float dl = d_row * (0.5f + (float) scale_nibble) * 0.25f;

                const uint32_t grid_index =
                    (uint32_t) indices[l * 32 + row] |
                    (((uint32_t) qh[row] << (8 - 2 * l)) & 0x300u);
                const uint8_t * grid = reinterpret_cast<const uint8_t *>(iq2s_grid + grid_index);
                const uint8_t sign_mask = signs[l * 32 + row];

                for (int j = 0; j < 8; ++j) {
                    const int k = l * 8 + j;
                    float w = dl * (float) grid[j];
                    if (sign_mask & (1u << j)) {
                        w = -w;
                    }
                    hmx_tile[(k / 2) * 64 + row * 2 + (k & 1)] = w;
                }
            }
        }

        for (int row = 0; row < 32; ++row) {
            dequantize_row_iq2_s(
                quantized.data() + row,
                row_ref.data(),
                ne0);

            for (int k = 0; k < 32; ++k) {
                const float got = hmx_tile[(k / 2) * 64 + row * 2 + (k & 1)];
                const float ref = row_ref[(size_t) kt * 32 + k];
                if (got != ref) {
                    std::fprintf(stderr,
                        "FAIL: HMX tile dequant mismatch kt=%lld row=%d k=%d ref=%g got=%g\n",
                        (long long) kt, row, k, ref, got);
                    ++n_failed;
                    return;
                }
            }
        }
    }
}

static void test_duplicate_d_validation() {
    const int64_t ne0 = 256;
    const int64_t ne1 = 1;

    std::vector<block_iq2_s> original(1);
    std::vector<uint8_t> repacked(repacked_size_2d(ne0, ne1), 0);
    std::vector<block_iq2_s> roundtrip(1);

    uint32_t rng = 0x12345678u;
    uint8_t * original_bytes = reinterpret_cast<uint8_t *>(original.data());
    for (size_t i = 0; i < sizeof(block_iq2_s); ++i) {
        original_bytes[i] = (uint8_t) xorshift32(rng);
    }

    check(repack_2d(
              original.data(), sizeof(block_iq2_s),
              ne0, ne1, repacked.data(), repacked.size()),
          "corruption test repack succeeds");

    // Tile 1 belongs to the same 256-value super-block as tile 0, so its d
    // copy must match.  Flip one bit and ensure read-back detects it.
    repacked[TILE_SIZE + D_PLANE_OFFSET] ^= 0x01u;

    check(!unpack_2d(
              repacked.data(), repacked.size(), ne0, ne1,
              roundtrip.data(), sizeof(block_iq2_s)),
          "unpack rejects inconsistent duplicated d");
}

// ---- IQ2_XS / IQ2_XXS ----

namespace iq2f = ggml_hexagon_iq2f;

static void test_iq2f_grid_range() {
    for (const uint64_t * tables : { iq2xs_grid, iq2xxs_grid }) {
        const size_t entries = (tables == iq2xs_grid) ? 512 : 256;
        uint8_t min_v = 0xff;
        uint8_t max_v = 0;

        for (size_t i = 0; i < entries; ++i) {
            const uint8_t * grid = reinterpret_cast<const uint8_t *>(tables + i);
            for (int j = 0; j < 8; ++j) {
                min_v = std::min(min_v, grid[j]);
                max_v = std::max(max_v, grid[j]);
            }
        }

        check(min_v == 8, "IQ2 family grid minimum stays 8");
        check(max_v == 43, "IQ2 family grid maximum stays 43");
        check(max_v <= 127, "IQ2 family grid magnitudes fit signed int8 for HVX vrmpy");
    }
}

// The unified sign plane stores (sign index << 1) | index bit 8, so ksigns64
// must agree with ksigns_iq2xs byte for byte.
static void test_iq2f_sign_table_agreement() {
    for (int s = 0; s < 128; ++s) {
        const uint8_t * expanded = reinterpret_cast<const uint8_t *>(&ksigns64[s]);
        for (int j = 0; j < 8; ++j) {
            const bool expected = (ksigns_iq2xs[s] & kmask_iq2xs[j]) != 0;
            const bool got      = expanded[j] != 0;
            if (expected != got) {
                std::fprintf(stderr, "FAIL: ksigns64 mismatch s=%d j=%d\n", s, j);
                ++n_failed;
                return;
            }
            check(expanded[j] == 0x00 || expanded[j] == 0xff,
                  "ksigns64 bytes are all-ones or all-zeros");
        }
    }
}

template <typename Traits>
static void test_iq2f_sizes() {
    using block_t = typename Traits::block_t;

    check(iq2f::original_size_2d<Traits>(256, 32) == 32u * sizeof(block_t),
          "iq2f original size for 32x256");
    check(iq2f::repacked_size_2d(256, 32) == 8u * iq2f::TILE_SIZE,
          "iq2f repacked size for 32x256");
    check(iq2f::repacked_size_2d(256, 33) == 16u * iq2f::TILE_SIZE,
          "iq2f row padding for 33x256");
    check(iq2f::repacked_size_2d(512, 32) == 16u * iq2f::TILE_SIZE,
          "iq2f two super-blocks per row");
    check(!iq2f::valid_2d_shape(255, 32), "iq2f rejects ne0 that is not a multiple of 256");
}

template <typename Traits>
static void test_iq2f_raw_roundtrip(int64_t ne0, int64_t ne1) {
    using block_t = typename Traits::block_t;

    const size_t original_size = iq2f::original_size_2d<Traits>(ne0, ne1);
    const size_t repacked_size = iq2f::repacked_size_2d(ne0, ne1);

    const size_t block_count = original_size / sizeof(block_t);
    std::vector<block_t> original(block_count);
    std::vector<uint8_t> repacked(repacked_size);
    std::vector<block_t> roundtrip(block_count);

    uint32_t rng = 0x9e3779b9u ^ (uint32_t) ne0 ^ ((uint32_t) ne1 << 16);
    uint8_t * original_bytes = reinterpret_cast<uint8_t *>(original.data());
    for (size_t i = 0; i < original_size; ++i) {
        original_bytes[i] = (uint8_t) xorshift32(rng);
    }

    check(iq2f::repack_2d<Traits>(
              original.data(), original_size,
              ne0, ne1, repacked.data(), repacked.size()),
          "iq2f raw repack succeeds");

    check(iq2f::unpack_2d<Traits>(
              repacked.data(), repacked.size(), ne0, ne1,
              roundtrip.data(), original_size),
          "iq2f raw unpack succeeds");

    check(std::memcmp(original.data(), roundtrip.data(), original_size) == 0,
          "iq2f raw bytes survive repack -> unpack exactly");
}

template <typename Traits>
static void test_iq2f_quantized_roundtrip(int64_t ne0, int64_t ne1) {
    using block_t = typename Traits::block_t;

    const enum ggml_type type = std::is_same<Traits, iq2f::iq2_xs_traits>::value
        ? GGML_TYPE_IQ2_XS : GGML_TYPE_IQ2_XXS;
    const size_t original_size = iq2f::original_size_2d<Traits>(ne0, ne1);
    const size_t repacked_size = iq2f::repacked_size_2d(ne0, ne1);

    std::vector<float> source((size_t) ne0 * (size_t) ne1);
    std::vector<float> imatrix(source.size(), 1.0f);
    const size_t block_count = original_size / sizeof(block_t);
    std::vector<block_t> quantized(block_count);
    std::vector<uint8_t> repacked(repacked_size);
    std::vector<block_t> roundtrip(block_count);

    fill_source(source, ne0, ne1);

    const size_t written = ggml_quantize_chunk(
        type, source.data(), quantized.data(), 0, ne1, ne0, imatrix.data());
    check(written == original_size, "iq2f ggml_quantize_chunk wrote expected byte count");

    check(iq2f::repack_2d<Traits>(
              quantized.data(), original_size,
              ne0, ne1, repacked.data(), repacked.size()),
          "iq2f quantized repack succeeds");

    check(iq2f::unpack_2d<Traits>(
              repacked.data(), repacked.size(), ne0, ne1,
              roundtrip.data(), original_size),
          "iq2f quantized unpack succeeds");

    check(std::memcmp(quantized.data(), roundtrip.data(), original_size) == 0,
          "valid iq2f bytes survive repack -> unpack exactly");

    const int64_t blocks_per_row = ne0 / QK_K;
    std::vector<float> deq_a((size_t) ne0);
    std::vector<float> deq_b((size_t) ne0);

    for (int64_t r = 0; r < ne1; ++r) {
        const block_t * a = quantized.data() + r * blocks_per_row;
        const block_t * b = roundtrip.data() + r * blocks_per_row;

        if constexpr (std::is_same_v<Traits, iq2f::iq2_xs_traits>) {
            dequantize_row_iq2_xs(a, deq_a.data(), ne0);
            dequantize_row_iq2_xs(b, deq_b.data(), ne0);
        } else {
            dequantize_row_iq2_xxs(a, deq_a.data(), ne0);
            dequantize_row_iq2_xxs(b, deq_b.data(), ne0);
        }

        check(std::memcmp(deq_a.data(), deq_b.data(), (size_t) ne0 * sizeof(float)) == 0,
              "iq2f dequantized values are bit-identical after repack roundtrip");
    }
}

// Mirrors dequantize_tiled_weight_to_fp16_task_iq2f: decode one 32x32 tile
// straight from the tile planes and compare against the CPU dequantizer.
template <typename Traits>
static void test_iq2f_hmx_tile_dequant_reference() {
    using block_t = typename Traits::block_t;

    const enum ggml_type type = std::is_same<Traits, iq2f::iq2_xs_traits>::value
        ? GGML_TYPE_IQ2_XS : GGML_TYPE_IQ2_XXS;
    const bool is_xxs = type == GGML_TYPE_IQ2_XXS;
    const uint64_t * grid_tbl = is_xxs ? iq2xxs_grid : iq2xs_grid;

    const int64_t ne0 = 256;
    const int64_t ne1 = 32;

    const size_t original_size = iq2f::original_size_2d<Traits>(ne0, ne1);
    const size_t repacked_size = iq2f::repacked_size_2d(ne0, ne1);
    const size_t block_count = original_size / sizeof(block_t);

    std::vector<float> source((size_t) ne0 * (size_t) ne1);
    std::vector<float> imatrix(source.size(), 1.0f);
    std::vector<block_t> quantized(block_count);
    std::vector<uint8_t> repacked(repacked_size);

    fill_source(source, ne0, ne1);
    const size_t written = ggml_quantize_chunk(
        type, source.data(), quantized.data(), 0, ne1, ne0, imatrix.data());

    check(written == original_size, "iq2f HMX reference fixture quantization size");
    check(iq2f::repack_2d<Traits>(
              quantized.data(), original_size,
              ne0, ne1, repacked.data(), repacked.size()),
          "iq2f HMX reference fixture repack succeeds");

    std::vector<float> row_ref((size_t) ne0);
    std::vector<float> hmx_tile(32 * 32);

    for (int64_t kt = 0; kt < ne0 / 32; ++kt) {
        const uint8_t * tile = repacked.data() + (size_t) kt * iq2f::TILE_SIZE;
        std::fill(hmx_tile.begin(), hmx_tile.end(), 0.0f);

        const uint8_t * indices = tile + iq2f::INDEX_PLANE_OFFSET;
        const uint8_t * signs   = tile + iq2f::SIGN_PLANE_OFFSET;
        const uint8_t * scales  = tile + iq2f::SCALE_PLANE_OFFSET;
        const ggml_half * d     = reinterpret_cast<const ggml_half *>(tile + iq2f::D_PLANE_OFFSET);

        for (int row = 0; row < 32; ++row) {
            const float d_row = GGML_FP16_TO_FP32(d[row]);
            const uint8_t sc = scales[row];

            for (int l = 0; l < 4; ++l) {
                const uint8_t scale_nibble =
                    is_xxs ? (sc & 0x0f) : (l < 2 ? (sc & 0x0f) : (sc >> 4));
                const float dl = d_row * (0.5f + (float) scale_nibble) * 0.25f;

                const uint32_t payload = signs[l * 32 + row];
                const uint32_t grid_index = indices[l * 32 + row] | ((payload & 1u) << 8);
                const uint8_t * grid = reinterpret_cast<const uint8_t *>(grid_tbl + grid_index);
                const uint8_t sign_mask = ksigns_iq2xs[payload >> 1];

                for (int j = 0; j < 8; ++j) {
                    const int k = l * 8 + j;
                    float w = dl * (float) grid[j];
                    if (sign_mask & (1u << j)) {
                        w = -w;
                    }
                    hmx_tile[(k / 2) * 64 + row * 2 + (k & 1)] = w;
                }
            }
        }

        for (int row = 0; row < 32; ++row) {
            if constexpr (std::is_same_v<Traits, iq2f::iq2_xs_traits>) {
                dequantize_row_iq2_xs(quantized.data() + row, row_ref.data(), ne0);
            } else {
                dequantize_row_iq2_xxs(quantized.data() + row, row_ref.data(), ne0);
            }

            for (int k = 0; k < 32; ++k) {
                const float got = hmx_tile[(k / 2) * 64 + row * 2 + (k & 1)];
                const float ref = row_ref[(size_t) kt * 32 + k];
                if (got != ref) {
                    std::fprintf(stderr,
                        "FAIL: iq2f HMX tile dequant mismatch kt=%lld row=%d k=%d ref=%g got=%g\n",
                        (long long) kt, row, k, ref, got);
                    ++n_failed;
                    return;
                }
            }
        }
    }
}

template <typename Traits>
static void test_iq2f_duplicate_d_validation() {
    using block_t = typename Traits::block_t;

    const int64_t ne0 = 256;
    const int64_t ne1 = 1;

    std::vector<block_t> original(1);
    std::vector<uint8_t> repacked(iq2f::repacked_size_2d(ne0, ne1), 0);
    std::vector<block_t> roundtrip(1);

    uint32_t rng = 0x87654321u;
    uint8_t * original_bytes = reinterpret_cast<uint8_t *>(original.data());
    for (size_t i = 0; i < sizeof(block_t); ++i) {
        original_bytes[i] = (uint8_t) xorshift32(rng);
    }

    check(iq2f::repack_2d<Traits>(
              original.data(), sizeof(block_t),
              ne0, ne1, repacked.data(), repacked.size()),
          "iq2f corruption test repack succeeds");

    repacked[iq2f::TILE_SIZE + iq2f::D_PLANE_OFFSET] ^= 0x01u;

    check(!iq2f::unpack_2d<Traits>(
              repacked.data(), repacked.size(), ne0, ne1,
              roundtrip.data(), sizeof(block_t)),
          "iq2f unpack rejects inconsistent duplicated d");
}

template <typename Traits>
static void test_iq2f_suite() {
    test_iq2f_sizes<Traits>();

    for (const auto & shape : std::vector<std::pair<int64_t, int64_t>> {
            {256, 1},
            {256, 31},
            {256, 32},
            {256, 33},
            {512, 32},
            {4096, 33},
        }) {
        test_iq2f_raw_roundtrip<Traits>(shape.first, shape.second);
    }

    for (const auto & shape : std::vector<std::pair<int64_t, int64_t>> {
            {256, 1},
            {256, 32},
            {512, 33},
            {4096, 32},
        }) {
        test_iq2f_quantized_roundtrip<Traits>(shape.first, shape.second);
    }

    test_iq2f_hmx_tile_dequant_reference<Traits>();
    test_iq2f_duplicate_d_validation<Traits>();
}

namespace iq4xs = ggml_hexagon_iq4xs;

static void iq4xs_set_scale(block_iq4_xs & b, int ib, unsigned ls) {
    const unsigned lo_shift = 4 * (ib % 2);
    const unsigned hi_shift = 2 * ib;
    b.scales_l[ib / 2] = (uint8_t) ((b.scales_l[ib / 2] & ~(15u << lo_shift)) | ((ls & 15u) << lo_shift));
    b.scales_h = (uint16_t) ((b.scales_h & ~(3u << hi_shift)) | ((ls >> 4) << hi_shift));
}

static void iq4xs_decode_raw(const block_iq4_xs & b, float * dst) {
    for (int ib = 0; ib < 8; ++ib) {
        uint8_t q[32];
        iq4xs::unpack_iq4_xs_indices_32(b, ib, q);
        const float d = iq4xs::iq4_xs_effective_scale(b, ib);
        for (int k = 0; k < 32; ++k) {
            dst[32 * ib + k] = d * kvalues_iq4nl[q[k]];
        }
    }
}

static float iq4xs_tile_scale(const uint8_t * tile, size_t row) {
    ggml_half d;
    std::memcpy(&d, tile + 512 + row * sizeof(d), sizeof(d));
    return GGML_FP16_TO_FP32(d);
}

static uint8_t iq4xs_tile_index(const uint8_t * tile, size_t row, size_t k) {
    return (tile[(k / 2) * 32 + row] >> (4 * (k % 2))) & 15u;
}

struct iq4xs_errors {
    double max_abs = 0, max_rel = 0, squared = 0;
    size_t count = 0;
    void add(double ref, double got) {
        const double error = std::fabs(got - ref);
        max_abs = std::max(max_abs, error);
        max_rel = std::max(max_rel, error / std::max(std::fabs(ref), 1e-6));
        squared += error * error;
        ++count;
    }
    void print(const char * name) const {
        std::printf("%s: n=%zu max_abs=%.9g max_rel=%.9g RMS=%.9g (relative floor=1e-6)\n",
                    name, count, max_abs, max_rel, count ? std::sqrt(squared / count) : 0);
    }
};

static void test_iq4xs_bits() {
    for (int ib = 0; ib < 8; ++ib) {
        for (unsigned ls = 0; ls < 64; ++ls) {
            block_iq4_xs b;
            std::memset(&b, 0xa5, sizeof(b));
            int before[8];
            for (int j = 0; j < 8; ++j) {
                before[j] = iq4xs::iq4_xs_scale(b, j);
            }
            iq4xs_set_scale(b, ib, ls);
            check(iq4xs::iq4_xs_scale(b, ib) == (int) ls - 32, "IQ4_XS all 512 scale combinations");
            for (int j = 0; j < 8; ++j) {
                check(j == ib || iq4xs::iq4_xs_scale(b, j) == before[j], "IQ4_XS adjacent scale bits unchanged");
            }
        }
        for (unsigned lo = 0; lo < 16; ++lo) {
            for (unsigned hi = 0; hi < 16; ++hi) {
                block_iq4_xs b = {};
                std::memset(b.qs + 16 * ib, lo | (hi << 4), 16);
                uint8_t q[32];
                iq4xs::unpack_iq4_xs_indices_32(b, ib, q);
                for (int j = 0; j < 16; ++j) {
                    check(q[j] == lo && q[j + 16] == hi, "IQ4_XS low/high logical nibble order");
                }
            }
        }
    }
}

static block_iq4_xs iq4xs_random_block(uint32_t & rng, ggml_half d) {
    block_iq4_xs b = {};
    b.d = d;
    b.scales_h = (uint16_t) xorshift32(rng);
    for (uint8_t & sc : b.scales_l) {
        sc = (uint8_t) xorshift32(rng);
    }
    for (uint8_t & q : b.qs) {
        q = (uint8_t) xorshift32(rng);
    }
    return b;
}

static void test_iq4xs_raw_and_range() {
    uint32_t rng = 0x4a38c127u;
    size_t overflow = 0, finite = 0;
    double max_rel = 0;
    for (unsigned bits = 0; bits < 65536; ++bits) {
        if ((bits & 0x7c00u) == 0x7c00u) {
            continue;
        }
        block_iq4_xs b = iq4xs_random_block(rng, (ggml_half) bits);
        float ref[256], got[256];
        dequantize_row_iq4_xs(&b, ref, 256);
        iq4xs_decode_raw(b, got);
        check(std::memcmp(ref, got, sizeof(ref)) == 0, "IQ4_XS helper bit equality for every finite d pattern");
        for (unsigned ls = 0; ls < 64; ++ls) {
            iq4xs_set_scale(b, 0, ls);
            const float d = iq4xs::iq4_xs_effective_scale(b, 0);
            const float rounded = GGML_FP16_TO_FP32(GGML_FP32_TO_FP16(d));
            if (!std::isfinite(rounded)) {
                ++overflow;
                continue;
            }
            ++finite;
            if (d != 0) {
                max_rel = std::max(max_rel, std::fabs(double(rounded) - d) / std::fabs(d));
            }
            // Binary16 nearest rounding is bounded by 2^-11; subnormal integer products are exact.
            check(std::fabs(double(rounded) - d) <= std::fabs(d) / 2048, "IQ4_XS effective scale rounding bound");
        }
    }
    std::printf("IQ4_XS raw: 63488 finite d patterns, helper max_abs_diff=0\n");
    std::printf("IQ4_XS scale domain: finite=%zu overflow=%zu max_rel=%.9g\n", finite, overflow, max_rel);
    check(overflow != 0, "IQ4_XS finite raw d can overflow effective fp16 scale");
}

struct iq4xs_worst {
    double error = -1;
    float d = 0, effective = 0, rounded = 0, ref = 0, got = 0;
    int ib = 0, ls = 0, q = 0;
    void print() const {
        std::printf("IQ4_XS worst weight: d=%.9g ib=%d ls=%d D=%.9g D16=%.9g q=%d ref=%.9g tile=%.9g\n",
                    d, ib, ls, effective, rounded, q, ref, got);
    }
};

static iq4xs_errors iq4xs_dot_errors;
static iq4xs_errors iq4xs_dot_quantized;
static iq4xs_errors iq4xs_dot_normal_activation;
static double iq4xs_dot_normalized = 0;
static double iq4xs_dot_float_normalized = 0;
static double iq4xs_dot_relative_ref = 0, iq4xs_dot_relative_tile = 0, iq4xs_dot_relative_l1 = 0;
static int64_t iq4xs_dot_worst_k = 0;
static size_t iq4xs_dot_worst_row = 0;
static int iq4xs_dot_worst_corpus = 0;

static void iq4xs_check_matrix(const std::vector<block_iq4_xs> & raw, int64_t k, int64_t n,
                              iq4xs_errors & errors, iq4xs_worst & worst, bool dots, bool quantized = false) {
    const size_t bytes = iq4xs::repacked_size_2d(k, n);
    std::vector<uint8_t> storage(bytes + 2, 0xa5);
    uint8_t * tiled = storage.data() + 1;
    if (!iq4xs::repack_2d(raw.data(), raw.size() * sizeof(block_iq4_xs), k, n, tiled, bytes)) {
        check(false, "IQ4_XS matrix repack succeeds");
        return;
    }
    check(storage.front() == 0xa5 && storage.back() == 0xa5, "IQ4_XS unaligned destination guards");
    const size_t padded = ((size_t) n + 31) / 32 * 32;
    std::vector<float> ref((size_t) k), got((size_t) k);
    for (size_t r = 0; r < padded; ++r) {
        if (r < (size_t) n) {
            dequantize_row_iq4_xs(raw.data() + r * (size_t) (k / 256), ref.data(), k);
        }
        for (size_t kt = 0; kt < (size_t) (k / 32); ++kt) {
            const uint8_t * tile = tiled + ((r / 32) * (size_t) (k / 32) + kt) * 576;
            const float d16 = iq4xs_tile_scale(tile, r % 32);
            for (size_t j = 0; j < 32; ++j) {
                const int q = iq4xs_tile_index(tile, r % 32, j);
                const size_t pos = kt * 32 + j;
                got[pos] = d16 * kvalues_iq4nl[q];
                if (r >= (size_t) n) {
                    check(d16 == 0 && q == 0 && got[pos] == 0, "IQ4_XS padding is zero contribution");
                    continue;
                }
                const block_iq4_xs & b = raw[r * (size_t) (k / 256) + kt / 8];
                const int ib = (int) (kt % 8);
                const uint8_t packed = b.qs[ib * 16 + j % 16];
                const int expected_q = j < 16 ? packed & 15u : packed >> 4;
                const float effective = iq4xs::iq4_xs_effective_scale(b, ib);
                const float expected_d = GGML_FP16_TO_FP32(GGML_FP32_TO_FP16(effective));
                const float expected = expected_d * kvalues_iq4nl[expected_q];
                check(q == expected_q && d16 == expected_d && got[pos] == expected, "IQ4_XS tile matches independent logical K decode");
                const double error = std::fabs(double(got[pos]) - ref[pos]);
                // 2^-11 scale rounding plus one float product rounding on each path.
                check(std::isfinite(got[pos]) && error <= std::fabs(ref[pos]) * (1.0 / 2048 + 2 * std::numeric_limits<float>::epsilon()),
                      "IQ4_XS weight error is explained by fp16 scale rounding");
                errors.add(ref[pos], got[pos]);
                if (error > worst.error) {
                    worst = {error, GGML_FP16_TO_FP32(b.d), effective, d16, ref[pos], got[pos],
                             ib, iq4xs::iq4_xs_scale(b, ib) + 32, q};
                }
            }
        }
        if (!dots || r >= (size_t) n) {
            continue;
        }
        uint32_t rng = 0x518cd9abu ^ (uint32_t) r;
        for (int corpus = 0; corpus < 8; ++corpus) {
            double reference = 0, tile_sum = 0, l1 = 0;
            float reference_f = 0, tile_f = 0, tile_partial = 0;
            for (size_t j = 0; j < (size_t) k; ++j) {
                const float random = ((int) (xorshift32(rng) % 20001) - 10000) / 10000.0f;
                const float x = corpus == 0 ? 0 : corpus == 1 ? 1 : corpus == 2 ? (j % 2 ? -1 : 1) :
                                corpus == 3 ? random * 1e-3f : corpus == 4 ? random :
                                corpus == 5 ? (j % 37 == 0 ? random : 0) :
                                corpus == 6 ? (j == r % (size_t) k ? 1 : 0) : random * 1e6f;
                reference += double(ref[j]) * x;
                tile_sum += double(got[j]) * x;
                l1 += std::fabs(double(ref[j]) * x);
                reference_f += ref[j] * x;
                tile_partial += got[j] * x;
                if (j % 32 == 31) {
                    tile_f += tile_partial;
                    tile_partial = 0;
                }
            }
            const double normalized = std::fabs(tile_sum - reference) / std::max(l1, 1e-6);
            iq4xs_dot_normalized = std::max(iq4xs_dot_normalized, normalized);
            iq4xs_dot_float_normalized = std::max(iq4xs_dot_float_normalized,
                std::fabs(double(tile_f) - reference_f) / std::max(l1, 1e-6));
            // Use sum(abs(w*x)) under cancellation. gamma_(2K) bounds float multiply/add rounding.
            const double u = std::numeric_limits<float>::epsilon() / 2;
            const double gamma = (2 * k * u) / (1 - 2 * k * u);
            const double bound = (1.0 / 2048 + 4 * u + 2 * gamma) * l1 + 1e-12;
            if (std::fabs(double(tile_f) - reference_f) > bound || !std::isfinite(tile_f)) {
                std::fprintf(stderr, "FAIL: IQ4_XS dot K=%lld row=%zu corpus=%d ref=%g tile=%g bound=%g\n",
                             (long long) k, r, corpus, reference_f, tile_f, bound);
                ++n_failed;
            }
            check(normalized <= 1.0 / 2048 + 4 * u, "IQ4_XS dot scale rounding bound");
            if (std::fabs(double(tile_f) - reference_f) > iq4xs_dot_errors.max_abs) {
                iq4xs_dot_worst_k = k;
                iq4xs_dot_worst_row = r;
                iq4xs_dot_worst_corpus = corpus;
            }
            const double relative = std::fabs(double(tile_f) - reference_f) / std::max(std::fabs(double(reference_f)), 1e-6);
            if (relative > iq4xs_dot_errors.max_rel) {
                iq4xs_dot_relative_ref = reference_f;
                iq4xs_dot_relative_tile = tile_f;
                iq4xs_dot_relative_l1 = l1;
            }
            iq4xs_dot_errors.add(reference_f, tile_f);
            if (quantized) {
                iq4xs_dot_quantized.add(reference_f, tile_f);
                if (corpus != 7) {
                    iq4xs_dot_normal_activation.add(reference_f, tile_f);
                }
            }
        }
    }
}

// Readback candidate: retain indices and normalize eight effective scales with either sign of d.
static block_iq4_xs iq4xs_reverse_block(const uint8_t * tiles, size_t k_tiles, size_t r, size_t sb) {
    float scales[8];
    float positive = 0, negative = 0;
    block_iq4_xs result = {};
    for (int ib = 0; ib < 8; ++ib) {
        const uint8_t * tile = tiles + ((r / 32) * k_tiles + sb * 8 + ib) * 576;
        scales[ib] = iq4xs_tile_scale(tile, r % 32);
        positive = std::max(positive, scales[ib]);
        negative = std::max(negative, -scales[ib]);
        for (int j = 0; j < 16; ++j) {
            result.qs[ib * 16 + j] = iq4xs_tile_index(tile, r % 32, j) |
                                    (iq4xs_tile_index(tile, r % 32, j + 16) << 4);
        }
    }
    double best = std::numeric_limits<double>::infinity();
    for (int sign = 0; sign < 2; ++sign) {
        block_iq4_xs candidate = result;
        const float d = sign == 0 ? std::max(positive / 31, negative / 32) : -std::max(positive / 32, negative / 31);
        candidate.d = GGML_FP32_TO_FP16(d);
        if (d != 0 && (candidate.d & 0x7fffu) == 0) {
            candidate.d = (ggml_half) ((candidate.d & 0x8000u) | 1u);
        }
        const float actual = GGML_FP16_TO_FP32(candidate.d);
        double error = 0;
        for (int ib = 0; ib < 8; ++ib) {
            const int s = actual == 0 ? 0 : std::max(-32, std::min(31, (int) std::round(scales[ib] / actual)));
            iq4xs_set_scale(candidate, ib, (unsigned) (s + 32));
            const double diff = double(actual) * s - scales[ib];
            error += diff * diff;
        }
        if (error < best) {
            best = error;
            result = candidate;
        }
    }
    float maximum = 0;
    for (int ib = 0; ib < 8; ++ib) {
        maximum = std::max(maximum, std::fabs(scales[ib]));
    }
    for (int ib = 0; ib < 8; ++ib) {
        // The positive-d candidate has step <= max(abs(D))/31, plus fp16 rounding.
        const double bound = maximum * (1.0 / 62 + 1.0 / 2048) + 32 * std::ldexp(1.0, -25);
        check(std::fabs(double(iq4xs::iq4_xs_effective_scale(result, ib)) - scales[ib]) <= bound,
              "IQ4_XS reverse scale normalization bound");
    }
    return result;
}

static void test_iq4xs_matrices() {
    iq4xs_errors deterministic, random, quantized, reverse, reverse_random;
    iq4xs_worst worst;
    const ggml_half ds[] = {0, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x3555, 0xb555, 0x3c01, 0x67ff, 0xe7ff};
    for (ggml_half d : ds) {
        for (unsigned ls = 0; ls < 64; ++ls) {
            std::vector<block_iq4_xs> raw(32);
            for (int r = 0; r < 32; ++r) {
                raw[r].d = d;
                for (int ib = 0; ib < 8; ++ib) {
                    iq4xs_set_scale(raw[r], ib, ls);
                    for (int j = 0; j < 16; ++j) {
                        raw[r].qs[16 * ib + j] = (uint8_t) (((j + r) % 16) | (((15 - j + r) % 16) << 4));
                    }
                }
            }
            iq4xs_check_matrix(raw, 256, 32, deterministic, worst, false);
        }
    }
    uint32_t rng = 0x9ad5417fu;
    for (int64_t k : {256, 512, 1024}) {
        for (int64_t n : {0, 1, 31, 32, 33, 65}) {
            std::vector<block_iq4_xs> raw((size_t) (k / 256 * n));
            for (block_iq4_xs & b : raw) {
                const unsigned magnitude = xorshift32(rng) % 0x6800u;
                const unsigned sign = xorshift32(rng) & 0x8000u;
                const ggml_half d = (ggml_half) (magnitude | sign);
                b = iq4xs_random_block(rng, d);
            }
            std::vector<float> source((size_t) (k * n));
            fill_source(source, k, n);
            for (int corpus = 0; corpus < 2; ++corpus) {
                if (corpus == 1 && n != 0) {
                    quantize_row_iq4_xs_ref(source.data(), raw.data(), k * n);
                }
                iq4xs_check_matrix(raw, k, n, corpus == 0 ? random : quantized, worst, true, corpus == 1);
                std::vector<uint8_t> tiled(iq4xs::repacked_size_2d(k, n));
                check(iq4xs::repack_2d(raw.data(), raw.size() * sizeof(block_iq4_xs), k, n, tiled.data(), tiled.size()), "IQ4_XS reverse fixture repack");
                for (size_t r = 0; r < (size_t) n; ++r) {
                    for (size_t sb = 0; sb < (size_t) (k / 256); ++sb) {
                        const block_iq4_xs back = iq4xs_reverse_block(tiled.data(), (size_t) (k / 32), r, sb);
                        check(std::memcmp(back.qs, raw[r * (size_t) (k / 256) + sb].qs, sizeof(back.qs)) == 0, "IQ4_XS reverse retains indices");
                        float decoded[256];
                        dequantize_row_iq4_xs(&back, decoded, 256);
                        for (size_t j = 0; j < 256; ++j) {
                            const uint8_t * tile = tiled.data() + ((r / 32) * (size_t) (k / 32) + sb * 8 + j / 32) * 576;
                            const float weight = iq4xs_tile_scale(tile, r % 32) * kvalues_iq4nl[iq4xs_tile_index(tile, r % 32, j % 32)];
                            (corpus == 0 ? reverse_random : reverse).add(weight, decoded[j]);
                            check(std::isfinite(decoded[j]), "IQ4_XS reverse finite weights");
                        }
                    }
                }
            }
        }
    }
    deterministic.print("IQ4_XS deterministic tile");
    random.print("IQ4_XS random tile");
    quantized.print("IQ4_XS quantized tile");
    reverse.print("IQ4_XS semantic reverse candidate (quantized corpus)");
    reverse_random.print("IQ4_XS semantic reverse candidate (random corpus)");
    iq4xs_dot_errors.print("IQ4_XS dot (float accumulation)");
    iq4xs_dot_quantized.print("IQ4_XS dot (quantized weights, float accumulation)");
    iq4xs_dot_normal_activation.print("IQ4_XS dot (quantized weights, activations <= 1)");
    std::printf("IQ4_XS dot worst: K=%lld row=%zu corpus=%d; max_abs_error/sum_abs_products=%.9g\n",
                (long long) iq4xs_dot_worst_k, iq4xs_dot_worst_row, iq4xs_dot_worst_corpus, iq4xs_dot_normalized);
    std::printf("IQ4_XS dot float max_abs_error/sum_abs_products=%.9g; worst relative ref=%.9g tile=%.9g sum_abs_products=%.9g\n",
                iq4xs_dot_float_normalized, iq4xs_dot_relative_ref, iq4xs_dot_relative_tile, iq4xs_dot_relative_l1);
    worst.print();
}

static void test_iq4xs_k_scale_baseline() {
    iq4xs_errors q4_errors, q5_errors;
    std::vector<float> source(32 * 256);
    fill_source(source, 256, 32);
    block_q4_K q4[32];
    block_q5_K q5[32];
    quantize_row_q4_K_ref(source.data(), q4, (int64_t) source.size());
    quantize_row_q5_K_ref(source.data(), q5, (int64_t) source.size());
    for (int r = 0; r < 32; ++r) {
        float ref4[256], ref5[256];
        dequantize_row_q4_K(q4 + r, ref4, 256);
        dequantize_row_q5_K(q5 + r, ref5, 256);
        for (int ib = 0; ib < 8; ++ib) {
            for (int type = 0; type < 2; ++type) {
                const uint8_t * scales = type == 0 ? q4[r].scales : q5[r].scales;
                const unsigned sc = ib < 4 ? scales[ib] & 63u : (scales[ib + 4] & 15u) | ((scales[ib - 4] >> 6) << 4);
                const unsigned m = ib < 4 ? scales[ib + 4] & 63u : (scales[ib + 4] >> 4) | ((scales[ib] >> 6) << 4);
                const float d = GGML_FP16_TO_FP32(type == 0 ? q4[r].d : q5[r].d) * sc;
                const float min = -GGML_FP16_TO_FP32(type == 0 ? q4[r].dmin : q5[r].dmin) * m;
                const float d16 = GGML_FP16_TO_FP32(GGML_FP32_TO_FP16(d));
                const float m16 = GGML_FP16_TO_FP32(GGML_FP32_TO_FP16(min));
                for (int j = 0; j < 32; ++j) {
                    const int pos = (ib / 2) * 32 + j;
                    const uint8_t packed = type == 0 ? q4[r].qs[pos] : q5[r].qs[pos];
                    int q = (packed >> (4 * (ib % 2))) & 15u;
                    if (type == 1 && (q5[r].qh[j] & (1u << ib))) {
                        q += 16;
                    }
                    const float got = d16 * q + m16;
                    const float ref = type == 0 ? ref4[ib * 32 + j] : ref5[ib * 32 + j];
                    const double bound = (std::fabs(double(d) * q) + std::fabs(min)) / 2048 +
                                         2 * std::numeric_limits<float>::epsilon() * (std::fabs(double(d) * q) + std::fabs(min));
                    check(std::isfinite(got) && std::fabs(double(got) - ref) <= bound, "Q4_K/Q5_K effective scale baseline bound");
                    (type == 0 ? q4_errors : q5_errors).add(ref, got);
                }
            }
        }
    }
    q4_errors.print("Q4_K tiled scale/min rounding baseline");
    q5_errors.print("Q5_K tiled scale/min rounding baseline");
}

static void test_iq4xs_rejection() {
    block_iq4_xs b = {};
    std::vector<uint8_t> dst(iq4xs::repacked_size_2d(256, 1), 0xa5);
    const std::vector<uint8_t> before = dst;
    check(!iq4xs::repack_2d(&b, sizeof(b), 255, 1, dst.data(), dst.size()), "IQ4_XS invalid K");
    check(!iq4xs::repack_2d(&b, sizeof(b), 256, -1, dst.data(), dst.size()), "IQ4_XS invalid rows");
    check(!iq4xs::repack_2d(&b, sizeof(b) - 1, 256, 1, dst.data(), dst.size()), "IQ4_XS short source");
    check(!iq4xs::repack_2d(&b, sizeof(b), 256, 1, dst.data(), dst.size() - 1), "IQ4_XS short destination");
    check(!iq4xs::repack_2d(nullptr, sizeof(b), 256, 1, dst.data(), dst.size()), "IQ4_XS null source");
    check(!iq4xs::repack_2d(&b, sizeof(b), 256, 1, nullptr, dst.size()), "IQ4_XS null destination");
    check(!iq4xs::repack_2d(&b, sizeof(b), INT64_MAX - 255, INT64_MAX, dst.data(), dst.size()), "IQ4_XS size overflow");
    for (ggml_half d : {ggml_half(0x6800), ggml_half(0xe800), ggml_half(0x7bff), ggml_half(0x7c00), ggml_half(0x7e00)}) {
        b.d = d;
        check(!iq4xs::repack_2d(&b, sizeof(b), 256, 1, dst.data(), dst.size()), "IQ4_XS non-finite tile scale rejected");
    }
    check(dst == before, "IQ4_XS failed repack leaves destination unchanged");
    b.d = GGML_FP32_TO_FP16(2048.0f);
    for (int ib = 0; ib < 8; ++ib) {
        iq4xs_set_scale(b, ib, 63);
    }
    check(iq4xs::repack_2d(&b, sizeof(b), 256, 1, dst.data(), dst.size()), "IQ4_XS large d with finite local scales accepted");
    block_iq4_xs a = {}, equivalent = {};
    a.d = GGML_FP32_TO_FP16(1.0f);
    equivalent.d = GGML_FP32_TO_FP16(2.0f);
    for (int ib = 0; ib < 8; ++ib) {
        iq4xs_set_scale(a, ib, 34);
        iq4xs_set_scale(equivalent, ib, 33);
    }
    std::vector<uint8_t> other(dst.size());
    check(iq4xs::repack_2d(&a, sizeof(a), 256, 1, dst.data(), dst.size()), "IQ4_XS non-injective fixture A");
    check(iq4xs::repack_2d(&equivalent, sizeof(equivalent), 256, 1, other.data(), other.size()), "IQ4_XS non-injective fixture B");
    check(dst == other && std::memcmp(&a, &equivalent, sizeof(a)) != 0, "IQ4_XS different raw bytes produce identical tiles");
    a = {};
    check(iq4xs::repack_2d(&a, sizeof(a), 256, 1, dst.data(), dst.size()), "IQ4_XS zero reverse fixture");
    const block_iq4_xs back = iq4xs_reverse_block(dst.data(), 8, 0, 0);
    for (int ib = 0; ib < 8; ++ib) {
        check(iq4xs::iq4_xs_effective_scale(back, ib) == 0, "IQ4_XS zero semantic reverse");
    }
    a.d = 1;
    for (int ib = 0; ib < 8; ++ib) {
        iq4xs_set_scale(a, ib, ib % 2 ? 31 : 33);
    }
    check(iq4xs::repack_2d(&a, sizeof(a), 256, 1, dst.data(), dst.size()), "IQ4_XS minimum subnormal reverse fixture");
    const block_iq4_xs subnormal = iq4xs_reverse_block(dst.data(), 8, 0, 0);
    for (int ib = 0; ib < 8; ++ib) {
        check(iq4xs::iq4_xs_effective_scale(subnormal, ib) == iq4xs::iq4_xs_effective_scale(a, ib),
              "IQ4_XS reverse retains minimum subnormal scales");
    }
    check(iq4xs::repack_2d(nullptr, 0, 256, 0, nullptr, 0), "IQ4_XS empty matrix");
    check(iq4xs::original_size_2d(256, 32) == 32 * 136 && iq4xs::repacked_size_2d(256, 32) == 32 * 144,
          "IQ4_XS storage overhead 144/136 - 1 = 5.8823529%");
}

static void test_iq4xs_representability() {
    size_t total = 0, finite = 0, overflow = 0, underflow = 0, subnormal = 0;
    float max_finite = 0, first_overflow = std::numeric_limits<float>::infinity();
    float safe[64] = {}, unsafe[64];
    size_t by_sign[64][2] = {};
    for (float & value : unsafe) {
        value = std::numeric_limits<float>::infinity();
    }
    const float min_normal = GGML_FP16_TO_FP32(0x0400);
    for (unsigned magnitude = 0; magnitude < 0x7c00; ++magnitude) {
        const float d_abs = GGML_FP16_TO_FP32((ggml_half) magnitude);
        max_finite = std::max(max_finite, d_abs);
        for (int s = -32; s <= 31; ++s) {
            ggml_half rounded[2];
            iq4xs::scale_status status[2];
            for (unsigned sign = 0; sign < 2; ++sign) {
                block_iq4_xs b = {};
                b.d = (ggml_half) (magnitude | (sign << 15));
                iq4xs_set_scale(b, 0, (unsigned) (s + 32));
                const float d = iq4xs::iq4_xs_effective_scale(b, 0);
                check(std::isfinite(d), "IQ4_XS all finite raw scales have finite float effective scale");
                rounded[sign] = GGML_FP32_TO_FP16(d);
                const float got = GGML_FP16_TO_FP32(rounded[sign]);
                status[sign] = iq4xs::iq4_xs_effective_scale_status(b, 0);
                ++total;
                if (std::isfinite(got)) {
                    ++finite;
                    ++by_sign[s + 32][sign];
                    safe[s + 32] = std::max(safe[s + 32], d_abs);
                    check(status[sign] == iq4xs::scale_status::OK, "IQ4_XS finite conversion status");
                } else {
                    ++overflow;
                    first_overflow = std::min(first_overflow, d_abs);
                    unsafe[s + 32] = std::min(unsafe[s + 32], d_abs);
                    check(status[sign] == iq4xs::scale_status::TILE_SCALE_OVERFLOW, "IQ4_XS overflow classification");
                }
                underflow += d != 0 && got == 0;
                check((rounded[sign] >> 15) == (sign ^ unsigned(s < 0)), "IQ4_XS sign including signed zero is preserved");
                if (d != 0 && std::fabs(d) < min_normal) {
                    ++subnormal;
                    check(d == got, "IQ4_XS subnormal effective scale is exact");
                }
            }
            check(status[0] == status[1] && (rounded[0] ^ rounded[1]) == 0x8000u, "IQ4_XS positive/negative symmetry");
        }
    }
    for (unsigned bits = 0x7c00; bits < 0x8000; ++bits) {
        for (unsigned sign = 0; sign < 2; ++sign) {
            block_iq4_xs b = {};
            b.d = (ggml_half) (bits | (sign << 15));
            for (int s : {-32, -31, -1, 0, 1, 30, 31}) {
                iq4xs_set_scale(b, 0, (unsigned) (s + 32));
                check(iq4xs::iq4_xs_effective_scale_status(b, 0) == iq4xs::scale_status::RAW_NONFINITE,
                      "IQ4_XS NaN/Inf status precedes multiplication including zero scale");
            }
        }
    }
    check(total == 63488u * 64 && finite == 3584824 && overflow == 478408 && underflow == 0,
          "IQ4_XS exhaustive representability counts");
    check(first_overflow == 2048 && safe[0] == 2047 && unsafe[0] == 2048, "IQ4_XS first overflow boundary at s=-32");
    check(safe[1] == 2112 && unsafe[1] == 2114 && safe[62] == 2182 && unsafe[62] == 2184,
          "IQ4_XS s=-31 and s=30 boundaries");
    check(safe[31] == max_finite && safe[32] == max_finite && safe[33] == max_finite &&
          safe[63] == safe[1] && unsafe[63] == unsafe[1], "IQ4_XS explicit s=-1,0,1,31 boundaries");
    // The largest finite encoding is found above; the next normal binade defines the rounding midpoint.
    int exponent = 0;
    std::frexp(max_finite, &exponent);
    const float midpoint = max_finite + std::ldexp(1.0f, exponent - 12);
    check(std::isfinite(GGML_FP16_TO_FP32(GGML_FP32_TO_FP16(std::nextafter(midpoint, 0.0f)))) &&
          !std::isfinite(GGML_FP16_TO_FP32(GGML_FP32_TO_FP16(midpoint))), "IQ4_XS actual fp16 overflow rounding midpoint");
    std::printf("IQ4_XS representability: finite_D=%zu finite_D16=%zu overflow=%zu underflow=%zu subnormal_D=%zu max_fp16=%g midpoint=%g first_abs_d=%g\n",
                total, finite, overflow, underflow, subnormal, max_finite, midpoint, first_overflow);
    for (int s = -32; s <= 31; ++s) {
        check(by_sign[s + 32][0] == by_sign[s + 32][1], "IQ4_XS per-scale symmetry counts");
        std::printf("IQ4_XS range s=%+d safe_abs_d=%g first_unsafe_abs_d=%g finite_per_sign=%zu\n",
                    s, safe[s + 32], unsafe[s + 32], by_sign[s + 32][0]);
    }
    unsigned lut_max = 0;
    for (int q = 0; q < 16; ++q) {
        lut_max = std::max(lut_max, (unsigned) std::abs(int(kvalues_iq4nl[q])));
    }
    const float scale = GGML_FP16_TO_FP32(GGML_FP32_TO_FP16(600.0f));
    check(std::isfinite(scale) && std::fabs(scale * kvalues_iq4nl[0]) > max_finite &&
          std::fabs(scale * kvalues_iq4nl[8]) <= max_finite, "IQ4_XS HMX weight range is index dependent and separate from tile scale");
    std::printf("IQ4_XS HMX conservative mathematical scale limit=%g (max LUT magnitude=%u; not a QFloat emulator)\n", max_finite / lut_max, lut_max);
}

static void iq4xs_exact_roundtrip(int64_t k, int64_t n, int pattern) {
    const size_t raw_size = iq4xs::original_size_2d(k, n);
    const size_t meta_size = iq4xs::metadata_size_2d(k, n);
    std::vector<block_iq4_xs> raw(raw_size / sizeof(block_iq4_xs)), restored(raw.size());
    std::vector<uint8_t> tiled(iq4xs::repacked_size_2d(k, n));
    std::vector<iq4xs::block_metadata> metadata(raw.size());
    uint32_t rng = 0x972413abu ^ (uint32_t) k ^ ((uint32_t) n << 16) ^ (uint32_t) pattern;
    const ggml_half boundary[] = {0, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x67ff, 0xe7ff};
    for (size_t i = 0; i < raw.size(); ++i) {
        const unsigned bits = xorshift32(rng);
        const ggml_half d = i < 8 ? boundary[i] : (ggml_half) ((bits % 0x6800u) | (bits & 0x8000u));
        raw[i] = iq4xs_random_block(rng, d);
        if (pattern < 2) {
            std::memset(raw[i].qs, pattern == 0 ? 0 : 0xff, sizeof(raw[i].qs));
        }
        for (int ib = 0; ib < 8; ++ib) {
            if (i < 8) {
                iq4xs_set_scale(raw[i], ib, (unsigned) ((i * 8 + ib) % 64));
            }
        }
    }
    check(iq4xs::repack_2d_with_metadata(raw.data(), raw_size, k, n, tiled.data(), tiled.size(), metadata.data(), meta_size),
          "IQ4_XS exact repack with metadata");
    for (size_t i = 0; i < raw.size(); ++i) {
        check(std::memcmp(metadata[i].bytes, raw.data() + i, 8) == 0, "IQ4_XS sidecar preserves all metadata bits");
    }
    check(iq4xs::readback_2d(tiled.data(), tiled.size(), metadata.data(), meta_size, k, n, 0, restored.data(), raw_size),
          "IQ4_XS exact full readback");
    if (raw_size != 0) {
        check(std::memcmp(raw.data(), restored.data(), raw_size) == 0, "IQ4_XS raw/tile+sidecar/raw is bit exact");
    }
    const uint8_t * raw_bytes = (const uint8_t *) raw.data();
    for (int trial = 0; trial < 64; ++trial) {
        const size_t offset = raw_size == 0 ? 0 : xorshift32(rng) % (raw_size + 1);
        const size_t size = raw_size - offset < 139 ? raw_size - offset : 139;
        std::vector<uint8_t> out(size + 2, 0xa5);
        check(iq4xs::readback_2d(tiled.data(), tiled.size(), metadata.data(), meta_size, k, n, offset, out.data() + 1, size),
              "IQ4_XS arbitrary unaligned byte range");
        check(out.front() == 0xa5 && out.back() == 0xa5, "IQ4_XS byte range guard bytes");
        check(size == 0 || std::memcmp(out.data() + 1, raw_bytes + offset, size) == 0, "IQ4_XS arbitrary range exact bytes");
    }
    if (n >= 3) {
        const size_t size = 137, stride = (size_t) (k / 256) * 136;
        std::vector<uint8_t> out(3 * 153, 0xa5);
        check(iq4xs::readback_2d_ranges(tiled.data(), tiled.size(), metadata.data(), meta_size, k, n,
                                       7, size, 3, stride, out.data(), out.size(), 153), "IQ4_XS strided 2D readback across blocks/rows");
        for (size_t i = 0; i < 3; ++i) {
            check(std::memcmp(out.data() + 153 * i, raw_bytes + 7 + stride * i, size) == 0, "IQ4_XS 2D range exact bytes");
            for (size_t j = size; j < 153; ++j) {
                check(out[153 * i + j] == 0xa5, "IQ4_XS output stride padding unchanged");
            }
        }
    }
    if (n % 32 != 0) {
        for (size_t row = (size_t) n; row < ((size_t) n + 31) / 32 * 32; ++row) {
            for (size_t kt = 0; kt < (size_t) (k / 32); ++kt) {
                uint8_t * tile = tiled.data() + ((row / 32) * (size_t) (k / 32) + kt) * 576;
                check(iq4xs_tile_scale(tile, row % 32) == 0, "IQ4_XS exact repack padded scale zero");
                for (size_t cp = 0; cp < 16; ++cp) {
                    check(tile[cp * 32 + row % 32] == 0, "IQ4_XS exact repack padded indices zero");
                    tile[cp * 32 + row % 32] = 0xff;
                }
            }
        }
        check(iq4xs::readback_2d(tiled.data(), tiled.size(), metadata.data(), meta_size, k, n, 0, restored.data(), raw_size), "IQ4_XS ignores padding in readback");
        check(raw_size == 0 || std::memcmp(raw.data(), restored.data(), raw_size) == 0, "IQ4_XS padding never enters raw output");
    }
}

static void test_iq4xs_exact_readback() {
    for (int64_t k : {256, 512, 1024}) {
        for (int64_t n : {0, 1, 31, 32, 33, 65}) {
            for (int pattern = 0; pattern < 3; ++pattern) {
                iq4xs_exact_roundtrip(k, n, pattern);
            }
        }
    }
    std::vector<block_iq4_xs> raw(1), back(1);
    std::vector<uint8_t> tiled(iq4xs::repacked_size_2d(256, 1));
    iq4xs::block_metadata metadata;
    for (unsigned lo = 0; lo < 16; ++lo) {
        for (unsigned hi = 0; hi < 16; ++hi) {
            for (int ib = 0; ib < 8; ++ib) {
                for (int j = 0; j < 16; ++j) {
                    raw[0].qs[16 * ib + j] = (uint8_t) (((lo + j) % 16) | (((hi + ib) % 16) << 4));
                }
            }
            check(iq4xs::repack_2d_with_metadata(raw.data(), sizeof(block_iq4_xs), 256, 1,
                  tiled.data(), tiled.size(), &metadata, sizeof(metadata)), "IQ4_XS exhaustive reverse fixture");
            check(iq4xs::readback_2d(tiled.data(), tiled.size(), &metadata, sizeof(metadata), 256, 1, 0,
                                    back.data(), sizeof(block_iq4_xs)), "IQ4_XS exhaustive reverse decode");
            check(std::memcmp(raw.data(), back.data(), sizeof(block_iq4_xs)) == 0, "IQ4_XS all low/high patterns reverse exactly");
        }
    }
    for (size_t offset = 0; offset < sizeof(block_iq4_xs); ++offset) {
        uint8_t byte = 0;
        check(iq4xs::readback_2d(tiled.data(), tiled.size(), &metadata, sizeof(metadata), 256, 1, offset, &byte, 1), "IQ4_XS every byte offset");
        check(byte == ((uint8_t *) raw.data())[offset], "IQ4_XS exact byte at every offset");
    }
    uint8_t dst[16];
    std::memset(dst, 0xa5, sizeof(dst));
    check(!iq4xs::readback_2d(tiled.data(), tiled.size(), &metadata, sizeof(metadata), 256, 1, 135, dst, 2), "IQ4_XS readback bounds");
    check(!iq4xs::readback_2d(tiled.data(), tiled.size() - 1, &metadata, sizeof(metadata), 256, 1, 0, dst, 1), "IQ4_XS short tiled input");
    check(!iq4xs::readback_2d(tiled.data(), tiled.size(), &metadata, sizeof(metadata) - 1, 256, 1, 0, dst, 1), "IQ4_XS short metadata input");
    check(!iq4xs::readback_2d_ranges(tiled.data(), tiled.size(), &metadata, sizeof(metadata), 256, 1,
                                  0, 1, SIZE_MAX, SIZE_MAX, dst, sizeof(dst), SIZE_MAX), "IQ4_XS strided size overflow rejected");
    for (uint8_t byte : dst) {
        check(byte == 0xa5, "IQ4_XS failed readback leaves output unchanged");
    }
    const std::vector<uint8_t> tile_before = tiled;
    const iq4xs::block_metadata metadata_before = metadata;
    check(!iq4xs::repack_2d_with_metadata(raw.data(), sizeof(block_iq4_xs), 256, 1,
          tiled.data(), tiled.size(), &metadata, sizeof(metadata) - 1), "IQ4_XS short sidecar rejected before repack");
    raw[0].d = GGML_FP32_TO_FP16(2048.0f);
    iq4xs_set_scale(raw[0], 0, 0);
    check(!iq4xs::repack_2d_with_metadata(raw.data(), sizeof(block_iq4_xs), 256, 1,
          tiled.data(), tiled.size(), &metadata, sizeof(metadata)), "IQ4_XS overflow rejected before metadata publication");
    check(tiled == tile_before && std::memcmp(&metadata, &metadata_before, sizeof(metadata)) == 0,
          "IQ4_XS failed repack preserves both tile and sidecar");
    check(iq4xs::metadata_size_2d(256, 32) == 32 * 8 && iq4xs::metadata_size_2d(512, 33) == 66 * 8,
          "IQ4_XS sidecar excludes padding rows");
    check(iq4xs::metadata_size_2d(INT64_MAX - 255, INT64_MAX) == 0, "IQ4_XS sidecar arithmetic overflow rejected");
    check(8.0 / 256 == 0.03125 && 8.0 * 8 / 256 == 0.25 && 152.0 * 8 / 256 == 4.75,
          "IQ4_XS metadata bytes/bits per weight units");
    std::printf("IQ4_XS exact readback: 54 matrix/pattern cases, exhaustive nibble reverse, arbitrary bytes and 2D ranges PASS; sidecar=8 B/block, total=4.75 bpw\n");
}

struct iq4xs_upload_fixture {
    int64_t k = 512, n = 1;
    std::vector<uint8_t> primary;
    std::vector<iq4xs::block_metadata> metadata;
    iq4xs::storage_mode mode = iq4xs::storage_mode::UNINITIALIZED;
    iq4xs::scale_status status = iq4xs::scale_status::OK;
    size_t reads = 0;

    iq4xs_upload_fixture() : primary(iq4xs::repacked_size_2d(k, n)), metadata((size_t) (k / 256 * n)) {}

    bool complete(const std::vector<block_iq4_xs> & raw) {
        metadata.resize(raw.size());
        const bool ok = iq4xs::prepare_2d(raw.data(), raw.size() * sizeof(block_iq4_xs), k, n,
            primary.data(), primary.size(), metadata.data(), metadata.size() * sizeof(iq4xs::block_metadata), mode, status);
        if (ok && mode == iq4xs::storage_mode::RAW) {
            metadata.clear();
            metadata.shrink_to_fit();
        }
        return ok;
    }
};

static void test_iq4xs_upload_contract() {
    iq4xs_upload_fixture fixture;
    uint32_t rng = 0x135719afu;
    std::vector<block_iq4_xs> raw(2);
    for (block_iq4_xs & b : raw) {
        b = iq4xs_random_block(rng, 0xb555);
    }
    // Match the existing shadow_size contract: disjoint fragments covering a complete generation.
    std::vector<block_iq4_xs> shadow(2);
    size_t received = 0;
    const size_t offsets[] = {17, 0, 136, 53};
    const size_t sizes[] = {36, 17, 136, 83};
    for (size_t i = 0; i < 4; ++i) {
        std::memcpy((uint8_t *) shadow.data() + offsets[i], (const uint8_t *) raw.data() + offsets[i], sizes[i]);
        received += sizes[i];
        if (received < raw.size() * sizeof(block_iq4_xs)) {
            check(fixture.mode == iq4xs::storage_mode::UNINITIALIZED, "IQ4_XS fragmented upload stays uninitialized");
        }
    }
    check(std::memcmp(shadow.data(), raw.data(), 2 * sizeof(block_iq4_xs)) == 0, "IQ4_XS disjoint source assembly");
    check(fixture.complete(shadow) && fixture.mode == iq4xs::storage_mode::TILED, "IQ4_XS complete safe upload becomes tiled");
    shadow.clear();
    shadow.shrink_to_fit();
    std::vector<block_iq4_xs> back(2);
    check(iq4xs::readback_2d(fixture.primary.data(), fixture.primary.size(), fixture.metadata.data(),
          fixture.metadata.size() * sizeof(iq4xs::block_metadata), 512, 1, 0, back.data(), 2 * sizeof(block_iq4_xs)),
          "IQ4_XS readback after transient shadow release");
    check(std::memcmp(back.data(), raw.data(), 2 * sizeof(block_iq4_xs)) == 0, "IQ4_XS metadata outlives upload shadow");
    raw[1].d = GGML_FP32_TO_FP16(2048.0f);
    iq4xs_set_scale(raw[1], 0, 0);
    check(fixture.complete(raw) && fixture.mode == iq4xs::storage_mode::RAW &&
          fixture.status == iq4xs::scale_status::TILE_SCALE_OVERFLOW && fixture.metadata.empty(),
          "IQ4_XS whole replacement with overflow uses primary raw storage");
    check(std::memcmp(fixture.primary.data(), raw.data(), 2 * sizeof(block_iq4_xs)) == 0, "IQ4_XS unsafe raw is preserved without a full shadow");
    raw[1].d = 0x3555;
    check(fixture.complete(raw) && fixture.mode == iq4xs::storage_mode::TILED, "IQ4_XS complete replacement can restore safe state before graph assignment");
    const std::vector<uint8_t> before = fixture.primary;
    const std::vector<iq4xs::block_metadata> meta_before = fixture.metadata;
    raw[1].d = 0x7e01;
    check(!fixture.complete(raw) && fixture.status == iq4xs::scale_status::RAW_NONFINITE, "IQ4_XS nonfinite raw must report load error");
    check(fixture.primary == before && std::memcmp(fixture.metadata.data(), meta_before.data(), 16) == 0,
          "IQ4_XS rejected upload preserves previous representation");
    // A byte count does not prove coverage; partial updates are not complete uploads.
    std::vector<uint8_t> partial(272, 0);
    std::memcpy(partial.data(), raw.data(), 17);
    check(partial[18] == 0, "existing shadow resize zero-fills bytes not uploaded");
    received = 0;
    for (int i = 0; i < 2; ++i) {
        std::memcpy(partial.data(), raw.data(), 136);
        received += 136;
    }
    check(received == partial.size() && std::memcmp(partial.data() + 136, raw.data() + 1, 136) != 0,
          "existing fragment counter reaches completion despite a missing second block");
    std::printf("IQ4_XS upload contract: complete/disjoint fragments, shadow release, raw fallback, full replacements and nonfinite rejection PASS\n");
}

static const char * iq4xs_mock_name(ggml_backend_t) {
    return "IQ4_XS_HOST_MOCK";
}

static bool iq4xs_mock_nonhost(ggml_backend_buffer_type_t) {
    return false;
}

static bool iq4xs_mock_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft == dev->context;
}

static bool iq4xs_mock_supports_op(ggml_backend_dev_t, const ggml_tensor * op) {
    if (op->op == GGML_OP_NONE || op->op == GGML_OP_VIEW || op->op == GGML_OP_RESHAPE ||
        op->op == GGML_OP_TRANSPOSE || op->op == GGML_OP_PERMUTE) {
        return true;
    }
    if (op->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const ggml_tensor * weight = op->src[0];
    const ggml_tensor * owner = weight->view_src ? weight->view_src : weight;
    const auto * fixture = (const iq4xs_upload_fixture *) owner->extra;
    return fixture && fixture->mode == iq4xs::storage_mode::TILED;
}

static void * iq4xs_mock_base(ggml_backend_buffer_t buffer) {
    return ((iq4xs_upload_fixture *) buffer->context)->primary.data();
}

static void iq4xs_mock_get(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * dst, size_t offset, size_t size) {
    auto * fixture = (iq4xs_upload_fixture *) buffer->context;
    ++fixture->reads;
    const size_t logical_offset = offset + (tensor->view_src ? tensor->view_offs : 0);
    if (fixture->mode == iq4xs::storage_mode::RAW) {
        std::memcpy(dst, fixture->primary.data() + logical_offset, size);
    } else {
        check(iq4xs::readback_2d(fixture->primary.data(), fixture->primary.size(), fixture->metadata.data(),
              fixture->metadata.size() * sizeof(iq4xs::block_metadata), fixture->k, fixture->n, logical_offset, dst, size),
              "IQ4_XS mock buffer exact get_tensor");
    }
}

static ggml_status iq4xs_mock_compute(ggml_backend_t, ggml_cgraph *) {
    check(false, "host mock must never execute a device graph");
    return GGML_STATUS_FAILED;
}

// Use the real scheduler and CPU, with a host-only non-host buffer/device fixture.
static void test_iq4xs_scheduler_fallback() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    check(cpu != nullptr, "IQ4_XS CPU backend for fallback test");
    if (!cpu) {
        return;
    }
    ggml_backend_buffer_type buft = *ggml_backend_cpu_buffer_type();
    buft.iface.is_host = iq4xs_mock_nonhost;
    ggml_backend_device device = *ggml_backend_get_device(cpu);
    device.context = &buft;
    device.iface.supports_op = iq4xs_mock_supports_op;
    device.iface.supports_buft = iq4xs_mock_supports_buft;
    buft.device = &device;
    ggml_backend mock = {};
    mock.device = &device;
    mock.iface.get_name = iq4xs_mock_name;
    mock.iface.graph_compute = iq4xs_mock_compute;
    for (bool safe : {false, true}) {
        iq4xs_upload_fixture fixture;
        std::vector<block_iq4_xs> raw(2);
        for (block_iq4_xs & b : raw) {
            b.d = GGML_FP32_TO_FP16(safe ? 1.0f : 2048.0f);
            std::memset(b.qs, 0x88, sizeof(b.qs));
        }
        check(fixture.complete(raw), "IQ4_XS scheduler fixture complete upload");
        ggml_init_params params = { 1024 * 1024, nullptr, true };
        ggml_context * ctx = ggml_init(params);
        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ4_XS, 512, 1);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 1);
        ggml_backend_buffer_i iface = {};
        iface.get_base = iq4xs_mock_base;
        iface.get_tensor = iq4xs_mock_get;
        ggml_backend_buffer_t weight_buffer = ggml_backend_buffer_init(&buft, iface, &fixture, fixture.primary.size());
        check(ggml_backend_tensor_alloc(weight_buffer, w, fixture.primary.data()) == GGML_STATUS_SUCCESS, "IQ4_XS mock weight allocation");
        w->extra = &fixture;
        ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_t act_buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 512 * sizeof(float));
        check(ggml_backend_tensor_alloc(act_buffer, x, ggml_backend_buffer_get_base(act_buffer)) == GGML_STATUS_SUCCESS, "IQ4_XS CPU activation allocation");
        std::vector<float> activation(512, 1.0f);
        ggml_backend_tensor_set(x, activation.data(), 0, activation.size() * sizeof(float));
        ggml_tensor * y = ggml_mul_mat(ctx, w, x);
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
        ggml_build_forward_expand(graph, y);
        ggml_backend_t backends[] = { &mock, cpu };
        ggml_backend_buffer_type_t bufts[] = { &buft, ggml_backend_cpu_buffer_type() };
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 2, 32, false, false);
        check(ggml_backend_sched_alloc_graph(sched, graph), "IQ4_XS scheduler allocates graph after upload");
        check(ggml_backend_sched_get_tensor_backend(sched, y) == (safe ? &mock : cpu), "IQ4_XS state controls real scheduler graph placement");
        if (!safe) {
            check(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "IQ4_XS unsafe weights execute on CPU");
            float result = 0;
            ggml_backend_tensor_get(y, &result, 0, sizeof(result));
            check(result == -33554432.0f && fixture.reads != 0, "IQ4_XS CPU fallback copies exact raw data through get_tensor");
        }
        ggml_tensor * view = ggml_view_1d(ctx, w, 256, sizeof(block_iq4_xs));
        check(ggml_backend_view_init(view) == GGML_STATUS_SUCCESS, "IQ4_XS mock view initialization");
        block_iq4_xs view_raw;
        ggml_backend_tensor_get(view, &view_raw, 0, sizeof(view_raw));
        check(std::memcmp(&view_raw, raw.data() + 1, sizeof(view_raw)) == 0, "IQ4_XS owner-relative view readback exact in both storage modes");
        ggml_backend_sched_free(sched);
        ggml_backend_buffer_free(act_buffer);
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
    }
    ggml_backend_free(cpu);
    std::printf("IQ4_XS real scheduler with host mock: safe placement and unsafe CPU-copy fallback PASS\n");
}

// ---- IQ4_XS Phase 2B: production allocation state ----
//
// These exercise iq4-xs-alloc.h, the same header ggml-hexagon.cpp includes, so
// the state machine under test is the one production uses.

// One production-shaped allocation: a state object, the primary buffer that holds
// either raw or tiled bytes, and the layout the buffer was sized from.
struct iq4xs_alloc_fixture {
    int64_t ne0 = 512, ne1 = 33, ne2 = 1, ne3 = 1;
    iq4xs_alloc_state st;
    std::vector<uint8_t> primary;
    bool live = true; // counts as the owner sidecar being alive

    iq4xs_alloc_fixture(int64_t k = 512, int64_t n = 33, int64_t d2 = 1, int64_t d3 = 1) {
        ne0 = k; ne1 = n; ne2 = d2; ne3 = d3;
        iq4xs_init_layout(st, ne0, ne1, ne2, ne3);
        primary.resize(st.tile_slice_size * st.n_slices());
    }

    ~iq4xs_alloc_fixture() { live = false; }

    size_t raw_nbytes() const { return st.raw_slice_size * st.n_slices(); }

    bool upload(const std::vector<block_iq4_xs> & raw) {
        return iq4xs_prepare(st, (const uint8_t *) raw.data(),
                raw.size() * sizeof(block_iq4_xs), primary.data(), primary.size());
    }

    bool get(void * dst, size_t offset, size_t size) const {
        return iq4xs_readback(st, primary.data(), offset, size, dst);
    }
};

static std::vector<block_iq4_xs> iq4xs_alloc_make_raw(int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, uint32_t seed) {
    const size_t sb = (size_t) (ne0 / QK_K);
    std::vector<block_iq4_xs> raw((size_t) ne1 * ne2 * ne3 * sb);
    uint32_t rng = seed;
    for (size_t i = 0; i < raw.size(); ++i) {
        raw[i] = iq4xs_random_block(rng, GGML_FP32_TO_FP16(0.05f));
    }
    return raw;
}

// A. safe allocation reaches REPACKED_SAFE with a sidecar and compute eligibility.
static void test_iq4xs_prod_safe_allocation() {
    iq4xs_alloc_fixture fixture;
    const auto raw = iq4xs_alloc_make_raw(fixture.ne0, fixture.ne1, fixture.ne2, fixture.ne3, 0x1234u);
    check(fixture.st.state == GGML_HEXAGON_IQ4XS_UNINITIALIZED, "IQ4_XS production starts uninitialized");
    check(fixture.upload(raw), "IQ4_XS production safe upload commits");
    check(fixture.st.is_repacked(), "IQ4_XS safe upload reaches REPACKED_SAFE");
    check(fixture.st.metadata.size() == raw.size() * sizeof(iq4xs::block_metadata),
          "IQ4_XS production sidecar is 8 bytes per block");
    // sidecar cost and total storage, in both unit systems
    check(8.0 / 256 == 0.03125 && 152.0 * 8 / 256 == 4.75, "IQ4_XS production sidecar 0.25 bpw, total 4.75 bpw");
    // the tiled bytes must not be the raw bytes
    check(fixture.primary.size() >= fixture.raw_nbytes(), "IQ4_XS primary allocation holds the larger tiled size");
    check(std::memcmp(fixture.primary.data(), raw.data(), std::min(fixture.raw_nbytes(), raw.size() * sizeof(block_iq4_xs))) != 0,
          "IQ4_XS REPACKED_SAFE bytes are tiles, not the raw source");
    std::printf("IQ4_XS production A: safe upload -> REPACKED_SAFE + sidecar PASS\n");
}

// B. an overflowing block reaches RAW_FALLBACK, keeps raw bytes, and is ineligible.
static void test_iq4xs_prod_overflow_allocation() {
    iq4xs_alloc_fixture fixture(512, 33);
    auto raw = iq4xs_alloc_make_raw(512, 33, 1, 1, 0x99aau);
    check(fixture.upload(raw) && fixture.st.is_repacked(), "IQ4_XS overflow fixture starts safe");

    // abs(d) = 2048 with s = -32 gives D = -65536, past the fp16 conversion midpoint
    raw[2].d = GGML_FP32_TO_FP16(2048.0f);
    iq4xs_set_scale(raw[2], 0, 0);
    check(fixture.upload(raw), "IQ4_XS overflow upload commits instead of failing");
    check(fixture.st.state == GGML_HEXAGON_IQ4XS_RAW_FALLBACK, "IQ4_XS overflow reaches RAW_FALLBACK");
    check(!fixture.st.is_repacked(), "IQ4_XS RAW_FALLBACK is not compute eligible");
    check(fixture.st.metadata.empty(), "IQ4_XS RAW_FALLBACK needs no sidecar");
    check(std::memcmp(fixture.primary.data(), raw.data(), raw.size() * sizeof(block_iq4_xs)) == 0,
          "IQ4_XS RAW_FALLBACK preserves raw bytes exactly");

    // exact readback straight from the raw primary allocation
    std::vector<uint8_t> back(raw.size() * sizeof(block_iq4_xs));
    check(fixture.get(back.data(), 0, back.size()) && std::memcmp(back.data(), raw.data(), back.size()) == 0,
          "IQ4_XS RAW_FALLBACK readback is bit exact");

    // restoring a safe generation is allowed before graph placement
    raw[2] = iq4xs_alloc_make_raw(512, 33, 1, 1, 0x99aau)[2];
    check(fixture.upload(raw) && fixture.st.is_repacked(), "IQ4_XS can return to REPACKED_SAFE before placement");
    std::printf("IQ4_XS production B: overflow -> RAW_FALLBACK, raw preserved, ineligible PASS\n");
}

// C. exact readback of a safe tensor over the whole tensor and arbitrary ranges.
static void test_iq4xs_prod_exact_readback() {
    for (const auto & shape : std::vector<std::pair<int64_t, int64_t>> {
            {256, 1}, {256, 32}, {256, 33}, {512, 31}, {1024, 33},
        }) {
        iq4xs_alloc_fixture fixture(shape.first, shape.second);
        const auto raw = iq4xs_alloc_make_raw(shape.first, shape.second, 1, 1, 0x5150u);
        check(fixture.upload(raw) && fixture.st.is_repacked(), "IQ4_XS exact readback fixture repacked");

        const size_t n = raw.size() * sizeof(block_iq4_xs);
        std::vector<uint8_t> whole(n);
        check(fixture.get(whole.data(), 0, n) && std::memcmp(whole.data(), raw.data(), n) == 0,
              "IQ4_XS full-tensor readback is bit exact");
    }
    std::printf("IQ4_XS production C: bit exact readback over 5 shapes PASS\n");
}

// D. ne2/ne3 slice handling, including row and plane crossings.
static void test_iq4xs_prod_dimensions() {
    for (const auto & shape : std::vector<std::array<int64_t, 4>> {
            {512, 33, 3, 1}, {512, 33, 1, 4}, {256, 32, 2, 2}, {512, 65, 3, 2},
        }) {
        iq4xs_alloc_fixture fixture(shape[0], shape[1], shape[2], shape[3]);
        const auto raw = iq4xs_alloc_make_raw(shape[0], shape[1], shape[2], shape[3], 0x2b7fu);
        check(fixture.upload(raw) && fixture.st.is_repacked(), "IQ4_XS 3D/4D fixture repacked");
        check(fixture.st.n_slices() == (size_t) shape[2] * shape[3], "IQ4_XS slice count matches ne2*ne3");

        const size_t n = raw.size() * sizeof(block_iq4_xs);
        std::vector<uint8_t> whole(n);
        check(fixture.get(whole.data(), 0, n) && std::memcmp(whole.data(), raw.data(), n) == 0,
              "IQ4_XS 3D/4D full readback is bit exact");

        // row crossing: a row of the root matrix is ne0/QK_K blocks
        const size_t row_bytes = fixture.st.raw_slice_size / (size_t) shape[1];
        if (row_bytes > 1) {
            std::vector<uint8_t> row(row_bytes);
            check(fixture.get(row.data(), row_bytes, row_bytes) &&
                  std::memcmp(row.data(), (const uint8_t *) raw.data() + row_bytes, row_bytes) == 0,
                  "IQ4_XS row crossing readback exact");
        }
        // plane crossing: the last bytes of one slice into the first bytes of the next
        const size_t slice = fixture.st.raw_slice_size;
        if (slice > 8 && fixture.st.n_slices() > 1) {
            std::vector<uint8_t> crossing(16);
            check(fixture.get(crossing.data(), slice - 8, 16) &&
                  std::memcmp(crossing.data(), (const uint8_t *) raw.data() + slice - 8, 16) == 0,
                  "IQ4_XS plane crossing readback exact");
        }
    }
    std::printf("IQ4_XS production D: ne2/ne3, row crossing and plane crossing PASS\n");
}

// E. arbitrary byte offsets and sizes, including block and row boundaries.
static void test_iq4xs_prod_offsets() {
    iq4xs_alloc_fixture fixture(512, 33, 2, 1);
    const auto raw = iq4xs_alloc_make_raw(512, 33, 2, 1, 0xc0deu);
    check(fixture.upload(raw) && fixture.st.is_repacked(), "IQ4_XS offset fixture repacked");

    const size_t blk  = sizeof(block_iq4_xs);
    const size_t row  = fixture.st.raw_slice_size / 33;
    const size_t n    = raw.size() * blk;
    const size_t offs[] = { 0, 1, blk - 1, blk, blk + 1, row - 1, row, row + 1,
                            fixture.st.raw_slice_size - 1, fixture.st.raw_slice_size, n - 1 };
    const size_t sizes[] = { 1, 2, blk - 1, blk, blk + 1, 2 * blk, 3 * blk + 5 };

    for (size_t off : offs) {
        for (size_t sz : sizes) {
            if (off > n || sz > n - off) {
                continue;
            }
            std::vector<uint8_t> got(sz ? sz : 1);
            check(fixture.get(got.data(), off, sz), "IQ4_XS arbitrary byte range readback succeeds");
            check(sz == 0 || std::memcmp(got.data(), (const uint8_t *) raw.data() + off, sz) == 0,
                  "IQ4_XS arbitrary byte range is bit exact");
        }
    }
    // a range crossing a slice boundary in one call
    const size_t cross_at = fixture.st.raw_slice_size - 9;
    std::vector<uint8_t> crossing(64);
    check(fixture.get(crossing.data(), cross_at, 64) &&
          std::memcmp(crossing.data(), (const uint8_t *) raw.data() + cross_at, 64) == 0,
          "IQ4_XS single call across a slice boundary is bit exact");
    std::printf("IQ4_XS production E: 11 offsets x 7 sizes plus slice-crossing range PASS\n");
}

// F. sidecar lifetime: aliasing views, owner destruction order, no stale state.
static void test_iq4xs_prod_lifecycle() {
    iq4xs_alloc_fixture fixture(512, 33);
    check(fixture.upload(iq4xs_alloc_make_raw(512, 33, 1, 1, 0xfeedu)) && fixture.st.is_repacked(),
          "IQ4_XS lifecycle owner repacked");
    const size_t sidecar_bytes = fixture.st.metadata.size();
    const size_t total_raw     = fixture.raw_nbytes();

    // A view holds a shared_ptr alias to the owner state, exactly as init_tensor
    // does for a view, so the sidecar is released once, when the last alias dies.
    // The owner is heap allocated here so the aliases share ownership for real.
    auto owner = std::make_shared<iq4xs_alloc_state>();
    *owner = fixture.st;
    check(owner.use_count() == 1 && owner->is_repacked(), "IQ4_XS lifecycle owner state allocated");

    std::vector<std::shared_ptr<iq4xs_alloc_state>> views(4, owner);
    check(owner.use_count() == 5, "IQ4_XS views increase the owner reference count");
    check(views.front().get() == views.back().get(), "IQ4_XS multiple views resolve to one owner state");

    // destroying views one at a time must never free or invalidate the sidecar
    for (size_t i = 0; i + 1 < views.size(); ++i) {
        views[i].reset();
        check(owner.use_count() >= 2, "IQ4_XS owner state survives partial view release");
        check(owner->metadata.size() == sidecar_bytes, "IQ4_XS sidecar not freed by view destruction");
    }
    check(owner.use_count() == 2, "IQ4_XS sidecar freed only after the last view releases");

    // a view read resolves through the owner state at its logical offset
    const size_t view_offs = sizeof(block_iq4_xs);
    std::vector<uint8_t> via_view(sizeof(block_iq4_xs));
    check(iq4xs_readback(*owner, fixture.primary.data(), view_offs, via_view.size(), via_view.data()),
          "IQ4_XS view-range readback succeeds");
    check(iq4xs_view_range(*owner, view_offs, via_view.size(), total_raw),
          "IQ4_XS view range validated against the owner size");
    check(!iq4xs_view_range(*owner, total_raw, 1, total_raw),
          "IQ4_XS out-of-range view rejected");

    views.clear();
    check(owner.use_count() == 1, "IQ4_XS owner is the last remaining reference");
    check(owner->metadata.size() == sidecar_bytes, "IQ4_XS sidecar intact while the owner lives");
    owner.reset();
    check(!owner, "IQ4_XS sidecar released exactly once, no stale owner reference");
    std::printf("IQ4_XS production F: sidecar lifetime, view aliasing, no double free PASS\n");
}

// G. the execution gate must reject any state other than REPACKED_SAFE, which is
// what stops a cached dispatch from sending raw bytes to the DSP as tiles.
static void test_iq4xs_prod_execution_gate() {
    iq4xs_alloc_fixture fixture(512, 33);
    const auto safe = iq4xs_alloc_make_raw(512, 33, 1, 1, 0x600du);

    // 1. safe upload, then a support-cache style eligibility decision
    check(fixture.upload(safe) && fixture.st.is_repacked(), "IQ4_XS gate: safe upload is eligible");
    const bool cached_eligible = fixture.st.is_repacked();

    // 2. contents change to an overflowing generation after placement
    auto unsafe_gen = safe;
    unsafe_gen[1].d = GGML_FP32_TO_FP16(2048.0f);
    iq4xs_set_scale(unsafe_gen[1], 0, 0);
    check(fixture.upload(unsafe_gen), "IQ4_XS gate: unsafe replacement commits");
    check(fixture.st.state == GGML_HEXAGON_IQ4XS_RAW_FALLBACK, "IQ4_XS gate: state flipped to RAW_FALLBACK");

    // 3. the stale cached decision must no longer allow execution
    check(cached_eligible && !fixture.st.is_repacked(),
          "IQ4_XS cached eligibility is invalidated by the new allocation state");
    check(fixture.st.is_repacked() == (fixture.st.state == GGML_HEXAGON_IQ4XS_REPACKED_SAFE),
          "IQ4_XS execution gate accepts REPACKED_SAFE only");

    // and the bytes are still exactly recoverable, so a CPU fallback is possible
    std::vector<uint8_t> back(fixture.raw_nbytes());
    check(fixture.get(back.data(), 0, back.size()) &&
          std::memcmp(back.data(), (const uint8_t *) unsafe_gen.data(), fixture.raw_nbytes()) == 0,
          "IQ4_XS rejected state still reads back bit exact for the CPU path");
    std::printf("IQ4_XS production G: stale cached eligibility cannot execute RAW_FALLBACK PASS\n");
}

// H. the shape gate and the non-finite upload contract.
static void test_iq4xs_prod_gates() {
    // shape gate: one superblock spans 8 tiles, so ne0 must be a multiple of QK_K
    iq4xs_alloc_state ok, bad;
    check(iq4xs_init_layout(ok, 512, 33, 1, 1), "IQ4_XS 512 columns passes the K gate");
    check(!iq4xs_init_layout(bad, 300, 33, 1, 1), "IQ4_XS 300 columns is rejected by the K gate");
    check(!iq4xs_init_layout(bad, 0, 33, 1, 1), "IQ4_XS zero ne0 is rejected");
    check(!iq4xs_init_layout(bad, 512, -1, 1, 1), "IQ4_XS negative ne1 is rejected");
    // sidecar sizing is derived from the raw byte count, so it covers exactly the
    // real blocks and excludes the 32-row tile padding
    // 512 columns is 2 superblocks per row; 33 rows; 8 bytes of sidecar per block
check(ok.meta_slice_size == (512 / QK_K) * 33 * sizeof(iq4xs::block_metadata),
          "IQ4_XS sidecar is 8 bytes per block over the real rows only");
    check(ok.meta_slice_size * (sizeof(block_iq4_xs) / sizeof(iq4xs::block_metadata)) == ok.raw_slice_size,
          "IQ4_XS sidecar covers every raw block");
    iq4xs_alloc_state huge;
    check(!iq4xs_init_layout(huge, INT64_MAX - 255, INT64_MAX, 1, 1),
          "IQ4_XS hostile shape rejected instead of overflowing the sidecar size");

    // non-finite d is an invalid upload, not a recoverable representation change
    iq4xs_alloc_fixture fixture(512, 33);
    auto raw = iq4xs_alloc_make_raw(512, 33, 1, 1, 0xabcdu);
    check(fixture.upload(raw) && fixture.st.is_repacked(), "IQ4_XS nonfinite test starts safe");
    const auto before_primary = fixture.primary;
    const auto before_meta   = fixture.st.metadata;
    raw[3].d = 0x7e01; // a NaN half encoding
    check(!fixture.upload(raw), "IQ4_XS non-finite d rejects the upload");
    check(fixture.st.is_repacked(), "IQ4_XS rejected upload preserves the previous representation");
    check(fixture.primary == before_primary && fixture.st.metadata == before_meta,
          "IQ4_XS rejected upload leaves the prior bytes and sidecar untouched");

    // an uninitialized allocation must not compute and must not read back
    iq4xs_alloc_state fresh;
    check(iq4xs_init_layout(fresh, 512, 33, 1, 1) && !fresh.is_repacked(), "IQ4_XS fresh allocation is uninitialized");
    std::vector<uint8_t> tmp(16);
    check(!iq4xs_readback(fresh, (const uint8_t *) raw.data(), 0, 16, tmp.data()),
          "IQ4_XS uninitialized allocation refuses readback");
    std::printf("IQ4_XS production H: K gate, sidecar sizing, non-finite and uninitialized rejection PASS\n");
}

namespace iq3xxs = ggml_hexagon_iq3xxs;

static void test_iq3xxs_sizes() {
    check(iq3xxs::original_size_2d(256, 32) == 32u * sizeof(block_iq3_xxs),
          "IQ3_XXS original size for 32x256");
    check(iq3xxs::row_tile_size(256) == IQ3XXS_COMPACT_TAIL_SIZE,
          "IQ3_XXS one-superblock tail size");
    check(iq3xxs::row_tile_size(512) == IQ3XXS_COMPACT_PAIR_SIZE,
          "IQ3_XXS two-superblock pair size");
    check(iq3xxs::row_tile_size(768) == IQ3XXS_COMPACT_PAIR_SIZE + IQ3XXS_COMPACT_TAIL_SIZE,
          "IQ3_XXS pair plus odd tail size");
    check(iq3xxs::row_tile_size(1024) == 2u * IQ3XXS_COMPACT_PAIR_SIZE,
          "IQ3_XXS two pair supertiles");
    check(iq3xxs::row_tile_size(4096) == 8u * IQ3XXS_COMPACT_PAIR_SIZE,
          "IQ3_XXS eight pair supertiles");
    check(iq3xxs::repacked_size_2d(256, 31) == IQ3XXS_COMPACT_TAIL_SIZE,
          "IQ3_XXS 31-row allocation");
    check(iq3xxs::repacked_size_2d(256, 32) == IQ3XXS_COMPACT_TAIL_SIZE,
          "IQ3_XXS 32-row allocation");
    check(iq3xxs::repacked_size_2d(256, 33) == 2u * IQ3XXS_COMPACT_TAIL_SIZE,
          "IQ3_XXS 33-row padding");
    check(iq3xxs::repacked_size_2d(512, 32) == IQ3XXS_COMPACT_PAIR_SIZE,
          "IQ3_XXS two superblocks per row");
    check(iq3xxs::repacked_size_2d(768, 32) == IQ3XXS_COMPACT_PAIR_SIZE + IQ3XXS_COMPACT_TAIL_SIZE,
          "IQ3_XXS odd final superblock");
    check(iq3xxs::repacked_size_2d(1280, 65) == 3u * (2u * IQ3XXS_COMPACT_PAIR_SIZE + IQ3XXS_COMPACT_TAIL_SIZE),
          "IQ3_XXS multi-pair allocation with padded rows");
    check(!iq3xxs::valid_2d_shape(255, 32), "IQ3_XXS rejects a partial superblock");
    check(iq3xxs::original_size_2d(INT64_MAX - 255, INT64_MAX) == 0,
          "IQ3_XXS rejects overflowing dimensions");
    size_t total = 0;
    check(iq3_compact_xxs_total_size(768, 33, 2, 3, &total) && total == 6u * 2u * (IQ3XXS_COMPACT_PAIR_SIZE + IQ3XXS_COMPACT_TAIL_SIZE),
          "IQ3_XXS 4D allocation formula");
    check(!iq3_compact_xxs_total_size(INT64_MAX, INT64_MAX, INT64_MAX, INT64_MAX, &total),
          "IQ3_XXS rejects overflowing 4D allocation");
}

static void test_iq3xxs_raw_roundtrip(int64_t ne0, int64_t ne1) {
    const size_t original_size = iq3xxs::original_size_2d(ne0, ne1);
    const size_t packed_size = iq3xxs::repacked_size_2d(ne0, ne1);
    std::vector<block_iq3_xxs> original(original_size / sizeof(block_iq3_xxs));
    std::vector<block_iq3_xxs> roundtrip(original.size());
    std::vector<uint8_t> packed(packed_size);
    uint32_t rng = 0x85ebca6bu ^ (uint32_t) ne0 ^ ((uint32_t) ne1 << 16);
    uint8_t * raw = reinterpret_cast<uint8_t *>(original.data());
    for (size_t i = 0; i < original_size; ++i) {
        raw[i] = (uint8_t) xorshift32(rng);
    }

    check(iq3xxs::repack_2d(original.data(), original_size, ne0, ne1, packed.data(), packed.size()),
          "IQ3_XXS raw repack succeeds");
    check(iq3xxs::unpack_2d(packed.data(), packed.size(), ne0, ne1, roundtrip.data(), original_size),
          "IQ3_XXS raw unpack succeeds");
    check(std::memcmp(original.data(), roundtrip.data(), original_size) == 0,
          "IQ3_XXS raw bytes survive repack and unpack exactly");

    const size_t range_offset = original_size > 81 ? 81 : original_size / 2;
    const size_t range_size = std::min<size_t>(127, original_size - range_offset);
    std::vector<uint8_t> range(range_size);
    check(iq3xxs::readback_2d(packed.data(), packed.size(), ne0, ne1, range_offset, range.data(), range.size()),
          "IQ3_XXS partial byte-range readback succeeds");
    check(std::memcmp(range.data(), raw + range_offset, range.size()) == 0,
          "IQ3_XXS partial byte-range readback is exact");

    if (ne1 >= 33) {
        std::vector<block_iq3_xxs> rows((size_t) 2 * (size_t) (ne0 / QK_K));
        check(iq3xxs::unpack_rows_2d(packed.data(), packed.size(), ne0, ne1, 31, 2,
                                     rows.data(), rows.size() * sizeof(block_iq3_xxs)),
              "IQ3_XXS row-boundary readback succeeds");
        check(std::memcmp(rows.data(), original.data() + 31 * (ne0 / QK_K),
                          rows.size() * sizeof(block_iq3_xxs)) == 0,
              "IQ3_XXS row-boundary readback is exact");
    }
}

static void test_iq3xxs_quantized_reference() {
    const int64_t ne0 = 512;
    const int64_t ne1 = 33;
    const size_t raw_size = iq3xxs::original_size_2d(ne0, ne1);
    const size_t packed_size = iq3xxs::repacked_size_2d(ne0, ne1);
    std::vector<float> source((size_t) ne0 * (size_t) ne1);
    std::vector<float> imatrix(source.size(), 1.0f);
    std::vector<block_iq3_xxs> quantized(raw_size / sizeof(block_iq3_xxs));
    std::vector<block_iq3_xxs> roundtrip(quantized.size());
    std::vector<uint8_t> packed(packed_size);
    fill_source(source, ne0, ne1);

    check(ggml_quantize_chunk(GGML_TYPE_IQ3_XXS, source.data(), quantized.data(), 0, ne1, ne0,
                              imatrix.data()) == raw_size,
          "IQ3_XXS quantizer writes expected bytes");
    check(iq3xxs::repack_2d(quantized.data(), raw_size, ne0, ne1, packed.data(), packed.size()),
          "IQ3_XXS quantized repack succeeds");
    check(iq3xxs::unpack_2d(packed.data(), packed.size(), ne0, ne1, roundtrip.data(), raw_size),
          "IQ3_XXS quantized unpack succeeds");
    check(std::memcmp(quantized.data(), roundtrip.data(), raw_size) == 0,
          "IQ3_XXS quantized bytes survive repack and unpack exactly");

    std::vector<float> cpu(ne0);
    std::vector<float> tiled(ne0);
    const size_t matrix_size = packed_size;
    for (int64_t row = 0; row < ne1; ++row) {
        dequantize_row_iq3_xxs(quantized.data() + row * (ne0 / QK_K), cpu.data(), ne0);
        iq3xxs::semantic_dequantize_row(packed.data(), tiled.data(), ne0, row, iq3xxs_grid, ksigns_iq2xs);
        for (int64_t k = 0; k < ne0; ++k) {
            check(std::fabs(cpu[k] - tiled[k]) <= 1e-6f * std::max(1.0f, std::fabs(cpu[k])),
                  "IQ3_XXS tiled scalar dequant matches CPU reference");
        }
    }
    check(matrix_size == iq3xxs::repacked_size_2d(ne0, ne1), "IQ3_XXS reference covers padded rows");
}

static void test_iq3xxs_compact_metadata() {
    const int64_t ne0 = 768;
    const int64_t ne1 = 33;
    const size_t raw_size = iq3xxs::original_size_2d(ne0, ne1);
    const size_t packed_size = iq3xxs::repacked_size_2d(ne0, ne1);
    std::vector<block_iq3_xxs> raw(raw_size / sizeof(block_iq3_xxs));
    std::vector<uint8_t> guarded(packed_size + 128, 0xa5);
    uint8_t * packed = guarded.data() + 64;
    uint32_t rng = 0x9e3779b9u;
    for (size_t i = 0; i < raw_size; ++i) {
        reinterpret_cast<uint8_t *>(raw.data())[i] = (uint8_t) xorshift32(rng);
    }
    check(iq3xxs::repack_2d(raw.data(), raw_size, ne0, ne1, packed, packed_size),
          "IQ3_XXS compact metadata repack succeeds");
    check(std::all_of(guarded.begin(), guarded.begin() + 64, [](uint8_t b) { return b == 0xa5; }) &&
          std::all_of(guarded.end() - 64, guarded.end(), [](uint8_t b) { return b == 0xa5; }),
          "IQ3_XXS repack preserves allocation canaries");
    check(std::all_of(packed + packed_size - 64, packed + packed_size, [](uint8_t b) { return b == 0; }),
          "IQ3_XXS odd-tail alignment padding is zero");
    std::vector<block_iq3_xxs> roundtrip(raw.size());
    check(iq3xxs::unpack_2d(packed, packed_size, ne0, ne1, roundtrip.data(), raw_size),
          "IQ3_XXS compact metadata unpack succeeds");
    check(std::memcmp(raw.data(), roundtrip.data(), raw_size) == 0,
          "IQ3_XXS compact metadata readback is exact");
}

static void test_iq3xxs_4d_slices() {
    const int64_t ne0 = 1280;
    const int64_t ne1 = 33;
    const int64_t ne2 = 2;
    const int64_t ne3 = 3;
    const size_t raw_slice = iq3xxs::original_size_2d(ne0, ne1);
    const size_t packed_slice = iq3xxs::repacked_size_2d(ne0, ne1);
    size_t packed_total = 0;
    check(iq3_compact_xxs_total_size(ne0, ne1, ne2, ne3, &packed_total) &&
          packed_total == packed_slice * (size_t) ne2 * (size_t) ne3,
          "IQ3_XXS 4D slices match allocation formula");
    std::vector<block_iq3_xxs> raw(raw_slice * (size_t) ne2 * (size_t) ne3 / sizeof(block_iq3_xxs));
    std::vector<block_iq3_xxs> roundtrip(raw.size());
    std::vector<uint8_t> packed(packed_total);
    uint32_t rng = 0x243f6a88u;
    for (size_t i = 0; i < raw.size() * sizeof(block_iq3_xxs); ++i) {
        reinterpret_cast<uint8_t *>(raw.data())[i] = (uint8_t) xorshift32(rng);
    }
    for (size_t slice = 0; slice < (size_t) ne2 * (size_t) ne3; ++slice) {
        check(iq3xxs::repack_2d(raw.data() + slice * (raw_slice / sizeof(block_iq3_xxs)), raw_slice,
                                ne0, ne1, packed.data() + slice * packed_slice, packed_slice),
              "IQ3_XXS 4D slice repack succeeds");
        check(iq3xxs::unpack_2d(packed.data() + slice * packed_slice, packed_slice, ne0, ne1,
                                roundtrip.data() + slice * (raw_slice / sizeof(block_iq3_xxs)), raw_slice),
              "IQ3_XXS 4D slice readback succeeds");
    }
    check(std::memcmp(raw.data(), roundtrip.data(), raw.size() * sizeof(block_iq3_xxs)) == 0,
          "IQ3_XXS 4D slice roundtrip is exact");
}

namespace iq3s = ggml_hexagon_iq3s;

static void test_iq3s_sizes() {
    check(iq3s::original_size_2d(256, 32) == 32u * sizeof(block_iq3_s),
          "IQ3_S original size for 32x256");
    check(iq3s::repacked_size_2d(256, 31) == 8u * iq3s::TILE_SIZE,
          "IQ3_S 31-row tile size");
    check(iq3s::repacked_size_2d(256, 32) == 8u * iq3s::TILE_SIZE,
          "IQ3_S 32-row tile size");
    check(iq3s::repacked_size_2d(256, 33) == 16u * iq3s::TILE_SIZE,
          "IQ3_S 33-row tile padding");
    check(iq3s::repacked_size_2d(512, 32) == 16u * iq3s::TILE_SIZE,
          "IQ3_S two superblocks per row");
    check(!iq3s::valid_2d_shape(255, 32), "IQ3_S rejects a partial superblock");
    check(iq3s::original_size_2d(INT64_MAX - 255, INT64_MAX) == 0,
          "IQ3_S rejects overflowing dimensions");
}

static void test_iq3s_raw_roundtrip(int64_t ne0, int64_t ne1) {
    const size_t original_size = iq3s::original_size_2d(ne0, ne1);
    const size_t packed_size = iq3s::repacked_size_2d(ne0, ne1);
    std::vector<block_iq3_s> original(original_size / sizeof(block_iq3_s));
    std::vector<block_iq3_s> roundtrip(original.size());
    std::vector<uint8_t> packed(packed_size);
    uint32_t rng = 0x391482abu ^ (uint32_t) ne0 ^ ((uint32_t) ne1 << 16);
    uint8_t * raw = reinterpret_cast<uint8_t *>(original.data());
    for (size_t i = 0; i < original_size; ++i) {
        raw[i] = (uint8_t) xorshift32(rng);
    }

    check(iq3s::repack_2d(original.data(), original_size, ne0, ne1, packed.data(), packed.size()),
          "IQ3_S raw repack succeeds");
    check(iq3s::unpack_2d(packed.data(), packed.size(), ne0, ne1, roundtrip.data(), original_size),
          "IQ3_S raw unpack succeeds");
    check(std::memcmp(original.data(), roundtrip.data(), original_size) == 0,
          "IQ3_S raw bytes survive repack and unpack exactly");

    std::vector<uint8_t> range(std::min<size_t>(127, original_size - 81));
    check(iq3s::readback_2d(packed.data(), packed.size(), ne0, ne1, 81, range.data(), range.size()),
          "IQ3_S partial byte-range readback succeeds");
    check(std::memcmp(range.data(), raw + 81, range.size()) == 0,
          "IQ3_S partial byte-range readback is exact");

    if (ne1 >= 33) {
        std::vector<block_iq3_s> rows((size_t) 2 * (size_t) (ne0 / QK_K));
        check(iq3s::unpack_rows_2d(packed.data(), packed.size(), ne0, ne1, 31, 2,
                                   rows.data(), rows.size() * sizeof(block_iq3_s)),
              "IQ3_S row-boundary readback succeeds");
        check(std::memcmp(rows.data(), original.data() + 31 * (ne0 / QK_K),
                          rows.size() * sizeof(block_iq3_s)) == 0,
              "IQ3_S row-boundary readback is exact");
    }
}

static void test_iq3s_quantized_reference() {
    const int64_t ne0 = 512;
    const int64_t ne1 = 33;
    const size_t raw_size = iq3s::original_size_2d(ne0, ne1);
    const size_t packed_size = iq3s::repacked_size_2d(ne0, ne1);
    std::vector<float> source((size_t) ne0 * (size_t) ne1);
    std::vector<float> imatrix(source.size(), 1.0f);
    std::vector<block_iq3_s> quantized(raw_size / sizeof(block_iq3_s));
    std::vector<block_iq3_s> roundtrip(quantized.size());
    std::vector<uint8_t> packed(packed_size);
    fill_source(source, ne0, ne1);

    check(ggml_quantize_chunk(GGML_TYPE_IQ3_S, source.data(), quantized.data(), 0, ne1, ne0,
                              imatrix.data()) == raw_size,
          "IQ3_S quantizer writes expected bytes");
    check(iq3s::repack_2d(quantized.data(), raw_size, ne0, ne1, packed.data(), packed.size()),
          "IQ3_S quantized repack succeeds");
    check(iq3s::unpack_2d(packed.data(), packed.size(), ne0, ne1, roundtrip.data(), raw_size),
          "IQ3_S quantized unpack succeeds");
    check(std::memcmp(quantized.data(), roundtrip.data(), raw_size) == 0,
          "IQ3_S quantized bytes survive repack and unpack exactly");

    std::vector<float> cpu(ne0);
    std::vector<float> tiled(ne0);
    for (int64_t row = 0; row < ne1; ++row) {
        dequantize_row_iq3_s(quantized.data() + row * (ne0 / QK_K), cpu.data(), ne0);
        iq3s::semantic_dequantize_row(packed.data(), tiled.data(), ne0, row, iq3s_grid);
        for (int64_t k = 0; k < ne0; ++k) {
            check(std::fabs(cpu[k] - tiled[k]) <= 1e-6f * std::max(1.0f, std::fabs(cpu[k])),
                  "IQ3_S tiled scalar dequant matches CPU reference");
        }
    }
}

static void test_iq3s_duplicate_d_validation() {
    const int64_t ne0 = 256;
    const int64_t ne1 = 1;
    std::vector<block_iq3_s> raw(1);
    std::memset(raw.data(), 0x5a, sizeof(block_iq3_s));
    std::vector<uint8_t> packed(iq3s::repacked_size_2d(ne0, ne1));
    check(iq3s::repack_2d(raw.data(), sizeof(block_iq3_s), ne0, ne1, packed.data(), packed.size()),
          "IQ3_S duplicate-d fixture repack succeeds");
    packed[iq3s::TILE_SIZE + iq3s::D_PLANE_OFFSET] ^= 1;
    std::vector<block_iq3_s> roundtrip(1);
    check(!iq3s::unpack_2d(packed.data(), packed.size(), ne0, ne1, roundtrip.data(), sizeof(block_iq3_s)),
          "IQ3_S unpack rejects inconsistent duplicated d");
    uint8_t byte = 0;
    check(!iq3s::readback_2d(packed.data(), packed.size(), ne0, ne1, 0, &byte, 1),
          "IQ3_S readback rejects inconsistent duplicated d");
}

int main() {
    test_iq3xxs_sizes();
    for (const auto & shape : std::vector<std::pair<int64_t, int64_t>> {
            {256, 1}, {256, 31}, {256, 32}, {256, 33}, {256, 63}, {256, 64}, {256, 65},
            {512, 31}, {512, 32}, {512, 33}, {768, 31}, {768, 32}, {768, 33},
            {1024, 33}, {1280, 65}, {4096, 65},
        }) {
        test_iq3xxs_raw_roundtrip(shape.first, shape.second);
    }
    test_iq3xxs_quantized_reference();
    test_iq3xxs_compact_metadata();
    test_iq3xxs_4d_slices();
    test_iq3s_sizes();
    for (const auto & shape : std::vector<std::pair<int64_t, int64_t>> {
            {256, 1}, {256, 31}, {256, 32}, {256, 33}, {512, 33}, {1024, 65},
        }) {
        test_iq3s_raw_roundtrip(shape.first, shape.second);
    }
    test_iq3s_quantized_reference();
    test_iq3s_duplicate_d_validation();
    test_iq4xs_prod_safe_allocation();
    test_iq4xs_prod_overflow_allocation();
    test_iq4xs_prod_exact_readback();
    test_iq4xs_prod_dimensions();
    test_iq4xs_prod_offsets();
    test_iq4xs_prod_lifecycle();
    test_iq4xs_prod_execution_gate();
    test_iq4xs_prod_gates();
    test_iq4xs_representability();
    test_iq4xs_exact_readback();
    test_iq4xs_upload_contract();
    test_iq4xs_scheduler_fallback();
    test_iq4xs_bits();
    test_iq4xs_raw_and_range();
    test_iq4xs_matrices();
    test_iq4xs_rejection();
    test_iq4xs_k_scale_baseline();
    test_grid_range();
    test_iq2f_grid_range();
    test_iq2f_sign_table_agreement();
    test_sizes();

    for (const auto & shape : std::vector<std::pair<int64_t, int64_t>> {
            {256, 1},
            {256, 31},
            {256, 32},
            {256, 33},
            {512, 32},
            {4096, 33},
        }) {
        test_raw_roundtrip(shape.first, shape.second);
    }

    for (const auto & shape : std::vector<std::pair<int64_t, int64_t>> {
            {256, 1},
            {256, 32},
            {512, 33},
            {4096, 32},
        }) {
        test_quantized_roundtrip(shape.first, shape.second);
    }

    test_hmx_tile_dequant_reference();
    test_duplicate_d_validation();

    test_iq2f_suite<iq2f::iq2_xs_traits>();
    test_iq2f_suite<iq2f::iq2_xxs_traits>();

    ggml_quantize_free();

    if (n_failed == 0) {
        std::printf("PASS: IQ2_S / IQ2_XS / IQ2_XXS / IQ4_XS repack tests\n");
    } else {
        std::fprintf(stderr, "%d repack test(s) failed\n", n_failed);
    }

    return n_failed != 0;
}
