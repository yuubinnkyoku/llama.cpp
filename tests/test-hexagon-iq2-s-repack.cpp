#include "ggml.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-hexagon/iq2-s-repack.h"
#include "ggml-hexagon/iq2-family-repack.h"

#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

int main() {
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
        std::printf("PASS: IQ2_S / IQ2_XS / IQ2_XXS repack tests\n");
    } else {
        std::fprintf(stderr, "%d repack test(s) failed\n", n_failed);
    }

    return n_failed != 0;
}
