#include "ggml.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-hexagon/iq2-s-repack.h"

#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

int main() {
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

    ggml_quantize_free();

    if (n_failed == 0) {
        std::printf("PASS: IQ2_S phase-1 repack tests\n");
    } else {
        std::fprintf(stderr, "%d IQ2_S repack test(s) failed\n", n_failed);
    }

    return n_failed != 0;
}
