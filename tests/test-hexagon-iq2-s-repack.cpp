#include "ggml.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-hexagon/iq2-s-repack.h"
#include "ggml-hexagon/iq3-xxs-repack.h"

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

static void test_iq3xxs_semantics() {
    using ggml_hexagon_iq3xxs::load_aux;
    const uint8_t little_endian[] = {0x81, 0x80, 0x60, 0xf0};
    const uint32_t packed = load_aux(little_endian);
    check(((packed >>  0) & 127) == 1, "IQ3_XXS bits 0..6 select sign 0");
    check(((packed >>  7) & 127) == 1, "IQ3_XXS bits 7..13 select sign 1");
    check(((packed >> 14) & 127) == 2, "IQ3_XXS bits 14..20 select sign 2");
    check(((packed >> 21) & 127) == 3, "IQ3_XXS bits 21..27 select sign 3");
    check((packed >> 28) == 15, "IQ3_XXS bits 28..31 select scale");

    // The CPU reference reads native words; these byte fixtures require a little-endian host.
    const uint32_t one = 1;
    check(*(const uint8_t *) &one == 1, "IQ3_XXS CPU comparison requires little-endian host");
    int cases = 0;
    for (float d : {0.00006103515625f, 1.0f, -0.75f}) {
        for (uint8_t index : {0, 255}) {
            for (uint32_t sign : {0u, 127u}) {
                for (uint32_t scale : {0u, 15u}) {
                    block_iq3_xxs block = {};
                    block.d = ggml_fp32_to_fp16(d);
                    std::memset(block.qs, index, 64);
                    for (int group = 0; group < 8; ++group) {
                        const uint32_t aux = sign | (sign << 7) | (sign << 14) | (sign << 21) | (scale << 28);
                        for (int b = 0; b < 4; ++b) {
                            block.qs[64 + 4 * group + b] = (uint8_t) (aux >> (8 * b));
                        }
                    }
                    float ref[QK_K], got[QK_K];
                    dequantize_row_iq3_xxs(&block, ref, QK_K);
                    ggml_hexagon_iq3xxs::semantic_dequantize(&block, got, QK_K, iq3xxs_grid, ksigns_iq2xs);
                    check(std::memcmp(ref, got, sizeof(ref)) == 0, "IQ3_XXS endpoint semantic fixture matches CPU exactly");
                    ++cases;
                }
            }
        }
    }
    for (int field = 0; field < 4; ++field) {
        for (uint32_t sign = 0; sign < 128; ++sign) {
            block_iq3_xxs block = {};
            block.d = ggml_fp32_to_fp16(0.5f);
            for (int i = 0; i < 64; ++i) {
                block.qs[i] = (uint8_t) (i * 7);
            }
            const uint32_t aux = (sign << (7 * field)) | (9u << 28);
            for (int group = 0; group < 8; ++group) {
                for (int b = 0; b < 4; ++b) {
                    block.qs[64 + 4 * group + b] = (uint8_t) (aux >> (8 * b));
                }
            }
            float ref[QK_K], got[QK_K];
            dequantize_row_iq3_xxs(&block, ref, QK_K);
            ggml_hexagon_iq3xxs::semantic_dequantize(&block, got, QK_K, iq3xxs_grid, ksigns_iq2xs);
            check(std::memcmp(ref, got, sizeof(ref)) == 0, "IQ3_XXS independent sign fields match CPU exactly");
            ++cases;
        }
    }
    std::printf("IQ3_XXS semantic fixtures: %d\n", cases);
}

static void test_iq3xxs_repack() {
    namespace iq3 = ggml_hexagon_iq3xxs;
    check(iq3::original_size_2d(256, 32) == 32 * 98, "IQ3_XXS raw size");
    check(iq3::repacked_size_2d(256, 32) == 8 * 512, "IQ3_XXS repacked size");
    check(!iq3::valid_2d_shape(255, 32), "IQ3_XXS rejects incomplete superblocks");
    check(!iq3::valid_2d_shape(256, -1), "IQ3_XXS rejects negative rows");
    check(iq3::repack_2d(nullptr, 0, 256, 0, nullptr, 0), "IQ3_XXS empty repack");
    int cases = 0;
    for (int64_t k : {256, 512, 1024, 4096, 8192}) {
        for (int64_t m : {1, 16, 31, 32, 33, 63, 64, 65}) {
            const size_t raw_size = iq3::original_size_2d(k, m);
            const size_t packed_size = iq3::repacked_size_2d(k, m);
            std::vector<block_iq3_xxs> raw(raw_size / sizeof(block_iq3_xxs)), out(raw.size());
            std::vector<uint8_t> packed(packed_size);
            uint32_t rng = 0x125738ab ^ (uint32_t) k ^ ((uint32_t) m << 16);
            for (size_t i = 0; i < raw_size; ++i) {
                ((uint8_t *) raw.data())[i] = (uint8_t) xorshift32(rng);
            }
            check(iq3::repack_2d(raw.data(), raw_size, k, m, packed.data(), packed_size), "IQ3_XXS random raw repack");
            check(iq3::unpack_2d(packed.data(), packed_size, k, m, out.data(), raw_size), "IQ3_XXS random raw unpack");
            check(std::memcmp(raw.data(), out.data(), raw_size) == 0, "IQ3_XXS random raw roundtrip is byte exact");
            for (int64_t r = 0; r < m; ++r) {
                check(iq3::unpack_rows_2d(packed.data(), packed_size, k, m, r, 1, out.data(), raw_size), "IQ3_XXS partial row readback");
                check(std::memcmp(raw.data() + r * (k / QK_K), out.data(), raw_size / (size_t) m) == 0, "IQ3_XXS partial readback is byte exact");
            }
            for (block_iq3_xxs & b : raw) {
                b.d = ggml_fp32_to_fp16(((int) (xorshift32(rng) % 257) - 128) / 8192.0f);
            }
            check(iq3::repack_2d(raw.data(), raw_size, k, m, packed.data(), packed_size), "IQ3_XXS finite fixture repack");
            std::vector<float> ref((size_t) k), got((size_t) k);
            for (int64_t r = 0; r < m; ++r) {
                dequantize_row_iq3_xxs(raw.data() + r * (k / QK_K), ref.data(), k);
                iq3::semantic_dequantize_row(packed.data(), got.data(), k, r, iq3xxs_grid, ksigns_iq2xs);
                check(std::memcmp(ref.data(), got.data(), (size_t) k * sizeof(float)) == 0, "IQ3_XXS repacked semantic decode matches CPU exactly");
            }
            for (int64_t r = m; r < iq3::padded_rows(m); ++r) {
                iq3::semantic_dequantize_row(packed.data(), got.data(), k, r, iq3xxs_grid, ksigns_iq2xs);
                for (float v : got) {
                    check(v == 0.0f, "IQ3_XXS padded rows decode to zero");
                }
            }
            for (int64_t kt = 0; kt < (k / 32) * (iq3::padded_rows(m) / 32); ++kt) {
                for (size_t i = iq3::PADDING_OFFSET; i < iq3::TILE_SIZE; ++i) {
                    check(packed[(size_t) kt * iq3::TILE_SIZE + i] == 0, "IQ3_XXS tile padding is zero");
                }
            }
            check(!iq3::repack_2d(raw.data(), raw_size - 1, k, m, packed.data(), packed_size), "IQ3_XXS rejects short source");
            check(!iq3::unpack_2d(packed.data(), packed_size - 1, k, m, out.data(), raw_size), "IQ3_XXS rejects short repack buffer");
            packed[iq3::TILE_SIZE + iq3::D_PLANE_OFFSET] ^= 1;
            check(!iq3::unpack_2d(packed.data(), packed_size, k, m, out.data(), raw_size), "IQ3_XXS detects corrupted duplicate d");
            ++cases;
        }
    }
    std::printf("IQ3_XXS repack shapes: %d\n", cases);
}

int main() {
    test_iq3xxs_semantics();
    test_iq3xxs_repack();
    test_grid_range();
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
        std::printf("PASS: IQ2_S and IQ3_XXS repack tests\n");
    } else {
        std::fprintf(stderr, "%d IQ2_S repack test(s) failed\n", n_failed);
    }

    return n_failed != 0;
}
