#include "ggml.h"
#include "ggml-quants.h"
#include "ggml-hexagon/iq3-s-repack.h"

#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace ggml_hexagon_iq3s;

static int n_failed = 0;
static int semantic_cases = 0;

static void check(bool cond, const char * msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++n_failed;
    }
}

static uint32_t random32(uint32_t & state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static void check_semantics(const block_iq3_s & b) {
    float ref[QK_K], got[QK_K];
    dequantize_row_iq3_s(&b, ref, QK_K);
    semantic_dequantize(&b, got, QK_K, iq3s_grid);
    check(std::memcmp(ref, got, sizeof(ref)) == 0, "raw semantic reference matches CPU exactly");
    std::vector<uint8_t> packed(repacked_size_2d(QK_K, 1));
    check(repack_2d(&b, sizeof(b), QK_K, 1, packed.data(), packed.size()), "semantic fixture repack");
    semantic_dequantize_row(packed.data(), got, QK_K, 0, iq3s_grid);
    check(std::memcmp(ref, got, sizeof(ref)) == 0, "tile semantic reference matches CPU exactly");
    ++semantic_cases;
}

static void test_semantics() {
    for (float d : {0.00006103515625f, 0.03125f, 1.0f, -0.75f}) {
        for (uint8_t low : {0, 1, 127, 128, 254, 255}) {
            for (uint8_t high : {0x00, 0x01, 0x02, 0x55, 0xaa, 0xff}) {
                for (uint8_t signs : {0x00, 0xff, 0x55, 0xaa, 0x01, 0x80}) {
                    for (uint8_t scale : {0, 1, 7, 15}) {
                        block_iq3_s b = {};
                        b.d = ggml_fp32_to_fp16(d);
                        std::memset(b.qs, low, sizeof(b.qs));
                        std::memset(b.qh, high, sizeof(b.qh));
                        std::memset(b.signs, signs, sizeof(b.signs));
                        std::memset(b.scales, scale | (scale << 4), sizeof(b.scales));
                        check_semantics(b);
                    }
                }
            }
        }
    }
    uint32_t rng = 0x391482ab;
    for (int i = 0; i < 512; ++i) {
        block_iq3_s b;
        for (size_t j = 0; j < sizeof(b); ++j) {
            ((uint8_t *) &b)[j] = (uint8_t) random32(rng);
        }
        b.d = ggml_fp32_to_fp16(((int) (random32(rng) % 257) - 128) / 8192.0f);
        check_semantics(b);
    }
    for (uint32_t entry : iq3s_grid) {
        for (int j = 0; j < 4; ++j) {
            check(((entry >> (8 * j)) & 255) <= 127, "grid magnitudes fit signed bytes for vrmpy");
        }
    }
    std::printf("IQ3_S semantic fixtures: %d\n", semantic_cases);
}

static void test_sizes() {
    check(sizeof(block_iq3_s) == 110, "raw block is 110 bytes");
    check(sizeof(block_iq3_s) * 8.0 / QK_K == 3.4375, "raw bpw is 3.4375");
    check(original_size_2d(256, 32) == 32 * 110, "raw size");
    check(repacked_size_2d(256, 32) == 8 * 512, "repacked size");
    check(TILE_SIZE * 8.0 / (TILE_ROWS * TILE_COLS) == 4.0, "repacked bpw is 4.0");
    check(std::fabs((512.0 - 440.0) / 440.0 * 100.0 - 16.363636363636) < 1e-10, "repack overhead is 16.363636 percent");
    check(!valid_2d_shape(255, 32) && !valid_2d_shape(32, 32), "reject incomplete superblocks");
    check(!valid_2d_shape(256, -1) && !valid_2d_shape(256, INT64_MAX), "reject invalid rows");
    check(original_size_2d(INT64_MAX - 255, INT64_MAX - 31) == 0, "raw size overflow");
    check(repacked_size_2d(INT64_MAX - 255, 32) == 0, "tile size overflow");
    check(repack_2d(nullptr, 0, 256, 0, nullptr, 0), "empty repack");
    check(unpack_rows_2d(nullptr, 0, 256, 0, 0, 0, nullptr, 0), "empty unpack");
}

static void test_repack() {
    int cases = 0;
    for (int64_t k : {256, 512, 1024, 4096, 8192}) {
        for (int64_t m : {1, 16, 31, 32, 33, 63, 64, 65}) {
            const size_t raw_size = original_size_2d(k, m);
            const size_t packed_size = repacked_size_2d(k, m);
            std::vector<block_iq3_s> raw(raw_size / sizeof(block_iq3_s)), out(raw.size());
            std::vector<uint8_t> packed(packed_size);
            uint32_t rng = 0x391482ab ^ (uint32_t) k ^ ((uint32_t) m << 16);
            for (size_t i = 0; i < raw_size; ++i) {
                ((uint8_t *) raw.data())[i] = (uint8_t) random32(rng);
            }
            check(repack_2d(raw.data(), raw_size, k, m, packed.data(), packed_size), "random raw repack");
            std::memset(out.data(), 0xff, raw_size);
            check(unpack_2d(packed.data(), packed_size, k, m, out.data(), raw_size), "random raw unpack");
            check(std::memcmp(raw.data(), out.data(), raw_size) == 0, "raw roundtrip is byte exact");
            for (int64_t first = 0; first < m; ++first) {
                const int64_t count = m - first;
                check(unpack_rows_2d(packed.data(), packed_size, k, m, first, count, out.data(), raw_size), "partial rows readback");
                check(std::memcmp(raw.data() + first * (k / QK_K), out.data(), (size_t) count * raw_size / m) == 0, "partial readback is byte exact");
            }
            for (block_iq3_s & b : raw) {
                b.d = ggml_fp32_to_fp16(((int) (random32(rng) % 257) - 128) / 8192.0f);
            }
            check(repack_2d(raw.data(), raw_size, k, m, packed.data(), packed_size), "finite fixture repack");
            std::vector<float> ref((size_t) k), got((size_t) k);
            for (int64_t r = 0; r < m; ++r) {
                dequantize_row_iq3_s(raw.data() + r * (k / QK_K), ref.data(), k);
                semantic_dequantize_row(packed.data(), got.data(), k, r, iq3s_grid);
                check(std::memcmp(ref.data(), got.data(), (size_t) k * sizeof(float)) == 0, "repacked semantics match CPU exactly");
            }
            for (int64_t r = m; r < padded_rows(m); ++r) {
                semantic_dequantize_row(packed.data(), got.data(), k, r, iq3s_grid);
                for (float v : got) {
                    check(v == 0.0f, "padded rows decode to zero");
                }
                for (int64_t kt = 0; kt < k / 32; ++kt) {
                    uint8_t * tile = packed.data() + ((r / 32) * (k / 32) + kt) * TILE_SIZE;
                    tile[D_PLANE_OFFSET + 2 * (r % 32)] = 0xff;
                    tile[SCALE_PLANE_OFFSET + r % 32] = 0xff;
                }
            }
            check(unpack_2d(packed.data(), packed_size, k, m, out.data(), raw_size), "padding corruption stays outside logical rows");
            check(std::memcmp(raw.data(), out.data(), raw_size) == 0, "padding does not leak into logical rows");
            check(!repack_2d(raw.data(), raw_size - 1, k, m, packed.data(), packed_size), "reject short raw source");
            check(!unpack_2d(packed.data(), packed_size - 1, k, m, out.data(), raw_size), "reject short packed source");
            check(!unpack_rows_2d(packed.data(), packed_size, k, m, m, 1, out.data(), raw_size), "reject padding readback");
            for (uint8_t mask : {0x01, 0xff}) {
                for (int g = 1; g < 8; ++g) {
                    const size_t pos = g * TILE_SIZE + D_PLANE_OFFSET + (g & 1);
                    packed[pos] ^= mask;
                    check(!unpack_2d(packed.data(), packed_size, k, m, out.data(), raw_size), "reject one-bit and one-byte duplicate d corruption");
                    packed[pos] ^= mask;
                }
            }
            ++cases;
        }
    }
    std::printf("IQ3_S repack shapes: %d; duplicate d bit/byte, partial rows and padding: PASS\n", cases);
}

int main() {
    test_sizes();
    test_semantics();
    test_repack();
    std::printf("%s: IQ3_S repack tests (%d failures)\n", n_failed ? "FAIL" : "PASS", n_failed);
    return n_failed != 0;
}
