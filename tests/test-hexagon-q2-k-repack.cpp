#include "ggml.h"
#include "ggml-quants.h"
#include "ggml-hexagon/q2-k-repack.h"

#include "ggml-hexagon/htp/q2-k-scalar.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static_assert(sizeof(block_q2_K) == 84, "Q2_K raw block size");

using namespace ggml_hexagon_q2k;

static int n_failed = 0;
static int semantic_cases = 0;
static int scalar_cases = 0;

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

static void check_semantics(const block_q2_K & b) {
    float ref[QK_K], got[QK_K];
    dequantize_row_q2_K(&b, ref, QK_K);
    semantic_dequantize(&b, got, QK_K);
    check(std::memcmp(ref, got, sizeof(ref)) == 0, "raw semantic reference matches CPU exactly");
    std::vector<uint8_t> packed(repacked_size_2d(QK_K, 1));
    check(repack_2d(&b, sizeof(b), QK_K, 1, packed.data(), packed.size()), "semantic fixture repack");
    semantic_dequantize_row(packed.data(), got, QK_K, 0);
    check(std::memcmp(ref, got, sizeof(ref)) == 0, "tile semantic reference matches CPU exactly");
    ++semantic_cases;
}

static void check_scalar(const block_q2_K * raw, const uint8_t * packed, int64_t k, int64_t rows, uint32_t & rng) {
    const size_t ntiles = (size_t) k / 32;
    std::vector<uint8_t> staged(ntiles * ALIGNED_TILE_SIZE, 0xa5);
    std::vector<uint8_t> act0(ntiles * HTP_MM_ACT_TILE_SIZE_Q8_0, 0x5a), act1(act0);
    std::vector<float> ref((size_t) k);
    std::vector<int8_t> aq0((size_t) k), aq1((size_t) k);
    std::vector<float> da0(ntiles), da1(ntiles);
    for (size_t kt = 0; kt < ntiles; ++kt) {
        std::memcpy(staged.data() + kt * ALIGNED_TILE_SIZE, packed + kt * TILE_SIZE, TILE_SIZE);
        for (int a = 0; a < 2; ++a) {
            uint8_t * act = (a ? act1 : act0).data() + kt * HTP_MM_ACT_TILE_SIZE_Q8_0;
            const ggml_half da = ggml_fp32_to_fp16((random32(rng) % 32) / 1024.0f);
            (a ? da1 : da0)[kt] = ggml_fp16_to_fp32(da);
            std::memcpy(act + HTP_MM_ACT_SCALE_PLANE_OFFSET_Q8_0, &da, sizeof(da));
            for (int j = 0; j < 32; ++j) {
                const int8_t aq = (int8_t) ((int) (random32(rng) % 255) - 127);
                (a ? aq1 : aq0)[kt * 32 + j] = aq;
                act[(j / 4) * 128 + (j & 3)] = (uint8_t) aq;
            }
        }
    }
    float got0[32], got1[32], single[32], bias[32];
    for (int r = 0; r < 32; ++r) {
        got0[r] = got1[r] = single[r] = 12345.0f;
        bias[r] = (r - 16) / 16.0f;
    }
    const float * sz = (rng & 1) ? bias : nullptr;
    tiled_vec_dot_q2_K_32x2((uint32_t) k, got0, got1, staged.data(), act0.data(), act1.data(), (uint32_t) rows, sz, sz);
    tiled_vec_dot_q2_K_32x1((uint32_t) k, single, staged.data(), act0.data(), (uint32_t) rows, sz);
    check(std::memcmp(single, got0, sizeof(got0)) == 0, "32x1 and 32x2 agree, including output guards");
    for (int64_t r = 0; r < rows; ++r) {
        dequantize_row_q2_K(raw + r * (k / QK_K), ref.data(), k);
        float sum0 = 0.0f, sum1 = 0.0f;
        for (int64_t j = 0; j < k; ++j) {
            sum0 += ref[j] * (float) aq0[j] * da0[j / 32];
            sum1 += ref[j] * (float) aq1[j] * da1[j / 32];
        }
        sum0 += sz ? sz[r] : 0.0f;
        sum1 += sz ? sz[r] : 0.0f;
        check(std::memcmp(&sum0, got0 + r, sizeof(float)) == 0, "scalar first activation matches CPU dequant dot exactly");
        check(std::memcmp(&sum1, got1 + r, sizeof(float)) == 0, "scalar second activation matches CPU dequant dot exactly");
        ++scalar_cases;
    }
    for (int64_t r = rows; r < 32; ++r) {
        check(got0[r] == 12345.0f && got1[r] == 12345.0f, "scalar writes only valid rows");
    }
}

static void test_semantics() {
    uint32_t rng = 0x291482ab;
    for (float d : {0.0f, 0.00006103515625f, 0.03125f, 1.0f, -0.75f}) {
        for (float dmin : {0.0f, 0.000000059604644775390625f, 0.03125f, 0.5f, -0.25f}) {
            for (uint8_t sc : {0x00, 0x01, 0x0f, 0x10, 0xf0, 0xff, 0x55, 0xaa}) {
                for (uint8_t qs : {0x00, 0xff, 0x55, 0xaa, 0x1b, 0xe4}) {
                    block_q2_K b = {};
                    b.d = ggml_fp32_to_fp16(d);
                    b.dmin = ggml_fp32_to_fp16(dmin);
                    std::memset(b.qs, qs, sizeof(b.qs));
                    for (int g = 0; g < 8; ++g) {
                        b.scales[2 * g] = sc;
                        b.scales[2 * g + 1] = sc ^ 0xff;
                    }
                    check_semantics(b);
                    std::vector<uint8_t> packed(repacked_size_2d(256, 1));
                    check(repack_2d(&b, sizeof(b), 256, 1, packed.data(), packed.size()), "scalar boundary repack");
                    check_scalar(&b, packed.data(), 256, 1, rng);
                }
            }
        }
    }
    for (int i = 0; i < 640; ++i) {
        block_q2_K b;
        for (size_t j = 0; j < sizeof(b); ++j) {
            ((uint8_t *) &b)[j] = (uint8_t) random32(rng);
        }
        b.d = ggml_fp32_to_fp16(((int) (random32(rng) % 257) - 128) / 8192.0f);
        b.dmin = ggml_fp32_to_fp16(((int) (random32(rng) % 257) - 128) / 4096.0f);
        check_semantics(b);
        std::vector<uint8_t> packed(repacked_size_2d(256, 1));
        check(repack_2d(&b, sizeof(b), 256, 1, packed.data(), packed.size()), "scalar random repack");
        check_scalar(&b, packed.data(), 256, 1, rng);
    }
    std::printf("Q2_K semantic fixtures: %d\n", semantic_cases);
}

static void test_sizes() {
    check(sizeof(block_q2_K) == 84, "raw block is 84 bytes");
    check(sizeof(block_q2_K) * 8.0 / QK_K == 2.625, "raw bpw is 2.625");
    check(ALIGNED_TILE_SIZE == 512 && TILE_SIZE == 448, "storage and staging sizes differ");
    check(repacked_size_2d(256, 32) / 32 == 112, "repacked row is 112 bytes");
    check(original_size_2d(256, 32) == 32 * 84, "raw size");
    check(repacked_size_2d(256, 32) == 8 * 448, "repacked size");
    check(TILE_SIZE * 8.0 / (TILE_ROWS * TILE_COLS) == 3.5, "repacked bpw is 3.5");
    check(std::fabs((448.0 - 336.0) / 336.0 * 100.0 - 33.333333333333) < 1e-10, "repack overhead is 33.333333 percent");
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
        for (int64_t m : {1, 2, 15, 31, 32, 33, 63, 64, 65}) {
            const size_t raw_size = original_size_2d(k, m);
            const size_t packed_size = repacked_size_2d(k, m);
            std::vector<block_q2_K> raw(raw_size / sizeof(block_q2_K)), out(raw.size());
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
            for (block_q2_K & b : raw) {
                b.dmin = ggml_fp32_to_fp16(((int) (random32(rng) % 257) - 128) / 4096.0f);
                b.d = ggml_fp32_to_fp16(((int) (random32(rng) % 257) - 128) / 8192.0f);
            }
            check(repack_2d(raw.data(), raw_size, k, m, packed.data(), packed_size), "finite fixture repack");
            std::vector<float> ref((size_t) k), got((size_t) k);
            for (int64_t r = 0; r < m; ++r) {
                dequantize_row_q2_K(raw.data() + r * (k / QK_K), ref.data(), k);
                semantic_dequantize_row(packed.data(), got.data(), k, r);
                check(std::memcmp(ref.data(), got.data(), (size_t) k * sizeof(float)) == 0, "repacked semantics match CPU exactly");
            }
            for (int64_t r = m; r < padded_rows(m); ++r) {
                semantic_dequantize_row(packed.data(), got.data(), k, r);
                for (float v : got) {
                    check(v == 0.0f, "padded rows decode to zero");
                }
                for (int64_t kt = 0; kt < k / 32; ++kt) {
                    uint8_t * tile = packed.data() + ((r / 32) * (k / 32) + kt) * TILE_SIZE;
                    tile[D_PLANE_OFFSET + 2 * (r % 32)] = 0xff;
                    tile[SCALE0_PLANE_OFFSET + r % 32] = 0xff;
                }
            }
            check(unpack_2d(packed.data(), packed_size, k, m, out.data(), raw_size), "padding corruption stays outside logical rows");
            check(std::memcmp(raw.data(), out.data(), raw_size) == 0, "padding does not leak into logical rows");
            check(!repack_2d(raw.data(), raw_size - 1, k, m, packed.data(), packed_size), "reject short raw source");
            check(!unpack_2d(packed.data(), packed_size - 1, k, m, out.data(), raw_size), "reject short packed source");
            check(!unpack_rows_2d(packed.data(), packed_size, k, m, m, 1, out.data(), raw_size), "reject padding readback");
            for (size_t plane : {D_PLANE_OFFSET, DMIN_PLANE_OFFSET}) {
                for (uint8_t mask : {0x01, 0xff}) {
                    for (int g = 1; g < 8; ++g) {
                        const size_t pos = g * TILE_SIZE + plane + (g & 1);
                        packed[pos] ^= mask;
                        check(!unpack_2d(packed.data(), packed_size, k, m, out.data(), raw_size), "reject duplicate d/dmin bit and byte corruption");
                        packed[pos] ^= mask;
                    }
                }
            }
            for (int64_t first = 0; first < m; first += 32) {
                const int64_t count = m - first < 32 ? m - first : 32;
                check_scalar(raw.data() + first * (k / QK_K), packed.data() + (first / 32) * (k / 32) * TILE_SIZE, k, count, rng);
            }
            ++cases;
        }
    }
    std::printf("Q2_K repack shapes: %d; duplicate d/dmin bit/byte, partial rows and padding: PASS\n", cases);
}

static void test_slices() {
    const int64_t k = 1024, rows = 33, slices = 2 * 3;
    const size_t raw_size = original_size_2d(k, rows);
    const size_t packed_size = repacked_size_2d(k, rows);
    std::vector<block_q2_K> raw(slices * raw_size / sizeof(block_q2_K)), out(raw.size());
    std::vector<uint8_t> packed(slices * packed_size);
    uint32_t rng = 0x382453ab;
    for (size_t i = 0; i < slices * raw_size; ++i) {
        ((uint8_t *) raw.data())[i] = (uint8_t) random32(rng);
    }
    for (int64_t slice = 0; slice < slices; ++slice) {
        const size_t offset = slice * raw_size / sizeof(block_q2_K);
        check(repack_2d(raw.data() + offset, raw_size, k, rows, packed.data() + slice * packed_size, packed_size), "4D slice repack");
        check(unpack_2d(packed.data() + slice * packed_size, packed_size, k, rows, out.data() + offset, raw_size), "4D slice unpack");
    }
    check(std::memcmp(raw.data(), out.data(), slices * raw_size) == 0, "4D slice roundtrip exact");
    // Read across a slice boundary with the same row split as backend get_tensor.
    const int64_t first = 31, count = 4;
    check(unpack_rows_2d(packed.data(), packed_size, k, rows, first, 2, out.data(), raw_size), "partial end of slice");
    check(unpack_rows_2d(packed.data() + packed_size, packed_size, k, rows, 0, 2, out.data() + 2 * (k / QK_K), raw_size), "partial next slice");
    check(std::memcmp(raw.data() + first * (k / QK_K), out.data(), count * (raw_size / rows)) == 0, "partial 4D readback exact");
}

int main() {
    test_sizes();
    test_semantics();
    test_repack();
    test_slices();
    std::printf("Q2_K scalar host cases: %d\n", scalar_cases);
    std::printf("%s: Q2_K repack tests (%d failures)\n", n_failed ? "FAIL" : "PASS", n_failed);
    return n_failed != 0;
}
