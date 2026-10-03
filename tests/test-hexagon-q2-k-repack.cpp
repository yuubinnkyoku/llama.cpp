#include "ggml.h"
#include "ggml-quants.h"
#include "ggml-hexagon/q2-k-repack.h"

#include "ggml-hexagon/htp/q2-k-scalar.h"

#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cstring>
#include <vector>

static_assert(sizeof(block_q2_K) == 84, "Q2_K raw block size");
static_assert(HTP_MM_WEIGHT_TILE_SIZE_Q2_K == 448 && HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q2_K == 512, "Q2_K storage/staging sizes");
static_assert(Q2K_QUANT_PLANE_OFFSET == 0 && Q2K_SCALE0_PLANE_OFFSET == 256 && Q2K_SCALE1_PLANE_OFFSET == 288 && Q2K_D_PLANE_OFFSET == 320 && Q2K_DMIN_PLANE_OFFSET == 384, "Q2_K frozen offsets");
static_assert(Q2K_D_PLANE_OFFSET + 128 == HTP_MM_WEIGHT_TILE_SIZE_Q2_K, "Q2_K D/DMIN vector load boundary");

using namespace ggml_hexagon_q2k;

static int n_failed = 0;
static int semantic_cases = 0;
static int scalar_cases = 0;
static int unpack_cases = 0;
static int integer_cases = 0;
static int algebra_cases = 0;
static double algebra_nmse_max = 0.0;

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

static uint32_t unpack_word(uint32_t packed) {
    return (packed & 3) | (((packed >> 2) & 3) << 8) | (((packed >> 4) & 3) << 16) | ((packed >> 6) << 24);
}

static void model_unpack_plane(const uint8_t * tile, int plane, uint32_t * words) {
    uint8_t load[128];
    std::memcpy(load, tile + Q2K_QUANT_PLANE_OFFSET + 32 * plane, sizeof(load));
    for (int row = 0; row < 32; ++row) {
        words[row] = unpack_word(load[row]);
    }
}

static void model_dot(int64_t k, float * out, const uint8_t * weights, const uint8_t * activation, int64_t rows, const float * bias) {
    float sums[32] = {};
    for (int64_t kt = 0; kt < k / 32; ++kt) {
        const uint8_t * tile = weights + kt * ALIGNED_TILE_SIZE;
        const int8_t * act = (const int8_t *) (activation + kt * HTP_MM_ACT_TILE_SIZE_Q8_0);
        const float da = q2_k_load_f16((const uint8_t *) act + HTP_MM_ACT_SCALE_PLANE_OFFSET_Q8_0);
        float tile_sum[32] = {};
        for (int subgroup = 0; subgroup < 2; ++subgroup) {
            int dot[32] = {}, suma[32] = {};
            for (int p = 4 * subgroup; p < 4 * subgroup + 4; ++p) {
                uint32_t words[32];
                model_unpack_plane(tile, p, words);
                for (int row = 0; row < 32; ++row) {
                    for (int j = 0; j < 4; ++j) {
                        const int a = act[p * 128 + 4 * row + j];
                        dot[row] += (int) ((words[row] >> (8 * j)) & 255) * a;
                        suma[row] += a;
                    }
                }
            }
            for (int row = 0; row < 32; ++row) {
                const uint8_t sc = tile[(subgroup ? Q2K_SCALE1_PLANE_OFFSET : Q2K_SCALE0_PLANE_OFFSET) + row];
                const float base_d = q2_k_load_f16(tile + Q2K_D_PLANE_OFFSET + 2 * row) * da;
                const float base_m = q2_k_load_f16(tile + Q2K_DMIN_PLANE_OFFSET + 2 * row) * da;
                tile_sum[row] += ((float) dot[row] * base_d) * (sc & 15) - ((float) suma[row] * base_m) * (sc >> 4);
            }
        }
        for (int row = 0; row < 32; ++row) {
            sums[row] += tile_sum[row];
        }
    }
    for (int64_t row = 0; row < rows; ++row) {
        out[row] = sums[row] + (bias ? bias[row] : 0.0f);
    }
}

static void check_algebra_float(const float * ref, const float * got, int64_t rows) {
    double error = 0.0, energy = 0.0;
    for (int64_t row = 0; row < rows; ++row) {
        const double diff = (double) ref[row] - got[row];
        error += diff * diff;
        energy += (double) ref[row] * ref[row];
    }
    const double nmse = energy ? error / energy : error;
    if (nmse > algebra_nmse_max) {
        algebra_nmse_max = nmse;
    }
    check(std::isfinite(nmse) && nmse <= 1e-7, "subgroup model meets backend NMSE contract");
    ++algebra_cases;
}

static void test_algebra() {
    uint32_t rng = 0x198c273a;
    std::vector<uint8_t> tile(ALIGNED_TILE_SIZE);
    for (int packed = 0; packed < 256; ++packed) {
        for (uint8_t & byte : tile) {
            byte = (uint8_t) random32(rng);
        }
        for (int p = 0; p < 8; ++p) {
            std::memset(tile.data() + QUANT_PLANE_OFFSET + 32 * p, packed, 32);
        }
        for (int p = 0; p < 8; ++p) {
            uint32_t words[32];
            model_unpack_plane(tile.data(), p, words);
            for (int row = 0; row < 32; ++row) {
                for (int j = 0; j < 4; ++j) {
                    check(((words[row] >> (8 * j)) & 255) == ((packed >> (2 * j)) & 3), "2-bit word unpack exact");
                }
                ++unpack_cases;
            }
        }
    }
    for (int sc = 0; sc < 256; ++sc) {
        for (int subgroup = 0; subgroup < 2; ++subgroup) {
            const size_t offset = subgroup ? SCALE1_PLANE_OFFSET : SCALE0_PLANE_OFFSET;
            std::memset(tile.data() + offset, sc, 32);
            uint8_t load[128];
            std::memcpy(load, tile.data() + offset, sizeof(load));
            for (int row = 0; row < 32; ++row) {
                const uint32_t widened = load[row];
                check((widened & 15) == (unsigned) sc % 16 && (widened >> 4) == (unsigned) sc / 16, "scale/min widened lanes exact");
            }
        }
    }
    const int8_t boundaries[] = {-128, -127, -1, 0, 1, 126, 127};
    const uint8_t nibbles[] = {0, 1, 15};
    for (int c = 0; c < 5000; ++c) {
        int8_t a[32];
        uint8_t q[32][32];
        for (int k = 0; k < 32; ++k) {
            a[k] = c < 63 ? boundaries[c % 7] : c < 126 ? boundaries[k % 7] : (int8_t) ((int) (random32(rng) % 256) - 128);
        }
        for (int row = 0; row < 32; ++row) {
            for (int p = 0; p < 8; ++p) {
                uint8_t packed = 0;
                for (int j = 0; j < 4; ++j) {
                    q[row][4 * p + j] = c < 63 ? (uint8_t) ((c / 7) % 4) : c < 126 ? (uint8_t) (j % 2 ? 2 : 1) : (uint8_t) (random32(rng) & 3);
                    packed |= q[row][4 * p + j] << (2 * j);
                }
                tile[QUANT_PLANE_OFFSET + 32 * p + row] = packed;
            }
        }
        for (int subgroup = 0; subgroup < 2; ++subgroup) {
            int dot[32] = {}, suma[32] = {};
            for (int p = 4 * subgroup; p < 4 * subgroup + 4; ++p) {
                uint32_t words[32];
                model_unpack_plane(tile.data(), p, words);
                for (int row = 0; row < 32; ++row) {
                    for (int j = 0; j < 4; ++j) {
                        dot[row] += (int) ((words[row] >> (8 * j)) & 255) * a[4 * p + j];
                        suma[row] += a[4 * p + j];
                    }
                }
            }
            for (int row = 0; row < 32; ++row) {
                int ref_dot = 0, ref_suma = 0;
                float ref = 0.0f, sum_abs = 0.0f;
                const int scale = c < 126 ? nibbles[(c / 7) % 3] : random32(rng) & 15;
                const int min = c < 126 ? nibbles[(c / 21) % 3] : random32(rng) & 15;
                const uint8_t sc = (uint8_t) (scale | (min << 4));
                check((sc & 15) == scale && (sc >> 4) == min, "scale/min nibble exact");
                const float d = ggml_fp16_to_fp32(ggml_fp32_to_fp16(((int) (random32(rng) % 257) - 128) / 8192.0f));
                const float dm = ggml_fp16_to_fp32(ggml_fp32_to_fp16(((int) (random32(rng) % 257) - 128) / 4096.0f));
                const float da = ggml_fp16_to_fp32(ggml_fp32_to_fp16((random32(rng) % 32) / 1024.0f));
                for (int k = 16 * subgroup; k < 16 * subgroup + 16; ++k) {
                    ref_dot += q[row][k] * a[k];
                    ref_suma += a[k];
                    const float term = (d * scale * q[row][k] - dm * min) * a[k] * da;
                    ref += term;
                    sum_abs += std::fabs(term);
                }
                check(dot[row] == ref_dot, "integer sum_qa exact");
                check(suma[row] == ref_suma, "integer sum_a exact");
                // 16 terms: abs(sum_qa) <= 6144, abs(sum_a) <= 2048.
                check(std::abs(dot[row]) <= 6144 && std::abs(suma[row]) <= 2048, "integer bounds");
                const float got = ((float) dot[row] * (d * da)) * scale - ((float) suma[row] * (dm * da)) * min;
                const float magnitude = std::fabs(d * da * scale * dot[row]) + std::fabs(dm * da * min * suma[row]);
                check(std::fabs(got - ref) <= 32 * FLT_EPSILON * (sum_abs + magnitude), "subgroup formula vs per-element rounding bound");
                ++integer_cases;
            }
        }
    }
    std::printf("Q2_K unpack lanes: %d; sum_qa/sum_a subgroup lanes: %d\n", unpack_cases, integer_cases);
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
            for (int lane = 0; lane < 64; ++lane) {
                std::memcpy(act + HTP_MM_ACT_SCALE_PLANE_OFFSET_Q8_0 + 2 * lane, &da, sizeof(da));
            }
            for (int j = 0; j < 32; ++j) {
                const int8_t aq = (int8_t) ((int) (random32(rng) % 255) - 127);
                (a ? aq1 : aq0)[kt * 32 + j] = aq;
                for (int lane = 0; lane < 32; ++lane) {
                    act[(j / 4) * 128 + 4 * lane + (j & 3)] = (uint8_t) aq;
                }
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
    float model0[32], model1[32], padding[32];
    for (int row = 0; row < 32; ++row) {
        model0[row] = model1[row] = padding[row] = 12345.0f;
    }
    model_dot(k, model0, staged.data(), act0.data(), rows, sz);
    model_dot(k, model1, staged.data(), act1.data(), rows, sz);
    check_algebra_float(got0, model0, rows);
    check_algebra_float(got1, model1, rows);
    for (size_t kt = 0; kt < ntiles; ++kt) {
        for (size_t j = TILE_SIZE; j < ALIGNED_TILE_SIZE; ++j) {
            staged[kt * ALIGNED_TILE_SIZE + j] = (uint8_t) random32(rng);
        }
    }
    model_dot(k, padding, staged.data(), act0.data(), rows, sz);
    check(std::memcmp(model0, padding, sizeof(model0)) == 0, "vector-load model is staging-padding independent");
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
        check(model0[r] == 12345.0f && model1[r] == 12345.0f, "algebra model writes only valid rows");
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
    test_algebra();
    test_semantics();
    test_repack();
    test_slices();
    std::printf("Q2_K scalar host cases: %d\n", scalar_cases);
    std::printf("Q2_K subgroup float batches: %d; max NMSE: %.9g (limit 1e-7)\n", algebra_cases, algebra_nmse_max);
    std::printf("%s: Q2_K repack tests (%d failures)\n", n_failed ? "FAIL" : "PASS", n_failed);
    return n_failed != 0;
}
