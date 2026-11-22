// Phase 60 - host side check of the NEON microkernel algebra.
//
// The NEON kernel cannot run on this host, but its arithmetic can: the kernel is
// "accumulate the exact integer subdot in int32, then scale once", while the
// scalar reference is "scale each subblock in float and accumulate in float".
// This program replicates both formulas exactly as written in
// ggml/src/ggml-cpu/tiled/tiled-kernel.cpp and compares them on random data,
// for the two shapes the model actually uses (SUBBLK=32 and SUBBLK=16, BIAS=128).
//
//   g++ -O2 -o p60_neon_algebra p60_neon_algebra.cpp && ./p60_neon_algebra
//
// It also reports the worst int32 magnitude reached by the accumulator, so the
// "does the int32 sum overflow where the float sum would not" question is
// answered with numbers instead of a guess.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>

static constexpr int TILE_K  = 256;
static constexpr int MICRO   = 16;
static constexpr int BIAS    = 128; // iq2/iq3 family
static constexpr int BIAS_QK = 0;   // q2_K / q4_K / q5_K (HAS_MIN, BIAS 0)
static constexpr int BIAS_Q6 = 32;  // q6_K
static constexpr int BIAS_Q3 = 4;   // q3_K

// scalar reference: tiled_run_microtile_scalar, HAS_MIN=false, BIAS=128
static void ref_microtile(int SUBBLK, const uint8_t * q0, const int8_t * q1,
                          const int32_t * scales, const float * d0, const float * d1,
                          float * buf, int buf_stride, int64_t * max_acc) {
    const int NB = TILE_K / SUBBLK;
    const int NS = SUBBLK / 16;

    float acc[MICRO][MICRO];
    for (int i = 0; i < MICRO; i++) for (int j = 0; j < MICRO; j++) acc[i][j] = 0.0f;

    for (int s = 0; s < NB; s++) {
        for (int j = 0; j < MICRO; j++) {
            // bsum of the activation subblock (q8_K bsums are the per-16 sums)
            int32_t bsum = 0;
            for (int u = 0; u < NS; u++) {
                int32_t t = 0;
                for (int e = 0; e < 16; e++) t += q1[j * TILE_K + s * SUBBLK + u * 16 + e];
                bsum += t;
            }
            for (int i = 0; i < MICRO; i++) {
                int32_t raw = 0;
                for (int e = 0; e < SUBBLK; e++) {
                    raw += (int32_t) q0[i * TILE_K + s * SUBBLK + e] * (int32_t) q1[j * TILE_K + s * SUBBLK + e];
                }
                int32_t corr = raw - BIAS * bsum;
                const int32_t scales_raw = scales[i * NB + s] * corr;
                acc[i][j] += (float) d0[i] * (float) scales_raw;
            }
        }
    }
    for (int i = 0; i < MICRO; i++) {
        for (int j = 0; j < MICRO; j++) {
            buf[i * buf_stride + j] += d1[j] * acc[i][j];
        }
    }
    (void) max_acc;
}

// NEON kernel algebra: w = q0 - BIAS (signed), exact int32 subdot, int32 accumulate
// across subblocks, one float scale at the end.
static void neon_microtile(int SUBBLK, const uint8_t * q0, const int8_t * q1,
                           const int32_t * scales, const float * d0, const float * d1,
                           float * buf, int buf_stride, int64_t * max_acc) {
    const int NB = TILE_K / SUBBLK;

    for (int i = 0; i < MICRO; i++) {
        int32_t acc[MICRO];
        for (int j = 0; j < MICRO; j++) acc[j] = 0;

        for (int s = 0; s < NB; s++) {
            for (int j = 0; j < MICRO; j++) {
                int32_t d = 0;
                for (int e = 0; e < SUBBLK; e++) {
                    const int32_t w = (int32_t) q0[i * TILE_K + s * SUBBLK + e] - BIAS;
                    d += w * (int32_t) q1[j * TILE_K + s * SUBBLK + e];
                }
                acc[j] += d * scales[i * NB + s];
            }
        }
        for (int j = 0; j < MICRO; j++) {
            if (std::llabs((long long) acc[j]) > *max_acc) *max_acc = std::llabs((long long) acc[j]);
            buf[i * buf_stride + j] += d1[j] * (d0[i] * (float) acc[j]);
        }
    }
}

static uint32_t rng_state = 12345;
static uint32_t rnd(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

// Narrow-path variant: num_k slabs held at row stride num_k*256, one slab per call.
// Both formulas are re-derived with the runtime strides the kernel uses, so this
// checks the NK=0 instantiation's indexing as well as its arithmetic.
static void ref_microtile_nk(int SUBBLK, int num_k, int slab, const uint8_t * q0, const int8_t * q1,
                             const int32_t * scales, const float * d0, const float * d1,
                             float * buf, int buf_stride) {
    const int NB = TILE_K / SUBBLK;
    const int NS = SUBBLK / 16;
    const int qk_stride = num_k * TILE_K;
    const int qk_off = slab * TILE_K;
    const int nb_stride = NB * num_k;
    const int nb_off = slab * NB;
    const int bs_stride = MICRO;
    const int bs_off = slab * MICRO;

    float acc[MICRO][MICRO];
    for (int i = 0; i < MICRO; i++) for (int j = 0; j < MICRO; j++) acc[i][j] = 0.0f;

    for (int s = 0; s < NB; s++) {
        for (int j = 0; j < MICRO; j++) {
            int32_t bsum = 0;
            for (int u = 0; u < NS; u++) {
                int32_t t = 0;
                for (int e = 0; e < 16; e++) t += q1[j * qk_stride + qk_off + s * SUBBLK + u * 16 + e];
                bsum += t;
            }
            for (int i = 0; i < MICRO; i++) {
                int32_t raw = 0;
                for (int e = 0; e < SUBBLK; e++) {
                    raw += (int32_t) q0[i * qk_stride + qk_off + s * SUBBLK + e] *
                           (int32_t) q1[j * qk_stride + qk_off + s * SUBBLK + e];
                }
                const int32_t corr = raw - BIAS * bsum;
                acc[i][j] += (float) d0[slab * MICRO + i] * (float) (scales[i * nb_stride + nb_off + s] * corr);
            }
        }
    }
    for (int i = 0; i < MICRO; i++) {
        for (int j = 0; j < MICRO; j++) {
            buf[i * buf_stride + j] += d1[bs_off + j] * acc[i][j];
        }
    }
    (void) bs_stride;
}

static void neon_microtile_nk(int SUBBLK, int num_k, int slab, const uint8_t * q0, const int8_t * q1,
                              const int32_t * scales, const float * d0, const float * d1,
                              float * buf, int buf_stride) {
    const int NB = TILE_K / SUBBLK;
    const int qk_stride = num_k * TILE_K;
    const int qk_off = slab * TILE_K;
    const int nb_stride = NB * num_k;
    const int nb_off = slab * NB;
    const int bs_off = slab * MICRO;

    for (int i = 0; i < MICRO; i++) {
        int32_t acc[MICRO];
        for (int j = 0; j < MICRO; j++) acc[j] = 0;
        for (int s = 0; s < NB; s++) {
            for (int j = 0; j < MICRO; j++) {
                int32_t d = 0;
                for (int e = 0; e < SUBBLK; e++) {
                    const int32_t w = (int32_t) q0[i * qk_stride + qk_off + s * SUBBLK + e] - BIAS;
                    d += w * (int32_t) q1[j * qk_stride + qk_off + s * SUBBLK + e];
                }
                acc[j] += d * scales[i * nb_stride + nb_off + s];
            }
        }
        for (int j = 0; j < MICRO; j++) {
            buf[i * buf_stride + j] += d1[bs_off + j] * (d0[slab * MICRO + i] * (float) acc[j]);
        }
    }
}

// HAS_MIN variant (qK family): the scalar reference also subtracts dmin*mins*bsum.
// Both formulas are written as in the source; the NEON side keeps the bsums term in
// the same int32 accumulator.
static void ref_microtile_min(int SUBBLK, int BIASV, const uint8_t * q0, const int8_t * q1,
                              const int32_t * scales, const int32_t * mins,
                              const float * d0, const float * dmin, const float * d1,
                              float * buf, int buf_stride) {
    const int NB = TILE_K / SUBBLK;
    const int NS = SUBBLK / 16;

    float acc[MICRO][MICRO];
    for (int i = 0; i < MICRO; i++) for (int j = 0; j < MICRO; j++) acc[i][j] = 0.0f;

    for (int s = 0; s < NB; s++) {
        for (int j = 0; j < MICRO; j++) {
            int32_t bsum = 0;
            for (int u = 0; u < NS; u++) {
                int32_t t = 0;
                for (int e = 0; e < 16; e++) t += q1[j * TILE_K + s * SUBBLK + u * 16 + e];
                bsum += t;
            }
            for (int i = 0; i < MICRO; i++) {
                int32_t raw = 0;
                for (int e = 0; e < SUBBLK; e++) {
                    raw += (int32_t) q0[i * TILE_K + s * SUBBLK + e] * (int32_t) q1[j * TILE_K + s * SUBBLK + e];
                }
                const int32_t scales_raw = scales[i * NB + s] * raw;
                const int32_t mins_bsum  = mins[i * NB + s] * bsum;
                acc[i][j] += (float) d0[i] * (float) scales_raw - (float) dmin[i] * (float) mins_bsum;
            }
        }
    }
    for (int i = 0; i < MICRO; i++) {
        for (int j = 0; j < MICRO; j++) {
            buf[i * buf_stride + j] += d1[j] * acc[i][j];
        }
    }
}

// Models the NEON lane structure, not just the arithmetic: acc[j] is an int32x4 whose
// lanes are summed by vaddvq_s32, while the min term is a plain scalar. A vector
// accumulator for the min term would be counted 4 times by the horizontal sum, so the
// emulation has to keep the two shapes apart to be able to catch that.
static void neon_microtile_min(int SUBBLK, int BIASV, const uint8_t * q0, const int8_t * q1,
                               const int32_t * scales, const int32_t * mins,
                               const float * d0, const float * dmin, const float * d1,
                               float * buf, int buf_stride) {
    const int NB = TILE_K / SUBBLK;
    const int NS = SUBBLK / 16;
    const int NV = SUBBLK / 16;

    for (int i = 0; i < MICRO; i++) {
        int32_t acc[MICRO][4], acc_min[MICRO];
        for (int j = 0; j < MICRO; j++) {
            for (int l = 0; l < 4; l++) acc[j][l] = 0;
            acc_min[j] = 0;
        }
        for (int s = 0; s < NB; s++) {
            for (int j = 0; j < MICRO; j++) {
                // vdotq_s32: lane l takes byte 4l..4l+3 of each 16-byte vector
                int32_t d[4] = { 0, 0, 0, 0 };
                for (int v = 0; v < NV; v++) {
                    for (int l = 0; l < 4; l++) {
                        for (int e = 4 * l; e < 4 * l + 4; e++) {
                            const int32_t w = (int32_t) q0[i * TILE_K + s * SUBBLK + 16 * v + e] - BIAS_QK;
                            d[l] += w * (int32_t) q1[j * TILE_K + s * SUBBLK + 16 * v + e];
                        }
                    }
                }
                for (int l = 0; l < 4; l++) acc[j][l] += d[l] * scales[i * NB + s];

                int32_t bsum = 0;
                for (int u = 0; u < NS; u++) {
                    int32_t t = 0;
                    for (int e = 0; e < 16; e++) t += q1[j * TILE_K + s * SUBBLK + u * 16 + e];
                    bsum += t;
                }
                acc_min[j] += bsum * mins[i * NB + s];
            }
        }
        for (int j = 0; j < MICRO; j++) {
            // vaddvq_s32(acc[j]) then the scalar min term
            const int32_t s1 = acc[j][0] + acc[j][1] + acc[j][2] + acc[j][3];
            float v = d0[i] * (float) s1 - dmin[i] * (float) acc_min[j];
            buf[i * buf_stride + j] += d1[j] * v;
        }
    }
}

int main(void) {
    const int buf_stride = MICRO;
    int failures = 0;

    for (int SUBBLK : {32, 16}) {
        const int NB = TILE_K / SUBBLK;
        // scale ranges that match the real formats: iq2/iq3 use 1..31 style scales
        for (int trial = 0; trial < 200; trial++) {
            std::vector<uint8_t> q0(MICRO * TILE_K);
            std::vector<int8_t>  q1(MICRO * TILE_K);
            std::vector<int32_t> scales(MICRO * NB);
            std::vector<float>   d0(MICRO), d1(MICRO);

            for (auto & v : q0) v = (uint8_t) (rnd() & 0xFF);
            for (auto & v : q1) v = (int8_t)  (rnd() & 0xFF);
            for (auto & v : scales) v = (int32_t) (1 + 2 * (rnd() % 16));
            for (auto & v : d0) v = (float) ((rnd() % 1000) + 1) / 1000.0f;
            for (auto & v : d1) v = (float) ((rnd() % 1000) + 1) / 1000.0f;

            std::vector<float> buf_ref(MICRO * buf_stride, 0.0f);
            std::vector<float> buf_neon(MICRO * buf_stride, 0.0f);
            int64_t max_acc = 0;

            ref_microtile (SUBBLK, q0.data(), q1.data(), scales.data(), d0.data(), d1.data(), buf_ref.data(),  buf_stride, &max_acc);
            neon_microtile(SUBBLK, q0.data(), q1.data(), scales.data(), d0.data(), d1.data(), buf_neon.data(), buf_stride, &max_acc);

            double worst_rel = 0.0;
            for (size_t k = 0; k < buf_ref.size(); k++) {
                const double a = buf_ref[k], b = buf_neon[k];
                const double den = std::fabs(a) > 1e-6 ? std::fabs(a) : 1e-6;
                const double rel = std::fabs(a - b) / den;
                if (rel > worst_rel) worst_rel = rel;
            }
            // float reassociation only: the two orders differ by rounding, not by algebra
            if (worst_rel > 1e-4) {
                printf("FAIL SUBBLK=%d trial=%d worst_rel=%.3e\n", SUBBLK, trial, worst_rel);
                failures++;
            }
            if (trial == 0) {
                printf("SUBBLK=%d max |int32 acc| = %lld (int32 limit 2147483647)\n",
                       SUBBLK, (long long) max_acc);
            }
        }
        printf("SUBBLK=%d: 200 trials done\n", SUBBLK);
    }

    // narrow path: num_k slabs, one slab per call, runtime strides
    for (int SUBBLK : {32, 16}) {
        const int NB = TILE_K / SUBBLK;
        for (int num_k : {2, 4}) {
            for (int trial = 0; trial < 50; trial++) {
                std::vector<uint8_t> q0(MICRO * num_k * TILE_K);
                std::vector<int8_t>  q1(MICRO * num_k * TILE_K);
                std::vector<int32_t> scales(MICRO * NB * num_k);
                std::vector<float>   d0(MICRO * num_k), d1(MICRO * num_k);

                for (auto & v : q0) v = (uint8_t) (rnd() & 0xFF);
                for (auto & v : q1) v = (int8_t)  (rnd() & 0xFF);
                for (auto & v : scales) v = (int32_t) (1 + 2 * (rnd() % 16));
                for (auto & v : d0) v = (float) ((rnd() % 1000) + 1) / 1000.0f;
                for (auto & v : d1) v = (float) ((rnd() % 1000) + 1) / 1000.0f;

                std::vector<float> buf_ref(MICRO * buf_stride, 0.0f);
                std::vector<float> buf_neon(MICRO * buf_stride, 0.0f);

                for (int slab = 0; slab < num_k; slab++) {
                    ref_microtile_nk (SUBBLK, num_k, slab, q0.data(), q1.data(), scales.data(), d0.data(), d1.data(), buf_ref.data(),  buf_stride);
                    neon_microtile_nk(SUBBLK, num_k, slab, q0.data(), q1.data(), scales.data(), d0.data(), d1.data(), buf_neon.data(), buf_stride);
                }

                double worst_rel = 0.0;
                for (size_t k = 0; k < buf_ref.size(); k++) {
                    const double a = buf_ref[k], b = buf_neon[k];
                    const double den = std::fabs(a) > 1e-6 ? std::fabs(a) : 1e-6;
                    const double rel = std::fabs(a - b) / den;
                    if (rel > worst_rel) worst_rel = rel;
                }
                if (worst_rel > 1e-4) {
                    printf("FAIL narrow SUBBLK=%d num_k=%d trial=%d worst_rel=%.3e\n", SUBBLK, num_k, trial, worst_rel);
                    failures++;
                }
            }
            printf("narrow SUBBLK=%d num_k=%d: 50 trials done\n", SUBBLK, num_k);
        }
    }

    // HAS_MIN (qK family): the bsums term must match too. In the dispatch table HAS_MIN
    // is true only for q2_K/q4_K/q5_K, and all three have BIAS = 0, so that is the only
    // combination to check here. (q6_K BIAS=32 and q3_K BIAS=4 are HAS_MIN=false and
    // are covered by the no-min sweep above.)
    for (int biasv : {BIAS_QK}) {
    for (int SUBBLK : {32, 16}) {
        const int NB = TILE_K / SUBBLK;
        for (int trial = 0; trial < 100; trial++) {
            std::vector<uint8_t> q0(MICRO * TILE_K);
            std::vector<int8_t>  q1(MICRO * TILE_K);
            std::vector<int32_t> scales(MICRO * NB), mins(MICRO * NB);
            std::vector<float>   d0(MICRO), dmin(MICRO), d1(MICRO);
            // qK codes are small unsigned values that must fit in int8 (BIAS=0 means no debias),
            // so generate 0..127 here, not 0..255.
            for (auto & v : q0) v = (uint8_t) (rnd() & 0x7F);
            for (auto & v : q1) v = (int8_t)  (rnd() & 0xFF);
            for (auto & v : scales) v = (int32_t) (1 + 2 * (rnd() % 16));
            for (auto & v : mins)   v = (int32_t) (rnd() % 32);
            for (auto & v : d0)   v = (float) ((rnd() % 1000) + 1) / 1000.0f;
            for (auto & v : dmin) v = (float) ((rnd() % 1000) + 1) / 1000.0f;
            for (auto & v : d1)   v = (float) ((rnd() % 1000) + 1) / 1000.0f;

            std::vector<float> buf_ref(MICRO * buf_stride, 0.0f);
            std::vector<float> buf_neon(MICRO * buf_stride, 0.0f);
            ref_microtile_min (SUBBLK, biasv, q0.data(), q1.data(), scales.data(), mins.data(), d0.data(), dmin.data(), d1.data(), buf_ref.data(),  buf_stride);
            neon_microtile_min(SUBBLK, biasv, q0.data(), q1.data(), scales.data(), mins.data(), d0.data(), dmin.data(), d1.data(), buf_neon.data(), buf_stride);

            double worst_rel = 0.0;
            for (size_t k = 0; k < buf_ref.size(); k++) {
                const double a = buf_ref[k], b = buf_neon[k];
                const double den = std::fabs(a) > 1e-6 ? std::fabs(a) : 1e-6;
                const double rel = std::fabs(a - b) / den;
                if (rel > worst_rel) worst_rel = rel;
            }
            if (worst_rel > 1e-4) {
                printf("FAIL hasmin SUBBLK=%d trial=%d worst_rel=%.3e\n", SUBBLK, trial, worst_rel);
                failures++;
            }
        }
        printf("HAS_MIN bias=%d SUBBLK=%d: 100 trials done\n", biasv, SUBBLK);
    }
    }

    printf(failures ? "RESULT FAIL (%d)\n" : "RESULT PASS\n", failures);
    return failures ? 1 : 0;
}
