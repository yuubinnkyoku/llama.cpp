// Phase 54 - host side check of the iq3_s two row sharing transformation.
//
// The device build of the shared tiled_vec_dot_iq3_s_32x2 changed the 32 step
// logits hash, so before touching the DSP again this program checks whether the
// transformation itself is sound. It models the scalar reference
// (tiled_vec_dot_iq3_s_scalar_32x1, copied from hvx-mm-kernels-tiled.h) and a
// scalar version of the shared two row form, then compares them on random
// tiles.
//
// If the two agree here, the transformation is mathematically identical and the
// device difference must come from something the scalar model cannot see
// (register pressure, the restrict qualifiers, scratch aliasing, or a compiler
// difference). If they disagree, the bug is in the transformation and this
// program says where.
//
//   cc -O2 -o iq3s_share_check iq3s_share_check.c -lm && ./iq3s_share_check

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// x86 has no native fp16, so decode the IEEE half by hand. The DSP uses
// __fp16 natively; this only has to be bit exact for the values the tiles hold.
static float half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            // subnormal half, normalise it
            uint32_t e = 127 - 15 + 1;
            uint32_t m = mant;
            while ((m & 0x400u) == 0) { m <<= 1; --e; }
            m &= 0x3ffu;
            bits = sign | (e << 23) | (m << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// Random bytes make mostly NaN halves, which compare unequal even when both
// paths compute the same thing. Write finite halves instead so the comparison
// is about the arithmetic and not about NaN payloads.
static void store_half(uint8_t * p, float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t) ((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    uint16_t h;
    if (exp <= 0) {
        h = (uint16_t) sign;
    } else if (exp >= 0x1f) {
        h = (uint16_t) (sign | 0x7c00u);
    } else {
        h = (uint16_t) (sign | ((uint32_t) exp << 10) | (mant >> 13));
    }
    memcpy(p, &h, sizeof(h));
}

static uint16_t load_half(const uint8_t * p) {
    uint16_t h;
    memcpy(&h, p, sizeof(h));
    return h;
}

// Tile layout, from ggml/src/ggml-hexagon/htp/matmul-ops.h
#define HTP_MM_WEIGHT_TILE_SIZE_IQ3_S 512
#define HTP_IQ3S_INDEX_PLANE_OFFSET    0
#define HTP_IQ3S_SIGN_PLANE_OFFSET     256
#define HTP_IQ3S_QH_PLANE_OFFSET       384
#define HTP_IQ3S_SCALE_PLANE_OFFSET    416
#define HTP_IQ3S_D_PLANE_OFFSET        448
#define HTP_MM_ACT_TILE_SIZE_Q8_0      1152

// The reference, copied verbatim in structure from
// tiled_vec_dot_iq3_s_scalar_32x1 so the comparison is against the same code
// the DSP falls back to when scratch is null.
static void ref_32x1(const uint32_t n, float * restrict s,
        const void * restrict vx, const void * restrict vy, uint32_t valid_rows,
        const float * restrict sz, const uint32_t * restrict grid) {
    const uint8_t * weights = vx;
    const uint8_t * activation = vy;
    for (uint32_t row = 0; row < valid_rows; ++row) {
        float sum = 0.0f;
        for (uint32_t kt = 0; kt < n / 32; ++kt) {
            const uint8_t * tile = weights + kt * HTP_MM_WEIGHT_TILE_SIZE_IQ3_S;
            const int8_t * act = (const int8_t *) (activation + kt * HTP_MM_ACT_TILE_SIZE_Q8_0);
            const uint8_t qh = tile[HTP_IQ3S_QH_PLANE_OFFSET + row];
            const uint8_t scale = tile[HTP_IQ3S_SCALE_PLANE_OFFSET + row];
            const float d  = half_to_float(load_half(tile + HTP_IQ3S_D_PLANE_OFFSET + 2 * row));
            const float da = half_to_float(load_half((const uint8_t *) act + 1024));
            const float db = d * (1 + 2 * scale);
            int32_t dot = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t signs = tile[HTP_IQ3S_SIGN_PLANE_OFFSET + 32 * l + row];
                for (int j = 0; j < 8; ++j) {
                    const int i = 2 * l + j / 4;
                    const unsigned index = tile[HTP_IQ3S_INDEX_PLANE_OFFSET + 32 * i + row] | (((qh >> i) & 1u) << 8);
                    const int magnitude = (grid[index] >> (8 * (j % 4))) & 255;
                    const int weight = (signs & (1u << j)) ? -magnitude : magnitude;
                    dot += weight * act[i * 128 + j % 4];
                }
            }
            sum += (float) dot * db * da;
        }
        s[row] = sum + (sz ? sz[row] : 0.0f);
    }
}

// The shared form. The weight side work (the qh byte, the scale byte, the fp16
// d, the sign byte and the grid index) is computed once per (kt, row) and used
// for both activations, which is exactly what the DSP version does when it
// hoists the gathers and the finish groups out of the per row loop.
static void shared_32x2(const uint32_t n, float * restrict s0, float * restrict s1,
        const void * restrict vx, const void * restrict vy0, const void * restrict vy1,
        uint32_t valid_rows, const float * restrict sz0, const float * restrict sz1,
        const uint32_t * restrict grid) {
    const uint8_t * weights = vx;
    const uint8_t * activation0 = vy0;
    const uint8_t * activation1 = vy1;
    for (uint32_t row = 0; row < valid_rows; ++row) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        for (uint32_t kt = 0; kt < n / 32; ++kt) {
            const uint8_t * tile = weights + kt * HTP_MM_WEIGHT_TILE_SIZE_IQ3_S;
            const int8_t * act0 = (const int8_t *) (activation0 + kt * HTP_MM_ACT_TILE_SIZE_Q8_0);
            const int8_t * act1 = (const int8_t *) (activation1 + kt * HTP_MM_ACT_TILE_SIZE_Q8_0);
            // shared weight side
            const uint8_t qh = tile[HTP_IQ3S_QH_PLANE_OFFSET + row];
            const uint8_t scale = tile[HTP_IQ3S_SCALE_PLANE_OFFSET + row];
            const float d = half_to_float(load_half(tile + HTP_IQ3S_D_PLANE_OFFSET + 2 * row));
            const float db = d * (1 + 2 * scale);
            const float da0 = half_to_float(load_half((const uint8_t *) act0 + 1024));
            const float da1 = half_to_float(load_half((const uint8_t *) act1 + 1024));
            int32_t dot0 = 0;
            int32_t dot1 = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t signs = tile[HTP_IQ3S_SIGN_PLANE_OFFSET + 32 * l + row];
                for (int j = 0; j < 8; ++j) {
                    const int i = 2 * l + j / 4;
                    const unsigned index = tile[HTP_IQ3S_INDEX_PLANE_OFFSET + 32 * i + row] | (((qh >> i) & 1u) << 8);
                    const int magnitude = (grid[index] >> (8 * (j % 4))) & 255;
                    const int weight = (signs & (1u << j)) ? -magnitude : magnitude;
                    dot0 += weight * act0[i * 128 + j % 4];
                    dot1 += weight * act1[i * 128 + j % 4];
                }
            }
            sum0 += (float) dot0 * db * da0;
            sum1 += (float) dot1 * db * da1;
        }
        s0[row] = sum0 + (sz0 ? sz0[row] : 0.0f);
        s1[row] = sum1 + (sz1 ? sz1[row] : 0.0f);
    }
}

static uint32_t rng_state = 0x12345678u;
static uint32_t rng(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

int main(int argc, char ** argv) {
    const uint32_t n = 512;              // 16 k tiles
    const uint32_t valid_rows = 32;
    const int trials = argc > 1 ? atoi(argv[1]) : 200;

    const size_t tile_bytes = (n / 32) * HTP_MM_WEIGHT_TILE_SIZE_IQ3_S;
    const size_t act_bytes  = (n / 32) * HTP_MM_ACT_TILE_SIZE_Q8_0;

    uint8_t * w      = malloc(tile_bytes);
    uint8_t * a0     = malloc(act_bytes);
    uint8_t * a1     = malloc(act_bytes);
    uint32_t * grid  = malloc(2048 * sizeof(uint32_t));
    float * sz0      = malloc(valid_rows * sizeof(float));
    float * sz1      = malloc(valid_rows * sizeof(float));
    float * r0       = malloc(valid_rows * sizeof(float));
    float * r1       = malloc(valid_rows * sizeof(float));
    float * s0       = malloc(valid_rows * sizeof(float));
    float * s1       = malloc(valid_rows * sizeof(float));

    if (!w || !a0 || !a1 || !grid || !sz0 || !sz1 || !r0 || !r1 || !s0 || !s1) {
        fprintf(stderr, "out of memory\n");
        return 2;
    }

    for (int i = 0; i < 2048; ++i) {
        grid[i] = rng();
    }

    int bad = 0;
    double worst = 0.0;
    for (int t = 0; t < trials; ++t) {
        for (size_t i = 0; i < tile_bytes; ++i) w[i] = (uint8_t) rng();
        for (size_t i = 0; i < act_bytes; ++i) { a0[i] = (uint8_t) rng(); a1[i] = (uint8_t) rng(); }
        // overwrite the fp16 scale fields with finite values in a plausible range
        for (uint32_t kt = 0; kt < n / 32; ++kt) {
            uint8_t * tile = w + kt * HTP_MM_WEIGHT_TILE_SIZE_IQ3_S;
            for (uint32_t row = 0; row < valid_rows; ++row) {
                store_half(tile + HTP_IQ3S_D_PLANE_OFFSET + 2 * row,
                           0.001f + (float) (rng() % 1000) * 0.0001f);
            }
            uint8_t * act0 = a0 + kt * HTP_MM_ACT_TILE_SIZE_Q8_0;
            uint8_t * act1 = a1 + kt * HTP_MM_ACT_TILE_SIZE_Q8_0;
            store_half(act0 + 1024, 0.001f + (float) (rng() % 1000) * 0.0001f);
            store_half(act1 + 1024, 0.001f + (float) (rng() % 1000) * 0.0001f);
        }
        for (uint32_t i = 0; i < valid_rows; ++i) {
            sz0[i] = (float) ((int32_t) rng() % 1000) / 100.0f;
            sz1[i] = (float) ((int32_t) rng() % 1000) / 100.0f;
        }

        ref_32x1(n, r0, w, a0, valid_rows, sz0, grid);
        ref_32x1(n, r1, w, a1, valid_rows, sz1, grid);
        shared_32x2(n, s0, s1, w, a0, a1, valid_rows, sz0, sz1, grid);

        for (uint32_t i = 0; i < valid_rows; ++i) {
            const double d0 = fabs((double) r0[i] - (double) s0[i]);
            const double d1 = fabs((double) r1[i] - (double) s1[i]);
            const double d = d0 > d1 ? d0 : d1;
            if (d > worst) worst = d;
            if (!(d == 0.0)) {
                if (bad < 5) {
                    printf("trial %d row %u: ref %.9g/%.9g shared %.9g/%.9g\n",
                           t, i, r0[i], r1[i], s0[i], s1[i]);
                }
                ++bad;
            }
        }
    }

    printf("trials %d rows_per_trial %u\n", trials, valid_rows);
    printf("mismatched rows %d\n", bad);
    printf("worst abs diff %.9g\n", worst);
    printf("%s\n", bad == 0 ? "IDENTICAL" : "DIFFERENT");
    return bad == 0 ? 0 : 1;
}


