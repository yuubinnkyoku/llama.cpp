#pragma once

#include <stdint.h>
#include <string.h>
#include "matmul-tile.h"

#ifndef __hexagon__
#include "ggml.h"
#endif

static inline float q2_k_load_f16(const uint8_t * src) {
#ifdef __hexagon__
    __fp16 value;
    memcpy(&value, src, sizeof(value));
    return (float) value;
#else
    ggml_fp16_t value;
    memcpy(&value, src, sizeof(value));
    return ggml_fp16_to_fp32(value);
#endif
}

static inline void tiled_vec_dot_q2_K_32x1(const uint32_t n, float * s, const void * vx, const void * vy, uint32_t valid_rows, const float * sz) {
#ifdef __clang__
#pragma clang fp reassociate(off) contract(off)
#endif
    const uint8_t * weights = (const uint8_t *) vx;
    const uint8_t * activation = (const uint8_t *) vy;
    for (uint32_t row = 0; row < valid_rows; ++row) {
        float sum = 0.0f;
        for (uint32_t kt = 0; kt < n / 32; ++kt) {
            // DMA copies 448 storage bytes into each 512-byte VTCM tile.
            const uint8_t * tile = weights + kt * HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q2_K;
            const int8_t * act = (const int8_t *) (activation + kt * HTP_MM_ACT_TILE_SIZE_Q8_0);
            const float d = q2_k_load_f16(tile + Q2K_D_PLANE_OFFSET + 2 * row);
            const float dmin = q2_k_load_f16(tile + Q2K_DMIN_PLANE_OFFSET + 2 * row);
            const float da = q2_k_load_f16((const uint8_t *) act + HTP_MM_ACT_SCALE_PLANE_OFFSET_Q8_0);
            for (int k = 0; k < 32; ++k) {
                const uint8_t sc = tile[(k < 16 ? Q2K_SCALE0_PLANE_OFFSET : Q2K_SCALE1_PLANE_OFFSET) + row];
                const float D = d * (sc & 15);
                const float M = dmin * (sc >> 4);
                const int q = (tile[Q2K_QUANT_PLANE_OFFSET + (k / 4) * 32 + row] >> (2 * (k & 3))) & 3;
                sum += (D * q - M) * (float) act[(k / 4) * 128 + (k & 3)] * da;
            }
        }
        s[row] = sum + (sz ? sz[row] : 0.0f);
    }
}

static inline void tiled_vec_dot_q2_K_32x2(const uint32_t n, float * s0, float * s1, const void * vx, const void * vy0, const void * vy1, uint32_t valid_rows, const float * sz0, const float * sz1) {
    tiled_vec_dot_q2_K_32x1(n, s0, vx, vy0, valid_rows, sz0);
    tiled_vec_dot_q2_K_32x1(n, s1, vx, vy1, valid_rows, sz1);
}
