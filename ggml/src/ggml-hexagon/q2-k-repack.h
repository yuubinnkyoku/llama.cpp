#pragma once

#include "ggml-quants.h"
#include "q2-k-layout.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace ggml_hexagon_q2k {

static_assert(QK_K == 256, "Q2_K requires 256 weights per superblock");
static_assert(sizeof(block_q2_K) == 84, "Q2_K source block layout changed");

// 32 rows x 32 weights: raw 336 B (2.625 bpw), repacked 448 B (3.5 bpw), overhead 33.333333% before row padding.
// Each 16-weight subgroup supports d*scale*da*sum(q*a) - dmin*min*da*sum(a).
static constexpr int64_t TILE_ROWS = 32;
static constexpr int64_t TILE_COLS = 32;
static constexpr int64_t GROUPS_PER_SUPERBLOCK = 8;
static constexpr size_t QUANT_PLANE_OFFSET = Q2K_QUANT_PLANE_OFFSET;
static constexpr size_t SCALE0_PLANE_OFFSET = Q2K_SCALE0_PLANE_OFFSET;
static constexpr size_t SCALE1_PLANE_OFFSET = Q2K_SCALE1_PLANE_OFFSET;
static constexpr size_t D_PLANE_OFFSET = Q2K_D_PLANE_OFFSET;
static constexpr size_t DMIN_PLANE_OFFSET = Q2K_DMIN_PLANE_OFFSET;
static constexpr size_t TILE_SIZE = HTP_MM_WEIGHT_TILE_SIZE_Q2_K;
static constexpr size_t ALIGNED_TILE_SIZE = HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q2_K;

static_assert(SCALE0_PLANE_OFFSET == QUANT_PLANE_OFFSET + 8 * TILE_ROWS, "Q2_K quant plane");
static_assert(SCALE1_PLANE_OFFSET == SCALE0_PLANE_OFFSET + TILE_ROWS, "Q2_K first scale plane");
static_assert(D_PLANE_OFFSET == SCALE1_PLANE_OFFSET + TILE_ROWS, "Q2_K second scale plane");
static_assert(DMIN_PLANE_OFFSET == D_PLANE_OFFSET + sizeof(ggml_half) * TILE_ROWS, "Q2_K d plane");
static_assert(TILE_SIZE == DMIN_PLANE_OFFSET + sizeof(ggml_half) * TILE_ROWS, "Q2_K storage size");
static_assert(ALIGNED_TILE_SIZE == (TILE_SIZE + 127) / 128 * 128, "Q2_K staging alignment");

inline bool valid_2d_shape(int64_t ne0, int64_t ne1) {
    return ne0 > 0 && ne0 % QK_K == 0 && ne1 >= 0 && ne1 <= INT64_MAX - 31;
}

inline int64_t padded_rows(int64_t ne1) {
    return (ne1 + 31) / TILE_ROWS * TILE_ROWS;
}

inline size_t original_size_2d(int64_t ne0, int64_t ne1) {
    if (!valid_2d_shape(ne0, ne1) || (uint64_t) (ne0 / QK_K) > SIZE_MAX / sizeof(block_q2_K)) {
        return 0;
    }
    const size_t row_size = (size_t) (ne0 / QK_K) * sizeof(block_q2_K);
    return (uint64_t) ne1 <= SIZE_MAX / row_size ? (size_t) ne1 * row_size : 0;
}

inline size_t repacked_size_2d(int64_t ne0, int64_t ne1) {
    if (!valid_2d_shape(ne0, ne1) || (uint64_t) (ne0 / TILE_COLS) > SIZE_MAX / TILE_SIZE) {
        return 0;
    }
    const size_t tile_row_size = (size_t) (ne0 / TILE_COLS) * TILE_SIZE;
    const size_t row_tiles = (size_t) (padded_rows(ne1) / TILE_ROWS);
    return row_tiles <= SIZE_MAX / tile_row_size ? row_tiles * tile_row_size : 0;
}

inline bool repack_2d(const block_q2_K * src, size_t src_size, int64_t ne0, int64_t ne1, uint8_t * dst, size_t dst_size) {
    if (!valid_2d_shape(ne0, ne1)) {
        return false;
    }
    const size_t raw = original_size_2d(ne0, ne1);
    const size_t packed = repacked_size_2d(ne0, ne1);
    if (ne1 == 0) {
        return true;
    }
    if (!raw || !packed || !src || !dst || src_size < raw || dst_size < packed) {
        return false;
    }
    std::memset(dst, 0, packed);
    const int64_t n_k_tiles = ne0 / TILE_COLS;
    for (int64_t r = 0; r < ne1; ++r) {
        for (int64_t kt = 0; kt < n_k_tiles; ++kt) {
            const block_q2_K & b = src[r * (ne0 / QK_K) + kt / GROUPS_PER_SUPERBLOCK];
            const int g = (int) (kt % GROUPS_PER_SUPERBLOCK);
            const size_t row = (size_t) (r % TILE_ROWS);
            uint8_t * tile = dst + ((size_t) (r / TILE_ROWS) * (size_t) n_k_tiles + (size_t) kt) * TILE_SIZE;
            const uint8_t * qbase = b.qs + (g / 4) * 32;
            const int shift = 2 * (g % 4);
            for (int p = 0; p < 8; ++p) {
                uint8_t packed = 0;
                for (int j = 0; j < 4; ++j) {
                    packed |= ((qbase[4 * p + j] >> shift) & 3) << (2 * j);
                }
                tile[QUANT_PLANE_OFFSET + p * TILE_ROWS + row] = packed;
            }
            tile[SCALE0_PLANE_OFFSET + row] = b.scales[2 * g];
            tile[SCALE1_PLANE_OFFSET + row] = b.scales[2 * g + 1];
            std::memcpy(tile + D_PLANE_OFFSET + 2 * row, &b.d, sizeof(b.d));
            std::memcpy(tile + DMIN_PLANE_OFFSET + 2 * row, &b.dmin, sizeof(b.dmin));
        }
    }
    return true;
}

inline bool unpack_rows_2d(const uint8_t * src, size_t src_size, int64_t ne0, int64_t ne1, int64_t first_row, int64_t nrows, block_q2_K * dst, size_t dst_size) {
    if (!valid_2d_shape(ne0, ne1) || first_row < 0 || nrows < 0 || first_row > ne1 || nrows > ne1 - first_row) {
        return false;
    }
    const size_t packed = repacked_size_2d(ne0, ne1);
    const size_t raw = original_size_2d(ne0, nrows);
    if (nrows == 0) {
        return true;
    }
    if (!packed || !raw || !src || !dst || src_size < packed || dst_size < raw) {
        return false;
    }
    std::memset(dst, 0, raw);
    const int64_t n_k_tiles = ne0 / TILE_COLS;
    for (int64_t r = first_row; r < first_row + nrows; ++r) {
        for (int64_t kt = 0; kt < n_k_tiles; ++kt) {
            block_q2_K & b = dst[(r - first_row) * (ne0 / QK_K) + kt / GROUPS_PER_SUPERBLOCK];
            const int g = (int) (kt % GROUPS_PER_SUPERBLOCK);
            const size_t row = (size_t) (r % TILE_ROWS);
            const uint8_t * tile = src + ((size_t) (r / TILE_ROWS) * (size_t) n_k_tiles + (size_t) kt) * TILE_SIZE;
            const uint8_t * d = tile + D_PLANE_OFFSET + 2 * row;
            const uint8_t * dmin = tile + DMIN_PLANE_OFFSET + 2 * row;
            if (g == 0) {
                std::memcpy(&b.d, d, sizeof(b.d));
                std::memcpy(&b.dmin, dmin, sizeof(b.dmin));
            } else if (std::memcmp(&b.d, d, sizeof(b.d)) || std::memcmp(&b.dmin, dmin, sizeof(b.dmin))) {
                return false;
            }
            uint8_t * qbase = b.qs + (g / 4) * 32;
            const int shift = 2 * (g % 4);
            for (int p = 0; p < 8; ++p) {
                const uint8_t packed = tile[QUANT_PLANE_OFFSET + p * TILE_ROWS + row];
                for (int j = 0; j < 4; ++j) {
                    qbase[4 * p + j] |= ((packed >> (2 * j)) & 3) << shift;
                }
            }
            b.scales[2 * g] = tile[SCALE0_PLANE_OFFSET + row];
            b.scales[2 * g + 1] = tile[SCALE1_PLANE_OFFSET + row];
        }
    }
    return true;
}

inline bool unpack_2d(const uint8_t * src, size_t src_size, int64_t ne0, int64_t ne1, block_q2_K * dst, size_t dst_size) {
    return unpack_rows_2d(src, src_size, ne0, ne1, 0, ne1, dst, dst_size);
}

inline void semantic_dequantize_row(const uint8_t * src, float * dst, int64_t k, int64_t row) {
#ifdef __clang__
#pragma clang fp reassociate(off) contract(off)
#endif
    for (int64_t kt = 0; kt < k / TILE_COLS; ++kt) {
        const uint8_t * tile = src + ((size_t) (row / TILE_ROWS) * (size_t) (k / TILE_COLS) + (size_t) kt) * TILE_SIZE;
        const size_t r = (size_t) (row % TILE_ROWS);
        ggml_half d, dmin;
        std::memcpy(&d, tile + D_PLANE_OFFSET + 2 * r, sizeof(d));
        std::memcpy(&dmin, tile + DMIN_PLANE_OFFSET + 2 * r, sizeof(dmin));
        for (int j = 0; j < 32; ++j) {
            const uint8_t sc = tile[(j < 16 ? SCALE0_PLANE_OFFSET : SCALE1_PLANE_OFFSET) + r];
            const float D = ggml_fp16_to_fp32(d) * (sc & 15);
            const float M = ggml_fp16_to_fp32(dmin) * (sc >> 4);
            const int q = (tile[QUANT_PLANE_OFFSET + (j / 4) * TILE_ROWS + r] >> (2 * (j & 3))) & 3;
            dst[32 * kt + j] = D * q - M;
        }
    }
}

inline void semantic_dequantize(const block_q2_K * src, float * dst, int64_t k) {
#ifdef __clang__
#pragma clang fp reassociate(off) contract(off)
#endif
    for (int64_t sb = 0; sb < k / QK_K; ++sb) {
        const block_q2_K & b = src[sb];
        const float d = ggml_fp16_to_fp32(b.d);
        const float dmin = ggml_fp16_to_fp32(b.dmin);
        for (int g = 0; g < GROUPS_PER_SUPERBLOCK; ++g) {
            const uint8_t * qbase = b.qs + (g / 4) * 32;
            const int shift = 2 * (g % 4);
            for (int j = 0; j < 32; ++j) {
                const uint8_t sc = b.scales[2 * g + j / 16];
                const float D = d * (sc & 15);
                const float M = dmin * (sc >> 4);
                const int q = (qbase[j] >> shift) & 3;
                dst[sb * QK_K + 32 * g + j] = D * q - M;
            }
        }
    }
}

} // namespace ggml_hexagon_q2k
