#pragma once

#include "ggml-quants.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace ggml_hexagon_iq3s {

static_assert(QK_K == 256, "IQ3_S requires 256 weights per superblock");
static_assert(sizeof(block_iq3_s) == 110, "IQ3_S source block layout changed");

// 32 rows x 32 weights: raw equivalent 440 B (3.4375 bpw), repacked 512 B (4 bpw), overhead 16.363636% before row padding.
// d is copied into all eight K tiles of a superblock; unpack checks every copy.
static constexpr int64_t TILE_ROWS = 32;
static constexpr int64_t TILE_COLS = 32;
static constexpr int64_t GROUPS_PER_SUPERBLOCK = 8;
static constexpr size_t INDEX_PLANE_OFFSET = 0;
static constexpr size_t SIGN_PLANE_OFFSET = 256;
static constexpr size_t QH_PLANE_OFFSET = 384;
static constexpr size_t SCALE_PLANE_OFFSET = 416;
static constexpr size_t D_PLANE_OFFSET = 448;
static constexpr size_t TILE_SIZE = 512;

static_assert(SIGN_PLANE_OFFSET == INDEX_PLANE_OFFSET + 8 * TILE_ROWS, "IQ3_S index plane");
static_assert(QH_PLANE_OFFSET == SIGN_PLANE_OFFSET + 4 * TILE_ROWS, "IQ3_S sign plane");
static_assert(SCALE_PLANE_OFFSET == QH_PLANE_OFFSET + TILE_ROWS, "IQ3_S high index plane");
static_assert(D_PLANE_OFFSET == SCALE_PLANE_OFFSET + TILE_ROWS, "IQ3_S scale plane");
static_assert(TILE_SIZE == D_PLANE_OFFSET + sizeof(ggml_half) * TILE_ROWS && TILE_SIZE % 128 == 0, "IQ3_S tile size and alignment");

inline bool valid_2d_shape(int64_t ne0, int64_t ne1) {
    return ne0 > 0 && ne0 % QK_K == 0 && ne1 >= 0 && ne1 <= INT64_MAX - 31;
}

inline int64_t padded_rows(int64_t ne1) {
    return (ne1 + 31) / TILE_ROWS * TILE_ROWS;
}

inline size_t original_size_2d(int64_t ne0, int64_t ne1) {
    if (!valid_2d_shape(ne0, ne1) || (uint64_t) (ne0 / QK_K) > SIZE_MAX / sizeof(block_iq3_s)) {
        return 0;
    }
    const size_t row_size = (size_t) (ne0 / QK_K) * sizeof(block_iq3_s);
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

inline bool repack_2d(const block_iq3_s * src, size_t src_size, int64_t ne0, int64_t ne1, uint8_t * dst, size_t dst_size) {
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
            const block_iq3_s & b = src[r * (ne0 / QK_K) + kt / GROUPS_PER_SUPERBLOCK];
            const int g = (int) (kt % GROUPS_PER_SUPERBLOCK);
            const size_t row = (size_t) (r % TILE_ROWS);
            uint8_t * tile = dst + ((size_t) (r / TILE_ROWS) * (size_t) n_k_tiles + (size_t) kt) * TILE_SIZE;
            for (int i = 0; i < 8; ++i) {
                tile[INDEX_PLANE_OFFSET + i * TILE_ROWS + row] = b.qs[8 * g + i];
            }
            for (int l = 0; l < 4; ++l) {
                tile[SIGN_PLANE_OFFSET + l * TILE_ROWS + row] = b.signs[4 * g + l];
            }
            tile[QH_PLANE_OFFSET + row] = b.qh[g];
            tile[SCALE_PLANE_OFFSET + row] = (b.scales[g / 2] >> (4 * (g & 1))) & 15;
            std::memcpy(tile + D_PLANE_OFFSET + 2 * row, &b.d, sizeof(b.d));
        }
    }
    return true;
}

inline bool unpack_rows_2d(const uint8_t * src, size_t src_size, int64_t ne0, int64_t ne1, int64_t first_row, int64_t nrows, block_iq3_s * dst, size_t dst_size) {
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
            block_iq3_s & b = dst[(r - first_row) * (ne0 / QK_K) + kt / GROUPS_PER_SUPERBLOCK];
            const int g = (int) (kt % GROUPS_PER_SUPERBLOCK);
            const size_t row = (size_t) (r % TILE_ROWS);
            const uint8_t * tile = src + ((size_t) (r / TILE_ROWS) * (size_t) n_k_tiles + (size_t) kt) * TILE_SIZE;
            const uint8_t * d = tile + D_PLANE_OFFSET + 2 * row;
            if (g == 0) {
                std::memcpy(&b.d, d, sizeof(b.d));
            } else if (std::memcmp(&b.d, d, sizeof(b.d))) {
                return false;
            }
            for (int i = 0; i < 8; ++i) {
                b.qs[8 * g + i] = tile[INDEX_PLANE_OFFSET + i * TILE_ROWS + row];
            }
            for (int l = 0; l < 4; ++l) {
                b.signs[4 * g + l] = tile[SIGN_PLANE_OFFSET + l * TILE_ROWS + row];
            }
            b.qh[g] = tile[QH_PLANE_OFFSET + row];
            const uint8_t nibble = tile[SCALE_PLANE_OFFSET + row];
            if (nibble > 15) {
                return false;
            }
            b.scales[g / 2] |= nibble << (4 * (g & 1));
        }
    }
    return true;
}

inline bool unpack_2d(const uint8_t * src, size_t src_size, int64_t ne0, int64_t ne1, block_iq3_s * dst, size_t dst_size) {
    return unpack_rows_2d(src, src_size, ne0, ne1, 0, ne1, dst, dst_size);
}

inline void semantic_dequantize_row(const uint8_t * src, float * dst, int64_t k, int64_t row, const uint32_t * grid) {
    for (int64_t kt = 0; kt < k / TILE_COLS; ++kt) {
        const uint8_t * tile = src + ((size_t) (row / TILE_ROWS) * (size_t) (k / TILE_COLS) + (size_t) kt) * TILE_SIZE;
        const size_t r = (size_t) (row % TILE_ROWS);
        ggml_half d;
        std::memcpy(&d, tile + D_PLANE_OFFSET + 2 * r, sizeof(d));
        const float db = ggml_fp16_to_fp32(d) * (1 + 2 * tile[SCALE_PLANE_OFFSET + r]);
        const uint8_t qh = tile[QH_PLANE_OFFSET + r];
        for (int l = 0; l < 4; ++l) {
            const uint8_t signs = tile[SIGN_PLANE_OFFSET + l * TILE_ROWS + r];
            for (int j = 0; j < 8; ++j) {
                const int i = 2 * l + j / 4;
                const unsigned idx = tile[INDEX_PLANE_OFFSET + i * TILE_ROWS + r] | (((qh >> i) & 1u) << 8);
                const uint8_t magnitude = (uint8_t) (grid[idx] >> (8 * (j % 4)));
                dst[32 * kt + 8 * l + j] = db * magnitude * ((signs & (1u << j)) ? -1.0f : 1.0f);
            }
        }
    }
}

inline void semantic_dequantize(const block_iq3_s * src, float * dst, int64_t k, const uint32_t * grid) {
    for (int64_t sb = 0; sb < k / QK_K; ++sb) {
        const block_iq3_s & b = src[sb];
        const float d = ggml_fp16_to_fp32(b.d);
        for (int g = 0; g < GROUPS_PER_SUPERBLOCK; ++g) {
            const float db = d * (1 + 2 * ((b.scales[g / 2] >> (4 * (g & 1))) & 15));
            for (int l = 0; l < 4; ++l) {
                const uint8_t signs = b.signs[4 * g + l];
                for (int j = 0; j < 8; ++j) {
                    const int i = 2 * l + j / 4;
                    const unsigned idx = b.qs[8 * g + i] | (((b.qh[g] >> i) & 1u) << 8);
                    const uint8_t magnitude = (uint8_t) (grid[idx] >> (8 * (j % 4)));
                    dst[sb * QK_K + 32 * g + 8 * l + j] = db * magnitude * ((signs & (1u << j)) ? -1.0f : 1.0f);
                }
            }
        }
    }
}

} // namespace ggml_hexagon_iq3s
