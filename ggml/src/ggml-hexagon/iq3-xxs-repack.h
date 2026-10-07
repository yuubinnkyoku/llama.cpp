#pragma once

#include "ggml-quants.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace ggml_hexagon_iq3xxs {

static_assert(QK_K == 256, "IQ3_XXS requires 256 weights per superblock");
static_assert(sizeof(block_iq3_xxs) == 98, "IQ3_XXS source block layout changed");

static constexpr int64_t TILE_ROWS = 32;
static constexpr int64_t TILE_COLS = 32;
static constexpr int64_t GROUPS_PER_SUPERBLOCK = 8;
static constexpr size_t INDEX_PLANE_OFFSET = 0;
static constexpr size_t AUX_PLANE_OFFSET = 256;
static constexpr size_t D_PLANE_OFFSET = 384;
static constexpr size_t PADDING_OFFSET = 448;
static constexpr size_t TILE_SIZE = 512;

static_assert(TILE_SIZE % 128 == 0, "IQ3_XXS tile size and alignment");
static_assert(AUX_PLANE_OFFSET == 8 * TILE_ROWS, "IQ3_XXS index plane");
static_assert(D_PLANE_OFFSET == AUX_PLANE_OFFSET + 4 * TILE_ROWS, "IQ3_XXS aux plane");
static_assert(PADDING_OFFSET == D_PLANE_OFFSET + sizeof(ggml_half) * TILE_ROWS && PADDING_OFFSET + 64 == TILE_SIZE,
              "IQ3_XXS d and padding planes");

inline bool valid_2d_shape(int64_t ne0, int64_t ne1) {
    return ne0 > 0 && ne0 % QK_K == 0 && ne1 >= 0 && ne1 <= INT64_MAX - (TILE_ROWS - 1);
}

inline int64_t padded_rows(int64_t ne1) {
    return (ne1 + TILE_ROWS - 1) / TILE_ROWS * TILE_ROWS;
}

inline size_t original_size_2d(int64_t ne0, int64_t ne1) {
    if (!valid_2d_shape(ne0, ne1)) {
        return 0;
    }
    const uint64_t blocks_per_row = (uint64_t) (ne0 / QK_K);
    const size_t limit = std::numeric_limits<size_t>::max();
    if (blocks_per_row > limit / sizeof(block_iq3_xxs)) {
        return 0;
    }
    const size_t row_size = (size_t) blocks_per_row * sizeof(block_iq3_xxs);
    if ((uint64_t) ne1 > limit / row_size) {
        return 0;
    }
    return (size_t) ne1 * row_size;
}

inline size_t repacked_size_2d(int64_t ne0, int64_t ne1) {
    if (!valid_2d_shape(ne0, ne1)) {
        return 0;
    }
    const uint64_t k_tiles = (uint64_t) (ne0 / TILE_COLS);
    const size_t limit = std::numeric_limits<size_t>::max();
    if (k_tiles > limit / TILE_SIZE) {
        return 0;
    }
    const size_t tiled_row_size = (size_t) k_tiles * TILE_SIZE;
    const uint64_t row_tiles = (uint64_t) (padded_rows(ne1) / TILE_ROWS);
    if (row_tiles > limit / tiled_row_size) {
        return 0;
    }
    return (size_t) row_tiles * tiled_row_size;
}

inline bool repack_2d(const block_iq3_xxs * src, size_t src_size, int64_t ne0, int64_t ne1,
                      uint8_t * dst, size_t dst_size) {
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
    const int64_t k_tiles = ne0 / TILE_COLS;
    const int64_t blocks_per_row = ne0 / QK_K;
    for (int64_t r = 0; r < ne1; ++r) {
        for (int64_t kt = 0; kt < k_tiles; ++kt) {
            const block_iq3_xxs & b = src[r * blocks_per_row + kt / GROUPS_PER_SUPERBLOCK];
            const int group = (int) (kt % GROUPS_PER_SUPERBLOCK);
            const size_t row = (size_t) (r % TILE_ROWS);
            uint8_t * tile = dst + ((size_t) (r / TILE_ROWS) * (size_t) k_tiles + (size_t) kt) * TILE_SIZE;
            for (int i = 0; i < 8; ++i) {
                tile[INDEX_PLANE_OFFSET + (size_t) i * TILE_ROWS + row] = b.qs[8 * group + i];
            }
            std::memcpy(tile + AUX_PLANE_OFFSET + 4 * row, b.qs + 64 + 4 * group, 4);
            std::memcpy(tile + D_PLANE_OFFSET + 2 * row, &b.d, sizeof(b.d));
        }
    }
    return true;
}

inline bool reconstruct_block_2d(const uint8_t * src, size_t src_size, int64_t ne0, int64_t ne1,
                                 int64_t row, int64_t sb, block_iq3_xxs & b) {
    const size_t packed = repacked_size_2d(ne0, ne1);
    if (!valid_2d_shape(ne0, ne1) || !src || src_size < packed || row < 0 || row >= ne1 ||
        sb < 0 || sb >= ne0 / QK_K) {
        return false;
    }
    const int64_t k_tiles = ne0 / TILE_COLS;
    const size_t tile_row = (size_t) (row % TILE_ROWS);
    for (int group = 0; group < GROUPS_PER_SUPERBLOCK; ++group) {
        const int64_t kt = sb * GROUPS_PER_SUPERBLOCK + group;
        const uint8_t * tile = src + ((size_t) (row / TILE_ROWS) * (size_t) k_tiles + (size_t) kt) * TILE_SIZE;
        const uint8_t * d = tile + D_PLANE_OFFSET + tile_row * sizeof(ggml_half);
        if (group == 0) {
            std::memcpy(&b.d, d, sizeof(b.d));
        } else if (std::memcmp(&b.d, d, sizeof(b.d)) != 0) {
            return false;
        }
        for (int i = 0; i < 8; ++i) {
            b.qs[8 * group + i] = tile[INDEX_PLANE_OFFSET + (size_t) i * TILE_ROWS + tile_row];
        }
        std::memcpy(b.qs + 64 + 4 * group, tile + AUX_PLANE_OFFSET + 4 * tile_row, 4);
    }
    return true;
}

inline bool readback_2d(const uint8_t * src, size_t src_size, int64_t ne0, int64_t ne1,
                        size_t offset, void * dst, size_t size) {
    const size_t raw = original_size_2d(ne0, ne1);
    const size_t packed = repacked_size_2d(ne0, ne1);
    if (!valid_2d_shape(ne0, ne1) || offset > raw || size > raw - offset) {
        return false;
    }
    if (size == 0) {
        return true;
    }
    if (!src || !dst || !raw || !packed || src_size < packed) {
        return false;
    }
    size_t done = 0;
    while (done < size) {
        const size_t pos = offset + done;
        const size_t block_index = pos / sizeof(block_iq3_xxs);
        const size_t block_offset = pos % sizeof(block_iq3_xxs);
        const size_t block_tail = sizeof(block_iq3_xxs) - block_offset;
        const size_t chunk = size - done < block_tail ? size - done : block_tail;
        block_iq3_xxs b;
        if (!reconstruct_block_2d(src, src_size, ne0, ne1,
                                  (int64_t) (block_index / (size_t) (ne0 / QK_K)),
                                  (int64_t) (block_index % (size_t) (ne0 / QK_K)), b)) {
            return false;
        }
        std::memcpy((uint8_t *) dst + done, (const uint8_t *) &b + block_offset, chunk);
        done += chunk;
    }
    return true;
}

inline bool unpack_rows_2d(const uint8_t * src, size_t src_size, int64_t ne0, int64_t ne1,
                           int64_t first_row, int64_t nrows, block_iq3_xxs * dst, size_t dst_size) {
    const size_t raw = original_size_2d(ne0, nrows);
    if (!valid_2d_shape(ne0, ne1) || first_row < 0 || nrows < 0 || first_row > ne1 || nrows > ne1 - first_row) {
        return false;
    }
    if (nrows == 0) {
        return true;
    }
    if (!src || !dst || !raw || dst_size < raw || src_size < repacked_size_2d(ne0, ne1)) {
        return false;
    }
    for (int64_t r = 0; r < nrows; ++r) {
        const size_t row_offset = (size_t) (first_row + r) * (size_t) (ne0 / QK_K) * sizeof(block_iq3_xxs);
        if (!readback_2d(src, src_size, ne0, ne1, row_offset,
                         dst + r * (ne0 / QK_K), (size_t) (ne0 / QK_K) * sizeof(block_iq3_xxs))) {
            return false;
        }
    }
    return true;
}

inline bool unpack_2d(const uint8_t * src, size_t src_size, int64_t ne0, int64_t ne1,
                      block_iq3_xxs * dst, size_t dst_size) {
    return unpack_rows_2d(src, src_size, ne0, ne1, 0, ne1, dst, dst_size);
}

inline uint32_t load_aux(const uint8_t * p) {
    return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
}

inline void semantic_dequantize_row(const uint8_t * src, float * dst, int64_t k, int64_t row,
                                    const uint32_t * grid, const uint8_t * sign_table) {
    for (int64_t kt = 0; kt < k / TILE_COLS; ++kt) {
        const uint8_t * tile = src + ((size_t) (row / TILE_ROWS) * (size_t) (k / TILE_COLS) + (size_t) kt) * TILE_SIZE;
        const size_t r = (size_t) (row % TILE_ROWS);
        ggml_half d;
        std::memcpy(&d, tile + D_PLANE_OFFSET + 2 * r, sizeof(d));
        const uint32_t aux = load_aux(tile + AUX_PLANE_OFFSET + 4 * r);
        const float db = ggml_fp16_to_fp32(d) * (0.5f + (aux >> 28)) * 0.5f;
        for (int l = 0; l < 4; ++l) {
            const uint8_t signs = sign_table[(aux >> (7 * l)) & 127];
            for (int j = 0; j < 8; ++j) {
                const uint32_t entry = grid[tile[INDEX_PLANE_OFFSET + (2 * l + j / 4) * TILE_ROWS + r]];
                const uint8_t magnitude = (uint8_t) (entry >> (8 * (j % 4)));
                dst[32 * kt + 8 * l + j] = db * magnitude * ((signs & (1u << j)) ? -1.0f : 1.0f);
            }
        }
    }
}

} // namespace ggml_hexagon_iq3xxs
