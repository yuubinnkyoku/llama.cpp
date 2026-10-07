#pragma once

#include "ggml-quants.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace ggml_hexagon_iq3s {

static_assert(QK_K == 256, "IQ3_S requires 256 weights per superblock");
static_assert(sizeof(block_iq3_s) == 110, "IQ3_S source block layout changed");

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
static_assert(TILE_SIZE == D_PLANE_OFFSET + sizeof(ggml_half) * TILE_ROWS && TILE_SIZE % 128 == 0,
              "IQ3_S tile size and alignment");

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
    if (blocks_per_row > limit / sizeof(block_iq3_s)) {
        return 0;
    }
    const size_t row_size = (size_t) blocks_per_row * sizeof(block_iq3_s);
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

inline bool repack_2d(const block_iq3_s * src, size_t src_size, int64_t ne0, int64_t ne1,
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
    const size_t blocks_per_row = (size_t) (ne0 / QK_K);
    for (int64_t r = 0; r < ne1; ++r) {
        for (int64_t kt = 0; kt < k_tiles; ++kt) {
            const size_t block_index = (size_t) r * blocks_per_row + (size_t) (kt / GROUPS_PER_SUPERBLOCK);
            const block_iq3_s & b = src[block_index];
            const int group = (int) (kt % GROUPS_PER_SUPERBLOCK);
            const size_t row = (size_t) (r % TILE_ROWS);
            uint8_t * tile = dst + ((size_t) (r / TILE_ROWS) * (size_t) k_tiles + (size_t) kt) * TILE_SIZE;
            for (int i = 0; i < 8; ++i) {
                tile[INDEX_PLANE_OFFSET + (size_t) i * TILE_ROWS + row] = b.qs[8 * group + i];
            }
            for (int l = 0; l < 4; ++l) {
                tile[SIGN_PLANE_OFFSET + (size_t) l * TILE_ROWS + row] = b.signs[4 * group + l];
            }
            tile[QH_PLANE_OFFSET + row] = b.qh[group];
            tile[SCALE_PLANE_OFFSET + row] = (b.scales[group / 2] >> (4 * (group & 1))) & 15;
            std::memcpy(tile + D_PLANE_OFFSET + 2 * row, &b.d, sizeof(b.d));
        }
    }
    return true;
}

inline bool reconstruct_block_2d(const uint8_t * src, size_t src_size, int64_t ne0, int64_t ne1,
                                 int64_t row, int64_t sb, block_iq3_s & b) {
    const size_t packed = repacked_size_2d(ne0, ne1);
    if (!valid_2d_shape(ne0, ne1) || !src || !packed || src_size < packed || row < 0 || row >= ne1 ||
        sb < 0 || sb >= ne0 / QK_K) {
        return false;
    }
    std::memset(&b, 0, sizeof(b));
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
        for (int l = 0; l < 4; ++l) {
            b.signs[4 * group + l] = tile[SIGN_PLANE_OFFSET + (size_t) l * TILE_ROWS + tile_row];
        }
        b.qh[group] = tile[QH_PLANE_OFFSET + tile_row];
        const uint8_t scale = tile[SCALE_PLANE_OFFSET + tile_row];
        if (scale > 15) {
            return false;
        }
        b.scales[group / 2] |= scale << (4 * (group & 1));
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
    const size_t blocks_per_row = (size_t) (ne0 / QK_K);
    while (done < size) {
        const size_t pos = offset + done;
        const size_t block_index = pos / sizeof(block_iq3_s);
        const size_t block_offset = pos % sizeof(block_iq3_s);
        const size_t chunk = size - done < sizeof(block_iq3_s) - block_offset
            ? size - done : sizeof(block_iq3_s) - block_offset;
        block_iq3_s block;
        if (!reconstruct_block_2d(src, src_size, ne0, ne1,
                                  (int64_t) (block_index / blocks_per_row),
                                  (int64_t) (block_index % blocks_per_row), block)) {
            return false;
        }
        std::memcpy((uint8_t *) dst + done, (const uint8_t *) &block + block_offset, chunk);
        done += chunk;
    }
    return true;
}

inline bool unpack_rows_2d(const uint8_t * src, size_t src_size, int64_t ne0, int64_t ne1,
                           int64_t first_row, int64_t nrows, block_iq3_s * dst, size_t dst_size) {
    if (!valid_2d_shape(ne0, ne1) || first_row < 0 || nrows < 0 || first_row > ne1 || nrows > ne1 - first_row) {
        return false;
    }
    const size_t raw = original_size_2d(ne0, nrows);
    if (nrows == 0) {
        return true;
    }
    if (!raw || !src || !dst || dst_size < raw || src_size < repacked_size_2d(ne0, ne1)) {
        return false;
    }
    for (int64_t r = 0; r < nrows; ++r) {
        const size_t row_offset = (size_t) (first_row + r) * (size_t) (ne0 / QK_K) * sizeof(block_iq3_s);
        if (!readback_2d(src, src_size, ne0, ne1, row_offset,
                         dst + (size_t) r * (size_t) (ne0 / QK_K),
                         (size_t) (ne0 / QK_K) * sizeof(block_iq3_s))) {
            return false;
        }
    }
    return true;
}

inline bool unpack_2d(const uint8_t * src, size_t src_size, int64_t ne0, int64_t ne1,
                      block_iq3_s * dst, size_t dst_size) {
    return unpack_rows_2d(src, src_size, ne0, ne1, 0, ne1, dst, dst_size);
}

inline void semantic_dequantize_row(const uint8_t * src, float * dst, int64_t k, int64_t row,
                                    const uint32_t * grid) {
    if (!src || !dst || !grid || k <= 0 || k % TILE_COLS != 0 || row < 0) {
        return;
    }
    for (int64_t kt = 0; kt < k / TILE_COLS; ++kt) {
        const uint8_t * tile = src + ((size_t) (row / TILE_ROWS) * (size_t) (k / TILE_COLS) + (size_t) kt) * TILE_SIZE;
        const size_t r = (size_t) (row % TILE_ROWS);
        ggml_half d;
        std::memcpy(&d, tile + D_PLANE_OFFSET + sizeof(d) * r, sizeof(d));
        const float db = ggml_fp16_to_fp32(d) * (1 + 2 * tile[SCALE_PLANE_OFFSET + r]);
        const uint8_t qh = tile[QH_PLANE_OFFSET + r];
        for (int l = 0; l < 4; ++l) {
            const uint8_t signs = tile[SIGN_PLANE_OFFSET + (size_t) l * TILE_ROWS + r];
            for (int j = 0; j < 8; ++j) {
                const int i = 2 * l + j / 4;
                const unsigned index = tile[INDEX_PLANE_OFFSET + (size_t) i * TILE_ROWS + r] | (((qh >> i) & 1u) << 8);
                const uint8_t magnitude = (uint8_t) (grid[index] >> (8 * (j % 4)));
                dst[32 * kt + 8 * l + j] = db * magnitude * ((signs & (1u << j)) ? -1.0f : 1.0f);
            }
        }
    }
}

inline void semantic_dequantize(const block_iq3_s * src, float * dst, int64_t k, const uint32_t * grid) {
    for (int64_t sb = 0; sb < k / QK_K; ++sb) {
        const block_iq3_s & block = src[sb];
        const float d = ggml_fp16_to_fp32(block.d);
        for (int group = 0; group < GROUPS_PER_SUPERBLOCK; ++group) {
            const float scale = d * (1 + 2 * ((block.scales[group / 2] >> (4 * (group & 1))) & 15));
            for (int l = 0; l < 4; ++l) {
                const uint8_t signs = block.signs[4 * group + l];
                for (int j = 0; j < 8; ++j) {
                    const int i = 2 * l + j / 4;
                    const unsigned index = block.qs[8 * group + i] | (((block.qh[group] >> i) & 1u) << 8);
                    const uint8_t magnitude = (uint8_t) (grid[index] >> (8 * (j % 4)));
                    dst[sb * QK_K + 32 * group + 8 * l + j] = scale * magnitude * ((signs & (1u << j)) ? -1.0f : 1.0f);
                }
            }
        }
    }
}

} // namespace ggml_hexagon_iq3s
