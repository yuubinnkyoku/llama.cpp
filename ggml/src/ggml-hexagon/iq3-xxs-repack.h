#pragma once

#include "ggml-quants.h"
#include "iq3-compact-layout.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace ggml_hexagon_iq3xxs {

static_assert(QK_K == 256, "IQ3_XXS requires 256 weights per superblock");
static_assert(sizeof(block_iq3_xxs) == 98, "IQ3_XXS source block layout changed");
static_assert(IQ3XXS_COMPACT_PAIR_SIZE == 6272 && IQ3XXS_COMPACT_PAIR_SIZE % IQ3_COMPACT_ALIGNMENT == 0,
              "IQ3_XXS pair supertile layout changed");
static_assert(IQ3XXS_COMPACT_TAIL_SIZE == 3200 && IQ3XXS_COMPACT_TAIL_SIZE % IQ3_COMPACT_ALIGNMENT == 0,
              "IQ3_XXS tail supertile layout changed");

static constexpr int64_t TILE_ROWS = IQ3_COMPACT_TILE_ROWS;
static constexpr int64_t TILE_COLS = IQ3_COMPACT_TILE_COLS;
static constexpr int64_t GROUPS_PER_SUPERBLOCK = IQ3_COMPACT_GROUPS_PER_SUPERBLOCK;
static constexpr size_t INDEX_PLANE_OFFSET = 0;
static constexpr size_t AUX_PLANE_OFFSET = 256;
static constexpr size_t DATA_TILE_SIZE = IQ3_COMPACT_DATA_TILE_SIZE;
static constexpr size_t TILE_SIZE = DATA_TILE_SIZE;
static constexpr size_t PAIR_SIZE = IQ3XXS_COMPACT_PAIR_SIZE;
static constexpr size_t TAIL_SIZE = IQ3XXS_COMPACT_TAIL_SIZE;

static_assert(AUX_PLANE_OFFSET == 8 * TILE_ROWS, "IQ3_XXS index plane");

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

inline size_t row_tile_size(int64_t ne0) {
    size_t bytes = 0;
    return iq3_compact_xxs_row_tile_size(ne0, &bytes) ? bytes : 0;
}

inline size_t repacked_size_2d(int64_t ne0, int64_t ne1) {
    if (!valid_2d_shape(ne0, ne1)) {
        return 0;
    }
    size_t bytes = 0;
    return iq3_compact_xxs_total_size(ne0, ne1, 1, 1, &bytes) ? bytes : 0;
}

inline bool tile_offsets(int64_t ne0, uint64_t kt, iq3_compact_offsets & offsets) {
    return iq3_compact_xxs_offsets(ne0, kt, &offsets) != 0;
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
    const uint64_t k_tiles = (uint64_t) ne0 / TILE_COLS;
    const uint64_t blocks_per_row = (uint64_t) ne0 / QK_K;
    const size_t row_tile_bytes = row_tile_size(ne0);
    for (int64_t r = 0; r < ne1; ++r) {
        const size_t tile_row = (size_t) r % TILE_ROWS;
        uint8_t * row_tile_base = dst + (size_t) (r / TILE_ROWS) * row_tile_bytes;
        const block_iq3_xxs * row_src = src + (size_t) r * (size_t) blocks_per_row;
        for (uint64_t kt = 0; kt < k_tiles; ++kt) {
            iq3_compact_offsets offsets;
            if (!tile_offsets(ne0, kt, offsets)) {
                return false;
            }
            const block_iq3_xxs & b = row_src[kt / GROUPS_PER_SUPERBLOCK];
            const int group = (int) offsets.group;
            uint8_t * tile = row_tile_base + offsets.data_tile;
            for (int i = 0; i < 8; ++i) {
                tile[INDEX_PLANE_OFFSET + (size_t) i * TILE_ROWS + tile_row] = b.qs[8 * group + i];
            }
            std::memcpy(tile + AUX_PLANE_OFFSET + 4 * tile_row, b.qs + 64 + 4 * group, 4);
            if (group == 0) {
                std::memcpy(row_tile_base + offsets.d_plane + 2 * tile_row, &b.d, sizeof(b.d));
            }
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
    const size_t row_tile_bytes = row_tile_size(ne0);
    const size_t tile_row = (size_t) row % TILE_ROWS;
    const uint8_t * row_tile_base = src + (size_t) (row / TILE_ROWS) * row_tile_bytes;
    for (int group = 0; group < GROUPS_PER_SUPERBLOCK; ++group) {
        const uint64_t kt = (uint64_t) sb * GROUPS_PER_SUPERBLOCK + (uint64_t) group;
        iq3_compact_offsets offsets;
        if (!tile_offsets(ne0, kt, offsets)) {
            return false;
        }
        const uint8_t * tile = row_tile_base + offsets.data_tile;
        for (int i = 0; i < 8; ++i) {
            b.qs[8 * group + i] = tile[INDEX_PLANE_OFFSET + (size_t) i * TILE_ROWS + tile_row];
        }
        std::memcpy(b.qs + 64 + 4 * group, tile + AUX_PLANE_OFFSET + 4 * tile_row, 4);
    }
    iq3_compact_offsets first_offsets;
    if (!tile_offsets(ne0, (uint64_t) sb * GROUPS_PER_SUPERBLOCK, first_offsets)) {
        return false;
    }
    std::memcpy(&b.d, row_tile_base + first_offsets.d_plane + 2 * tile_row, sizeof(b.d));
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
    const size_t row_tile_bytes = row_tile_size(k);
    const uint8_t * row_tile_base = src + (size_t) (row / TILE_ROWS) * row_tile_bytes;
    for (uint64_t kt = 0; kt < (uint64_t) k / TILE_COLS; ++kt) {
        iq3_compact_offsets offsets;
        if (!tile_offsets(k, kt, offsets)) {
            return;
        }
        const uint8_t * tile = row_tile_base + offsets.data_tile;
        const size_t r = (size_t) row % TILE_ROWS;
        ggml_half d;
        std::memcpy(&d, row_tile_base + offsets.d_plane + 2 * r, sizeof(d));
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
