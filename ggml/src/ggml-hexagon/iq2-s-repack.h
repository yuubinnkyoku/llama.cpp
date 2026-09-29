#pragma once

#include "ggml-quants.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace ggml_hexagon_iq2s {

// Phase-1 host-side layout for an IQ2_S 32x32 weight tile.
//
// Source IQ2_S is organized in 256-value super-blocks.  Each 32-value group
// uses 4 low-index bytes, 4 sign bytes, one qh byte, one scale byte, and the
// super-block-wide fp16 scale d.
//
// The HTP tiled path addresses weights in 32x32 tiles, so the initial layout
// deliberately duplicates d into every 32-column tile.  The resulting tile is
// exactly three 128-byte HVX vectors:
//
//   [0,   128)  grid-index plane: index[l][row], l=0..3, row=0..31
//   [128, 256)  sign plane:       sign [l][row], l=0..3, row=0..31
//   [256, 288)  qh[row]
//   [288, 320)  scale[row]
//   [320, 384)  d[row] as raw ggml_half bits
//
// This expands IQ2_S from 2.5625 bpw to 3.0 bpw in repacked storage, but keeps
// every 32x32 tile self-contained and 128-byte aligned.  A later optimization
// may share d across the eight tiles that belong to one 256-value super-block.

static_assert(QK_K == 256, "IQ2_S repack assumes QK_K == 256");
static_assert(sizeof(block_iq2_s) == 82, "IQ2_S source block layout changed");

static constexpr int64_t TILE_ROWS = 32;
static constexpr int64_t TILE_COLS = 32;
static constexpr int64_t GROUPS_PER_SUPERBLOCK = QK_K / TILE_COLS; // 8

static constexpr size_t INDEX_PLANE_OFFSET = 0;
static constexpr size_t SIGN_PLANE_OFFSET  = 128;
static constexpr size_t QH_PLANE_OFFSET    = 256;
static constexpr size_t SCALE_PLANE_OFFSET = 288;
static constexpr size_t D_PLANE_OFFSET     = 320;
static constexpr size_t TILE_SIZE          = 384;

static_assert(TILE_SIZE % 128 == 0, "IQ2_S HTP tile must stay HVX aligned");
static_assert(D_PLANE_OFFSET + TILE_ROWS * sizeof(ggml_half) == TILE_SIZE,
              "IQ2_S tile layout does not fill exactly three HVX vectors");

inline bool valid_2d_shape(int64_t ne0, int64_t ne1) {
    return ne0 > 0 && ne1 >= 0 && ne0 % QK_K == 0;
}

inline int64_t padded_rows(int64_t ne1) {
    return (ne1 + TILE_ROWS - 1) / TILE_ROWS * TILE_ROWS;
}

inline size_t original_size_2d(int64_t ne0, int64_t ne1) {
    if (!valid_2d_shape(ne0, ne1)) {
        return 0;
    }

    const size_t superblocks_per_row = (size_t) (ne0 / QK_K);
    return (size_t) ne1 * superblocks_per_row * sizeof(block_iq2_s);
}

inline size_t repacked_size_2d(int64_t ne0, int64_t ne1) {
    if (!valid_2d_shape(ne0, ne1)) {
        return 0;
    }

    const size_t row_tiles = (size_t) (padded_rows(ne1) / TILE_ROWS);
    const size_t k_tiles   = (size_t) (ne0 / TILE_COLS);
    return row_tiles * k_tiles * TILE_SIZE;
}

// Repack a row-major IQ2_S matrix into fixed 32x32 HTP tiles.
//
// src contains ne1 rows, each with ne0 values.  ne0 must be a multiple of 256.
// Padded rows in dst are zero-filled.  The function is byte-lossless for all
// source IQ2_S fields; only d is duplicated across the eight 32-column tiles.
inline bool repack_2d(
        const block_iq2_s * src,
        size_t src_size,
        int64_t ne0,
        int64_t ne1,
        uint8_t * dst,
        size_t dst_size) {
    if (!valid_2d_shape(ne0, ne1)) {
        return false;
    }

    const size_t expected_src = original_size_2d(ne0, ne1);
    const size_t expected_dst = repacked_size_2d(ne0, ne1);
    if (src_size < expected_src || dst_size < expected_dst) {
        return false;
    }
    if ((expected_src != 0 && src == nullptr) || (expected_dst != 0 && dst == nullptr)) {
        return false;
    }

    if (expected_dst != 0) {
        std::memset(dst, 0, expected_dst);
    }

    const int64_t superblocks_per_row = ne0 / QK_K;
    const int64_t n_k_tiles = ne0 / TILE_COLS;

    for (int64_t r = 0; r < ne1; ++r) {
        const int64_t ct  = r / TILE_ROWS;
        const int64_t row = r % TILE_ROWS;
        const block_iq2_s * src_row = src + r * superblocks_per_row;

        for (int64_t kt = 0; kt < n_k_tiles; ++kt) {
            const int64_t group = kt % GROUPS_PER_SUPERBLOCK;
            const block_iq2_s * b = src_row + kt / GROUPS_PER_SUPERBLOCK;
            uint8_t * tile = dst + ((size_t) ct * (size_t) n_k_tiles + (size_t) kt) * TILE_SIZE;

            for (int l = 0; l < 4; ++l) {
                tile[INDEX_PLANE_OFFSET + (size_t) l * TILE_ROWS + (size_t) row] =
                    b->qs[4 * group + l];
                tile[SIGN_PLANE_OFFSET + (size_t) l * TILE_ROWS + (size_t) row] =
                    b->qs[QK_K / 8 + 4 * group + l];
            }

            tile[QH_PLANE_OFFSET    + (size_t) row] = b->qh[group];
            tile[SCALE_PLANE_OFFSET + (size_t) row] = b->scales[group];
            std::memcpy(tile + D_PLANE_OFFSET + (size_t) row * sizeof(ggml_half),
                        &b->d, sizeof(ggml_half));
        }
    }

    return true;
}

// Reverse repack_2d for host-side validation/read-back.
//
// Because d is duplicated in every 32-column tile, this also checks that all
// eight copies agree.  A mismatch returns false and is treated as corrupted
// repacked data.
inline bool unpack_2d(
        const uint8_t * src,
        size_t src_size,
        int64_t ne0,
        int64_t ne1,
        block_iq2_s * dst,
        size_t dst_size) {
    if (!valid_2d_shape(ne0, ne1)) {
        return false;
    }

    const size_t expected_src = repacked_size_2d(ne0, ne1);
    const size_t expected_dst = original_size_2d(ne0, ne1);
    if (src_size < expected_src || dst_size < expected_dst) {
        return false;
    }
    if ((expected_src != 0 && src == nullptr) || (expected_dst != 0 && dst == nullptr)) {
        return false;
    }

    if (expected_dst != 0) {
        std::memset(dst, 0, expected_dst);
    }

    const int64_t superblocks_per_row = ne0 / QK_K;
    const int64_t n_k_tiles = ne0 / TILE_COLS;

    for (int64_t r = 0; r < ne1; ++r) {
        const int64_t ct  = r / TILE_ROWS;
        const int64_t row = r % TILE_ROWS;
        block_iq2_s * dst_row = dst + r * superblocks_per_row;

        for (int64_t kt = 0; kt < n_k_tiles; ++kt) {
            const int64_t group = kt % GROUPS_PER_SUPERBLOCK;
            block_iq2_s * b = dst_row + kt / GROUPS_PER_SUPERBLOCK;
            const uint8_t * tile =
                src + ((size_t) ct * (size_t) n_k_tiles + (size_t) kt) * TILE_SIZE;
            const uint8_t * d_src =
                tile + D_PLANE_OFFSET + (size_t) row * sizeof(ggml_half);

            if (group == 0) {
                std::memcpy(&b->d, d_src, sizeof(ggml_half));
            } else if (std::memcmp(&b->d, d_src, sizeof(ggml_half)) != 0) {
                return false;
            }

            for (int l = 0; l < 4; ++l) {
                b->qs[4 * group + l] =
                    tile[INDEX_PLANE_OFFSET + (size_t) l * TILE_ROWS + (size_t) row];
                b->qs[QK_K / 8 + 4 * group + l] =
                    tile[SIGN_PLANE_OFFSET + (size_t) l * TILE_ROWS + (size_t) row];
            }

            b->qh[group]     = tile[QH_PLANE_OFFSET    + (size_t) row];
            b->scales[group] = tile[SCALE_PLANE_OFFSET + (size_t) row];
        }
    }

    return true;
}

} // namespace ggml_hexagon_iq2s
