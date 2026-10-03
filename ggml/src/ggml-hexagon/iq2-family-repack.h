#pragma once

#include "ggml-quants.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

// Host-side layout for the IQ2_XS and IQ2_XXS 32x32 HTP weight tiles.
//
// Both types reuse the IQ2_S tile shape so the DMA/tile plumbing in
// matmul-ops.c stays identical (384 B per 32x32 tile, HVX aligned):
//
//   [0,   128)  grid-index plane: low 8 index bits, [l][row]
//   [128, 256)  sign plane:       sign payload,     [l][row]
//   [256, 288)  unused (IQ2_S qh slot)
//   [288, 320)  scale[row]
//   [320, 384)  d[row] as raw ggml_half
//
// Both types use one sign-plane encoding so the HVX unpack is shared:
//   bit 0      = grid-index bit 8 (always 0 for IQ2_XXS, whose index is 8 bit)
//   bits 1..7  = ksigns_iq2xs sign index of subgroup l
//
// Scale plane: IQ2_XS keeps the source byte (low nibble covers subgroups 0,1,
// high nibble covers 2,3).  IQ2_XXS scales the whole 32-value group, so the
// single source nibble is duplicated into both halves.

namespace ggml_hexagon_iq2f {

static_assert(QK_K == 256, "IQ2 family repack assumes QK_K == 256");
static_assert(sizeof(block_iq2_xs)  == 74, "IQ2_XS source block layout changed");
static_assert(sizeof(block_iq2_xxs) == 66, "IQ2_XXS source block layout changed");

static constexpr int64_t TILE_ROWS = 32;
static constexpr int64_t TILE_COLS = 32;
static constexpr int64_t GROUPS_PER_SUPERBLOCK = QK_K / TILE_COLS; // 8

static constexpr size_t INDEX_PLANE_OFFSET = 0;
static constexpr size_t SIGN_PLANE_OFFSET  = 128;
static constexpr size_t SPARE_PLANE_OFFSET = 256;
static constexpr size_t SCALE_PLANE_OFFSET = 288;
static constexpr size_t D_PLANE_OFFSET     = 320;
static constexpr size_t TILE_SIZE          = 384;

static_assert(TILE_SIZE % 128 == 0, "IQ2 family HTP tile must stay HVX aligned");
static_assert(D_PLANE_OFFSET + TILE_ROWS * sizeof(ggml_half) == TILE_SIZE,
              "IQ2 family tile layout does not fill three HVX vectors plus tail");
static_assert(SCALE_PLANE_OFFSET + TILE_ROWS == D_PLANE_OFFSET,
              "IQ2 family scale plane must sit right before d");

// One (block, 32-value group, subgroup l) triple as stored in the tile.
struct fields {
    uint8_t index; // low 8 bits of the grid index
    uint8_t sign;  // sign-plane payload (see file header)
    uint8_t scale; // scale byte of the 32-value group
};

struct iq2_xs_traits {
    using block_t = block_iq2_xs;

    static fields read(const block_t & b, int group, int l) {
        const uint16_t q = b.qs[4 * group + l];
        fields f;
        f.index = (uint8_t) (q & 0xffu);
        f.sign  = (uint8_t) (q >> 8);
        f.scale = b.scales[group];
        return f;
    }

    static void write_group(block_t & b, int group, const fields (&fs)[4]) {
        for (int l = 0; l < 4; ++l) {
            b.qs[4 * group + l] = (uint16_t) (fs[l].index | ((uint16_t) fs[l].sign << 8));
        }
        b.scales[group] = fs[0].scale;
    }
};

struct iq2_xxs_traits {
    using block_t = block_iq2_xxs;

    // the packed 4 x 7-bit sign field + 4-bit scale lives in source bytes 4..7;
    // read it as a little-endian uint32 (all supported targets are LE)
    static uint32_t packed(const block_t & b, int group) {
        uint32_t v = 0;
        std::memcpy(&v, (const uint8_t *) &b.qs[4 * group] + 4, sizeof(v));
        return v;
    }

    static fields read(const block_t & b, int group, int l) {
        const uint8_t * gb = (const uint8_t *) &b.qs[4 * group];
        const uint32_t p   = packed(b, group);

        fields f;
        f.index = gb[l];
        f.sign  = (uint8_t) (((p >> (7 * l)) & 127u) << 1);
        const uint8_t nibble = (uint8_t) (p >> 28);
        f.scale = (uint8_t) (nibble | (nibble << 4));
        return f;
    }

    static void write_group(block_t & b, int group, const fields (&fs)[4]) {
        uint8_t * gb = (uint8_t *) &b.qs[4 * group];
        uint32_t p   = 0;
        for (int l = 0; l < 4; ++l) {
            gb[l] = fs[l].index;
            p |= (uint32_t) ((fs[l].sign >> 1) & 127u) << (7 * l);
        }
        p |= (uint32_t) (fs[0].scale & 0x0fu) << 28;
        std::memcpy(gb + 4, &p, sizeof(p));
    }
};

inline bool valid_2d_shape(int64_t ne0, int64_t ne1) {
    return ne0 > 0 && ne1 >= 0 && ne0 % QK_K == 0;
}

inline int64_t padded_rows(int64_t ne1) {
    return (ne1 + TILE_ROWS - 1) / TILE_ROWS * TILE_ROWS;
}

template <typename Traits>
inline size_t original_size_2d(int64_t ne0, int64_t ne1) {
    if (!valid_2d_shape(ne0, ne1)) {
        return 0;
    }
    const size_t superblocks_per_row = (size_t) (ne0 / QK_K);
    return (size_t) ne1 * superblocks_per_row * sizeof(typename Traits::block_t);
}

inline size_t repacked_size_2d(int64_t ne0, int64_t ne1) {
    if (!valid_2d_shape(ne0, ne1)) {
        return 0;
    }
    const size_t row_tiles = (size_t) (padded_rows(ne1) / TILE_ROWS);
    const size_t k_tiles   = (size_t) (ne0 / TILE_COLS);
    return row_tiles * k_tiles * TILE_SIZE;
}

// Repack a row-major source matrix into fixed 32x32 HTP tiles.
// Padded rows in dst are zero-filled.  Only d is duplicated across the eight
// 32-column tiles of one superblock; every other field is stored byte-exact.
template <typename Traits>
inline bool repack_2d(
        const typename Traits::block_t * src,
        size_t src_size,
        int64_t ne0,
        int64_t ne1,
        uint8_t * dst,
        size_t dst_size) {
    using block_t = typename Traits::block_t;

    if (!valid_2d_shape(ne0, ne1)) {
        return false;
    }

    const size_t expected_src = original_size_2d<Traits>(ne0, ne1);
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
        const block_t * src_row = src + r * superblocks_per_row;

        for (int64_t kt = 0; kt < n_k_tiles; ++kt) {
            const int64_t group = kt % GROUPS_PER_SUPERBLOCK;
            const block_t * b = src_row + kt / GROUPS_PER_SUPERBLOCK;
            uint8_t * tile = dst + ((size_t) ct * (size_t) n_k_tiles + (size_t) kt) * TILE_SIZE;

            for (int l = 0; l < 4; ++l) {
                const fields f = Traits::read(*b, (int) group, l);
                tile[INDEX_PLANE_OFFSET + (size_t) l * TILE_ROWS + (size_t) row] = f.index;
                tile[SIGN_PLANE_OFFSET  + (size_t) l * TILE_ROWS + (size_t) row] = f.sign;
            }

            tile[SCALE_PLANE_OFFSET + (size_t) row] = Traits::read(*b, (int) group, 0).scale;
            std::memcpy(tile + D_PLANE_OFFSET + (size_t) row * sizeof(ggml_half),
                   &b->d, sizeof(ggml_half));
        }
    }

    return true;
}

// Reverse repack_2d for host-side validation/read-back.
// Because d is duplicated in every 32-column tile, all eight copies must agree.
template <typename Traits>
inline bool unpack_2d(
        const uint8_t * src,
        size_t src_size,
        int64_t ne0,
        int64_t ne1,
        typename Traits::block_t * dst,
        size_t dst_size) {
    using block_t = typename Traits::block_t;

    if (!valid_2d_shape(ne0, ne1)) {
        return false;
    }

    const size_t expected_src = repacked_size_2d(ne0, ne1);
    const size_t expected_dst = original_size_2d<Traits>(ne0, ne1);
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
        block_t * dst_row = dst + r * superblocks_per_row;

        for (int64_t kt = 0; kt < n_k_tiles; ++kt) {
            const int64_t group = kt % GROUPS_PER_SUPERBLOCK;
            block_t * b = dst_row + kt / GROUPS_PER_SUPERBLOCK;
            const uint8_t * tile =
                src + ((size_t) ct * (size_t) n_k_tiles + (size_t) kt) * TILE_SIZE;
            const uint8_t * d_src = tile + D_PLANE_OFFSET + (size_t) row * sizeof(ggml_half);

            if (group == 0) {
                std::memcpy(&b->d, d_src, sizeof(ggml_half));
            } else if (std::memcmp(&b->d, d_src, sizeof(ggml_half)) != 0) {
                return false;
            }

            fields fs[4];
            for (int l = 0; l < 4; ++l) {
                fs[l].index = tile[INDEX_PLANE_OFFSET + (size_t) l * TILE_ROWS + (size_t) row];
                fs[l].sign  = tile[SIGN_PLANE_OFFSET  + (size_t) l * TILE_ROWS + (size_t) row];
            }
            fs[0].scale = tile[SCALE_PLANE_OFFSET + (size_t) row];
            fs[1].scale = fs[0].scale;
            fs[2].scale = fs[0].scale;
            fs[3].scale = fs[0].scale;
            Traits::write_group(*b, (int) group, fs);
        }
    }

    return true;
}

} // namespace ggml_hexagon_iq2f
