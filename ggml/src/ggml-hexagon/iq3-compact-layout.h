#pragma once

#include <stddef.h>
#include <stdint.h>

#define IQ3_COMPACT_TILE_ROWS 32
#define IQ3_COMPACT_TILE_COLS 32
#define IQ3_COMPACT_GROUPS_PER_SUPERBLOCK 8
#define IQ3_COMPACT_DATA_TILE_SIZE 384
#define IQ3_COMPACT_SUPERBLOCK_K 256
#define IQ3_COMPACT_ALIGNMENT 128

#define IQ3XXS_COMPACT_DATA_PAIR_SIZE (16 * IQ3_COMPACT_DATA_TILE_SIZE)
#define IQ3XXS_COMPACT_PAIR_SIZE (IQ3XXS_COMPACT_DATA_PAIR_SIZE + 2 * IQ3_COMPACT_TILE_ROWS * 2)
#define IQ3XXS_COMPACT_TAIL_SIZE (8 * IQ3_COMPACT_DATA_TILE_SIZE + IQ3_COMPACT_TILE_ROWS * 2 + IQ3_COMPACT_TILE_ROWS * 2)

#define IQ3S_COMPACT_DATA_PAIR_SIZE (16 * IQ3_COMPACT_DATA_TILE_SIZE)
#define IQ3S_COMPACT_QH_PAIR_SIZE (2 * 8 * IQ3_COMPACT_TILE_ROWS)
#define IQ3S_COMPACT_SCALE_PAIR_SIZE (2 * 4 * IQ3_COMPACT_TILE_ROWS)
#define IQ3S_COMPACT_D_PAIR_SIZE (2 * IQ3_COMPACT_TILE_ROWS * 2)
#define IQ3S_COMPACT_PAIR_SIZE (IQ3S_COMPACT_DATA_PAIR_SIZE + IQ3S_COMPACT_QH_PAIR_SIZE + IQ3S_COMPACT_SCALE_PAIR_SIZE + IQ3S_COMPACT_D_PAIR_SIZE)
#define IQ3S_COMPACT_TAIL_SIZE (8 * IQ3_COMPACT_DATA_TILE_SIZE + 8 * IQ3_COMPACT_TILE_ROWS + 4 * IQ3_COMPACT_TILE_ROWS + IQ3_COMPACT_TILE_ROWS * 2 + IQ3_COMPACT_TILE_ROWS * 2)

typedef struct iq3_compact_offsets {
    size_t data_tile;
    size_t qh_plane;
    size_t scale_plane;
    size_t d_plane;
    uint32_t group;
    uint32_t lane;
} iq3_compact_offsets;

static inline int iq3_compact_mul_size(size_t a, size_t b, size_t * out) {
    if (b != 0 && a > SIZE_MAX / b) {
        return 0;
    }
    *out = a * b;
    return 1;
}

static inline int iq3_compact_add_size(size_t a, size_t b, size_t * out) {
    if (a > SIZE_MAX - b) {
        return 0;
    }
    *out = a + b;
    return 1;
}

static inline int iq3_compact_total_size(size_t row_tile_bytes, int64_t ne1, int64_t ne2, int64_t ne3, size_t * out) {
    if (row_tile_bytes == 0 || ne1 < 0 || ne2 < 0 || ne3 < 0) {
        return 0;
    }
    const uint64_t rows = (uint64_t) ne1;
    const uint64_t row_tiles_u64 = rows / IQ3_COMPACT_TILE_ROWS + (rows % IQ3_COMPACT_TILE_ROWS != 0);
    if (row_tiles_u64 > SIZE_MAX) {
        return 0;
    }
    size_t total = 0;
    if ((uint64_t) ne2 > SIZE_MAX || (uint64_t) ne3 > SIZE_MAX ||
        !iq3_compact_mul_size(row_tile_bytes, (size_t) row_tiles_u64, &total) ||
        !iq3_compact_mul_size(total, (size_t) ne2, &total) ||
        !iq3_compact_mul_size(total, (size_t) ne3, &total)) {
        return 0;
    }
    *out = total;
    return 1;
}

static inline int iq3_compact_xxs_row_tile_size(int64_t ne0, size_t * out) {
    if (ne0 <= 0 || ne0 % IQ3_COMPACT_SUPERBLOCK_K != 0) {
        return 0;
    }
    const uint64_t n_superblocks = (uint64_t) ne0 / IQ3_COMPACT_SUPERBLOCK_K;
    const uint64_t n_pairs = n_superblocks / 2;
    const size_t tail = n_superblocks % 2 ? IQ3XXS_COMPACT_TAIL_SIZE : 0;
    if (n_pairs > SIZE_MAX / IQ3XXS_COMPACT_PAIR_SIZE) {
        return 0;
    }
    return iq3_compact_add_size((size_t) n_pairs * IQ3XXS_COMPACT_PAIR_SIZE, tail, out);
}

static inline int iq3_compact_xxs_total_size(int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t * out) {
    size_t row_tile_bytes;
    return iq3_compact_xxs_row_tile_size(ne0, &row_tile_bytes) &&
           iq3_compact_total_size(row_tile_bytes, ne1, ne2, ne3, out);
}

static inline int iq3_compact_xxs_offsets(int64_t ne0, uint64_t kt, iq3_compact_offsets * out) {
    if (out == NULL || ne0 <= 0 || ne0 % IQ3_COMPACT_SUPERBLOCK_K != 0 || kt >= (uint64_t) ne0 / IQ3_COMPACT_TILE_COLS) {
        return 0;
    }
    const uint64_t n_superblocks = (uint64_t) ne0 / IQ3_COMPACT_SUPERBLOCK_K;
    const uint64_t superblock = kt / IQ3_COMPACT_GROUPS_PER_SUPERBLOCK;
    const uint32_t group = (uint32_t) (kt % IQ3_COMPACT_GROUPS_PER_SUPERBLOCK);
    const uint64_t n_pairs = n_superblocks / 2;
    const uint64_t pair = superblock / 2;
    const uint32_t lane = (uint32_t) (superblock % 2);
    size_t base = 0;
    if (superblock < n_pairs * 2) {
        if (pair > SIZE_MAX / IQ3XXS_COMPACT_PAIR_SIZE) {
            return 0;
        }
        base = (size_t) pair * IQ3XXS_COMPACT_PAIR_SIZE;
        out->data_tile = base + lane * (8 * IQ3_COMPACT_DATA_TILE_SIZE) + group * IQ3_COMPACT_DATA_TILE_SIZE;
        out->d_plane = base + IQ3XXS_COMPACT_DATA_PAIR_SIZE + lane * IQ3_COMPACT_TILE_ROWS * 2;
    } else {
        if (n_pairs > SIZE_MAX / IQ3XXS_COMPACT_PAIR_SIZE) {
            return 0;
        }
        base = (size_t) n_pairs * IQ3XXS_COMPACT_PAIR_SIZE;
        out->data_tile = base + group * IQ3_COMPACT_DATA_TILE_SIZE;
        out->d_plane = base + 8 * IQ3_COMPACT_DATA_TILE_SIZE;
    }
    out->qh_plane = 0;
    out->scale_plane = 0;
    out->group = group;
    out->lane = lane;
    return 1;
}

static inline int iq3_compact_s_row_tile_size(int64_t ne0, size_t * out) {
    if (ne0 <= 0 || ne0 % IQ3_COMPACT_SUPERBLOCK_K != 0) {
        return 0;
    }
    const uint64_t n_superblocks = (uint64_t) ne0 / IQ3_COMPACT_SUPERBLOCK_K;
    const uint64_t n_pairs = n_superblocks / 2;
    const size_t tail = n_superblocks % 2 ? IQ3S_COMPACT_TAIL_SIZE : 0;
    if (n_pairs > SIZE_MAX / IQ3S_COMPACT_PAIR_SIZE) {
        return 0;
    }
    return iq3_compact_add_size((size_t) n_pairs * IQ3S_COMPACT_PAIR_SIZE, tail, out);
}

static inline int iq3_compact_s_total_size(int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t * out) {
    size_t row_tile_bytes;
    return iq3_compact_s_row_tile_size(ne0, &row_tile_bytes) &&
           iq3_compact_total_size(row_tile_bytes, ne1, ne2, ne3, out);
}

static inline int iq3_compact_s_offsets(int64_t ne0, uint64_t kt, iq3_compact_offsets * out) {
    if (out == NULL || ne0 <= 0 || ne0 % IQ3_COMPACT_SUPERBLOCK_K != 0 || kt >= (uint64_t) ne0 / IQ3_COMPACT_TILE_COLS) {
        return 0;
    }
    const uint64_t n_superblocks = (uint64_t) ne0 / IQ3_COMPACT_SUPERBLOCK_K;
    const uint64_t superblock = kt / IQ3_COMPACT_GROUPS_PER_SUPERBLOCK;
    const uint32_t group = (uint32_t) (kt % IQ3_COMPACT_GROUPS_PER_SUPERBLOCK);
    const uint64_t n_pairs = n_superblocks / 2;
    const uint64_t pair = superblock / 2;
    const uint32_t lane = (uint32_t) (superblock % 2);
    size_t base = 0;
    if (superblock < n_pairs * 2) {
        if (pair > SIZE_MAX / IQ3S_COMPACT_PAIR_SIZE) {
            return 0;
        }
        base = (size_t) pair * IQ3S_COMPACT_PAIR_SIZE;
        out->data_tile = base + lane * (8 * IQ3_COMPACT_DATA_TILE_SIZE) + group * IQ3_COMPACT_DATA_TILE_SIZE;
        out->qh_plane = base + IQ3S_COMPACT_DATA_PAIR_SIZE + lane * 8 * IQ3_COMPACT_TILE_ROWS;
        out->scale_plane = base + IQ3S_COMPACT_DATA_PAIR_SIZE + IQ3S_COMPACT_QH_PAIR_SIZE + lane * 4 * IQ3_COMPACT_TILE_ROWS;
        out->d_plane = base + IQ3S_COMPACT_DATA_PAIR_SIZE + IQ3S_COMPACT_QH_PAIR_SIZE + IQ3S_COMPACT_SCALE_PAIR_SIZE + lane * IQ3_COMPACT_TILE_ROWS * 2;
    } else {
        if (n_pairs > SIZE_MAX / IQ3S_COMPACT_PAIR_SIZE) {
            return 0;
        }
        base = (size_t) n_pairs * IQ3S_COMPACT_PAIR_SIZE;
        out->data_tile = base + group * IQ3_COMPACT_DATA_TILE_SIZE;
        out->qh_plane = base + 8 * IQ3_COMPACT_DATA_TILE_SIZE;
        out->scale_plane = out->qh_plane + 8 * IQ3_COMPACT_TILE_ROWS;
        out->d_plane = out->scale_plane + 4 * IQ3_COMPACT_TILE_ROWS;
    }
    out->group = group;
    out->lane = lane;
    return 1;
}

