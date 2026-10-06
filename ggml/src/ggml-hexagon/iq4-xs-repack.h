#pragma once

#include "ggml-quants.h"
#include "ggml-impl.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace ggml_hexagon_iq4xs {

static_assert(QK_K == 256 && sizeof(block_iq4_xs) == 136, "IQ4_XS raw layout changed");
static_assert(offsetof(block_iq4_xs, d) == 0 && offsetof(block_iq4_xs, scales_h) == 2 &&
              offsetof(block_iq4_xs, scales_l) == 4 && offsetof(block_iq4_xs, qs) == 8,
              "IQ4_XS raw field offsets changed");

static constexpr size_t TILE_ROWS = 32;
static constexpr size_t TILE_COLS = 32;
static constexpr size_t SCALE_PLANE_OFFSET = 512;
static constexpr size_t TILE_SIZE = 576;
static_assert(SCALE_PLANE_OFFSET + TILE_ROWS * sizeof(ggml_half) == TILE_SIZE, "IQ4_NL tile size changed");
// Eight tiles use 144 bytes per row, versus 136 raw bytes: 5.8823529% overhead before row padding.
static_assert(8 * TILE_SIZE / TILE_ROWS == 144, "IQ4_XS tiled storage changed");

inline int iq4_xs_scale(const block_iq4_xs & b, int ib) {
    GGML_ASSERT(ib >= 0 && ib < 8);
    const unsigned ls = ((unsigned(b.scales_l[ib / 2]) >> (4 * (ib % 2))) & 0x0fu) |
                        (((unsigned(b.scales_h) >> (2 * ib)) & 0x03u) << 4);
    return int(ls) - 32;
}

inline void unpack_iq4_xs_indices_32(const block_iq4_xs & b, int ib, uint8_t * q) {
    GGML_ASSERT(ib >= 0 && ib < 8);
    const uint8_t * qs = b.qs + ib * 16;
    for (int j = 0; j < 16; ++j) {
        q[j]      = qs[j] & 0x0fu;
        q[j + 16] = qs[j] >> 4;
    }
}

inline float iq4_xs_effective_scale(const block_iq4_xs & b, int ib) {
    const float d = GGML_FP16_TO_FP32(b.d);
    const int s = iq4_xs_scale(b, ib);
    return d * (float) s;
}

enum class scale_status {
    OK,
    RAW_NONFINITE,
    TILE_SCALE_OVERFLOW,
};

inline scale_status iq4_xs_effective_scale_status(const block_iq4_xs & b, int ib) {
    const float d = GGML_FP16_TO_FP32(b.d);
    if (!std::isfinite(d)) {
        return scale_status::RAW_NONFINITE;
    }
    const float effective = d * (float) iq4_xs_scale(b, ib);
    if (!std::isfinite(effective)) {
        return scale_status::TILE_SCALE_OVERFLOW;
    }
    const ggml_half rounded = GGML_FP32_TO_FP16(effective);
    return std::isfinite(GGML_FP16_TO_FP32(rounded)) ? scale_status::OK : scale_status::TILE_SCALE_OVERFLOW;
}

struct block_metadata {
    uint8_t bytes[offsetof(block_iq4_xs, qs)];
};
static_assert(sizeof(block_metadata) == 8, "IQ4_XS sidecar must contain only original metadata");
static_assert(144 + sizeof(block_metadata) == 152, "IQ4_XS tile and sidecar storage changed");

inline bool layout_sizes(int64_t ne0, int64_t ne1, size_t & raw, size_t & tiled) {
    if (ne0 <= 0 || ne1 < 0 || ne0 % QK_K != 0) {
        return false;
    }
    const uint64_t rows = (uint64_t) ne1;
    const uint64_t sb = (uint64_t) ne0 / QK_K;
    const uint64_t row_tiles = rows / TILE_ROWS + (rows % TILE_ROWS != 0);
    const size_t limit = std::numeric_limits<size_t>::max();
    if (sb > limit / sizeof(block_iq4_xs) || sb > limit / (8 * TILE_SIZE)) {
        return false;
    }
    const size_t raw_row = (size_t) sb * sizeof(block_iq4_xs);
    const size_t tile_row = (size_t) sb * 8 * TILE_SIZE;
    if (rows > limit / raw_row || row_tiles > limit / tile_row) {
        return false;
    }
    raw = (size_t) rows * raw_row;
    tiled = (size_t) row_tiles * tile_row;
    return true;
}

inline size_t original_size_2d(int64_t ne0, int64_t ne1) {
    size_t raw = 0, tiled = 0;
    return layout_sizes(ne0, ne1, raw, tiled) ? raw : 0;
}

inline size_t repacked_size_2d(int64_t ne0, int64_t ne1) {
    size_t raw = 0, tiled = 0;
    return layout_sizes(ne0, ne1, raw, tiled) ? tiled : 0;
}

inline size_t metadata_size_2d(int64_t ne0, int64_t ne1) {
    size_t raw = 0, tiled = 0;
    return layout_sizes(ne0, ne1, raw, tiled) ? raw / sizeof(block_iq4_xs) * sizeof(block_metadata) : 0;
}

// Host prototype only. Source and destination must not overlap.
// Reject non-finite effective fp16 scales before writing dst.
inline bool repack_2d(const block_iq4_xs * src, size_t src_size, int64_t ne0, int64_t ne1,
                      uint8_t * dst, size_t dst_size) {
    size_t raw = 0, tiled = 0;
    if (!layout_sizes(ne0, ne1, raw, tiled) || src_size < raw || dst_size < tiled ||
        (raw != 0 && src == nullptr) || (tiled != 0 && dst == nullptr)) {
        return false;
    }
    for (size_t i = 0; i < raw / sizeof(block_iq4_xs); ++i) {
        for (int ib = 0; ib < 8; ++ib) {
            if (iq4_xs_effective_scale_status(src[i], ib) != scale_status::OK) {
                return false;
            }
        }
    }
    if (tiled == 0) {
        return true;
    }
    std::memset(dst, 0, tiled);
    const size_t sb_per_row = (size_t) (ne0 / QK_K);
    const size_t k_tiles = sb_per_row * 8;
    for (size_t r = 0; r < (size_t) ne1; ++r) {
        for (size_t kt = 0; kt < k_tiles; ++kt) {
            const block_iq4_xs & b = src[r * sb_per_row + kt / 8];
            const int ib = (int) (kt % 8);
            const size_t row = r % TILE_ROWS;
            uint8_t * tile = dst + ((r / TILE_ROWS) * k_tiles + kt) * TILE_SIZE;
            uint8_t q[32];
            unpack_iq4_xs_indices_32(b, ib, q);
            for (int cp = 0; cp < 16; ++cp) {
                tile[cp * TILE_ROWS + row] = q[2 * cp] | (q[2 * cp + 1] << 4);
            }
            const ggml_half d = GGML_FP32_TO_FP16(iq4_xs_effective_scale(b, ib));
            std::memcpy(tile + SCALE_PLANE_OFFSET + row * sizeof(d), &d, sizeof(d));
        }
    }
    return true;
}

// All buffers must be disjoint. Failures leave both outputs unchanged.
inline bool repack_2d_with_metadata(const block_iq4_xs * src, size_t src_size, int64_t ne0, int64_t ne1,
                                    uint8_t * dst, size_t dst_size, block_metadata * metadata, size_t metadata_size) {
    size_t raw = 0, tiled = 0;
    if (!layout_sizes(ne0, ne1, raw, tiled)) {
        return false;
    }
    const size_t needed = raw / sizeof(block_iq4_xs) * sizeof(block_metadata);
    if (metadata_size < needed || (needed != 0 && metadata == nullptr) ||
        !repack_2d(src, src_size, ne0, ne1, dst, dst_size)) {
        return false;
    }
    for (size_t i = 0; i < raw / sizeof(block_iq4_xs); ++i) {
        std::memcpy(metadata[i].bytes, src + i, sizeof(block_metadata));
    }
    return true;
}

inline void pack_iq4_xs_indices_32(const uint8_t * tile, size_t row, uint8_t * qs) {
    GGML_ASSERT(row < TILE_ROWS);
    for (size_t j = 0; j < 16; ++j) {
        const unsigned shift = 4 * (j % 2);
        const unsigned lo = (tile[(j / 2) * TILE_ROWS + row] >> shift) & 15u;
        const unsigned hi = (tile[((j + 16) / 2) * TILE_ROWS + row] >> shift) & 15u;
        qs[j] = (uint8_t) (lo | (hi << 4));
    }
}

inline void reconstruct_block(const uint8_t * tiled, const block_metadata * metadata,
                              size_t sb_per_row, size_t index, block_iq4_xs & b) {
    const size_t row = index / sb_per_row;
    const size_t sb = index % sb_per_row;
    const size_t k_tiles = sb_per_row * 8;
    std::memcpy(&b, metadata[index].bytes, sizeof(block_metadata));
    for (size_t ib = 0; ib < 8; ++ib) {
        const uint8_t * tile = tiled + ((row / TILE_ROWS) * k_tiles + sb * 8 + ib) * TILE_SIZE;
        pack_iq4_xs_indices_32(tile, row % TILE_ROWS, b.qs + ib * 16);
    }
}

// Read logical raw bytes with one block of scratch space. Padded rows are not read.
inline bool readback_2d(const uint8_t * tiled, size_t tiled_size, const block_metadata * metadata, size_t metadata_size,
                        int64_t ne0, int64_t ne1, size_t offset, void * dst, size_t size) {
    size_t raw = 0, needed_tiled = 0;
    if (!layout_sizes(ne0, ne1, raw, needed_tiled) || offset > raw || size > raw - offset) {
        return false;
    }
    if (size == 0) {
        return true;
    }
    const size_t needed_metadata = raw / sizeof(block_iq4_xs) * sizeof(block_metadata);
    if (tiled_size < needed_tiled || metadata_size < needed_metadata || tiled == nullptr || metadata == nullptr || dst == nullptr) {
        return false;
    }
    size_t done = 0;
    while (done < size) {
        const size_t pos = offset + done;
        const size_t block_offset = pos % sizeof(block_iq4_xs);
        const size_t available = sizeof(block_iq4_xs) - block_offset;
        const size_t chunk = size - done < available ? size - done : available;
        block_iq4_xs b;
        reconstruct_block(tiled, metadata, (size_t) (ne0 / QK_K), pos / sizeof(block_iq4_xs), b);
        std::memcpy((uint8_t *) dst + done, (const uint8_t *) &b + block_offset, chunk);
        done += chunk;
    }
    return true;
}

inline bool readback_2d_ranges(const uint8_t * tiled, size_t tiled_size,
                              const block_metadata * metadata, size_t metadata_size, int64_t ne0, int64_t ne1,
                              size_t offset, size_t size, size_t n_copies, size_t stride_tensor,
                              void * dst, size_t dst_size, size_t stride_data) {
    size_t raw = 0, needed_tiled = 0;
    if (!layout_sizes(ne0, ne1, raw, needed_tiled) || offset > raw || size > raw - offset) {
        return false;
    }
    if (n_copies == 0 || size == 0) {
        return true;
    }
    const size_t steps = n_copies - 1;
    if ((stride_tensor != 0 && steps > (raw - offset - size) / stride_tensor) || dst_size < size ||
        (stride_data != 0 && steps > (dst_size - size) / stride_data) || dst == nullptr ||
        tiled == nullptr || tiled_size < needed_tiled || metadata == nullptr ||
        metadata_size < raw / sizeof(block_iq4_xs) * sizeof(block_metadata)) {
        return false;
    }
    for (size_t i = 0; i < n_copies; ++i) {
        if (!readback_2d(tiled, tiled_size, metadata, metadata_size, ne0, ne1,
                         offset + i * stride_tensor, (uint8_t *) dst + i * stride_data, size)) {
            return false;
        }
    }
    return true;
}

enum class storage_mode {
    UNINITIALIZED,
    TILED,
    RAW,
};

// Prototype for complete uploads before graph placement. Unsafe finite data stays raw in the primary allocation.
inline bool prepare_2d(const block_iq4_xs * src, size_t src_size, int64_t ne0, int64_t ne1,
                       uint8_t * dst, size_t dst_size, block_metadata * metadata, size_t metadata_size,
                       storage_mode & mode, scale_status & status) {
    size_t raw = 0, tiled = 0;
    if (!layout_sizes(ne0, ne1, raw, tiled) || src_size < raw || dst_size < tiled ||
        (raw != 0 && (src == nullptr || dst == nullptr))) {
        return false;
    }
    status = scale_status::OK;
    for (size_t i = 0; i < raw / sizeof(block_iq4_xs); ++i) {
        for (int ib = 0; ib < 8; ++ib) {
            const scale_status current = iq4_xs_effective_scale_status(src[i], ib);
            if (current == scale_status::RAW_NONFINITE) {
                status = current;
                return false;
            }
            if (current == scale_status::TILE_SCALE_OVERFLOW) {
                status = current;
            }
        }
    }
    if (status == scale_status::TILE_SCALE_OVERFLOW) {
        if (raw != 0) {
            std::memcpy(dst, src, raw);
        }
        mode = storage_mode::RAW;
        return true;
    }
    if (!repack_2d_with_metadata(src, src_size, ne0, ne1, dst, dst_size, metadata, metadata_size)) {
        return false;
    }
    mode = storage_mode::TILED;
    return true;
}

} // namespace ggml_hexagon_iq4xs
