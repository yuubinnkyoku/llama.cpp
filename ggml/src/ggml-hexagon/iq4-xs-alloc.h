#pragma once

#include "iq4-xs-repack.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

// Production allocation state for an IQ4_XS weight.
//
// The tile folds fp16(d) * signed 6-bit scale into one fp16 scale per row. That
// is not representable for every valid raw tensor, so an allocation records which
// representation its bytes currently hold. Only REPACKED_SAFE may reach the DSP.
//
// The root allocation owns this object. Views alias it through a shared_ptr and
// address the owner through their logical raw offset, so a view can never hold a
// diverging copy of the state or of the metadata sidecar.

enum ggml_hexagon_iq4xs_alloc_state {
    // allocated, upload incomplete: not representable, must not be computed on
    GGML_HEXAGON_IQ4XS_UNINITIALIZED = 0,
    // primary allocation holds raw IQ4_XS bytes; scheduler copies it to CPU
    GGML_HEXAGON_IQ4XS_RAW_FALLBACK  = 1,
    // primary allocation holds IQ4_NL-compatible tiles plus the metadata sidecar
    GGML_HEXAGON_IQ4XS_REPACKED_SAFE = 2,
};

struct iq4xs_alloc_state {
    // root layout, per (ne2,ne3) matrix slice
    int64_t ne[4] = { 0, 0, 0, 0 };
    size_t  raw_slice_size  = 0; // raw bytes of one ne0 x ne1 matrix
    size_t  tile_slice_size = 0; // tiled bytes of one ne0 x ne1 matrix
    size_t  meta_slice_size = 0; // sidecar bytes of one ne0 x ne1 matrix

    int state = GGML_HEXAGON_IQ4XS_UNINITIALIZED;

    // original 8 bytes per block (d, scales_h, scales_l). Kept so readback is
    // bit exact; reconstructing these from the folded tile scale is lossy.
    std::vector<uint8_t> metadata;

    size_t n_slices() const { return (size_t) ne[2] * (size_t) ne[3]; }

    bool is_repacked() const { return state == GGML_HEXAGON_IQ4XS_REPACKED_SAFE; }
};

// Record the layout of a new allocation. False means the shape is unsupported
// and the allocation stays UNINITIALIZED.
inline bool iq4xs_init_layout(iq4xs_alloc_state & st, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    size_t raw = 0, tiled = 0;
    if (!ggml_hexagon_iq4xs::layout_sizes(ne0, ne1, raw, tiled)) {
        return false;
    }
    st.ne[0] = ne0; st.ne[1] = ne1; st.ne[2] = ne2; st.ne[3] = ne3;
    st.raw_slice_size  = raw;
    st.tile_slice_size = tiled;
    st.meta_slice_size = ggml_hexagon_iq4xs::metadata_size_2d(ne0, ne1);
    st.metadata.clear();
    st.state = GGML_HEXAGON_IQ4XS_UNINITIALIZED;
    return true;
}

// Scan a complete raw source, then commit either tiles plus sidecar, or the raw
// bytes themselves. The scan finishes before any write and the sidecar is sized
// before the repack, so a failure cannot leave a half-repacked allocation.
//
// Return value: true on commit, false when nothing was committed. The caller
// distinguishes a rejected upload (non-finite d) from a committed RAW_FALLBACK
// by reading st.state, which is left untouched on failure.
inline bool iq4xs_prepare(
        iq4xs_alloc_state & st,
        const uint8_t * src, size_t src_size,
        uint8_t * dst, size_t dst_size) {
    if (st.raw_slice_size == 0 || st.tile_slice_size == 0 || st.meta_slice_size == 0) {
        return false;
    }
    const size_t n_slices = st.n_slices();
    if (src == nullptr || dst == nullptr ||
        src_size < st.raw_slice_size * n_slices ||
        dst_size < st.tile_slice_size * n_slices) {
        return false;
    }

    // valid raw IQ4_XS can still overflow the fp16 tile scale, so scan first
    bool overflow = false;
    for (size_t s = 0; s < n_slices; ++s) {
        const block_iq4_xs * src_slice =
            (const block_iq4_xs *) (src + s * st.raw_slice_size);
        for (size_t i = 0; i < st.raw_slice_size / sizeof(block_iq4_xs); ++i) {
            for (int ib = 0; ib < 8; ++ib) {
                const ggml_hexagon_iq4xs::scale_status status =
                    ggml_hexagon_iq4xs::iq4_xs_effective_scale_status(src_slice[i], ib);
                if (status == ggml_hexagon_iq4xs::scale_status::RAW_NONFINITE) {
                    // not a representation change, it is an invalid upload
                    return false;
                }
                if (status == ggml_hexagon_iq4xs::scale_status::TILE_SCALE_OVERFLOW) {
                    overflow = true;
                }
            }
        }
    }

    if (overflow) {
        // RAW_FALLBACK: the raw bytes are already the exact source of truth, so
        // they stay in the primary allocation and no sidecar is built
        for (size_t s = 0; s < n_slices; ++s) {
            std::memcpy(dst + s * st.tile_slice_size, src + s * st.raw_slice_size, st.raw_slice_size);
        }
        st.metadata.clear();
        st.metadata.shrink_to_fit();
        st.state = GGML_HEXAGON_IQ4XS_RAW_FALLBACK;
        return true;
    }

    std::vector<uint8_t> metadata(st.meta_slice_size * n_slices);
    for (size_t s = 0; s < n_slices; ++s) {
        const block_iq4_xs * src_slice =
            (const block_iq4_xs *) (src + s * st.raw_slice_size);
        if (!ggml_hexagon_iq4xs::repack_2d_with_metadata(
                src_slice, st.raw_slice_size, st.ne[0], st.ne[1],
                dst + s * st.tile_slice_size, st.tile_slice_size,
                (ggml_hexagon_iq4xs::block_metadata *) (metadata.data() + s * st.meta_slice_size),
                st.meta_slice_size)) {
            return false;
        }
    }
    st.metadata = std::move(metadata);
    st.state = GGML_HEXAGON_IQ4XS_REPACKED_SAFE;
    return true;
}

// Exact raw readback of a logical raw byte range.
//
// base must be the start of the owner allocation (not the view pointer), and
// logical_offset must already be root-relative. Only the requested blocks are
// reconstructed, one block of scratch at a time, and requests are split at
// matrix slice boundaries so ne2/ne3 stay correct.
inline bool iq4xs_readback(
        const iq4xs_alloc_state & st,
        const uint8_t * base,
        size_t logical_offset, size_t size,
        void * dst) {
    if (st.state == GGML_HEXAGON_IQ4XS_RAW_FALLBACK) {
        if (dst == nullptr) {
            return false;
        }
        std::memcpy(dst, base + logical_offset, size);
        return true;
    }
    if (!st.is_repacked() || dst == nullptr) {
        return false;
    }

    const size_t n_slices = st.n_slices();
    size_t done = 0;
    while (done < size) {
        const size_t pos = logical_offset + done;
        const size_t slice = st.raw_slice_size ? pos / st.raw_slice_size : n_slices;
        if (slice >= n_slices) {
            return false;
        }
        const size_t in_slice = pos % st.raw_slice_size;
        const size_t remaining = st.raw_slice_size - in_slice;
        const size_t chunk = remaining < (size - done) ? remaining : (size - done);
        if (!ggml_hexagon_iq4xs::readback_2d(
                base + slice * st.tile_slice_size, st.tile_slice_size,
                (const ggml_hexagon_iq4xs::block_metadata *)
                    (st.metadata.data() + slice * st.meta_slice_size),
                st.meta_slice_size, st.ne[0], st.ne[1], in_slice,
                (uint8_t *) dst + done, chunk)) {
            return false;
        }
        done += chunk;
    }
    return true;
}

// Byte range of the owner allocation that a view of `view_offset` raw bytes and
// `size` bytes addresses, validated against the root tensor size.
inline bool iq4xs_view_range(const iq4xs_alloc_state & st, size_t view_offset, size_t size, size_t owner_nbytes) {
    if (view_offset > owner_nbytes || size > owner_nbytes - view_offset) {
        return false;
    }
    const size_t total = st.raw_slice_size * st.n_slices();
    return view_offset <= total && size <= total - view_offset;
}