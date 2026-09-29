# IQ2_S Phase-1 Hexagon repack

This document freezes the initial host-side repack contract for adding
`GGML_TYPE_IQ2_S` to the Snapdragon/Hexagon backend.

The goal of this phase is correctness and a layout that maps directly onto the
backend's existing fixed 32x32 tiled matmul machinery.  HTP dispatch, HVX
dequantization, and HMX execution are intentionally not enabled yet.

## Source IQ2_S block

Current `block_iq2_s` stores 256 weights in 82 bytes:

```text
d       2 bytes   fp16 super-block scale
qs     64 bytes   first 32 bytes: 8-bit grid indices
                  second 32 bytes: sign masks
qh      8 bytes   2 high index bits for each 32-value group
scales  8 bytes   two 4-bit sub-scales for each 32-value group
----------------
total  82 bytes / 256 weights = 2.5625 bits/weight
```

A 256-value super-block contains eight consecutive 32-value groups.  For one
32-value group and one row, decoding needs:

- 4 low grid-index bytes;
- 4 sign bytes;
- 1 `qh` byte;
- 1 scale byte;
- the shared fp16 `d`.

## Initial HTP tile

The phase-1 repack stores one 32-row x 32-column weight tile in exactly
384 bytes, or three 128-byte HVX vectors:

```text
offset       size      contents
0            128       index[l][row], l = 0..3, row = 0..31
128          128       sign [l][row], l = 0..3, row = 0..31
256           32       qh[row]
288           32       scale[row]
320           64       d[row] as raw fp16 bits
384                    end
```

The first two planes are already transposed into the natural
`[four 8-value groups][32 rows]` order.  The last 128-byte vector is a metadata
plane containing `qh`, scale, and `d`.

Every tile is self-contained and every tile starts at a 128-byte boundary when
the repacked matrix buffer itself is 128-byte aligned.

## Why duplicate `d`

The current Hexagon matmul code addresses a fixed-size object per 32x32
weight tile.  Keeping `d` inside each tile means the first IQ2_S HTP kernels
can follow the same addressing model as Q4_0/Q5_K/Q6_K without adding a second
shared-metadata address calculation.

The cost is measurable:

```text
source IQ2_S     2.5625 bits/weight
phase-1 repack   3.0000 bits/weight
increase        17.073%
```

This is a deliberate phase-1 tradeoff, not a final memory target.  If device
profiling shows memory pressure or bandwidth pressure, a later compact layout
can share one 64-byte `d[32]` plane across the eight 32-column tiles belonging
to the same 256-value super-block.

## Matrix ordering

For a 2D source matrix:

- `ne0` is the reduction dimension and must be a multiple of 256;
- `ne1` is the number of rows;
- source storage is normal row-major IQ2_S blocks;
- rows are padded to a multiple of 32 in the repacked buffer;
- tile order is `[row_tile][k_tile]`;
- `k_tile` advances in units of 32 source columns.

Within each source 256-value super-block, `k_tile % 8` selects the 32-value
group.

## Phase-1 validation contract

`ggml/src/ggml-hexagon/iq2-s-repack.h` provides a host-only forward and reverse
repack.  The reverse path checks that all eight duplicated copies of `d`
match.

`tests/test-hexagon-iq2-s-repack.cpp` verifies:

1. expected packed and repacked sizes;
2. arbitrary-byte round-trip equality;
3. valid IQ2_S produced by `ggml_quantize_chunk` round-trips byte-for-byte;
4. dequantized floats are bit-identical before and after the round trip;
5. inconsistent duplicated `d` metadata is detected.

No Hexagon SDK or device is required for these tests.

## Next implementation step

After this contract is stable, the HTP side can add:

- `HTP_TYPE_IQ2_S`;
- `HTP_MM_WEIGHT_TILE_SIZE_IQ2_S = 384`;
- an HVX IQ2_S -> fp16 dequantizer for the HMX prefill path;
- `tiled_vec_dot_iq2_s_32x1` and `32x2` for decode;
- dispatch and backend support checks.

The host repack should then be wired into `ggml-hexagon.cpp` rather than
redesigned at the same time as the HTP kernels.
