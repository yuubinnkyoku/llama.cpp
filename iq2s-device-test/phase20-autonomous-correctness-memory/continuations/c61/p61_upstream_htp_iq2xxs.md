# Phase 61 - upstream HTP IQ2_XXS kernel vs this fork's

Upstream merged IQ2_XXS HTP support on 2026-11-14 in `61dbf4922`
("Hexagon: add support for IQ2_XXS and NVFP4", PR #31954). This fork has its own IQ2_XXS
HTP kernel. The two target the same quant type, so they overlap.

**Read only. Nothing taken.**

## 1. Upstream's approach (`61dbf4922`)

Files: `htp/hvx-mm-kernels-tiled.h` (+363), `htp/hmx-mm-kernels-tiled.h` (+104),
`htp/iq2xxs-tables.h` (+54), `htp-repack.cpp` (+148), `matmul-ops.c` (+48).

Design, from the constants and the gather code:

- **Tile = 32 rows x 8 B of codes**, plus 4 fp16 `d` values at offset 256
  (`IQ2XXS_D_SLICE_OFF`). So the tile holds 4 sub-blocks of 8 rows.
- **Two halfword gathers per tile.** `iq2xxs_gather_issue` does a `Q6_W_vdeal_VVR` to split
  the tile into index words and sign/scale words, then issues two `Q6_vgather_ARMVh` calls
  into a per-tile scratch area. Lane `i` gets the codes of groups `2h` and `2h+1` in
  halfwords 0 and 1.
- **Scratch lives inside the tile**, at
  `tile + htp_tensor_get_tile_padded_size(HTP_TYPE_IQ2_XXS) - HTP_TENSOR_TILE_SCRATCH_IQ2_XXS`.
  So the gather results are written back into the tile's own padding.
- **Sign handling by bit prediction**, not by a table lookup:
  `IQ2XXS_SIGN_PRED_LO/HI` are byte-position masks, and the sign bits are extracted with
  shifts and masks rather than a `ksigns64` load.
- **Scale**: `db = d * (2 * scale + 1) / 8`, folded as `IQ2XXS_D_SCALE = 0.125f`.
- **NVFP4 came in the same commit**, so the commit is two features in one.

## 2. This fork's approach

Files: `htp/matmul-ops.c`, `htp/hvx-mm-kernels-tiled.h`.

- **Codebooks in VTCM**, copied once by `htp_iq2f_tables_copy`: `iq2xs_grid`, `iq2xxs_grid`,
  `ksigns64`, laid out after the IQ2_S grid inside `HTP_IQ2S_GRID_VTCM_RESERVE`.
- **Per-thread gather scratch** in VTCM (`IQ2S_SCRATCH_STRIDE = 1024`, one slot pair per
  group of a tile), shared between the IQ2_S and IQ2 family paths.
- **Sign handling by a 16-entry LUT** (`iq2_s_sign4_lut`) plus an arithmetic mask builder
  (`iq2_s_sign_mask_arith`), rather than bit prediction.
- **Four-column kernel** (the fork's own tuning), not the 32-row x 8 B tile.

## 3. The differences that matter

| aspect | upstream | fork |
| --- | --- | --- |
| tile | 32 rows x 8 B codes + 4 fp16 d | fork's own 4-column shape |
| gather scratch | inside the tile's padding | per-thread VTCM slot |
| codebook location | passed as a pointer per call | VTCM, copied once at init |
| sign decode | bit prediction with masks | 16-entry LUT + arithmetic mask |
| scale | `(2*scale+1)/8` folded to 0.125f | fork's own |
| scope | IQ2_XXS + NVFP4 in one commit | IQ2_XXS only |

**The VTCM-resident codebook is the fork's advantage.** Upstream passes the grid pointer per
call; the fork copies it into VTCM once, so the gather reads from VTCM instead of DDR. On a
phone with a slow memory path that should matter.

**Upstream's in-tile scratch is the cleaner idea.** It avoids a separate per-thread VTCM
reservation, which is what forces the fork's `_Static_assert` on
`HTP_IQ2S_GRID_VTCM_RESERVE`. Upstream's approach scales to more threads without growing the
reserve.

**Upstream's bit-prediction sign decode** avoids a LUT load per group. Whether that is faster
than the fork's LUT on this SoC is not measured.

## 4. Why this is a `skip` in the triage, not a `welcome`

Taking upstream's IQ2_XXS would **replace** the fork's kernel, not add to it. That is a
strategy decision, not a merge:

- The fork's kernel is tuned for this device (VTCM codebook, 4-column shape, gather batching
  from Phase 46-54).
- Upstream's is general and comes with NVFP4, which the 27B does not use.
- The fork's IQ2_XXS path has been through the Phase 58 bit-exactness work.

**Recommendation: do not take #31954 for IQ2_XXS.** Consider taking only the NVFP4 part if a
future model needs it. Revisit if the fork's kernel ever needs to be replaced wholesale.

## 5. What is not known

- No measurement of either kernel against the other on this device. The fork's Phase 58
  numbers (3.1x kernel speedup for iq3_xxs) are for a different type.
- Upstream's `htp-repack.cpp` (+148) implies a repack layout change. Whether that is
  compatible with the fork's `iq2-family-repack.h` is not checked.
- The `tests/test-backend-ops.cpp` change (+7) is upstream's test gating; the fork has its
  own test setup.
