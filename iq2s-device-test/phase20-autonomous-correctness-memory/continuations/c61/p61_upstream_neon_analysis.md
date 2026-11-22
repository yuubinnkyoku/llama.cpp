# Phase 61 - upstream NEON tiled kernel vs this fork's kernel

Source: [hardwiredagent/silicon-forge-arm-neon-study](https://github.com/hardwiredagent/silicon-forge-arm-neon-study)
(MIT, written 2026-11-11 by an AI agent on Orbofi). Read only, nothing taken.

## 1. The two kernels side by side

| aspect | upstream study | this fork |
| --- | --- | --- |
| gate | `__aarch64__ && __ARM_FEATURE_DOTPROD` | `__ARM_NEON` (dotprod checked by the dispatch chain) |
| src1 layout | **interleaved in place** by `tiled_repack_src1` | natural layout, `tiled_repack_src1` stays a no-op |
| dot instruction | `vdotq_laneq_s32` (SDOT by lane) | `vdotq_s32` |
| tile shape | 4 src0 rows x 16 src1 rows per pass | 1 src0 row x 16 src1 rows |
| accumulators | `float32x4_t acc[16][4]` = 64 registers | `int32x4_t acc[16]` = 16 registers |
| reduction | **none** (lane = row, no horizontal sum) | `vaddvq_s32` per column at the end |
| scale application | per subblock, float, inside the loop | once at the end, int32 then float |
| min term | float, per subblock, `vmlaq`/`vfmsq` | int32 scalar accumulator |
| bias handling | `vsubq_u8` on load (same idea) | `vsubq_s8` on load (same idea) |

## 2. What is genuinely different

**The interleaved src1 layout is the big one.** The study repacks the activation band so that
for one 4-byte k-group the 16 rows' chunks are contiguous. Then one `vdotq_laneq_s32` adds four
partial sums for four different rows, and **nothing is reduced afterwards** - the lane index
carries the row identity. This fork keeps the natural layout and does a `vaddvq_s32` horizontal
sum per column at the end.

**The 4x16 tile shape follows from that.** With the interleaved layout you can hold 4 weight
rows x 16 prompt rows in 64 float registers and never spill. This fork's 1x16 shape uses 16
int32 registers and needs the horizontal reduction.

**Float accumulation vs int32.** The study accumulates in float per subblock. This fork keeps
int32 exact until the very end. The fork's choice is numerically tighter (no per-subblock
rounding) but the study's is what makes the 4x16 shape fit in registers.

## 3. Why 6.5x there and 2.14x here

Three separate factors, in order of likely size:

1. **The baseline is different.** The study's "before" is stock llama.cpp v0.9.0 on Arm, where
   the tiled routine is compiled out entirely and the IQ formats "fall back to a path that
   unpacks the weights again for every dot product". This fork's "before" is the same stock
   path, so the baselines are comparable in kind. But the study measures **IQ3_M**, which is
   the worst case for the stock path. This fork measured the whole 27B model, where most
   tensors are not IQ3_M and the CPU share is only part of the wall.
2. **The SoC is different.** Neoverse N1 is a server core with a large L2 and no thermal
   limit. The 8 Elite Gen 5 is a phone SoC with mixed cores, a small L2 per cluster and a
   thermal cap that was sitting at 1.78 GHz for most of Phase 60's runs.
3. **The kernel is different.** The interleaved layout removes the horizontal reduction and
   lets 4 rows share each SDOT. That is a real advantage this fork does not have.

**The honest split**: factor 1 and 2 are probably most of the gap. Factor 3 is real but the
study itself notes its Arm figures are "still well below the x86 reference" and that what
bounds them "is not profiled".

## 4. Candidates this fork could adopt (NOT taken, Phase 62 material)

Ordered by expected value / risk:

1. **Interleaved src1 + `vdotq_laneq_s32` + 4x16 tile.** The main idea. Removes the horizontal
   reduction and quadruples the work per SDOT. Risk: needs a real `tiled_repack_src1` for Arm,
   which changes the activation tile layout and therefore every kernel that reads it. Must be
   gated so the scalar/AVX paths keep the natural layout.
2. **Float accumulation per subblock.** Would let the 4x16 shape fit. Costs the int32 exactness,
   so it needs the quality gate to prove it is still fine.
3. **`vdotq_laneq_s32` without the interleave.** Even with the natural layout, using the
   by-lane form on the src0 side could cut some register pressure. Small win, low risk.
4. **Check whether the 16-register int32 accumulator array spills.** This fork uses
   `int32x4_t acc[16]` = 64 bytes of vector state plus `acc_min[16]`. A spill would show up in
   the disassembly; worth checking before any redesign.

## 5. What the study did NOT do (so this fork should not assume it)

- No phone was measured. The study says so explicitly.
- Only Qwen2.5 1.5B, 4 threads, 512-token prompt.
- No `test-backend-ops`, no sanitizers, no real-prompt output comparison.
- 128-bit NEON only; no SVE, no I8MM.
- Not submitted upstream (the study cites llama.cpp's AI-contribution rules).

## 6. Status

Analysis only. **No code was taken.** The study is MIT licensed, so taking it would be legal,
but the Phase 61 brief forbids intake on this branch and the fork's own policy is to do
upstream intake on a dedicated branch with a regression gate.
