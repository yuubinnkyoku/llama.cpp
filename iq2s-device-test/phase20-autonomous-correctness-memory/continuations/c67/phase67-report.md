# Phase 67 - dotprod 判断（不採用）と M タイル診断の準備

## §0 結論

- **dotprod/i8mm CPU variant は不採用。** 同一セッションの IQ2_XXS マイクロベンチ（test-backend-ops perf、n=8）では有意差なし（base 平均 57.2 GFLOPS、dot 平均 60.9 GFLOPS、ペア差 t≈1.27）。さらに dot は logit が変わる（27.2549171 vs 27.2587242）ため、凍結 hash `65dd1c56…` と一致しない。§1 の採用条件（hash 一致＋有意な速度差）を満たさない。
- **常用構成は不変**: down HTP 既定 23 層、NUB=8、FA_SELECT=1、DMA64=0、host lib `c87765f7`、skel `76dd89e3`、hash `65dd1c56…`。
- M タイル診断の発射台を用意した: 診断用 lib（skel `cced2257`＝タイル化 DSP、host `76848910`＝`GGML_HEXAGON_ID_MCHUNK` で m_chunk を強制）をビルドし、5-token プロンプト・m_chunk=32 の 1 回実行（最初の実 2 チャンク）を 11 GB ゲート付きでキューに投入した。

## §1 dotprod マイクロベンチ（同一セッション、交互 8 ペア）

手法: `test-backend-ops perf -b CPU -o MUL_MAT -p 'type_a=iq2_xxs.*n=8,'` を、CPU lib だけ差し替えた 2 つのディレクトリ（`variant-base` `ac7cc94c` / `variant-dot` `a3b7d4e7`）で交互に 8 回ずつ実行。テスト側の内部反復（63〜81 run）で平均化済み。

| pair | base GFLOPS | dot GFLOPS | diff |
| --- | --- | --- | --- |
| 1 | 65.05 | 60.72 | -4.33 |
| 2 | 54.54 | 61.02 | +6.48 |
| 3 | 44.00 | 60.52 | +16.52 |
| 4 | 54.66 | 60.95 | +6.29 |
| 5 | 54.72 | 61.01 | +6.29 |
| 6 | 65.17 | 60.50 | -4.67 |
| 7 | 64.79 | 61.06 | -3.73 |
| 8 | 54.63 | 60.99 | +6.36 |

- base 平均 57.20、dot 平均 60.85。ペア差の平均 +3.65、t ≈ 1.27（有意でない）。
- base は 44〜65 GFLOPS とばらつく（周波数/熱状態）。dot は 60.5〜61.1 と安定。dot が常に速いわけではなく、**採用に足る再現性のある差はない**。
- 数値: A/B v3 で最初の token の logit が base 27.2587242 / dot 27.2549171 と差。HTP カーネルでは Phase 65 で hash 再凍結を受け入れたが、CPU variant は指示の採用条件が「基準 hash 一致」であり、満たさない。**このため不採用で確定。**

## §2 M タイル診断の準備

- タイル化コードを再適用（Phase 65 のリバートから復元）:
  - host: `ggml_hexagon_precompute_hvx_mm_params` の ID 分岐で VTCM に収まる最大 m_chunk を探索し `kparams->m_chunk` に設定。診断用に `GGML_HEXAGON_ID_MCHUNK` で強制縮小。
  - DSP: `hvx_mm_matmul_id` のチャンクループ、チャンクごとの activation 転送・量子化、`hvx_mm_id` の行フィルタ。
- ビルド成果物: skel `cced22579c1205576350d98b280724d76d159665132e081cc65c25a3a6f6a529`、host `76848910c092d929b7e72184273bfee7a2157a1a5ebab9a2036594b0c801002f`（`c66/mtile-diag/`）。
- 診断 1: 5-token プロンプト（40 行）＋ `GGML_HEXAGON_ID_MCHUNK=32` → 2 チャンク（32+8）の最小実タイル実行。prefill 512 は診断 PASS まで禁止。
- 診断 2（PASS 後）: チャンク境界行の CPU 一致検証（5-token 出力の logit を CPU 参照と比較、最初の token 182416）。
- 診断 3（PASS 後）: VTCM/DMA canary。DSP 側に境界センチネルを置き、チャンクごとに破壊がないことを確認する実装を追加する。
- 実行は `gate with --project inkling2 --min-mem 11` でキュー投入済み（他プロジェクトのロックと 11 GB を待つ）。

## §3 端末・ゲート状態（21:00〜21:40）

- 21:00 の `gate status`: free、MemAvailable 4 GB。
- 21:02〜21:32 は haruvoice がロック保持。待機中に USB adb の serial が `324753221196` から `192.168.55.15:5555` に変化（スクリプトは `adb` 単体指定に修正）。
- test-backend-ops は Android 用に手動クロスビルド（`aarch64-linux-android29-clang++`、1.44 MB）。HTP backend のロードには `ADSP_LIBRARY_PATH` が必要。
- 診断キュー投入時点の MemAvailable は 4〜5 GB で、11 GB 待ち。GATE_MEM_LOW の場合は再投入する。

## §4 成果物

- `c66/test-backend-ops`（Android / arm64、perf モード）
- `c66/bench_pairs.sh`（同一セッション交互ペア測定）
- `c66/mtile-diag/`（診断 lib 2 種）
- `c66/mtile_diag1.sh`（診断 1 スクリプト）
- コミットは本 report とともに push する。

## §5 追記（22:15）: M タイル診断の結果 — 3 構成で bit 一致、prefill 128 まで無事故

すべて `gate with --project inkling2` 経由。診断は 6 層 override ではなく **24 層 down override（HTP buffer 5,531.11 MiB）** で実施した（診断 host lib が 8-token cap の既定配置で 23 層を載せるため、6 層 override は実質 24 層になる。配置の実態は buffer サイズで確認）。

| 診断 | 構成 | 結果 |
| --- | --- | --- |
| 1 | 5-token（40 行）、m_chunk=32 → 2 chunk（32+8） | rc=0、first token 182416、logit 27.2261276、swap 1.37 GiB |
| 2 | 同 5-token、非タイル vs タイル（m_chunk=32） | **logits sha256 一致** `74f01fcacd…d3d2d239`、swap 0.72 / 0.76 GiB |
| 3 | 17-token（136 行）、m_chunk=128 → 2 chunk（128+8） | **logits 一致** `2c65522086…5a019a51`、first token 151435、swap 0.86 / 0.78 GiB |
| 4 | 32-token（256 行）、m_chunk=128 → 2 chunk（128+128） | rc=0、first token 159033（CPU 参照も 159033、logit 差 0.098）、swap 0.75 GiB |
| 5 | prefill 128 token（32-token ubatch × 4）、m_chunk=128 | prefill 21.6 s（5.92 t/s）、first token 149890、**crash なし**、swap 1.55 GiB で guard 停止（測定は無効、経路は生存） |

- **チャンク境界の数値は非タイルと bit 一致**（診断 2/3）。タイル化の行分割・転送・量子化はこの範囲で正しい。
- 診断 4 の 256 行は Phase 65 の prefill 512 で再起動したときの ubatch 形状そのもの。今回は 2 chunk で完走し、CPU 参照と first token 一致。
- 診断 5 で tiled prefill 128 token（4 ubatch）が完走（crash なし）。swap 1.55 GiB は 10.7 GB 開始時の page cache churn で、測定としては無効。
- **一連の診断で端末再起動なし**（uptime 21 時間 44 分、00:32 の再起動以来）。
- 残る工程: ③ VTCM/DMA canary（DSP 側センチネル）、④ 11 GB + warmup での prefill 512（Phase 65 の再起動条件）。canary 実装は次回。

## §6 追加の成果物

- `c66/mtile_diag2.sh`〜`mtile_diag6.sh`（診断 1〜5）
- `c66/clean_runs.sh`（古い logits の削除。`runs/` 989 MB → 392 MB、/data 4.3 → 4.9 GB）
- 診断 lib: skel `cced2257…`、host `76848910…`（`c66/mtile-diag/`、端末では `variant-diag/`）
