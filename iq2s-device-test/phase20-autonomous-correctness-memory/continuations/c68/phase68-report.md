# Phase 68 - VTCM canary: タイル経路の領域外書き込みを検出

## §0 結論

- **VTCM/DMA canary を実装し、タイル経路が VTCM の未使用領域（act_raw の tail）を上書きしていることを検出した。** DSP は `VTCM-TOO-SMALL` を返して停止（ホストは `dsp-error VTCM-TOO-SMALL`）、端末は再起動せず。Phase 65 の prefill 512 再起動と同系統のメモリ安全性の問題を、診断で再現できた形。
- 使用行の数値は Phase 67 で非タイルと bit 一致済み。**prefill 512 は canary PASS まで実行しない**（Phase 68 の残作業）。
- 診断用に `GGML_HEXAGON_ID_PROBE_TOKENS`（配置 probe の token 数、既定 8）を追加した。ただし tiling 有効な host lib では probe が常に VTCM に収まるため、既定配置を止める用途には使えない（placement は override で制御する）。

## §1 canary の実装

DSP `hvx_mm_matmul_id` のタイルループに、以下を追加した（`matmul-ops.c`）:

- レイアウト末尾（`base + L.total_bytes`、`+128 <= ctx->vtcm_size` のとき）に 0xA5 の 128 バイト番兵。
- 各 chunk で `act_raw` の未使用 tail（`m_rows * act_raw_stride` 以降）と `src1` の未使用 tail に 0xA5 を書き、転送・量子化・worker の後に検証。
- 破損時は領域ごとに異なる status を返す: act_raw tail → `VTCM-TOO-SMALL`、src1 tail → `INVAL-PARAMS`、レイアウト末尾 → `NO-SUPPORT`。FARF に m0/rows/各 guard の結果を記録。

ビルド:
- canary 入り skel `60822a1a5bf7a66ec4b7a0346de5522b864fc3f26169f692d70b93e9f8c5bc9e`（status 共通版）、`e8450acb78567799ce6f6232f03f6b088dcfd4f0d1312c49710e959be8cd8df4`（status 分離版）
- host（probe token ノブ付き）`fac8335e6cae19f17fbd6a1e0326789290ce7ccc1f3d5ceafb84a26515e81b99`
- 計測用 host（`GGML_HEXAGON_ID_MCHUNK` ノブのみ）`76848910c092d929b7e72184273bfee7a2157a1a5ebab9a2036594b0c801002f`

## §2 実行結果

5-token プロンプト、m_chunk=32（40 行 → 2 chunk: 32+8）、24 層 down override（HTP buffer 5,531.11 MiB）:

| 試行 | skel | 結果 |
| --- | --- | --- |
| canary 共通 status | `60822a1a` | rc=134、`dsp-error INTERNAL-ERROR`（canary 検出） |
| canary status 分離 | `e8450acb` | rc=134、**`dsp-error VTCM-TOO-SMALL` = act_raw tail の破損** |

- 破損検出は最初の部分チャンク（chunk 2、8 行）で発生。使用行（0..7）の出力は正しいが、未使用 tail に書き込みが及んでいる。
- 原因候補: (a) 量子化 worker が src1 領域を越えて隣接の act_raw tail に書く、(b) DMA 転送が 8 行を超えて書く、(c) レイアウトの端数アラインメントの不一致。次は writer を特定する（act_raw の tail 番兵を src1 側だけに絞る、m_chunk=40 の単一 chunk では発生しないことの確認、量子化カーネルの row サイズと layout の src1_row_size の一致確認）。
- 端末は再起動していない（uptime 23 時間超）。安全側の停止が機能した。

## §3 状態

- 端末: uptime 23 時間超（00:32 の再起動以来、追加再起動なし）、MemAvailable 8.3〜10.1 GB、`/data` 空き 5.6 GB、残存 probe なし。
- 常用構成は不変（down HTP 既定 23 層、NUB=8、hash `65dd1c56…`）。NUB=32 は canary PASS まで禁止のまま。
- 変数メモ: 今回の診断で「tiling 有効 host lib では配置 probe が常に成功する」ことが判明。既定配置を止めたい場合は override で明示的に CPU を指定する。

## §4 次回

1. act_raw tail の writer 特定（上記 (a)-(c) の切り分け）。
2. 修正後 canary PASS を確認 → prefill 512（11 GB＋warmup）。
3. 採用判断（hash `65dd1c56…` bit 一致＋prefill/decode A/B＋split timing）→ NUB=32 解禁の是非。

## §5 成果物

- 本 report（c68）、`c66/mtile-diag/libggml-htp-v81-canary*.so`、`c66/canary_diag*.sh`、`c66/cleanup.sh` 更新。
- ソース: `matmul-ops.c`（canary）、`ggml-hexagon.cpp`（probe token ノブ）は診断用で、常用 lib には未適用のまま端末の既定 lib は安全版（`c87765f7`/`76dd89e3`）。
