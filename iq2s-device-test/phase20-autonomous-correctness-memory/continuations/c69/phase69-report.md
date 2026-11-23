# Phase 69 - canary 誤検出の解消、bit 一致回帰、prefill 512 完走

## §0 結論

- Phase 68 の `VTCM-TOO-SMALL` は **canary の配置ミスによる誤検出**だった。VTCM レイアウトは Group C（act_raw）が Group B（src0）と**意図的にオーバーラップ**する。act_raw の tail を番兵にしたため、weight prefetch の正当な書き込みを破損と誤認していた。
- canary を **src1 tail + レイアウト末尾**に修正（`f7def94095d19ed7e6e8c3786a2a477b4ed44b57bf786d93597ab88dba47e126`）。修正後:
  - **bit 一致回帰 PASS**: 5-token（40 行、32+8 chunk）`0ba39d56…`、17-token（136 行、128+8 chunk）`2c655220…` が非タイルと完全一致。256 行（128+128 chunk）も canary 付きで完走、first token 159033（CPU 参照と一致）。
  - **prefill 512 が完走**（Phase 65 の再起動条件）: 512 token を 84,104 ms（6.09 t/s）で処理、first token 153697、decode 1 step 468.8 ms、dsp-error 0、**端末再起動なし**。
- 凍結 hash の維持も確認: 診断 skel + 安全 host（`c87765f7`）の replay 4 token が **`65dd1c563da7f1fb24dc53a362d4acff4afb184bd98f04df96f4d513d5ca661d`**、HTP buffer 5,348.86 MiB（既定 23 層）。

## §1 誤検出の原因と修正

- レイアウト（`htp_mm_hvx_vtcm_layout_build` の ID 分岐）は次の 3 グループ:
  - Group A: src1（量子化 activation）
  - Group B: src0（weight prefetch）と dst。off_group_a から開始
  - Group C: act_raw。**Group B と同じ off_group_a から開始（時分割で再利用）**
- Phase 68 は act_raw の未使用 tail `[m_rows*stride, act_raw_bytes)` を番兵にしたが、この範囲は src0（weight tile）が実際に使う。canary が正当なアクセスを検出していた。
- 修正: act_raw tail を削除し、**src1 tail とレイアウト末尾のみ**を番兵に。status は src1 → `INVAL-PARAMS`、末尾 → `NO-SUPPORT`。
- 診断 host の `GGML_HEXAGON_ID_MCHUNK` と `GGML_HEXAGON_ID_PROBE_TOKENS`（既定 8）はそのまま。

## §2 回帰と prefill 512（`gate with --min-mem 11`、開始 MemAvailable 11.64 GB）

| run | 構成 | 結果 |
| --- | --- | --- |
| hash | 診断 skel + 安全 host、replay 4 token、NUB=8 | **`65dd1c56…`**、HTP 5,348.86 MiB、first token 182416 |
| a | PT=5、m_chunk=0（非タイル） | rc=0、`0ba39d56…` |
| b | PT=5、m_chunk=32（2 chunk） | rc=0、**`0ba39d56…`（a と一致）** |
| c | PT=17、m_chunk=0 | rc=0、`2c655220…` |
| d | PT=17、m_chunk=128（2 chunk） | rc=0、**`2c655220…`（c と一致）** |
| e | PT=32（256 行）、m_chunk=128 | rc=0、first token 159033、canary error 0 |
| **prefill 512** | NUB=32、m_chunk=128、TWO_PHASE=1、NOLOGITS=1 | **rc=0**、512 token 84,104 ms（**6.09 t/s**）、first token 153697、decode 468.8 ms/step、HTP 5,531.11 MiB、dsp-error 0、**再起動なし** |

- prefill 512 の swap は合計 2.88 GiB（TWO_PHASE のため prefill 中の guard は作動しない）。**測定値としては 1.5 GiB 規則を超えるため無効**だが、今回の目的（crash しないことの確認）は達成。
- e の 256 行は Phase 65 の再起動条件の ubatch 形状。canary 付きで完走し、CPU 参照 first token 159033 と一致。
- 端末は再起動なし（uptime 1 日 1 時間超）。canary PASS 後の prefill 512 は安全に完走した。

## §3 状態・次工程

- 端末（01:50、ゲート経由）: uptime 1 日 1 時間 49 分、MemAvailable 11.99 GB、SwapFree 10.45 GB、`/data` 空き 5.4 GB、残存 probe なし、gate lock free。
- 診断 lib: skel `f7def940…`（canary 最終）、診断 host `76848910…`、安全 host `c87765f7…`、安全 skel `76dd89e3…`。
- 次工程（採用判断）:
  1. prefill 512 と decode の A/B（NUB=8 非タイル vs NUB=32 タイル）を同一セッションで実施。prefill 512 は warmup を前置して swap 1.5 GiB 以内で測る。
  2. split timing 記録（graph の pp/tg splits、HTP/DSP の所要）。
  3. 基準 hash `65dd1c56…` の bit 一致（診断 skel で確認済み）と合わせて採用を判定。
  4. 採用時は NUB=32 解禁判断（Phase 65 実績: CPU buffer 49.24 MiB、decode 472.8 ms/step、hash 一致 2 回）。
- 修正版は `GGML_HEXAGON_ID_MCHUNK` ノブ裏のまま。既定経路・常用構成・凍結 hash・品質ゲートは不変。

## §4 成果物

- 本 report（c69）、`c66/validate_canary.sh`、`c66/regress_canary.sh`、`c66/compare_prompts.sh` 等の調査スクリプト。
- ソース: `matmul-ops.c` の canary 配置修正。診断 host の probe token ノブ。
