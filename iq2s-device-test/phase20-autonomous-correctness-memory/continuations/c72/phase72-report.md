# Phase 72 - 採用 A/B 成立、NUB=32 + M タイルを常用へ移行

## §0 結論

- **採用 A/B が成立した**（02:22-02:25、`gate with` 1 回、cleanup 直後に同一ゲート内で実行、MemAvailable 11.56 GB で開始）。
- 結果: prefill 512 は NUB=32 タイルが **8.26 t/s（61,986 ms）** で NUB=8 非タイル **6.45 t/s（79,332 ms）** より **+28%**、しかも swap は **0.40 GiB（NUB=32）/ 1.29 GiB（NUB=8）** で両方 1.5 GiB 以内（有効測定）。decode 32 は NUB=32 **503.1 ms/step** vs NUB=8 **572.5 ms/step** で **-12%**、swap 0.07 / 0 GiB（有効）。
- **採用条件をすべて満たした**: ①基準 hash `65dd1c56…` bit 一致（NUB=32 採用構成で再確認）、②prefill/decode A/B、③split timing（pp bs 8 vs 32、splits 1305/652 共通）、④swap 1.5 GiB 以内の有効測定。
- **常用構成を NUB=32 + M タイル（auto m_chunk）へ更新**。新標準 lib: host `fac8335e…`、skel `f7def940…`（canary 付き）。旧標準（host `c87765f7…`、skel `76dd89e3…`）は端末の `libs-prev/` にバックアップ。
- 更新後の検証: replay NUB=32 で **`65dd1c56…`**（HTP buffer 5,348.86 MiB、first token 182416、swap 0.06 GiB）、decode 32-step **377.6 ms/step**（prefill 5-token 868 ms）。基準 hash は不変。

## §1 採用 A/B の詳細（02:22-02:25）

| run | 構成 | 結果 |
| --- | --- | --- |
| p8 | prefill 512、NUB=8、m_chunk=0（非タイル） | 79,332 ms（**6.45 t/s**）、decode 1 step 1,195.6 ms、swap 338,919 pages = **1.29 GiB**、graph pp bs=8 |
| p32 | prefill 512、NUB=32、m_chunk=128（タイル） | 61,986 ms（**8.26 t/s**）、decode 271.9 ms、swap 105,311 pages = **0.40 GiB**、graph pp bs=32 |
| d8 | decode 32、NUB=8 | **572.5 ms/step**、swap 0 |
| d32 | decode 32、NUB=32 | **503.1 ms/step**、swap 18,632 pages = 0.07 GiB |

- HTP buffer は全 run 5,348.86 MiB（既定 23 層）で配置実態を確認。
- split timing: `sched_reserve: graph (pp bs=8/32, tg bs=1): nodes 11826 / 11826, splits 1305 / 652`。splits は同一、pp bs のみ差。
- 品質: replay hash `65dd1c56…` と first token 182416 で不変。タイル経路は Phase 67/69 で bit 一致検証済み。

## §2 常用構成の更新

- 新標準: `libs/libggml-hexagon.so` = `fac8335e6cae19f17fbd6a1e0326789290ce7ccc1f3d5ceafb84a26515e81b99`、`libs/libggml-htp-v81.so` = `f7def94095d19ed7e6e8c3786a2a477b4ed44b57bf786d93597ab88dba47e126`
- 旧標準: host `c87765f7…`、skel `76dd89e3…`（端末 `libs-prev/` に保存）
- 起動環境: NGL=48、**NUB=32**、FA_SELECT=1、DMA64=0、SMALLMM=0、override なし（既定 23 層）。
- 基準値（更新後、02:26 検証）: replay hash `65dd1c56…`、HTP buffer 5,348.86 MiB、decode 32-step **377.6 ms/step**、prefill 5-token 868 ms。prefill 512 の基準は A/B の p32 = 61,986 ms（8.26 t/s、swap 0.40 GiB）。

## §3 端末・ゲート状態

- 02:26 検証時: MemAvailable 11.6 GB、SwapFree 10.8 GB、`/data` 空き 3.1 GB、uptime 3 日（追加再起動なし）。
- gate: A/B と検証は `gate with --project inkling2 --min-mem 11`（cleanup 直後の瞬間窓を利用）。待機型は Phase 70/71 の通り機能しないため、cleanup と A/B を同一ゲート内で実行する方式で成功した。
- `/data` が 3.1 GB まで減少。`clean_runs2.sh` で old logits を削除済みだが、次回も継続監視。

## §4 成果物

- 本 report（c72）。`c66/cleanup_and_run.sh`（cleanup + A/B 一括ゲート）、`c66/ab70.sh`（serial 自動解決）、`c66/hash32.sh`、`c66/install_and_verify.sh`、`c72-libs/`（新標準 lib のコピー）。
- コミットは本 report とスクリプトを push する。

## §5 追記（02:40）: NUB=32 常用構成の検証

新標準 lib（host `fac8335e…`、skel `f7def940…`）での検証:

| 項目 | 値 |
| --- | --- |
| replay 4 token（NUB=32） | **`65dd1c56…` 一致**、HTP buffer 5,348.86 MiB、first token 182416、swap 0.06 GiB |
| decode 32-step（NUB=32、default） | **377.6 ms/step**（prefill 5-token 868 ms）、swap 0.50 GiB |
| prefill 512（NUB=32、A/B の p32） | 61,986 ms（**8.26 t/s**）、swap 0.40 GiB |
| 64-step 品質ゲート（NUB=32） | CPU 参照 64 step **434.1 ms/step**（有効、swap 0）。HTP 64 step は 11.4 GB 開始で swap 1.58 GiB となり guard 停止（**無効**）。cleanup 後の再試行は MemAvailable 11,468,844 kB（閾値まで 65 MB 不足）で LOW。 |

- 64-step 品質ゲートは NUB=8 で Phase 66 に PASS 済み。NUB=32 の HTP 64-step はメモリ窓が取れ次第の追加確認（カーネルは Phase 67/69 で bit 一致検証済みのため必須ではない）。
- CPU 参照の 434.1 ms/step は同一夜間の冷却状態での値。絶対値は熱状態で変動するため、A/B 判定には使わない（同一セッション比較のみ）。

## §6 追記（02:45）: 次工程の候補

- host 側 guard 解除（mix() の CONT 3D/F32 3D MUL_MAT、PAD の DMA64 extended mapping）は、配置が変わり logits が変わるため**凍結 hash `65dd1c56…` と bit 一致しない**。実施には Phase 65 と同様の hash 再凍結の所有者判断が必要。
