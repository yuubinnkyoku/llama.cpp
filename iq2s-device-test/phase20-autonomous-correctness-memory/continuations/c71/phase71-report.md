# Phase 71 - 採用 A/B: ナイトキュー実行も rc=76（メモリ窓なし）、再投入

## §0 結論

- ナイトキューで A/B（`01-inkling-ab70.job`、`# npu-mem=11`）を実行したが、**rc=76 GATE_MEM_LOW**。1810 秒（30 分）の待機中に MemAvailable は 9.4 GB から 6.1 GB へ低下し、11 GiB に届かなかった。**測定は未実施（BLOCKED）**。
- その後 haruvoice がロックを取得して run を開始したため、当面のメモリ窓は見込めない。ジョブは `failed/` から `jobs/` へ**再投入済み**。ランナーは STOP で停止済み（STOP ファイルは削除、次回起動可能）。
- 次回は端末がアイドルになる時間帯（就寝前のランナー起動）に同じジョブを流す。11 GiB 窓ができれば prefill/decode A/B と split timing が取得できる。

## §1 実行記録

| 時刻 | 状態 |
| --- | --- |
| 22:00:29 | 前面 launcher、MemAvailable 5.42 GB、SwapFree 6.82 GB、`/data` 3.6 GB |
| 22:00:49 | cleanup → 9.64 GB、SwapFree 10.65 GB |
| 22:01:30 | ランナー起動（`runner-launch.sh` 経由、--watch） |
| 22:01:31 | ジョブ開始、gate が `--min-mem 11` で待機（lock 保持） |
| 22:06:43 | MemAvailable 7.86 GB（低下） |
| 22:16:57 | 7.40 GB |
| 22:25:03 | 6.13 GB |
| 22:31:41 | **rc=76 で failed/**（last の MemAvailable は 3 GB 台、haruvoice の run 開始後） |
| 22:33:42 | STOP でランナー終了、ジョブを `jobs/` へ再投入、STOP ファイル削除 |

- gate status（22:33:52）: `lock: HELD project=haruvoice pid=1752 since=22:31:43`、`queue: 1 jobs`、`mem: 3 GB`。
- 端末: uptime 1 日 21 時間 57 分（追加再起動なし）、`/data` 空き 3.6 GB（減少傾向、logits の追加削除が必要）、残存 probe なし。

## §2 次回

1. 端末がアイドルな時間帯にランナーを起動（`bash /d/npu-gate/npu-queue-runner.sh --watch > /d/npu-gate/runner.out 2>&1 &`）。ジョブは `jobs/01-inkling-ab70.job` に投入済み。
2. 11 GiB 窓ができ次第、A/B（prefill 512 と decode 32、NUB=8 非タイル vs NUB=32 タイル）が実行され、`done/` に結果が出る。
3. レポートには hash bit 一致・prefill/decode 値・split timing・HTP buffer・swap・有効性（swap 1.5 GiB 内）を記載する。prefill 512 の swap 2.88 GiB（Phase 69）が構造的に超える場合は所有者判断へ。

## §3 成果物

- 本 report（c71）。`D:\npu-gate\jobs\01-inkling-ab70.job`（再投入済み）。`runner-launch.sh`（ランナー起動用）。
