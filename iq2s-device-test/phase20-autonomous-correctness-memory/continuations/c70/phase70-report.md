# Phase 70 - 採用 A/B: 11 GiB 窓が取れずナイトキューへ

## §0 結論

- 採用判断 A/B（NUB=8 非タイル vs NUB=32 タイル、prefill 512 + decode 32、split timing）を `gate with --min-mem 11` で試行したが、**MemAvailable が 11 GiB に届かず GATE_MEM_LOW（76）**。
- 前面アプリ（YouTube）が重く、cleanup 後でも 9.55 GB までしか回復しない。30 分待機の途中で MemAvailable は 7.2 GB まで低下したため、ロックを解放し、**A/B をナイトキューに投入**した（`D:\npu-gate\jobs\01-inkling-ab70.job`、`# npu-mem=11` `# project=inkling2`）。
- キューランナーは未起動。ユーザーが寝る前に `bash /d/npu-gate/npu-queue-runner.sh --watch > /d/npu-gate/runner.out 2>&1 &` で起動すれば、11 GiB 窓ができ次第 A/B が実行される。

## §1 試行の記録

| 時刻 | 状態 |
| --- | --- |
| 19:59:11 | gate free、MemAvailable 5.22 GB |
| 19:59:41 | cleanup（前面 YouTube を残す）→ 9.55 GB、`/data` 4.4 GB |
| 19:59:51 | A/B gate がロック取得、`--min-mem 11` で待機開始 |
| 20:10:06 | MemAvailable 7.28 GB（低下傾向） |
| 20:11:30 | MemAvailable 7.17 GB、11 GiB 到達見込みなし |
| 20:11:39 | 自ゲートを SIGTERM で解放（lock free、token none） |
| 20:11 | `01-inkling-ab70.job` を投入 |

- A/B スクリプト `ab70.sh` は prefill 512（NUB=8 m_chunk=0 / NUB=32 m_chunk=128、TWO_PHASE、NOLOGITS）、warmup decode、decode 32（NUB=8/32）を同一セッションで実行し、各 run の `sched_reserve: graph (pp bs=...)` と HTP buffer、swap を記録する。
- 採用条件は Phase 70 の通り: hash `65dd1c56…` bit 一致＋prefill/decode A/B＋split timing＋**測定として有効（swap 1.5 GiB 以内）**。prefill 512 の Phase 69 実測は swap 2.88 GiB（TWO_PHASE で prefill 中 guard 不作動）であり、A/B の prefill が規則内に収まるかを確認する必要がある。

## §2 端末状態（20:11）

- uptime 1 日 19 時間 38 分（追加再起動なし）、MemAvailable 7.2 GB、SwapFree 10.4 GB、`/data` 空き 4.4 GB、残存 probe なし、gate lock free。

## §3 成果物

- `c66/ab70.sh`（A/B スクリプト）、`D:\npu-gate\jobs\01-inkling-ab70.job`（ナイトキュー）。
- 本 report（c70）。
