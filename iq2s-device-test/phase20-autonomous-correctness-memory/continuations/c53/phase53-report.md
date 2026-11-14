# Phase 53 — 夜間自律（自由裁量モード）

対象: 2026-11-14 22:20 〜 2026-11-15 朝（所有者は約 12 時間不在）
作業 worktree: `D:\ghq\github.com\yuubinnkyoku\llama.cpp-c51`
ブランチ: `codex/c53-summary`（`codex/c52-quality` / `fc6e5198c` から作成）
push 許可: `codex/c53-summary` のみ

---

## §0 現在地の固定（開始時の実測、2026-11-14 22:24）

指示の「想定基準」と一致していることを実測で確認した。

| 項目 | 実測 |
| --- | --- |
| c51 worktree ブランチ | `codex/c52-quality` |
| HEAD | `fc6e5198c` |
| worktree | clean（未コミット差分なし） |
| メイン worktree `llama.cpp` | `codex/phase20-htp-correctness-memory` / `2717a9293`、**未コミット 9 ファイル**（指示どおり保護、一切触れていない） |
| 端末 | NX741J / Android 16 / uptime 8 日 21 時間（再起動なし） |
| ADB | USB `324753221196` と TCP `192.168.55.15:5555` の 2 経路。同じ端末なので以後 `-s 324753221196` を明示 |
| スケル | `288e6712df9c8201ef05c12ec5704968d4f71407992cff8e0f03aa002ee0861d`（Phase 50 D） |
| probe（開始時） | `aeaa1b62...`（Phase 52 終了時に戻した 32-step 版） |
| モデル | `/data/local/tmp/llama.cpp/models/Qwen3.8-27B-GSQ-RCO-IQ2_S.gguf` 9,259,510,912 B |
| skel / libs | `/data/local/tmp/c46-27b/libs` |
| 直前の報告書 | `continuations/c52/phase52-report.md`（§1〜§17） |

Skill `overnight-autonomy`（`SKILL.md` と `references/hexagon.md`）を読了。恒久的規則は
そのまま適用し、Phase 52 までの数値を Phase 53 の既定値として流用しない。

---

## §1 今夜の自由裁量の判断（すべてここに理由を残す）

指示は「測定上の縛りは撤廃、実機と repo の保護だけ守る」だった。以下は自分で決めた。

| 判断 | 内容 | 理由 |
| --- | --- | --- |
| **swap guard** | 262144 pages（1024 MiB）を**既定のまま**用意しつつ、Phase 53 の測定 run では `GUARD_PAGES=393216`（1536 MiB）を使った | 1024 MiB が「目安」と明示され裁量が認められたため。実測では 1 run あたりの pswpout 増分が 268k〜374k pages で、**1024 MiB では model load だけで毎回発動**した（§3 に全記録）。一方この端末の swap は ZRAM（`/proc/swaps` は権限で読めないが `SwapTotal 12287M`）で、発動時も MemAvailable は 10 GB 台、SwapFree も 10 GB 台、OOM なし。**停止線（OOM・不安定・DSP 異常）は緩めていない**。1536 MiB を超える発動が出たら、その run は測定不成立として扱い、条件を変える。 |
| **TVer** | 停止した | 2026-11-14 の所有者許可。§17 の共存手順は今回不要 |
| **CPU 単独測定** | 実行した（NGL=0） | Phase 52 で BLOCKED にしたのは「fast cap 窓での CPU 単独比」であって、CPU 側の絶対測定ではない。今夜は窓待ちをせず、cap を記録して測定した |
| **1 試験の長さ** | 最大 1800 s（測定 run）、持続試験は 3600 s まで | 180 秒制限の撤廃が明示されたため。`run53.sh` は `RUN_TIMEOUT` で可変 |
| **長時間試験の長さ** | 連続 decode 約 24 分 ×2 系列（HTP / CPU）を目標 | 30〜60 分の例示に対し、CPU 側は熱制限で打ち切りが早いと予想されたため、まず 24 分で 1 系列とし、余力があれば延長 |
| **上流取り込み** | しなかった。監視と報告のみ | 指示どおり |
| **repo** | `codex/c53-summary` のみに push。merge / force push / 過去コミット改変なし | 指示どおり |

---

## §2 ハーネスの変更（今夜作ったもの）

### §2.1 `continuations/c53/c53_decode_probe.cpp`（probe 第 3 世代）

Phase 52 の probe（`c52/decode_probe_64.cpp` = c27 の step 上限を 64 にしたもの）を基に、
**環境変数でだけ挙動を変える**上位版を作った。何も設定しなければ Phase 52 と同一の経路を
通るので、既知 hash の回帰がそのまま使える（§3 で確認）。

| 環境変数 | 既定 | 内容 |
| --- | --- | --- |
| `P53_STEPS_MAX` | 64 | STEPS 引数の上限。持続試験用に最大 4096 |
| `P53_N_CTX` | 128 | コンテキスト長。足りなければ自動引き上げ（`NCTX_AUTO` を出力） |
| `P53_N_BATCH` | 32 | 論理バッチ。**このビルドの llama.cpp は `llama_decode` 1 回のトークン数が n_batch 以下でないと assert で落ちる**ので、プロンプト長に合わせて自動引き上げ（`NBATCH_AUTO` を出力） |
| `P53_N_UBATCH` | 32 | micro batch |
| `P53_PROMPT_FILE` | なし | プロンプト本文のファイル |
| `P53_PROMPT_TOKENS` | 0 | プロンプトを**トークン数で**指定。本文を繰り返して長さを確保し、先頭 N トークンに切り詰める |
| `P53_PROMPT_LIST` | なし | prefill 掃引する長さの列（`32,64,128,256,512`）。**1 回の model load で複数長を測る**ため |
| `P53_SEGMENTS` | 1 | 「prefill → steps 回 decode」を S 回繰り返す。毎 segment KV を clear するので、長時間連続 decode を小さい n_ctx のまま走らせられる |
| `P53_TEMP` / `P53_TOP_K` / `P53_TOP_P` / `P53_SEED` | 0（=argmax） | サンプリング。0 のときは Phase 52 と同じ argmax 経路 |
| `P53_NO_LOGITS` | 未設定 | logits を書かない。500 step × 248320 float = 0.5 GB になるため長時間試験では必須 |

追加した出力行:

- `PREFILL seg %d tokens %d ms %.6f tokens_s %.6f` — prefill 時間
- `PREFILL_SWEEP len %d ms %.6f tokens_s %.6f status %d` — 掃引 1 点
- `SWEEP_WARM ms %.6f` — 掃引前の捨て打ち 1 decode（グラフ構築コストを掃引から除く）
- `TTFT_MS %.6f` — prefill 開始から最初のトークンが得られるまで
- `STEPT seg %d step %d t_ms %.3f` — 各 step の絶対時刻（tok/s 推移のため）
- `RESULT_SEG seg %d forwards %zu decode_ms %.6f tokens_s %.9f` — segment ごとの集計
- `GENERATED n %zu` と `<logits>.gen.txt` / `.gen.ids` — 自由生成の全文と ID 列

`PREFILL`/`DECODE`/`RESULT`/`GENERATION` の既存行は書式を変えていないので、Phase 50〜52 の
解析スクリプトがそのまま使える。

### §2.2 `run53.sh` / `run_split53.sh` / `p53_run.sh`

`run50.sh` / `run_split50.sh` / `p50_run.sh` の複製に 3 点だけ足した。

- `RUN_TIMEOUT`（既定 180）— `timeout -s TERM` の値を可変にした
- `GUARD_PAGES`（既定 262144）— swap guard の閾値を明示的に記録・変更できるようにした
- `p53_run.sh` は `P53_*` 環境を run ヘッダに書き出す。`TVer` を force-stop リストに加えた

安全側は変えていない: `MEM_MIN_KB=9500000` / `SWAP_MIN_KB=8000000` / `RES_MIN_PCT=95` /
`APPROVAL_TTL_S=900` / 2 回の prewarm / スケル同定 / 承認ファイルは Phase 50 のまま。

---

## §3 32-step 回帰（probe 第 3 世代の妥当性ゲート）

（結果を追記）

---

## §4 prefill / TTFT

（結果を追記）

---

## §5 自由生成サンプル

（結果を追記）

---

## §6 持続性評価

（結果を追記）

---

## §7 実用入口（llama-cli / llama-server）

（結果を追記）

---

## §8 品質補強（評価文 2 本の 32/64-step）

（結果を追記）

---

## §9 decode 性能の新規仮説

（結果を追記）

---

## §10 上流 Hexagon 関連変更の監視

（結果を追記）

---

## §11 Phase 46〜52 総括

（結果を追記）

---

## §12 判定

（結果を追記）
