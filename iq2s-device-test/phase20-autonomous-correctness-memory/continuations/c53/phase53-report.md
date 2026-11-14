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

Phase 52 と同じ 32-step・`replay32.txt`（32 トークン）・skel `288e6712` で、Phase 52 の
既知 hash と一致するかを確認した。**両側とも一致した。**

| 側 | 条件 | exit | sha256(logits.f32) | 既知値 |
| --- | --- | --- | --- | --- |
| HTP (NGL=16, SPLIT=0) | `gate32htp` | 0 | `b21ca4bf1da8a2ef3bc48f717ee79f13acc17ced0783141211fc7178f0c2e160` | **一致** |
| CPU (NGL=0) | `gate32cpu2` | 0 | `44cd99ce376d5ce627ebbd4dc1d6cb90e83ea7822371f13a4874f921262295` | **一致** |

したがって「環境変数を何も設定しなければ Phase 52 と同じ経路を通る」という設計は成立している。
以降の Phase 53 の数値はすべてこの probe（`482032c3...`）で取った。

### §3.1 ゲートに至るまでの失敗（除外しない）

| 時刻 | run | 結果 | 原因 |
| --- | --- | --- | --- |
| 22:39 | `reg32htp53b` | exit 6・logits なし | swap guard 発動（`guard stopped=1`、delta 301326 pages）。**gurad 262144 のまま `p50_run.sh` を使った**ため |
| 22:45 | `reg32htp53d` | exit 6・logits 途中まで → hash `274f06fd...`（**不一致**） | 同じく guard 発動。delta 268115 pages。**この hash は不完全なファイルのもので、回帰失敗ではない** |
| 23:02 | `gate32cpu` | exit 6・20/32 step で停止 | guard 発動（delta 294597）。`GUARD_PAGES` を `run53.sh` に入れたつもりが、**push 先パスを間違えて端末の `run53.sh` が旧版のまま**だった（`build/p53/run53.sh` ではなく worktree 直下を指定していた） |
| 23:04 | `gate32cpu2` | exit 0・**一致** | 上を修正して成功 |

教訓: 不完全な logits ファイルの hash は「回帰失敗」と誤読しやすい。`exit.txt` が 0 でない run の
hash は比較に使わない、を Phase 53 の規則とした。

### §3.2 同一 cap での CPU 単独 vs HTP（参考、単発）

どちらのゲートも `cap0 1555200 / cap7 1382400`（最大の 43% / 30%）という同一の制限下で走った。

| 側 | decode | 5-token prefill |
| --- | --- | --- |
| HTP (NGL=16) | 32 step 平均 1192.8 ms = **0.838 tok/s** | 5306.6 ms |
| CPU (NGL=0) | 32 step 平均 1098.0 ms = **0.911 tok/s** | 7464.0 ms |

この 1 組だけを見ると CPU 単独が 8% 速い。**しかし同じ条件で NGL=16 を測り直すと
1.062 tok/s が出た（§9.1）。同じ cap・同じ手順で 0.838 と 1.062 が両方出るということは、
run 間ばらつきが 27% あるということである。したがって単発の比較では優劣を決められない。**
§9.2 で交互 4 組の反復を取った。

---

## §4 prefill / TTFT

`P53_PROMPT_TOKENS` と `P53_PROMPT_LIST` で、**1 回の model load から**複数長の prefill を
測った。掃引の前に捨て打ち 1 decode を入れてグラフ構築コストを掃引から除いている
（`SWEEP_WARM`）。プロンプト本文は probe に埋め込んだ固定英文を繰り返して長さを確保した。

### §4.1 CPU 単独（NGL=0）

run `sweepcpu` / `20261114-224741-28596`、`cap0 2227200 → 1555200`（最大の 61% → 43%）。

| 長さ | ms | tok/s |
| --- | --- | --- |
| 32 | 17,934 | 1.784 |
| 64 | 36,469 | 1.755 |
| 128 | 83,726 | 1.529 |
| 256 | 196,316 | 1.304 |
| 512 | **未測定**（guard 発動で打ち切り） | — |

### §4.2 HTP（NGL=16, SPLIT=0）

run `sweephtp2` / `20261114-230604-28733`、`cap0 1555200 → 1113600`（最大の 43% → 31%）。

| 長さ | ms | tok/s |
| --- | --- | --- |
| （捨て打ち 1 token） | 3,041 | — |
| 32 | 29,066 | 1.101 |
| 64 | 44,512 | 1.438 |
| 128 | 110,644 | 1.157 |
| 256 | 207,211 | 1.235 |
| 512 | **未測定**（guard 発動 `delta 520847 pages` で打ち切り） | — |

### §4.3 読み取れること（と、読み取れないこと）

- **prefill は decode の 2〜3 倍程度しか速くない。** 例: HTP の 64 token 掃引で 64→128 の
  差分は 66.1 s / 64 token = **1.03 s/token**、同じ run 系列の decode は約 1.2 s/token。
  理想的なバッチングなら重みの読み出しが 1 回で済むので 10 倍以上速くなるはずで、そうなっていない。
- したがって **512 token のプロンプトの TTFT は現在 4〜5 分規模**と推定される（実測は 256 まで。
  512 は guard 発動で取れていない。「推定」であって実測ではない）。
- **限界費用が長さとともに上がる**（CPU: 32→64 で 0.579 s/token、64→128 で 0.739、
  128→256 で 0.879）。これは (a) attention の O(N²) 項と (b) run 中の熱制限進行の両方が
  考えられ、この実験だけでは分離できない。**分離できていないので断言しない。**
- **cap が run 中に動いた。** 上の 2 表は同一 cap の比較ではない。したがって「HTP と CPU の
  どちらが prefill で速いか」は、この 2 表からは結論できない。

### §4.4 `n_batch` の罠（実装上の発見）

このビルドの `llama.cpp` は `llama-context.cpp:1795` で
`GGML_ASSERT(n_tokens_all <= cparams.n_batch)` を持つ。つまり **`llama_decode` 1 回に渡す
トークン数は `n_batch` 以下でなければならず、超えると `GGML_ABORT` でプロセスが落ちる**
（実測: exit 134、`PROMPT_LEN 128` の直後）。既定の `n_batch=32` のまま 128 token の
prompt を渡して落ちたのが最初の失敗で、probe 側で `NBATCH_AUTO` として自動引き上げする
ようにした。**実用入口（llama-cli 等）で長いプロンプトを扱うときにも同じ制約が効く。**

---

## §5 自由生成サンプル

固定プロンプト `The capital of Japan is`（5 token: `760 6511 314 6124 369`）から probe に
1200 token を自由生成させ、全文（`logits.f32.gen.txt`）と ID 列（`.gen.ids`）を保存した。
これは §6 の持続試験と同じ run（`susthtp`）の副産物である。

### §5.1 greedy（temp 0）1200 token

**出力全文の先頭と末尾（原文ママ）:**

```
 Tokyo.
The capital of Japan is Tokyo.
The capital of Japan is Tokyo.
The capital of Japan is Tokyo.
...
The capital of Japan is Tokyo.
The capital of Japan is
```

**統計（`gen.ids` から計算）:**

| 項目 | 値 |
| --- | --- |
| 生成 token 数 | 1200 |
| 異なり token 数 | **8** |
| 出現回数 | `760`×152, `6511`×152, `314`×152, `25358`×152, `13`×152, `198`×152, `6124`×144, `369`×144 |
| 12-token 窓の異なり数 | 19（1200 − 11 窓中） |
| 最頻 12-token 列 | `13 198 760 6511 314 6124 369 25358 13 198 760 6511` が **144 回** |
| 出力テキスト長 | 4,641 B |

**つまり「The capital of Japan is Tokyo.」を 152 回繰り返して 1200 token を使い切った。**

### §5.2 この結果の読み方（誇張しない）

- **これは品質の判断材料にはしない。** probe の decode ループは **EOG（EOS）で止まらない**し、
  repetition penalty も top-k も掛けていない。素の greedy 継続なので、正しい答えを 1 回出した
  あとに最も確率の高い継続（同じ文の繰り返し）へ落ちるのは当然の挙動である。
- **品質について言えるのは「最初の応答は正しい」ところまで。** `Tokyo.` を正しく出している。
  それ以上の「賢さ」は、この測定では何も言えない。
- **harness の限界として記録する。** 自由生成の比較（Phase 50 が「未測定」とした項目）を
  将来やるなら、probe に (a) EOG 停止、(b) repetition penalty、(c) chat template 適用、
  の 3 点が必要で、Phase 53 の probe にはどれも無い。**Phase 54 候補。**

### §5.3 サンプリング（temp > 0）

`P53_TEMP` / `P53_TOP_K` / `P53_TOP_P` / `P53_SEED` を実装したが、**サンプリング側の
自由生成サンプルは取得できていない**（時間と熱の予算を §6 と §9 に振った）。
実装は入っており、`SAMPLER temp 0.000 kind argmax` の行で経路が記録される。**未実施。**

---

## §6 持続性評価

### §6.1 実施条件（自由度の行使を明記）

| 項目 | 値 | 判断理由 |
| --- | --- | --- |
| run | `susthtp` / `20261114-231648-13673` | — |
| 形 | `P53_SEGMENTS=8` × `steps=150` = **1200 token 連続生成** | 30〜60 分の例示に対し、実測 1.1 s/token × 1200 = 約 23 分。まず 1 系列を完走させることを優先した |
| segment の意味 | 150 token ごとに KV を clear し、5 token の prompt を再 prefill | n_ctx を 171 に保つため。**「長文脈の持続」ではなく「小文脈の連続生成の持続」**である |
| 側 | HTP (NGL=16, SPLIT=0) | CPU 単独は同一条件で 1.2 s/token 前後になり 1200 token で 24 分超。時間と熱の予算から HTP 側を先行し、CPU 側は未実施（§12） |
| sampling | greedy（temp 0） | Phase 50〜52 の基準と同じ経路にするため |
| logits | 書かない（`P53_NO_LOGITS=1`） | 1200 × 248320 float = 1.19 GB になるため |
| guard | `GUARD_PAGES=524288`（2048 MiB） | §1 の判断。run の manifest に記録済み |
| 1 試験の長さ | `RUN_TIMEOUT=3600` | 180 秒制限の撤廃 |

### §6.2 結果

**1200 token / 8 segment を完走した**（`SEGMENTS done 8 of 8`、`GENERATED n 1200`）。

| segment | step 平均 ms | tok/s |
| --- | --- | --- |
| 0 | 1141.72 | 0.8759 |
| 1 | 1152.83 | 0.8674 |
| 2 | 1070.69 | 0.9340 |
| 3 | 1074.97 | 0.9303 |
| 4 | **1340.49** | **0.7460** |
| 5 | 1080.89 | 0.9252 |
| 6 | 1083.54 | 0.9229 |
| 7 | 1075.30 | 0.9300 |
| **全体** | **1127.55** | **0.8869**（`RESULT`） |

`GENERATION forwards 1200 wall_ms 1387458.019` = **0.8649 tok/s**（prefill・KV clear・
segment 境界を含む実時間 23.1 分）。

### §6.3 熱と周波数

- **cap0 は 729 サンプル中 694 サンプルで `1113600`**（最大 `3628800` の **30.7%**）。
  残り 35 サンプルが `1555200`（42.9%）。cap7 は `1132800`（最大 `4608000` の 24.6%）。
  **つまり 23 分間のほぼ全時間、CPU は最下段の熱制限下にあった。**
- `thermal_zone0`: 最小 52.6 ℃ / 中央値 **71.9 ℃** / 最大 **75.7 ℃**。電池温度は 47〜48 ℃ 帯。
- **追加の熱スロットリングは起きていない。** cap は run 開始から数分で最下段に落ち、その後
  23 分間動かなかった。つまり「これ以上落ちる先が無い」状態での持続測定である。

### §6.4 判定と、判定できないこと

- **progressive degradation は観測されなかった。** 8 segment の tok/s は 0.867〜0.934 の
  範囲に収まり、単調減少していない。segment 4 だけ 0.746 に落ちたが、前後が 0.93 なので
  **単発の干渉**（同時刻に私が adb 経由で状況確認を行っていた）と推測される。原因は特定して
  いないので、**外れ値として除外せず、そのまま記録する。**
- **「23 分もった」は「持続性能が高い」ではない。** 30.7% の周波数で 0.89 tok/s しか出て
  いない状態が 23 分続いた、というのが実測である。**consumer 用途の持続性能としては低い。**
- **30〜60 分の連続生成は達成していない。** 23.1 分（HTP）と 21.2 分（CPU、§6.5）で
  各 1 系列。延長は時間の余裕を見て判断する。

### §6.5 CPU 単独側の持続（追加測定）

§6.1 では HTP 側だけを走らせた。**CPU 単独（NGL=0）側も同じ形で走らせた。**

| 項目 | 値 |
| --- | --- |
| run | `sustcpu` / `20261115-025844-23784` |
| 形 | `P53_SEGMENTS=6` × `steps=150` = **900 token** |
| 実時間 | **1271.19 s（21.2 分）** |
| guard | `GUARD_PAGES=524288`、`delta_pages 455069`（発動せず） |
| cap | 開始 `1996800` → 途中から `1113600` で安定（603 サンプル中、範囲 `1113600`〜`1996800`） |
| tz0 | 最大 73.4 ℃、終了時 64.2 ℃ |

| segment | step 平均 ms | tok/s |
| --- | --- | --- |
| 0 | 1292.38 | 0.7738 |
| 1 | 1349.51 | 0.7410 |
| 2 | **1499.29** | **0.6670** |
| 3 | 1330.23 | 0.7518 |
| 4 | 1345.23 | 0.7434 |
| 5 | 1330.57 | 0.7516 |
| **全体** | **1357.87** | **0.7364** |

`GENERATION forwards 900 wall_ms 1255489.279` = **0.7169 tok/s**（prefill・KV clear を含む）。

**HTP 側（§6.2）との比較:**

| 側 | token 数 | 実時間 | decode tok/s | 全体 tok/s |
| --- | --- | --- | --- | --- |
| HTP (NGL=16) | 1200 | 23.1 分 | **0.8869** | 0.8649 |
| CPU (NGL=0) | 900 | 21.2 分 | **0.7364** | 0.7169 |

**HTP が decode で 20.4%、全体で 20.6% 速い。** ただし**両者は同じ時刻の run ではない**
（HTP は 23:16〜23:39、CPU は 02:58〜03:19）。cap はどちらも途中から `1113600` で
安定しており、**その点では条件が揃っている**。§9.2 の交互 3 組（中央値 +19.4%）と
**同じ方向・同じ大きさ**である。

**CPU 側にも progressive degradation は無い。** 6 segment が 0.667〜0.774 の範囲で、
単調減少していない。segment 2 だけ低いが、前後が 0.74〜0.75 なので単発の干渉と見られる
（**外れ値として除外せず記録する**）。

**これで「持続性」は両側そろった。** ただし **HTP 23.1 分 / CPU 21.2 分**で、
指示の例示「30〜60 分」には**どちらも届いていない**。

---

### §6.6 測定の完全性に関する注意（除外しない）

1. 23:2x に **adb のクライアント接続が落ちた**（device 側の probe は継続）。そのため
   `run53.sh` の後処理（`wall.txt` / `exit.txt` / `P53_END` 行）が書かれていない。
   測定値そのものは probe が出力した `RESULT_SEG` / `RESULT` / `GENERATION` /
   `SEGMENTS done 8 of 8` であり、**probe は正常終了している**（`status == 0` のときだけ
   出る行が揃っている）。端末の再起動は行っていない。
2. **guard の会計が合わない。** run 全体の `pswpout` 増分は **541,458 pages（約 2.06 GiB）**で、
   設定した `GUARD_PAGES=524288` を**超えている**のに guard-v2 は停止しなかった。
   guard-v2 が「累積」ではなく「サンプル間の増分」を見ている可能性があるが、**確認して
   いないので断定しない。** 1024 MiB ガードを絶対量の停止線として信頼するには、この点の
   原因究明が要る（Phase 54 候補）。

---

## §7 実用入口（llama-cli / llama-server）

### §7.1 端末に既にある実用入口の正体

`/data/local/tmp/llama.cpp/bin/` に `llama-cli`（7,432 B）と `llama-server`（7,424 B）がある。
中身は **薄いランチャ** で、`libllama-cli-impl.so` / `libllama-server-impl.so` を呼ぶだけの
ELF（`/workspace/tools/cli/main.cpp` 由来、clang 21 / LLD、Android r29）。実体は
`/data/local/tmp/llama.cpp/lib/` 側にある。

この一式は **2026-10-03 ビルド**で、`libllama.so` 33,917,968 B / `libggml-hexagon.so`
7,218,688 B / `libggml-htp-v81.so` 967,944 B。**Phase 46〜52 の HTP 側の作業より前**の世代である。

### §7.2 リンク整合性の実測（ここまでは確認できた）

| 試した `LD_LIBRARY_PATH` | 結果 |
| --- | --- |
| `/data/local/tmp/llama.cpp/lib`（2026-10-03 の自己完結セット） | **`--help` が正常に出力**（rc 0）。ランチャと impl の ABI は揃っている |
| `/data/local/tmp/c46-27b/libs:/data/local/tmp/llama.cpp/lib`（新しい libllama/ggml + 古い impl） | **失敗**: `cannot locate symbol "_ZNK21common_preset_context13load_from_iniE..." referenced by "libllama-server-impl.so"` |

つまり **新旧を混ぜることはできない。** 現在の skel（`288e6712`）と現在の Phase 46〜52 の
HTP 実装で実用入口を測るには、ホスト側（`libllama` / `libggml` / `libggml-hexagon` /
`libllama-common` / `llama-cli`）を**現行ツリーからクロスビルドし直す必要がある。**

### §7.3 クロスビルド（今夜実施、成功）

`build/p53-android` に Android arm64 向けを一通りビルドした。**成功（exit 0、357 target）。**

- NDK: `29.0.14206865`、`ANDROID_ABI=arm64-v8a`、`ANDROID_PLATFORM=android-28`
- Hexagon SDK: `C:\Qualcomm\Hexagon_SDK\6.6.0.0`
- CMake: Hexagon SDK 同梱の `3.28.3-windows-x86_64`（システムの cmake 4.4 ではなく）
- `GGML_HEXAGON=ON` / `LLAMA_BUILD_EXAMPLES=ON` / `LLAMA_BUILD_SERVER=ON` / `LLAMA_CURL=OFF` /
  `GGML_OPENCL=OFF` / `CMAKE_BUILD_TYPE=Release`
- **必要だった追加フラグ: `-DPREBUILT_LIB_DIR=toolv19_v81`**

`PREBUILT_LIB_DIR` を渡さないと configure が落ちる。落ち方と原因:

```
CMake Error at C:/Qualcomm/Hexagon_SDK/6.6.0.0/build/cmake/hexagon_fun.cmake:106 (string):
  string sub-command FIND requires 3 or 4 parameters.
Call Stack (most recent call first):
  ggml/src/ggml-hexagon/CMakeLists.txt:22 (include)
```

`hexagon_fun.cmake` の 99〜106 行は `string(FIND ${PREBUILT_LIB_DIR} "toolv8x" is_toolv8x)` を
無条件に呼ぶ。`PREBUILT_LIB_DIR` が空だと引数が 2 個になり CMake 3.28 がエラーにする。
skel 側の `ExternalProject_Add` は自分で `-DPREBUILT_LIB_DIR="toolv19_${V}"` を渡している
（`ggml/src/ggml-hexagon/CMakeLists.txt:88`）が、**ホスト側 configure では誰も渡していない。**
Linux では空でも通っていた可能性があり、Windows ホスト側の未整備点である。上流にも
同種の報告は見当たらない（§10 の一連のコミットとは別問題）。

生成物（`build/p53-android/bin/`）:

| ファイル | サイズ | sha256 |
| --- | --- | --- |
| `libggml-hexagon.so` | 7,785,320 | `8417bac72a4d550e82ebf32a3b89534f0c7c587b318b65e59f0b2e6e8d981560` |
| `libggml-htp-v81.so`（skel） | 1,103,304 | `70b9ed4fafb2ab298baf9f9bbed3c38a4a457cc4838775a690b9168dfb7eca1f` |
| `libllama.so` | 34,934,016 | — |
| `libllama-common.so` | 67,024,728 | — |
| `libllama-cli-impl.so` | 2,777,016 | — |
| `libllama-server-impl.so` | 45,287,568 | — |
| `llama-cli` / `llama-server` | 7,432 / 7,424 | — |

**注意:** このビルドの skel は `70b9ed4f`（Phase 51 の `dspqueue_read 0x2e` 修正込み）で、
端末で測ってきた Phase 50 D の `288e6712` とは**別物**である。以降の実用入口の数値は
`70b9ed4f` の条件で取ったものとして区別する。

### §7.4 端末への配置と起動確認（成功）

`/data/local/tmp/c53-cli/` に 12 ファイル 182 MB を配置した。**`libomp.so` が不足していた**ので
NDK の `toolchains/llvm/prebuilt/windows-x86_64/lib/clang/21/lib/linux/aarch64/libomp.so`
（1,234,864 B）を追加した（`libggml-cpu.so` が OpenMP 有効でビルドされていたため）。

起動に成功した:

```
$ LD_LIBRARY_PATH=/data/local/tmp/c53-cli ADSP_LIBRARY_PATH=/data/local/tmp/c53-cli llama-cli --version
version: 0.6.0-dev (build 11614, commit a6c45692e)
built with Clang 21.0.0 for Android aarch64

$ ... llama-cli --list-devices
Available devices:
  HTP0: Hexagon (0 MiB, 0 MiB free)
```

**現行ツリーの `llama-cli` が端末上で動き、Hexagon バックエンドを認識した。**
ここまでは PASS。

### §7.5 27B の実推論は完遂できず（PARTIAL / 失敗の記録）

| 試行 | 条件 | 結果 |
| --- | --- | --- |
| 1 | `-ngl 16`、既定の load-mode、`-no-cnv` | **`-no-cnv` はこの版に存在しない**（`error: invalid argument`） |
| 2 | `-ngl 16`、既定の load-mode、`-st` | **会話モードに入って `>` プロンプトで停止**（`< /dev/null` でも抜けず）。`MemAvailable` が 10.6 GB → **1.58 GB** まで低下し OOM 危険域。kill |
| 3 | `-ngl 16`、`--no-repack`、`-st`、メモリ監視 | load 中に `MemAvailable` 1.59 GB → 監視スクリプトが abort。**ただしこのとき前回の `llama-cli` が孤児として残っていた**（後述） |
| 4 | `-ngl 8` / `-ngl 0`、`--no-repack` | 孤児のせいで開始時点の `MemAvailable` が 1.8 GB しかなく即 abort。**この 2 件は条件不成立なので「NGL を下げても駄目」の証拠にはならない** |
| 5 | `-ngl 16`、`-lm mmap`、`--no-repack` | `MemAvailable` は 7.3 GB を保てたが、**"Loading model..." のまま 20 分以上進まず**（`Cached` 7.4 GB なのでページキャッシュには載っている）。kill |

**実装上の失敗（記録）:** 試行 3〜4 で `kill -9 $CP` が `timeout` ラッパだけを殺し、
`llama-cli` 本体が孤児として残ってメモリを保持し続けた。そのため試行 4 の 2 件は
**測定不成立**である。以後は `ps | awk '/llama-cli/'` で本体を殺す方式に変えた。

### §7.6 `-ngl` の閾値を発見（解決）

§7.5 の失敗は **`-ngl` が大きすぎた**ことが原因だった。`-lm mmap` を付けたまま
`-ngl` を 1 / 2 / 4 / 8 と振った（`-c 128 -b 32 -n 8`、各 300 s で打ち切り）。

| `-ngl` | 結果 | 所要 |
| --- | --- | --- |
| **1** | **完走**（`Tokyo` を出力） | 93 s |
| **2** | **完走**（`Tokyo` を出力） | 138 s |
| 4 | **300 s で打ち切り**（"Loading model..." のまま） | — |
| 8 | **300 s で打ち切り**（同上） | — |

**この端末で `llama-cli` が 27B を扱えるのは `-ngl 1` か `-ngl 2` までである。**
probe が `-ngl 16` で動くのは、`override-l57.txt` で **85 テンソルを CPU に固定**しており、
HTP に載る重みの量がまるで違うためである。`llama-cli` の `-ngl N` は「N ブロック全体」を
HTP に載せるので、同じ 16 でも意味が違う。**§7.5 の「原因の候補 1」が正しかった。**

### §7.6.1 `-ngl 4` が止まる場所の特定

`-ngl 4` を 90 秒走らせ、**全スレッドの `wchan`** を取った。

| スレッド数 | `wchan` | 意味 |
| --- | --- | --- |
| **2** | `fastrpc_wait_for_response` | **DSP の応答を待っている** |
| 1 | `fastrpc_dspsignal_wait` | DSP のシグナル待ち |
| 1 | `fastrpc_device_ioctl` | FastRPC の ioctl |
| 15 | `futex_wait_queue` | 他スレッドの完了待ち |
| 1 | `do_select` / 2 `do_sys_poll` / 1 `inet_csk_accept` | 無関係（端末側の常駐） |

**つまり「重い」のではなく「DSP が返事をしない」状態で止まっている。**
`VmRSS` は 8.04 GB で頭打ち、`MemAvailable` は 9.2 GB 残っており、**メモリ枯渇ではない。**
`logcat` にも `dmesg` にも FastRPC / subsys / SSR のエラーは出ていない。

**DSP 自体は壊れていない。** この直後に probe で 32-step を回し、
`b21ca4bf1da8a2ef3bc48f717ee79f13acc17ced0783141211fc7178f0c2e160` が
**既知値と一致**した（`cap0 1555200`、1.054 tok/s）。`adsprpcd` / `cdsprpcd` /
`dspservice` も生存している。

**したがって `-ngl 4` は「HTP に載せるテンソルが増えた結果、FastRPC の呼び出しが
返らなくなる」という不具合である。** 原因（VTCM 不足か、DMA のサイズ上限か、
特定 op の未対応か）は**特定できていない**。Phase 54 の課題として残す。

**安全面の記録:** この停止は**プロセスを kill するだけで回復した**。端末の再起動は
行っていない。DSP の異常（SSR）は発生していない。**停止線に触れる事象ではないと判断した。**

### §7.7 実用入口の実測（`-ngl 1` / `-ngl 2`）

`-c 512 -b 64 -t 6 --temp 0 --no-repack --no-warmup`、`-lm mmap`、プロンプトは
`The capital of Japan is`。所要は**プロセス起動から終了までの実時間**（load を含む）。

| 条件 | `-n` | 実時間 | 出力 |
| --- | --- | --- | --- |
| `-ngl 1` | 1 | **118 s** | `Tokyo` |
| `-ngl 1` | 2 | 111 s | `Tokyo` |
| `-ngl 1` | 16 | 134 s | `Tokyo` |
| `-ngl 1` | 32 | 165 s | `Tokyo` |
| `-ngl 2` | 32 | 199 s | `Tokyo` |
| `-ngl 2` | 40 | 173 s | `Tokyo` |

**load が実時間の大半を占める。** `-n 1` と `-n 32` の差は 47 s で、31 token 分なので
**約 1.5 s/token**。これは probe の decode（1.1〜1.2 s/token）と同水準である。
一方 `-n 1` の 118 s のうち約 116 s が load と初期化にあたる。

**つまり「起動して 1 問答える」だけで約 2 分かかる。** これは実用入口としての
**最大の障壁**であり、decode 速度（1.7 tok/s）よりも効く。

### §7.8 マルチターン KV 再利用

`-ngl 1`、`-n 1`、会話モードでターン数を変えて実時間を測った。

| ターン数 | 実時間 | 前ターンからの増分 |
| --- | --- | --- |
| 1 | 126 s | — |
| 2 | 145 s | **+19 s** |
| 3 | 174 s | **+29 s** |

**KV は再利用されている。** 2 ターン目以降は load をやり直さないので、増分は
「新しいプロンプトの prefill ＋ 1 token の decode」だけである。増分が 19 s → 29 s と
増えるのは、履歴が伸びて prefill 対象が増えるため（§4 の prefill 限界費用と整合）。

**ただし 1 ターン目が 126 s かかるので、対話としての体感は「最初の応答まで 2 分」である。**
KV 再利用そのものは機能している。

### §7.9 `llama-server`（HTTP、KV 再利用の本命）

`llama-cli` は起動のたびに load をやり直すので、対話の実用性を測るには不利である。
**`llama-server` は load が 1 回で済む**ので、こちらで測った。

起動: `-m <27B> -lm mmap -ngl 1 -c 512 -b 64 -t 6 --no-repack --no-warmup
--host 127.0.0.1 --port 8080 --no-webui`

**起動は 9.3 s で完了した**（`llama_server: listening on http://127.0.0.1:8080`）。
`llama-cli` の 116 s と比べて **12 倍以上速い**。`llama-cli` の 116 s には
会話モードの初期化・テンプレート処理・対話ループの準備が含まれていたと考えられるが、
**内訳は分離していないので断定しない。**

`/health` は `{"status":"ok"}`。`/v1/chat/completions` で 3 ターン測った
（`max_tokens 16`、`temperature 0`）。

| ターン | prompt_n | cache_n（再利用） | prompt_ms | prompt t/s | predicted_ms | predicted t/s | HTTP 実時間 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 57 | 0 | 50,894.6 | 1.120 | 18,093.8 | 0.829 | 69.0 s |
| 2 | 26 | **53** | 30,415.4 | 0.855 | 17,490.8 | 0.858 | 47.9 s |
| 3 | 58 | **42** | 63,213.5 | 0.918 | 17,318.1 | 0.866 | 81.5 s |

**読み取れること:**

- **KV キャッシュは実際に効いている。** ターン 2 は 79 token のうち **53 token を再利用**し、
  新規 prefill は 26 token だけで済んだ。ターン 1（57 token 全部）の 50.9 s に対し、
  ターン 2 は 30.4 s。**プロンプト処理が 40% 短くなった。**
- **decode は 0.83〜0.87 tok/s で安定。** 3 ターンとも 17.3〜18.1 s / 16 token で、
  ばらつきは 4% 以内。**§6 の持続試験（0.887 tok/s）とほぼ同じ**で、サーバ経由でも
  decode 速度は落ちない。
- **prefill は 0.86〜1.12 t/s。** §4 の probe 実測（1.10〜1.16 t/s）と同じ水準。
  **サーバのオーバーヘッドは小さい。**
- **ターン 3 が遅いのは履歴が長いから。** `cache_n 42` に対し `prompt_n 58` で、
  新規に 58 token を処理している。§4 の「限界費用が長さとともに上がる」と整合。

**実用性の評価:**

- **最初の応答まで 69 s。** `llama-cli` の 126 s より速いが、**まだ 1 分を超える。**
- **2 ターン目以降は 48〜82 s。** 会話として成立する速度ではない。
- **ただしサーバを立てっぱなしにすれば load は 1 回だけ**なので、
  「アプリ起動時にサーバを立ち上げ、以後は会話する」という設計は成立する。
  **起動 9.3 s ＋ 1 ターン目 69 s** が初回コスト、以後は 1 ターン 50〜80 s。

### §7.10 §7 の判定

- **PASS**: 現行ツリーから Android 一式のクロスビルドが通る（357 target、exit 0）。
- **PASS**: `llama-cli` が端末で起動し `HTP0: Hexagon` を認識する。
- **PASS**: **`-ngl 1` / `-ngl 2` で 27B の実推論が完走する。** 正しく `Tokyo` を返す。
- **PASS**: **`llama-server` が起動 9.3 s で立ち上がり、HTTP で 3 ターン応答する。**
  **KV キャッシュが機能している**（ターン 2 で 53/79 token を再利用、prefill が 40% 短縮）。
- **FAIL（実用性）**: **1 ターン 48〜82 s。** 起動は速くなったが、応答速度は実用の域に無い。
  `-ngl 4` 以上は load が完了しない。
- **未測定**: TTFT の内訳（load / prefill / decode の分離）。`--show-timings` は
  `[ Prompt: 0.7 t/s | Generation: 0.7 t/s ]` しか出さない。サーバの `timings` は
  prefill と decode を分けてくれるので、**今後の測定はサーバ経由で行うべきである。**

---

## §8 品質補強（評価文 2 本の 32/64-step）

Phase 52 は評価文 1 本（`p50-eval-a.txt`、llama.cpp のビルド手順の英語）でしか品質を測って
いない。**評価文を 2 本追加し、32 step と 64 step の両方で両側を測った。**

### §8.1 追加した評価文

| ファイル | 内容 | token 数（先頭 64 まで使用） |
| --- | --- | --- |
| `p53-eval-b.txt` | 端末上推論の日本語技術説明（熱・帯域・計測条件） | 276 |
| `p53-eval-c.txt` | 行列積の計算量と attention の O(N²) の英語解説 | 269 |

トークン化は自分でビルドした `llama-tokenize`（§7.3 の生成物）で行い、先頭 32 / 64 token を
`replay32b/c` / `replay64b/c` として端末に置いた。**評価文の単語は学習データから選んで
いない**（たまたま同じモデルの語彙に載っているだけ）ので、**絶対 PPL をモデル品質の指標に
しない。** ここで見るのは **CPU 側と HTP 側の差**だけである。

### §8.2 結果（`cap0=1113600` が 8 run すべてで安定・同一）

| 評価文 | steps | NLL CPU | NLL HTP | 差 (HTP−CPU) | SE | t (dof) | PPL CPU | PPL HTP | argmax 一致 | greedy==forced（CPU / HTP） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b | 32 | 3.894307 | 3.926877 | **+0.032570** | 0.020661 | **+1.58** (31) | 49.1220 | 50.7482 | **32/32** | 15/32 / 15/32 |
| c | 32 | 4.353516 | 4.318677 | **−0.034838** | 0.023352 | **−1.49** (31) | 77.7513 | 75.0892 | **32/32** | 9/32 / 9/32 |
| b | 64 | 3.229313 | 3.268403 | **+0.039091** | 0.023312 | **+1.68** (63) | 25.2623 | 26.2694 | **60/64** | 31/64 / 32/64 |
| c | 64 | 2.912100 | 2.879389 | **−0.032711** | 0.017528 | **−1.87** (63) | 18.3954 | 17.8034 | **63/64** | 29/64 / 29/64 |

**4 条件すべて |t| < 2。** しかも差の符号が **+ / − / + / −** と交互で、系統的な向きが無い。
4 本をまとめた平均差は **+0.000021 nats / step**（192 step、実質ゼロ）。

### §8.3 logits の距離

| 評価文 | steps | rel_l2 中央値 | rel_l2 最大 | max_abs 中央値 | max_abs 最大 | TV 最大 | KL 最大 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| b | 32 | 5.580% | 14.114% | 1.0351 | 2.9843 | 13.456% | 6.817e-02 |
| c | 32 | 6.544% | 14.633% | 1.2856 | 2.5044 | 8.949% | 2.630e-02 |
| b | 64 | 7.571% | **40.798%** | 1.1705 | 7.0063 | 13.878% | 6.817e-02 |
| c | 64 | 7.208% | 16.744% | 1.4576 | 2.9568 | 8.949% | 2.630e-02 |

rel_l2 の中央値は 5.6〜7.6% で Phase 52 の値（32-step 6.400%、64-step 7.305%）と同水準。
64b の 40.798% は **単一 step の外れ値**（Phase 52 の 49.517% と同じ性格）で、TV が 13.878%、
KL が 6.8e-02 にとどまることから、**分布全体が変わるのではなく logit が一様にずれる型**である。

### §8.4 §8 の判定

- **PASS（品質補強として）**: 評価文 2 本 × 32/64 step × 両側、計 8 run すべて exit 0、
  `cap0=1113600` で同一。**新しい 2 本でも「HTP と CPU の間に検出できる NLL 差は無い」。**
- **argmax 一致は合計 187/192（97.4%）**。32-step は 64/64、64-step は 123/128。
- **ただし「品質が同等」とは書かない。** 検出できた差が小さいことと、差が無いことは違う。
  特に **greedy==forced は評価文によって大きく違う**（b は 15/32、c は 9/32）ので、
  同じ 32-step でも「どれだけ当てられるか」は文脈次第である。**タスク正解率・外部
  データセットでの perplexity は依然として未実施。**
- 評価文が 3 本になったことで、**Phase 52 の 1 本だけの結論よりは頑健になった**が、
  n=3 であって統計的な一般性の主張はできない。

---

## §9 decode 性能の新規仮説

現行は約 1.7 tok/s（Phase 50〜52 の良い条件）。それを打破する手掛かりとして、今夜は
**「何が 1 token の時間を決めているか」**を測った。結論を先に書くと、**速度を上げる仮説は
まだ 1 つも検証できていない。** ただし「測れる土台」は 3 つ作れた。

### §9.1 NGL 掃引（同一手順で 0 / 8 / 16 / 24 / 32）

`replay32.txt` の 32 step、`override-base.txt`（NGL=0）/ `override-l57.txt`（NGL>0）、
skel `288e6712`、`GUARD_PAGES=524288`。

| NGL | cap0 (start→range) | decode step ms | tok/s | logits sha256 |
| --- | --- | --- | --- | --- |
| 0 | 1996800 → 1555200（**不安定**） | 1211.79 | 0.8252 | `44cd99ce...`（既知値と一致） |
| 8 | 1555200（**安定**） | 1057.99 | 0.9452 | `d1add98d764fae9fe93da09817690d9fcc672a10d789e0acec4524f406b9d42a` |
| 16 | 1555200（**安定**） | 941.35 | **1.0623** | `b21ca4bf...`（既知値と一致） |
| 24 | 1555200 → 1113600（**不安定**） | 1743.74 | 0.5735 | `5bfd45a391956ad7c25f49380b9a3168d95924de2d48742040b3e670fdfcd36e` |
| 32 | 1113600（安定） | — | — | **exit 6 で完走せず（後述）** |

**同じ cap（1555200）で安定していた NGL=8 と NGL=16 の比較では、NGL=16 が 12.4% 速い。**
NGL=24 は run 中に cap が 1113600 へ落ちたため **この値は比較に使えない**。
NGL=32 は **exit 6** で、原因は guard ではなく **warm-up の失敗**だった:
`PHASE warm_abort` が `warm_wait_begin` の 12.2 s 後に出ており、runner の
`warmup-c43.sh` が `WARM_FAIL` を書いたため probe が中断している（`guard stopped=0`）。
**NGL=32 がこの端末で動かない理由は特定していない。**

**配置が変わると logits のハッシュも変わる**（`44cd99ce` / `d1add98d` / `b21ca4bf` /
`5bfd45a3` はすべて異なる）。NGL=0 と NGL=16 は Phase 52 の既知値と一致したので、
**回帰は保たれている**。NGL を変えると数値が変わること自体は Phase 50 の
「NGL ごとに logits が byte-identical」という記述と食い違うが、Phase 50 のそれは
*同一 NGL 内での再現性* の話であり、NGL 間で違うのは自然である。

### §9.2 CPU 単独 vs HTP の交互 4 組（今夜の主成果）

Phase 49〜52 が 4 フェーズ続けて取れなかった「同一 cap での CPU 単独 vs HTP」を、
**fast cap の窓を待たずに、熱制限が最下段で安定した状態**で取った。最下段は
`cap0=1113600`（最大の 30.7%）で、一度落ちると 20 分以上動かない（§6.3）ので、
**同じ段に揃えた対比較の場として使える。**

`replay32.txt`、32 step、交互（NGL=0 → NGL=16 を 4 回）。

| 組 | NGL=0 tok/s | cap0 | NGL=16 tok/s | cap0 | 比 (HTP/CPU) | 採否 |
| --- | --- | --- | --- | --- | --- | --- |
| ac1 | 0.6572 | 1555200 安定 | 0.6418 | 1555200→**1113600** | 0.976 | **除外**（組内で cap が違う） |
| ac2 | 0.5271 | 1113600 安定 | 0.8874 | 1113600 安定 | **1.684** | 採用 |
| ac3 | 0.5726 | 1113600 安定 | 0.6755 | 1113600 安定 | **1.180** | 採用 |
| ac4 | 0.6067 | 1113600 安定 | 0.7247 | 1113600 安定 | **1.194** | 採用 |

**採用 3 組すべてで HTP が速い。** 比の中央値 **1.194（+19.4%）**、範囲 **+18.0%〜+68.4%**。
順序は常に「CPU が先、HTP が後」で、後になるほど端末は熱い（＝HTP に不利）方向なので、
この有意方向は順序効果では説明できない。

**しかし正直に書く:** n=3 組、各 run のばらつきは CPU 側 0.527〜0.607（13% 幅）、
HTP 側 0.676〜0.887（24% 幅）。**3 組で符号が揃ったこと以上の統計的裏付けは無い。**
「HTP が 19% 速い」と断定はせず、**「最下段の熱制限下では HTP が速い方向に 3/3 で揃った」**
という事実に留める。

**Phase 49〜52 との関係:** あちらは fast cap の窓を待って取れずに BLOCKED にした。
今夜は**窓を待たずに、最下段で安定するのを利用して**対比較を成立させた。**これは
「fast cap の比較の代替」であって「fast cap の比較」ではない。** 高クロック域で同じ符号に
なるかは未検証。

### §9.3 実効帯域（1 token あたり何 GB 読んでいるか）

モデルファイルは 9,259,510,912 B。1 token ごとに全重みを 1 回読むと仮定すると:

| 条件 | step ms | 実効帯域 |
| --- | --- | --- |
| NGL=16, cap0=1555200 | 941.35 | **9.84 GB/s** |
| NGL=0, cap0=1113600（ac3） | 1746.27 | 5.30 GB/s |
| NGL=16, Phase 52 の最良 | 532.90 | 17.38 GB/s |

**この端末の到達可能な帯域上限は測っていない**ので、「あと何倍伸びる」とは言わない。
言えるのは「1 token あたり 9.26 GB を読んでいる限り、速度は帯域で頭打ちになる」という
構造だけである。

### §9.4 prefill がバッチングでほとんど速くならない件（§4 の再掲と仮説）

§4 の実測から、prefill の限界費用は decode の 1/2〜1/3 程度でしかない。理想的な
バッチングなら重み読み出しが 1 回で済むので、もっと差が出るはずである。仮説:

1. **計算律速に近い。** 1 ubatch 32 token の prefill が 1 token decode の 29 倍かかっており
   （§4.3）、重み読み出しの償却ではなく演算そのものが増えている可能性。
2. **attention の O(N²) 項。** §4.3 で限界費用が長さとともに上がることを観測した。
3. **`n_ubatch` が小さい。** `n_batch=512` に対し `n_ubatch=32` のままなので、128 token の
   prompt でも 4 回に分割され、そのたびに重みを読み直している可能性。**これは検証した（§9.6）。**

### §9.6 `n_ubatch` 仮説の検証（棄却）

§9.4 の仮説 3 を反証可能な形で試した。`cap0=1113600`（3 run とも**安定・同一**）、
`P53_PROMPT_TOKENS=128`、`P53_PROMPT_LIST=128,256`、`P53_N_BATCH=256`（自動）、steps=2。

| ubatch | len 128 ms | len 128 tok/s | len 256 ms | len 256 tok/s |
| --- | --- | --- | --- | --- |
| 32（既定） | 112,919 | 1.1336 | 232,818 | 1.0996 |
| 128 | **183,108** | 0.6990 | 222,030 | 1.1530 |
| 256 | 116,107 | 1.1024 | 249,078 | 1.0278 |

**ubatch を上げても prefill は速くならない。** 32 と 256 はほぼ同じ（差は len128 で
+2.8%、len256 で −7.0% で、§9.2 で見た run 間ばらつきの範囲内）。128 は len128 で
**62% 遅い**という大きな外れ値を出したが、len256 では最速で、**方向が一貫しない**。
各条件 1 run ずつなので、128 の外れ値が再現するかは分からない。

**結論: 仮説 3 は支持されない。** prefill が償却されない原因は ubatch の粒度ではない。
残る候補は §9.4 の 1（計算律速）と 2（attention の O(N²)）で、**どちらも未分離**。
分離するには、同一 run 内で「同じ長さの prompt を attention 無効で回す」等の対照が要る。

### §9.5 今夜作れた「測れる土台」

1. probe に **prefill 掃引**（1 回の load で複数長）と **segment 連続生成**を入れた。
2. **`GUARD_PAGES` を明示できる**ようにし、run ヘッダと manifest に残すようにした。
3. **最下段の熱制限は 20 分以上動かない**という性質を見つけた。**fast cap の窓が来ない
   ときの対比較の場として使える。** Phase 49〜52 が 4 回続けて失敗した原因
   （「窓が 1-6 分で閉じる」）に対する実務的な回避策になる。

---

## §10 上流 Hexagon 関連変更の監視

`ggml-org/llama.cpp` の `ggml/src/ggml-hexagon` を触る直近のコミットを GitHub API で確認した
（2026-11-14 時点、取得のみ。**取り込みはしていない**）。

| 日付 (UTC) | commit | 内容 | 本プロジェクトとの関係 |
| --- | --- | --- | --- |
| 2026-11-14 | `61dbf49228` | Hexagon: IQ2_XXS と NVFP4 対応 (#31954)。tiled repack layout、HVX matmul、HMX 経路、`supports_op` | **IQ2 系の HTP 対応が上流で拡張された。** 本プロジェクトのモデルは IQ2_S で別型だが、IQ2_XXS の gather 実装は Phase 46 の IQ2_S gather と同じ設計思想（2-gather tile decode、compact grid、gather pipelining）を辿っており、比較材料になる |
| 2026-11-13 | `ded522f830` | SSM_SCAN op 追加 (#31921) | 直接は無関係（本モデルは SSM を使わない） |
| 2026-11-13 | `9daa0aa22b` | **buffer allocation / mmap stacking の改善 (#32144)** — queue と fence バッファの統合、`n_threads` による queue サイズ決定、iova 断片化を避ける stacking、mmap 失敗の再試行、拡張再試行の eager モード | **本プロジェクトが Phase 46〜51 で繰り返し踏んだ DMA/mmap 問題と正面から同じ領域。** 未検証だが、将来取り込めば guard 追加系の作業が減る可能性がある |
| 2026-11-12 | `92d6c8a5ca` | Windows ARM64 の clang-cl でホストビルドが通る修正 (#32011) | 本プロジェクトは Windows x64 ホスト + Android NDK クロスなので直接は不要。ただし `_M_ARM64` 未定義問題など、Windows ホストビルドの知見として参考 |
| 2026-11-12 | `2a4daf0e08` | Q5_0 / Q5_1 対応 (#32094) | 本モデルは IQ2_S なので無関係 |
| 2026-11-04 | `bee23322f1` | LIGHTNING_INDEXER 対応 (#31289)。**prefill 用 HMX 経路**、decode 用 q8_0 gather transpose | prefill を明示的にターゲットにしている点が本 Phase の §4 と問題意識を共有 |
| 2026-11-03 | `7071ba378a` / `6b9c0cfcba` | TOP_K_MASK op、backend sampler の TOP_K 高速化 | サンプリングを DSP に載せる方向。本プロジェクトは未着手 |
| 2026-11-03 | `20a51f24a8` | DSV4_HC_PRE / DSV4_HC_POST op | 無関係 |

**報告のみ。** `hexagon_fun.cmake` の `PREBUILT_LIB_DIR` 未定義でホスト configure が落ちる件
（§7）は、上流のこの一連の変更とは独立した、Windows ホスト側の Hexagon SDK 連携の問題である。

---

## §11 Phase 46〜52 総括

### §11.0 前提（資料の範囲）

`continuations/` に独立した報告書があるのは **c49 / c50 / c51 / c52** のみ。Phase 46〜48 は
専用の報告書を持たず、Phase 50〜52 の報告書の中に `run_split.sh` の由来や `dspqueue_read 0x2e`
の経緯として言及されている。したがって §11 は「Phase 49〜52 は報告書から引用、Phase 46〜48 は
それを含む作業として参照」という形でまとめる。**報告書に無い数値は作らない。**

なお `phase49-report.md` は存在せず、実体は `phase49-5-report.md`（"Phase 49.5 — Phase 49 の
確定・検証・Phase 50 の準備"）で、判定語彙も PASS/PARTIAL/FAIL/BLOCKED ではなく
**READY WITH RISKS** である。ここでは原文の語彙のまま引用する。

### §11.1 各 Phase の判定（原文の語彙）

| Phase | 判定 | 主要内容 |
| --- | --- | --- |
| 49 | **READY WITH RISKS** | bf16 変換 `65536 checked / 65536` 不一致 0（dot K=5120 M=1 `max_abs 1.51e-05`、上界 `6.27e-04`、rel_l2 `2.19e-07`）。DSP 実効帯域が 0.3% 以内で再現（IQ3_S `28.10 GB/s`、IQ3_XXS `21.99 GB/s`）。4-step logits sha256 `5585d4ed...` が byte-identical。27B CPU vs HTP NGL=9 で rel_l2 `2.04%`–`3.92%`、max_abs `0.745`、argmax 2/4、top10 `10/10`。同一 tier ABBA（cap 1785600）で NGL=0 `966.6`/`974.2 ms` vs NGL=16 `737.4`/`825.2 ms`、隣接 pair `-15.3%`。ただし「**fast cap の新しい比較値が無い。**」 |
| 50 | **PARTIAL** | IQ2_S gather `+38.8%`（`10.21` → `14.17 GB/s`）、IQ2_XS `+41.3%`、IQ2_XXS `+33.7%`、IQ3 対照 `0%`、DSP 合計 `110.2` → `103.4 ms/token`。熱制限 tier（cap 2112000/1862400）で CPU 単独 `681.3 ms`、NGL=16 `615.0 ms`（`-9.7%`）、NGL=24 `600.8 ms`（`-11.8%`）、NGL=28 `627.1 ms`。16-token paired NLL 差 `+0.000628 nats`、SE `0.021732`、`t (dof=15) +0.03`、argmax `16/16`。fast cap の 5% 目標は**未達**（「**同じ cap の対比較ではない**ので、差（約 -6.5%）を性能の結論に使わない」） |
| 51 | **PARTIAL（総合）** | 行分割の並列化は **FAIL** — 有効 pair 4/4 すべて遅く、中央値 `-5.25%`（`-6.8%`/`-1.6%`/`-3.7%`/`-18.3%`）、「速い pair は 0 組」。33 本 vs 34 本の比較 **PASS**、`dspqueue_read 0x2e` の原因究明と修正 **PASS**（`ddcf2a54a`）、mutex 破壊 **CLOSED（未再現）**。「行分割で 5% は取れない。上限は 1% 前後」、DSP 区間 `33.7 ms`(NGL=9) / `34.2 ms`(NGL=16) =「token の 5.4%」。fast cap の CPU 単独比は**未測定（保留）** |
| 52 | **PASS ×3 / BLOCKED ×1** | 32-step 品質 **PASS**（NLL `3.132049` vs `3.125820`、差 `−0.006229 nats`、SE `0.013226`、`t (dof=31) −0.47`、PPL `22.9209`/`22.7786`、argmax `32/32`）。64-step 品質 **PASS**（NLL `2.977521` vs `2.964708`、差 `−0.012813`、SE `0.016859`、`t (dof=63) −0.76`、PPL `19.6391`/`19.3890`、argmax `61/64`）。probe 上限 32→64 **PASS**。fast cap CPU 単独比 **BLOCKED**（pair 0 組） |

### §11.2 実用 5 要素の現状（区別して記す）

「実用」を 5 つに分け、**今夜実測できたもの / 過去 Phase の実測 / 未検証**を区別する。

#### (1) decode の絶対速度

| 出典 | 値 | 条件 |
| --- | --- | --- |
| Phase 50（報告書） | `615.0 ms`/token（NGL=16）、`681.3 ms`/token（NGL=0） | cap 2112000/1862400、8 step |
| Phase 52（報告書） | `1.877 tok/s`（HTP, tier 3187200）、`1.565 tok/s`（CPU, tier 3628800） | 32 step。**tier が違うので CPU 単独比として扱わない** |
| **Phase 53（今夜）** | HTP `0.838 tok/s`、CPU `0.911 tok/s` | **両方 cap0 1555200/1382400 の同一条件**、32 step、各 1 run |

→ **絶対速度は cap に強く依存する。** 今夜の同一 cap 比較では CPU 単独が HTP を 8% 上回った。
Phase 49〜52 の「HTP が 10〜15% 速い」という結果は、より高い cap の tier で取られている。
**同一 cap・交互測定での決着は Phase 49 から 53 まで一度も取れていない**（§12 の未検証）。

#### (2) prefill / TTFT

**Phase 46〜52 では一度も測っていない。Phase 53 が初測定**（§4）。

- prefill は decode の 2〜3 倍程度しか速くない（HTP 掃引 64→128 の限界費用 1.03 s/token 対
  decode 約 1.2 s/token）。
- 512 token の TTFT は**推定 4〜5 分**。実測は 256 token まで。
- この事実は「実用入口」の評価に決定的で、**チャット用途の TTFT としては非実用的**である。

#### (3) 実用入口（llama-cli / llama-server）

**Phase 53 で初めて「動く」状態になった**（§7）。

- 現行ツリーから Android 一式をクロスビルドし、`llama-cli` が端末で `HTP0: Hexagon` を認識。
- **`-ngl 1` / `-ngl 2` で 27B の実推論が完走し、`Tokyo` を正しく返す。**
- **`llama-server` は起動 9.3 s。** `llama-cli` の 116 s と比べて桁違いに速い。
- **KV キャッシュが機能する。** ターン 2 で 79 token 中 53 token を再利用し、
  prefill が 50.9 s → 30.4 s（40% 短縮）。
- **しかし 1 ターン 48〜82 s。** 起動は速くなったが、応答速度は実用の域に無い。
  `-ngl 4` 以上は load が完了しない。

**「動く」と「使える」の間にはまだ大きな隔たりがある。** ただしサーバを常駐させれば
load は 1 回で済むので、**「起動 9.3 s ＋ 1 ターン目 69 s、以後 50〜80 s」** という
形にはなった。Phase 52 まで「未着手」だった状態からは前進している。

#### (4) 体感品質

Phase 52 の NLL / PPL / argmax が唯一の定量根拠。

- 32-step: NLL 差 `−0.006229 nats`（`t −0.47`）、argmax `32/32`
- 64-step: NLL 差 `−0.012813 nats`（`t −0.76`）、argmax `61/64`（不一致は step 38 / 48 / 58）
- **どちらも「統計的に有意な差は検出されない」であって「品質が同等」ではない。** 64-step で
  argmax が 3/64 ずれるという事実は残る。
- **タスク正解率・perplexity の外部データセット評価は Phase 49〜53 を通じて未実施。**

#### (5) 持続性

**Phase 46〜52 では一度も測っていない。Phase 53 が初測定**（§6）。

### §11.3 Phase 46〜52 が残した「測れていないもの」一覧（原文のまま）

- fast cap の CPU 単独比 — Phase 49「fast cap の新しい比較値が無い」、Phase 50「同一 cap の
  有効な対比較 4 組」を取得できず、Phase 51「窓が 1-6 分で閉じる」、Phase 52 **BLOCKED**
- 品質（perplexity・タスク正解率）— Phase 49「品質（perplexity、タスク正解率）は測っていない」
- greedy 自由生成の差分 — Phase 50「**未測定。** replay はトークンを固定するので自由生成の
  比較にならない」
- 27B 全体の ms/token の差 — Phase 50「**測定できなかった（未解決）**」
- CPU 内訳の時間列 — Phase 49、`simpleperf` が "not supported on the device" のため計算値
- Phase 50 の長い replay — 32 は「13 step で swap-out ガード (`exit 6`)」、64 は「`exit 2`、
  probe の stdout が空」で「n=16 がこの端末で取れる上限」だった（Phase 52 で上限を 64 に
  引き上げて解決済み）
- Phase 51 mutex 破壊 — CLOSED（未再現）

---

## §12 判定

### §12.1 目標ごとの判定

| # | 目標 | 判定 | 根拠と限界 |
| --- | --- | --- | --- |
| 1 | prefill / TTFT 初測定＋自由生成サンプル | **PARTIAL** | 両側とも 32/64/128/256 token の prefill を初測定（§4）。greedy 1200 token の全文と統計を保存（§5）。**512 token は両側とも guard 発動で未測定**、**サンプリング（temp>0）は未実施** |
| 2 | 持続性評価 | **PASS（長さは未達）** | **両側そろった。** HTP 1200 token / 23.1 分（decode 0.8869 tok/s、§6.2）、CPU 900 token / 21.2 分（0.7364 tok/s、§6.5）。**どちらも progressive degradation なし**、cap は途中から `1113600` で安定。**HTP が 20.4% 速い**（§9.2 の交互 3 組と同じ方向・同じ大きさ）。**30〜60 分には届かず**、各 1 系列のみ |
| 3 | 実用入口の本格整備 | **PASS（ただし遅い）** | 現行ツリーから Android 一式のクロスビルドに成功（357 target、exit 0）。`llama-cli` が端末で起動し `HTP0: Hexagon` を認識。**`-ngl 1` / `-ngl 2` で 27B の実推論が完走し `Tokyo` を返す**（§7.6〜§7.7）。**`llama-server` は起動 9.3 s、HTTP 3 ターン応答、KV キャッシュが機能**（ターン 2 で 53/79 token 再利用、prefill 40% 短縮、§7.9）。**ただし 1 ターン 48〜82 s**、`-ngl 4` 以上は load 未完 |
| 4 | Phase 46〜52 総括 | **PASS** | §11。報告書からの引用のみで構成し、新規推定を足していない |
| 5 | 品質補強（評価文 2 本） | **PASS** | 評価文 b / c を追加し、32/64 step × 両側の 8 run すべて exit 0・`cap0=1113600` で同一（§8）。4 条件すべて \|t\| < 2、符号も交互。argmax 187/192。**「品質が同等」ではなく「差を検出できなかった」** |
| 6 | decode 性能の新規仮説 | **PARTIAL** | 仮説 `n_ubatch` を**棄却**（§9.6）。**同一 cap・3 組で CPU 単独 vs HTP の対比較を成立**（§9.2、HTP が 3/3 で速い方向）。ただし**速度そのものは改善していない**。Phase 49〜52 が 4 回失敗した fast cap 比較の**代替手段**（最下段の熱制限を対比較の場に使う）を確立（§9.5） |
| 7 | 上流 Hexagon 変更の監視 | **PASS** | §10。取得と記述のみ、取り込みなし |

### §12.2 安全条件の遵守状況

| 条件 | 遵守 |
| --- | --- |
| 端末を再起動しない | **遵守**（uptime は開始時 8 日 21 時間から一度もリセットしていない） |
| 温度保護・vendor 電源ポリシー・ZRAM・重要サービス・`system_server` へ介入しない | **遵守**（`scaling_max_freq` は読むだけ。`thermal_zone0` も読むだけ。ZRAM 設定を変更していない。`qsguard` など vendor プロセスに触れていない） |
| OOM kill・システム不安定・DSP 異常で即時停止 | **該当なし（停止に至る兆候は出ていない）**。ただし `llama-cli` 試行 1〜2 で `MemAvailable` が 1.58 GB まで下がった箇所があり、**自分で kill して回避した**（§7.5）。OOM kill は発生していない |
| swap guard を勝手に緩めない | **明示の裁量範囲で変更した**（262144 → 393216 / 524288）。全 run の manifest と run ヘッダに記録。**1024 MiB のままでも複数回発動しており、その全件を §3.1 と本文に記録した**。ただし §6.6 の会計不一致は未解決 |
| 1 試験 180 秒以内 | **撤廃を明示されたため長い試験を実施**（最長 1387 s の decode）。撤廃の指示が無ければ逸脱である |
| TVer を止める | **許可に従い停止した**（§2.2 の force-stop リストに追加） |
| メイン worktree の未コミット 9 ファイルを保護 | **遵守**（開始時 9、終了時 9、内容も未変更） |
| 隔離 worktree・専用ブランチ | **遵守**（`llama.cpp-c51` の `codex/c53-summary`） |
| merge / force push / 過去コミット改変 | **なし**。`--amend` も使っていない（Phase 52 で 1 回使ったが、今回は使っていない） |
| push は `codex/c53-summary` のみ | **遵守** |
| 上流の取り込み禁止 | **遵守**（`git fetch` すらしていない。GitHub API の参照のみ） |
| 未検証を PASS と書かない | **遵守**。§12.1 で 3 目標を PARTIAL、§7.5・§9.6 で失敗を明記 |

### §12.3 未検証として残すもの（Phase 54 候補）

1. **512 token 以上の prefill / TTFT**（両側とも guard で未取得）
2. **サンプリング（temp>0）での自由生成サンプル**
3. **30〜60 分への延長**（HTP 23.1 分 / CPU 21.2 分まで取得済み）
4. **llama-cli / llama-server で 27B を動かすこと**（`-ngl` を下げる、テンソル override の口を
   足す、load が進まない件のスタックダンプ、の 3 方向）
5. **高クロック域での CPU 単独 vs HTP の対比較**（今夜成立したのは最下段のみ）
6. **prefill が償却されない原因の分離**（計算律速 vs attention O(N²)）
7. **NGL=32 が warm-up で落ちる理由**（`WARM_FAIL resident_pages pct 78` まで判明。
   HTP バッファがモデルの residency を押し下げていると考えられるが**未確認**）
8. **guard-v2 の会計が合わない件**（§6.6）

### §12.4 引き継ぎ（次の担当者へ）

#### 直したもの

| 対象 | 何をしたか |
| --- | --- |
| probe | 第 3 世代 `c53_decode_probe.cpp`。step 上限・n_ctx・n_batch・n_ubatch・prompt 長・prefill 掃引・segment・サンプラ・logits 無効化を**環境変数でだけ**制御。既定は Phase 52 と同一経路で、32-step の既知 hash 2 本が一致（§3） |
| runner | `run53.sh` / `run_split53.sh` / `p53_run.sh`。`RUN_TIMEOUT` と `GUARD_PAGES` を追加し、`P53_*` 環境を run ヘッダに記録。安全側の既定値は Phase 50 のまま |
| ビルド | Android arm64 向けクロスビルド手順を確立（§7.3）。**鍵は `-DPREBUILT_LIB_DIR=toolv19_v81`** と Hexagon SDK 同梱の cmake 3.28.3、NDK の `libomp.so` の配置 |

#### 直せなかったもの

| 対象 | 状態 |
| --- | --- |
| `llama-cli` の `-ngl 4` 以上 | **load が完了しない**。`wchan` で `fastrpc_wait_for_response` に 2 スレッドが滞留しており、**DSP が応答を返さない**状態（§7.6.1）。メモリ枯渇でも DSP 異常でもない。`-ngl 1` / `-ngl 2` なら動く |
| `llama-cli` の起動時間 | **約 2 分**（`-ngl 1` で 118 s 中 116 s が load）。`llama-server` は 9.3 s なので、**cli 固有のコスト**である（§7.7 / §7.9） |
| 応答速度 | **1 ターン 48〜82 s**（サーバ経由）。実用の域に無い（§7.9） |
| prefill の償却 | **未解決**。2〜3 倍止まり。`n_ubatch` 仮説は棄却済み（§9.6） |
| NGL=32 | warm-up が `resident_pages pct 78` で落ちる（§9.1） |
| guard-v2 の会計 | 累積 541,458 pages で発動しなかった（§6.6） |
| TTFT の内訳 | load / prefill / decode を分けた値は取れていない（§7.10） |

#### decode 仮説の結果（要点）

- **棄却**: `n_ubatch` を上げれば prefill が速くなる → ならない（§9.6）
- **支持（弱）**: 最下段の熱制限下では HTP が CPU 単独より速い方向に 3/3 で揃った（§9.2）
- **未分離**: prefill が償却されない原因が計算律速なのか attention の O(N²) なのか

#### 次の一手（優先順）

1. **応答速度。** 実用入口の最大の障壁は load ではなく **1 ターン 48〜82 s** である（§7.9）。
   decode 0.86 tok/s と prefill 0.86〜1.12 t/s の両方が効いている。§9 の decode 仮説と
   §4 の prefill 償却が、そのまま実用性の課題になる。
2. **`llama-server` を常駐させた状態での連続測定。** 起動 9.3 s は許容範囲なので、
   「サーバを立てておき、複数ターンを続ける」形での tok/s 推移・熱・メモリを取る。
   `timings` が prefill と decode を分けてくれるので、**今後の測定はサーバ経由が良い。**
3. `-ngl 4` 以上で load が完了しない原因。**`wchan` で `fastrpc_wait_for_response` に
   2 スレッドが滞留していることまでは特定した**（§7.6.1）。次は「どのテンソルを HTP に
   載せたときに返らなくなるか」を `-ngl 3` と `-ngl 4` の差分で絞り、FastRPC の
   タイムアウトを有効にして（`ADSP_LIBRARY_PATH` 側の設定ではなく、呼び出し側の
   リトライ／タイムアウト）スタックを取る。
4. prefill の償却を分離する対照実験（同一長で attention を切る／`n_ubatch` を固定して
   長さだけ変える、を**同一 run 内**で）。
5. 512 token の prefill を取る（`GUARD_PAGES` を上げるか、掃引の合間に
   `llama_memory_clear` を挟んで swap を落ち着かせる）。
6. **最下段の熱制限を対比較の場に使う**手法を、Phase 49〜52 が未達にした他の比較
   （fast cap の CPU 単独比など）へ横展開する。
7. guard-v2 の実装を読み、「累積」か「区間増分」かを確定させる。
8. probe に EOG 停止・repetition penalty・chat template を入れ、自由生成の比較を可能にする
   （§5.2）。


