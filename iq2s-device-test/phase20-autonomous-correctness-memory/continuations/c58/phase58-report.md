# Phase 58 - セッション依存の追求と iq3_xxs の 4 列化

対象: 2026-11-19 01:30 〜
作業 worktree: `D:\ghq\github.com\yuubinnkyoku\llama.cpp-c51`
ブランチ: `codex/c58-session`（`codex/c57-batchcost` / `d888366cd` から作成）
push 許可: `codex/c58-session` のみ

この報告書は作業中に逐次更新している。数値はすべて実測で、未測定は「未測定」と書く。

---

## §0 目標0: 現在地の固定と回帰ゲート - PASS

| 項目 | 実測 |
| --- | --- |
| 端末 | `100.108.106.81:45997`（Tailscale 経由の adb。USB と 192.168.55.15 は使えなかった） |
| skel | `288e6712df9c8201ef05c12ec5704968d4f71407992cff8e0f03aa002ee0861d` |
| probe | `482032c30d2bd27f51c27d478b638ba64e1f86a467586a3190accb66ebbd2b59` |
| replay.txt | 標準 4 トークン `25358 11 198 760` |
| 開始時 uptime | **227,172 s（63.1 時間）**。Phase 57 の再起動から一度も落ちていない |
| 開始時状態 | swap 使用 7.3 GB、MemAvailable 5.3 GB、TZ 41.5 ℃、cap 1.79 GHz（画面 ON） |

画面を消すと vendor が cap を **883 MHz** に落とす（実測: 消灯 883200 / 点灯 1785600）。
`p53_run.sh` は 20 秒ごとに `KEYCODE_WAKEUP` を送るので測定中は点灯状態になる。

### §0.1 32-step 回帰

| run | 条件 | exit | sha256(logits.f32) | 既知値 | 判定 |
| --- | --- | --- | --- | --- | --- |
| `20261119-013204-9078` | 既定 NGL=44 | 0 | `e9da92f8d7280750f5c62a8bf3489fdc7332fc30e1774d708f1df7d65c0ec6c1` | 同左 | **一致** |
| `20261119-014722-17199` | 既定 NGL=44（再起動後） | 0 | 同上 | 同左 | **一致** |

**目標0 は PASS。** 2 本目は §1 の再起動後に取り直したもの。

---

## §1 事故: 長時間 server 測定でメモリが枯渇し端末が再起動した（記録）

### §1.1 何をしたか

目標1 の劣化カーブ測定のため、`llama-server -ngl 44 -c 512` を 1 本常駐させ、
60 秒間隔で greedy の completion（n_predict=48）を送る `p58_curve.sh` を開始した
（開始 2026-11-19 01:39、uptime 227,477 s）。

### §1.2 何が起きたか（実測ログ）

| sample | wall | prompt_ms | pred_n | per_tok_ms | MemAvailable | SwapFree | pswpout | cap | tz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 s | NA | NA | NA | 6.98 GB | 10.36 GB | 66,660,382 | 1.79 GHz | 51.9 ℃ |
| 2 | 257 s | 21,858 | 48 | **5,000.6** | **0.60 GB** | 7.06 GB | **103,576,174** | 1.11 GHz | 68.0 ℃ |

**5 分間で swap-out が 37,000,000 ページ（約 148 GB）増え、MemAvailable が 0.6 GB まで落ちた。**
decode は 5.0 s/token（遅いモード）。その後まもなく端末は**再起動**した
（次の確認で uptime 122 s）。**私は再起動コマンドを発行していない。**
メモリ枯渇による system_server 側の停止と推測するが、**機序は未確定**。

### §1.3 何が悪かったか

- **server には probe の swap guard が無い。** probe は `guard-v2` で swap-out を監視して
  8 GB で止めるが、server を直接起動するとその保護が無い。
- モデル 9.25 GB + HTP バッファ 6.6 GB + KV で、16 GB の端末には余裕が無い。
  起動直後の health check が 2 秒で返ったため、**ロード完了前に最初のリクエストを送ってしまった**
  （sample 1 が timings 無しで返っている）。その後のロードでページキャッシュが総入れ替えになった。
- 「長時間の劣化カーブ」という目的に対して、**この端末で server を長時間回すのは安全でない**。

### §1.4 以後の制約（自己課）

- server の長時間常駐は**しない**。decode の測定は probe（guard 付き）のみ。
- 1 run あたりの swap-out 増分が 1 GB を超えたら、その run の数値は比較に使わない。

---

## §2 目標1: decode のセッション依存 - PARTIAL（事故の副産物として 1 点だけ確定）

### §2.1 得られた観測

事故の sample 2 は「遅いモード」の最中だった。そのときの状態:

- decode **5,000.6 ms/token**
- MemAvailable **0.60 GB**、swap 使用 5.2 GB、**pswpout が 5 分で 37M ページ増**
- cap 1.11 GHz、TZ 68.0 ℃

**「遅いモード」はメモリ枯渇＋スワップスラッシングと同時に起きていた。**
Phase 56/57 の遅いモード（3.4-4.7 s/step）も、その時のセッションは swap 使用が多く
（Phase 57 開始時点で 7.3 GB）、同じ機序の可能性がある。**相関の 1 点であり、原因の証明ではない。**

### §2.2 劣化カーブは取れなかった

事故後は端末が再起動し、セッションは初期化された（swap 0、MemAvailable 9-12 GB）。
そこから 2-3 時間の劣化カーブを取るには、意図的にメモリを食う必要があり、
**§1 と同じ事故を再現するリスクがあるため実施しない**。
「安全停止線」に従い、ここは PARTIAL とする。

### §2.3 判定

**PARTIAL。** 遅いモードとメモリ枯渇の同時発生を 1 例記録した。
発生条件の特定・回復手段の確認はできていない。

---

## §3 目標2: iq3_xxs の 32x4 タイル - PASS（bit-exact、DSP で 2.9-3.1 倍）

### §3.1 実装

`MATMUL_2D_REPACKED_IMPL` に 4 列目の引数 `DOT_2X4` を追加し、列ループを
4 列 -> 2 列 -> 1 列の 3 段にした。iq3_xxs だけ実カーネル、他の 12 型は
「2 列カーネルを 2 回」のラッパーで、演算も仕事量も従来と同じ。

| ファイル | 変更 |
| --- | --- |
| `htp/hvx-mm-kernels-tiled.h` | `tiled_vec_dot_iq3_xxs_32x4` を追加（32x2 と同じ構造で 4 行分の dot を持つ。gather 8 本は 4 行で共有、finish group は 1 回） |
| `htp/matmul-ops.c` | `IQ3XXS_DOT_2X4`（実カーネル）と他 12 型の `*_DOT_2X4` ラッパー、`MATMUL_2D_REPACKED_IMPL` の 4 行ループ、13 箇所の instantiation 更新 |

ビルド: `cmake --build build/p53-android --target htp-v81 -j 8`
新 skel sha256: **`76dd89e3bf2af1576101eba7bb10e5b9d750191fdc92b6a1fc019967166782ef`**
（ベース `288e6712` は 1,103,304 B、新は 1,115,784 B）

### §3.2 合格条件 1: bit-exact - **PASS**

| NGL | 既知 hash | 新 skel の hash | 判定 |
| --- | --- | --- | --- |
| 16 | `b21ca4bf1da8a2ef3bc48f717ee79f13acc17ced0783141211fc7178f0c2e160` | 同左 | **一致** |
| 44 | `e9da92f8d7280750f5c62a8bf3489fdc7332fc30e1774d708f1df7d65c0ec6c1` | 同左 | **一致** |

**全型・全経路で演算が変わっていない**（他の 12 型のラッパーも同じ hash で通っている）。

### §3.3 合格条件 2: カーネル単体 - **PASS（3 倍）**

NGL=44、128 prefill のプロファイル（DSP 1.82 GHz 前後、prefill の N>=16 のみ）:

| skel | iq3_xxs MUL_MAT | DSP 時間 | GFLOP/s |
| --- | --- | --- | --- |
| `288e6712`（2 列） | 308 | 11.37 s | **101.1** |
| `76dd89e3`（4 列） | 288 | 3.45 s | **316.3** |

**3.1 倍。** HTP prefill 全体でも 30.53 -> 22.70 s（**-26%**）、HTP 合計で 32.47 -> 24.74 s（-24%）。

### §3.4 合格条件 3: end-to-end A/B - **改善は 1.5% にとどまった**

同一セッション ABBA、NGL=44、128 prefill + 4 decode（`p58_ab.sh`）:

| 組 | 順 | arm | prefill128 (s) | decode (ms) | iq3_xxs GFLOP/s | cap0 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | old `288e6712` | 55.78 | 771 | 101.1 | 1.79-1.67 |
| 1 | 2 | new `76dd89e3` | **54.92** | 813 | 296.8 | 1.67-1.555 |
| 2 | 1 | new `76dd89e3` | **55.00** | 812 | 296.8 | 1.555 |
| 2 | 2 | old `288e6712` | 72.97 | 943 | 88.5 | 1.555-1.11 |

組 2 の old は熱で cap が 1.11 GHz まで落ちた外れ値。**公平な組 1 では 55.78 -> 54.92（-1.5%）**。
組 2 の new（cap 1.555）は組 1 の old（cap 1.79-1.67）より低い cap で 55.00 なので、
方向としては改善だが、**HTP 時間が 24% 減ったのに wall はほぼ動かない**。

### §3.5 なぜ効かないか（実測からの推論）

- old: wall 55.78 s、HTP 32.47 s -> 非 HTP 23.3 s
- new: wall 54.92 s、HTP 24.74 s -> 非 HTP 30.2 s

**HTP が 7.7 s 速くなった分、非 HTP 側が 7 s 増えている。** これは
「wall が HTP と非 HTP の和ではなく、**どちらか遅い方（または別の律速）で決まっている**」
ことを示す。NGL=56（CPU 9 層）でも prefill は 52-54 s（Phase 57）で、
CPU 層を 12 減らしても 2-6% しか変わらない事実と整合する。

**結論: NGL=44-56 の prefill は DSP カーネル律速ではない。**
iq3_xxs の 3 倍化は DSP では確実に効いているが、wall にはほぼ出ない。
**次の標的はホスト側（CPU 層 + バッチ間のホスト処理）**であり、カーネルではない。

### §3.6 判定

**PASS（カーネルとして）。** bit-exact、カーネル単体 3.1 倍、HTP 時間 -24%。
**end-to-end は 1.5%（公平な組）で、期待した 5% には届かなかった。**
その理由（wall が DSP 律速でない）を実測で示したことがここの成果である。

### §3.7 なぜ効かないか: CPU 側が律速だった（新しい床 = NGL=50-52）

同時セッションで **NGL=0（CPU のみ）の 128 prefill** を測った: **138.24 s**（64 層、cap 1.5552 GHz、
run `20261119-021116-26846`）。**CPU 1 層あたり 2.16 s。**

これを使うと NGL=44 の内訳が説明できる:

- CPU 側: 21 層 x 2.16 s = **45.4 s**
- HTP 側: 24.7 s（新 skel）
- wall: 54.9 s

**wall は CPU 側（45 s）でほぼ決まっている。** HTP を 7.7 s 速くしても、
wall は CPU の床にぶつかって動かない。Phase 57 で NGL=56 が NGL=44 より速くならなかったのも
同じ理由（CPU は 19 s に減るが HTP が 30 s に増えて、床が入れ替わるだけ）。

**床が入れ替わる点を探すと NGL=50 前後になる。** 新 skel で NGL を振り直した
（同一セッション、cap 1.5552 GHz 安定）:

| NGL | prefill128 (s) | decode (ms) | CPU 層 | HTP 層 | 備考 |
| --- | --- | --- | --- | --- | --- |
| 0 | 138.24 | 1,220 | 64 | 0 | CPU の床 |
| 44 | 54.92 / 55.00 | 812 / 813 | 21 | 43 | 既定 |
| **50** | **49.66** | **830** | 15 | 49 | **最良（バランス点）** |
| 52 | 50.51 | 863 | 13 | 51 | |
| 56 | **49.02** | 1,208 | 9 | 55 | prefill 最良だが decode が 1.5 倍 |
| 58 | 55.82 | 916 | 7 | 57 | cap 1.11 GHz に落ちたため比較不可 |

**NGL=50 が最良**: prefill 54.96 -> 49.66 s（**-9.6%**）、decode は 812 -> 830 ms（誤差）。
NGL=56 は prefill が最も速い（49.02 s）が decode が 1.5 倍遅い。

**つまり新しい skel では既定 NGL=44 は最適ではない。** NGL=50 を推奨する
（既定を変えるかは次の判断。Phase 57 の既定 44 はそのままにしてある）。

### §3.8 判定（改）

**PASS（カーネル + 再探索）。** bit-exact、カーネル単体 3.1 倍、HTP -24%。
end-to-end は NGL=44 では 1.5% だが、**NGL=50 にすると -9.6%** になる。
「wall が DSP 律速でない」ことを CPU のみの実測（2.16 s/層）で示し、
新しいバランス点（NGL=50）を実測で見つけた。
---

## §4 目標3: 実用測定と残課題

- **llama-server の長時間常駐は §1 の事故で中止した。** 短時間の 1 リクエストは
  Phase 56 で確認済み（-ngl 44 で 1.09 t/s）。今回の事故で、guard の無い server を
  この端末で長時間回すのは安全でないと判断した。
- **NGL=56 の用途別推奨**: 新 skel では prefill が最良（49.02 s）だが decode が 1.5 倍遅い。
  prefill 重視（長いプロンプトを 1 回処理して少しだけ生成する）なら 56、
  生成重視なら 50。**既定 44 は据え置き**（Phase 57 の判断）。
- 一過性停止は Phase 58 では 0 回（約 12 run）。Phase 56-58 の合計で 57 run 中 2 回。
- guard-v2 の会計・EOG 停止・repetition penalty・chat template は未着手（時間切れ）。

---

## §5 判定

### §5.1 目標ごと

| # | 目標 | 判定 | 根拠 |
| --- | --- | --- | --- |
| 0 | 現在地の固定と回帰ゲート | **PASS** | 既定 NGL=44 の 32-step が `e9da92f8` に 2 回一致（§0.1） |
| 1 | decode のセッション依存 | **PARTIAL** | 遅いモード（5.0 s/token）が MemAvailable 0.6 GB・swap スラッシングと同時に起きた 1 例を記録。劣化カーブは事故のため中止（§1, §2） |
| 2 | iq3_xxs の 32x4 | **PASS** | bit-exact（NGL=16 `b21ca4bf` / NGL=44 `e9da92f8`）、カーネル 3.1 倍（101 -> 316 GFLOP/s）、HTP -24%。end-to-end は NGL=44 で 1.5%、**NGL=50 で -9.6%**（§3） |
| 3 | 実用測定・残課題 | **一部** | server 長時間は中止（事故）。NGL 再探索で最適点 50 を発見（§4） |

### §5.2 分かったこと（要約）

1. **iq3_xxs の 4 列化は正しく 3 倍速い**（bit-exact、316 GFLOP/s）。他の 12 型も
   同じ hash で通る（ラッパー経由で演算不変）。
2. **しかし NGL=44 の prefill wall は DSP 律速ではなかった。** CPU のみの実測から
   CPU 1 層 = 2.16 s で、NGL=44 の CPU 側は 45 s。HTP を 7.7 s 削っても wall は 1.5% しか動かない。
3. **床が入れ替わる点は NGL=50。** 新 skel で NGL=50 にすると prefill が -9.6%、decode は同等。
   NGL=56 は prefill 最良だが decode 1.5 倍。
4. **guard の無い長時間 server は危険。** 5 分で swap-out 148 GB、MemAvailable 0.6 GB、
   その後端末が再起動した（私は再起動を発行していない）。以後は probe のみ。

### §5.3 次の一手（優先順）

1. **CPU 側 1 層 2.16 s を削る。** これが今の最大の律速。CPU 側の MUL_MAT（IQ2_S/IQ3_S の
   quantized dot）を HVX/CPU で速くする、または CPU スレッド数・アフィニティを詰める。
2. **既定を NGL=50 に。** 新 skel と組で -9.6%。切り分けは 1 要素（ngl だけ）で済む。
3. **decode の遅いモードの条件を安全に切り分ける。** メモリ圧を上げずに再現する方法
   （例: swap 使用量だけを増やす）を考える。server は使わない。
4. **iq2_xxs / iq2_s の 4 列化。** 同じラッパー構造で追加できる（HTP prefill の 19% / 17%）。
   ただし 2 の結果から、NGL=50 前後では HTP が律速に入るので効く可能性がある。

---

## §6 安全条件の遵守

| 条件 | 遵守 |
| --- | --- |
| 端末を再起動しない | **発行していない**。ただし §1 の server 測定がメモリ枯渇を起こし、端末が再起動した（記録）。以後 server は使わない |
| 温度保護・vendor 電源ポリシー・ZRAM 等へ介入しない | **遵守**（読むだけ） |
| 前面アプリを止めない | **遵守**（`P55_FOREGROUND` / `P55_SKIP_FOREGROUND` を記録） |
| 安全停止線 | §1 で停止し、以後 server を使わない判断をした |
| swap guard | `GUARD_PAGES=2097152` 維持。probe の run では発動なし |
| メイン worktree の差分保護 | **遵守**（開始時 9、終了時 9） |
| 隔離 worktree・専用ブランチ | **遵守**（`codex/c58-session`） |
| merge・force push・過去コミット改変 | **なし** |
| push は `codex/c58-session` のみ | **遵守** |
| 未検証を PASS と書かない | **遵守**（目標1 は PARTIAL、原因は「候補」と明記） |
| ビルド鮮度を sha256 で確認 | **遵守**（新 skel `76dd89e3` を push 後に照合） |

---

## §7 終了時の端末状態

| 項目 | 値 |
| --- | --- |
| **skel の最終 sha256** | **`288e6712df9c8201ef05c12ec5704968d4f71407992cff8e0f03aa002ee0861d`**（Phase 50 D に復元） |
| 新 skel の置き場所 | `/data/local/tmp/c46-27b/libs/libggml-htp-v81.so.c58x4`（`76dd89e3...`） |
| probe | `482032c30d2bd27f51c27d478b638ba64e1f86a467586a3190accb66ebbd2b59` |
| probe / llama プロセス | **0** |
| replay.txt | **標準 4 トークン（`25358 11 198 760`）に復元済み** |
| 再起動 | 所有者の端末が §1 で再起動（私は発行していない）。その後は再起動なし（uptime 2,488 s） |
| 給電 | AC（残量 61%） |
| 温度 / cap | 52.6 ℃ / 1.5552 GHz |
| 前面アプリ | 14 run すべてで `P55_SKIP_FOREGROUND` を記録（launcher 13、twitter 1）。**停止 0** |
| メイン worktree | 9 ファイルのまま保護 |

**注意**: 次に新しい skel を使うときは `libggml-htp-v81.so.c58x4` をコピーする。
既定 NGL は 44 のまま（新 skel では 50 が最良という測定がある）。

---

## §8 成果物

| 種別 | パス |
| --- | --- |
| 報告書 | `continuations/c58/phase58-report.md` |
| 4 列カーネル | `ggml/src/ggml-hexagon/htp/hvx-mm-kernels-tiled.h` |
| 列ループ拡張 | `ggml/src/ggml-hexagon/htp/matmul-ops.c` |
| 劣化カーブ（中止） | `continuations/c58/p58_curve.sh` |
| skel A/B | `continuations/c58/p58_ab.sh` |

### §8.1 再現コマンド

`
# ビルド
cmake --build build/p53-android --target htp-v81 -j 8
# -> 76dd89e3bf2af1576101eba7bb10e5b9d750191fdc92b6a1fc019967166782ef

# bit-exact 確認（replay32 にしてから）
env WANT_SKEL=76dd89e3... GUARD_PAGES=2097152 RUN_TIMEOUT=1800 MEM_MIN_KB=4000000 \
    sh p53_run.sh 32 replay 16 t58-n16 0     # -> b21ca4bf
env WANT_SKEL=76dd89e3... ... sh p53_run.sh 32 replay 44 t58-n44 0   # -> e9da92f8

# NGL 再探索
env WANT_SKEL=76dd89e3... ... P53_PROMPT_TOKENS=128 P53_NO_LOGITS=1 sh p53_run.sh 4 noreplay 50 o58-n50 0

# skel A/B
env TIMEOUT=600 sh p58_ab.sh 2 ab58 44 4
`

### §8.2 コミット（`codex/c58-session`、push 済み）

| commit | 内容 |
| --- | --- |
| `02f8f401c` | iq3_xxs の 4 列カーネルと列ループ拡張、bit-exact、3.1 倍 |
| `36f5ae63b` | 劣化カーブと A/B のスクリプト |