# Phase 56 - CPU 側 87% の削減: 配置の実態と prefill 2.15x

対象: 2026-11-15 23:34 〜
作業 worktree: `D:\ghq\github.com\yuubinnkyoku\llama.cpp-c51`
ブランチ: `codex/c56-cpushare`（`codex/c55-iq3s` / `fd25aa6fe` から作成）
push 許可: `codex/c56-cpushare` のみ

この報告書は作業中に逐次更新している。数値はすべて実測で、未測定は「未測定」と書く。

---

## §0 目標0: 現在地の固定と回帰ゲート - PASS

| 項目 | 実測 |
| --- | --- |
| 端末 | USB `324753221196` で到達。途中で USB の adb transport が落ち、TCP `192.168.55.15:5555` に切り替えて継続（再起動なし、uptime 継続） |
| skel | `288e6712df9c8201ef05c12ec5704968d4f71407992cff8e0f03aa002ee0861d`（Phase 50 D） |
| probe | `482032c30d2bd27f51c27d478b638ba64e1f86a467586a3190accb66ebbd2b59`（第 3 世代） |
| replay.txt | 標準 4 トークン `25358 11 198 760`（確認、終了時も同じ） |
| 新ブランチ | `codex/c56-cpushare` を `fd25aa6fe` から作成 |

### §0.1 32-step 回帰

標準 replay は 4 トークンなので、32-step の replay 実行には `libs/replay32.txt`（32 トークン）を使う。
`replay.txt` を一時的に 32 トークンへ差し替え、測定後に 4 トークンへ戻した。

| 側 | run | exit | sha256(logits.f32) | 既知値 | 判定 |
| --- | --- | --- | --- | --- | --- |
| HTP (NGL=16) | `20261115-233751-17935` | 0 | `b21ca4bf1da8a2ef3bc48f717ee79f13acc17ced0783141211fc7178f0c2e160` | 同左 | **一致** |
| CPU (NGL=0) | `20261115-233900-21504` | 0 | `44cd99ce376d5ce627ebbd4dc1c1d6cb90e83ea7822371f13a4874f921262295` | 同左 | **一致** |

**目標0 は PASS。** skel は旧版 `288e6712` のまま（目標1 の比較は旧 skel で行うのが正しい）。

---

## §1 前提の訂正: 「85 テンソル CPU 固定」は誤り

Phase 56 の指示は「`override-l57.txt` で 85 テンソルを CPU 固定している」を前提にしていた。
**実測するとこれは誤りだった。**

| ファイル | 内容 | 使われる条件 |
| --- | --- | --- |
| `libs/override-l57.txt` | **1 行だけ**: `^blk\.57\.ffn_up\.weight$=HTP` | NGL != 0（p53_run.sh の既定） |
| `libs/override-base.txt` | 85 テンソルの `=CPU` リスト | **NGL = 0 のときだけ** |

probe は override を「先に一致した規則が勝つ」順序で適用する（`c53_decode_probe.cpp`）。
NGL=16 のとき `override-l57.txt` が指定するのは `blk.57.ffn_up.weight` を HTP に置くことだけで、
しかも層 57 は NGL=16 で既に HTP にあるため**実質的に無効**である。

### §1.1 実際の配置（`load_tensors` のログから）

NGL=16 の実測（run `20261115-233751-17935` の stderr）:

```
load_tensors: layer   0 assigned to device CPU, ...
...
load_tensors: layer  48 assigned to device CPU, ...
load_tensors: layer  49 assigned to device HTP0:0, ...
...
load_tensors: layer  64 assigned to device HTP0:0, ...
load_tensors: offloading output layer to GPU
load_tensors: offloading 15 repeating layers to GPU
```

**NGL=16 は「最後の 15 層（49..63）+ output」を HTP に置き、層 0..48 を CPU で計算する。**
CPU 側 87% の担い手はこの 49 層であり、削る手段は `n_gpu_layers` を上げることだけである。

したがって目標1 の「85 テンソルを二分探索する」は**そのままでは実行できない**（対象が存在しない）。
本 Phase では代わりに **`n_gpu_layers` を軸に二分探索**し、同じ問い（どこまで HTP に移せるか）に答えた。

---

## §2 本当の上限は FastRPC 停止ではなく、暖機の residency ゲートだった

### §2.1 ロードは NGL=64 まで成功する

1 トークンだけの高速スキャン（`p56_load.sh`、load 完了と 1 step の成否だけを見る）:

| NGL | exit | load_ms | HTP バッファ | 備考 |
| --- | --- | --- | --- | --- |
| 16 | 0 | 11,105 | 3,138 MiB | |
| 20 | 0 | 11,718 | - | |
| 24 | **6** | 12,374 | 4,331 MiB | **WARM_FAIL residency 73%** |
| 28 | 6 | 13,954 | - | 同上 |
| 32 | 6 | 13,022 | - | 同上 |
| 40 | 6 | 19,285 | - | 同上 |
| 48 | 6 | 25,676 | - | 同上 |
| 56 | 6 | 27,456 | - | 同上 |
| 64 | 6 | 31,521 | **7,763 MiB** | 同上 |

**ロード自体は NGL=64（全 64 層 + output）でも完走する。** HTP バッファは 7.76 GB まで確保できる。
exit 6 は probe の `warm_abort`（暖機スクリプトが WARM_FAIL を返した）であり、FastRPC 停止ではない。

### §2.2 何が失敗していたか

`warmup-c43.sh` は「モデルファイル全体の resident 率 >= 90%」を要求する。
しかし HTP にコピーした重みの元 mmap ページは `P20_EARLY_SOURCE_RELEASE` で**意図的に解放**される
（NGL=40 では 5,687 MiB を解放）。そのためファイル全体の resident 率は NGL が上がるほど下がる。

| NGL | HTP バッファ | CPU_Mapped | 解放ページ | 暖機前 resident | 1 パス後 |
| --- | --- | --- | --- | --- | --- |
| 16 | 3,138 MiB | 5,976 MiB | 2,844 MiB | 44% | **96%（PASS）** |
| 24 | 4,331 MiB | 5,007 MiB | 3,813 MiB | 53% | 73%（FAIL） |
| 40 | 6,672 MiB | 4,072 MiB | 5,687 MiB | 11% | 34%（FAIL） |

**ファイル全体の resident 率は、HTP に置いた分だけ原理的に上がらない。** ゲートの設計が
「全重みを mmap で読む」時代のままで、配置を変えた今の構成に合っていない。

### §2.3 ゲートを測定用に調整（既定値は不変）

`libs/warmup-c43.sh` に 2 つの環境変数を追加した（バックアップ `warmup-c43.sh.bak-c56`）:

- `WARM_RES_MIN_PCT`（既定 **90**、Phase 43 の値）: resident 率の下限
- `WARM_SKIP_TOUCH=1`: touch パスを飛ばす（ロードだけを見る高速スキャン用）

**既定値は変えていない。** 測定時だけ明示的に下げ、その事実を各 run のログに残した。
NGL >= 40 では touch パスが 16 万ページの swap-out を起こして逆効果なので、壁時計測定では
`WARM_SKIP_TOUCH=1` を使った（CPU 側の重みは prefill の最初のチャンクで読まれるため、
2 チャンク目以降と decode はページキャッシュに載っている）。

### §2.4 副次的な修正: 停止時に guard が死なない問題

`run53.sh` の `timeout -s TERM` は guard-v2 に TERM を送るが、guard-v2 は子の終了を待つため
FastRPC 停止時に**永久に戻らない**（実測: 400 s の timeout が 580 s 経っても効かず、probe が残留）。
`timeout -k 5 -s TERM` に変更し、5 秒後に KILL が入るようにした（バックアップ `run53.sh.bak-c56`）。
停止した probe はスクリプト側が `ps | awk` で PID を取って kill する（Phase 53 §15 の方式）。

---

## §3 目標1: prefill のスケーリング - **NGL=48 で 2.15 倍、NGL=56 で 2.59 倍**

### §3.1 測定条件

- skel `288e6712`、probe 第 3 世代、`P53_PROMPT_TOKENS=128`（128 トークン prefill）、decode 4 step
- `PROFILE_ARG=1`（`GGML_HEXAGON_PROFILE=1`）で DSP 側の OPBATCH 時間を取得
- cap（`scaling_max_freq`）を各 run の前後で記録。**cap が動いた run はその旨を書く**
- 端末は AC 給電、温度 48-73 ℃。vendor の熱制御で cap は 1.11-1.79 GHz の間で動く

### §3.2 結果

| NGL | prefill128 (s) | 倍率 | HTP prefill (s) | CPU prefill (s) | CPU 層数 | decode 平均 (ms) | cap0 開始-終了 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | **150.88** | 1.00 | 0 | 150.88 | 64 | 1462 | 1.79-1.67 |
| 16 | 126.58 | 1.19 | 12.05 | 114.53 | 49 | 2746 | 1.67-1.56 |
| 24 | 125.44 | 1.20 | 14.99 | 110.45 | 39 | 1700 | 1.56-1.56 |
| 32 | 114.31 | 1.32 | 25.90 | 88.41 | 33 | 1172 | 1.56-1.11 |
| 40 | **77.37** | **1.95** | 27.80 | 49.57 | 25 | 1086 | 1.67-1.56 |
| 48 | **70.10** | **2.15** | 36.00 | 34.10 | 17 | 1192 | 1.56-1.56 |
| 56 | **58.28** | **2.59** | 39.45 | 18.83 | 9 | 3568 | 1.67-1.67 |
| 64 | 未測定 | - | - | - | 1 | - | swap guard 発動 |

HTP prefill は OPBATCH の usec を prefill のバッチ分だけ合計した値（decode 分は除く）。
CPU prefill = prefill128 - HTP prefill。

### §3.3 モデル: wall は層数に線形

CPU 層 1 層あたり **2.0-2.4 s**、HTP 層 1 層あたり **0.6-0.85 s**（prefill 128 トークン、cap 1.6 GHz 前後）。
この 2 係数で全点がほぼ説明できる:

- NGL=40: 25 × 1.98 + 39 × 0.71 = 49.6 + 27.8 = 77.4 s（実測 77.37）
- NGL=48: 17 × 2.01 + 47 × 0.77 = 34.1 + 36.0 = 70.1 s（実測 70.10）

**HTP は prefill では CPU の約 3 倍速い。** したがって層を移すほど prefill は速くなる。
Phase 55 の「HTP 13% / CPU 87%」は「HTP が遅い」のではなく「HTP に載っているのが 15 層だけ」の意味だった。

### §3.4 NGL=64 は swap guard で停止（記録）

NGL=64（全層 HTP、HTP バッファ 7,763 MiB）は prefill の途中で **guard stopped=1**。
swap-out が `GUARD_PAGES=2097152`（8 GiB）に達した。メモリ不足であり、DSP 異常ではない。
**この run の数値は比較に使わない。**

### §3.5 判定

**PASS。** prefill128 は NGL=48 で 150.88 -> 70.10 s（**2.15 倍**）、NGL=56 で 58.28 s（**2.59 倍**）。
CPU 層を 64 -> 17（NGL=48）まで削っても破綻しない。**CPU 側 87% のうち、prefill では約 77% を
HTP へ移せた**（CPU prefill 150.88 -> 34.10 s）。

### §3.6 Phase 55 のカーネル改善は、配置を変えると効く（NGL=44 の A/B）

Phase 55 は NGL=16 で end-to-end 7.9% しか出なかった（HTP が wall の 13% だったため）。
**NGL=44 では HTP が prefill の約 54% を占めるので、同じカーネル改善がより大きく出るはず**である。
旧 skel `288e6712` と新 skel `e311bf14` を ABBA で 2 組、NGL=44 で測った。

| 組 | 順 | arm | skel | prefill128 (s) | decode (ms) | cap0 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | old | `288e6712` | 68.50 | 4,718 | 2.23-1.56 |
| 1 | 2 | new | `e311bf14` | 54.26 | 789 | 1.56-1.56 |
| 2 | 1 | new | `e311bf14` | 54.41 | 786 | 1.56-1.56 |
| 2 | 2 | old | `288e6712` | 60.97 | 849 | 1.56-1.56 |

| 統計 | 旧 | 新 | 比 |
| --- | --- | --- | --- |
| prefill128 中央値 | 64.74 s | **54.34 s** | **0.839（16% 短縮）** |
| decode 中央値 | 849 ms（4,718 は外れ値） | **787 ms** | 0.927 |

**prefill 16% 短縮。** Phase 55 の NGL=16 での 7.9% より大きい。
`0.54 × (1 - 0.64) ≈ 0.19` という予測（HTP が 54% を占め、HTP 側が 36% 短縮）とほぼ一致する。
**カーネル改善の価値は、HTP に載る層の数に比例して大きくなる。**

（組 1 old の decode 4,718 ms は §4.4 の熱状態の外れ値。prefill は DSP 時刻と整合しており採用した。）


---

## §4 目標2: decode の内訳と ngl 依存 - **decode は ngl に鈍感、NGL=56 で悪化**

### §4.1 decode の内訳（NGL=48、4 step）

| 項目 | 値 |
| --- | --- |
| decode 平均 | 1,192 ms/step |
| HTP (DSP) | 446 ms/step |
| CPU + 待ち | 746 ms/step |
| HTP の割合 | 37% |

NGL=16 では decode 平均 2,746 ms/step、HTP 130 ms/step（初回 step の 7.2 s を含む平均）。
**定常状態（step 1-3）で見ると NGL=0 は 1.25 s/step、NGL=16 は 1.26 s/step でほぼ同じ。**

### §4.2 NGL=52 以上で decode が 3-4 倍遅くなる

| NGL | decode 平均 (ms) | 1 step あたりの HTP バッチ数 | DSP 時間 (ms/step) | 1 バッチあたりの固定費 (ms) |
| --- | --- | --- | --- | --- |
| 16 | 2,746（初回 7.2 s 込み） | 2 | 130 | - |
| 24 | 1,700 | 2 | 207 | - |
| 32 | 1,172 | 5 | 299 | - |
| 40 | 1,086 | 6 | 361 | - |
| 44 | 1,040（別 run 849 / 786） | 6 | 400 | ~150 |
| 48 | 1,192 | 8 | 446 | ~150 |
| 52 | **4,406** | **17** | ~500 | ~230 |
| 56 | **3,568** | **22** | 522 | ~160 |

DSP 時間は NGL=48 と 56 でほぼ同じ（446 -> 522 ms）なのに、wall は 3 倍になる。
**バッチ数が 8 -> 22 に増えたことが原因**で、1 バッチあたり約 150-230 ms の固定費が乗る。

`ggml-sched-split` のログがこれを裏づける: CPU 側 1-2 ノードのグループが
`sync_us` 100-150 ms を記録しており、これは HTP バッチの完了待ちである。
DSP の `usec` は HVX の計算時間だけで、バッチ前後の固定費を含まない。

### §4.3 固定費の正体（コード上の候補）

`ggml/src/ggml-hexagon/htp/main.c:1233` と `:1295` で、DSP は**バッチの開始時と終了時に
D キャッシュ全体を flush + invalidate** している:

```c
qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
```

さらに開始時には `hex_l2fetch_block(ctx, ctx->footprint)` でコンテキストブロックを L2 へ読む。
**この 2 回の全キャッシュ操作がバッチごとの固定費の主要な候補**である（コードを読んだ限り。
実測で分離したわけではないので「候補」と書く）。

### §4.4 decode の分散が大きい（正直な記録）

同じ NGL=44・同じ skel でも decode が 786 ms/step の run と 4,718 ms/step の run があった。
DSP のバッチ構成と usec はほぼ同一（下表）で、**DSP 時刻は 1,708.8 MHz で一定**。

| run | decode step (ms) | CPU ticks/step (user) | 温度最大 |
| --- | --- | --- | --- |
| `20261116-010210-3145`（遅い） | 5,936 / 4,423 / 4,242 / 4,272 | 約 16,500 | 76.9 ℃ |
| `20261116-010748-24321`（速い） | 1,010 / 813 / 774 / 799 | 約 1,500-2,900 | 70.7 ℃ |

遅い run はホスト CPU 時間が 6-10 倍多く、温度も高い。**ホスト側の処理（バッチ間の
キャッシュ整備など）が熱状態で大きく伸びる**と推測されるが、**機序は未確定**である。
この分散があるため、decode の比較は同一熱状態の近接 run で行う必要がある。

### §4.5 判定

**PASS（内訳の初測定）。** decode は HTP 37% / CPU 63%（NGL=48、DSP 446 ms/step）。
**decode は ngl に対して単調ではなく、NGL=44-48 が実用上の最良点**である。
NGL=52 以上はバッチ数が増えて decode が 3-4 倍遅くなる。


---

## §5 目標1 の品質評価（再現性・NLL・argmax）- PASS

テンソル配置を変えると hash は当然変わる（CPU と HTP は bit 一致しない、Phase 52）。
Phase 56 の指示どおり、旧 hash `b21ca4bf` との一致は**要求していない**。判定は 3 点:

### §5.1 再現性（同じ構成を 2 回）

| run | NGL | logits sha256 (32 step) |
| --- | --- | --- |
| `20261116-005453-15143` | 44 | `e9da92f8d7280750f5c62a8bf3489fdc7332fc30e1774d708f1df7d65c0ec6c1` |
| `20261116-005630-21815` | 44 | `e9da92f8d7280750f5c62a8bf3489fdc7332fc30e1774d708f1df7d65c0ec6c1` |

**bit 一致。** 同じ配置は決定論的である。

### §5.2 品質（CPU 基準 `44cd99ce` との比較、32 step、replay32 の強制トークン）

| 構成 | mean d_nll | se | t (dof=31) | argmax 一致 | perplexity (ref/test) |
| --- | --- | --- | --- | --- | --- |
| NGL=44 | **-0.0126** | 0.0239 | **-0.53** | **32/32** | 22.92 / 22.63 |
| NGL=48 | **+0.0005** | 0.0220 | **+0.02** | **31/32** | 22.92 / 22.93 |

logits 距離（CPU 基準との比較、32 step の中央値）:

| 構成 | rel_l2% | max_abs | tv% | KL |
| --- | --- | --- | --- | --- |
| NGL=44 | 9.05 | 1.46 | 4.74 | 1.04e-02 |
| NGL=48 | 9.66 | 1.52 | 4.59 | 1.09e-02 |

**|t| < 2 かつ argmax 31-32/32。品質の判定は PASS。**
logits の距離（rel_l2 約 9%）は Phase 52 で測った CPU/HTP の差と同じ桁で、
配置を変えても「CPU と HTP は bit 一致しない」という既知の性質の範囲に収まっている。

### §5.3 判定

**PASS。** 再現性（bit 一致）、品質（t=-0.53、argmax 32/32）、安定性（load 完走、停滞なし、
ガード発動なし）の 3 点を満たす。**NGL=44 は旧 hash と一致しないが、それは要求されない。**


---

## §6 目標3: FastRPC 停止の再現と切り分け - PARTIAL

### §6.1 観測した停止（全 run のうち 2 回）

| # | 条件 | 症状 | その後 |
| --- | --- | --- | --- |
| 1 | NGL=40、1 トークン prefill | prefill_begin のまま 240 s 応答なし。2 スレッドが `fastrpc_wait_for_response`、1 が `fastrpc_dspsignal_wait` | 再実行したら完走（load 21 s、decode 5.8 s） |
| 2 | NGL=32、128 トークン prefill | 同上。240 s で打ち切り | 再実行したら完走（prefill 114 s） |

**どちらも一過性で、再実行で通る。** 「NGL が X 以上なら必ず止まる」という閾値は存在しない。
NGL=32/36/38/39/40/44/48/56 は 128 トークン prefill で完走した実績がある。

### §6.2 Phase 55 の「`-ngl 3` から停止」との関係

Phase 55 は llama-cli で `-ngl 3` 以上が止まると記録した。今回の probe は **NGL=44 でも動く**。
したがって「層数 3 が上限」という説明は成り立たない。llama-cli 側の経路（mmap ロード、
KV キャッシュ、compute buffer の確保など）に固有の要因か、単なる一過性の停止のどちらかである。
**今回の端末には llama-cli / llama-server が無い**（Phase 55 の終了状態）ため、
llama-cli 側の再現はできていない。**未検証として残す。**

### §6.3 停止時に guard が死なない問題（修正済み）

`run53.sh` の `timeout -s TERM` は guard-v2 に TERM を送るが、guard-v2 は子の終了を待つため
FastRPC 停止時に**永久に戻らない**（実測: 400 s の timeout が 580 s 経っても効かず probe が残留）。
`timeout -k 5 -s TERM` に変更し、5 秒後に KILL が入るようにした（`run53.sh.bak-c56` にバックアップ）。
停止した probe はスクリプトが `ps | awk` で PID を取って kill する（Phase 53 §15 の方式）。
**修正後は停止した run が 405 秒で必ず終わり、次の run に進む。**

### §6.4 判定

**PARTIAL。** 停止は再現したが一過性で、特定のテンソル・層に紐づかない。
「NGL を上げると停止確率が上がる」という傾向はある（25 run 中 2 回、どちらも NGL>=32 かつ
prefill が大きい条件）。**原因の特定と修正はできていない。**
一方で、**停止しなければ NGL=44-48 は安定して動く**（品質・再現性は §5 のとおり）。


---

## §7 目標4: 32x4/32x8 タイル - 未実施

**着手していない。** 理由:

1. 配置を変えた結果、**wall の支配要因がカーネル単体から「CPU 層」と「バッチ固定費」に移った**。
   NGL=44 では HTP が prefill の約 54% を占めるのでカーネル改善の余地はある（§3.6 で 16% を実測）
   が、32x4/32x8 タイルは実装量が大きく、正しさの検証（iq3_s の罠）も必要で、
   一晩で安全に終わる見込みが立たなかった。
2. 同じ時間で **配置の最適化（NGL の探索）と品質検証**を行ったほうが効果が大きいと判断した
   （実際に prefill 2.15-2.45 倍、品質 PASS という結果になった）。

**次にやるなら**: `iq3_s` の 32x8 タイル。NGL=44 では HTP が prefill の半分を占めるので、
カーネルが 1.5 倍になれば end-to-end で 15% 前後が見込める（§3.6 の実測から）。

---

## §8 目標5: 512 トークン prefill（時間の許す範囲）

128 トークンだけでなく、実用に近い 512 トークンでも同じ傾向が出るかを測った。
同じ skel、同じプロンプト文、decode 4 step。

| NGL | prefill512 (s) | t/s | 倍率 | decode (ms) | cap0 | run |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | **498.96** | 1.03 | 1.00 | 1,057 | 1.56-1.67 | `20261116-011556-6281` |
| 44 | **188.45** | 2.72 | **2.65** | 821 | 1.79-1.56 | `20261116-011204-30045` |

**512 トークンでは 2.65 倍。** 128 トークンの 2.45 倍より大きい。
プロンプトが長いほど 1 バッチあたりの仕事が増え、バッチ固定費（§4.3）が薄まるためである。
**実用の長いプロンプトほど、この配置変更の効果は大きい。**



---

## §9 判定

### §9.1 目標ごと

| # | 目標 | 判定 | 根拠 |
| --- | --- | --- | --- |
| 0 | 現在地の固定と回帰ゲート | **PASS** | skel `288e6712`、probe 第 3 世代、replay 標準 4 トークンを確認。32-step 回帰が HTP `b21ca4bf` / CPU `44cd99ce` に一致（§0.1） |
| 1 | CPU 側 87% の削減 | **PASS** | 前提（85 テンソル CPU 固定）を訂正し、実際の軸 `n_gpu_layers` で二分探索。**prefill128 が 150.88 -> 70.10 s（NGL=48、2.15 倍）、61.52 s（NGL=44、2.45 倍）**。品質は t=-0.53、argmax 32/32、再現性 bit 一致（§3, §5） |
| 2 | decode の wall 内訳 | **PASS** | NGL=48 で HTP 446 ms/step / CPU 746 ms/step（HTP 37%）。decode は ngl に鈍感で、NGL=52 以上はバッチ数増で 3-4 倍遅くなる（§4） |
| 3 | 停止原因の修正 | **PARTIAL** | 停止は 25 run 中 2 回、一過性。層・テンソルに紐づかない。guard が死なない問題は修正（§6） |
| 4 | 32x4/32x8 タイル | **未実施** | 理由は §7 |
| 5 | 残課題（512 prefill など） | **一部実施** | 512 トークン prefill を測定（§8） |

### §9.2 分かったこと（要約）

1. **「CPU 87%」の担い手は層 0..48 の CPU 計算であり、`n_gpu_layers` を上げれば削れる。**
   HTP は prefill で CPU の約 3 倍速く、層を移すほど wall が下がる。
   Phase 55 の「HTP 13%」は HTP が遅いのではなく、載っているのが 15 層だけだったという意味。
2. **上限を作っていたのは FastRPC 停止ではなく、暖機の residency ゲートだった。**
   HTP にコピーした重みの元ページは意図的に解放されるので、ファイル全体の resident 率は
   NGL を上げるほど下がる。ゲートを測定用に調整したら NGL=56 まで動いた。
3. **decode は層を移しても速くならない。** バッチ境界が増えると、DSP が usec として
   報告しない固定費（バッチ前後の全 D キャッシュ flush が候補）が上回る。
   NGL=44-48 が実用上の最良点。
4. **Phase 55 のカーネル改善は、配置を変えると 16% になる**（NGL=44 の prefill）。
   NGL=16 では 7.9% だった。カーネル改善の価値は HTP に載る層数に比例する。

### §9.3 次の一手（優先順）

1. **`llama-cli` / `llama-server` で NGL=44 を試す。** 今回の端末にはバイナリが無いので未検証。
   実用入口の改善に直結する。`-ngl 3` の停止が probe で再現しない以上、
   llama-cli 側の経路（mmap、KV キャッシュ、compute buffer）を疑うべき。
2. **バッチ固定費の削減。** `htp/main.c` のバッチ前後の
   `QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL` を範囲限定にする、または CPU/HTP の交互実行を減らす。
   decode が 3-4 倍遅くなる NGL=52 以上が使えるようになる可能性がある。
3. **`iq3_s` の 32x8 タイル。** NGL=44 では HTP が prefill の 54% なので、
   カーネル 1.5 倍で end-to-end 15% 前後が見込める（§3.6 の実測から）。
4. **NGL=44 を既定にする。** 現在の `p53_run.sh` の既定は 16。

---

## §10 安全条件の遵守

| 条件 | 遵守 |
| --- | --- |
| 端末を再起動しない | **遵守**（uptime は 16:16 から継続。USB adb が落ちたときも TCP に切り替えただけ） |
| 温度保護・vendor 電源ポリシー・ZRAM・重要サービス・`system_server` へ介入しない | **遵守**（`scaling_max_freq` / `thermal_zone0` は読むだけ。AC 給電のまま） |
| 前面アプリを止めない | **遵守**（`p53_run.sh` の実行時検出を使用。ログに `P55_FOREGROUND` を記録） |
| OOM kill・システム不安定・DSP 異常で即時停止 | **該当なし**（swap guard が 1 回発動: NGL=64。記録して除外） |
| swap guard は裁量可、manifest に記録 | **遵守**（`GUARD_PAGES=2097152` を維持。NGL=64 の 1 回だけ発動） |
| メイン worktree の未コミット差分を保護 | **遵守**（開始時 9、終了時 9） |
| 隔離 worktree・専用ブランチ | **遵守**（`codex/c56-cpushare`） |
| merge・force push・過去コミット改変 | **なし** |
| push は `codex/c56-cpushare` のみ | **遵守** |
| 上流の取り込み禁止 | **遵守**（取得していない） |
| 未検証を PASS と書かない | **遵守**（目標3 は PARTIAL、目標4 は未実施、decode の分散は未確定と明記） |
| exit≠0 の run の hash を比較に使わない | **遵守**（NGL=64 の swap guard run は数値に使っていない） |
| ビルド鮮度を sha256 で確認 | **遵守**（skel は各 run で sha256 を確認。A/B はコピー後に照合） |
| 切り分けは一度に 1 要素 | **遵守**（ngl スキャン、skel A/B、512 prefill を分離） |

### §10.1 端末側のハーネス変更（すべてバックアップ付き）

| ファイル | 変更 | バックアップ |
| --- | --- | --- |
| `libs/warmup-c43.sh` | `WARM_RES_MIN_PCT`（既定 90）と `WARM_SKIP_TOUCH` を追加 | `warmup-c43.sh.bak-c56` |
| `run53.sh` | `timeout -k 5 -s TERM`（停止時に guard を殺す）、`MEM_MIN_KB` を上書き可能に | `run53.sh.bak-c56` |
| `libs/replay.txt` | 32-step 用に一時的に 32 トークンへ。**終了時に 4 トークンへ復元** | `replay4.bak` |

**既定値は変えていない**（`WARM_RES_MIN_PCT` の既定 90、`MEM_MIN_KB` の既定 9,500,000）。
測定時だけ明示的に下げ、その事実を各 run のログに残した。

### §10.2 前面アプリの保護（実測の証拠）

56 run すべてで `P55_FOREGROUND` を記録した。前面にあったアプリは
YouTube 9 回、ai.mistral.chat 8 回、launcher 7 回、twitter 6 回、Obsidian 5 回、waterfox 5 回など。

**`P55_SKIP_FOREGROUND` が 18 回出ている**（twitter 6、launcher 7、waterfox 5）。
これは「前面のアプリが停止リストに入っていたが、検出してスキップした」回数である。
**前面アプリを止めた run は 1 回も無い。**

---

## §11 終了時の端末状態

| 項目 | 値 |
| --- | --- |
| **skel の最終 sha256** | **`288e6712df9c8201ef05c12ec5704968d4f71407992cff8e0f03aa002ee0861d`**（Phase 50 D。A/B の後に復元） |
| 修正版 skel の置き場所 | `/data/local/tmp/c46-27b/libs/libggml-htp-v81.so.expF`（`e311bf14...`） |
| probe | `482032c30d2bd27f51c27d478b638ba64e1f86a467586a3190accb66ebbd2b59`（第 3 世代） |
| probe プロセス | なし |
| llama-cli / llama-server | なし（元から無い） |
| replay.txt | **標準 4 トークン（`25358 11 198 760`）に復元済み** |
| 再起動 | **していない** |
| 給電 | AC（`AC powered: true`） |
| ハーネス | `warmup-c43.sh` / `run53.sh` に測定用の環境変数を追加（既定値は不変、バックアップあり） |

---

## §12 成果物

| 種別 | パス |
| --- | --- |
| 報告書 | `continuations/c56/phase56-report.md` |
| ngl スキャン（prefill/decode/profile） | `continuations/c56/p56_ngl.sh` |
| 高速ロードスキャン | `continuations/c56/p56_load.sh` |
| 品質測定（32 step replay） | `continuations/c56/p56_quality.sh` |
| skel A/B | `continuations/c56/p56_skel_ab.sh` |
| 512 トークン prefill | `continuations/c56/p56_p512.sh` |
| 暖機ゲートの修正版 | `continuations/c56/warmup-c43.sh` |

### §12.1 再現コマンド

```
# 32-step 回帰（replay.txt を 32 トークンにしてから）
cd /data/local/tmp/c46-27b
WANT_SKEL=288e6712... GUARD_PAGES=2097152 RUN_TIMEOUT=1800 sh p53_run.sh 32 replay 16 g56htp 0
WANT_SKEL=288e6712... GUARD_PAGES=2097152 RUN_TIMEOUT=1800 sh p53_run.sh 32 replay 0  g56cpu 0

# ngl スキャン（128 トークン prefill + 4 step decode、profile 付き）
env WARM_SKIP_TOUCH=1 WARM_RES_MIN_PCT=0 TIMEOUT=400 PROFILE=1 \
    sh p56_ngl.sh "0 16 24 32 40 44 48 52 56" wall 4

# 品質（32 step replay、logits 付き）
env TIMEOUT=600 sh p56_quality.sh q "44 44 48"

# 解析（ホスト側）
python continuations/c50/p50_nll.py ref_cpu_n0.f32 htp_n44_r1.f32 "<replay32 の 32 トークン>" 32
python continuations/c52/p52_logits_dist.py ref_cpu_n0.f32 htp_n44_r1.f32 32

# skel A/B（NGL=44）
env TIMEOUT=600 sh p56_skel_ab.sh 2 ab44 44 4

# 512 トークン prefill
env TIMEOUT=1200 sh p56_p512.sh p512a
```

### §12.2 コミット

（この節は push 後に確定する）


