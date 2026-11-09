# Phase 49.5 - Phase 49 の確定・検証・Phase 50 の準備

2026-11-09 / 2026-11-10。作業ブランチ `codex/c49-perf`、HEAD `533b03638`。
Phase 49 の報告 (`phase49-final-report.md`, `phase49-round1..3`) は変更していない。

判定: **READY WITH RISKS**。保存・正当性・測定基準は揃った。ただし (1) 今夜の端末は
性能測定の環境条件を満たさず、fast cap の新しい比較値を取れなかった、(2) 27B の logits 差分が
生成品質に与える影響は未評価、の 2 点を明示する。

---

## 1. Phase 49 の GitHub 保存

- 監査: 5 コミット、10 ファイル、+397 / -41。`git diff --check` は無出力。
  追加は全てテキスト (C/C++ ヘッダ、Markdown)。バイナリ、巨大ログ、鍵、個人情報、
  端末固有パスの混入なし。
  変更ファイル: `.agents/skills/...` 2 件、`ggml/src/ggml-backend.cpp`、
  `ggml/src/ggml-hexagon/ggml-hexagon.cpp`、`ggml/src/ggml-hexagon/htp/` 6 件。
- push: `codex/c49-perf` を origin へ通常 push。強制 push なし。main への merge なし。
- 確認: `git rev-parse HEAD` = `git ls-remote origin codex/c49-perf` = `533b03638ad263b54bc6c3a6674cfc9753041f9d`。
- 端末に配備した実体も同じ: `libggml-htp-v81.so` sha256
  `a87330ad...` (ローカルビルドと一致)、`libggml-hexagon.so` `d6f60f65...`、
  `libggml-base.so` `d950aa96...`、`libllama.so` `29d198a4...`。

## 2. Skill の統合

GitHub `codex/overnight-autonomy-phase49` (`1b7c599`, `f72f222`) と、ローカルの
`ef4482259` を比較した。前者は 28 行の作業契約、後者は 93 行の規則集で、重複は
「Phase 48 を上書きしない」「34 テンソル目」「HexaTrain」の 3 点だった。

cherry-pick はせず、上流にしか無い内容だけをローカルへ取り込んだ:

- Phase 49 のファイルが Phase 48 の引き継ぎを上書きしない、という前置き。
- `FLOAT32` 宣言は実行精度を保証しない (HexaTrain の Muon 反復で実測) という検証原則。
- 34 テンソル目の欠陥で優先作業を停止しない。
- `GGML_HEXAGON_SPLIT=1` を既定にしない、33 テンソルの分割実装は比較用に残す。

新規に `references/hexagon-phase50.md` を作り、Phase 49 の確定値と Phase 50 の測定条件・
合格条件を書いた。`SKILL.md` は Phase 50 の資料を先に読む形へ 1 箇所だけ変更した。
共通規則は `SKILL.md` に一本化したまま。Phase 47/48 の古い制約は復活させていない。

## 3. 正当性の再確認

### 3.1 BF16 行列積

`p49_bf16_test` を再実行 (2026-11-10 00:07)。生出力は `p495-bf16-test.log`。

| 項目 | 結果 |
|---|---|
| BF16 全ビットパターンの変換 | 65536 checked / 65536、不一致 0 |
| one-hot 順列の同一性 | bad 0 |
| dot K=5120 M=1 | max_abs 1.51e-05、上界 6.27e-04、rel_l2 2.19e-07、超過 0、非有限 0 |
| dot K=5120 M=2 | max_abs 1.35e-05、上界 6.35e-04、rel_l2 2.32e-07、超過 0 |
| dot K=5120 M=3 | max_abs 2.39e-05、上界 6.20e-04、rel_l2 2.30e-07、超過 0 |
| dot K=5152 (端数あり) | max_abs 1.68e-05、rel_l2 2.90e-07、超過 0 |
| dot K=96 | max_abs 1.39e-06、上界 1.34e-05、rel_l2 1.47e-07 |
| 非有限出力 | 0 |
| 型・形状ゲート | 連続 bf16、K%32==0、F32 活性化のみ許可。他は CPU へ戻る (Phase 49 から変更なし) |
| 実行 | `LD_LIBRARY_PATH=libs ADSP_LIBRARY_PATH=... ./p49_bf16_test` (`exact`/`dot`/`tail`/`perm`/`bench` で部分実行可) |

上界は測定前に決めた `4 eps sum|w x|`。結果を見て緩めていない。
`ADSP_LIBRARY_PATH` を渡さないと skel を開けずに落ちる (0x80000406)。

### 3.2 IQ3 gather 最適化

- 4 step replay (NGL=9) の logits sha256 = `5585d4edc47ee7505c558aeb1b929a7eb7089e238d1b2d81a3ffcfa4958bd480`。
  Phase 49 の基準ファイル `lg-ngl9.f32` と**バイト一致**。3973120 バイト。
- 同じく NGL=0 の logits sha256 = `577417e9fd33912282f964f8c99e250fadc1b5605453f04fcd5bbea23ef3560b`。
  Phase 49 の `lg-ngl0.f32` と**バイト一致**。
- つまり命令の並べ替え後も値は変わらず、しかも別セッション・別温度状態で再現する。
- スクラッチは 1024 B/thread (IQ3_S, IQ3_XXS とも)。`HTP_IQ2S_GRID_VTCM_RESERVE` 128 KB の
  中に収まり、ワーカーごとに `l` (0..HTP_MAX_NTHREADS-1) で区切られる。重複なし。
  VTCM 使用量は Phase 49 から変わっていない (確保量を増やしていない)。
- DSP 異常 (セッション切断、ハング、センチネル異常、離散的な値の化け) は今夜のモデル run 12 回で
  発生していない。落ちたのは swap-out ガードによる停止 (exit 6) 2 回だけ。
- **帯域の再現**: `GGML_HEXAGON_PROFILE=1` の 5 step 実行 (NGL=9、cap 1785600) で、DSP の
  型別実効帯域を測り直した。step 2 でガードに掛かったので 2 step 分 (706 op) の集計。

  | 型 | 本数 | 時間 us | MB | GB/s | Phase 49 の値 |
  |---|---:|---:|---:|---:|---:|
  | iq3_s | 32 | 28698 | 806.5 | **28.10** | 28.02 |
  | iq3_xxs | 36 | 39801 | 875.1 | **21.99** | 21.90 |
  | iq4_xs | 13 | 37024 | 2293.7 | 61.95 | 64.95 |
  | iq2_s | 6 | 7529 | 73.9 | 9.81 | 9.85 |
  | iq2_xs | 2 | 6596 | 51.5 | 7.81 | 7.81 |
  | q2_K | 8 | 3591 | 154.8 | 43.12 | - |
  | q4_K | 4 | 3360 | 200.5 | 59.68 | 61.90 |
  | bf16 | 12 | 176 | 5.9 | 33.51 | 34.45 |

  IQ3_S と IQ3_XXS が Phase 49 の最適化後の値と 0.3% 以内で一致した。**IQ3 gather の効果は
  再現している**。bf16 の op が 12 本、33.51 GB/s で動いていることも同時に確認できた
  (`blk.56.ssm_alpha.weight`、usec 27)。全体は 35.20 GB/s、76.7 GMAC/s。

### 3.3 スケジューラ計測

独立試験 `p495_sched_timing_test` を新規に書いた (`p495_sched_timing_test.cpp`)。

- `mix` モード: 2 スレッドがそれぞれ自分の sched で 300 graph を回す。
  ログ 60 行、全行 `calls 10`。共有状態なら 20 になる。nodes 1 と nodes 3 が 30 行ずつで、
  2 つのスケジューラが混ざっていない。
- 不変条件 `sync + copy + compute == total`: 60 行で最大差 1 us (整数除算の丸め)、
  負値 0、`compute > total` 0。二重計上なし。
- `async` モード: 20 ms かかる処理を非同期に投入するモック backend を使う。
  投入は `compute_us 80` (20 ms ではない)、待ちは消費側 split の `sync_us 23375`、
  最終 synchronize は `ggml-sched-wait backend MOCK avg_us 21138`。
  **非同期投入時間を実行時間として数えていない**ことを独立に示す。
- `GGML_SCHED_SPLIT_TIMING` 未設定では `ggml-sched-*` 行が 1 行も出ない。
  コード上も `split_timing ? ggml_time_us() : 0` で、無効時に時計を読まない。
- 計測値の帰属: `total` は前 split の完了待ち + 入力コピー + backend 呼び出し。
  HTP の `compute_us` は投入復帰までで、DSP 実行時間は timeline (`wait_us`) 側で測る。

## 4. 27B 全体の数値

同じ replay token 列 (25358 11 198 760)、4 step、同じビルド。指標は
`p495_logits_cmp.py` で計算した (vocab 248320)。

| 比較 | step0 rel_l2 | step1 | step2 | step3 | max_abs (最大) | 非有限 |
|---|---:|---:|---:|---:|---:|---:|
| CPU vs HTP NGL=9 | 2.04% | 2.74% | 3.92% | 3.06% | 0.745 | 0 |
| CPU vs Phase 48 HTP | 1.99% | 2.24% | 2.77% | 1.59% | 0.337 | 0 |
| Phase 48 HTP vs Phase 49 HTP | 0.88% | 2.02% | 2.91% | 2.61% | 0.725 | 0 |

分布と順位 (CPU を基準):

| step | mean_abs | RMS | TV 距離 | KL(CPU||HTP) | argmax | top10 重複 | top1 確率 |
|---|---:|---:|---:|---:|---|---:|---|
| 0 | 0.044 | 0.056 | 1.83% | 1.0e-03 | 25358 -> 25358 | 10/10 | 0.572 -> 0.588 |
| 1 | 0.052 | 0.066 | 0.56% | 1.9e-04 | 11 -> 11 | 10/10 | 0.461 -> 0.458 |
| 2 | 0.072 | 0.091 | 1.76% | 1.3e-03 | 864 -> 264 | 10/10 | 0.211 -> 0.211 |
| 3 | 0.093 | 0.117 | 2.71% | 3.6e-03 | 7949 -> 760 | 10/10 | 0.142 -> 0.132 |

読み方:

- **誤差は bf16 で新たに増えた分と、既存の HTP 経路の分に分けられる。**
  Phase 48 の HTP は CPU に対して既に最大 2.77% ずれていた。Phase 49 は 3.92%。
  Phase 48 と Phase 49 の差は最大 2.91% で、bf16 追加と IQ3 並べ替えを含む変更全体が
  この大きさ。bf16 単独演算の誤差は rel_l2 2.3e-07 なので、この差は積和の順序と
  オフロード範囲の違いによるもので、変換の誤りではない。
- 上位候補は常に 10/10 が同じ集合で、入れ替わるのは 1 位と 2 位だけ。
  step 2 と 3 で argmax が CPU と違うが、top1 確率は 0.211 -> 0.211、0.142 -> 0.132 で、
  僅差の入れ替わりである。
- **生成品質への影響は未確定。** replay は token を教師強制で与えるので、argmax が
  違っても列は最後まで走る。品質 (perplexity、タスク正解率) は測っていない。
  定量的な根拠なしに品質の合格基準を作らない。Phase 50 の課題として残す。

## 5. Phase 50 の測定環境 (固定)

- 標準構成: HTP は NGL=16 / `GGML_HEXAGON_SPLIT=0` / bf16 有効 / IQ3 gather 有効。
  CPU は NGL=0。NGL=24/32 は追加評価のみ。33 テンソルの行分割は残すが既定にしない。
- 参照値 (Phase 49、同じ手順): 熱制限下 cap 2112/1862 で CPU 単独 約 735、NGL=16 約 645 ms。
  fast cap 3628/4608 で CPU 単独 467-482、NGL=9 467-474、NGL=16 476 ms。条件が違うので混ぜない。
- **今夜分かった環境の落とし穴** (Phase 50 で必ず記録する):
  - 画面 OFF (Dozing) で CPU が 883200/768000 に固定される。充電していない端末では
    `svc power stayon true` は効かず、`screen_off_timeout` を 30 分にしても約 45 秒で Dozing に戻る。
    20 秒ごとの `input keyevent KEYCODE_WAKEUP` でしか fast cap を保てなかった。
  - 8 step の途中で 3628800 -> 2112000 に落ちた run がある。**前後両方の cap を記録し、
    変わった run は捨てる。**
  - 背景アプリの anon page が 1-2 GB あると、モデルの page cache を広げるために
    swap-out が 1073-1139 MiB 発生し、1024 MiB ガードに掛かる。試験前の `am force-stop` で通る。
  - 負荷 (load average 6-10) が高いと CPU split が 900 ms まで伸びる。Phase 49 の 735 ms と
    比べてはいけない。
  - **run を長くしてもばらつきは減らない。** 16 step (NGL=16、cap 1785600) を 1 回取ると
    714-2005 ms で、3 step 目に 2005 ms の外れ値が入った。中央値は 782-787 ms で 8 step の
    737-825 ms と同じ。ばらつきは step 数ではなく背景負荷で決まるので、長い run 1 回より
    8 step を 4 組の方が良い。外れ値を除く規則を先に決めておくこと。
- 手順は `hexagon-phase50.md` に固定した。実装は `p495_run.sh` (1 run、画面維持と
  背景停止を含む) と `p495_abba.sh` (交互比較、cap 変化を CAP_CHANGED として記録)。

### 5.5 今夜取れた同 tier の新しい比較 (cap 1785600 / 1497600)

端末が AC 給電に変わったあと、8 step x 4 run の ABBA を回した。1 run 目は開始時
2112000 で終了時 1785600 だったので CAP_CHANGED として捨てた。残り 3 run は
cap 1785600 / 1497600 で一致している。

| run | NGL | step1-7 の中央値 ms | ばらつき (最小-最大) | 備考 |
|---|---:|---:|---|---|
| s1 | 0 | 966.6 | 878.6 - 1267.2 | CAP_CHANGED、比較に使わない |
| s2 | 16 | 737.4 | 730.4 - 1083.2 | 同一 cap |
| s3 | 0 | 974.2 | 850.6 - 1222.2 | 同一 cap |
| s4 | 16 | 825.2 | 741.6 - 1019.5 | 同一 cap |

時間的に隣り合う組 (s3, s4) で **-15.3%**。step 0 は 1238-2641 ms で、初回を含めると差は
さらに大きく出るが、初回は除いて判定する。ばらつきは ±25% と大きい (背景負荷)。この tier は
Phase 49 の 2112/1862 でも 3628/4608 でもないので、参照値とは混ぜない。

同じ run の split 構成 (NGL=16、2 split):

| split | ノード | 定常 step の時間 |
|---|---:|---:|
| CPU | 2950 | 610-618 ms |
| HTP (投入復帰まで) | 898 | 0.72-0.75 ms |
| 差し引きの DSP 実行 | | 約 125-130 ms (step 合計 740 ms から CPU 分を引いた値) |

CPU split 610 ms + DSP 130 ms = 740 ms で、**この構成では重なりがほぼ無い**。
NGL=0 の 974 ms から 16 層を HTP へ移すと CPU が 364 ms 減り、DSP が 130 ms 増える。
1 層あたり CPU 22.8 ms 減 / DSP 8.1 ms 増で、**この tier では DSP が 1 バイトあたり 2.5-2.8 倍
効率が良い**。fast cap では CPU が約 2 倍速くなるので両者が並ぶ (Phase 49 の同等という結果と整合)。

## 6. CPU 側の内訳 (約 400 ms の正体)

`GGML_SCHED_DEBUG=2` のノードダンプ (NGL=0、2024 ノード + 1824 view = 3848) から、
1 token あたりの CPU 側の読み出し量を実測した。

| 種別 | ノード数 | 読み出し MiB | 割合 |
|---|---:|---:|---:|
| MUL_MAT (重み) | 497 | 8183.0 | 96.1% |
| それ以外 (GET_ROWS, GATED_DELT, CONCAT, CPY, MUL, RMS_NORM, SWIGLU, SILU, ADD, SSM_CONV, SCALE, ROPE, ...) | 1527 | 331.2 | 3.9% |
| 合計 | 2024 | 8514.2 | 100% |

MUL_MAT を重み種別に分けたもの (上位):

| クラス | 本数 | MiB | 割合 |
|---|---:|---:|---:|
| IQ3_XXS K5120 N17408 | 20 | 640.0 | 7.5% |
| IQ4_XS K5120 N248320 (出力ヘッド) | 1 | 644.0 | 7.6% |
| IQ2_S K5120 N17408 | 39 | 1053.0 | 12.4% |
| IQ2_XXS K5120 N17408 | 25 | 525.0 | 6.2% |
| IQ2_S K17408 N5120 | 19 | 513.0 | 6.0% |
| IQ3_XXS K6144 N5120 | 43 | 503.0 | 5.9% |
| IQ2_XS K5120 N17408 | 19 | 456.0 | 5.4% |
| IQ3_XXS K17408 N5120 | 14 | 448.0 | 5.3% |
| IQ3_S K17408 N5120 | 12 | 432.0 | 5.1% |
| IQ3_S K5120 N17408 | 9 | 324.0 | 3.8% |
| 型別小計 | IQ3_XXS 2093 / IQ2_S 1881 / IQ2_XXS 946 / IQ2_XS 906 / IQ3_S 840 / IQ4_XS 750 / Q4_K 188 / IQ3_M 87 / 型不明 492 = 8183 | |

型不明の 492 MiB は 1 要素あたり 0.17-0.24 バイトで、既知のブロック長と 6% 以上ずれたもの。
K=6144 の端数か、IQ1 系の可能性があるが確認していない。

CPU 側の実効帯域を `p495_cpu_matvec_test` で実測した (K=5120 N=17408 M=1、7 スレッド、
cap 3628800 だが `scaling_cur_freq` は 1900-2500 MHz):

| 型 | GB/s |
|---|---:|
| iq2_xxs | 12.13 |
| iq2_xs | 7.93 |
| iq2_s | 7.52 |
| iq3_xxs | 7.29 |
| iq3_s | 4.42 |
| iq4_xs | 5.70 |

同じ実行内の数値なので型の間の比較はできる。**実測した読み出し量と実測した帯域から
計算した時間**は合計約 1.13 s で、同じ日の実測 CPU split (NGL=0、約 900 ms) と同じ桁。
モデル側が約 20% 速いのは、単体試験が毎回重みを cold で読むのに対し、モデルでは
直前の層の重みが cache に残るためと解釈できる (未検証)。

結論: **CPU の 400 ms は「特定の遅い演算」ではなく、1 token あたり 8.31 GiB の重みを
読む時間そのもの。** 割合で見ると MUL_MAT が 96% で、内訳は IQ2_S 1.8 GiB、IQ3_XXS 2.0 GiB、
IQ2_XXS 0.9 GiB、IQ2_XS 0.9 GiB、IQ3_S 0.8 GiB、IQ4_XS 0.7 GiB。1 個の演算を直しても
数 % しか動かない。減らすには (a) CPU が読む重みを減らす (層を HTP へ移す)、(b) 同じ重みを
少ないバイト数で読む (より低ビットの量子化)、(c) 重みを複数 token で再利用する、のどれかが要る。

| 演算 | 回数/ token | 時間 | CPU 時間に占める割合 | HTP 対応状況 |
|---|---:|---|---|---|
| MUL_MAT 全体 | 497 | 計算値 約 1.09 s (実測 split 900 ms と同桁) | 96% (バイト基準) | IQ3_S / IQ3_XXS / IQ4_XS / Q4_K / bf16 あり。IQ2_S / IQ2_XS / IQ2_XXS / IQ3_M は低速または無し |
| GET_ROWS | 99 | 未測定 | 1.8% (バイト基準) | - |
| GATED_DELT | 48 | 未測定 | 1.7% | 無し |
| RMS_NORM | 305 | 未測定 | 0.06% | CPU |
| SWIGLU / SILU | 64 / 96 | 未測定 | 0.09% | CPU |
| CONCAT / CPY / CONT | 48 / 48 / 16 | 未測定 | 0.25% | CPU |
| ROPE / FLASH_ATTN | 32 / 16 | 未測定 | 0.03% | CPU |
| 同期・コピー | split あたり 1 | 実測 8-19 us (sched) | 0.002% | - |

「未測定」は測っていないという意味で、0 ではない。時間の列は、バイトが実測・帯域が実測・
積が計算値であることを明記しておく。

## 7. Phase 50 候補の優先順位

| 候補 | 現在の実測 | 理論上限 | 27B への効き | 難度 | 数値リスク | メモリリスク | 他への影響 | 実機試験 |
|---|---|---|---|---|---|---|---|---|
| A. CPU の重みを HTP へ (IQ2_S / IQ3_XXS など) | CPU split 610 ms / 2950 ノード、重み 8183 MiB (cap 1785)。fast cap では 394-401 ms | 1 層あたり CPU 22.8 ms 減 / DSP 8.1 ms 増。全 48 層を DSP が同じ効率で処理できれば CPU split は 331 MiB 分 (約 15 ms) まで落ちる | **最大。いまの 740 ms が約 400 ms になる余地** | 高 | 中 (積和順序の差。bf16 変換自体は厳密) | 低 (重みは mmap のまま DMA) | DSP 実行が 130 ms から 400 ms へ増える | カーネル単体 + 27B 8 step |
| B. IQ2_S / IQ2_XS の gather 最適化 | DSP 9.85 / 7.81 GB/s、合計 約 6 ms/token | IQ3 並みの 21 GB/s で 6 -> 2.5 ms | **小。単独では 0.5%** | 中 | 低 (並べ替えのみ) | 低 (スクラッチ 1024 B/thread) | A の前提条件になる | カーネル単体 + logits バイト一致 |
| C. CPU と DSP の重複 | NGL=16 で CPU 610 + DSP 130 = 740 ms。重なりは 10 ms 未満 (cap 1785) | 構造上ほぼ 0。CPU split がグラフの先頭にあり、HTP split がその出力に依存する | **小。いまの分割順では伸びない** | 高 | 低 | 中 (2 面分の copy が要る) | グラフ配置と同期の全面見直し | 27B 8 step、ABBA |
| D. 32x2 の IQ3 実装 | 32x1 を 2 回呼ぶ (prefill のみ) | prefill の短縮のみ | **decode には 0** | 中 | 低 | 低 | prefill の数値検証が要る | prefill の試験 |
| E. キャッシュ dirty 区間と flush | 未測定 | 未測定 | 不明 (小さい見込み) | 中 | 高 (整合性) | 低 | DMA 経路全体 | 専用の整合性試験が必須 |

優先順位: **A > B > E > C > D**。

- A は B を前提にする。IQ2_S を HTP へ移すと、いまの 9.85 GB/s では 1881 MiB に約 200 ms
  かかり、CPU に置くより遅い。30 GB/s 以上にしてから移す。
- C は「重なりを増やす」候補だが、いまの分割順 (CPU が先、HTP が後) では重なりは
  原理的に作れない。NGL=32 で Phase 49 が見た 126 ms の重なりは、HTP split が 4 つに
  分かれていた構成のもの。2 split の標準構成では期待できない。順序を変える
  (HTP を先に置く) と CPU 側が DSP の出力を待つので、同じ理由で伸びない。
- D は decode の指標に出ない。E は費用が未測定で、整合性の別ゲートが要る。

## 8. Phase 50 の合格条件 (提案)

性能:

- fast cap で、同一条件の NGL=0 より NGL=16 が再現性をもって速いこと。
- 目標は 5% 以上。ただし **8 step 4 組以上の ABBA、同じ cap の組だけ、中央値で判定**とし、
  ばらつき (最小-最大) が 5% を超えるなら組を増やす。1-2% の差を PASS にしない。
- 熱制限下の優位 (Phase 49: 約 12%) を壊さないこと。

正当性:

- bf16 単独試験 PASS (65536 パターンのビット一致、M=1/2/3、K=5120/5152/96、非有限 0、上界内)。
- IQ3 単独試験 PASS (4 step logits が `5585d4ed...` とバイト一致)。
- 2 / 4 / 16 step が exit 0、非有限 logits なし、CPU split の logits が `577417e9...` と一致。
- 既存の 33 テンソル分割のビット一致を壊さない。数値基準を緩めない。

安全:

- メモリガード違反なし、DSP 異常なし、CPU 単独経路の回帰なし。
- 1 試験 180 秒、swap-out 増分 1024 MiB、OOM 兆候・強い熱制限で停止。ガードは変更しない。

比較の残し方:

- 変更前後で同じモデル・NGL・cap・入力 (同じ replay token 列) を使い、
  変更前後と CPU 単独の 3 つを必ず並べる。

## 9. 今夜の実測ログ (再現用)

| 内容 | 値 |
|---|---|
| NGL=0 4 step replay (CPU logits) | exit 0、8.3 s、sha256 `577417e9...` = Phase 49 と一致 |
| NGL=9 4 step replay (IQ3 回帰) | exit 0、sha256 `5585d4ed...` = Phase 49 と一致 |
| NGL=16 4 step replay (21:26、画面のみ) | **guard exit 6** (swap-out 1024 MiB 超) |
| NGL=16 4 step replay (後の ABBA 内) | exit 0 (s2/s4)。給電が変わり背景アプリ停止で通った |
| NGL=0 8 step (22:19) | exit 0 だが run 途中で cap 3628800 -> 2112000。比較に使わない |
| NGL=0 2 step (グラフダンプ) | exit 0、ノード 2024 + view 1824 |
| ABBA 8 step x 4 (cap 1785600) | s1 CAP_CHANGED、s2/s3/s4 は同一 cap。上記 5.5 |
| NGL=16 16 step (cap 1785600) | exit 0、中央値 782-787 ms、外れ値 2005 ms が 1 回 |
| BF16 単体 | fail 0 |
| sched 計測 (mix / async) | 上記 3.3 |
| CPU matvec 帯域 | 上記 6 |

## 10. 使ったコマンド

ホスト (worktree `D:/ghq/github.com/yuubinnkyoku/llama.cpp-c49`):

```
git log --oneline be78b2451..HEAD
git diff --check be78b2451..HEAD
git push origin codex/c49-perf
git ls-remote --heads origin codex/c49-perf
git fetch origin codex/overnight-autonomy-phase49
git diff --stat 6ef8a9c60..f72f2223a
```

ビルド (既存の build-c49 をそのまま使用、再ビルドなし):

```
clang++ --target=aarch64-linux-android28 -O2 -std=c++17 -I ggml/include -I ggml/src \
  -o p495_sched_timing_test p495_sched_timing_test.cpp \
  -L build-c49/bin -lggml -lggml-base -llog -ldl -lm -static-libstdc++
clang++ ... -o p495_cpu_matvec_test p495_cpu_matvec_test.cpp \
  -L build-c49/bin -lggml -lggml-cpu -lggml-base -llog -ldl -lm -static-libstdc++
```

端末 (serial は `adb devices` のもの、今夜は `100.108.106.81:37641`):

```
adb shell "cd /data/local/tmp/c46-27b; LD_LIBRARY_PATH=libs ADSP_LIBRARY_PATH=/data/local/tmp/c46-27b/libs ./p49_bf16_test"
adb shell "cd /data/local/tmp/c46-27b; LD_LIBRARY_PATH=libs GGML_SCHED_SPLIT_TIMING=10 ./p495_sched_timing_test mix 300 10"
adb shell "cd /data/local/tmp/c46-27b; LD_LIBRARY_PATH=libs GGML_SCHED_SPLIT_TIMING=1 ./p495_sched_timing_test async 15 20000"
adb shell "sh /data/local/tmp/c46-27b/p495_run.sh 4 replay 0 c1cpu"
adb shell "sh /data/local/tmp/c46-27b/p495_run.sh 4 replay 9 c3ngl9"
adb shell "sh /data/local/tmp/c46-27b/p495_run.sh 2 noreplay 0 d2cpu 2"
adb shell "sh /data/local/tmp/c46-27b/p495_abba.sh 8 0:s1 16:s2 0:s3 16:s4"
adb shell "export PROFILE_ARG=1; sh /data/local/tmp/c46-27b/p495_run.sh 5 noreplay 9 prof9"
```

解析:

```
python p495_logits_cmp.py lg-ngl0.f32 lg-ngl9.f32 4 248320
python p495_graph_ops.py p495-d2cpu-stderr.txt
python p495_mulmat_classes.py p495-d2cpu-stderr.txt
python p49_ops.py p495-prof9.txt 1
```

## 11. Git 状態

- ブランチ `codex/c49-perf`。Phase 49 の HEAD は `533b03638`、Phase 49.5 の 3 コミットを
  積んだあとの HEAD は `1ac7112e2`。作業ツリー clean。
- `1ac7112e2` は origin/codex/c49-perf と一致。Phase 49 の 5 コミットも保存済み。
- メイン worktree (`llama.cpp`、`codex/phase20-htp-correctness-memory`) の未コミット 9 ファイルは
  触っていない。Phase 48 の `hexagon.md` も変更していない。
- このフェーズのコミット: 3 件 (`b32e90cc6` Skill 統合と記録、`724617bf1` 同 tier の実測と
  CPU 内訳、`1ac7112e2` IQ3 帯域の再現)。すべて push 済み。
  強制 push、main への merge はしていない。

## 12. 未解決と次の一手

1. **fast cap の新しい比較値が無い。** 今夜の端末は画面維持ができず、8 step の途中で
   熱制限に落ちた。給電が入ってからは cap 1785600 / 1497600 で安定したので、
   その tier の組は取れた (5.5)。fast cap (3628800 / 4608000) は冷却後に
   `p495_abba.sh 8 0:a 16:b 0:c 16:d` で取り直す。
2. **NGL=16 の 4 step replay が swap-out ガードに掛かった。** 背景アプリを止めても
   1073-1139 MiB 出る。メモリ余裕を増やす (充電して画面 ON、アプリ停止) か、
   NGL を下げた系列で代替する。ガードは緩めない。
3. **品質が未評価。** logits の差 (rel_l2 最大 3.9%、top10 は常に一致) が生成品質に
   効くかは perplexity かタスク評価でしか分からない。Phase 50 で先に決める。
4. **CPU 内訳の時間列が計算値。** 演算ごとの実時間は on-device のプロファイラが
   無い (`simpleperf` は perf_event を開けず、`cpu-cycles`/`cpu-clock` とも
   "not supported on the device") ため、バイト実測 x 帯域実測で代用した。
   実時間が要るなら計測用のビルドを別に作る。
5. **`attn_output` (34 本目) は未試験のまま。** 既定は 33 テンソル。
