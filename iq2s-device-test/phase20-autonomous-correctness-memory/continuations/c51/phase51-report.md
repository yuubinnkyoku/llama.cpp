# Phase 51 報告書

ブランチ `codex/c51-perf`、基準は Phase 50 の `8e5b850a3`。
本文は 2026-11-13 02:03 時点の実測に基づく。未測定の項目は「未測定」と明記する。

## 1. 総合判定

| 項目 | 判定 | 根拠 |
|---|---|---|
| A. 34 テンソル目の不具合の解明 | **PARTIAL** | 現行ツリーには 34 本目固有の不具合が**存在しない**ことを行ごとの実測で示した。ただし過去に起きたとされる異常終了そのものは再現していない (実物の記録が DMA64=1 の別事象だけ) |
| B. 並列実行の高速化 | **PARTIAL** | 行分割の費用を 1 token あたり 163-171 ms 削り、非同期経路の数値不整合を直した。27B 全体で SPLIT=0 を上回る再現性のある優位は**未確認**。上限は実測から 1% 前後と見積もられる |
| 総合 | **PARTIAL** | A と B を独立に評価。安全性と数値正当性は維持 |

B が PASS しない理由は「測れなかったから」ではなく、
**測れる範囲で上限が 5% に届かないと分かったから**である (7 節と 8 節)。
測定そのものが環境に阻まれた部分 (8.2) も別途明記する。

27B 全体の比較は 02:02 に 4 組を試みて 2 run で中断し (8.1.1)、その直後に端末が
ネットワークから落ちて adb も ping も通らなくなった。再起動は禁止されているため、
残る実測は端末が戻ってから `sh p51_finish.sh 9 4` で取る。

## 2. 34 テンソル目の異常終了の調査結果

### 2.1 実在する記録の棚卸し

端末 `/data/local/tmp/c46-27b/` に Phase 46-47 の一次資料が残っていた。
`c47` / `c48` / `c49` の主報告書は git のどのブランチにも無いため、生ログを読んだ。

| 資料 | 内容 | 読み |
|---|---|---|
| `A.log` `A2-4.log` `B.log` `C.log` `D.log` | 単一テンソル分割、DMA64=0 | すべて rc=0、`sentinel=0`、`htp_vs_htpfull=0.000000` |
| `E1.log` `E2.log` | 同じ試験、**DMA64=1** | `dspqueue_read failed: 0x0000002e` で abort |
| `p47-f33/f34/f35/f5/f6-*.log` | 実モデル ABBA (ngl 9、5 step、frac 0.9) | rc=0。ただし後述 |
| `runs/` 747 ディレクトリ | 2026-11-06 から 11-10 までの全 run | `dspqueue` 0 件、`ggml_abort` 0 件 |
| `split33.txt` / `split34.txt` | 分割リスト | 差は `blk.63.attn_output.weight` の 1 本だけ |

決定的だったのは 2 点。

1. **唯一の abort は DMA64=1 のときだけ出る。** E1/E2 は A2-4 と同じ
   1 テンソル試験で、環境変数 `GGML_HEXAGON_DMA64` だけが違う。
   DMA64=0 では 5 回とも rc=0。**テンソル本数の関数ではない。**
   エラーは `dspqueue_read` が返す 0x2e (AEE_EBADPARM 相当) で、
   分割の CPU シェアを実行する前、DSP 応答待ちで起きる。
2. **「34 本で試験した」とされた run は、実際には 33 本しか分割していなかった。**
   `p47-f34-a-b1.log` の非同期 run (`runs/20261108-074614-1264`) の `c46-env.txt` を見ると
   `SPLIT_WEIGHTS` に `blk.63.attn_output.weight` が入っておらず、stderr の
   `split weight` 行も 33 本。ラベル `f34` はテンソル本数ではない。

つまり「34 本目の異常終了」を裏付ける一次記録は存在しない。
`P47_FAIL rc 2` は承認ファイル切れ (`REFUSING: no approval file`、経過 0.1-0.2 秒、
run ディレクトリが前回のまま) であり、異常終了ではない。

### 2.2 34 本目を実際に分割した (今回)

NGL=16、skel D (`288e6712...`)、4 step replay、`split34.txt`。

```
ggml-hex: split weight blk.63.attn_output.weight (iq3_s): k 6144 m 5120 host rows 2560
  dsp rows 2560 row 2640 B, host 6.45 MiB, dsp 6.45 MiB in 7.50 MiB at off 1032157184
  usage 1, alias 1, r mod 32 0, readback ndiff 0 first -1, in buf 1
```

- 34 本すべて捕捉 (`SPLIT_COUNT 34`)、readback は ndiff 0、alias はモデルバッファ内。
- frac 0.5 (host share 合計 400.93 MiB) の非同期 run は 2 回とも
  **swap-out ガード (1.24-1.25 GiB > 1024 MiB)** で停止した。decode は step 1 まで到達し、
  DSP エラーも `dspqueue_read` も出ていない。
- `GGML_HEXAGON_SPLIT_CPU_ROWS=32` (host share 数 MiB) にすると 4 step 完走、exit 0。

### 2.3 行ごとの数値検証 (Gate 2)

`P51_DUMP=d34.bin` の診断 run (34 本、CPU_ROWS=32、NGL=16、4 step、exit 0) で、
出力を `-12345` で汚してから両シェアに書かせ、full DSP 参照と比較した。

| 検査 | 結果 |
|---|---|
| CPU 担当行 (32 行) が独立な CPU 再計算と一致 | **136/136** (`cpu_vs_cpuref same nonbit 0`) |
| DSP 担当行が full DSP 参照と一致 | CPU シェアの外で**差分行 0** (136 record) |
| 未書き込み行 (センチネル) | **0** |
| 非有限値 | **0** |
| 異常終了 / DSP 異常 | なし |

`blk.63.attn_output.weight` の alias された tiled 領域からの行読みに、
行番号のずれも端数も無い。**34 本目固有の境界処理の問題は無い。**

### 2.4 仮説の切り分け

| # | 仮説 | 判定 | 根拠 |
|---|---|---|---|
| 1 | バッファ寿命・所有権 | 否定 | 34 本で 4 step 完走、センチネル 0、再実行で安定 |
| 2 | 非同期キューの順序・同期 | **肯定 (本数と無関係)** | async が入力を待たずに読み、run ごとに値が変わる。3 節で詳述 |
| 3 | キャッシュ整合性 | 肯定 (2 と同じ機構) | 待たずに host がデバイスメモリを読む。2 の修正で解消 |
| 4 | 行分割の境界処理 | 否定 | 2.3 の行ごと検証。先頭 32 行のみ CPU、他は DSP とビット一致 |
| 5 | テンソル数に依存する管理構造 | 否定 | 33 本でも同じ不整合が出る (3 節)。34 本目だけの差は host share 6.45 MiB |
| 6 | メモリ圧迫による間接的な失敗 | **肯定 (別要因)** | frac 0.5 の 400.93 MiB host share がガードに触れる。本数ではなく比率の問題 |
| 7 | 特定の型・形状・経路 | 否定 | 34 本目は IQ3_S 6144x5120 で、33 本側の `blk.59.attn_output.weight` と同型 |

## 3. 見つけて直した不具合: 非同期分割の入力待ち漏れ

### 3.1 症状

同じバイナリ、同じ入力で、`SPLIT=1 async` の logits が run ごとに変わる。
NGL=9、33 本、frac 0.25、4 step replay で:

| lib | mode | logits sha256 (先頭 16) |
|---|---|---|
| Phase 50 配備 | async | `788611da9d15a5fb` |
| c51 修正なし | async | `9456dc7458643944` |
| 同上を再実行 | async | `019160106423dc12` (**一致しない**) |
| c51 修正なし | seq | `2d4693c43c0cc307` |
| 同上を再実行 | seq | `2d4693c43c0cc307` (一致) |

step 0 の logits はどの run でも完全一致し、step 1 以降が全要素で変わる
(max |d| 3.5-4.8、argmax は 4 step 中 3 つ一致、top10 は 6-10 個一致)。

### 3.2 原因

`hex_split_run` の async 分岐が「DSP シェアを submit → **待たずに** CPU シェア →
flush_sync」の順だった。CPU シェアは `plan->x_cpu->data = x->data` として
**デバイスメモリ上の入力を直接読む**。その入力を書いた DSP op は同じバッチで
まだ実行中かもしれず、待たずに読めば古い値とキャッシュの不整合が混ざる。
`GGML_HEXAGON_SPLIT_NOSYNC=1` も同様 (CPU シェアを先に走らせる)。

Phase 46 の合成ハーネス (`c46_split_repeat`) が `split-async.bin` と
`split-seq.bin` の一致を確認できていたのは、入力がロード時の固定データで
DSP の producer が無かったためである。

### 3.3 修正 (commit `bb2607d4f`)

`hex_split_run` の先頭で、この op より前に投入された op を join してから submit し、
CPU シェアを DSP シェアと重ねる。

```cpp
    // The host share reads its input from device memory, and the op that writes
    // that input may still be in flight. Join the ops submitted before this one
    // first: the DSP runs its queue in order, so they own every input of this op.
    tw0 = hex_now_us();
    sess->flush_sync();
    tw1 = hex_now_us();
```

DSP はキュー順に実行するので、先行 op を待てば入力は確定する。この op 自身の
DSP シェアは待たないので重なりは残る。`seq` と診断時だけ CPU シェアの後にも待つ。

パラメータの安全性も確認した。`htp_op_batch::add_op` は `kernel_params` を
`memcpy` でバッチ側に複製する (`ggml-hexagon.cpp:3735`)。したがって join の待ちで
別 op のパラメータを書き換えても、飛行中の op は影響を受けない。

## 4. 最小再現試験 (モデル不要)

`c51_split_repro.cpp`。詳細は `phase51-small-repro.md`。

- 34 本の**実名・実型・実形状**を HTP の weights バッファに作り、分割の捕捉経路を通す。
- 各 MUL_MAT の入力の前に **SCALE を 1 本置く**。これで入力が「同じバッチの DSP op が
  書くテンソル」になり、実モデルと同じ依存関係になる。
- `--repeats T` で T 回計算し、毎回の出力ハッシュを出す。

結果 (`--small`、34 本、3 repeat):

| lib | mode | 3 回のハッシュ |
|---|---|---|
| 修正前 `5c55b585` | async | `64b0e335` , `a638f0e6` , `a638f0e6` |
| 修正前 | seq | `a638f0e6` x3 |
| 修正後 `c646b4db` | async | `a638f0e6` x3 |
| 修正後 | seq | `a638f0e6` x3 |
| 修正後 | off (分割なし) | `d4e8f225` x2 |

**修正前は 1 回目だけ誤り、修正後は 1 回目から seq とビット一致。**
モデルも 9 GB の読み込みも不要で、1 回 2 秒。

### 4.1 診断を入れると不具合が消える

同じ修正前 lib で `--dump` を付けると、ハッシュが 2 回とも `a638f0e6` になり
再現しなくなる。診断経路が分割の前に元ノードを DSP で 1 回実行して `flush_sync`
するため、先行 op が回収されてしまうからである。
**診断ありの結果だけで「正常」と判断してはいけない** (指示書 5.1 の注意そのもの)。

## 5. 修正内容の一覧

| commit | 内容 | 検証 |
|---|---|---|
| `bb2607d4f` | 分割の host シェアの前に先行 DSP op を join | 小型試験で再現 -> 解消。実モデルで async == seq |
| `302685f2d` | 分割時も HTP の node cache を使う | prep_us 163414 -> 48120。logits は変更前と一致 |
| `36b407da5` | host シェアのスレッド既定を 6 から 3 へ | 実形状で 164.6 -> 143.7 ms。dump がバイト一致 (値は不変) |

`302685f2d` の根拠: 分割を有効にすると `hex_split_enabled()` が node cache を
無効化しており、1 token ごとに 229 ops の `htp_opnode` 構築とパラメータ再計算を
やり直していた。分割 op 自身は毎回パラメータを作り直し、テンソルの現在値も
追い直しているので、キャッシュを使って問題ない。

## 6. 行分割の数値正当性

| 対象 | 結果 |
|---|---|
| 34 本、NGL=16、4 step、行ごと (2.3) | PASS (136 record、CPU 行ビット一致、DSP 行差分 0、センチネル 0) |
| 33 本、NGL=9、frac 0.25、async と seq | **ビット一致** `2d4693c43c0cc307` |
| 同、2 回実行の再現性 | 一致 |
| 分割なしの回帰 (NGL=16) | `f3a0d235c0da32d0c40ed901b9236026...` Phase 50 の基準と一致 |
| 分割なしの回帰 (NGL=9) | `5585d4edc47ee7505c558aeb1b929a7e` Phase 50 の基準と一致 |
| 小型試験、34 本、2 repeat、行ごと | PASS (68 record、CPU 128 行のみ差分、他は一致) |
| 小型試験、34 本、**frac 0.35**、行ごと | PASS (34 record、CPU 行 352/1792/2144/3584/6080 のみ差分、他は一致、センチネル 0) |
| 小型試験、**分割なし**、Phase 50 配備 lib と c51 lib | **同一ハッシュ** `3f4f81c9fc776b83` (real 34 shapes、805.82 MiB) |
| 小型試験、`SPLIT_NOSYNC=1` (修正後) | async / nosync / seq の 3 経路すべて `a638f0e61f5f1269` |
| 最終ビルド (3 スレッド既定) の行ごと | PASS (34 record、CPU 128 行のみ差分、外側 0、センチネル 0) |

新しい比率 (frac 0.25、0.35) の境界で、上記のとおり検証し直している。
Phase 48-49 の比率の結果は流用していない。
分割なしの経路は、Phase 50 配備のホストライブラリと c51 のライブラリで
**ビット一致**した。2 つの修正が分割を使わない経路に触れていないことの直接確認である。

## 7. CPU/HTP の実際の重複時間

### 7.1 1 演算の内訳 (GGML_HEXAGON_SPLIT_TRACE、33 本、NGL=9)

中央値の和 (1 decode step あたり、us)。投入は 1 op あたり 40-56 us で、
33 本合わせて約 1.5 ms/step なので表からは外した。

| 構成 | cpu (host シェア) | wait | op | wait の意味 |
|---|---|---|---|---|
| async frac 0.5 (修正前) | 55367 | 1614 | 62498 | CPU シェアの後の DSP 待ち |
| async frac 0.25 (修正前) | 33723 | 84 | 49302 | 同上 |
| async frac 0.125 (修正前) | 22894 | 5082 | 49834 | 同上 |
| async frac 0.25 (修正後) | 32879 | 17828 | 34345 | **投入前の join** |

- 33 本を分割せずに DSP だけで実行した時間は **33684 us/step** (profile 実測、
  NGL=9、skel A。IQ3_S の帯域は A と D で同一)。
- CPU シェアの実効帯域: host コピー 394.48 MiB を 55.4 ms で読むので **7.5 GB/s**
  (33 本、frac 0.5、6 スレッド、cpu_us の最小値の和から算出)。1 op あたり平均 11.95 MiB。
  DSP の IQ3_S 28 GB/s に対して 3.7 倍遅く、これが「CPU に 22% より多く渡すと損」の根拠である。
- 修正後の op 合計 34345 us は、分割しない場合とほぼ同じ。
  **重なりは join の待ち (17828 us) に食われている。**
- `wait_us == 0` の op は修正前 frac 0.5 で 132 中 2 本。ほぼ全 op で
  CPU シェアが DSP シェアを隠しており、1 演算の中の重なり自体は成立している。

### 7.2 HTP backend の内訳 (GGML_HEXAGON_TIMELINE、229 ops/step)

| 構成 | batches | submit_us | wait_us | prep_us |
|---|---|---|---|---|
| 分割なし (cache 有効) | 1 | 251 | 60842 | **306** |
| 分割あり (cache 無効) | 34 | 5917 | 17074 | **170801** |
| 分割あり (cache 修正後) | 34 | 1670 | 24576 | **39026** |
| 分割あり (cache + async 修正) | 62 | 3287 | 35624 | **46075** |

`prep_us` は「投入でも待ちでもない host 時間」で、node 再構築と CPU シェアを含む。
Phase 49 が記録した「分割が 11-12% 遅い」の主因はこの 163-171 ms であり、
分割そのものの計算ではなかった。

2 つの計器は一致する。timeline の分割なしの wait 60.8 ms と、profile (skel D、NGL=9) の
decode MUL_MAT 合計 60,178 us/step はほぼ同じ値で、HTP 区間がほぼ全部 MUL_MAT であること
を示す。33 本分は 33,752 us/step で、上の D = 33.7 ms と 0.3% 以内で一致する。

### 7.3 モデル無しで測った重なりの効果 (ばらつきが大きい)

`c51_split_repro` の実形状 (805.82 MiB、行はコピー) で、34 本だけを繰り返した
ときの 1 回の計算時間 (ms)。`off` は DSP だけ、`async` は行分割。

| 構成 | frac | 観測値 (run ごと) |
|---|---|---|
| off (分割なし) | - | 180.0, 180.0, 180.0, 180.0, 180.3 (極めて安定) |
| async (修正後) | 0.25 | 154.9, 156.6, 242.0, 249.1 |
| async (修正後) | 0.35 | 126.7, 128.6, 207.3 |
| async (修正後) | 0.15 / 0.20 / 0.22 | 177.0 / 166.0 / 165.2 |
| seq (直列) | 0.25 | 297.3, 309.3 |

**分割側の値は run 間で 2 倍近く動く。** 一方 `off` は 0.1% の精度で再現する。
この差は、host シェアが**追加の CPU スレッド** (測定当時は 6、現在の既定は 3) で走るためで、端末で他の
アプリ (このときは動画再生と systemui) が CPU を使っていると、その分だけ
host シェアが遅くなる。`off` は DSP 待ちだけなのでホストの負荷に影響されない。

読み取れること:

- 空いているときは重なりが効く (154.9 対 180.0 で 0.86 倍、126.7 で 0.70 倍)。
- 端末が使われているときは、分割側が DSP だけより遅くなる (242-249 対 180)。
- `seq` は常に遅い (297-309)。直列であることは確か。
- **「13.5% 速い」のような単一の数字は出せない。** 7.4 の上限計算と、
  ホストが空いていることを前提にしたときの値として扱う。

この性質は実機の利用状況に直結する。画面が点いて他のアプリが動いている端末で
行分割を既定にすると、平均では DSP だけより遅くなりうる。

### 7.4 上限

- 分割が触れるのは、分割リストに入れたテンソルの DSP 実行時間だけ。33 本で NGL=9 は
  33.7 ms、34 本で NGL=16 は 34.2 ms (Phase 50 の skel D プロファイル、decode 4 step の
  MUL_MAT 合計 413086 us のうち 136739 us = 33.1%)。**リストは固定なので、NGL を上げても
  この窓は増えない。** NGL=16 の DSP 合計 103.3 ms の残りは分割していない層である。
- 33 本の CPU 合計 C = 118 ms、DSP 合計 D = 33.7 ms。最適点の op 合計は D*C/(D+C) = 26.2 ms。
- したがって行分割の上限は 33.7 - 26.2 = **7.5 ms/token (約 1.1%)**。
  ホストが空いていてスレッド既定を 3 にした実測は 143.7/180.0 = 0.80 倍なので、
  33.7 ms の窓が 27 ms 程度になる計算で、上限は 6-7 ms (約 1.0%) と読むのが実測に近い。
  残り 27 ms の DSP 区間は層の依存関係で分割できない。

### 7.5 host シェアのスレッド数 (改善)

GGML_HEXAGON_SPLIT_THREADS の既定は 6 だった。1 つの host シェアは数 MiB なので、
6 スレッドを起こす費用が仕事を上回る。実形状 (805.82 MiB、frac 0.25) の
定常 repeat の実測:

| スレッド数 | compute_ms |
|---|---|
| 1 | 170.0 |
| 2 | 148.2 |
| **3** | **142.3 / 143.2 / 151.0 / 143.7 (既定を 3 にした後)** |
| 4 | 154.8 |
| 6 (旧既定) | 164.6 / 171.3 / 171.3 / 225.6 |

3 が最良で、旧既定の 6 より 13-20% 速い。同時に、分割なしの 180.0 ms に対して
**0.80 倍**になり、これまでの既定 (6) では 0.91 倍だったものが改善した。

行は 1 スレッドあたり 1 行ずつ計算されるため、値はスレッド数に依存しない。
小型ハーネスの dump が 3 と 6 で**バイト一致** (9a5ffdc4a7298231...) することと、
出力ハッシュが一致する (`a638f0e6...`) ことで確認した。

36b407da5 でこの既定を 3 にした。SPLIT=0 の経路には影響しない。

### 7.6 入力を「producer だけ待つ」形にしても速くならない理由

join の待ち 17.8 ms を減らせば速くなるように見えるが、DSP はキュー順に実行するので、
入力 x を書いた op を待つことは、それより前に投入された op を全部待つことと同じである。
したがって join を部分的にしても待ち時間は減らない。実装もしない。

## 8. 27B 全体の性能比較

### 8.1 測れたこと

同一条件 (NGL=9、4 step replay、cap 2112000/1862400、skel D) の run ごとの
最小 step (ms):

| 構成 | lib | 最小 step | swap-out |
|---|---|---|---|
| SPLIT=0 | Phase 50 配備 | 697.5 | 43 MiB |
| SPLIT=0 | c51 (両修正) | 660.4 / 674.9 | - |
| async frac 0.25 | c51 (両修正) | 670.2 | - |
| seq frac 0.25 | c51 (両修正) | 780.5 (stall 込み) | - |

**これらの run では SPLIT=0 と async の差は 1.5% 前後で、向きも定まらない。**
指示書の合格条件 (4 組以上の有効な pair、cap が動いた run の除外) を満たしていない。

### 8.1.1 4 組の試行 (11-13 02:02、最優先で実施したが成立せず)

利用者が就寝する旨を伝えたため、01:56 の待機スクリプトを止めて 02:02 に 4 組を開始した。
結果は 2 run で中断した。

| run | mode | exit | 最小 step | cap の内訳 |
|---|---|---|---|---|
| s1 | off | 0 | 690.0 ms | 2112000_1862400 x14, 2496000_2668800 x3 |
| s2 | async | 0 | 744.6 ms | 2112000_1862400 x18, 2496000_2668800 x1 |
| s3 | async | **6** | なし | 2112000_1862400 x17 (swap-out 1.36 GiB) |

- s1 は cap が 17 サンプル中 3 つ別 tier に動いており、**同じ tier の比較として使えない**
  (p51_ser_pairs.py の 90% 規則で除外される)。
- s3 は swap-out 1.36 GiB でガードが停止した。この時間帯は run ごとに swap が増えており、
  1 組目 (off 292409 pages = 1.14 GiB、async 157135 pages = 614 MiB) ですでに境界だった。
- 02:03 に端末がネットワークから落ち、以降 adb も ping も通らない (再起動はしていない)。

したがって 02:02 の試行から**有効な pair は 0 組**。判定材料には加えない。

### 8.2 測れなかったこと

有効な 4 組を取れていない。理由は 2 つ。

1. cap が 1 run の間に動く (2112000 -> 3187200 -> 1555200 -> 3033600 を観測)。
   Phase 50 の 2 つの tier より細かく動くため、pair の多くが除外になる。
2. 端末のメモリが run の前提 (MemAvailable 9.5 GB) を満たさない時間帯がある。
   11-12 23 時台は 8.5-9.4 GB で、ガードが run を拒否した。11-13 00:15 には 6.26 GB、
   01:51 には 6.90 GB まで下がった (いずれも画面点灯・アプリ利用中)。
3. 画面が自動で消えない設定になっている。`stay_on_while_plugged_in` が 15 で、
   充電中は消灯しない。「画面が消えるまで待つ」方法では計測の窓が来ない (01:51 に確認)。
   画面を強制的に消す操作は利用者の邪魔になるのでしていない。
4. バッテリーは 11-12 深夜に 15% まで下がったが、11-13 01:42 に AC 接続 31% へ回復した。
   充電中の長時間 run は電力面では問題なくなった。
5. 11-13 01:53 に p51_finish.sh の動作確認を兼ねて 1 条件だけ実行してしまい、
   ガードが swap-out 1.91 GiB で停止した (exit 6、decode は stall 込みで `min_ms 1246`)。
   開始時の MemAvailable は 6.90 GB で、force-stop 後も足りなかった。
   記録として残すが、**比較には使わない**。以後、画面点灯中は実行しない。
   なお、この run の後 MemAvailable は 9.80 GB まで戻っており、force-stop が効けば
   次回はガードを通る可能性がある。

### 8.3 Phase 47 の記録との関係

Phase 47 の生ログには、fast cap で分割が 10.4% 速いという ABBA が 17 組残っている
(中央値 async/off = 0.896)。ただしこれは**入力待ちの無い (壊れた) 非同期経路**で
測られた値である。値を変えずに 10% を説明できる機構は 7.4 の上限計算に無い。
現行ツリーでは再現しないため、**性能の根拠としては使わない。**

### 8.4 fast cap ならどうなるか (算術。実測ではない)

cap が 3628800 に戻れば CPU シェアは速くなる。2112000 に対する比を 1.72 とすると、
C = 118 ms は 68.6 ms になり、最適点の op 合計は D*C/(D+C) = 22.6 ms になる。
窓の削減は 33.7 - 22.6 = 11.1 ms で、fast cap の token (Phase 50 の実測で 635 ms) に
対して **1.7%**。capped の 1.0% から倍増するが、5% には届かない。
これは C と D の実測からの算術であって、fast cap での実測ではない。

### 8.5 結論

行分割で 5% は取れない。上限は 1% 前後で、NGL を上げても変わらない。
分割リストに入れた 34 本の DSP 時間は NGL=9 で 33.7 ms、NGL=16 で 34.2 ms とほぼ同じで、
これは token の 5.4% にすぎない (7.4)。リストを広げれば窓は増えるが、CPU 側の層を
HTP へ移すと split 境界が増え、Phase 50 は層 36 の 3 本で split 2 -> 4、時間は改善せずと
記録している。5% を出すには DSP 区間全体を CPU 区間と重ねる必要があり、
層の依存関係 (層 49 は層 48 の出力を要る) からそれは作れない。

## 9. メモリ使用量

| 項目 | 実測 |
|---|---|
| 34 本 frac 0.5 の host share 合計 | **400.93 MiB** (`added_total_B 420392960`) |
| 34 本目の単体 | host 6.45 MiB / dsp 6.45 MiB (alias。新規確保なし) |
| CPU_ROWS=32 のときの 34 本合計 | 数 MiB |
| SPLIT=0、NGL=9 の swap-out | 43 MiB |
| SPLIT=1 frac 0.5、NGL=16 | 1.24-1.25 GiB -> **ガード停止** |
| SPLIT=1 CPU_ROWS=32、NGL=16 | 587 MiB (SPLIT=0 と同等) |

分割の host share は匿名メモリで、token ごとに読むためスワップアウトすると
戻ってくる。ガードを作動させたのは本数ではなくこの量である。

## 10. fast cap の状態

| 時刻 | cap0 / cap7 | 状態 |
|---|---|---|
| 11-10 20:00-21:00 | 2112000 / 1862400 | 固定 (Phase 50 と同じ) |
| 11-10 20:45-21:00 | 1555200 / 1382400 | さらに低下 |
| 11-12 23:15 | **3628800 / 3648000** | fast cap が一時的に復帰 |
| 11-12 23:20 | 2112000 / 1862400 | 数分で戻る |
| 11-12 23:32 | 3033600 / 2112000 | 中間 tier |
| 11-12 23:59 | **3628800 / 3648000** | fast cap が再び復帰。MemAvailable 8.98 GB で run の前提に届かず |

23:15 の fast cap では run を実行できなかった (MemAvailable 8.96 GB < 9.5 GB で拒否)。
**fast cap での比較は取得できていない。** 指示書どおり、これは合格条件にしない。
vendor 電源ポリシー、温度保護、周波数制限には一切介入していない。

## 11. 生成品質への影響

- 分割の計算結果は分割なしと**ビット一致しない** (CPU 行の丸めが違う)。これは
  Phase 50 の 33 本分割と同じ扱いで、行ごとの参照一致で正当性を確認している。
- **async は seq とビット一致する (修正後)。** つまり今回の修正で、非同期経路の
  数値は「逐次の正しい経路」と同一になった。修正前の async は 1 token 目から
  静かに間違っていた。
- NLL / perplexity / 品質指標は今回**新たに測っていない**。分割なしの logits が
  Phase 50 の基準 (`f3a0d235...` / `5585d4ed...`) と一致したため、既存の品質評価の
  前提は変わっていない。品質指標を都合よく変更はしていない。

## 12. Git

| 項目 | 値 |
|---|---|
| ブランチ | `codex/c51-perf` |
| 基準 | `8e5b850a3` (Phase 50 の HEAD) |
| HEAD | `143f1e43d` 以降の作業を含む (本文執筆時点) |
| コミット | 36 個 (Skill 取り込み 2、不具合修正 1、費用改善 1、スレッド改善 1、記録と解析 31) |
| 状態 | clean、**未 push** |
| メイン worktree | 未変更 (未コミット 9 ファイルはそのまま) |
| Phase 50 worktree | `8e5b850a3` のまま |

## 13. 未解決事項と再現コマンド

### 未解決
1. 27B 全体での 4 組以上の有効な pair (8.2)。cap とメモリの安定した窓が要る。
   `p51_finish.sh <ngl> <pairs>` が前提を検査してから 4 組をまとめて回す。前提を満たさないときは理由を出して止まる。
   **2026-11-14 追試: 4 組を実行し、cap が一致した有効 pair は 1 組だけだった (§15)。
   その 1 組では async が off より 6.7 % 遅く、5 % の優位は出ていない。追加測定は未完。**
2. join の待ち 17.8 ms を「入力の producer だけ待つ」形にする案は、DSP がキュー順に実行するため待ち時間が減らず、採らない (7.6)。
3. 修正前 lib で `--n 1 --repeats 1` を実行すると `FORTIFY: pthread_mutex_lock called on a destroyed mutex` で落ちる。
   修正後は落ちない。終了処理順の別問題と思われるが未特定。
4. `dspqueue_read` の 0x2e は DMA64=1 でのみ出る。AEE_EBADPARM と読めるが、定数の定義はこの repo に無い。分割経路とは独立。

### 再現コマンド

`phase51-notes.md` の 11 節と `phase51-small-repro.md` の 5 節に全文がある。
要点だけ:

```
SKEL=288e6712df9c8201ef05c12ec5704968d4f71407992cff8e0f03aa002ee0861d
cd /data/local/tmp/c46-27b

# 34 本を分割して 4 step (host share を小さくしてガードに触れない)
SPLIT_WEIGHTS=$(cat libs/split34.txt) GGML_HEXAGON_SPLIT_CPU_ROWS=32 \
  P51_DUMP=d34.bin P51_MODE=async WANT_SKEL=$SKEL sh p51_run.sh 4 replay 16 d34 1

# 行ごとの比較
python p51_dump_cmp.py d34.bin d34.bin.ref

# モデル無しの再現 (2 秒)
LD_LIBRARY_PATH=libs ADSP_LIBRARY_PATH=libs ./c51_split_repro --mode async --small --repeats 3
LD_LIBRARY_PATH=libs ADSP_LIBRARY_PATH=libs ./c51_split_repro --mode seq   --small --repeats 3

# 内訳
SPLIT_WEIGHTS=$(cat libs/split33.txt) TRACE=1 TIMELINE=1 GGML_HEXAGON_SPLIT_CPU_FRAC=0.25 \
  P51_MODE=async WANT_SKEL=$SKEL sh p51_run.sh 4 replay 9 t1 1
```

## 14. Phase 52 への引き継ぎ

1. **34 本目はもう怖がらなくてよい。** 34 本構成は正常に動き、行ごとの数値も
   参照と一致する。既定を 33 本にしておく理由はもう無いが、切り替えるなら
   Phase 50 の 33 本基準 (`1570917a...`) との比較を 1 回取ること。
2. **分割の非同期経路は「待ってから読む」が必須。** 診断を入れると不具合が
   消えるので、診断ありだけで正常と判断しない。小型ハーネス
   (`c51_split_repro`) を回帰試験として使うこと。
3. **行分割で 5% は取れない。** 上限は分割リストの DSP 時間で決まり、NGL を上げても増えない (1.0% 前後)。
   CPU/HTP の並列化で 5% を狙うなら、行分割ではなく
   「DSP 区間を CPU 区間と重ねる」構造 (層の依存関係を跨ぐ) が要るが、
   Phase 49 の実測では HTP<->CPU の往復 1 回が 7-12 ms で、往復を増やす案は
   利得を食い潰す。
4. **性能を測る前に cap とメモリの窓を確認する。** cap は 1 run の間に
   2112000 / 3187200 / 3033600 / 1555200 と動く。MemAvailable 9.5 GB を
   満たさないとガードが run を拒否する。
5. 端末は 7 日連続稼働。バッテリーは 11-13 00:22 に 16%、AC 非接続まで下がった。
   長時間の連続 run は避け、1 組ごとに冷却と充電の余裕を見ること。

6. **未検証の候補 (構造が違うので Phase 51 では試していない)。** 2 つの要求を同時に扱い、
   片方の token の DSP 区間ともう片方の CPU 区間を重ねる案。層の依存関係に縛られないので
   上限は DSP 区間の割合 (NGL=9 で 9%) に近づけるが、`llama_decode` が logits を返す前に
   同期する現在の構造を、投入と回収を分ける形に変える必要がある。

## 15. 2026-11-14 27B ABBA 追試 (NGL=9、33 本、4 pairs)

端末が復帰し、AC 給電 90 %、MemAvailable 10.2 GB、SwapFree 9.5 GB、
cap 3628800 / 3648000 の状態で `sh p51_finish.sh 9 4` を実行した。
8 run すべて exit 0、DSP abort なし、swap-out ガード停止なし。
総所要は 4 分 24 秒 (00:53:49-00:58:13)。

| pair | 先 | off min_ms | async min_ms | async/off | cap 一致 | 採用 |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 1 | off → async | 532.054 | 537.554 | +1.0 % | off は途中で 3513600 へ、async は 3187200 | no |
| 2 | async → off | 626.364 | 561.725 | **-10.3 %** | async に 3187200_3129600 が 1 sample、他は 3187200_3648000 | no |
| 3 | off → async | 526.582 | 559.334 | +6.2 % | off は 3187200、async は 2496000 | no |
| 4 | async → off | 506.060 | 539.834 | **+6.7 %** | 両方 2496000_2668800 で一致 | **yes** |

- 有効 pair は 4 組中 1 組だけ。その 1 組では **async が off より 6.7 % 遅い**。
- pair 2 は async が 10.3 % 速いが、cap sample が 1 点だけ異なるため採用しない。
- したがって「27B 全体で SPLIT=0 を 5 % 上回る」という再現性のある優位は、
  今回の追試でも確認できていない。B の判定は **PARTIAL のまま**。
- `swap` の生値は 40,558-314,273 (スクリプト出力単位)。ガード停止は 0 件。
- logits は off が全 run `5585d4ed...` (Phase 50 の分割なし基準)、
  async が全 run `2d4693c4...` (既知の分割経路のハッシュ) で、
  それぞれ決定的。off と async のハッシュが違うのは、
  分割が CPU host share を使う別の算術経路だからで、
  async 内の非決定性ではない (§2.3 の行単位検証と整合)。
- 34 本目は今回の対象外 (split33.txt)。34 本目の不存在確認 (§2) を
  この追試の結果で書き換えない。

## 16. 2026-11-14 継続: descriptor 最小再現・新規2バグ・ABBA

### 16.1 descriptor キャッシュの形状変更 — 再現した

`c51_desc_repro.cpp` を追加した。同じ `ggml_tensor` オブジェクトの `ne[1]` を
8 → 16 に変え、**synchronize を挟まず** 2 回の graph compute を同じ opbatch に
入れる。重みは F32 なので、descriptor が古いままだと数値が一致しない。

| lib | 結果 | 2 回目の shape1_diff |
| --- | --- | ---: |
| 修正前 `c51final` (44baedd) | **FAIL** | 8 / 16 |
| 修正後 (descriptor 検証 + sync 入り) | **PASS** | 0 / 16 |

`GGML_HEXAGON_VERBOSE=1` の `add-tensor` 列:

- 修正前: 2 回目に `w` の add-tensor が**出ない**。`size 8192 : 256:8:1:1` の
  descriptor が再利用され、DSP は 8 行のまま読む。
- 修正後: 2 回目に `add-tensor #3 w : ... offset 0 size 16384 : 256:16:1:1` が
  出る。新しい index と新しい offset/size が得られる。

これが `e30c6b7cd` の直接の reproduction である。ただし 34 本目全体の
異常終了を再現したものではない (16.3)。

### 16.2 33/34 本・split on/off の小型試験

`c51_split_repro` で n=34、mode off/seq/async、`--small`、repeats 3。

- async と seq は同一ハッシュ (`8eee2669...`、3 回とも一致)。
- off は別ハッシュ (`76f9261c...`、行の担当が違うため当然)。
- n=34 の split は 34 本すべて捕捉、readback ndiff 0。
- n=33 も同じコマンド列で回す予定だったが、実行前に端末が offline になったため
  **未測定**。33 vs 34 の比較は再開後に取り直す。
- NGL はモデル側の変数なので、この小型試験の変数には含めない。

### 16.3 DMA64=1 の `dspqueue_read 0x2e` — 条件を固定したが未修正

`c46_split_repeat` で再現:

| 条件 | 結果 |
| --- | --- |
| DMA64=0 | rc 0 |
| DMA64=1 | rc 134、`dspqueue_read failed: 0x2e` |
| type f32 / iq2_s、m 512/1024/2048/17408 | DMA64=1 はすべて 0x2e |
| OPPOLL=1 + DMA64=1 | 0x2e ではなく **timeout (124)** |

一方、`c51_split_repro` では split on + `GGML_HEXAGON_MM_SELECT=1` + DMA64=1 でも
**再現しない** (async/seq/off の n=4 と n=34 `--small`、n=1 の full)。
n=34 を full shape で回す試みは重みの構築が 180 秒を超えて timeout (124) したため、
判定に使っていない。したがって 0x2e は「分割経路一般」ではなく、
`c46_split_repeat` のバッファ構成 (WEIGHTS usage の w1/w2 + 2 buffer/opbatch) に
依存する。原因は未特定。DMA64=1 は Phase 49 以降の既知正常条件 (DMA64=0) から
外れるため、分割試験では引き続き DMA64=0 を使う。

### 16.4 `pthread_mutex_lock called on a destroyed mutex` — 再現せず

`c51base` (5c55b585、修正前)、`c51final` (44baedd)、descriptor 修正後の 3 つで、
`--n 1 --repeats 1` を off/async、full/`--small`、`--dump`/`--threads 6`/
`--rows 32` の組み合わせで各 3 回試した。**全件 rc 0、FORTIFY なし。**
Phase 51 の 1 回の観測は再現できなかった。修正前 lib 固有と断定はできないので、
「未再現・原因未特定」としてクローズする。

### 16.5 ABBA 追加測定 — BLOCKED

端末が TCP ADB (`192.168.55.15:5555`) のみになり、その後 offline。
再起動は禁止のため実施していない。offline 直前の状態は
cap 3628800/3648000 (fast)、AC 非給電、battery 89 %、
MemAvailable 6.09 GB で、27B runner の MemAvailable 9.5 GB ゲートにも
到達していなかった。追加 pair と fast cap の CPU 単独比 -5 % は未測定。

## 17. 2026-11-14 再接続後: 33/34 比較・ABBA 追加・0x2e の切り分け

端末は USB (`324753221196`) で復帰。skel は Phase 50 D のまま
(`288e6712...`)。再起動はしていない。

### 17.1 33 本 vs 34 本 (小型試験、`--small`、repeats 3)

| n | mode | split 本数 | hash (3 回とも同一) |
| ---: | --- | ---: | --- |
| 33 | off | 0 | `2fc532b4872e72a7` |
| 33 | seq | 33 | `32447938dc3d6159` |
| 33 | async | 33 | `32447938dc3d6159` |
| 34 | off | 0 | `76f9261ced66594d` |
| 34 | seq | 34 | `8eee266925de9793` |
| 34 | async | 34 | `8eee266925de9793` |

- 33/34 とも **async == seq**、3 回とも再現。
- 33 と 34 でハッシュが違うのは分割対象が 1 本増えるため当然で、
  どちらも同一構成内では決定的。
- **34 本目固有の異常は出ていない。** ただしこれは小型試験であり、
  27B 全体の 34 本構成を保証しない。

### 17.2 ABBA 追加測定 (2 回試行、有効 pair 0)

1 回目 (`p51_finish.sh 9 4`、14:38-14:46): 8 run 中 7 run exit 0、
1 run は guard stop (exit 6、`guard stopped=1`)。cap は
3187200 → 2496000 → 2227200 と降下し、**同一 cap の pair は 0**。

2 回目 (`p51_finish.sh 9 3`、14:55-15:02): 6 run 中 5 run exit 0、
1 run は guard stop。cap は 3187200 → 2496000 → 2227200 と再び降下し、
**同一 cap の pair は 0**。

| 試行 | run 数 | exit 0 | guard stop | 有効 pair |
| --- | ---: | ---: | ---: | ---: |
| 1 | 8 | 7 | 1 | 0 |
| 2 | 6 | 5 | 1 | 0 |

- 熱制限 (thermal_zone0 40-45 ℃) で cap が 2 段以上動くため、
  4 step × 8 run を 1 つの cap tier に収められない。
- guard stop は `pswpout` 由来で、`swap` 列が 272,827-427,478 と大きい run で発生。
  1024 MiB ガードは緩めていない。
- **fast cap (3628800/3648000) の窓は 14:34 時点で観測したが、
  27B を 4 step 回す間に 3187200 へ落ちた。** fast cap での
  「CPU 単独比 -5 %」は依然未測定。
- したがって split 並列化の性能判定は **「現構成では 5 % 優位を確認できず」**
  のまま。コードは保持する。

### 17.3 `dspqueue_read 0x2e` の切り分け (前進)

`c46_split_repeat` で再現する条件を絞った。

| 条件 | 結果 |
| --- | --- |
| DMA64=0 | rc 0 |
| DMA64=1 + `MM_SELECT=1` | rc 134、0x2e |
| DMA64=1 + `MM_SELECT=0` | rc 6 (`rep: op not supported`、0x2e ではない) |
| DMA64=1、threads 1/2/3/6 | すべて 0x2e |
| DMA64=1、type f32/iq2_s、m 512-17408 | すべて 0x2e |
| DMA64=1、`weights`/`noweights` | どちらも 0x2e |

`GGML_HEXAGON_VERBOSE=1` の差:

- DMA64=0: `fd 24 ... size 5271552 ... extended 0`、`add-buffer #0 ... vmem 5271552`
- DMA64=1: `fd 24 ... size 6291456 ... extended 1`、`add-buffer #0 ... vmem 0`

つまり DMA64=1 では重みバッファが **extended map** になり、skel 側は
`HAP_mem_request(HAP_MEM_MAP, HAP_MEM_FLAGS_EXTENDED_MAP)` で DSP VA を取る。
`libggml-htp-v81.so` に `extended mmap failed` の文字列はあるが、
今回のログにはそれが**出ていない**。0x2e は `dspqueue_read` の戻りであり、
DSP 側の応答が返らない/不正という形で現れる。

一方 `c51_split_repro` は同じ DMA64=1 + extended map でも
n=1/4/8/16/34、off/async のすべてで **0x2e を出さない**。
差は「1 opbatch に 2 buffer (w1/w2) を入れるか」で、
`c46_split_repeat` は w1/w2 の 2 buffer、`c51_split_repro` は 1 buffer。

**結論: 0x2e は DMA64=1 の extended map と、複数 buffer を 1 opbatch に
入れる構成の組み合わせで出る。分割経路そのものではない。原因は未特定。**
分割試験は DMA64=0 を維持する。

### 17.4 mutex 破壊 — 再確認

`c51base`/`c51final`/descriptor 修正後の 3 つで、off/async × full/`--small` ×
`--dump`/`--threads`/`--rows` を各 3 回。**全件 rc 0、FORTIFY なし。**
未再現のままクローズ。

### 17.5 4 step の 1 組 (同一 cap、15:22-15:27)

`p51_series.sh 9 4 replay` を 2 方向で回した。3 step は replay のトークン数 (4) と
一致せず exit 2 になるため使わない。

| 順序 | off min_ms | async min_ms | async/off | cap |
| --- | ---: | ---: | ---: | --- |
| off → async | 605.377 | 581.929 | **-3.9 %** | 両方 2496000_2668800 |
| async → off | 679.882 | 700.906 | **+3.1 %** | async は 2227200 へ降下、off は 2227200 |

- 1 組目は同一 cap で async が 3.9 % 速い。ただし 4 step の min_ms は
  step 1-3 の最小値で、step 0 のロード影響を含まない。
- 2 組目は cap が途中で降下したため比較不成立。
- 3.9 % は 5 % に届かず、方向も 2 組で一致しない。
  **split 並列化は「現構成では 5 % 優位を確認できず」** のまま。
- 参考: 15:10 の 2 組では off 656.295 / async 579.918 (-11.6 %) と
  async 656.752 / off 615.554 (+6.7 %) で、これも方向が一致しない。

## 18. 2026-11-14 夕: 0x2e の原因特定と修正、fast cap 窓の測定

### 18.1 前回の結論は誤りだった (buffer 本数ではない)

§17 は「0x2e は 1 opbatch に複数 buffer を入れる構成で出る」と結論したが、
これは**測定に使ったバイナリが古かった**ためである。

端末の `c51_split_repro` は 2026-11-13 00:14 のビルドで、
`GGML_HEXAGON_DMA64` を尊重する変更より前のものだった。そのため
`GGML_HEXAGON_DMA64=1` を渡しても実際は DMA64=0 で走っていた。

```
$ GGML_HEXAGON_DMA64=1 ./c51_split_repro --mode off --n 1 --repeats 1
ggml-hex: Hexagon Arch version v81, DMA64 disabled      <- 渡したのに無効
```

再ビルド後は同じコマンドで `DMA64 enabled` になり、weight buffer が
`extended 1` で map される。**その状態でも 0x2e は出ない。**
§17 の「1 buffer なら出ない」は DMA64=0 の測定であり、比較になっていなかった。

### 18.2 真のトリガー: 出力が extended mapping の中にあること

`c46_split_repeat` と `c51_split_repro` の構造差を 1 つずつ潰した。
残った差は「MUL_MAT の**出力**が WEIGHTS buffer の中にあるか」だけである。

- `c46_split_repeat`: `w2` `x2` `y2` を同じ context から確保するので、
  `y2` も WEIGHTS buffer に入り、DMA64=1 では extended mapping になる。
- `c51_split_repro`: 出力は activation buffer にあり、extended にならない。

これを直接試す `c51_dma64_repro` を書いた。op と形状は同一で、
出力の置き場所だけを変える。

| DMA64 | 出力の場所 | rc | 0x2e | logits hash |
| ---: | --- | ---: | ---: | --- |
| 0 | 別 buffer | 0 | なし | `041f7d7d77f25d83` |
| 0 | weights buffer | 0 | なし | `041f7d7d77f25d83` |
| 1 | 別 buffer | 0 | なし | `041f7d7d77f25d83` |
| 1 | weights buffer | **134** | **あり** | — |

3 回ずつ実行し、4 行とも毎回同じ結果だった。**トリガーは
「DMA64=1 かつ op の出力が extended mapping 内」で確定。**

### 18.3 原因

skel の `htp_mmap` は extended mapping を **読み取り専用**で作る。

```c
payload.mmap.prot  = HAP_MEM_CACHE_NON_SHARED | HAP_PROT_READ;
payload.mmap.flags = HAP_MEM_FLAGS_EXTENDED_MAP;
```

書き込み先がこの mapping だと DSP 側でフォールトし、応答が返らないまま
ホストの `dspqueue_read` が 0x2e になる。

`matmul-ops.c` には同じ理由のガードが**一部の入口にしか無かった**。

| 入口 | extended 出力のガード |
| --- | --- |
| `op_matmul_id` | あり (`dst` と `src1` と `ids`) |
| `op_matmul_id_nx` | あり (`dsts[p]` と `act` と `ids`) |
| `op_matmul` | **無し** |
| `op_matmul_nx` | **無し** |

`c46_split_repeat` の MUL_MAT は `op_matmul` を通るので、ガードを素通りして
フォールトしていた。`op_matmul_id` 系を使う実モデルの MoE 経路が無事だったのは
このためである。

### 18.4 修正

`op_matmul` と `op_matmul_nx` に、`op_matmul_id` と同じ判定を足した (13 行)。

```c
    // An extended (DMA64) mapping is read only, so a kernel that writes into one
    // faults instead of returning an error. op_matmul_id and op_matmul_id_nx
    // already refuse that case; without the same check here the plain MUL_MAT
    // paths take the fault and the host sees dspqueue_read 0x2e.
    if (htp_tensor_is_extended(octx->src[1]) || htp_tensor_is_extended(octx->dst)) {
        return HTP_STATUS_NO_SUPPORT;
    }
```

`op_matmul_nx` は活性の位置が `src[n_weights]` なので、そちらを参照する。

### 18.5 修正の検証

skel を再ビルドして端末に置いた。

| 項目 | 値 |
| --- | --- |
| 修正前 skel (Phase 50 D) | `288e6712df9c8201ef05c12ec5704968d4f71407992cff8e0f03aa002ee0861d` |
| 修正後 skel | `70b9ed4fafb2ab298baf9f9bbed3c38a4a457cc4838775a690b9168dfb7eca1f` |

`c51_dma64_repro --out wbuf`、DMA64=1、3 回:

| 項目 | 修正前 | 修正後 |
| --- | --- | --- |
| rc | 134 | 134 |
| `dspqueue_read 0x2e` | 1 | **0** |
| ホストのメッセージ | `dspqueue_read failed: 0x0000002e` | `dspcall : dsp-rsp NO-SUPPORT` |

**0x2e は消え、理由の分かるエラーになった。** 落ちるのは
「extended mapping を出力に使う」という構成自体が不正だからで、
不正が静かなフォールトではなく明示的な拒否として返るようになった。

`--out abuf` は修正前後とも rc 0、hash `041f7d7d77f25d83` で不変。

### 18.6 27B 本線の回帰 (修正 skel)

修正 skel で 27B を 4 step、NGL=9、33 本分割で回した。

| mode | exit | min_ms | logits sha256 (先頭 16) | split 本数 |
| --- | ---: | ---: | --- | ---: |
| off | 0 | 458.539 | `5585d4edc47ee750` | 0 |
| async | 0 | 532.010 | `2d4693c43c0cc307` | 33 |

- off の hash は Phase 50 の ngl=9 分割なし基準 `5585d4edc47ee750` と一致。
- async の hash は §17 の async と一致 (`2d4693c4...`)。
- **修正 skel は 27B の分割経路を変えていない。** 分割は
  `op_matmul` を通るが、出力は activation buffer にあり extended ではないため
  新しいガードに掛からない。
- この run は cap `3187200_3648000` (capped)。性能比較には使わない。

### 18.7 fast cap 窓での ABBA (不成立)

16:22-16:28 に fast cap (`3628800/3648000`、tz0 30.7 ℃) の窓が開いたので
1 組取った。**両 run とも cap が 1 sample だけ動いたため、§15/§17 と同一基準で
不成立**として除外する。数値は参考として残す。

| run | exit | min_ms | cap の内訳 | logits |
| --- | ---: | ---: | --- | --- |
| off | 0 | 461.042 | `3628800_3648000` x10、`3398400_3648000` x1 | `5585d4ed...` |
| async | 0 | 527.604 | `3628800_3648000` x10、`3187200_*` x2 | `2d4693c4...` |

- 両方とも step 0 のロードを含む 4 step。off 461 ms は今回観測した最小。
- 方向は async が 14 % **遅い**。ただし不成立 run なので判定には使わない。
- 窓は 16:28 に閉じた (cap 3187200、tz0 49.2 ℃)。以後 16:32 まで再開せず。

### 18.8 2 度目の fast cap 窓 (16:35、これも不成立)

16:35 に cap が `3628800` へ戻り、tz0 37.3 ℃、MemAvailable 10.8 GB だったので
もう 1 組取った。**また不成立。**

| run | exit | min_ms | cap の内訳 | logits |
| --- | ---: | ---: | --- | --- |
| off | 0 | 488.411 | `3628800_3648000` x7、`3398400_*` x1、`3187200_*` x3 | `5585d4ed...` |
| async | 0 | 556.898 | `3187200_3648000` x11 (開始時から既に降下) | `2d4693c4...` |

- off は途中で 3187200 へ降下、async は開始時点で既に 3187200。
  **同一 cap の pair にならない。**
- async の step は 604.6 → 556.9 → 690.0 → 798.4 と後半で悪化しており、
  熱制限の影響が見える。
- 16:36 に cap 3187200、tz0 48.4 ℃。窓は約 1 分で閉じた。

**fast cap の窓は 16:22 と 16:35 の 2 回観測したが、どちらも 1 組 (2 run、
約 2 分) を同一 cap で完走できなかった。** 27B の 4 step は 1 run あたり
step 0 のロードを含めて 30-60 秒かかり、その間に熱制限が入る。
画面消灯とアイドル待ちだけでは窓を延ばせない。

### 18.9 判定

| 目標 | 判定 | 根拠 |
| --- | --- | --- |
| 目標1 pair 確保 | **BLOCKED** | fast cap 窓を 2 回 (16:22-16:28、16:35-16:36) 観測したが、4 run すべてで cap が動き、同一 cap の pair が 0 組 |
| 目標1 並列高速化 | **PARTIAL** | 有効 pair 0 組。§17 までの全測定でも 5 % 優位は未確認。SPLIT=0 既定を維持 |
| 目標1 CPU 単独比 | **未測定** | pair が揃わず、窓も閉じたため実施せず |
| 目標2 トリガー特定 | **PASS** | 「DMA64=1 かつ出力が extended mapping 内」で確定。4 条件 × 3 回が全て一致 |
| 目標2 最小修正 | **PASS** | 13 行のガードで 0x2e → `NO-SUPPORT`。27B 回帰も off/async とも hash 不変 |

### 18.10 コミット

| commit | 内容 |
| --- | --- |
| `ddcf2a54a` | `fix(hexagon): refuse an extended mapping as a MUL_MAT output` |
| `7bdcc82f8` | `test(c51): add the DMA64 output-location reproduction` |
| `365b12be5` | `docs(c51): record the 0x2e root cause, the fix and the fast cap window` |

### 18.11 再現コマンド

```sh
# トリガーの確認 (DMA64=1、出力の場所だけを変える)
cd /data/local/tmp/c46-27b
for o in abuf wbuf; do
  env LD_LIBRARY_PATH=libs ADSP_LIBRARY_PATH=libs \
    GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_OPPOLL=0 GGML_HEXAGON_DMA64=1 \
    ./c51_dma64_repro --out $o --m 17408 --k 5120 --repeats 1
done
# abuf -> rc 0 / hash 041f7d7d77f25d83、wbuf -> 修正前 0x2e、修正後 NO-SUPPORT

# 27B 回帰 (修正 skel)
export SPLIT_WEIGHTS=$(cat libs/split33.txt)
export WANT_SKEL=70b9ed4fafb2ab298baf9f9bbed3c38a4a457cc4838775a690b9168dfb7eca1f
sh p51_series.sh 9 4 replay off:0 async:0.25
```

skel のビルド:

```powershell
$env:HEXAGON_SDK_ROOT="C:\Qualcomm\Hexagon_SDK\6.6.0.0"
$env:HEXAGON_TOOLS_ROOT="C:\Qualcomm\Hexagon_SDK\6.6.0.0\tools\HEXAGON_Tools\19.0.07"
cmake --build build-c51 --target htp-v81
```

### 18.12 次の一手

1. **fast cap の窓は 1-6 分しか保たない。** 1 組 (2 run) に約 2 分かかるので、
   窓に入ったら 2 組を続けて取るには冷却を強める必要がある。
   画面消灯とアイドル待ちだけでは足りない。
2. 0x2e は**修正済み**。DMA64=1 を使う経路の安全性は上がったが、
   分割既定は DMA64=0 のままにする (Phase 49 以降の既知正常条件)。
3. 分割並列化の 5 % は依然未確認。有効 pair が 0 組なので、
   性能判定は「現構成では不採用」を維持する。

## 19. 2026-11-14 夜: 熱制限安定 tier での ABBA 4 組と性能判定の確定

§18 で 0x2e が解決したので、残る課題は分割並列化の性能判定だけになった。
fast cap の窓は 1-6 分で閉じて pair が取れないため、**熱制限で cap が
安定 tier に張り付いた状態**で測る方針に切り替えた。

### 19.1 ウォームアップと tier の推移

AC 給電のまま run を重ね、cap が動かなくなる tier を探した。

| 時刻 | cap0 / cap7 | 状態 |
| --- | --- | --- |
| 16:43 | 3628800 / 4396800 | fast。1 run 目はロード中に swap guard (275,896 pages) |
| 16:45 | 3628800 / 4396800 | fast だが 1 run 中に 5 種類の cap が混在 |
| 16:45 | 3187200 / 3648000 | 11 sample すべて同一。**安定** |
| 17:00-17:20 | 2496000 → 2227200 | 熱上昇で降下。load 8-17 で不安定 |
| 17:50 | 2496000 / 2668800 | 28-29 sample すべて同一。**安定** |
| 18:05 | 2496000 / 2668800 | 33-49 sample すべて同一。**安定** |

**`2496000_2668800` が最も長く保つ安定 tier** だった。§15/§17 で使った
tier と同じで、比較の継続性もある。

### 19.2 環境側の障害 (測定を妨げたもの)

測定を妨げたのは分割ではなく端末の環境だった。すべて記録する。

| 事象 | 内容 | 対処 |
| --- | --- | --- |
| swap guard | ロード/prefill 中に pswpout 増分が 1024 MiB 超。17:09 以降 6 回 | ガードは緩めず、メモリを空けて再試行 |
| 背景アプリ | `dex2oat64` 160 %、GMS・Play Store・ブラウザが次々復帰 | 既知の安全な範囲で force-stop |
| USB 切断 | 16:56 に USB ADB が切れ TCP (`192.168.55.15:5555`) へ | 再起動せず TCP で継続。17:44 に USB 復帰 |
| load 上昇 | 最大 17.3。run の step 時間が 3-37 秒に乱れる | load が下がるのを待ってから測定 |

**swap guard は 6 回発動し、いずれも「モデルのロード/prefill」で起きた。**
分割の有無と無関係で、off でも async でも同じように出た。
ガードは 1024 MiB のまま一切緩めていない。

### 19.3 有効 pair (4 組、すべて tier `2496000_2668800`)

`P51_CAP_STABLE YES` (全 sample で cap0/cap7 が不変) の run だけを pair にした。
**4 組すべてが同一 tier・同一 cap。**

| pair | 順序 | off min_ms | async min_ms | async/off | off の cap sample | async の cap sample |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | off → async | 525.184 | 560.691 | **+6.8 %** | 40 (YES) | 37 (YES) |
| D | off → async | 508.291 | 516.331 | **+1.6 %** | 34 (YES) | 49 (YES) |
| F | async → off | 510.444 | 529.426 | **+3.7 %** | 42 (YES) | 45 (YES) |
| G | async → off | 532.384 | 630.040 | **+18.3 %** | 34 (YES) | 34 (YES) |

- 改善率 (async が off より速い場合を正) は **-6.8 %, -1.6 %, -3.7 %, -18.3 %**。
- **4 組すべてで async が遅い。** 速い pair は 0 組。
- 中央値は (-3.7 + -6.8) / 2 = **-5.25 %**、つまり async が約 5 % 遅い。
- logits は全 run で off `5585d4edc47ee750...`、async `2d4693c43c0cc307...`。
  §17/§18 と同一で、数値の再現性は保たれている。

### 19.4 不成立 run (除外せず記録)

| 時刻 | 内容 | 不成立の理由 |
| --- | --- | --- |
| 16:46 | off 763.364 / async 517.167 | async の cap7 が 1 sample 変化 |
| 16:48 | async 514.441 / off 512.744 | off の cap7 が 1 sample 変化 |
| 17:00 | off 550.280 / async 681.241 | async が 2496000 へ降下、step が 9257/15462 ms |
| 17:08 | off 495.329 / async guard | async が prefill 中に swap guard |
| 17:11 | off guard / async guard | 両方 swap guard (354,029 / 348,277 pages) |
| 17:45 | off 517.344 / async 490.714 | 両方 cap7 が 1 sample 変化 |
| 17:47 | async 544.831 / off 472.517 | off が 2 sample 変化 |
| 17:48 | off 461.253 | cap7 が 1 sample 変化 |
| 17:55 | async 546.921 / off 612.465 | off が 2227200 へ降下 |
| 17:59 | off guard / async 629.380 | off が swap guard |
| 18:08 | async 529.359 / off 598.283 | off が 2227200 へ降下 |
| 18:14 | off guard / async 529.426 | off が swap guard、cap も混在 |
| 18:21 | NGL=0 off guard | CPU 単独は全層 RAM 常駐で swap guard |

**不成立 13 件、成立 4 件。** 不成立の主因は cap7 (prime core) の 1 sample 変化で、
これは熱制限が 96 ℃ 付近で入るため。cap0 は全 run で安定していたが、
§15/§17 と同じ基準 (1 sample 差も不成立) を適用した。

### 19.5 CPU 単独比 (任意) — 実施できず

同 tier で NGL=0 を 1 run 試したが、**swap guard で停止**した
(307,455 pages ≈ 1200 MiB)。NGL=0 は 27B の全層を CPU 側に置くため、
分割時より多くの匿名メモリを必要とする。ガードは緩めず、指示の
「熱や時間の都合で難しければスキップする」に従って**未測定**とする。
**計測不能な時間を 0 として扱わない。**

### 19.6 判定

| 目標 | 判定 | 根拠 |
| --- | --- | --- |
| pair 確保 | **PASS** | 同一 tier `2496000_2668800`・同一 cap の有効 pair が **4 組** (基準 3 組以上) |
| 並列高速化 | **FAIL** | 4 組すべてで async が遅い (速い pair 0 組)。中央値 **-5.25 %** (async が遅い) |
| CPU 単独比 | **未測定** | NGL=0 が swap guard で停止。ガードは緩めていない |

**並列高速化は FAIL。** 合格条件は「過半数で async が速く、中央値 5 % 以上」だが、
速い pair が 0 組で中央値もマイナスなので、明確に FAIL に当たる。
**SPLIT=0 既定を維持する。** 分割コードは残すが、既定経路には昇格しない。

### 19.7 なぜ遅いか (観測事実のみ)

- async は DSP シェアを投入してから CPU シェアを実行する。CPU シェアは
  3 スレッドで数 MiB の行列積を担当する。
- 4 組の async は 516-630 ms、off は 508-532 ms。差は 8-98 ms。
- 端末の load が 5-10 と高く、CPU 側は他プロセスと資源を奪い合う
  (`references/hexagon.md` の「CPU と DSP に分ける構成は CPU 側が不利」)。
- **この観測から原因を断定しない。** 今回測ったのは「この構成・この tier で
  5 % 優位が無い」ことだけである。

### 19.8 コミット

| commit | 内容 |
| --- | --- |
| (本節) | `docs(c51): settle the split speedup at a stable thermal tier` |

### 19.9 再現コマンド

```sh
# 安定 tier に張り付かせる (1-2 run)
cd /data/local/tmp/c46-27b
export SPLIT_WEIGHTS=$(cat libs/split33.txt)
export WANT_SKEL=288e6712df9c8201ef05c12ec5704968d4f71407992cff8e0f03aa002ee0861d
sh p51_series.sh 9 4 replay off:0

# cap が動かないことを確認してから pair を取る
sh p51_series.sh 9 4 replay off:0 async:0.25
sh p51_series.sh 9 4 replay async:0.25 off:0

# 各 run の判定は P51_CAP_STABLE を見る (YES の run だけを pair にする)
grep P51_CAP_STABLE p51-series-*.log
```

### 19.10 次の一手

1. **分割並列化は FAIL で確定。** SPLIT=0 を既定のまま維持する。
   分割コードは残すが、既定経路へ昇格させない。
2. 高速化を目指すなら、CPU シェアの担当行数 (`SPLIT_CPU_FRAC`) を
   下げて CPU 側の負担を減らす方向が考えられる。ただし今回は測っていない。
3. CPU 単独比は未測定のまま。NGL=0 は全層常駐でメモリが足りないため、
   より小さいモデルか、メモリの空いた状態で別途測る必要がある。
4. fast cap での pair も未測定のまま (§18 と同じ理由)。

### 19.11 追記: fast cap の窓が再び開いたので試した (18:44-18:48)

§19 の測定後、18:44 に cap が `3628800_3648000` へ戻り、tz0 32.3 ℃
(これまでで最も低い) だったので、未測定だった fast cap の pair を試した。
**pair は取れなかったが、fast cap で完全安定の off run を 1 本確保した。**

| 試行 | run | exit | min_ms | cap の内訳 | 判定 |
| --- | --- | ---: | ---: | --- | --- |
| 1 | off | 0 | 659.759 | `3628800_3648000` x12、`3513600_*` x1 | 不成立 (1 sample 変化) |
| 1 | async | 6 | — | `3628800_*` x10、`3513600_*` x1、`3398400_*` x1 | 不成立 (swap guard 273,691 pages) |
| 2 | off | 0 | **457.534** | `3628800_3648000` x11 | **CAP_STABLE YES (12/12)** |
| 2 | async | 0 | 503.909 | `3628800_*` x7、`3513600_*` x1、`3187200_*` x4 | 不成立 (cap 降下) |

- 試行 2 の off は **fast cap で 12 sample すべて同一**、457.534 ms。
  これは今回観測した off の最速値で、§19 の安定 tier (508-532 ms) より約 10 % 速い。
  **fast cap では CPU 側が速くなるという §8.4 の算術と方向は一致する。**
- しかし async は同じ fast cap を保てず (7 sample で降下)、pair にならなかった。
- 18:48 に cap 3187200 へ降下。窓は約 4 分で閉じた。

**fast cap の pair と CPU 単独比は依然未測定。** 1 本の off だけでは
比較にならないので、判定には使わない。数値は次回の窓の参考として残す。

## 20. Phase 51 最終判定 (クローズ)

Phase 51 をここで閉じる。分析は各節にあり、本節は判定と参照だけを残す。

### 20.1 最終判定一覧

| 項目 | 判定 | 参照 |
| --- | --- | --- |
| 33 本 vs 34 本の比較 | **PASS** | §17.1 (async == seq、3 回再現、34 本目固有の異常なし) |
| `dspqueue_read 0x2e` | **PASS** | §18.2-18.5 (原因特定＋修正、`ddcf2a54a`) |
| mutex 破壊 | **CLOSED (未再現)** | §16.4 / §17.4 |
| 同一 cap の pair 確保 | **PASS** | §19.3 (安定 tier で 4 組) |
| 分割並列化の高速化 | **FAIL** | §19.3 (4 組すべて async が遅い、中央値 -5.25 %) |
| fast cap の CPU 単独比 | **未測定 (保留)** | §18.7-18.8 / §19.11 (窓が 1-6 分で閉じる) |

### 20.2 分割並列化の扱い

**行分割のコードは保持するが、既定経路には昇格しない。** `SPLIT=0` が既定のまま。
同一 cap の 4 組すべてで async が遅く、中央値 -5.25 % だったため。

### 20.3 0x2e の修正内容

skel の `matmul-ops.c` で、`op_matmul` と `op_matmul_nx` に extended mapping を
出力に使えないようにするガードを追加した (13 行)。`op_matmul_id` 系には
元からあったが、plain MUL_MAT の経路に無かった。修正前 skel
`288e6712...`、修正後 `70b9ed4f...`。

### 20.4 コミット一覧

| commit | 内容 |
| --- | --- |
| `ddcf2a54a` | `fix(hexagon): refuse an extended mapping as a MUL_MAT output` |
| `7bdcc82f8` | `test(c51): add the DMA64 output-location reproduction` |
| `365b12be5` | `docs(c51): record the 0x2e root cause, the fix and the fast cap window` |
| `b711127a9` | `docs(c51): record the second fast cap window and renumber the section` |
| `712b80314` | `docs(c51): settle the split speedup at a stable thermal tier` |
| `2ab746abc` | `docs(c51): record the third fast cap window` |

ブランチ `codex/c51-perf`、HEAD `2ab746abc`、push 済み。

## 付録 A. 変更前後のコード

3 つの修正はすべて `ggml/src/ggml-hexagon/ggml-hexagon.cpp` の中にある。

### A.1 非同期分割の入力待ち (bb2607d4f)

```diff
@@ static ggml_status hex_split_run(ggml_hexagon_session * sess, const htp_opnode &
     uint64_t    tw0 = 0, tw1 = 0;
     ggml_status rc  = GGML_STATUS_SUCCESS;
 
+    // The host share reads its input from device memory, and the op that writes
+    // that input may still be in flight. Join the ops submitted before this one
+    // first: the DSP runs its queue in order, so they own every input of this op.
+    tw0 = hex_now_us();
+    sess->flush_sync();
+    tw1 = hex_now_us();
+
     if (nosync) {
@@ async 分岐の最後
-        if (!seq) {
-            tw0 = hex_now_us();
+        if (seq || diag) {
             sess->flush_sync();
-            tw1 = hex_now_us();
         }
```

### A.2 分割時の node cache (302685f2d)

```diff
-    // Check for cache hit
-    bool cache_hit = (graph->uid != 0 && sess->cached_uid == graph->uid && !hex_split_enabled());
+    // Check for cache hit. The split op rebuilds its kernel params and follows the
+    // current tensors on every call, so the cached node list stays valid there too.
+    bool cache_hit = (graph->uid != 0 && sess->cached_uid == graph->uid);
```

### A.3 host シェアのスレッド既定 (5996ec0c4)

```diff
         const char * e = getenv("GGML_HEXAGON_SPLIT_THREADS");
-        const int    n = e ? atoi(e) : 6;
+        // One host share is a few MiB, so six threads cost more than the work:
+        // 142-151 ms with three against 171-226 ms with six on the real shapes.
+        const int    n = e ? atoi(e) : 3;
```
