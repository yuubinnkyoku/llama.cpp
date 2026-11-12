# Phase 51 作業記録

## 0. Git と環境の基準

| 項目 | 値 |
|---|---|
| リポジトリ | `D:/ghq/github.com/yuubinnkyoku/llama.cpp` |
| メイン worktree | 2717a9293 `codex/phase20-htp-correctness-memory`、未コミット 9 ファイル (保護) |
| Phase 50 worktree | `llama.cpp-c50` = 8e5b850a3 `codex/c50-perf`、clean、32 コミット、未 push |
| Phase 51 worktree | `llama.cpp-c51` = 8e5b850a3 から作成、ブランチ `codex/c51-perf` |
| 汎用 Skill 取り込み | 618a0a5 -> 173b6f54b、a937506 -> 3ee45e81a (conflict なし) |
| 端末 | 192.168.55.16:5555、cap0 2112000 / max 3628800、cap7 1862400 / max 4608000 |
| 端末 skel | 288e6712df9c8201ef05c12ec5704968d4f71407992cff8e0f03aa002ee0861d (Phase 50 D) |
| 端末 host lib | libggml-hexagon.so d6f60f65e2dc3ca18eadd95c7f6839758d8d82f716ef577a3bba15af68f3ec08 |

Phase 50 報告との相違: なし (c50 の HEAD、コミット数、clean、未 push すべて一致)。

## 1. Gate 0: 34 テンソル目の不具合について、実在する記録の棚卸し

端末 `/data/local/tmp/c46-27b/` に Phase 46-47 の生ログが残っていたので、
推測ではなく実物を読んだ。以下はすべて一次資料の引用。

### 1.1 唯一の異常終了記録は DMA64=1 のもの

`A.log` から `E2.log` は `run_abcd.sh` と `run_repeat.sh` が作った。
両スクリプトの差は 1 つだけで、`GGML_HEXAGON_DMA64` の値である。

```
run_abcd.sh   : GGML_HEXAGON_DMA64=0  ./c46_split_repeat iq2_s 17408 5120 <iters> 6 s2 weights <ref>
run_repeat.sh : GGML_HEXAGON_DMA64=$dma ./c46_split_repeat iq2_s 17408 5120 1 6 s2 weights noref
```

| ログ | DMA64 | 結果 |
|---|---|---|
| A, A2, A3, A4, B, C, D | 0 | `C46EXIT rc=0`、`sentinel=0`、`htp_vs_htpfull=0.000000` |
| E1, E2 | 1 | `ggml-hexagon.cpp:4713: ggml-hex: dspqueue_read failed: 0x0000002e` -> `ggml_abort` |

E1 の再現手順 (実物):

```
cd /data/local/tmp/c46-27b
LD_LIBRARY_PATH=libs ADSP_LIBRARY_PATH=libs GGML_HEXAGON_DEVICES=HTP0:0 \
GGML_HEXAGON_MM_SELECT=1 GGML_HEXAGON_IQ2S_GATHER=1 GGML_HEXAGON_IQ3XXS_GATHER=1 \
GGML_HEXAGON_IQ3S_GATHER=1 GGML_HEXAGON_FFN_OFFLOAD=1 GGML_HEXAGON_OPPOLL=0 \
GGML_HEXAGON_DMA64=1 ./c46_split_repeat iq2_s 17408 5120 1 6 s2 weights noref
```

- 失敗点は `compute_htp_begin 0` の直後。CPU シェアは完了済み、DSP シェアの
  投入後の応答待ちで `dspqueue_read` が 0x2e (AEE_EBADPARM) を返す。
- 同じ 1 テンソル試験が DMA64=0 では 5 回とも rc=0。したがってこの異常終了は
  **テンソル本数の関数ではなく DMA64 の関数**である。
- `run_split49.sh` 以降の分割試験はすべて `GGML_HEXAGON_DMA64=0` を明示しており、
  Phase 49/50 の既知の正常条件も DMA64=0 である。

### 1.2 34 本目を含む実モデル試験は、実際には 33 本しか分割していなかった

`p47-f34-*.log` は ngl=9 / frac 0.9 / 5 step の実モデル ABBA。
非同期 3 本はいずれも rc=0。

しかし、その run ディレクトリの `c46-env.txt` を見ると
`SPLIT_WEIGHTS` に **`blk.63.attn_output.weight` が入っていない**。
stderr の `split weight` 行も 33 本で、34 本目は捕捉されていない。

つまりこの記録は「34 テンソル構成が動いた」証拠にはならない。
ラベル `f34` はテンソル本数を意味していない (本数の 33 と一致する)。

### 1.3 過去の run ログに異常終了は 1 件もない

```
adb shell "cd /data/local/tmp/c46-27b/runs && grep -rl 'dspqueue' . | wc -l"   -> 0
adb shell "cd /data/local/tmp/c46-27b/runs && grep -rl 'ggml_abort' . | wc -l" -> 0
```

747 ディレクトリ、`DECODE step` を含むファイル 1397 件。分割の rc 2 は
`approve.sh` の単発承認が消費済みで `REFUSING: no approval file` になったもので、
経過時間 0.1-0.2 秒、run ディレクトリは前 run のまま、という形で区別できる。
Phase 47 の `P47_FAIL` はこの拒否である。

### 1.4 分割リストの実体

端末 `/data/local/tmp/c46-27b/libs/split33.txt` (745 B, 2026-11-08 18:18) と
`split34.txt` (770 B, 2026-11-08 19:36) が現存する。差は 1 本だけ:

```
+blk.63.attn_output.weight
```

34 本の内訳は層 56-63 の `attn_qkv` / `attn_gate` / `ssm_out` / `ffn_*` /
`attn_k` / `attn_v` / `attn_output`。`blk.59.attn_output.weight` は 33 本側にも入っており、
34 本目だけが除外されていた。

`blk.63.attn_output.weight` の実体は IQ3_S、ne0 6144、ne1 5120 で、
Phase 50 の ngl=16 プロファイル (`prof-A-ngl16.txt`) でも
`MUL_MAT|blk.63.attn_output.weight ... iq3_s` として DSP 上で動いている。
型・形状・配置のいずれも 33 本目までのテンソルと同種である。

## 2. Gate 1 の設計 (この記録の時点で未実施)

最小再現の第一段は、27B を増やさずに 34 本目を実際に分割することにする。
理由: 34 本目は ngl=16 ですでに HTP 上にあり、ホストシェアの追加は
IQ3_S 6144x5120 / frac 0.5 で 2560 行 x 2640 B = 6.76 MiB にすぎない。
メモリガードに触れずに、実モデルで直接「34 本目を分割したとき」を観測できる。

手順 (p51_run.sh で実行):

1. `SPLIT=1 seq` + split34 + ngl=16 + 4 step replay (逐次経路の正当性)
2. `SPLIT=1 async` + split34 + ngl=16 + 4 step replay (非同期経路)
3. logits を Phase 50 の基準 `5585d4ed...` (分割なし) と比較
4. 33 本 + ngl=16 の `1570917a...` との比較で、34 本目だけの寄与を分離

縮小試験が成功しても実モデルの不具合が解決した証拠にはしない。
逆に、実モデルで 34 本目が再現しない場合は「再現せず」と記録し、
1.1 の DMA64 経路を別仮説として切り分ける。

## 3. 試験スクリプト

- `p51_run.sh` (端末 `/data/local/tmp/c46-27b/p51_run.sh`)
  p50_run.sh の force-stop / cap 記録 / メモリ記録をそのまま使い、
  分割モードだけを `P51_MODE` で切り替える。

## 4. 実測: 34 本目を実際に分割した (2026-11-10、cap 2112000/1862400)

スクリプト `p51_run.sh` は p50_run.sh の force-stop / cap 記録 / メモリ記録を
そのまま使い、分割モードだけを `P51_MODE` で切り替える。skel は Phase 50 D
(288e6712...)、ホストライブラリは d6f60f65... 。

| tag | 構成 | 4 step の結果 | swap-out |
|---|---|---|---|
| base1 | NGL=16 SPLIT=0 | exit 0、1091.9 / 716.2 / 701.7 / 6989.3 ms | 594 MiB |
| s34a1 | NGL=16 async split34 frac 0.5 | **exit 6 (ガード)**、decode 未到達 | 1.24 GiB |
| s34a2 | 同上 2 回目 | **exit 6 (ガード)**、step 0-1 は完走 (1564.7 / 722.4 ms) | 1.25 GiB |
| s34r32 | NGL=16 async split34 CPU_ROWS=32 | **exit 0、4 step 完走** (829.1 / 756.5 / 972.0 / 9652.7 ms) | 587 MiB |

### 4.1 34 本目は捕捉され、動く

```
ggml-hex: split weight blk.63.attn_output.weight (iq3_s): k 6144 m 5120 host rows 2560
  dsp rows 2560 row 2640 B, host 6.45 MiB, dsp 6.45 MiB in 7.50 MiB at off 1032157184
  usage 1, alias 1, r mod 32 0, readback ndiff 0 first -1, in buf 1
```

`SPLIT_COUNT 34`。readback は ndiff 0、alias はモデルバッファ内、境界は 32 行整列。
CPU_ROWS=32 の run でも `host rows 32 dsp rows 5088 ... readback ndiff 0`。

**34 本構成の非同期分割は、この端末で異常終了しない。**
frac 0.5 の 2 回が止まったのは swap-out ガード (1.24-1.25 GiB > 1024 MiB) で、
DSP エラーも `dspqueue_read` も出ていない。

### 4.2 ガードを作動させたのは host share のメモリ

分割は 33/34 本で `host %.2f MiB` の匿名メモリを確保する。34 本 frac 0.5 で
**合計 400.93 MiB** (`added_total_B 420392960`)。SPLIT=0 の 594 MiB に足すと
1.05 GiB を超え、ガードの 1024 MiB に触れる。

CPU_ROWS=32 では host share が 34 x 32 行に縮み、合計で数 MiB。
このとき swap-out は 587 MiB で SPLIT=0 と同等、4 step 完走。

したがって「34 本目だから落ちる」のではなく、
「frac 0.5 の host share がこの端末のメモリに収まらない」が正しい。

### 4.3 logits

| run | logits sha256 |
|---|---|
| base1 (SPLIT=0, NGL=16, 4 step replay) | f3a0d235c0da32d0c40ed901b9236026e328358a05b77b852b1debdd680adca7 |
| s34r32 (34 本分割、async) | 4d4c311786f735f8f2eb874c3735bc058e76f0386d9622fdbfe69a7f807beb01 |

base1 のハッシュは Phase 50 が記録した分割なし基準 `f3a0d235...` と**一致**した。
別セッション・別日でも同じ入力なら同じ logits になることの確認である。

分割側のハッシュが違うのは設計どおり (分割行は CPU が計算するので丸めが変わる)。
Phase 50 も 33 本分割で `1570917a...` という別ハッシュを、行ごとの参照一致を
確認したうえで受け入れている。34 本目も同じ行ごとの確認が必要で、未実施。

## 5. Gate 3: 損失の出どころ (NGL=9 で分離)

NGL=9 は NGL=16 より遥かにメモリに優しい。SPLIT=0 の swap-out は **43 MiB**。

`GGML_HEXAGON_TIMELINE=1` の 1 行は 1 decode step の HTP backend 内訳である
(submit = 投入、wait = DSP 応答待ち、prep = そのどちらでもない host 時間)。

| 構成 | step | ops | batches | submit_us | wait_us | prep_us |
|---|---|---|---|---|---|---|
| SPLIT=0 | 1 | 229 | 1 | 251 | 60842 | **306** |
| SPLIT=0 | 2 | 229 | 1 | 263 | 59448 | **196** |
| SPLIT=1 (33 本) | 1 | 229 | 34 | 5917 | 17074 | **170801** |
| SPLIT=1 (33 本) | 2 | 229 | 34 | 1374 | 14807 | **163414** |

**損失の正体は prep の 163-171 ms/token。** キャッシュが効いている SPLIT=0 では
同じ 229 ops の prep が 0.2-0.3 ms である。差は約 170 ms/token。

原因はコード上で 1 行:

```cpp
bool cache_hit = (graph->uid != 0 && sess->cached_uid == graph->uid && !hex_split_enabled());
```

分割を有効にすると、1 token ごとに 229 ops の `htp_opnode` 構築と
パラメータ再計算をやり直す。33 本の CPU シェア (実測 14.9 ms) の 10 倍以上を
ここで失っている。

### 5.1 分割そのものの費用対効果 (SPLIT_TRACE)

NGL=9、33 本、frac 0.5、4 step の trace 集計 (中央値の和):

- `cpu_us` 合計 59,558 (1 step あたり 14,890 us)
- `wait_us` 合計 5,301 (1 step あたり 1,325 us)
- `op_us` 合計 68,272 (1 step あたり 17,068 us)
- `wait_us == 0` の op は 132 中 2 本。ほぼ全 op で CPU シェアが DSP シェアを隠している。

同じ 33 本を分割せずに DSP だけで実行した時間は、Phase 50 の profile
(`c50/prof-A-ngl9.txt`、skel A。IQ3_S の帯域は A と D で同一) の usec 合計から
1 step あたり **33,684 us**。

つまり分割は 33.7 ms あった DSP 区間を 17.1 ms に縮められる立場にありながら、
実際には token を 634 -> 762 ms へ **+128 ms 遅くしている**。
差のほぼ全部が 5 節の prep 170 ms である。

## 6. Phase 47 の記録の再集計 (fast cap では分割が速い)

`p51_p47_pairs.py` で Phase 47 の ABBA ログを再集計した。cap が 2 run の間で
動いた block と、承認切れ (rc 2、経過 0.1-0.2 秒) は除外している。

| tier | 有効 pair | 中央値 async/off | 読み方 |
|---|---|---|---|
| fast 3628800 | 17 | **0.8959** | 分割が 10.4% 速い |
| capped 2112000 | 1 | 0.9806 | 分割が 2% 速い |

条件は NGL=9、33 本、frac 0.9、5 step。Phase 51 の指示書にある
「Phase 49 では分割が 11-12% 遅かった」という数字に対応する記録は、
この repo のどのブランチにも残っていない (`c47`/`c48`/`c49` の主報告書は
git に存在しない)。一方、Phase 47 の実測は fast cap で明確に速い。

矛盾を説明できる仮説は 5 節のとおりで、当時の分割は
「CPU シェアの分だけ DSP 区間が縮む + キャッシュ無効の 170 ms」の差であった。
fast cap では CPU が速いので前者が勝ち、capped では負ける。

## 7. 発見: async 分割は再現性がない (今回の主要な不具合)

34 本目の話から離れて、分割経路そのものを測っているときに見つかった。

同じバイナリ、同じ環境、同じ replay 4 step で、`SPLIT=1 async` の logits が
run ごとに変わる。

| lib | mode | frac | logits sha256 (先頭 16) | prep_us (step 1-4) |
|---|---|---|---|---|
| Phase 50 配備 (11-09 02:02) | async | 0.25 | 788611da9d15a5fb | 170801 / 163414 (分割 0.5 の run) |
| c51 c51base (変更なし) | async | 0.25 | 9456dc7458643944 | 207249 / 178654 / 204150 |
| 同上、同じ条件で再実行 | async | 0.25 | 019160106423dc12 | - |
| c51 seq | seq | 0.25 | 2d4693c43c0cc307 | - |
| 同上、同じ条件で再実行 | seq | 0.25 | **2d4693c43c0cc307 (一致)** | - |

`seq` は 2 回とも同一ハッシュ。`async` は同じバイナリでも一致しない。
step 0 の logits はどの run でも完全一致し、step 1 以降が全要素で変わる
(max |d| 3.5-4.8、argmax は 4 step 中 3 つ一致、top10 は 6-10 個一致)。

### 7.1 原因

`hex_split_run` の async 分岐は

1. DSP シェアを submit (flush_async)
2. **待たずに** CPU シェアを実行
3. flush_sync で待つ

の順だった。CPU シェアは `plan->x_cpu->data = x->data` として
**デバイスメモリ上の入力 x を直接読む**。x を書いたのは直前に投入された
DSP op であり、2 の時点でそれが完了している保証は無い。さらに
DSP が書いた領域を host が読むためのキャッシュ整合も、待たずに読めば取れない。

`GGML_HEXAGON_SPLIT_NOSYNC=1` も同じで、CPU シェアを先に走らせてから
submit するため、入力の producer がまだ飛んでいる。

単一テンソルの合成ハーネス (Phase 46 の `c46_split_repeat`) が
`split-async.bin` と `split-seq.bin` の一致を確認できていたのは、
その試験の入力 x がロード時に用意した固定データで、DSP の producer が
無かったためである。実モデルでは入力が DSP の出力なので、この危険が表に出る。

### 7.2 修正

`hex_split_run` の先頭で、この op より前に投入された op を join してから
submit し、CPU シェアを DSP シェアと重ねる。

```cpp
    // The host share reads its input from device memory, and the op that writes
    // that input may still be in flight. Join the ops submitted before this one
    // first: the DSP runs its queue in order, so they own every input of this op.
    tw0 = hex_now_us();
    sess->flush_sync();
    tw1 = hex_now_us();
```

DSP はキュー順に実行するので、先行 op を待てばこの op の入力は確定する。
この op 自身の DSP シェアは待たないため、CPU シェアとの重なりは残る。
診断時と `seq` のときだけ、CPU シェアの後にも待つ。

### 7.3 合格条件 (この修正に対して)

1. `async` を 2 回実行して logits が一致する (再現性)。
2. `async` の logits が `seq` の logits と**ビット一致**する。
   行の割り当てと計算が同じで、違うのは順序だけだからである。
3. `wait_us == 0` の op が残り、`op_us` が `cpu_us` に近い (重なりが残っている)。
4. 速度が `seq` より速い。

## 8. 修正の検証結果 (2026-11-10 20:30 時点)

修正は 2 つ、別コミットにした。

| commit | 内容 |
|---|---|
| bb2607d4f | `hex_split_run` で、この op より前の DSP op を join してから submit する |
| 302685f2d | 分割時も HTP の node cache を使う (1 token あたり 163-171 ms の再構築を消す) |

### 8.1 数値 (NGL=9、33 テンソル、frac 0.25、4 step replay)

| 構成 | lib | logits sha256 (先頭 16) |
|---|---|---|
| SPLIT=0 | c51 両修正 | 5585d4edc47ee750 (Phase 50 の ngl=9 分割なし基準と一致) |
| SPLIT=0 (再実行) | 同 | 5585d4edc47ee750 |
| async | 同 | 2d4693c43c0cc307 |
| async (再実行) | 同 | 2d4693c43c0cc307 |
| seq (2 回) | 修正前 lib | 2d4693c43c0cc307 |

**async は再現し、かつ seq とビット一致した。** 修正前の async は
同一バイナリでも run ごとに違う値 (9456dc74 / 01916010) を出していた。

### 8.2 prep_us の変化

NGL=9、33 テンソル、frac 0.25 の timeline (1 step あたり):

| lib | prep_us |
|---|---|
| 修正前 | 207249 / 178654 / 204150 |
| cache 修正のみ | 39026 / 41011 |
| cache + async 修正 | 46075 / 61113 (batches 62) |

cache 修正で prep が約 1/4 になった。async 修正で batches が 34 -> 62 に増え、
pre-wait の分だけ prep/wait の配分が動く。

### 8.3 分割の費用 (SPLIT_TRACE、1 step あたりの最小値の和)

| 構成 | cpu_us | wait_us | op_us |
|---|---|---|---|
| async frac 0.5 (修正前) | 55367 | 1614 | 62498 |
| async frac 0.25 (修正前) | 33723 | 84 | 49302 |
| async frac 0.125 (修正前) | 22894 | 5082 | 49834 |
| async frac 0.25 (両修正) | 32879 | 17828 | 34345 |

DSP だけで同じ 33 本を実行した時間は 33,684 us/step (profile 実測)。
**両修正後の op_us 34,345 us は、分割しない場合とほぼ同じ**。
分割で得られるはずの重なりは、ほぼ pre-wait に食われている。

### 8.4 上限の見積り

NGL=9 の 1 token は約 660-700 ms。そのうち HTP split の DSP 区間は 61 ms。
分割が触れるのはこの 61 ms だけで、しかも型ごとの CPU/DSP 比から決まる
調和平均までしか縮まない。

- 33 本の DSP 合計 D = 33.7 ms
- 同じ 33 本の CPU 合計 C = 118 ms (trace の cpu_us から)
- 最適点での op 合計 = D*C/(D+C) = 26.2 ms

したがって分割の上限は 33.7 - 26.2 = **7.5 ms/token (約 1.1%)**。
残りの DSP 区間 27 ms は分割できない (層の依存関係)。

**行分割では 5% に届かない。** 5% を出すには DSP 区間 61 ms のほぼ全部を
CPU 区間 (約 570 ms) と重ねる必要があり、それは層の依存関係上できない。
Phase 49 が記録した「HTP と CPU の往復 1 回 7-12 ms」を考えると、
層を交互に配置して重なりを作る案は、往復の固定費で利得を食い潰す。

## 9. 第 2 ラウンド: 34 本目の行ごとの数値検証 (Gate 2) PASS

2026-11-10 20:49 の run (`runs/20261110-204919-763`) は、34 本の分割リスト
(`split34.txt`)、`GGML_HEXAGON_SPLIT_CPU_ROWS=32`、NGL=16、4 step replay、
`P51_DUMP=d34.bin` で実行した。exit 0、decode 4 step 完走、DSP エラーなし。

診断モードは出力を `-12345` で汚してから両シェアに書かせる。したがって
残った `-12345` は「書かれなかった行」、範囲違いは「prefix 以外の差分」として出る。

### 9.1 結果

| 検査 | 結果 |
|---|---|
| 分割の捕捉 | 34 本すべて (`SPLIT_COUNT 34`)。`blk.63.attn_output.weight` は `readback ndiff 0`、alias 有効 |
| CPU 担当行の参照一致 | `split verify ... cpu_vs_cpuref same nonbit 0 max 0.000000` が **136/136** (34 本 x 4 step) |
| DSP 担当行の参照一致 | dump と ref を record ごとに比較し、**CPU シェアの外で差分行 0** (136 record すべて) |
| CPU シェアの行数 | 全 record で先頭 32 行のみ差分 = 設定した 32 行と一致 |
| センチネル | **0 行** (全行が書かれている) |
| 非有限値 | **0** |
| 異常終了 / DSP 異常 | なし |
| logits ハッシュ | `4d4c311786f735f8f2eb874c3735bc05` (修正前 lib、NGL=16) |

`d34.bin` = `94777f61e9dc4606a14db41bd6592bd81cb096fb0f59bd910d3456d784d8f38d`
`d34.bin.ref` = `40bf44dff3e115b8efd2d7ab8ffb7af93e3e0dd488b49042ae0bc76a59b9697c`
(各 5,195,360 B = 136 record)

DSP 側の行が full DSP 参照とビット一致したことは、
`blk.63.attn_output.weight` の alias された tiled 領域からの行読みが
行番号のずれも端数も無いことを意味する。34 本目に固有の境界処理の問題は無い。

### 9.2 Gate A の判定材料

- 34 本構成は**正常に動作する**。行ごとの数値も参照と一致する。
- 過去に「34 本目で異常終了」とされた記録のうち、実物が残っているのは
  DMA64=1 の `dspqueue_read failed: 0x0000002e` だけで、これは 1 テンソルの
  試験でも再現し、テンソル本数に依存しない。
- 分割経路には本数と無関係な実害 (async の入力待ち漏れ) があり、それは修正した。

したがって「34 本目固有の不具合」は**現行ツリーには存在しない**。
ただし「過去に別のツリーで起きた異常終了を再現した」わけではないので、
判定は PARTIAL とする (下記 12 節)。

## 10. 第 1 ラウンドの到達点と、次にやること

### 10.1 できたこと

1. Gate 0 の棚卸し: 34 本目の異常終了とされてきた記録を端末の一次資料まで遡った。
   唯一の abort は DMA64=1 の `dspqueue_read failed: 0x2e`。747 個の run ディレクトリに
   abort は 1 件も無い。34 本目は Phase 47 の時点で実際には分割されていなかった。
2. Gate 1: NGL=16 で 34 本目を実際に分割した。捕捉され、`readback ndiff 0`、alias 有効。
   非同期で decode まで到達し、異常終了しない。
   frac 0.5 で止まったのは host share 401 MiB による swap-out ガード。
3. 分割経路の実害を 2 つ見つけて直した。
   - async が入力を待たずに読み、run ごとに違う logits を出していた (再現性ゼロ)。
     修正後は再現し、`seq` とビット一致する。
   - 分割時に node cache を捨てており、1 token あたり 163-171 ms を再構築に使っていた。
     修正後は 39-61 ms。
4. 分割の利得の上限を実測から見積もった: 7.5 ms/token (約 1.1%、NGL=9)。
   op_us は 34.3 ms で、分割しない場合の DSP 33.7 ms とほぼ同じ。重なりは pre-wait に
   食われている。

### 10.2 未了

- 34 本目の**行ごとの数値検証** (SPLIT_DUMP / SPLIT_DUMP_REF)。-> 第 2 ラウンドで完了 (9 節)。
- 27B 全体の性能比較 (off / async / seq) の有効な pair。今ラウンドの後半は
  端末がサーマルと swap で崩れ、6 run 中 5 run がガード停止したため取得できていない。
- pre-wait を「入力の producer だけ待つ」形にすれば 17.8 ms を削れる可能性がある。
  ただし上限は 1.1% のまま。

### 10.3 端末の状態 (2026-11-10 20:47)

- cap 1555200/1382400 (vendor が強い制限側に移った)、tz0 64.2 C、load 24.25、
  MemAvailable 9.9 GB、SwapFree 10.8 GB、cooling_device 全 0、残留プロセスなし。
- 1 時間で 30 回近く 9.26 GB のモデルを読み込んだため、page cache と ZRAM が
  飽和している。しばらく放置してから計測を再開する。
- 画面は 30 秒で消灯、`stayon` は false に戻した。

## 11. 再現コマンド

```
SKEL=288e6712df9c8201ef05c12ec5704968d4f71407992cff8e0f03aa002ee0861d

# 分割なしの基準 (NGL=16 / NGL=9、4 step replay)
adb shell "cd /data/local/tmp/c46-27b && P51_MODE=off WANT_SKEL=$SKEL sh p51_run.sh 4 replay 16 base16 1"
adb shell "cd /data/local/tmp/c46-27b && P51_MODE=off WANT_SKEL=$SKEL sh p51_run.sh 4 replay 9  base9 1"

# 34 本目を含む分割 (host share を小さくしてガードに触れない)
adb shell "cd /data/local/tmp/c46-27b && SPLIT_WEIGHTS=\$(cat libs/split34.txt) \
  GGML_HEXAGON_SPLIT_CPU_ROWS=32 P51_MODE=async WANT_SKEL=$SKEL sh p51_run.sh 4 replay 16 s34 1"

# 分割の内訳
adb shell "cd /data/local/tmp/c46-27b && SPLIT_WEIGHTS=\$(cat libs/split33.txt) TRACE=1 TIMELINE=1 \
  GGML_HEXAGON_SPLIT_CPU_FRAC=0.25 P51_MODE=async WANT_SKEL=$SKEL sh p51_run.sh 4 replay 9 t1 1"
python p51_split_trace.py <stderr.txt>
python p51_frac_model.py <trace.stderr> <profile.stderr> 0.25

# 回文系列 (cap が動いた pair は捨てる)
adb shell "cd /data/local/tmp/c46-27b && SPLIT_WEIGHTS=\$(cat libs/split33.txt) WANT_SKEL=$SKEL \
  sh p51_series.sh 9 4 replay off:0 async:0.25 async:0.25 off:0"
```

`WANT_SKEL` を省くと run50.sh が Phase 49 の既定 skel を要求して `REFUSING` になる。
`SPLIT_WEIGHTS` を省くと 1 テンソルだけの分割になる。

## 12. 未解決事項

- 34 本目を分割したときの**行ごとの数値検証** (CPU 行 = CPU 参照、DSP 行 = 全 DSP 参照、
  センチネル 0) は未実施。SPLIT_DUMP / SPLIT_DUMP_REF で取る。
- `0x2e` の意味は AEE_EBADPARM と読めるが、定数の定義はこの repo に無い。
  DMA64=1 のときにだけ出る。Phase 51 の分割経路とは独立。
- キャッシュを分割時にも有効化できるかは未検証 (5 節)。実装と検証は次段。
- frac 0.5 は型ごとの CPU/DSP 比と無関係な固定値である。型別の比は
  IQ3_S 6.4x / IQ3_XXS 3.0x / IQ2_S 1.9x / IQ2_XXS 0.7x (Phase 50 実測)。


## 13. 第 3 ラウンド: モデル不要の再現試験と、報告書

### 13.1 追加したもの

- `c51_split_repro.cpp`: 34 本の実名・実型・実形状を HTP に作り、入力の前に
  SCALE を置いて「同じバッチの DSP op が入力を書く」実モデルと同じ依存を作る。
  モデルも 9 GB の読み込みも不要で 1 回 2 秒。
- `phase51-small-repro.md`: その結果。修正前 lib は 1 回目だけ誤り
  (`64b0e335` -> `a638f0e6` x2)、修正後は 1 回目から seq とビット一致。
  診断 (`--dump`) を入れると修正前でも一致してしまう (診断が先行 op を join するため)
  ことも記録した。
- `phase51-report.md`: 報告書 (1-14 節)。8 節だけ未測定の項目を明記している。

### 13.2 実形状 (805.82 MiB) でも確認

`--replicate` (1 行だけ量子化してコピー) で実形状を数秒で回せる。
async 2 回、seq 2 回、いずれも `7305aed116d7cd83` で一致した。

### 13.3 27B の性能比較を実施しなかった理由

2026-11-12 23:43 時点で、端末は画面点灯 (`mWakefulness=Awake`)、Morphe が
799 MB を保持、バッテリー 24%、**AC 非接続**。この状態で 9.26 GB を読み込む
27B run を回すと、利用中のアプリを追い出して数分間端末が使えなくなり、
バッテリーも数 % 減る。夜間に無人のまま回すと、充電できない状態で残量が
尽きて端末が落ちる危険がある。

したがって、性能比較は次の条件が揃ったときに行う:
画面消灯かつアイドル、MemAvailable 9.5 GB 以上、AC 接続または十分な残量、
cap が 1 つの tier で安定。

### 13.4 現時点の判定

- A: PARTIAL (34 本目固有の不具合は無いと実測で示したが、過去の異常終了自体は未再現)
- B: PARTIAL (分割の費用を 1 token あたり 163-171 ms 削り、非同期の数値不整合を
  直した。27B 全体での優位は未確認。上限は約 1.1%)