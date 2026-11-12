# Phase 51 小さな再現試験 (モデル不要)

27B を動かさずに、34 テンソル構成の行分割を再現・検証する試験を作った。
`c51_split_repro.cpp` (約 260 行) を NDK でクロスコンパイルし、端末で動かす。

## 1. 何を作ったか

- 分割リスト 34 本の**実名・実型・実形状** (層 56-63、IQ3_S / IQ3_XXS) を、
  そのまま HTP の weights バッファに作る。
- 各テンソルは `ggml_quantize_chunk` で決定的な疑似乱数から量子化して書き込む。
  書き込みは `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` のバッファに対して行うので、
  `hex_split_capture` がモデルと同じ経路で走る。
- グラフは 1 列 MUL_MAT を 34 本。出力はデバイスメモリ (分割が末尾行を書ける条件)。
- **入力の前に SCALE を 1 本置く** (`--chain`、既定で有効)。これで分割の入力が
  「同じバッチの DSP op が書くテンソル」になり、実モデルと同じ依存関係になる。
- `--repeats T` で同じグラフを T 回計算し、毎回の出力の FNV-1a ハッシュを出す。
  再現性はこのハッシュの一致で判定する。
- `--dump P` で `GGML_HEXAGON_SPLIT_DUMP` を有効にし、行ごとの検証もできる。

モデルも 9 GB の読み込みも不要。1 回 2 秒程度。

## 2. 結果: 非同期の不具合がモデル無しで再現した

`--small` (m=512)、34 テンソル、3 repeat。

### 2.1 修正前の lib (`5c55b585`)

```
$ ./c51_split_repro --mode async --small --repeats 3
graph nodes 68
REPEAT 0 status 0 hash 64b0e3350d90e0d1 nonfinite 0
REPEAT 1 status 0 hash a638f0e61f5f1269 nonfinite 0
REPEAT 2 status 0 hash a638f0e61f5f1269 nonfinite 0

$ ./c51_split_repro --mode seq --small --repeats 3
REPEAT 0 status 0 hash a638f0e61f5f1269 nonfinite 0
REPEAT 1 status 0 hash a638f0e61f5f1269 nonfinite 0
REPEAT 2 status 0 hash a638f0e61f5f1269 nonfinite 0
```

**1 回目だけ値が違う。** 分割の CPU シェアが、直前の SCALE がまだ書いていない
入力バッファを読んだためである。2 回目以降は 1 回目が書いた正しい値が見えるので
一致する。つまり「最初の 1 token だけ静かに間違う」という形の不具合である。

### 2.2 修正後の lib (`c646b4db`)

```
$ ./c51_split_repro --mode async --small --repeats 3
REPEAT 0..2  hash a638f0e61f5f1269 (3 回とも同じ)

$ ./c51_split_repro --mode seq --small --repeats 3
REPEAT 0..2  hash a638f0e61f5f1269 (async と同一)

$ ./c51_split_repro --mode off --small --repeats 2
REPEAT 0..1  hash d4e8f2258e7c13a8 (分割なし。行の担当が違うので別値で当然)
```

**async が 1 回目から seq とビット一致。** 3 回とも同一。

### 2.3 行ごとの検証 (モデル不要版)

```
$ rm -f h34f.bin*
$ ./c51_split_repro --mode async --small --repeats 2 --dump /data/local/tmp/c46-27b/h34f.bin
$ python p51_dump_cmp.py h34f.bin h34f.bin.ref
dump 68 records 140080 B, ref 68 records 140080 B
records with a leading CPU-share difference: 68, row counts [128]
rows differing outside the CPU share: 0
sentinel rows total 0, nonfinite values total 0
PASS
```

68 record = 34 テンソル x 2 回。CPU シェアの 128 行だけが full DSP 参照と違い、
それ以外の行は完全一致、未書き込み行 0、非有限値 0。

## 3. 実形状でも同じ (805.82 MiB)

`--replicate` は 1 行だけ量子化して残りの行にコピーする。実形状
(`m = 5120 / 10240 / 17408`) でも数秒で回せるようにするためである。
行の中身が同じなので行番号のずれは検出できない。メモリ経路と境界の確認用。

```
$ ./c51_split_repro --mode async --replicate --repeats 2
weights 34, 805.82 MiB set
graph nodes 68
REPEAT 0 status 0 hash 7305aed116d7cd83 nonfinite 0
REPEAT 1 status 0 hash 7305aed116d7cd83 nonfinite 0

$ ./c51_split_repro --mode seq --replicate --repeats 2
REPEAT 0 status 0 hash 7305aed116d7cd83 nonfinite 0
REPEAT 1 status 0 hash 7305aed116d7cd83 nonfinite 0
```

実形状 805.82 MiB でも async は 1 回目から seq とビット一致する。
行番号の検証は 2.3 の実モデル dump (136 record、CPU シェアの外で差分行 0) が担う。

## 4. 診断を入れると不具合が消える (重要な注意)

同じ修正前 lib で `--dump` を付けると、2 回のハッシュが**どちらも** `a638f0e6` に
なって再現しなくなる。

理由は `hex_split_run` の診断経路が、分割の前に元のノードを DSP で 1 回実行して
`flush_sync` するためである。この待ちが先行 op をすべて回収してしまい、
CPU シェアが読む入力が確定してしまう。

つまり **診断ありの試験だけを見て「正常」と判断してはいけない**。
Phase 51 の指示書 5.1 の「診断処理自体が競合や同期関係を変える可能性があるため、
診断あり・なしの両方を確認してください」は、まさにこの現象である。

## 5. 副次的に観測したこと

- 修正前 lib で `--n 1 --repeats 1` を実行すると、
  `FORTIFY: pthread_mutex_lock called on a destroyed mutex` で落ちる。
  同じ条件が修正後 lib では正常終了する (hash d073d873...)。
  終了処理の順序に依存した別の症状と思われるが、深追いしていない。

## 6. 再現コマンド

```
# ビルド (ホスト)
NDK=C:/Users/yuubi/AppData/Local/Android/Sdk/ndk/29.0.14206865/toolchains/llvm/prebuilt/windows-x86_64/bin
$NDK/clang++.exe --target=aarch64-linux-android28 -O2 -std=c++17 -static-libstdc++ -fPIC \
  -I <repo>/ggml/include c51_split_repro.cpp -o c51_split_repro \
  -L <repo>/build-c51/bin -lggml -lggml-base -lggml-cpu
adb push c51_split_repro /data/local/tmp/c46-27b/

# 実行 (端末)
cd /data/local/tmp/c46-27b && chmod 755 c51_split_repro
LD_LIBRARY_PATH=libs ADSP_LIBRARY_PATH=libs ./c51_split_repro --mode async --small --repeats 3
LD_LIBRARY_PATH=libs ADSP_LIBRARY_PATH=libs ./c51_split_repro --mode seq   --small --repeats 3

# lib の入れ替え (A/B)
cp libs/libggml-hexagon.so.c51base libs/libggml-hexagon.so   # 修正前
cp libs/libggml-hexagon.so.c51head libs/libggml-hexagon.so   # 修正後

# 実形状で回す (任意、802 MiB の重み)
LD_LIBRARY_PATH=libs ADSP_LIBRARY_PATH=libs ./c51_split_repro --mode async --repeats 3
```

## 7. この試験の位置づけ

指示書 4.2 の段階に当てはめると、A (単独テンソル) から D (34 テンソル) までを
1 つの試験で通り、E (逐次と非同期の比較) を同じ条件で行える。
F (実モデルへの統合) は別途、27B の 4 step replay で確認済み (logits が
`seq` と `async` でビット一致)。

縮小構成で再現したからといって実モデルの問題が解決した証拠にはならない、という
指示書 4.3 の注意に対しては、実モデル側でも同じ結論 (async == seq) を
得ているので、両方が揃っている。
