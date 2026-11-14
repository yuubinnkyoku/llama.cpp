# Phase 52 — 生成品質評価と CPU 単独比

Phase 51 は §20 でクローズした。Phase 52 は新規の性能最適化を行わず、
「実用的な高速生成」の裏付けに残る 2 項目を完了することを目的とする。

- 目標 1: 生成品質の長めの評価 (32-step 以上)
- 目標 2: fast cap での CPU 単独比 (機会があれば)

## 1. 現在地と測定条件

### 1.1 開始時の状態

| 項目 | 値 |
| --- | --- |
| ブランチ | `codex/c52-quality` (`codex/c51-perf` の `2ab746abc` から分岐) |
| 基準 HEAD | `2ab746abc` (指示の想定 `712b80314` より新しい。§19.11 の追記を含む) |
| 端末 | USB `324753221196` と TCP `192.168.55.15:5555` の両方が同時に見えていた |
| skel | `288e6712df9c8201ef05c12ec5704968d4f71407992cff8e0f03aa002ee0861d` (Phase 50 D) |
| libggml-hexagon.so | `44baedd81e5d7498b89430c2b9a60c612a0f14903be74de267780ab9b2c8c7e0` |
| 再起動 | していない |

### 1.2 評価条件

Phase 50 の品質評価と同一条件を選んだ。

| 項目 | 値 |
| --- | --- |
| モデル | `Qwen3.8-27B-GSQ-RCO-IQ2_S.gguf` (9,259,510,912 bytes) |
| 評価文 | `/data/local/tmp/c50/p50-eval-a.txt` (llama.cpp README 冒頭、Phase 50 と同一) |
| トークン列 | 同ファイルを `llama-tokenize` で符号化した先頭 32 個 |
| CPU 側 | NGL=0 (全層 CPU) |
| HTP 側 | NGL=16、SPLIT=0 (Phase 50 の品質評価と同一) |
| 文脈 | 生テキスト (chat template なし) |
| 比較方法 | 同一の強制トークン列に対する per-step NLL の paired 比較 |

トークン列 (32 個):

```
760 1822 1918 314 411 2313 369 279 1510 636 2933 63 6522 13 11116 351
10970 3622 628 381 1669 303 498 965 14 636 2933 834 9251 1202 965 14
```

先頭 16 個は Phase 50 が使った列と一致する (同じ評価文の同じ位置)。
残り 16 個は同じ文の続きで、Phase 52 で延長した部分である。

### 1.3 実行環境の実測値

| 時刻 | MemAvailable | SwapFree | cap0 / cap7 | tz0 | load |
| --- | ---: | ---: | --- | ---: | ---: |
| 19:14 (CPU 32 試行 1) | 10,726,284 kB | 11,276,096 kB | 3628800 / 3648000 | 29.2 ℃ | 5.2 |
| 19:14 (CPU 32 成功) | 11,052,724 kB | 10,758,276 kB | 3628800 / 3648000 | 29.2 ℃ | 6.2 |
| 19:16 (HTP 32 成功) | 11,295,180 kB | 10,743,524 kB | 3187200 / 3648000 | — | 7.0 |
| 19:18 (CPU 32 再現) | 11,414,796 kB | 10,604,852 kB | 3187200 / 3648000 | — | 6.6 |
| 19:19 (HTP 32 再現) | — | — | 2496000 / 2668800 | — | — |

**tier は run 間で一定ではない。** しかし本節の比較は logits の数値比較であり、
速度比較ではないため、tier の差は結果に影響しない。数値の再現性は
各側 2 回の hash 一致で確認した (1.5)。

### 1.4 ガード発動の記録

| 時刻 | run | 内容 |
| --- | --- | --- |
| 19:14 | CPU 32 試行 1 | prefill 中に swap guard (310,838 pages ≈ 1214 MiB) |
| 19:15 | HTP 32 試行 1 | prefill 中に swap guard (380,085 pages ≈ 1485 MiB) |

いずれも**モデルのロード/prefill 中**に発生し、分割や NGL の設定とは無関係。
ガードは 1024 MiB のまま**一切緩めていない**。メモリを空けて再試行し、
2 回目で両側とも完走した。

### 1.5 再現確認 (各側 2 回)

| 側 | run | logits sha256 |
| --- | --- | --- |
| CPU (NGL=0) | 1 回目 | `44cd99ce376d5ce627ebbd4dc1c1d6cb90e83ea7822371f13a4874f921262295` |
| CPU (NGL=0) | 2 回目 | `44cd99ce376d5ce627ebbd4dc1c1d6cb90e83ea7822371f13a4874f921262295` |
| HTP (NGL=16) | 1 回目 | `b21ca4bf1da8a2ef3bc48f717ee79f13acc17ced0783141211fc7178f0c2e160` |
| HTP (NGL=16) | 2 回目 | `b21ca4bf1da8a2ef3bc48f717ee79f13acc17ced0783141211fc7178f0c2e160` |

**両側とも 2 回で完全一致。** 32 step × 248,320 vocab = 7,946,240 float が
ビット単位で再現している。

## 2. 目標 1: 32-step の NLL 評価

### 2.1 結果

| 量 | 値 |
| --- | ---: |
| 平均 NLL CPU (NGL=0) | 3.132049 |
| 平均 NLL HTP (NGL=16) | 3.125820 |
| **平均差 (HTP − CPU)** | **−0.006229 nats** |
| 標準誤差 | 0.013226 |
| t (dof=31) | **−0.47** |
| PPL CPU | 22.9209 |
| PPL HTP | 22.7786 |
| **argmax 一致** | **32/32** |
| greedy == forced (CPU) | 11/32 |
| greedy == forced (HTP) | 11/32 |

**NLL の差は −0.0062 ± 0.0132 nats (t = −0.47) で、ゼロと区別できない。**
argmax は 32 step すべてで一致した。

### 2.2 logits の距離

| 量 | 最小 | 中央値 | 最大 |
| --- | ---: | ---: | ---: |
| 相対 L2 差 | 1.677 % | 6.400 % | **11.768 %** |
| 最大絶対差 | 0.3171 | 1.1284 | **2.8008** |
| 全変動距離 (TV) | 0.015 % | 2.880 % | 5.697 % |
| KL (CPU‖HTP) | 4.799e-05 | 3.095e-03 | 1.155e-02 |

**logits は最大 11.8 % (相対 L2)、絶対 2.80 までずれる。** それでも
argmax は 32/32 一致し、NLL 差は検出できない。Phase 50 の n=16 の値
(相対 L2 最大 9.60 %、最大絶対差 2.80) と同じ範囲で、step を倍にしても
傾向は変わらなかった。

### 2.3 読み方

- 誤差があることと品質が落ちることは別、という Phase 50 の結論は
  n=32 でも同じ形で再現した。
- ただし **n=32、評価文 1 本、生テキスト**である。この範囲で
  「差が検出できない」までで、**品質が同等という証明ではない。**
- 品質の合否判断は所有者に委ねる。本報告は差の実測のみを提示する。

## 3. 目標 1 の追加試行: 64-step は probe の上限で実行不能

指示は「可能なら 64-step も」だったので試した。**実行できなかった。**

### 3.1 症状

`decode-probe-wait` に 64 step を渡すと、**何も出力せず 2.5 秒で exit 2** になる。
stdout も stderr も空で、`OVERRIDE` も `PHASE load_begin` も出ない。
Phase 50 が記録した「64 step は probe が起動していない」と同じ症状である。

### 3.2 境界の特定 (本節の新しい情報)

step 数を変えて直接叩き、境界を確定した。

| steps | rc | 備考 |
| ---: | ---: | --- |
| 32 | 0 (または 3) | 完走する。3 は override ファイル指定ミスによるもので、probe は起動している |
| 33 | **2** | 出力なしで即終了 |
| 34 | **2** | 同上 |
| 36 | **2** | 同上 |
| 38 | **2** | 同上 |
| 40 | **2** | 同上 |
| 48 | **2** | 同上 |
| 56 | **2** | 同上 |
| 64 | **2** | 同上 |

**境界は 32 と 33 の間**で、replay トークン数を 33 個に揃えても変わらない。
したがって「トークン列が足りない」ことが原因ではなく、
**probe 側の step 上限が 32** である。

### 3.3 結論

**64-step は probe の実装上限により実行不能。** これは環境 (メモリ・熱) の
制約ではなく、測定ツールの制約である。64-step を行うには probe の
再ビルドが必要で、本指示の範囲 (新規の性能最適化を行わない) を超えるため
実施していない。**32-step が本 Phase で到達できる上限**である。

## 4. 目標 2: fast cap での CPU 単独比

### 4.1 結果: BLOCKED (窓が取れず)

本 Phase の測定中、fast cap (`3628800`) の窓は観測していない。
19:14 の CPU 32-step 試行 1 の開始時点で cap0 が `3628800` だったが、
その run は swap guard で停止し、成功した run の時点では既に
`3187200` へ降下していた。

| 時刻 | cap0 | 状態 |
| --- | --- | --- |
| 19:14 | 3628800 | CPU 32 試行 1 開始。prefill 中に guard 停止 |
| 19:14 | 3628800 | CPU 32 成功 (この時点では fast) |
| 19:16 | 3187200 | HTP 32 成功。既に降下 |
| 19:18 | 3187200 | CPU 32 再現 |
| 19:19 | 2496000 | HTP 32 再現 |
| 19:20 以降 | 3187200 | 64-step 試行と境界特定の間 |

**同一 cap の CPU 単独 vs HTP の pair は 1 組も取れていない。**
§18/§19 と同じく、fast cap の窓は 1-6 分で閉じ、27B の 1 run (ロード含む
30-60 秒) の間に熱制限が入るためである。

指示の「窓が取れない場合は冷却や設定変更で無理に延ばさず、BLOCKED として
環境要因とともに記録する」に従い、**BLOCKED** とする。
温度保護・電源ポリシーへの介入は行っていない。

### 4.2 fast cap 窓での試行 (19:25、不成立)

報告をまとめている最中に 19:25 に fast cap (`3628800_3648000`、tz0 35.3 ℃、
MemAvailable 10.8 GB) の窓が開いたので、§19.11 の知見に従って
**熱を持ちやすい側 (NGL=0 の CPU 単独) を先に**回した。

| run | exit | cap の内訳 | 判定 |
| --- | ---: | --- | --- |
| CPU (NGL=0) 4-step | 0 | `3628800_3648000` x11、`3513600_3648000` x1 | **不成立** (1 sample 変化) |

- 完走はしたが cap0 が 1 sample だけ `3513600` へ落ちた。§15〜§19 と同一基準で
  **不成立**として除外する。
- この run の直後に cap は `3187200` へ降下し、窓は閉じた (tz0 42.6 ℃)。
  **HTP 側を測る前に窓が終わったため、pair は 0 組。**
- 順序の工夫 (熱を持つ側を先) は正しかったが、1 run 目で既に 1 sample 動いた。
  CPU 単独は 27B 全層を CPU で回すため発熱が大きく、fast cap を保てない。

**fast cap の CPU 単独比は依然未測定。** 不成立 run も記録として残す。

### 4.3 参考: 本 Phase で観測した tier と速度

速度は本 Phase の判定対象ではないが、記録として残す。

| 側 | tier | decode_ms (32 step) | tokens/s |
| --- | --- | ---: | ---: |
| CPU (NGL=0) | 3628800 | 639.097 | 1.565 |
| HTP (NGL=16) | 3187200 | 532.902 | 1.877 |

tier が異なるため**比較に使えない**。CPU 側が fast cap だったにもかかわらず
HTP 側より遅いが、これは NGL=0 (全層 CPU) と NGL=16 (16 層 HTP) の
構成差であり、tier の差ではない。**この 2 値を CPU 単独比として扱わない。**

## 5. 判定

| 目標 | 判定 | 根拠 |
| --- | --- | --- |
| 目標 1 (生成品質 32-step) | **PASS** | CPU/HTP 両側で 32-step 完走。NLL 差 −0.0062 ± 0.0132 nats (t=−0.47)、argmax 32/32、相対 L2 最大 11.77 %、最大絶対差 2.80。各側 2 回の hash 一致で再現確認済み |
| 目標 1 の 64-step | **実行不能** | probe の step 上限が 32。33 以上は出力なしで exit 2。ツールの制約であり環境要因ではない |
| 目標 2 (fast cap CPU 単独比) | **BLOCKED** | fast cap の窓を 2 回観測したが、CPU 単独 run が 1 sample で cap を外し、HTP 側を測る前に窓が閉じた。同一 cap の pair は 0 組。ガード・温度保護への介入は行っていない |

**品質が同等であるとは結論しない。** n=32・評価文 1 本・生テキストの範囲で
「差が検出できない」までを報告し、合否判断は所有者に委ねる。
根拠のない精度閾値は設定していない。

## 6. コミット

| commit | 内容 |
| --- | --- |
| (本節) | `docs(c52): evaluate 32-step generation quality and record the probe limit` |
| (本節) | `docs(c51): close Phase 51 with the final verdicts` |

ブランチ `codex/c52-quality`。`codex/c51-perf` の `2ab746abc` を基準にする。

## 7. 再現コマンド

### 7.1 32-step の NLL 評価

```sh
# 端末側: 32 トークン列を作る (評価文は Phase 50 と同一)
cd /data/local/tmp/c46-27b
cp libs/replay.txt libs/replay.txt.bak
echo "760 1822 1918 314 411 2313 369 279 1510 636 2933 63 6522 13 11116 351 \
10970 3622 628 381 1669 303 498 965 14 636 2933 834 9251 1202 965 14" > libs/replay.txt

export WANT_SKEL=288e6712df9c8201ef05c12ec5704968d4f71407992cff8e0f03aa002ee0861d
PROFILE_ARG=0 sh p50_run.sh 32 replay 0  nll32cpu 1   # CPU 単独
PROFILE_ARG=0 sh p50_run.sh 32 replay 16 nll32htp 1   # HTP NGL=16

cp libs/replay.txt.bak libs/replay.txt
```

```powershell
# ホスト側: NLL と距離の統計
adb pull /data/local/tmp/c46-27b/runs/<cpu-run>/logits.f32 build/p52/lg-cpu32.f32
adb pull /data/local/tmp/c46-27b/runs/<htp-run>/logits.f32 build/p52/lg-htp32.f32

python iq2s-device-test/phase20-autonomous-correctness-memory/continuations/c50/p50_nll.py `
  build/p52/lg-cpu32.f32 build/p52/lg-htp32.f32 `
  "760 1822 1918 314 411 2313 369 279 1510 636 2933 63 6522 13 11116 351 10970 3622 628 381 1669 303 498 965 14 636 2933 834 9251 1202 965 14" `
  32 248320

python iq2s-device-test/phase20-autonomous-correctness-memory/continuations/c52/p52_logits_dist.py `
  build/p52/lg-cpu32.f32 build/p52/lg-htp32.f32 32 248320
```

### 7.2 probe の step 上限の確認

```sh
cd /data/local/tmp/c46-27b
for n in 32 33 34 40 64; do
  LD_LIBRARY_PATH=libs ADSP_LIBRARY_PATH=libs P19_REPLAY_TOKENS=libs/replay.txt \
    ./libs/decode-probe-wait \
      /data/local/tmp/llama.cpp/models/Qwen3.8-27B-GSQ-RCO-IQ2_S.gguf \
      libs/override-base.txt $n /tmp/t.f32 mmap HTP0:0 0 >/dev/null 2>&1
  echo "steps=$n rc=$?"
done
# 32 は 0 (または 3)、33 以上は 2 で即終了する
```

## 8. 次の一手

1. **64-step を行うには probe の再ビルドが必要。** 32-step 上限の根拠を
   probe のソースで確認し、上限を外せるか判断する。本 Phase では
   範囲外として実施していない。
2. **fast cap の CPU 単独比は未測定のまま。** 窓が 1-6 分で閉じるため、
   1 組 (2 run) を同一 cap で完走させるには、窓が開いた直後に
   熱を持ちやすい側 (NGL=0 の CPU 単独) を先に回す順序が有望。
   §19.11 の知見 (最初の run が最も cap 安定) と整合する。
   ただし 19:25 の試行では CPU 単独 1 run で既に 1 sample 動いた。
   **CPU 単独は 27B 全層を回すため発熱が大きく、fast cap を保てない。**
   次に試すなら、CPU 単独を 4-step より短くするか、冷却を強めてから
   窓を待つ必要がある。
3. 品質評価は n=32・評価文 1 本・生テキストで完了した。より強い裏付けが
   必要なら、評価文を増やすか chat template を適用した条件で測る。
   ただしそれは本 Phase の範囲外。

## 9. probe の step 上限の特定と修正

### 9.1 原因

`decode-probe-wait` のソースは
`iq2s-device-test/phase20-autonomous-correctness-memory/tools/c27_decode_probe_wait.cpp`
にあり、step 数を**引数の検証で 32 に制限**していた。

```c
    const int steps = std::atoi(argv[3]);
    if (steps < 1 || steps > 32) {
        return 2;
    }
```

`return 2` が**何も出力しない**ため、§3.1 で観測した「stdout も stderr も空で
2.5 秒で exit 2」という症状になっていた。バッファサイズや静的配列ではなく、
単純な引数上限である。

`n_ctx` は 128、prompt は数トークンなので、64 step は元から収まる。
他に step 数依存の制約は無い (decode ループは `steps` をそのまま使う)。

### 9.2 変更内容 (最小)

上限だけを 32 → 64 に変えた。**1 行の定数変更のみ**で、decode の計算経路には
触れていない。

```diff
     const int steps = std::atoi(argv[3]);
-    if (steps < 1 || steps > 32) {
+    // Phase 52: was 32. n_ctx is 128 below and the prompt is a few tokens, so
+    // 64 steps fit. Raising the limit does not touch the decode loop, which the
+    // 32-step regression against the known hashes checks before 64 is used.
+    if (steps < 1 || steps > 64) {
         return 2;
     }
```

ソースは `continuations/c52/decode_probe_64.cpp` として保存した
(元ファイルはメイン worktree にあり、そちらは変更していない)。

### 9.3 ビルド

端末のライブラリと ABI を合わせるため、**端末から取り出した .so にリンク**した。

```powershell
# 端末から 4 本の .so を取り出す (libggml.so と libggml-cpu.so は symlink の先)
adb pull /data/local/tmp/c46-27b/libs/libllama.so      build/p52/devlibs/
adb pull /data/local/tmp/c46-27b/libs/libggml-base.so  build/p52/devlibs/
adb pull /data/local/tmp/c44-27b/libs/libggml.so       build/p52/devlibs/
adb pull /data/local/tmp/c44-27b/libs/libggml-cpu.so   build/p52/devlibs/

$NDK="C:/Users/yuubi/AppData/Local/Android/Sdk/ndk/29.0.14206865/toolchains/llvm/prebuilt/windows-x86_64/bin"
& "$NDK/clang++.exe" --target=aarch64-linux-android28 -O2 -std=c++17 `
  -static-libstdc++ -fPIC -I <repo>/include -I <repo>/ggml/include `
  decode_probe_64.cpp -o decode-probe-wait `
  -L build/p52/devlibs -lllama -lggml -lggml-base -lggml-cpu `
  -Wl,--allow-shlib-undefined
```

`--allow-shlib-undefined` は、端末の `libggml-cpu.so` が
`_Unwind_Resume@LIBC_R` を未解決のまま持っているため必要だった。
生成物の依存 .so 一覧は元の probe と同一 (`libc`, `libdl`, `libm`,
`libggml-base`, `libggml-cpu`, `libggml`, `libllama`)。

| 項目 | sha256 |
| --- | --- |
| 元の probe (バックアップ) | `aeaa1b6236d597cc04ad0b642074f451abb70d409b66625a2dd3083c9bd13f3f` |
| 再ビルドした probe | `ad15340c5c0447d949c867a4327c86118c8d162490e73179c385401e9145725d` |

### 9.4 32-step 回帰 (既知 hash との一致)

**probe 差し替えが計算結果に影響しないことの証明。** 再ビルド後、§1.5 と
同じ条件で 32-step を回した。

| 側 | 既知 hash (§1.5) | 再ビルド後 | 一致 |
| --- | --- | --- | --- |
| CPU (NGL=0) | `44cd99ce376d5ce627ebbd4dc1c1d6cb90e83ea7822371f13a4874f921262295` | `44cd99ce...` | **一致** |
| HTP (NGL=16) | `b21ca4bf1da8a2ef3bc48f717ee79f13acc17ced0783141211fc7178f0c2e160` | `b21ca4bf...` | **一致** |

**両側とも完全一致。** 上限変更は数値に影響していない。

### 9.5 ガード発動 (記録)

32-step 回帰の途中で 4 回、64-step の途中で 5 回、swap guard が発動した。
すべて**モデルのロード/prefill 中**で、ガードは 1024 MiB のまま緩めていない。

| 時刻 | run | delta_pages | 備考 |
| --- | --- | ---: | --- |
| 20:09 | 32-step CPU | 337,357 | 再試行で成功 |
| 20:15 | 32-step CPU | — | 再試行で成功 |
| 20:16 | 32-step CPU | 464,482 | 再試行で成功 |
| 20:21 | 32-step CPU | — | 再試行で成功 |
| 20:30 | 64-step CPU 再現 | 271,416 | 再試行で成功 |
| 20:38 | 64-step CPU 再現 | — | 再試行で成功 |
| 20:42 | 64-step HTP 再現 | 348,391 | **未成功** |
| 20:47 | 64-step HTP 再現 | 527,476 | **未成功** |
| 20:48 | 64-step HTP 再現 | 481,808 | **未成功** |
| 20:55 | 64-step HTP 再現 | 471,186 | **未成功** |

**失敗した 4 回はいずれも HTP 64-step の再現 run。** 成功した run の
AnonPages は約 1.4 GB、失敗した run は 2.0-2.4 GB だった。背景アプリ
(`ai.mistral.chat`、`com.google.android.gms`、キーボードの `zenz` など) が
force-stop 後も数秒で復帰し、AnonPages を押し上げるためである。

## 10. 目標 1 の 64-step 結果

### 10.1 完走した run

| 側 | run | exit | logits sha256 |
| --- | --- | ---: | --- |
| CPU (NGL=0) | 1 回目 | 0 | `de62b90c81521463075bbb0f7fe02d1b9d3fc9262799cd984e39a1e0d6021db4` |
| CPU (NGL=0) | 2 回目 | 0 | `de62b90c81521463075bbb0f7fe02d1b9d3fc9262799cd984e39a1e0d6021db4` |
| HTP (NGL=16) | 1 回目 | 0 | `d7b272a076bfb79aff543db9cf3042d5663a82a740dd7915228e4a073b53c627` |
| HTP (NGL=16) | 2 回目 | 0 | `d7b272a076bfb79aff543db9cf3042d5663a82a740dd7915228e4a073b53c627` |

**両側とも 2 回で完全一致。** 64 step × 248,320 vocab = 15,892,480 float が
ビット単位で再現している。HTP の 2 回目は §17 で取得した (下記)。

### 10.2 NLL 統計 (64 step)

| 量 | 値 |
| --- | ---: |
| 平均 NLL CPU (NGL=0) | 2.977521 |
| 平均 NLL HTP (NGL=16) | 2.964708 |
| **平均差 (HTP − CPU)** | **−0.012813 nats** |
| 標準誤差 | 0.016859 |
| t (dof=63) | **−0.76** |
| PPL CPU | 19.6391 |
| PPL HTP | 19.3890 |
| **argmax 一致** | **61/64** |
| greedy == forced (CPU) | 24/64 |
| greedy == forced (HTP) | 25/64 |

**NLL の差は −0.0128 ± 0.0169 nats (t = −0.76) で、ゼロと区別できない。**
ただし **argmax は 3 step で不一致**だった (32-step では 32/32 一致)。

### 10.3 argmax が不一致だった step

| step | 正解 | NLL CPU | NLL HTP | 差 | argmax CPU | argmax HTP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 38 | 2614 | 3.925153 | 3.872234 | −0.052919 | 1510 | **351** |
| 48 | 321 | 1.187739 | 1.012883 | −0.174856 | 11 | **321** |
| 58 | 68 | 2.797592 | 2.845907 | +0.048315 | 2186 | **34** |

- step 48 では HTP 側の argmax が**正解トークンと一致**しており、
  CPU 側が外している。差の方向は一様でない。
- 3 step とも NLL 差は小さい (−0.17 から +0.05)。argmax が入れ替わるのは
  上位 2 候補の logit が近い step である。

### 10.4 logits の距離 (64 step)

| 量 | 最小 | 中央値 | 最大 |
| --- | ---: | ---: | ---: |
| 相対 L2 差 | 1.677 % | 7.305 % | **49.517 %** |
| 最大絶対差 | 0.3171 | 1.6619 | **6.5177** |
| 全変動距離 (TV) | 0.004 % | 3.269 % | 11.597 % |
| KL (CPU‖HTP) | 6.773e-06 | 4.863e-03 | 4.407e-02 |

**最大 49.5 % (相対 L2) は step 34 の 1 点**で、その step の TV は 0.608 %、
KL は 6.75e-03 と小さい。logit 全体が平行移動した形で、分布の形は
ほとんど変わっていない。中央値 (7.3 %) は 32-step の 6.4 % と同程度。

### 10.5 32-step と 64-step の比較

| 量 | 32-step | 64-step |
| --- | ---: | ---: |
| 平均 NLL 差 | −0.006229 | −0.012813 |
| 標準誤差 | 0.013226 | 0.016859 |
| t | −0.47 | −0.76 |
| argmax 一致 | **32/32** | **61/64** |
| 相対 L2 中央値 | 6.400 % | 7.305 % |
| 相対 L2 最大 | 11.768 % | 49.517 % |
| 最大絶対差 最大 | 2.8008 | 6.5177 |

- NLL 差はどちらもゼロと区別できない。
- **argmax は 32-step では完全一致だったが、64-step では 3 件不一致。**
  step を延ばすと不一致が出る可能性があることを示す実測である。
- 相対 L2 の最大は 64-step で大きいが、これは 1 step の外れ値
  (step 34) による。中央値は同程度。

## 11. 目標 2: fast cap CPU 単独比のクローズ

**再挑戦せず BLOCKED (環境制約で測定不能) として確定する。**

### 11.1 実測の対比

同じ端末・同じ fast cap 条件で、2 つの構成の cap 保持に差があった。

| 構成 | 時刻 | fast cap の保持 | 出典 |
| --- | --- | --- | --- |
| CPU 単独 (NGL=0) 4-step | 19:25 | **1 run 目で cap0 が 1 sample 降下** (`3513600`) | §4.2 |
| HTP off (SPLIT=0) 4-step | 18:47 | **12/12 sample すべて保持** | §19.11 |

- CPU 単独は 27B の全層を CPU で回すため発熱が大きく、fast cap の窓
  (1-6 分) の中で 1 run すら cap を保てなかった。
- HTP off は同じ 4-step でも fast cap を 12 sample 保持した。
- **これは観測された対比であり、「HTP が持続性能で優位」といった結論では
  ない。** 測定条件 (時刻・熱履歴・背景負荷) が異なる 2 点の比較で、
  同一条件の対比較ではない。

### 11.2 判定

**BLOCKED。** 同一 cap の CPU 単独 vs HTP の pair は 0 組。
fast cap の窓が 1-6 分で閉じ、CPU 単独 1 run がその窓を保てないため、
この端末では測定不能という環境制約として確定する。
温度保護・電源ポリシーへの介入は行っていない。

## 12. Phase 52 最終判定

| 目標 | 判定 | 根拠 |
| --- | --- | --- |
| 目標 1: 32-step 品質 | **PASS** | 両側完走、NLL 差 −0.0062 ± 0.0132 (t=−0.47)、argmax 32/32、各側 2 回 hash 一致 (§2) |
| 目標 1: probe 修正 | **PASS** | 上限 32 → 64 の 1 行変更。32-step 回帰が両側とも既知 hash と一致 (§9.4) |
| 目標 1: 64-step 品質 | **PASS** | 両側完走、NLL 差 −0.0128 ± 0.0169 (t=−0.76)、argmax 61/64、**両側とも 2 回 hash 一致** (§10, §17) |
| 目標 2: fast cap CPU 単独比 | **BLOCKED** | 同一 cap の pair 0 組。CPU 単独が fast cap を 1 run も保てない (§11) |

**品質が同等であるとは結論しない。** 64-step では argmax が 3/64 で
不一致だった。差の実測を報告し、合否の判断は所有者に委ねる。
根拠のない精度閾値は設定していない。

## 13. コミット

| commit | 内容 |
| --- | --- |
| (本節) | `docs(c52): raise the probe step limit and record the 64-step quality` |

ブランチ `codex/c52-quality`。基準は `151751a3a`。

## 14. 再現コマンド

### 14.1 probe のビルドと差し替え

§9.3 のとおり。差し替え前のバックアップ:

```sh
adb shell 'cp /data/local/tmp/c45-27b/libs/decode-probe-wait \
  /data/local/tmp/c45-27b/libs/decode-probe-wait.p19bak'
```

### 14.2 32-step 回帰

```sh
cd /data/local/tmp/c46-27b
cp libs/replay32.txt libs/replay.txt
export WANT_SKEL=288e6712df9c8201ef05c12ec5704968d4f71407992cff8e0f03aa002ee0861d
PROFILE_ARG=0 sh p50_run.sh 32 replay 0  reg32cpu 1   # -> 44cd99ce...
PROFILE_ARG=0 sh p50_run.sh 32 replay 16 reg32htp 1   # -> b21ca4bf...
```

### 14.3 64-step

```sh
cd /data/local/tmp/c46-27b
cp libs/replay64.txt libs/replay.txt
export WANT_SKEL=288e6712df9c8201ef05c12ec5704968d4f71407992cff8e0f03aa002ee0861d
PROFILE_ARG=0 sh p50_run.sh 64 replay 0  nll64cpu 1   # -> de62b90c...
PROFILE_ARG=0 sh p50_run.sh 64 replay 16 nll64htp 1   # -> d7b272a0...
```

```powershell
# 統計
python .../c50/p50_nll.py build/p52/lg-cpu64.f32 build/p52/lg-htp64.f32 `
  "<64 tokens>" 64 248320
python .../c52/p52_logits_dist.py build/p52/lg-cpu64.f32 build/p52/lg-htp64.f32 64 248320
```

### 14.4 後片付け

```sh
adb shell 'cp /data/local/tmp/c45-27b/libs/decode-probe-wait.p19bak \
  /data/local/tmp/c45-27b/libs/decode-probe-wait'
adb shell 'chmod 755 /data/local/tmp/c45-27b/libs/decode-probe-wait'
adb shell 'cp /data/local/tmp/c46-27b/libs/replay.txt.p52bak \
  /data/local/tmp/c46-27b/libs/replay.txt'
```

## 15. 次の一手

1. **HTP 64-step の再現確認が残っている。** 4 回試行して 4 回とも
   swap guard で停止した。成功した run の AnonPages は約 1.4 GB、
   失敗した run は 2.0-2.4 GB で、背景アプリの復帰が原因。
   端末が静穏な時間帯 (深夜など) に 1 run 取れば埋まる。
2. **64-step で argmax が 3/64 不一致だった事実は記録として重要。**
   32-step では 32/32 一致だったので、step を延ばすと差が出る。
   品質判断の材料として所有者に提示する。
3. probe の上限変更は 1 行で、32-step 回帰が両側とも既知 hash と一致した。
   この probe は実験ツールであり、skel や端末設定には触れていない。
   終了時は元の probe に戻した。

## 16. 終了時の端末状態

| 項目 | 値 |
| --- | --- |
| skel | `288e6712df9c8201ef05c12ec5704968d4f71407992cff8e0f03aa002ee0861d` (Phase 50 D、変更なし) |
| probe | `aeaa1b6236d597cc04ad0b642074f451abb70d409b66625a2dd3083c9bd13f3f` (元に復元) |
| replay | `25358 11 198 760` (標準 4 トークン列に復元) |
| decode-probe プロセス | なし |
| cap | 2496000 |
| tz0 | 43.4 ℃ |
| MemAvailable | 8,571,608 kB |
| SwapFree | 11,291,024 kB |

**給電について**: 測定中は AC 給電だったが、終了時点で `AC powered: false` /
`Discharging` (74 %) になっていた。**本作業で給電設定は変更していない。**
USB ケーブルの抜去など外部要因によるものと思われるが、断定はしない。
温度保護・vendor 電源ポリシー・ZRAM・重要サービスへの介入は行っていない。
端末の再起動もしていない。

## 17. HTP 64-step の再現確認 (TVer を止めずに取得)

§10.1 で未取得だった HTP 64-step の 2 回目を、**TVer を起動したまま**取得した。

### 17.1 TVer を止めずに済んだ理由

TVer は `jp.hamitv.hamiand1` で、測定開始時点で**フォアグラウンド再生中**だった
(`topResumedActivity` が `jp.hamitv.hamiand1/.tver.ui.activities.MainActivity`、
音声フォーカス保持、`mIsActive=true`、RSS 745 MB)。

`p50_run.sh` の force-stop リスト (16 個) に **TVer は入っていない**。
リストは GMS・Play Store・ブラウザ・SNS などで、TVer は対象外である。
したがってランナーをそのまま実行しても TVer は止まらない。

実際、測定の前後で TVer のプロセスは同じ pid 2827 のまま生存し、
フォアグラウンドのままだった。

### 17.2 手順

TVer 以外のアプリだけを止めてメモリを空け、静穏な窓で 1 run 取った。

```sh
# TVer (jp.hamitv.hamiand1) を除いたリストだけを止める
for p in com.google.android.gms com.android.vending com.twitter.android.piko \
         dev.shiggy.cord net.waterfox.android.release org.mozilla.fenix \
         com.google.android.googlequicksearchbox de.mm20.launcher2.nightly \
         com.kazumaproject.markdownhelperkeyboard ai.mistral.chat com.facebook.aura; do
    am force-stop $p >/dev/null 2>&1
done
am kill-all >/dev/null 2>&1

# 承認を入れて 1 run
sh /data/local/tmp/c46-27b/approve.sh "p52-nll64htp2" >/dev/null
cd /data/local/tmp/c46-27b
export WANT_SKEL=288e6712df9c8201ef05c12ec5704968d4f71407992cff8e0f03aa002ee0861d
PROFILE_ARG=0 sh p50_run.sh 64 replay 16 nll64htp2 1
```

### 17.3 結果

| 項目 | 値 |
| --- | --- |
| exit | **0** |
| DECODE step 数 | **64** |
| logits sha256 | `d7b272a076bfb79aff543db9cf3042d5663a82a740dd7915228e4a073b53c627` |
| 1 回目との一致 | **一致** |
| cap | 3187200 / 3648000 |
| AnonPages (開始時) | 1,376,488 kB |
| TVer | **pid 2827 のまま生存、フォアグラウンド** |

**HTP 64-step の再現確認が取れた。** これで §10 の 4 run がすべて揃った。

### 17.4 失敗した試行 (除外せず記録)

| 時刻 | 内容 | 理由 |
| --- | --- | --- |
| 21:34 | exit 2 | 承認ファイル切れ (`approve.sh` の単発承認を消費済み) |
| 21:35 | exit 2 | 同上。加えて probe が元の 32-step 版に戻っていた |
| 21:36 | exit 6 | swap guard (265,841 pages)。AnonPages 2.18 GB |
| 21:37 | **exit 0** | **成功**。AnonPages 1.38 GB |

- 21:35 の exit 2 は、前回終了時に probe を元へ戻したため 64-step が
  再び拒否されたことによる。再ビルド版 (`ad15340c...`) を再配置して解消した。
- 21:36 の guard は AnonPages が 2.18 GB だったため。アプリを止めて
  1.38 GB まで下がった直後の run で成功した。
- **ガードは 1024 MiB のまま緩めていない。**

### 17.5 端末の状態 (本節終了時)

| 項目 | 値 |
| --- | --- |
| TVer | 生存 (pid 2827)、フォアグラウンド |
| skel | `288e6712...` (変更なし) |
| probe | `ad15340c...` (64-step 版を配置したまま) |
| replay | 64 トークン列 (要復元) |
| AC 給電 | true に復帰 |
| 再起動 | していない |



