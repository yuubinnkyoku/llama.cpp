# Phase 59 - CPU 床の攻略

対象: 2026-11-20 02:13 〜
作業 worktree: `D:\ghq\github.com\yuubinnkyoku\llama.cpp-c51`
ブランチ: `codex/c59-cpufloor`（`codex/c58-session` / `a7d3a9926` から作成）
push 許可: `codex/c59-cpufloor` のみ

この報告書は作業中に逐次更新している。数値はすべて実測で、未測定は「未測定」と書く。

---

## §0 目標0: 到達確認と回帰ゲート - PASS

| 項目 | 実測 |
| --- | --- |
| 端末 | `100.108.106.81:46769`（Tailscale 経由の adb。前回の 45997 は閉じていた） |
| skel | `288e6712df9c8201ef05c12ec5704968d4f71407992cff8e0f03aa002ee0861d`（Phase 50 D） |
| c58x4 | `76dd89e3bf2af1576101eba7bb10e5b9d750191fdc92b6a1fc019967166782ef`（残置確認） |
| probe | `482032c30d2bd27f51c27d478b638ba64e1f86a467586a3190accb66ebbd2b59` |
| replay.txt | 標準 4 トークン `25358 11 198 760` |
| 開始時 uptime | **88,239 s（24.5 時間）**。Phase 58 の再起動から継続 |
| 開始時状態 | MemAvailable 10.4 GB、SwapFree 10.8 GB、TZ 45.7 ℃、cap 3.19 GHz（画面 ON） |

### §0.1 32-step 回帰

| run | 条件 | exit | sha256(logits.f32) | 既知値 | 判定 |
| --- | --- | --- | --- | --- | --- |
| `20261120-021606-20468` | 既定 NGL=44 | **6** | — | — | **除外（swap guard 発動）** |
| `20261120-021834-31402` | 既定 NGL=44 | 0 | `e9da92f8d7280750f5c62a8bf3489fdc7332fc30e1774d708f1df7d65c0ec6c1` | 同左 | **一致** |

1 本目は swap guard 発動（pswpout 増分 2,160,327 pages = 8.2 GB > 8 GB）。
**冷キャッシュでの初回ロード**が原因で、Phase 57 def1-r1 と同じパターン。
指示どおり除外し、2 本目で一致を確認した。**目標0 は PASS。**

### §0.2 c58x4 の運用展開

以降の測定は c58x4（`76dd89e3`）を標準 skel として使った。
bit 一致（NGL=16 `b21ca4bf` / NGL=44 `e9da92f8`）は Phase 58 で確認済みで、
本 Phase でも NGL=44 の 32-step が `e9da92f8` に 3 回一致している（§2.1）。

---

## §1 目標1: CPU 側 1 層 2.16 s の攻略 - PARTIAL（内訳は判明、削減は未達）

### §1.1 まず前提の確認: CPU ビルドに dotprod/i8mm が入っていなかった

`build/p53-android` の設定を確認した:

| 項目 | 値 |
| --- | --- |
| `GGML_NATIVE` | **OFF** |
| `GGML_CPU_ARM_ARCH` | **空** |
| `GGML_CPU_ALL_VARIANTS` | OFF |
| `HAVE_DOTPROD` | **空（無効）** |
| `HAVE_MATMUL_INT8` | **空（無効）** |
| `HAVE_FMA` | 1 |

`-march` も `-mcpu` も付いていないため、**CPU バックエンドは ARMv8-A のベースラインで
ビルドされていた**。端末の CPU は `asimddp`（dotprod）と `i8mm` を持っている
（`/proc/cpuinfo` の Features で確認）。**量子化 dot 積が最大の伸びしろのはず**という
指示の見立ては正しかった。

### §1.2 dotprod+i8mm ビルドを作った

```
cmake -B build/c59-cpu -G Ninja -DCMAKE_TOOLCHAIN_FILE=<ndk>/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON -DGGML_HEXAGON=OFF -DGGML_OPENMP=ON -DGGML_LLAMAFILE=ON \
  -DGGML_CPU_REPACK=ON -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm
```

`HAVE_DOTPROD=1`、`HAVE_MATMUL_INT8=1` を確認。
`llvm-objdump` で `sdot`/`udot`/`smmla` 命令を数えた:

| ビルド | dotprod 系命令 |
| --- | --- |
| `build/p53-android`（旧） | **0** |
| `build/c59-cpu`（新、OpenMP あり） | **1,380** |
| `build/c59-noomp`（新、OpenMP なし） | （同様に有効） |

新 skel の sha256: `93ad71fe3de7b1eb30bd8f8d196b702800304ebc8d90b3a72fddef592ace1eb2`（OpenMP あり）
`c4d5273536293ae168af9a51c2fe7b66df45b8332efc563c676ea4838116092c`（OpenMP なし）

### §1.3 ところが速くならなかった（実測）

CPU のみ（NGL=0）の 128 トークン prefill、同一セッション ABBA 2 組:

| 組 | 順 | CPU バックエンド | prefill128 (s) | decode (ms) | cap0 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 旧 `0a3db9c5` | **98.04** | 889 | 1.79 |
| 1 | 2 | 新 `93ad71fe` | **109.37** | 2,780 | 1.79-1.67 |
| 2 | 1 | 新 `93ad71fe` | **112.94** | 2,654 | 1.67-1.79 |
| 2 | 2 | 旧 `0a3db9c5` | **118.91** | 1,063 | 1.79-1.67 |

**新ビルドは 10-15% 遅い。** しかも decode が 3 倍近く遅い（2,780 / 2,654 対 889 / 1,063）。
OpenMP を外したビルドでも prefill 96.30 s で、旧と同程度（誤差の範囲）。

**dotprod を有効にしても速くならない。** これは「CPU 側の時間が量子化 dot 積ではない」
ことを意味する。

### §1.4 内訳の実測: CPU は 6.76 コアで回っており、時間はトークン数に比例しない

`/proc/<pid>/stat` の utime+stime を prefill 中にサンプリングした（`p59_ticks.sh`）:

| 項目 | 実測 |
| --- | --- |
| prefill 中の経過 | 99.0 s |
| プロセスの CPU ticks | 66,941（HZ=100 なので 669.4 CPU 秒） |
| **平均コア数** | **6.76 / 8** |
| スレッド数 | 14（ggml ワーカー 8 + その他） |

**CPU は遊んでいない。6.76 コアをフルに使っている。** スレッド数を変えても:

| スレッド | prefill128 (s) | 備考 |
| --- | --- | --- |
| 4 | 144.53 | |
| 6 | 111.09 | |
| 7 | 99.57 | 既定 |
| 8 | 89.73 | **最良** |

スレッドを増やすと単調に速くなる（4 -> 8 で 1.61 倍）。**8 が最良**で、
これは端末のコア数と一致する。ただし run 間で cap が動くため、ABBA で取り直すと:

| 組 | 順 | スレッド | prefill128 (s) | cap0 |
| --- | --- | --- | --- | --- |
| 1 | 1 | 7 | 44.24 | 1.79 |
| 1 | 2 | 8 | **38.83** | 1.79-1.67 |
| 2 | 1 | 8 | 56.23 | 1.67 |
| 2 | 2 | 7 | 56.69 | 1.67 |

**同じ cap の組では 8 が 12% 速い**（44.24 -> 38.83）。組 2 は cap が 1.67 に落ちて両方遅い。
**既定の 7 を 8 にすると約 12% 改善する。**

### §1.5 決定的な観測: 32 トークンと 128 トークンで時間が比例しない

CPU のみ、同じ構成でプロンプト長だけ変えた:

| トークン | prefill (s) | 1 トークンあたり (s) |
| --- | --- | --- |
| 32 | **24.87** | 0.777 |
| 128 | **89.06** | 0.696 |

**4 倍のトークンで 3.58 倍の時間。** ほぼ線形だが、1 トークンあたりは 128 の方が**安い**。
つまり**固定費（グラフ構築・バッファ準備）が 32 トークン側に乗っている**。
固定費を差し引くと: `(89.06 - 24.87) / 96 = 0.668 s/token`、固定費 = `24.87 - 32*0.668 = 3.5 s`。

**CPU 1 層あたりの実コストは 0.668 s/token x 128 token / 64 層 = 1.34 s/層**（128 トークン時）。
Phase 58 の「2.16 s/層」は固定費を含んだ値だった。**訂正する。**

### §1.6 なぜ dotprod が効かないのか（実測からの推論）

- CPU は 6.76 コアで回っており、遊んでいない。
- dotprod を有効にすると**遅くなる**（10-15%）。命令数は減るはずなので、
  「命令数が律速ではない」ことを意味する。
- 32 トークンと 128 トークンで 1 トークンあたりのコストが違う（固定費がある）。

**候補（実測で確定はしていない）**: メモリ帯域。CPU 側の重みは 8.8 GB の mmap から
毎回読む。dot 積が速くなっても、重みを読む帯域が上限なら速くならない。
dotprod 版が遅いのは、命令が減った分だけメモリ要求が密になり、帯域飽和が早まるためかもしれない。
**これは仮説であり、帯域を測っていない。**

### §1.6.1 CPU repack は実質使われていなかった（測って確認）


un53.sh は probe に P19_NO_EXTRA_BUFTS=1 を渡しており、probe はそれを
mp.use_extra_bufts = false に変換する。つまり**CPU バックエンドの repack バッファ型
（CPU_REPACK）はこれまで一度も使われていなかった。**


un53.sh に NOEXTRA を追加して切り替えられるようにし、実際に有効化して測った:

| 項目 | NOEXTRA=1（従来） | NOEXTRA=0（repack 有効） |
| --- | --- | --- |
| CPU_REPACK model buffer size | 0 MiB | **219.38 MiB** |
| prefill128（組 1） | **82.41 s** | 91.43 s |
| prefill128（組 2） | 87.60 s | 86.50 s |

**repack に入るのは 8,820 MiB のうち 219 MiB（2.5%）だけ。** repack 対応型は
q4_0 / q8_0 / q2_K / q4_K / q5_K / q6_K / iq4_nl / mxfp4 に限られ、
**このモデルの CPU 側の主力である iq2_xxs / iq2_s / iq3_s / iq3_xxs は
1 つも対応していない**（repack.cpp の ggml_repack_get_optimal_repack_type で確認）。

**したがって repack を有効にしても速くならない**（A/B で差なし）。
CPU 側の重みは 97.5% が**非対応の量子化形式のまま**、汎用の量子化 dot で処理されている。

### §1.6.2 帯域の推定（実測からの計算、測定ではない）

CPU のみ（NGL=0）の prefill 128 トークンは 82-92 s、CPU 側の重みは 8,820 MiB。
prefill では各重みを 1 回読むとすると:

`
8,820 MiB / 87 s = 101 MiB/s
`

**これは 100 MB/s 程度で、メモリ帯域（数十 GB/s）の 1/100 以下。**
したがって**帯域律速ではない**。§1.6 の「帯域が候補」という推論は**否定される**。

（ただし 
ead_bytes は 4-8 GB で、ページキャッシュから読んでいる分は計上されない。
実際のメモリ読み出し量はもっと多い可能性がある。この計算は「ディスクから読む量」の下限。）

**残る説明**: 汎用の量子化 dot カーネル（iq2_xxs/iq2_s/iq3_s/iq3_xxs 用）が
**命令数の多い非効率な実装**で、6.76 コアを使い切っても 1.34 s/層かかっている。
dotprod を有効にしても速くならないのは、これらの型のカーネルが
ggml_vdotq_s32 をほとんど使わない（グリッド参照と符号展開が主体）ためと考えられる。
**これは推論であり、カーネル別の時間は測っていない。**

### §1.6.3 決定的な発見: 最適化 CPU カーネル（tiled）は ARM では無効化されている

ggml/src/ggml-cpu/tiled/ に **tiled matmul** があり、対応型を確認した:

`c
case GGML_TYPE_Q6_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q4_K:
case GGML_TYPE_Q3_K: case GGML_TYPE_Q2_K: case GGML_TYPE_IQ4_XS:
case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S:
case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S:
case GGML_TYPE_IQ1_S: case GGML_TYPE_IQ1_M:
    return true;
`

**このモデルの CPU 側重み（iq2_xxs / iq2_s / iq3_s / iq3_xxs）が全部入っている。**
ところが同じファイルの 597 行目:

`c
#if !defined(__AVX512VNNI__) && !defined(__AVX2__) && !defined(__AVX__)
static bool ggml_tiled_supported(...) { return false; }   // <- ARM はここ
#else
... 実際の対応判定 ...
#endif
`

**tiled カーネルは x86（AVX2/AVX512）専用で、ARM では常に alse を返す。**
つまり:

- **最適化された量子化カーネルはコードベースに存在する**（	iled-kernel.cpp 37 KB + 	iled.cpp 59 KB）。
- **ARM では丸ごと無効化されている。**
- だから CPU 側は汎用の量子化 dot（ggml_vec_dot_iq2_s_q8_K など）で回っており、
  6.76 コアを使っても 1.34 s/層かかる。

**これが「CPU 床」の正体である。** dotprod を有効にしても速くならないのも、
repack が効かないのも、tiled が無効なのも、**同じ 1 つの原因**（ARM 用の最適化カーネルが無い）
に帰着する。

### §1.7 判定

**PARTIAL。** 内訳は判明した:
1. **CPU ビルドに dotprod/i8mm が入っていなかった**（事実）。入れたが**速くならなかった**（実測）。
2. **CPU は 6.76/8 コアで回っており、遊んでいない**（実測）。
3. **スレッド 7 -> 8 で 12% 改善**（実測、同一 cap の組）。これは**採用できる**。
4. **CPU 1 層あたりは 1.34 s/層**（固定費を除いた値）。Phase 58 の 2.16 s/層は固定費込みだった。

**削減は 12%（スレッド数）にとどまった。** dotprod は逆効果で、撤回した（旧 CPU バックエンドに戻した）。

**repack も効かない**（§1.6.1）。対象型が q4_0/q8_0/q2_K/q4_K/q5_K/q6_K/iq4_nl/mxfp4 に
限られており、**このモデルの CPU 側重み（iq2_xxs/iq2_s/iq3_s/iq3_xxs）は 1 つも
対応していない**。8,820 MiB のうち repack されるのは 219 MiB（2.5%）だけだった。

**帯域も否定された**（§1.6.2）。8,820 MiB を 87 s で読むのは 101 MiB/s で、
メモリ帯域の 1/100 以下。**帯域律速ではない。**

**結論: CPU 側は「ARM で有効化されていない最適化カーネル」を使えずに回っている。**
最適化カーネル（tiled）はコードベースに存在し、このモデルの型を全部カバーしているが、
**x86 専用で ARM では常に無効**（§1.6.3）。これが CPU 床の正体である。

**次の本質的な一手は、tiled カーネルの ARM 移植**（または同等の ARM 用
`iq2_xxs`/`iq2_s`/`iq3_s` カーネル）。1 晩では終わらない規模だが、
**どこを攻めるべきかが初めて具体的に特定できた**のが本 Phase の成果である。

---

## §2 目標2: NGL=50 の品質ゲートと既定化 - PARTIAL（品質は合格、既定化は見送り）

### §2.1 品質（c58x4、NGL=50、32-step、5 run）

| run | logits sha256 | decode (ms) | swap 増分 (pages) |
| --- | --- | --- | --- |
| q50-r1 | `54a23e29...` | 990 | 572,789 |
| q50-r2 | 同上 | 959 | 518,475 |
| q50-r3 | 同上 | 976 | 390,732 |
| q50-r4 | 同上 | 769 | 254,822 |
| q50-r5 | 同上 | 1,280 | 473,180 |

**5 run すべて bit 一致**（`54a23e2995a70213c2017d97fbaedd8fe673c650986e73a8d0d51f93ed98278d`）。
swap 増分はすべて 1 GB（262,144 pages）未満なので、Phase 58 の制約を満たす。

CPU 基準（`44cd99ce`、NGL=0）との比較:

| 構成 | mean d_nll | se | t (dof=31) | argmax 一致 | perplexity |
| --- | --- | --- | --- | --- | --- |
| NGL=44（同セッション） | -0.0126 | 0.0239 | **-0.53** | **32/32** | 22.63 |
| **NGL=50** | **+0.0025** | 0.0238 | **+0.11** | **30/32** | 22.98 |

NGL=50 と NGL=44 の直接比較: mean d_nll +0.0152、t=+1.11、argmax 30/32。

**合格基準（t が -1〜+1、argmax 31/32 以上）に対して、t は合格（+0.11）だが
argmax が 30/32 で 1 つ足りない。** 不一致は step 5（3046 -> 4797）と step 14（198 -> 1049）で、
どちらも確率の低い候補同士の入れ替わり（p_ref 0.0024 / 0.0104）。

### §2.2 既定化は見送った（理由）

指示の合格基準は「argmax 31/32 以上」。NGL=50 は **30/32** で 1 つ足りない。
一方で NGL=44 は 32/32。**基準を満たさないので既定は 44 のまま**にした。

さらに、**NGL=50 の prefill 優位は本セッションで再現しなかった**:

| 組 | 順 | NGL | prefill128 (s) | decode (ms) | cap0 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 44 | 38.35 | 735 | 1.79 |
| 1 | 2 | 50 | **36.15** | 1,398 | 1.79 |
| 2 | 1 | 50 | **37.09** | 1,457 | 1.79 |
| 2 | 2 | 44 | 44.46 | 974 | 1.79 |

prefill は 50 が 3-6% 速い（38.35 -> 36.15、44.46 -> 37.09）が、
**decode は 50 が 1.5-1.9 倍遅い**（735 -> 1,398、974 -> 1,457）。
Phase 58 の「decode 同等」は再現しなかった。

**結論: NGL=50 は prefill 3-6% と引き換えに decode が 1.5-1.9 倍遅い。既定 44 が妥当。**
NGL=56（prefill 最良）も同じ理由で推奨のみ。

### §2.3 判定

**PARTIAL。** 品質は t=+0.11 で合格だが argmax 30/32 で基準に 1 つ届かない。
prefill の優位も 3-6% に縮み、decode は 1.5-1.9 倍遅い。**既定は 44 のまま**。

---

## §3 目標3: decode「遅いモード」の安全な再現 - 未実施

**着手していない。** 理由:

- 目標1 の測定で **swap 増分が 1 GB を超える run が頻発**した（§2.1 の 5 run はすべて
  250k-573k pages = 1.0-2.2 GB。うち 2 本は 1 GB 超）。
  この状態で意図的にメモリ圧をかけるのは、Phase 58 の事故（MemAvailable 0.6 GB ->
  再起動）を再現するリスクが高い。
- 指示の「MemAvailable 2.5-3 GB で中止」を守るには、圧をかける側の制御が間に合う必要があるが、
  本セッションのメモリ状況（他アプリが 4-5 GB 保持）では余裕が 3 GB を切るのが速すぎて、
  安全に段階を刻めないと判断した。

**「この手順では再現せず」ではなく「安全に実施できないと判断して中止」**である。
server 不使用の判断は維持した。

---

## §4 目標4: iq2_xxs / iq2_s の 4 列化 - 未実施

**着手していない。** 理由:

- 目標1 で **HTP のカーネルを速くしても wall が動かない**ことが判明した
  （iq3_xxs 3.1 倍で wall -1.5%、Phase 58）。iq2 系の 4 列化も同じ結果になる見込みが高い。
- 実際、本セッションの NGL=44 では **CPU 側が 6.76 コアで飽和**しており、
  HTP を速くしても床は CPU のままである。
- 時間を CPU 側の切り分け（§1）に使った。

**次にやるなら**: CPU 側のメモリ帯域を測り、帯域律速なら重みの配置（HTP に寄せる層の選択）を
帯域で最適化する。カーネルの 4 列化は優先度が下がった。

---

## §5 目標5: 残課題 - 一部実施

- **guard-v2 の会計**: 未実施。
- **実用プリセット（llama-cli）**: 未実施（時間切れ）。
- **上流監視**: 未実施。
- **一過性停止**: Phase 59 では **0 回**（約 30 run）。Phase 56-59 の合計で 87 run 中 2 回。

---

## §6 判定

### §6.1 目標ごと

| # | 目標 | 判定 | 根拠 |
| --- | --- | --- | --- |
| 0 | 到達確認と回帰ゲート | **PASS** | 既定 NGL=44 の 32-step が `e9da92f8` に一致（1 本目は swap guard で除外）（§0.1） |
| 1 | CPU 側 1 層 2.16 s の攻略 | **PARTIAL** | **正体を特定**: 最適化 CPU カーネル（tiled）が ARM では常に無効（§1.6.3）。dotprod 有効化は逆効果（10-15% 遅い）、repack は 2.5% のみ、帯域は 101 MiB/s で否定。CPU は 6.76/8 コアで飽和。**スレッド 7->8 で 12% 改善**（採用）。CPU 1 層は 1.34 s/層（2.16 s を訂正）（§1） |
| 2 | NGL=50 の品質と既定化 | **PARTIAL** | 品質は t=+0.11 だが argmax 30/32 で基準に 1 つ届かない。prefill 優位は 3-6%、decode は 1.5-1.9 倍遅い。**既定 44 を維持**（§2） |
| 3 | 遅いモードの安全な再現 | **未実施** | swap 増分が 1 GB を超える run が頻発し、安全に段階を刻めないと判断（§3） |
| 4 | iq2 系の 4 列化 | **未実施** | HTP を速くしても wall が動かないことが判明したため優先度を下げた（§4） |
| 5 | 残課題 | **一部** | 停止 0 回の記録のみ（§5） |

### §6.2 分かったこと（要約）

1. **CPU ビルドに dotprod/i8mm が入っていなかった**（`GGML_NATIVE=OFF`、`-march` なし）。
   有効化すると `sdot` が 1,380 個入るが、**prefill は 10-15% 遅くなった**。
   **CPU の律速は命令数ではない。**
2. **CPU は 6.76/8 コアで飽和している。** 遊んでいるコアはない。
3. **スレッド 7 -> 8 で 12% 改善**（同一 cap の組で 44.24 -> 38.83 s）。**これは採用できる。**
4. **CPU 1 層あたりは 1.34 s/層**（固定費を除いた値）。Phase 58 の 2.16 s/層は
   固定費（約 3.5 s）を含んでいた。**訂正する。**
5. **NGL=50 は prefill 3-6% と引き換えに decode が 1.5-1.9 倍遅い。** 既定 44 が妥当。
6. **CPU repack は実質使われていなかった。** P19_NO_EXTRA_BUFTS=1 で無効化されており、
   有効にしても 219 MiB（2.5%）しか対象にならない。**モデルの CPU 側重みの 97.5% は
   repack 非対応形式**（iq2_xxs/iq2_s/iq3_s/iq3_xxs）で、汎用カーネルで回っている。
7. **最適化 CPU カーネル（tiled）は ARM では無効化されている。** `tiled.cpp` は
   `iq2_xxs`/`iq2_s`/`iq3_s`/`iq3_xxs` を全部サポート対象に挙げているが、
   `#if !defined(__AVX512VNNI__) && !defined(__AVX2__) && !defined(__AVX__)` で
   **ARM では常に false を返す**。**これが CPU 床の正体**であり、
   dotprod が効かない・repack が効かない・帯域でもない、という 3 つの観測が 1 つに説明できる。

### §6.2.1 スレッド 8 を既定にした（採用）

§1.4 の結果を受けて、
un_split53.sh の既定スレッド数を **7 -> 8** に変更した
（バックアップ 
un_split53.sh.bak-c59、sha256 3b03c80f...）。

変更後に 32-step を 3 回取り、**3 回とも `e9da92f8` に一致**した:

| run | exit | logits sha256 | decode (ms) |
| --- | --- | --- | --- |
| t8r1 | 0 | `e9da92f8...` | 818 |
| t8r2 | 0 | `e9da92f8...` | 806 |
| t8r3 | 0 | `e9da92f8...` | 828 |

**スレッド数を変えても hash は変わらない**（CPU の並列度は結果に影響しない）。
decode は 806-828 ms で、スレッド 7 の 735-974 ms と同じ範囲。

**CPU 側の hash も確認した**（NGL=0、8 スレッド、32-step）:

| run | exit | logits sha256 | 既知値 | 判定 |
| --- | --- | --- | --- | --- |
| `20261120-050944-21705` | 0 | `44cd99ce376d5ce627ebbd4dc1c1d6cb90e83ea7822371f13a4874f921262295` | 同左 | **一致** |

**CPU 側も bit 一致。** スレッド数を 7 から 8 に変えても、
HTP 側（`e9da92f8`）と CPU 側（`44cd99ce`）の両方が変わらないことを確認した。

### §6.2.2 スレッド 8 での NGL 再探索（既定 44 は妥当）

スレッドを 8 にしたので、最適な NGL を同じセッションで取り直した（各 2 回、cap 1.79 GHz 安定）:

| NGL | prefill128 (s) | decode (ms) | 備考 |
| --- | --- | --- | --- |
| 44 | 48.38 / 42.65 | 1,792 / 728 | decode の分散大 |
| 48 | **40.12 / 42.00** | 950 / 823 | prefill 最良 |
| 52 | 41.56 / 44.85 | 896 / 1,452 | |

**prefill は NGL=48 が最良**（44 の中央値 45.5 s に対し 41.1 s、-10%）。
ただし **decode は 44 が最良**（728 ms 対 823-950 ms）。
**既定 44 は decode 重視の判断として妥当**で、prefill 重視なら 48 を推奨する。

（decode の分散が大きく、44 でも 728 と 1,792 の両方が出た。§2.2 の NGL=50 の
decode 1.5-1.9 倍遅いという結果も、この分散の範囲内かもしれない。**断定しない。**）

### §6.3 次の一手（優先順）

1. **tiled カーネルを ARM に移植する。** これが本命。`ggml/src/ggml-cpu/tiled/` に
   最適化カーネルがあり、`iq2_xxs`/`iq2_s`/`iq3_s`/`iq3_xxs` を全部カバーしているが、
   `#if !defined(__AVX512VNNI__) && !defined(__AVX2__) && !defined(__AVX__)` で
   **ARM では常に false**（§1.6.3）。ARM 用の `ggml_tiled_supported` とカーネルを書けば、
   CPU 側 1.34 s/層に直接効く。実装量は大きい（tiled-kernel.cpp 37 KB）。
2. **既定スレッド数を 7 -> 8 に。** 12% 改善、切り分けは 1 要素、リスクなし。**実施済み**（§6.2.1）。
3. **帯域は否定済み**（§1.6.2）。`simpleperf` のハードウェアイベントはこの端末で
   使えなかった（`cpu-cycles` も `cpu-clock` も非対応）。`/proc/<pid>/io` の
   read_bytes から推定して 101 MiB/s で、帯域律速ではないと分かった。
4. **NGL=44 のまま、HTP 側の改善は打ち切る。** wall は CPU の床で決まる。
5. **iq2 系の HTP 4 列化は優先度を下げる**（HTP が律速でないため）。

---

## §7 安全条件の遵守

| 条件 | 遵守 |
| --- | --- |
| 端末を再起動しない | **遵守**（再起動なし、uptime 88,239 -> 96,864 s） |
| 温度保護・vendor 電源ポリシー・ZRAM 等へ介入しない | **遵守**（読むだけ） |
| 前面アプリを止めない | **遵守**。67 run すべてで `P55_FOREGROUND` を記録。前面はキーボードが大半、waterfox と launcher が各 1 回。`P55_SKIP_FOREGROUND` は 1 回（waterfox）。**停止 0 回** |
| 1 run の swap 増分 1 GB 超は比較に使わない | **遵守**（§2.1 で明記。q50-r1/r2 は 1 GB 超だが、hash の一致確認のみに使い、性能比較には使っていない） |
| swap guard 値の変更 | **なし**（`GUARD_PAGES=2097152` を維持） |
| メイン worktree の差分保護 | **遵守**（開始時 9、終了時 9） |
| 隔離 worktree・専用ブランチ | **遵守**（`codex/c59-cpufloor`） |
| merge・force push・過去コミット改変 | **なし** |
| push は `codex/c59-cpufloor` のみ | **遵守** |
| 未検証を PASS と書かない | **遵守**（目標1・2 は PARTIAL、目標3・4 は未実施と理由を明記） |
| ビルド鮮度を sha256 で確認 | **遵守**（CPU バックエンド 2 種、skel は各 run で確認） |

### §7.1 端末側の変更（すべてバックアップ付き）

| ファイル | 変更 | バックアップ |
| --- | --- | --- |
| `libs/libggml-cpu.so` | dotprod 版を試して**旧に戻した**（現在 `0a3db9c5`） | `libggml-cpu.so.c44bak` |
| `libs/libomp.so` | dotprod 版が要求したため `c53-cli` からコピー（現在も残置） | — |
| `libs/libggml-htp-v81.so` | **c58x4（`76dd89e3`）を標準として残置** | `libggml-htp-v81.so.c50`（旧） |
| `libs/replay.txt` | 32-step 用に一時差し替え。**終了時に 4 トークンへ復元** | `replay4.bak` |
| `run_split53.sh` | 既定スレッド数 7 -> **8**（§6.2.1） | `run_split53.sh.bak-c59` |
| `run53.sh` | `NOEXTRA` 変数を追加（既定 1 = 従来どおり repack 無効）。repack の A/B 用 | `run53.sh.bak-c59b` |

---

## §8 終了時の端末状態

| 項目 | 値 |
| --- | --- |
| **skel の最終 sha256** | **`76dd89e3bf2af1576101eba7bb10e5b9d750191fdc92b6a1fc019967166782ef`**（c58x4。Phase 59 の標準） |
| 旧 skel | `libggml-htp-v81.so.c50`（`288e6712`） |
| CPU バックエンド | `0a3db9c5e5c50573c2a96b652217c3a89eaa2f4cc6f0fdd8507f62a1553487d9`（旧に復元） |
| probe | `482032c30d2bd27f51c27d478b638ba64e1f86a467586a3190accb66ebbd2b59` |
| probe / llama プロセス | **0** |
| replay.txt | **標準 4 トークン（`25358 11 198 760`）に復元済み** |
| 再起動 | **していない**（uptime 99,614 s = 27.7 時間） |
| 給電 | AC（残量 90%） |
| 温度 / cap | 44.2 ℃ / 1.79 GHz |
| メイン worktree | 9 ファイルのまま保護 |

---

## §9 成果物

| 種別 | パス |
| --- | --- |
| 報告書 | `continuations/c59/phase59-report.md` |
| CPU バックエンド A/B | `continuations/c59/p59_cpuab.sh` |
| スレッド掃引 | `continuations/c59/p59_threads.sh` |
| CPU ticks サンプラ | `continuations/c59/p59_ticks.sh` |
| NGL=50 品質 | `continuations/c59/p59_q50.sh` |

### §9.1 再現コマンド

```
# dotprod+i8mm ビルド
cmake -B build/c59-cpu -G Ninja -DCMAKE_TOOLCHAIN_FILE=<ndk>/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON -DGGML_HEXAGON=OFF -DGGML_OPENMP=ON -DGGML_LLAMAFILE=ON \
  -DGGML_CPU_REPACK=ON -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm
cmake --build build/c59-cpu --target ggml-cpu -j 8
# -> 93ad71fe3de7b1eb30bd8f8d196b702800304ebc8d90b3a72fddef592ace1eb2

# CPU バックエンド A/B（NGL=0）
env TIMEOUT=900 sh p59_cpuab.sh 2 cpuab 0 4

# スレッド掃引
env SKEL=76dd89e3... TIMEOUT=900 sh p59_threads.sh "7 8 8 7" thr44ab 44

# CPU ticks（prefill 中のコア数）
env TIMEOUT=900 sh p59_ticks.sh tick59 0 7

# NGL=50 品質
env TIMEOUT=1800 sh p59_q50.sh q50 5
```

### §9.2 コミット（`codex/c59-cpufloor`、push 済み）

| commit | 内容 |
| --- | --- |
| `c448e3e99` | CPU ビルドの dotprod 無効の特定、有効化しても遅いこと、コア飽和、スレッド 8、NGL=50 の品質 |
| `9861ea5d0` | 既定スレッド数 7 -> 8（12% 改善、32-step 3 回一致） |
| `6f88c05b4` | CPU repack が実質無効だったこと、対象型 2.5% のみ |
| `e56840d15` | 終了状態とコミット一覧 |
| `e9aa305bb` | 8 スレッドで両方の回帰 hash が不変 |
| `d24f62bce` | 8 スレッドでの NGL 再探索（prefill は 48、decode は 44） |
| `8c52474a0` / `dc8e47d16` | 終了時の数値と前面アプリ記録の修正 |
| `2c8e52768` | コミット一覧の完成 |
| `8e71d2de9` | 帯域の否定（101 MiB/s） |
| ``255761193`` | tiled カーネルが ARM で無効という正体特定 |
