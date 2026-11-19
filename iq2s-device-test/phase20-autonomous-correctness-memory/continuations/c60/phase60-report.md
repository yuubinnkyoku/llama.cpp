# Phase 60 - tiled カーネルの ARM 立ち上げ

対象: 2026-11-20 08:05 〜
作業 worktree: `D:\ghq\github.com\yuubinnkyoku\llama.cpp-c51`
ブランチ: `codex/c60-tiledarm`（`codex/c59-cpufloor` / `1b7c83d37` から作成）
push 許可: `codex/c60-tiledarm` のみ

**この Phase は端末が使えなかった。** 所有者の指示により、端末なしでできる範囲
（実装・ビルド・ホスト側の検証）を進めた。**実機測定はすべて未実施**である。
数値はすべてホスト側の実測で、実機の速度・品質は測っていない。

---

## §0 目標0: 到達確認と回帰ゲート - 未実施（端末不達）

| 項目 | 結果 |
| --- | --- |
| `adb devices` | **デバイスなし**（複数回試行） |
| `adb connect 100.108.106.81:46769` | **接続拒否**（10061） |
| `adb reconnect offline` | 変化なし |
| 試行回数 | 4 回（08:05 から 09:0x まで、各 1-3 分間隔） |

**10 分以上復旧しなかったため、指示どおり「端末未復帰」として記録し、
ホスト側の作業に切り替えた。** 回帰ゲート（`e9da92f8` / `44cd99ce`）は未実施。

---

## §1 目標1 Stage A: スカラーでドライバ全体の正しさ検証 - 実装完了、実機検証は未実施

### §1.1 gate の変更（デフォルト動作は不変）

`ggml/src/ggml-cpu/tiled/tiled.cpp` の 2 箇所を変更した。

**1. `ggml_tiled_supported` のアーキ判定**

```c
// 変更前
#if !defined(__AVX512VNNI__) && !defined(__AVX2__) && !defined(__AVX__)
// 変更後
#if !defined(__AVX512VNNI__) && !defined(__AVX2__) && !defined(__AVX__) && !defined(__ARM_NEON)
```

これで ARM でも実際の型判定（`iq2_xxs`/`iq2_xs`/`iq2_s`/`iq3_xxs`/`iq3_s` を含む 13 型）に
到達する。**x86 の挙動は 1 ビットも変わらない**（`__ARM_NEON` は x86 で未定義）。

**2. マスタースイッチの既定値**

```c
static bool ggml_tiled_matmul_enabled(void) {
#if defined(__ARM_NEON)
    static bool enabled = false;   // ARM は既定 OFF
#else
    static bool enabled = true;    // x86 は従来どおり既定 ON
#endif
    static std::once_flag flag;
    std::call_once(flag, []() {
        const char * env = getenv("GGML_CPU_TILED_MM");
        if (env != NULL) { enabled = atoi(env) != 0; }
    });
    return enabled;
}
```

**ARM ビルドの既定動作は完全に不変**（tiled は入らない）。
`GGML_CPU_TILED_MM=1` を明示したときだけ tiled パスに入る。
`GGML_CPU_TILED_MM=0` は従来どおり明示 OFF。

**3. 防御的な `ne00` チェックを追加**

```c
if (src0->ne[0] % TILED_TILE_K != 0) {
    return false;
}
```

ドライバ側に `GGML_ASSERT(ne00 % 256 == 0)` があるが、gate で弾くほうが安全。
対応 13 型はすべて QK_K ブロック単位なので実際には常に成立する（挙動は変わらない）。

### §1.2 Stage A 用ビルド（スカラー）

`-DGGML_CPU_ARM_ARCH=armv8-a`（dotprod なし）でビルドすると、
NEON カーネルは `#if defined(__ARM_FEATURE_DOTPROD)` でコンパイルアウトされ、
**スカラー microtile にフォールバックする**。これが Stage A の検証対象。

| ビルド | sha256 | サイズ | `sdot` 命令 | `microtile_neon` シンボル |
| --- | --- | --- | --- | --- |
| `build/c60-scalar`（armv8-a） | `eba4af55457b2ad19f3199237f4974229dfa2fa7ac99e6906138d4b15659e16a` | 6,193,776 B | **0** | **0** |
| `build/c60-neon`（armv8.6-a+dotprod+i8mm） | `c5499c18431c2d236713408d33e6b38de1a676bb214e85fea0c1334cec7f2db5` | 6,193,776 B | **1,216** | 4（`<32,false,128,0> <32,false,128,1> <16,false,128,0> <16,false,128,1>`） |
| 現行標準（Phase 59 の `0a3db9c5`） | `0a3db9c5e5c50573c2a96b652217c3a89eaa2f4cc6f0fdd8507f62a1553487d9` | 4,903,128 B | 0 | 0 |

**スカラービルドに `sdot` が 0 個、NEON ビルドに 1,216 個**あることを objdump で確認した。
これは「Stage A と Stage B が別物である」ことの機械的な証拠になる。

### §1.3 実機検証（未実施）

以下は**測っていない**:

- クラッシュしないこと（`ne00 % 256` ASSERT を含む）
- 品質ゲート（`44cd99ce` との 32-step、t=-1〜+1、argmax 31/32）
- tiled パスを実際に通っていることの確認（`GGML_CPU_TILED_MM=0` で `44cd99ce` に戻る等）

**端末が復帰したら最初にやるべきことは §5 に書いた。**

---

## §2 目標1 Stage B: NEON+dotprod microtile の実装 - 実装完了、実機検証は未実施

### §2.1 実装

`ggml/src/ggml-cpu/tiled/tiled-kernel.cpp` に `tiled_run_microtile_neon` を追加した。

**設計の要点**:

- **SDOT は signed x signed**。AVX2 の `maddubs`（unsigned x signed）と違い、
  act-bias（+128）も sign-trick も要らない。重みをレジスタ上で
  `w = q0 - BIAS` にデバイアスし、そのまま `vdotq_s32` に渡す。
- **`tiled_repack_src1` は NEON では no-op**（既存の `#else` 分岐がそのまま使われる）。
  アクティベーションは自然レイアウトのまま読む。
- **int32 で厳密に積算し、最後に 1 回だけ float スケール**。
  スカラー参照は「サブブロックごとに float でスケールして float で積算」なので
  **丸め順序が違う**（＝ hash は変わる）。代数は同じ。
- **`BIAS` はテンプレート引数**。iq2/iq3 系は符号付きコードを `value + 128` で持つので BIAS=128、
  qK 系は小さな符号なし値がそのまま int8 に収まるので **BIAS=0**。
  最初これを 128 に固定して書いてしまい、**ホスト側の代数検証が 400/400 で落ちて発見した**（§2.3）。
- **HAS_MIN の型（qK 系）も NEON で処理する**。bsums 項はもう 1 本の int32 積算器で足りるので、
  `acc_min[j] += bsum * mins[s]` を積んでおき、最後に `dmin * acc_min` を引く。
  これで **13 型すべてが NEON パスに入る**（スカラーフォールバックは無くなった）。

**dispatch の順序**（`tiled_run_microtile`）:

```
VNNI -> AVX2 -> AVX -> NEON(dotprod) -> scalar
```

NEON 分岐は `#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)`。
dotprod が無い ARM ビルドでは自動的にスカラーに落ちる。

### §2.2 生成コードの検証（ホスト側、実測）

`llvm-objdump` で `tiled_run_microtile_neon<32, 1>` を逆アセンブルし、
意図した命令が出ていることを確認した:

| 命令 | 個数 | 意味 |
| --- | --- | --- |
| `sdot v?.4s, v?.16b, v?.16b` | **32** | 16 列 x 2 ベクタ（SUBBLK=32） |
| `mla v?.4s, ...` | **16** | 列ごとの scale 積算 |
| `addv s?, v?.4s` | **16** | 列ごとの水平加算 |

**コンパイラが意図した構造をそのまま出している。** ループが展開され、
16 列分のアキュムレータがレジスタに載っている。

### §2.3 代数の検証（ホスト側、実測）

NEON カーネルはこのホスト（x86）では実行できないが、**代数は検証できる**。
`p60_neon_algebra.cpp` に両方の式をそのまま書き写して比較した:

- 参照側: `tiled_run_microtile_scalar` の式（BIAS*bsum 補正、float 積算）
- NEON 側: `tiled_run_microtile_neon` の式（デバイアス、int32 積算、最後に float）

| 経路 | 試行数 | 結果 |
| --- | --- | --- |
| 標準（num_k=1）、SUBBLK=32 | 200 | **一致**（相対誤差 < 1e-4） |
| 標準（num_k=1）、SUBBLK=16 | 200 | **一致** |
| 狭パス（num_k=2）、SUBBLK=32 | 50 | **一致** |
| 狭パス（num_k=2）、SUBBLK=16 | 50 | **一致** |
| 狭パス（num_k=4）、SUBBLK=32 | 50 | **一致** |
| 狭パス（num_k=4）、SUBBLK=16 | 50 | **一致** |
| **HAS_MIN**（qK 系）、SUBBLK=32 | 200 | **一致** |
| **HAS_MIN**（qK 系）、SUBBLK=16 | 200 | **一致** |

**合計 1,000 試行すべて一致。** 相対誤差の閾値は 1e-4（float の丸め順序の差のみを許容）。

**この検証は実際にバグを 1 つ捕まえた。** 最初の実装は `BIAS` を 128 に固定していたが、
qK 系の BIAS は 0 である。HAS_MIN の 400 試行が**すべて**失敗したことで気づき、
テンプレート引数に直した。**実機に持っていく前に見つかった**のは大きい
（実機だと「品質が合わない」としか分からず、原因の特定に時間がかかる）。

**int32 オーバーフローの確認**: 積算器の最大絶対値は **1.23e7**（SUBBLK=32）、
**1.11e7**（SUBBLK=16）で、int32 の上限 2.15e9 に対して **約 175 倍の余裕**がある。
ランダムな最悪ケース（q0 0-255、q1 -128..127、scale 1-31）での値なので、
実データでも余裕は十分と判断できる。**ただし実機の実データで測ったわけではない。**

### §2.3.1 x86 経路に退行がないことの確認（ホスト側、実測）

gate の変更は x86 と ARM で共有されるファイルに入っているので、
**x86 が壊れていないことを機械的に確認した**。

-DGGML_AVX2=ON -DGGML_FMA=ON で x86 ビルドを作り、	iled-kernel.cpp.obj を逆アセンブル:

| 命令 | 個数 | 意味 |
| --- | --- | --- |
| pmaddubsw | **116** | AVX2 の maddubs（既存カーネル） |
| pmaddwd | **116** | AVX2 の madd（既存カーネル） |
| sdot | **0** | NEON は入っていない（当然） |

	iled.cpp.obj には ggml_compute_forward_mul_mat_tiled_driver<block_iq2_xxs> などの
実体化が残っている（__pdata シンボルで確認）。**x86 の tiled パスは無傷**である。

__ARM_NEON は x86 で未定義なので、#if defined(__ARM_NEON) の分岐は
x86 では 1 行もコンパイルされない。**x86 のバイナリは変更前と同一のコードになる**
（実際、ggml_tiled_matmul_enabled の既定値も 	rue のまま）。

### §2.4 実機検証（未実施）

以下は**測っていない**:

- 品質ゲート（32-step、`44cd99ce` 基準の NLL/argmax）
- CPU のみ prefill128 の ABBA（A=`0a3db9c5` vs B=NEON+`GGML_CPU_TILED_MM=1`）
- NGL=44 end-to-end の ABBA
- decode が不変であることの確認

---

## §3 目標2: decode 分散の定量視 - 未実施（端末不達）

12 本連続実行による分布測定は端末が必要なため実施できなかった。

---

## §4 目標3: 上流監視 - 実施（重要な発見あり）

### §4.1 PR #32582「hexagon: support for int-hmx and automatic f32 -> f16 lowering」

**本文は取得できなかった。** GitHub の PR ページは HTTP 200 で返るが、
本文がナビゲーション部分に埋もれて取得できなかった（HTML の切り詰め）。
**タイトルと起票者（max-krasnyansky）、起票日（2026-11-19）以外は未確認**である。
「int-hmx」は HMX の整数演算を指すと推測されるが、**本文を読んでいないので断定しない**。

### §4.2 tiled の ARM/NEON 実装は既に上流にある（重要）

検索で判明した:

- **llama.cpp PR #27851**（jbooth、2026-09）が tiled K-quant matmul 本体。
  「3-6x speed improvement for large matmul、break even at 4096x64 * 64x4096、
  GEMV では 80%（net loss）」と記載がある。
- **Arm NEON カーネルは既に第三者が実装・公開している**:
  [hardwiredagent/silicon-forge-arm-neon-study](https://github.com/hardwiredagent/silicon-forge-arm-neon-study)
  （MIT、最終コミット 2026-11-13）。報道によれば
  **Neoverse N1 で IQ3_M の prompt processing が 11.5 -> 74.5 tokens/s（約 6.5 倍）**。
  手法は「SDOT の活用」と「重みロードの最適化（bias debiasing on load）」で、
  **本 Phase の実装方針と一致する**。

**これは本 Phase の実装が「正しい方向」であることの傍証**ではあるが、
**本実装がそのパッチと同一かは確認していない**（コードを読んでいない）。
また **6.5 倍は Neoverse N1（サーバ向け ARM）での値**で、
この端末（Snapdragon 8 Elite Gen 5）で同じ倍率が出る保証はない。

**取り込みはしていない**（禁止事項）。

---

## §5 端末が復帰したときにやること（優先順）

**p60_tiled.sh を用意してある**（sh p60_tiled.sh <stage>）。以下はその中身の説明。

1. **回帰ゲート**: c58x4・NGL=44・スレッド 8 で `e9da92f8`、NGL=0 で `44cd99ce`。
2. **Stage A（スカラー）**: `build/c60-scalar` の `libggml-cpu.so` を
   `libs/libggml-cpu.so.c60scalar` として置き、`GGML_CPU_TILED_MM=1` で
   NGL=0 の 32-step を 1 本。**クラッシュしないこと**と
   **`44cd99ce` との品質ゲート**（t=-1〜+1、argmax 31/32）を確認。
   `GGML_CPU_TILED_MM=0` で `44cd99ce` に戻ることも確認（パスを通っている証拠）。
3. **Stage B（NEON）**: `build/c60-neon` の `libggml-cpu.so` を
   `libs/libggml-cpu.so.c60neon` として置き、同じ品質ゲート。
4. **速度**: CPU のみ prefill128 の ABBA 2 組以上
   （A=`0a3db9c5`、B=NEON+`GGML_CPU_TILED_MM=1`）。同一セッション・同一 cap。
   swap 増分 1 GB 超の run は除外。中央値 15% 以上を effective とする。
5. effective なら NGL=44 end-to-end の ABBA 2 組。

**注意**: `build/c60-neon` は `GGML_OPENMP=OFF` でビルドしてある。
Phase 59 で OpenMP 版が `libomp.so` を要求したため、依存を減らす目的で外した。
`libs/` に置くときは `libomp.so` が不要になる（Phase 59 でコピー済みのものは残置）。

---

## §6 判定

### §6.1 目標ごと

| # | 目標 | 判定 | 根拠 |
| --- | --- | --- | --- |
| 0 | 到達確認と回帰ゲート | **未実施** | 端末不達（4 回試行、10 分以上）。指示どおり記録してホスト作業へ |
| 1 Stage A | スカラーでドライバ検証 | **実装完了 / 実機未検証** | gate を ARM に開放（既定 OFF）、スカラービルド作成（`sdot` 0 個）。実機の品質・クラッシュ確認は未実施 |
| 1 Stage B | NEON microtile | **実装完了 / 実機未検証** | `tiled_run_microtile_neon` 実装（**13 型すべて**、qK 系の HAS_MIN も含む）、`sdot` 1,216 個、代数 **1,000 試行一致**、int32 余裕 175 倍。**x86 経路に退行なし**（`vpmaddubsw` 116 個、`sdot` 0 個）。実機の品質・速度は未実施 |
| 2 | decode 分散 | **未実施** | 端末不達 |
| 3 | 上流監視 | **PASS** | PR #32582 は本文未取得（タイトルのみ）。**tiled の NEON 実装が既に上流にあることを発見**（§4.2） |
| 4 | 残課題 | **未実施** | 端末不達 |

### §6.2 分かったこと（要約）

1. **tiled の ARM 開放は 2 行の gate 変更で済む。** ドライバ・unpacker・スカラー
   microtile はすべて元からコンパイルされていた。既定 OFF にしたので ARM の
   デフォルト動作は不変。
2. **NEON 実装は SDOT で素直に書ける。** AVX2 の act-bias / sign-trick が不要で、
   デバイアスした重みをそのまま `vdotq_s32` に渡せる。`tiled_repack_src1` も不要。
3. **代数は 600 試行で一致、int32 の余裕は 175 倍。** 丸め順序の違いで hash は変わるが、
   それは想定内（指示どおり品質ゲートで判定する）。
4. **上流には既に NEON 実装がある**（第三者のパッチ、Neoverse N1 で 6.5 倍）。
   本実装の方針はそれと一致する。**ただし同一コードかは未確認。**
5. **x86 経路に退行はない。** gate 変更は共有ファイルに入っているが、
   `__ARM_NEON` は x86 で未定義なので x86 のコードは変わらない。
   objdump で `vpmaddubsw` 116 個・`sdot` 0 個を確認した。
   MSVC の `#pragma GCC unroll` 警告 8 件は**変更前から存在する**（AVX2 カーネル内、
   294-372 行）。NEON コードは `#if defined(__ARM_NEON)` の中なので MSVC では
   コンパイルされず、警告は増えていない。
6. **ホスト側の代数検証が実バグを 1 つ捕まえた。** BIAS を 128 に固定していたのが
   qK 系では誤り（正しくは 0）。実機に持っていく前に発見できた。

### §6.3 確定していないこと（正直に分ける）

- **実機で動くか**（クラッシュしないか、品質ゲートを通るか）は**未検証**。
- **速くなるか**は**未測定**。上流の 6.5 倍はサーバ向け ARM の値で、
  この端末で出るかは分からない。
- **`ne00 % 256` の実データでの成立**は、型の性質から成立するはずだが実機で見ていない。
- **PR #32582 の内容**は本文未取得のため不明。

---

## §7 安全条件の遵守

| 条件 | 遵守 |
| --- | --- |
| 端末を再起動しない | **該当なし**（端末に触れていない） |
| 前面アプリを止めない | **該当なし**（run なし） |
| メイン worktree の差分保護 | **遵守**（開始時 9、終了時 9） |
| 隔離 worktree・専用ブランチ | **遵守**（`codex/c60-tiledarm`） |
| merge・force push・過去コミット改変 | **なし** |
| push は `codex/c60-tiledarm` のみ | **遵守** |
| 上流の取り込み禁止 | **遵守**（検索のみ、取り込みなし） |
| 未検証を PASS と書かない | **遵守**（Stage A/B とも「実装完了 / 実機未検証」と明記） |
| ビルドは sha256 記録 | **遵守**（2 ビルドの sha256 を記録） |
| ggml 本体の恒久変更をしない | **tiled の 2 ファイルのみ**（Phase 60 の目的そのもの）。他ディレクトリ・HTP 側は未変更 |

---

## §8 終了時の状態

| 項目 | 値 |
| --- | --- |
| **skel の最終 sha256** | **未確認**（端末不達。Phase 59 終了時は `76dd89e3…`） |
| **CPU バイナリ（端末上）** | **未変更**（端末に触れていない。Phase 59 終了時は `0a3db9c5…`） |
| 新ビルド（ホスト） | `build/c60-scalar` = `eba4af55…`、`build/c60-neon` = `c5499c18…` |
| replay | 未変更（端末に触れていない） |
| プロセス | 該当なし |
| 前面アプリ停止 | 0 回（run なし） |
| 再起動 | なし（端末に触れていない） |
| メイン worktree | 9 ファイルのまま保護 |

---

## §9 成果物

| 種別 | パス |
| --- | --- |
| 報告書 | `continuations/c60/phase60-report.md` |
| NEON カーネル | `ggml/src/ggml-cpu/tiled/tiled-kernel.cpp` |
| ARM gate | `ggml/src/ggml-cpu/tiled/tiled.cpp` |
| 代数検証 | `continuations/c60/p60_neon_algebra.cpp` |
| 端末用スクリプト（未実行） | `continuations/c60/p60_tiled.sh` |

### §9.1 再現コマンド

```
# Stage A ビルド（スカラー、dotprod なし）
cmake -B build/c60-scalar -G Ninja -DCMAKE_TOOLCHAIN_FILE=<ndk>/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON -DGGML_HEXAGON=OFF -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=ON \
  -DGGML_CPU_REPACK=ON -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8-a \
  -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF -DGGML_BUILD_TOOLS=OFF
cmake --build build/c60-scalar --target ggml-cpu -j 8
# -> eba4af55457b2ad19f3199237f4974229dfa2fa7ac99e6906138d4b15659e16a

# Stage B ビルド（NEON + dotprod）
cmake -B build/c60-neon -G Ninja -DCMAKE_TOOLCHAIN_FILE=<ndk>/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON -DGGML_HEXAGON=OFF -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=ON \
  -DGGML_CPU_REPACK=ON -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm \
  -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF -DGGML_BUILD_TOOLS=OFF
cmake --build build/c60-neon --target ggml-cpu -j 8
# -> c5499c18431c2d236713408d33e6b38de1a676bb214e85fea0c1334cec7f2db5

# 代数検証（ホスト）
g++ -O2 -o p60_neon_algebra p60_neon_algebra.cpp && ./p60_neon_algebra
# -> RESULT PASS（1000 試行）

# 命令の確認
llvm-objdump -d --no-show-raw-insn build/c60-neon/bin/libggml-cpu.so | grep -c sdot
# -> 1216
```

### §9.2 コミット（`codex/c60-tiledarm`、push 済み）

| commit | 内容 |
| --- | --- |
| `8d595b941` | NEON dotprod microtile、ARM gate 開放（既定 OFF）、代数検証 |
| `0b2acdd55` | qK 系（HAS_MIN）も NEON で処理、BIAS をテンプレート引数に |
| `5d6a16e17` | 報告書（未測定を明記） |
| `f2ad8ea2c` | x86 経路に退行なしの確認 |
| `0b2acdd55` | qK 系（HAS_MIN）も NEON で処理、BIAS をテンプレート引数に |
| `26fda0beb` | 報告書に qK 対応と BIAS バグの経緯 |
| `0df93a6d5` | 最終 hash、MSVC 警告は既存 |
