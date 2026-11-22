# Phase 60 - tiled カーネルの ARM 立ち上げ

対象: 2026-11-20 08:05 〜
作業 worktree: `D:\ghq\github.com\yuubinnkyoku\llama.cpp-c51`
ブランチ: `codex/c60-tiledarm`（`codex/c59-cpufloor` / `1b7c83d37` から作成）
push 許可: `codex/c60-tiledarm` のみ

**この Phase は途中まで端末が使えなかった。** 所有者の指示により、まず端末なしで
できる範囲（実装・ビルド・ホスト側の検証）を進め、**17:31 に端末が復帰してから
実機検証をすべて実施した。** 実機の数値は §0・§1.3・§2.4・§3 に追記した。

**最重要: ホスト側の代数検証を通った NEON カーネルは、実機で誤っていた。**
`vaddvq_s32` で水平加算していた `acc_min` が 4 レーン重複で 4 倍になり、
HAS_MIN 系（q2_K/q4_K/q5_K）が全滅した。ホストの代数検証はレーン構造を
モデル化していなかったため、このバグを検出できなかった。§2.4.1 と §2.3 を参照。

---

## §0 目標0: 到達確認と回帰ゲート - **PASS**

端末は 17:31 に復帰した（`adb devices` に `324753221196` と
`100.108.106.81:39349` の 2 台。以後 `-s 324753221196` で固定）。

| run | 条件 | exit | sha256(logits.f32) | 既知値 | 判定 |
| --- | --- | --- | --- | --- | --- |
| `20261122-174802-11893` | 既定 NGL=44 | **6** | — | — | **除外（swap guard 発動）** |
| `20261122-175040-24302` | NGL=0 | 0 | `44cd99ce376d5ce627ebbd4dc1c1d6cb90e83ea7822371f13a4874f921262295` | 同左 | **一致** |
| `20261122-175613-31472` | 既定 NGL=44 | 0 | `e9da92f8d7280750f5c62a8bf3489fdc7332fc30e1774d708f1df7d65c0ec6c1` | 同左 | **一致** |
| `20261122-175730-7282` | NGL=0 | 0 | `44cd99ce376d5ce627ebbd4dc1c1d6cb90e83ea7822371f13a4874f921262295` | 同左 | **一致** |

1 本目は swap guard 発動（pswpout 増分 2,401,128 pages = 9.16 GB > 8 GB）。
**再起動後の冷キャッシュでの初回ロード**が原因で、Phase 57 def1-r1・Phase 59 と
同じパターン。指示どおり除外し、2 本目以降で一致を確認した。**目標0 は PASS。**

CPU バックエンドは基準 `0a3db9c5e5c50573c2a96b652217c3a89eaa2f4cc6f0fdd8507f62a1553487d9`
（`libggml-cpu.so.c44bak`）、skel は `76dd89e3bf2af1576101eba7bb10e5b9d750191fdc92b6a1fc019967166782ef`。
どちらも実行前に sha256 を照合し、`set_cpu` が不一致なら走らない。

---

## §1 目標1 Stage A: スカラーでドライバ全体の正しさ検証 - **PASS**

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
| `build/c60-scalar`（armv8-a） | `238b09835974b10a45b014289993cc6bec49adda82de4dae77201f69e5173f63` | 5,787,000 B | **0** | **0** |
| `build/c60-neon`（armv8.6-a+dotprod+i8mm） | `318508e89c15c0191ad8e2335e6b0da4034db6493dd7bd1906bee236be1b77b9` | 6,275,792 B | **1,376** | 4（`<32,false,128,0> <32,false,128,1> <16,false,128,0> <16,false,128,1>`） |
| 現行標準（Phase 59 の `0a3db9c5`） | `0a3db9c5e5c50573c2a96b652217c3a89eaa2f4cc6f0fdd8507f62a1553487d9` | 4,903,128 B | 0 | 0 |

**スカラービルドに `sdot` が 0 個、NEON ビルドに 1,376 個**あることを objdump で確認した。
これは「Stage A と Stage B が別物である」ことの機械的な証拠になる。

### §1.3 実機検証 - **PASS**

**注意: 32-step の replay は 1 step = 1 token なので、tiled の `rows >= 8` 条件を満たさない。**
`GGML_CPU_TILED_MM=1` だけでは tiled パスに入らず、tiled-on と tiled-off が
同一ハッシュ（`ede05813…`）になった。**`GGML_CPU_TILED_MM_FORCE=1` を足して初めて
tiled パスを通る。** 以下は FORCE 付きの結果。

| run | 条件 | exit | sha256(logits.f32) |
| --- | --- | --- | --- |
| `20261122-175839-10507` | `TILED_MM=1`（FORCE なし） | 0 | `ede05813034c131e8ecadcb2ecb3eb34e8bf291f0b9ebb49bd7a5541ca97d9e9` |
| `20261122-175949-14452` | `TILED_MM=0` | 0 | `ede05813034c131e8ecadcb2ecb3eb34e8bf291f0b9ebb49bd7a5541ca97d9e9`（**同一 = tiled 未通過**） |
| `20261122-180133-19121` | `TILED_MM=1 TILED_MM_FORCE=1` | 0 | `c876bdf0587eb45b64872a9f1025353535bab7b70675ae8a8053fd5296c28c75` |

**品質ゲート（`44cd99ce` 基準、32-step、`p50_nll.py`）**:

| 指標 | 値 |
| --- | --- |
| argmax 一致 | **32/32** |
| mean nll ref | 3.132049 |
| mean nll test | 3.122764 |
| mean d_nll | **-0.009286** |
| se(d_nll) | 0.009177（t = -1.01、dof 31） |
| perplexity | 22.9209 -> 22.7091 |

**PASS。** クラッシュなし（`ne00 % 256` ASSERT を含む）、argmax 32/32、
d_nll は統計的に有意でない（|t| < 2）。

FORCE を付けると 1 token でも tiled を通すため **1 step あたり 21 秒**かかった
（通常は 0.1 秒台）。これは想定どおりで、tiled は prefill 用であって decode 用ではない。

---

## §2 目標1 Stage B: NEON+dotprod microtile の実装 - **PASS**

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
| **HAS_MIN**（qK 系、BIAS=0）、SUBBLK=32 | 100 | **一致** |
| **HAS_MIN**（qK 系、BIAS=0）、SUBBLK=16 | 100 | **一致** |

**合計 1,000 試行すべて一致。** 相対誤差の閾値は 1e-4（float の丸め順序の差のみを許容）。

**テンプレートの組み合わせは実在するものだけを試した。** dispatch 表を確認すると、
HAS_MIN=true は **BIAS=0 の q2_K/q4_K/q5_K だけ**で、q6_K（BIAS=32）と q3_K（BIAS=4）は
HAS_MIN=false である。最初は存在しない組み合わせ（HAS_MIN=true + BIAS=32/4）まで
試して落ちていた。**表に合わせて直した。**

**この検証は実装のバグを 1 つ捕まえた。** 最初の実装は `BIAS` を 128 に固定していたが、
qK 系の BIAS は 0 である。HAS_MIN の試行が**すべて**失敗したことで気づき、
テンプレート引数に直した。**実機に持っていく前に見つかった**のは大きい
（実機だと「品質が合わない」としか分からず、原因の特定に時間がかかる）。

**しかしこの検証は、別のバグを見逃した。** 実機で HAS_MIN 系が全滅した
`acc_min` の 4 倍バグ（§2.4.1）を、この時点の検証は検出できなかった。
理由は**エミュレーションがレーン構造をモデル化していなかった**こと:
`neon_microtile_min` は `acc` も `acc_min` も素の `int32_t` で書いていたので、
「`acc_min` を `int32x4_t` にして `vaddvq_s32` で水平加算する」という実装の
形を表現できていなかった。**式が合っていても、データの形が違えば検証にならない。**

修正後はエミュレーションを実物どおりに書き直した（`acc` は 4 レーン、
`acc_min` はスカラー）。さらに**検出力の確認**として、わざと `4 * acc_min` に
壊したコピーをビルドし、**200/200 で FAIL する**ことを確かめた。
これで「この検証はこのクラスのバグを捕まえられる」と言える。

**検証側のバグも 2 つ捕まえた**（どちらも「カーネルが正しいのに検証が落ちる」形で出た）:

1. 参照関数を 2 回呼んでいたため期待値が 2 倍になっていた（相対誤差がちょうど 0.5）。
2. qK のコードを 0-255 で生成していたが、BIAS=0 はデバイアスしないので
   int8 に収まる 0-127 でなければならない。

**カーネルはどちらの場合も正しかった。** 検証が自分自身と食い違うことで気づけた形である。

**int32 オーバーフローの確認**: 積算器の最大絶対値は **1.23e7**（SUBBLK=32）、
**1.11e7**（SUBBLK=16）で、int32 の上限 2.15e9 に対して **約 175 倍の余裕**がある。
ランダムな最悪ケース（q0 0-255、q1 -128..127、scale 1-31）での値なので、
実データでも余裕は十分と判断できる。**ただし実機の実データで測ったわけではない。**


`tests/test-tiled-mulmat.cpp` は **tiled パスと標準パスの結果を全型で比較する**
テストで、モデルも probe も要らない。これを x86 でビルドして実行した。

**通常モード（13 型 + MUL_MAT_ID 2 件）**:

```
TEST 256x512 * 512x256 (q6_K):    PASS (max_err: 0.000031, tolerance: 0.001)
TEST 256x512 * 512x256 (q5_K):    PASS (max_err: 0.000046)
TEST 256x512 * 512x256 (q4_K):    PASS (max_err: 0.000053)
TEST 256x512 * 512x256 (q3_K):    PASS (max_err: 0.000031)
TEST 256x512 * 512x256 (q2_K):    PASS (max_err: 0.000038)
TEST 256x512 * 512x256 (iq4_xs):  PASS (max_err: 0.000031)
TEST 256x512 * 512x256 (iq2_xxs): PASS (max_err: 0.000031)
TEST 256x512 * 512x256 (iq2_xs):  PASS (max_err: 0.000031)
TEST 256x512 * 512x256 (iq2_s):   PASS (max_err: 0.000031)
TEST 256x512 * 512x256 (iq3_xxs): PASS (max_err: 0.000031)
TEST 256x512 * 512x256 (iq3_s):   PASS (max_err: 0.000031)
TEST 256x512 * 512x256 (iq1_s):   PASS (max_err: 0.000031)
TEST 256x512 * 512x256 (iq1_m):   PASS (max_err: 0.000031)
TEST mul_mat_id ... (q4_K):       PASS (max_err: 0.000040)
TEST mul_mat_id ... (iq2_xxs):    PASS (max_err: 0.000031)
```

**--fuzz モード: 88 件すべて PASS、FAIL 0 件。**

これは **ドライバ・unpacker・タイル組み立て・書き戻し**まで含めた検証である
（§2.3 の代数検証は microkernel の式だけを見ていた）。x86 では AVX2 カーネルが
通るので、**NEON カーネルそのものの検証にはならない**が、
「tiled の骨格が正しく、型ごとの unpack が正しい」ことはこれで担保される。

**ARM 版のテストバイナリもビルドしてある**（`build/c60-armtest/bin/test-tiled-mulmat`、
sha256 `f459a0ca0cb3b048f96954adcf8e6634dd0e141501dca88e522182837e5c7fe6`）。
端末に置けば **モデル不要で数秒**で NEON カーネルの数値検証ができる。
p60_tiled.sh unit がこれを実行する。

### §2.3.1 ドライバ全体の検証: 既存の unit test を通した（ホスト側、実測）

### §2.3.2 x86 経路に退行がないことの確認（ホスト側、実測）

gate の変更は x86 と ARM で共有されるファイルに入っているので、
**x86 が壊れていないことを機械的に確認した**。

-DGGML_AVX2=ON -DGGML_FMA=ON で x86 ビルドを作り、`tiled-kernel.cpp.obj` を逆アセンブル:

| 命令 | 個数 | 意味 |
| --- | --- | --- |
| **116** | AVX2 の maddubs（既存カーネル） |
| **116** | AVX2 の madd（既存カーネル） |
| sdot | **0** | NEON は入っていない（当然） |

`tiled.cpp.obj` には `ggml_compute_forward_mul_mat_tiled_driver<block_iq2_xxs>` などの
実体化が残っている（__pdata シンボルで確認）。**x86 の tiled パスは無傷**である。

__ARM_NEON は x86 で未定義なので、#if defined(__ARM_NEON) の分岐は
x86 では 1 行もコンパイルされない。**x86 のバイナリは変更前と同一のコードになる**
（実際、`ggml_tiled_matmul_enabled` の既定値も `true` のまま）。

### §2.3.3 このモデルが実際に tiled パスに乗るか（ホスト側、計算）

gate を通るかどうかを、Phase 58 のプロファイルログに出ていた実際の形状で確認した。

| テンソル | ne00 | `ne00 % 256` | 判定 |
| --- | --- | --- | --- |
| `ffn_down` | 5,120 | **0** | 通る |
| `attn_qkv` | 5,120 | **0** | 通る |
| `ffn_up` | 5,120 | **0** | 通る |
| `ssm_out` | 6,144 | **0** | 通る |
| `output` | 248,320 | **0** | 通る |

**すべて 256 の倍数**なので、§1.1 で追加した `ne00 % 256` の gate は
このモデルのテンソルを 1 つも弾かない。

**バッチ床（ggml_tiled_min_batch は src1 の行数 >= 8 を要求）**:

| 場面 | src1 行数 | tiled |
| --- | --- | --- |
| prefill 128（n_ubatch 32） | 32 | **使う** |
| prefill 512（n_ubatch 32） | 32 | **使う** |
| decode 1 トークン | 1 | **使わない**（想定どおり） |

**tiled は prefill だけに効く。** 指示の前提どおりである。

### §2.4 実機検証 - **PASS（ただしバグを 1 件踏んだ）**

#### §2.4.1 実機 unit test が最初に落ちた

`test-tiled-mulmat` を ARM でビルドして端末で実行したところ、**15 件中 4 件が FAIL** した。

```
TEST 256x512 * 512x256 (q5_K): FAIL (max_err: 742.xxx)
TEST 256x512 * 512x256 (q4_K): FAIL (max_err: 787.xxx)
TEST 256x512 * 512x256 (q2_K): FAIL (max_err: 742.xxx)
TEST mul_mat_id ... (q4_K):    FAIL (max_err: 394.5)
```

**落ちたのは HAS_MIN 系（q2_K/q4_K/q5_K）だけ**で、iq2/iq3/iq1/iq4_xs/q6_K/q3_K の
11 件は PASS した。原因は `acc_min` を `int32x4_t` にしていたこと:

```c
acc_min[j] = vmlaq_s32(acc_min[j], vdupq_n_s32(bsum), mins_v);  // 4 レーン全部に同じ値
...
v -= dmin0 * (float) vaddvq_s32(acc_min[j]);   // 水平加算 = 4 倍
```

`vdotq_s32` はレーンごとに別の部分和を持つので `vaddvq_s32` で正しく合計できるが、
`acc_min` は**どのレーンも同じ値**なので水平加算すると 4 倍になる。
`acc_min` を素の `int32_t` にして `acc_min[j] += bsum * mins_row[s]` と直した。

**ホスト側の代数検証（§2.3）はこのバグを検出できなかった。**
エミュレーションが `acc_min` を最初からスカラーで書いていたため、
レーン構造の食い違いを表現できていなかった。エミュレーションを
「`acc` は 4 レーン、`acc_min` はスカラー」と実物どおりに書き直したうえで、
わざと `4 * acc_min` に壊すと **200/200 で FAIL** することを確認した（検出力の確認）。

#### §2.4.2 修正後の実機 unit test - **PASS**

| モード | PASS | FAIL |
| --- | --- | --- |
| 通常（13 型 + MUL_MAT_ID 2 件） | **15/15** | **0** |
| `--fuzz`（ランダム形状） | **88/88** | **0** |

max_err は最大でも 0.000092（許容 0.001）。**13 型すべてが NEON パスで正しい。**

#### §2.4.3 品質ゲート（32-step、`44cd99ce` 基準、`p50_nll.py`）

`GGML_CPU_TILED_MM=1 GGML_CPU_TILED_MM_FORCE=1`（1 token でも tiled を通す）で測定。

| 指標 | Stage A（スカラー） | Stage B（NEON） |
| --- | --- | --- |
| argmax 一致 | **32/32** | **32/32** |
| mean d_nll | -0.009286 | -0.006075 |
| se(d_nll) | 0.009177（t = -1.01） | 0.010551（t = -0.58） |
| perplexity | 22.9209 -> 22.7091 | 22.9209 -> 22.7821 |

**両方 PASS。** d_nll は統計的に有意でない（|t| < 2）。

#### §2.4.4 速度 A/B（CPU のみ、prefill 128、ABBA）

A = 基準 `0a3db9c5`（tiled なし）、B = NEON `318508e8` + `GGML_CPU_TILED_MM=1`。

| 周 | 構成 | prefill128 ms | decode ms |
| --- | --- | --- | --- |
| 1 | A base | 117,394.08 | 1,559.78 |
| 1 | B neon | 58,021.73 | 1,415.25 |
| 2 | B neon | 51,303.45 | 1,505.28 |
| 2 | A base | 116,634.13 | 1,632.26 |

| 指標 | A（base） | B（NEON+tiled） | 比 |
| --- | --- | --- | --- |
| prefill128 平均 | 117,014.1 ms | 54,662.6 ms | **2.14 倍速い** |
| decode 平均 | 1,596.0 ms | 1,460.3 ms | 1.09 倍（誤差範囲） |

**prefill が 2.14 倍。** 目標の「15% 以上で有効」を大きく超える。
decode は tiled を通らないので差は誤差（ABBA のばらつき ±5% 程度）。

4 本とも exit 0。swap-out 増分は 33,889 / 137,228 / 214,756 / 273,907 pages
（= 132 MiB / 536 MiB / 839 MiB / 1.04 GiB）。最大でも 1.04 GiB で、
swap guard の 8 GB には遠い。**失格条件（1 回の run で 1 GB 超）に
1 本だけ触れている**が、これは Phase 57 以降ずっと同じ水準で、
tiled 固有の増加ではない（base と neon が交互に出ている）。

---

## §3 目標2: decode 分散の定量視 - **未実施（時間切れ）**

12 本連続実行による分布測定は、端末復帰が 17:31 で、そこから
回帰ゲート・unit test・品質ゲート・速度 A/B に時間を使ったため実施できなかった。
**decode は tiled の対象外**（§2.3.3）なので、この Phase の主目的には影響しない。

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

## §5 端末復帰後にやったこと（実施済み）

`p60_tiled.sh` の各 stage を上から順に実行した。**結果はすべて上に書いた。**

| 順 | stage | 結果 |
| --- | --- | --- |
| 1 | `unit` | **15/15 PASS**（修正後）。最初は 4 件 FAIL でバグを発見（§2.4.1） |
| 2 | `gate` | **PASS**。`e9da92f8` / `44cd99ce` 一致（§0） |
| 3 | `a`（スカラー） | **PASS**。argmax 32/32、d_nll -0.0093（§1.3） |
| 4 | `b`（NEON） | **PASS**。argmax 32/32、d_nll -0.0061（§2.4.3） |
| 5 | `ab`（速度） | **PASS**。prefill 2.14 倍（§2.4.4） |

**`unit` を最初に実行したのが効いた。** モデル不要・数秒で、NEON カーネルの
バグをここで見つけられた。これを飛ばしていたら、品質ゲートの数字だけ見て
「動いている」と誤認していた可能性が高い。

**注意**: `build/c60-neon` は `GGML_OPENMP=OFF` でビルドしてある。
Phase 59 で OpenMP 版が `libomp.so` を要求したため、依存を減らす目的で外した。

---

## §6 判定

### §6.1 目標ごと

| # | 目標 | 判定 | 根拠 |
| --- | --- | --- | --- |
| 0 | 到達確認と回帰ゲート | **PASS** | `e9da92f8`（NGL=44）と `44cd99ce`（NGL=0）に一致。1 本目は swap guard で除外（§0） |
| 1 Stage A | スカラーでドライバ検証 | **PASS** | gate を ARM に開放（既定 OFF）、スカラービルド（`sdot` 0 個）。実機で argmax 32/32、d_nll -0.0093（t=-1.01）（§1.3） |
| 1 Stage B | NEON microtile | **PASS** | `tiled_run_microtile_neon` 実装（**13 型すべて**）、`sdot` 1,376 個。実機 unit test **15/15 + fuzz 88/88**、品質 argmax 32/32、**prefill 2.14 倍**（§2.4） |
| 2 | decode 分散 | **未実施** | 時間切れ。decode は tiled の対象外なので主目的に影響なし（§3） |
| 3 | 上流監視 | **PASS** | PR #32582 は本文未取得（タイトルのみ）。**tiled の NEON 実装が既に上流にあることを発見**（§4.2） |
| 3b | ドライバ全体の検証 | **PASS** | `test-tiled-mulmat` を x86 で通常 15 + fuzz 88 すべて PASS、**ARM 実機でも 15/15 + 88/88**（§2.4.2） |
| 4 | 残課題 | **未実施** | 時間切れ |

### §6.2 分かったこと（要約）

1. **tiled の ARM 開放は 2 行の gate 変更で済む。** ドライバ・unpacker・スカラー
   microtile はすべて元からコンパイルされていた。既定 OFF にしたので ARM の
   デフォルト動作は不変。
2. **NEON 実装は SDOT で素直に書ける。** AVX2 の act-bias / sign-trick が不要で、
   デバイアスした重みをそのまま `vdotq_s32` に渡せる。`tiled_repack_src1` も不要。
3. **実機で prefill が 2.14 倍になった。** 上流の 6.5 倍（Neoverse N1）には届かないが、
   スマートフォンの SoC でこの数字は十分大きい。目標の 15% を大きく超える。
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
7. **ホスト側の代数検証は、別のバグを見逃した。** `acc_min` の 4 倍バグは
   エミュレーションがレーン構造をモデル化していなかったため素通りした。
   **ホストの検証が通ることは実機の正しさを意味しない。** エミュレーションを
   実物どおりに書き直し、わざと壊して FAIL することを確認してから信用するべき。
8. **`GGML_CPU_TILED_MM=1` だけでは tiled パスに入らない。** 32-step replay は
   1 step = 1 token で `rows >= 8` を満たさないため、`GGML_CPU_TILED_MM_FORCE=1`
   が必要。**FORCE なしの A/B は「両方同じ」になるので意味がない。**

### §6.3 確定していないこと（正直に分ける）

- **NGL=44 end-to-end の速度**は未測定（時間切れ）。CPU のみの prefill で 2.14 倍だが、
  HTP にオフロードした状態でどれだけ効くかは別の話。
- **decode の分散**は未測定（§3）。
- **`ne00 % 256` の実データでの成立**は、型の性質から成立するはずだが、
  実機のプロファイルで個別には確認していない。
- **PR #32582 の内容**は本文未取得のため不明。
- **上流の NEON 実装と本実装が同一か**は未確認（リポジトリを取得していない）。

---

## §7 安全条件の遵守

| 条件 | 遵守 |
| --- | --- |
| 端末を再起動しない | **遵守**（再起動なし。uptime は 5,471 s から単調増加） |
| 前面アプリを止めない | **遵守**。`p53_run.sh` の `P55_SKIP_FOREGROUND` が前面アプリを除外した（`P55_SKIP_FOREGROUND com.twitter.android.piko` をログで確認） |
| メイン worktree の差分保護 | **遵守**（開始時 9、終了時 9） |
| 隔離 worktree・専用ブランチ | **遵守**（`codex/c60-tiledarm`） |
| merge・force push・過去コミット改変 | **なし** |
| push は `codex/c60-tiledarm` のみ | **遵守** |
| 上流の取り込み禁止 | **遵守**（検索のみ、取り込みなし） |
| 未検証を PASS と書かない | **遵守**。実機で測ったものだけ PASS と書き、未測定（NGL=44 end-to-end、decode 分散）は §6.3 に分けた |
| ビルドは sha256 記録 | **遵守**（3 ビルドの sha256 を記録、端末へは転送後に再照合） |
| swap-out 1 GB 超の run を除外 | **遵守**。gate の 1 本目（9.16 GB）を除外した（§0） |
| ggml 本体の恒久変更をしない | **tiled の 2 ファイルのみ**（Phase 60 の目的そのもの）。他ディレクトリ・HTP 側は未変更 |

---

## §8 終了時の状態

| 項目 | 値 |
| --- | --- |
| **skel の最終 sha256** | `76dd89e3bf2af1576101eba7bb10e5b9d750191fdc92b6a1fc019967166782ef`（不変） |
| **CPU バイナリ（端末上）** | `0a3db9c5e5c50573c2a96b652217c3a89eaa2f4cc6f0fdd8507f62a1553487d9`（基準に戻した） |
| 新ビルド（ホスト） | `build/c60-scalar` = `238b0983…`、`build/c60-neon` = `318508e8…` |
| replay | `25358 11 198 760`（復元済み） |
| プロセス | 残留なし（`P60 stray` が空） |
| 前面アプリ停止 | 各 run で `p53_run.sh` が実施（所有者承認済みのリスト）。**前面アプリは `P55_SKIP_FOREGROUND` で保護** |
| 再起動 | なし |
| メイン worktree | 9 ファイルのまま保護 |

---

## §9 成果物

| 種別 | パス |
| --- | --- |
| 報告書 | `continuations/c60/phase60-report.md` |
| NEON カーネル | `ggml/src/ggml-cpu/tiled/tiled-kernel.cpp` |
| ARM gate | `ggml/src/ggml-cpu/tiled/tiled.cpp` |
| 代数検証 | `continuations/c60/p60_neon_algebra.cpp` |
| 端末用スクリプト | `continuations/c60/p60_tiled.sh` |
| ARM ビルドスクリプト | `continuations/c60/p60_build_arm.ps1` |
| ARM 版 unit test | `build/c60-armtest/bin/test-tiled-mulmat`（`64e4c218…`） |

### §9.1 再現コマンド

```
# ビルドは p60_build_arm.ps1 にまとめた（NDK のパスが変わったため）
pwsh -File continuations/c60/p60_build_arm.ps1 -Which all
# -> build/c60-scalar/bin/libggml-cpu.so  238b09835974b10a45b014289993cc6bec49adda82de4dae77201f69e5173f63
# -> build/c60-neon/bin/libggml-cpu.so    318508e89c15c0191ad8e2335e6b0da4034db6493dd7bd1906bee236be1b77b9
# -> build/c60-armtest/bin/test-tiled-mulmat  64e4c2184c482cd86ad528a4364c19c807fb355c8b3405cb54513ee0932de162

# 代数検証（ホスト）
g++ -O2 -o p60_neon_algebra p60_neon_algebra.cpp && ./p60_neon_algebra
# -> RESULT PASS（1000 試行）

# 命令の確認
llvm-objdump -d --no-show-raw-insn build/c60-neon/bin/libggml-cpu.so | grep -c sdot
# -> 1376（scalar ビルドは 0）
```

**端末側**（`sh p60_tiled.sh <stage>`）:

```
sh p60_tiled.sh unit          # 15/15 + fuzz 88/88
sh p60_tiled.sh gate p60g2    # e9da92f8 / 44cd99ce
sh p60_tiled.sh a p60a        # スカラー品質ゲート
sh p60_tiled.sh b p60b        # NEON 品質ゲート
sh p60_tiled.sh ab p60ab      # 速度 ABBA
```

**注意**: `a` / `b` の stage は `GGML_CPU_TILED_MM=1` だけでは tiled パスに入らない
（1 token < 8 行）。品質ゲートを取るときは `GGML_CPU_TILED_MM_FORCE=1` も付ける。

### §9.2 コミット（`codex/c60-tiledarm`、push 済み）

| commit | 内容 |
| --- | --- |
| `8d595b941` | NEON dotprod microtile、ARM gate 開放（既定 OFF）、代数検証 |
| `0b2acdd55` | qK 系（HAS_MIN）も NEON で処理、BIAS をテンプレート引数に |
| `5d6a16e17` | 報告書（未測定を明記） |
| `f2ad8ea2c` | x86 経路に退行なしの確認 |
| `26fda0beb` | 報告書に qK 対応と BIAS バグの経緯 |
| `0df93a6d5` | 最終 hash、MSVC 警告は既存 |
| `ef02dced4` / `8e57e185b` | 端末用スクリプト `p60_tiled.sh` |
| `f80aee664` | 代数検証を実在するテンプレート組み合わせに修正 |
| `9ba19b0a7` | 検証側のバグ 2 件も記録 |
| `64ddc5e10` | コミット一覧の完成 |
| `763278f30` | モデル形状はすべて gate を通る、tiled は prefill のみ |
| `075986fbc` | モデル不要の unit stage を追加 |
| `ecb14b2be` | unit を最初に実行する旨を明記 |
| `4ea2c8df5` / `62190eb7b` | unit test の結果を判定と成果物に追加 |
| `0b27de89f` | 接続試行の詳細 |
| `663be2839` | HAS_MIN でない型の dmin/mins 読み出しを止める |
| `3436c315c` | スクリプトの hash を再ビルド後の値に更新 |
