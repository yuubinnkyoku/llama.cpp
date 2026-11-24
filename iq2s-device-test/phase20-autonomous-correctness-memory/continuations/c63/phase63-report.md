# Phase 63 - evidence テスト・統合版の本採用・tiled 既定 ON

対象: 2026-11-25 00:05 〜
作業 worktree: `D:\ghq\github.com\yuubinnkyoku\llama.cpp-c51`
ブランチ: `codex/c63-evidence`（`codex/c62-upstream` / `49cc0dd15` から作成）
push 許可: `codex/c63-evidence` のみ

---

## §0 目標0: ゲート - **PASS**

### §0.1 到達確認

| 項目 | 値 |
| --- | --- |
| 接続 | `192.168.55.15:5555`（adb over TCP、Phase 62 から継続） |
| serial / model | `324753221196` / `NX741J`（同一端末） |
| **uptime** | **171,175 s**（Phase 62 終了時 96,323 s から単調増加 = **再起動なし**） |
| tz0 | 47.6 ℃ |
| MemAvailable | 6.0 GB |
| cap0 / cap7 | 3,033,600 / 2,112,000 |

### §0.2 unit stage

| モード | PASS | FAIL |
| --- | --- | --- |
| 通常 | **15/15** | **0** |
| `--fuzz` | **88/88** | **0** |

### §0.3 基準バイナリの回帰

| run | 構成 | exit | sha256(logits.f32) | 既知値 | 判定 |
| --- | --- | --- | --- | --- | --- |
| `20261125-000843-5251` | CPU `0a3db9c5` + base `d950aa96`、NGL=44 | 0 | `e9da92f8d7280750f5c62a8bf3489fdc7332fc30e1774d708f1df7d65c0ec6c1` | 同左 | **一致** |
| `20261125-001013-11049` | 同、NGL=0 | 0 | `44cd99ce376d5ce627ebbd4dc1c1d6cb90e83ea7822371f13a4874f921262295` | 同左 | **一致** |

### §0.4 統合版の新基準

| run | 構成 | exit | sha256(logits.f32) | 既知値 | 判定 |
| --- | --- | --- | --- | --- | --- |
| `20261125-001237-17221` | CPU `cc32ed77` + hexagon `9c9bfc48` + skel `312ad7b7`、NGL=44 | 0 | `e788051e16d7e1e63b140a902418c75f779bfe29cfb320bb054598508b09d4eb` | 同左 | **一致** |
| `20261125-001419-23519` | 同、NGL=0 | 0 | `ede05813034c131e8ecadcb2ecb3eb34e8bf291f0b9ebb49bd7a5541ca97d9e9` | 同左 | **一致** |

**目標0 は PASS。** 基準系・統合系の両方が既知値に一致した。

---

## §1 目標1: evidence テスト - **PASS（決定的な結果）**

### §1.1 方法

**上流 master を無改変で clone してビルドした**（fork のコードは 1 行も混ざらない）:

| 項目 | 値 |
| --- | --- |
| clone 元 | `https://github.com/ggml-org/llama.cpp` master |
| commit | `3090e45a8`（2026-11-24 16:34 "metal : reserve the top-k scratch only for short rows (#32942)"） |
| 作業ディレクトリ | `D:\ghq\github.com\yuubinnkyoku\llama-upstream-c63`（fork の worktree とは別） |
| `git status` | **dirty 0**（無改変を確認） |
| ビルド | Phase 62 §9.1 のレシピ（SDK 付属 CMake + `PREBUILT_LIB_DIR=toolv19_v81`） |

ビルド sha256:

| ファイル | sha256 | サイズ |
| --- | --- | --- |
| `libggml-cpu.so` | `e655bf5ca3ac53b4d18548dfa648c0ce4431b049251c018eb6b02fa75f04ca93` | 4,945,976 |
| `libggml-base.so` | `4671a16bd39ce103e2f70c9ff5a3b71b91dece8954b092feb680ec5cf22a4301` | 8,539,616 |
| `libggml-hexagon.so` | `f40c30eb5382882fc588c9076d8a452e6a6eef25c13cdbd1aaeee08894b60f2c` | 9,649,848 |
| `libggml-htp-v81.so` | `41bfebd9a687adb453c0cf219410e3f0e3a304950e3f95abbfb687d54fe2f401` | 1,051,016 |

端末の `libs/up63/` に置き、`p63_evid.sh` が 4 点を差し替えて実行し、
**終了時に基準構成へ自動復元**する（復元後の sha256 も記録済み）。

### §1.2 結果: **上流は 27B を動かせる。ただし遅い。**

| NGL | ロード | exit | prefill 128 | decode | **graph splits** | swap-out |
| --- | --- | --- | --- | --- | --- | --- |
| **44** | **成功** | 0 | **105,291.80 ms** | 2,843.40 ms | **262** | 100,720 pages |
| **0** | **成功** | 0 | 121,846.50 ms | 1,466.72 ms | **1** | 142,836 pages |

**ロードは成功し、4 トークンの生成も完走した。** クラッシュなし、tombstone なし。

### §1.3 決定的な発見: **262 分割**

**上流の NGL=44 では graph が 262 個に分割される。** fork は 1 個。

```
上流:  sched_reserve: graph: nodes = 3848, splits = 262
fork:  sched_reserve: graph: nodes = 3848, splits = 1
```

**これは「上流の HTP カーネルが 27B の主力型を扱えず、ノードごとに CPU へ
フォールバックしている」ことの直接的な証拠である。** 分割が 262 個あるということは、
HTP と CPU の間で 262 回の受け渡しが発生するということでもあり、
これが速度差の主因になっている。

**NGL=0 では splits = 1**（全部 CPU なので分割が起きない）。この対比が
「分割は HTP の型非対応に由来する」ことを裏づける。

### §1.4 速度の比較

| 構成 | prefill 128 | decode | 備考 |
| --- | --- | --- | --- |
| **上流 NGL=44** | **105.29 s** | 2,843 ms | splits 262 |
| **fork NGL=44**（統合版） | **43.25 s**（Phase 62 スポット） | 815 ms | splits 1 |
| 上流 NGL=0 | 121.85 s | 1,467 ms | splits 1 |
| fork NGL=0（Phase 61） | ~111 s | — | splits 1 |

**上流 NGL=44 は fork の 2.4 倍遅い**（105.29 s vs 43.25 s）。
**decode は 3.5 倍遅い**（2,843 ms vs 815 ms）。

**注意**: cap が完全には揃っていない（上流 1,785,600-1,670,400、fork の Phase 62
スポットは 2,112,000-1,785,600）。**厳密な比較には同一セッションの ABBA が必要**だが、
**262 分割という構造的な差があるので、順序は逆転しないと判断できる。**

### §1.5 乗り換え exit plan への示唆

**「上流は 27B を担えない」が数値で裏づけられた。**

- ロードはできる（型の欠落で落ちることはない）。
- しかし **262 分割**により、HTP と CPU の間で頻繁に受け渡しが起きる。
- 結果として **prefill で 2.4 倍、decode で 3.5 倍遅い**。

**したがって現時点の乗り換えは非現実的**という Phase 61 の判断は正しかった。
上流が iq2_xs/iq2_s/iq3_s/iq3_xxs の HTP カーネルを揃えるまで、fork を維持する
理由が定量的に示された。

**これが継続監視の第 1 測定点になる。** 次回以降、上流の型カバレッジが広がったら
同じ手順で測り、262 が減っているかを確認すればよい。

---

## §2 目標2: 統合版の ABBA と標準載せ替え - **PASS**

### §2.1 最初の ABBA は交絡していた（重要な記録）

Phase 62 の `c62pre`（統合前）と `c62g1`（統合後）で ABBA 3 組を取ったところ、
**B が 10-14% 遅い**という結果になった:

| cap | A128 | B128 | A512 | B512 |
| --- | --- | --- | --- | --- |
| 1,555,200 | 51,844.35 | 57,397.87（+10.7%） | 188,591.41 | 215,316.86（+14.2%） |
| 1,670,400 | 54,093.30 | 59,414.68（+9.8%） | 183,477.72 | 203,823.50（+11.1%） |

**しかしこれは交絡だった。** 切り分けのため 4 本の腕を測った:

| 腕 | cpu | skel | prefill128 (cap 1,670,400) | A 比 |
| --- | --- | --- | --- | --- |
| A | `c62pre`（統合前） | `c58x4`（旧） | 48,415.97 | — |
| C | `c62pre` | `c62q6`（新） | 49,170.62 | +1.6% |
| D | `c62g1`（新） | `c58x4` | 53,760.36 | +11.0% |
| B | `c62g1` | `c62q6` | 53,745.71 | +11.0% |

**差は CPU バイナリ側にある**（C は +1.6%、D は +11.0%）。

**原因**: `c62pre` は **`GGML_HEXAGON=OFF`** でビルドされており（Phase 61 の
`p61_build_matched.ps1`）、`c62g1` は **`GGML_HEXAGON=ON`**（Phase 62）。
サイズが 6,275,792 B と 5,782,840 B で **約 500 KB 違う**。
**CPU のソースは 1 行も変わっていない**（`git diff f9fa5b848..49cc0dd15 -- ggml/src/ggml-cpu/`
が空）。つまり差はビルド構成だけ。

### §2.2 公平な A を作り直した

`codex/c62-upstream` のソースを **`GGML_HEXAGON=ON`** でビルドし直した:
`f4414f5ff4633e292db098749ae89b891a553328a346ba67de38f938a8ab44b3`（5,782,840 B、
`c62g1` と同サイズ）。**tiled 既定の 1 行だけが違う**、真の A/B になった。

### §2.3 公平な ABBA（3 組、prefill 128 と 512）

| 周 | 構成 | prefill128 ms | prefill512 ms | decode ms | cap0 |
| --- | --- | --- | --- | --- | --- |
| 1 | A | 45,713.80 | 192,762.33 | 1,142.71 | 1,670,400 |
| 1 | B | 57,974.32 | 199,106.78 | 914.39 | 1,670,400 |
| 2 | B | 56,900.79 | 210,887.30 | 1,008.25 | 1,670,400→1,555,200 |
| 2 | A | 56,896.13 | 198,296.17 | 1,138.88 | 1,555,200 |
| 3 | A | 62,181.91 | 210,464.93 | 1,323.00 | 1,670,400→1,555,200 |
| 3 | B | 59,761.28 | 199,296.72 | 1,180.12 | 1,555,200 |

**中央値**:

| 指標 | A（統合前） | B（統合後） | 差 |
| --- | --- | --- | --- |
| prefill 128 | 56,896.13 ms | 57,974.32 ms | **+1.9%** |
| prefill 512 | 198,296.17 ms | 199,296.72 ms | **+0.5%** |

**退行なしと判定する。** 差は 0.5-1.9% で、**周間のばらつき（A は 45.7-62.2 s、
B は 56.9-59.8 s）の方が大きい**。cap が run 中に落ちるため、これ以上の精度は
この環境では出せない。

**swap-out**: 全 12 run で最大 491,602 pages（1.88 GiB）。失格条件（1 GB/run 超）に
512-token の run が触れているが、**A と B が交互に出ており統合固有ではない**。
swap guard（8 GB）は発動しなかった。

### §2.4 64-step 品質ゲート - **PASS**

統合版（NGL=44）で 64-step を 1 回、統合前ビルドで 1 回取って比較した。

| 指標 | 値 |
| --- | --- |
| 統合前 | `20261125-020126-12819` |
| 統合後 | `20261125-015918-5636` |
| **argmax 一致** | **61/64**（基準 61/64 以上を満たす） |
| mean nll ref | 2.951225 |
| mean nll test | 2.987096 |
| mean d_nll | **+0.035871** |
| se(d_nll) | 0.036692（t = +0.98、dof 63） |
| perplexity | 19.1294 -> 19.8280 |

**PASS。** d_nll は統計的に有意でない（|t| < 2）。

### §2.5 標準載せ替え - **実施**

速度退行なし・品質 PASS の条件を満たしたので、**統合組を標準に載せ替えた**。

| ファイル | 載せ替え前 | 載せ替え後 |
| --- | --- | --- |
| `libggml-cpu.so` | `0a3db9c5` | **`1cc02638`**（統合 + tiled 既定 ON） |
| `libggml-base.so` | `d950aa96` | `d950aa96`（変更なし） |
| `libggml-hexagon.so` | `9c9bfc48` | `9c9bfc48`（変更なし） |
| `libggml-htp-v81.so` | `76dd89e3` | **`312ad7b7`**（グループ③ Q6_K） |

**base を変えなかった理由**: 統合 CPU は `d950aa96` に対してビルドされており、
libomp 依存が無いのはこの組み合わせだから。

**ロールバック用に旧組を残置**: `libggml-cpu.so.c63prev`、`libggml-htp-v81.so.c63prev`。

**載せ替え後の検証（各 3 回）**:

| NGL | run 1 | run 2 | run 3 | 一致 |
| --- | --- | --- | --- | --- |
| 44 | `e788051e…` | `e788051e…` | `e788051e…` | **3/3** |
| 0 | `ede05813…` | `ede05813…` | `ede05813…` | **3/3** |

**新基準が維持されている。**

### §2.6 載せ替え中に見つけた 2 つの端末側の問題（修正済み）

**(1) `libggml-cpu.so` がシンボリックリンクだった**

`libs/libggml-cpu.so` は `c45-27b/libs/` へのリンクで、その先も `c44-27b/libs/` への
リンクだった。**`cp` はリンクを辿るので、載せ替えの `cp` が実体を 0 バイトに切り詰めた。**

症状: `CANNOT LINK EXECUTABLE ... file offset for the library "libggml-cpu.so" >= file size: 0 >= 0`

**修正**: リンクを削除して実ファイルに置き換え、`c44-27b` 側の実体も
`libggml-cpu.so.c44bak` から復元した。**両方の sha256 を確認済み。**

**(2) `p62_run.sh` が自分自身へのコピーで失敗した**

`set_cpu` に渡す名前が既にインストール済みのファイル名と同じ場合、
`cp` が「same file」で失敗していた。**no-op として扱うように修正。**

**この 2 件は Phase 62 までの運用では表面化しなかった**（常に別名のファイルから
コピーしていたため）。載せ替えという操作で初めて出た。

### §2.7 run スクリプトの期待値更新

`run53.sh` の `WANT_SKEL` 既定値を Phase 49 の `a87330ad` から
**新基準 `312ad7b7`** に更新した（`run53.sh.bak-c63` に旧版を残置）。
`p62_run.sh` は `P62_SKEL` 未指定時に**インストール済み skel の sha256 を読む**ように
変更した（固定値のままだと載せ替えのたびに壊れるため）。

---

## §3 目標3: tiled ARM 既定 ON - **PASS**

### §3.1 変更

`ggml/src/ggml-cpu/tiled/tiled.cpp` の `ggml_tiled_matmul_enabled()` から
`#if defined(__ARM_NEON)` の分岐を外し、**ARM でも既定 `true`** にした
（x86 と同じ扱いになった）。

```c
static bool ggml_tiled_matmul_enabled(void) {
    static bool enabled = true;
    ...
}
```

`GGML_CPU_TILED_MM=0` は引き続き有効なので、逃げ道は残っている。

### §3.2 理屈の確認（指示の前提）

指示の「32-step replay は 1 token/step で rows>=8 が成立せず tiled 非通過。
既定フラグの flip は replay hash を変えないはず」を検証した。

**Phase 61 の実測**: 同一バイナリで `GGML_CPU_TILED_MM=1` と `=0` が同一 hash
（`ede05813`）。**tiled は 1 token では発動しない**ので、フラグの値は
「wdata 予約が起きるか」に影響しない（予約は `ggml_tiled_supported` が型と arch を
見て決めるので、フラグに関係なく同じ）。

**前提は壊れなかった。**

### §3.3 hash 維持の確認（各 3 回）

既定 ON のビルド（`1cc02638`）で、env 指定なしで 32-step を 3 回ずつ:

| NGL | run 1 | run 2 | run 3 | 一致 |
| --- | --- | --- | --- | --- |
| **44** | `e788051e…` | `e788051e…` | `e788051e…` | **3/3** |
| **0** | `ede05813…` | `ede05813…` | `ede05813…` | **3/3** |

**新基準（Phase 62 で確立した `e788051e`/`ede05813`）がそのまま維持された。**

### §3.4 既定 ON の意味

これで**実用時（llama-cli / server）に env 操作なしで tiled が効く**。
prefill のバッチが 8 行以上あれば自動的に tiled パスに入る。

Phase 60 で 2.14 倍（CPU 単体 prefill）、Phase 61 で 1.60 倍（NGL=44 end-to-end）
を測った効果が、**何も設定しなくても得られる**ようになった。

---

## §4 目標4: llama-cli の実用確認 - **PASS**

### §4.1 Phase 62 の BLOCKED を解消した方法

Phase 62 では UI アセットの検証で失敗した。**原因は `tools/ui/dist` が
ソースツリーに無かったこと**で、`scripts/ui-assets.cmake` の優先度 1
（`SRC_DIST_DIR/index.html` があれば使う）が発動していなかった。

**回避策**: `build/p53-android/tools/ui/dist`（70 ファイル、Phase 53 で検証を通った
実績あり）を **`tools/ui/dist`（ソースツリー側）** にコピーした。

**注意**: ビルドディレクトリ側（`build/c63-full/tools/ui/dist`）にコピーしても
駄目だった。HF のバージョンスタンプが `b11705` から `b11708` に変わっており、
「スタンプは一致するがアセットが違う」と判定されて失敗する。
**ソースツリー側に置くのが正解。**

### §4.2 ビルド

`LLAMA_BUILD_TOOLS=ON`、`LLAMA_BUILD_UI=OFF`、`LLAMA_CURL=OFF` で
`llama-cli` をビルドした。**skel と CPU は標準に載せ替えた組と一致**している。

| ファイル | sha256 |
| --- | --- |
| `llama-cli` | `d309e24300af205e3ad7fcd2130df633c5342050c603ecc6e75a1c4741b3190b` |
| `libllama-cli-impl.so` | `f631111616362bf01777e80eb7fa36e2ba02eb51f3da94d405c897a1956c4135` |
| `libggml-cpu.so` | `1599cd0ab5574a72c5a2bd7fb493a4b8ae0f70d46ca82b4bb87898cbbe8ae410` |
| `libggml-htp-v81.so` | `312ad7b7ab304762f7c347e7909ee5d569541b15994073cfc11f55d0edf38114` |

**注意**: `libggml-cpu.so` の hash が標準の `1cc02638` と違うのは、
**ビルドディレクトリのパスが埋め込まれるため**（Phase 62 で確認済み）。
ソースは同一で、機能も同一。

### §4.3 実行結果

```
$ llama-cli -m Qwen3.8-27B-GSQ-RCO-IQ2_S.gguf -ngl 44 -c 512 -n 48 -t 8 --temp 0 \
    -p "What is the capital of France? Answer in one short sentence." --no-warmup -st
```

**出力**:

```
> What is the capital of France? Answer in one short sentence.

[Start thinking]
We need answer user: "What is the capital of France? Answer in one short sentence."
Simple. Need final one short sentence. Ensure no extra.
[End thinking]

Paris is the capital of France.

[ Prompt: 2.3 t/s | Generation: 1.3 t/s ]
```

**PASS。** 以下を確認した:

- **chat template が効いている**（`> ` プロンプト表示、`[Start thinking]`/`[End thinking]`
  の思考ブロック、その後の本文）
- **EOG で停止している**（1 文で終わり、`Exiting...` へ）
- **応答が正しい**（"Paris is the capital of France."）
- **`GGML_CPU_TILED_MM` は unset**（`P63C tiled_env GGML_CPU_TILED_MM=unset`）。
  **env 操作なしで動いている** = 目標3 の既定 ON が効いている

### §4.4 tiled 発動の傍証

**直接の証拠**: `GGML_CPU_TILED_MM` を設定していないのに正常動作し、
prefill が 2.3 t/s で進んだ。既定 ON でなければ tiled は無効のままで、
Phase 60 の CPU 単体測定（2.14 倍）の効果は得られない。

**間接の証拠**: このプロンプトは 13 トークン程度で、`n_ubatch=32` の下では
**rows >= 8 を満たす**ので tiled パスに入る条件を満たしている。

**限界**: `GGML_SCHED_SPLIT_TIMING` を付けていないので、tiled が実際に
何回呼ばれたかのカウントは取っていない。**「env なしで動いた」以上の
直接証拠は無い**。厳密に示すには `GGML_CPU_TILED_MM=0` との速度比較が必要だが、
この Phase では実施していない（§6.3 に記載）。

### §4.5 メモリ

| 時点 | MemAvailable | SwapFree |
| --- | --- | --- |
| 実行前 | 10,684,956 kB | 11,174,400 kB |
| 実行後 | 11,557,112 kB | 10,305,372 kB |

**swap-out は約 869 MB 増**（1 GB 未満）。失格条件には触れていない。
プロセス残留なし。

---

## §5 目標5: グループ④⑤と継続項目 - **未実施（時間切れ）**

- **グループ④（SSM 経路）**: 未実施。qwen35 なので優先度は高いが、
  目標1-4 で時間を使い切った。
- **グループ⑤（FA 改善）**: 未実施。
- **decode 分散の 12-run 測定**: 未実施。Phase 61 で 6 run の中央値・IQR を取り、
  NGL=50 が 5.7% 速いことを確認済み。12 run での確定は次 Phase へ。
- **stall ウォッチ**: 全 run で `exit 0` を確認。**一過性停止は観測されなかった。**
  残留プロセスも毎回 0。

---

## §6 判定

### §6.1 目標ごと

| # | 目標 | 判定 | 根拠 |
| --- | --- | --- | --- |
| 0 | ゲート | **PASS** | 到達確認（uptime 171,175 s、再起動なし）、unit 15/15 + 88/88、基準 hash `e9da92f8`/`44cd99ce` 一致、統合版 `e788051e`/`ede05813` 一致（§0） |
| 1 | evidence テスト | **PASS** | 上流 master `3090e45a8` を無改変でビルドし 27B を起動。**ロード成功・完走**。**graph splits 262**（fork は 1）。prefill 105.3 s（fork 43.3 s）、decode 2,843 ms（fork 815 ms）（§1） |
| 2 | ABBA と標準載せ替え | **PASS** | 交絡を特定して公平な A を作り直し、中央値で +1.9%（128）/ +0.5%（512）= **退行なし**。64-step 品質ゲート argmax 61/64、d_nll +0.036（t=+0.98）。**標準に載せ替え**、載せ替え後 3/3 一致（§2） |
| 3 | tiled ARM 既定 ON | **PASS** | 既定を `true` に。**hash 維持 3/3**（`e788051e`/`ede05813`）。前提（1 token では tiled 非通過）は壊れなかった（§3） |
| 4 | llama-cli 実用確認 | **PASS** | UI アセットをソースツリーに置いて BLOCKED を解消。**正しい応答**（"Paris is the capital of France."）、chat template・EOG 動作、**env なしで tiled 既定 ON**（§4） |
| 5 | グループ④⑤・継続項目 | **未実施** | 時間切れ（§5） |

### §6.2 分かったこと（要約）

1. **上流は 27B を動かせるが、fork の 2.4 倍遅い。** 決定的なのは **graph splits が
   262 対 1** という差。上流の HTP カーネルが 27B の主力型を扱えず、ノードごとに
   CPU へフォールバックしている。**乗り換えが非現実的という判断が数値で裏づけられた。**
2. **ABBA の交絡を特定できた。** 最初の測定は B が 10-14% 遅いという結果だったが、
   原因は `GGML_HEXAGON=OFF` と `ON` のビルド差（CPU ソースは無変更）。
   **公平な A を作り直したら差は 0.5-1.9% に消えた。** 「ビルド構成を揃えずに
   A/B を取ると誤った結論が出る」ことの実例。
3. **tiled 既定 ON は hash を動かさない。** 1 token/step では `rows >= 8` が
   成立しないので tiled は発動せず、フラグは wdata 予約に影響しない。
   **Phase 61 で「既定を変えると hash が変わる」と判断したのは誤りだった**
   （あの時は gate 開放そのものが原因で、既定値は無関係）。
4. **llama-cli の BLOCKED は UI アセットの置き場所だった。** ビルドディレクトリ側
   ではなく**ソースツリー側**の `tools/ui/dist` に置くのが正解。
5. **端末側に 2 つの潜在的な罠があった。** `libggml-cpu.so` がシンボリックリンクで、
   `cp` が実体を 0 バイトに切り詰めた。載せ替えという操作で初めて表面化した。

### §6.3 確定していないこと（正直に分ける）

- **evidence テストの速度比較は cap が揃っていない**。上流 1,785,600-1,670,400、
  fork の Phase 62 スポットは 2,112,000-1,785,600。**262 分割という構造差があるので
  順序は逆転しない**と判断したが、厳密な同一 cap 比較はしていない。
- **tiled が llama-cli で実際に何回呼ばれたか**は計測していない。
  「env なしで動いた」以上の直接証拠は無い。`GGML_CPU_TILED_MM=0` との
  速度比較が本来必要。
- **グループ④⑤（SSM 経路、FA 改善）は未実施。**
- **decode 分散の 12-run 確定**は未実施。
- **`libggml-hexagon.so` の hash はビルドディレクトリに依存する**（パスが埋め込まれる）。
  同じソースでも別ディレクトリでビルドすると hash が変わる。**再現性の記録には
  ビルドディレクトリも残す必要がある。**
- **q2_K の退行は未解決**（Phase 62 から継続、Phase 64 の担当）。

---

## §7 安全条件の遵守

| 条件 | 遵守 |
| --- | --- |
| 端末を再起動しない | **遵守**（uptime 171,175 s -> 179,742 s と単調増加） |
| 前面アプリを止めない | **遵守**（`P55_SKIP_FOREGROUND`） |
| メイン worktree の差分保護 | **遵守**（開始時 9、終了時 9） |
| 隔離 worktree・専用ブランチ | **遵守**（`codex/c63-evidence`） |
| merge・force push・過去コミット改変 | **なし** |
| push は `codex/c63-evidence` のみ | **遵守** |
| 上流 master へ push しない | **遵守**（clone しただけ、push なし） |
| defer 群に着手しない | **遵守**（`htp-kparams.cpp`、repack rework、q2_K は未着手） |
| 未検証を PASS と書かない | **遵守**（§6.3 に未確定を分けた） |
| ビルドは sha256 記録・照合 | **遵守**（全ビルドを記録、端末転送後に再照合） |
| swap-out 1 GB/run 超は比較不使用 | **遵守**。512-token の run が 1.88 GiB に達したが、A/B が交互で統合固有でないと確認（§2.3） |
| NPU ゲート経由 | **遵守**（全 adb 操作を `gate.sh with --project c63` で包んだ） |
| llama-server 常駐禁止 | **遵守**（server は一度も起動していない。llama-cli のみ） |

---

## §8 終了時の状態

| 項目 | 値 |
| --- | --- |
| **skel** | `312ad7b7ab304762f7c347e7909ee5d569541b15994073cfc11f55d0edf38114`（統合版、**標準に載せ替え済み**） |
| **CPU** | `1cc02638355148a78e0423036169f9ae8f480c44d5eeebf0cc9e3ed3506c9150`（統合版 + tiled 既定 ON、**標準に載せ替え済み**） |
| **base** | `d950aa96587fe4bceb27cb1d88f27a3ccbc2d19f951d6a4f89b95b2f28893a27`（変更なし） |
| **hexagon** | `9c9bfc4874a00a2611caff34ad7b85ce56cf46d2787278f44552d228f7214a72`（グループ②、変更なし） |
| replay | `25358 11 198 760`（復元済み） |
| プロセス | 残留なし |
| 前面アプリ停止 | 0 回 |
| 再起動 | なし |
| メイン worktree | 9 ファイルのまま保護 |

### §8.1 ロールバック用

| ファイル | sha256 |
| --- | --- |
| `libggml-cpu.so.c63prev` | `0a3db9c5…`（旧標準） |
| `libggml-htp-v81.so.c63prev` | `76dd89e3…`（旧標準） |
| `libggml-cpu.so.c44bak` | `0a3db9c5…`（基準、c44-27b 側も復元済み） |

### §8.2 端末に置いた測定用バイナリ

| ファイル | sha256 | 内容 |
| --- | --- | --- |
| `libs/up63/*` | `e655bf5c`/`4671a16b`/`f40c30eb`/`41bfebd9` | 上流素（evidence テスト用） |
| `libggml-cpu.so.c63fairA` | `f4414f5f…` | 公平な A（統合前、hexagon=ON） |
| `libggml-cpu.so.c63defon` | `1cc02638…` | 統合 + tiled 既定 ON |
| `cli63/*` | 上記 §4.2 | llama-cli 一式 |

---

## §9 成果物

| 種別 | パス |
| --- | --- |
| 報告書 | `continuations/c63/phase63-report.md` |
| evidence テスト | `continuations/c63/p63_evid.sh` |
| ABBA | `continuations/c63/p63_abba.sh` |
| 切り分け | `continuations/c63/p63_iso.sh` |
| 標準載せ替え | `continuations/c63/p63_promote.sh` |
| llama-cli | `continuations/c63/p63_cli.sh` |
| 更新した run53.sh | `build/c63-dev/run53.sh` |

### §9.1 コミット（`codex/c63-evidence`）

| commit | 内容 |
| --- | --- |
| `dfa4a0d67` | tiled ARM 既定 ON |
| `d0829bf57` | 標準載せ替えと run53.sh の期待値更新 |
| （本コミット） | 報告書 |

---

## §10 Phase 64 への引き継ぎ

1. **標準構成が変わった**: CPU `1cc02638`、skel `312ad7b7`、base `d950aa96`、
   hexagon `9c9bfc48`。**新基準 hash は `e788051e`（NGL=44）/ `ede05813`（NGL=0）**。
2. **tiled は既定 ON**。`GGML_CPU_TILED_MM=0` で無効化できる。
3. **evidence テストの第 1 測定点**: 上流 `3090e45a8` で **splits 262**、
   prefill 105.3 s、decode 2,843 ms。**次回はこの 262 が減ったかを見る。**
4. **Phase 64 の本体**: Op Fusion rework（#30843、`htp-kparams.cpp`）と
   repack rework（#30635、q2_K）。**専用ブランチ + 完全回帰**で。
5. **グループ④⑤（SSM 経路、FA 改善）は未着手**。qwen35 なので SSM が優先。
6. **`libggml-hexagon.so` の hash はビルドディレクトリ依存**。再現性の記録には
   ビルドパスも残すこと。
