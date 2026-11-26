# Phase 65 - 測定フェーズ（cap統制・NGL再掃引・Bonsai Q2_0）

対象: 2026-11-27 01:11 〜
作業 worktree: `D:\ghq\github.com\yuubinnkyoku\llama.cpp-c51`
ブランチ: `codex/c65-tuning`（`aca91523b8f52813d35106b5a666ce70640aa14f` から作成）
push 許可: `codex/c65-tuning` のみ

ビルドディレクトリ: `build/c63-test`（fork、hexagon=ON）、
`D:\ghq\github.com\yuubinnkyoku\llama-upstream-c63\build\c65-upcli`（**上流 master**、hexagon=ON）。

---

## §0 目標0: ゲート - **PASS**

| 項目 | 値 |
| --- | --- |
| 接続 | `192.168.55.15:5555`（再接続で復帰。`100.108.106.81:38769` でも到達可） |
| serial / model | `324753221196` / `NX741J` |
| **uptime** | 348,116 s（Phase 64 終了時 281,608 s から単調増加 = **再起動なし**） |
| **cap0 / tz0（開始時）** | **1,785,600 / 53.0 ℃**（低 cap、温まり気味） |
| MemAvailable / SwapFree | 5,610,508 kB / 5,485,772 kB |

### §0.1 unit stage

| モード | PASS | FAIL |
| --- | --- | --- |
| 通常 | **15/15** | **0** |
| `--fuzz` | **88/88** | **0** |

### §0.2 基準 hash（標準組）

標準組 = CPU `abecaf60`（dotprod+i8mm）・skel `312ad7b7`・hexagon `9c9bfc48`・base `d950aa96`。

| NGL | sha256(logits.f32) | 既知値 | 判定 |
| --- | --- | --- | --- |
| 44 | `e788051e16d7e1e63b140a902418c75f779bfe29cfb320bb054598508b09d4eb` | 同左 | **一致** |
| 0 | `ede05813034c131e8ecadcb2ecb3eb34e8bf291f0b9ebb49bd7a5541ca97d9e9` | 同左 | **一致** |

### §0.3 `sdot` カウント（恒久規則）

`libggml-cpu.so`（`abecaf60`）= **1,376 sdot**（Phase 64 の値と一致）。
上流ビルド（`c65-upcli` の `libggml-cpu.so`）= **1,120 sdot**（0 ではない）。
どちらも 0 でないことを確認した。

---

## §1 目標3: Bonsai 2.5 27B Q2_0 evidence - **完了（決定的な結果）**

3フェーズ連続で繰り越されていた項目。**結論: fork では「警告なしにゴミを出す」を実測で確認。上流では CPU は正常、HTP はロード時に異常終了。**

### §1.1 下調べ（端末不要・最初に実施）

#### (a) fork が持っているもの / 持っていないもの

| 要素 | fork | 上流 master |
| --- | --- | --- |
| `GGML_TYPE_Q2_0`（型 42） | **あり** | あり |
| `GGML_TYPE_Q1_0`（型 41） | **あり** | あり |
| `ggml_compute_forward_fwht`（CPU FWHT カーネル） | **あり**（`ggml-cpu.c` / `ops.cpp`） | あり |
| `GGML_HINT_SRC0_IS_HADAMARD` | **あり** | あり |
| `ggml_vec_dot_q2_0_q8_0` | **あり** | あり |
| **`LLM_KV_PRISM_HADAMARD_*`（モデル側の結線）** | **0 件（無い）** | あり |
| `llama_verify_hadamard_graph` | **無い** | あり |

**つまり fork は「カーネルは持っているが、モデル側で Hadamard を適用する配線が無い」。**
上流の `a9d27ac69`（#29600, Prism Bonsai 2 27B 対応）が入っていないため。
（この commit は上流 master では rebase されて別 SHA になっているが、`src/llama-arch.h` に
`LLM_KV_PRISM_HADAMARD_VERSION` 等が存在することで内容の取り込みを確認した。）

上流のコードには次のコメントがある:

```
// without this check, an arch that skips the transform helpers loads and computes wrong results
```

**警告なしにゴミを出す、と上流自身が明記している。**

#### (b) PrismML 自身の記述（HF の `presets/llama.cpp/mtp-hexagon.json` より）

```json
"released_build": null,
"released_build_note": "No public build runs this model correctly on the Hexagon NPU yet;
                        it needs a fix that is not released."
"runtime_note": "The whole model runs on the NPU over four sessions (embedding, 64 layers,
                 draft head, output layer); only sampling runs on the CPU.
                 Without the fix the output is wrong."
```

**「修正なしでは出力が間違っている」とベンダーが明言している。**
指示が警告していた「warningなしにゴミを出す」リスクは、**ベンダーの文書で裏付けられた。**

### §1.2 モデルの入手

| 項目 | 値 |
| --- | --- |
| リポジトリ | `prism-ml/Ternary-Bonsai-2.5-27B-GGUF`（公開、apache-2.0） |
| ファイル | `Ternary-Bonsai-2.5-27B-Q2_0.gguf` |
| サイズ | **7,673,194,848 B（7.15 GiB）** |
| sha256 | `84496171ca79d4d33fb45612a2afd8303ec494baf916281023b7dc9cb3b3b9fa` |
| 取得方法 | PC で curl（約 95 秒）→ sha256 照合 → adb push（722 秒）→ 端末側で **sha256 再照合一致** |

**注意**: 端末の空きが 1.9 GB しか無かったため、`runs/` 配下の古い `logits.f32`
（968 個、約 9 GB）を削除して 10 GB を確保した。**直近 42 個（Phase 64 の分）は保持。**

MTP 版（`-mtp.gguf`、7.37 GB）は**非公開リポジトリ**（`...-gguf-mtp`、カードに
「Private, unreleased model ... do not redistribute」）なので取得していない。

### §1.3 測定結果（4 通り）

同一モデル・同一端末で、ビルドだけを変えた。

| # | ビルド | 実行 | 結果 | 出力 |
| --- | --- | --- | --- | --- |
| 1 | **fork** `cli63`（b11708-d0829bf57） | CPU（`-ngl 0`） | **ロード成功・エラーなし** | **`ỉ` / `essiutique柏emer`（ゴミ）** |
| 2 | **上流 master** `cli65`（fa9f928） | CPU（`--device none`） | ロード成功 | **`The user's message is just "The`（正常）** |
| 3 | 上流 master | HTP `-ngl 99` | **rc 134（SIGABRT）で異常終了** | `dspqueue_read failed: 0x0000002e` |
| 4 | 上流 master | HTP `-ngl 8` / `-ngl 20` | **どちらも rc 134** | 同上 |

#### (1) fork は「無警告でゴミ」

```
ftype      : Q2_0
modalities : text

> The
[Start thinking]
essiutique柏emer
```

**`ftype Q2_0` と正しく認識し、警告もエラーも出さず、無意味な文字列を生成する。**
指示が警告した通りの挙動である。**Hadamard 変換が適用されていないことの直接的証拠。**

#### (2) 上流の CPU は正常

```
ftype      : Q2_0
modalities : text

> The
[Start thinking]
The user's message is just "The
[ Prompt: 1.9 t/s | Generation: 1.6 t/s ]
```

**同じモデル・同じ端末で、上流は意味の通る続きを生成する。**
差分はビルドだけ（fork に `#29600` が無い）なので、**原因は Hadamard の配線**と確定できる。

**CPU のみの速度: prefill 1.9 t/s、decode 1.6 t/s**（`-c 256`、6 スレッド）。
参考として fork の（ゴミを出す）CPU 実行は 1.4 / 1.2 t/s だった。

#### (3)(4) 上流の HTP は**ロード時に落ちる**

```
ggml-hexagon.cpp:1889: ggml-hex: dspqueue_read failed: 0x0000002e
ggml_abort
```

`-ngl 99`（全層）、`-ngl 20`、`-ngl 8` の**すべてで同じ箇所で rc 134**。
層数を減らしても変わらないので、**メモリ量の問題ではなく、このモデルの
重み repack 段階で DSP 側が失敗している**と判断できる。

**したがって graph splits は取得できていない（`splits=0`、計測不能）。**
これは測定失敗ではなく、**「この状態では HTP で走らない」という結果そのもの**である。

### §1.4 GSQ-RCO（現行 27B）との対比

| 項目 | GSQ-RCO IQ2_S（現行） | **Bonsai 2.5 Q2_0（今回）** |
| --- | --- | --- |
| サイズ | 9.26 GB | **7.15 GB**（77%） |
| 型 | IQ2_S/IQ3_S/IQ3_XXS/IQ2_XXS/IQ2_XS + Q2_K 等 | **Q2_0 / Q1_0**（ternary） |
| **fork での HTP** | **splits 1（全面オフロード、品質ゲート済み）** | **ロード可・出力はゴミ** |
| 上流での HTP | splits 262（型の大半が CPU フォールバック） | **ロード時に異常終了** |
| 上流での CPU | 測定対象外 | 1.9 t/s prefill / 1.6 t/s decode（正常出力） |
| 速度の比較 | prefill 128 で 27-38 秒（Phase 64 実測） | HTP で走らないため比較不能 |

**構造差の読み方**: GSQ は「fork が型を足したので全面 NPU で速い」。
Bonsai は「上流が型（Q2_0）を持っているが、**NPU 側の修正が未リリース**なので上流でも走らない」。
**fork は Hadamard の配線が無いので、そもそも正しい計算ができない。**
**どちらの経路も現時点では実用にならない。**

### §1.5 採用判断に必要な材料（まとめ）

1. **fork で Bonsai を使うには `#29600`（モデル側 Hadamard 配線）の取り込みが必須。**
   3 ファイル競合（`gguf_writer.py` / `llama-graph.cpp` / `llama-model.h`）で、
   core 側の変更なので hexagon の積み上げとは別系統。**未実施**。
2. **`#29600` を入れても HTP では走らない可能性が高い**（上流 master で既に rc 134）。
   ベンダーが「未リリースの修正が必要」と明言しており、**その修正の入手が前提**。
3. **CPU のみなら上流で動く**（1.9 / 1.6 t/s）。ただし 27B を CPU だけで回すのは
   実用速度ではない。
4. **品質ゲート（hash/NLL）は別モデルなので使えない。**
   今回の evidence は「ロード可否・出力の妥当性・splits・速度・メモリ」に限る。

**推奨: 現時点では採用しない。** ベンダーの Hexagon fix がリリースされ、
かつ fork に `#29600` 相当を取り込めた時点で再評価する。
**「fork に Hadamard 配線を足せば動く」という単純な話ではない**ことを今回の
4 通りの測定が示している。

---

## §2 目標4: 小刻み follow

### §2.1 `ceefd16da` / `27c7a769d` - **既に着地済み**

指示は「競合ゼロで適用済み・未コミットの 2 件を commit して着地」だったが、
**Phase 64 で既にコミット済み**であることを確認した:

- `ac99cebf3 hexagon: take the HMX FA sink fix (#30581) and HVX softplus (#30435)`
- `ceefd16da` と `27c7a769d` を 1 コミットにまとめたもの。`in_HEAD=True`。
- 品質は Phase 64 §18 で検証済み（64-step argmax 64/64、d_nll +0.0012、t=+0.51）。

**追加作業なし。**

### §2.2 上流の型対応 - **IQ2_XXS のみ追加されていた**

指示の「型対応（IQ2/IQ3/XS系）が入っていたら evidence 第2測定点を即実施」を判定するため、
上流 master（`fa9f928`）の tiled 型表を確認した:

```
Q1_0, Q2_0, Q4_0, Q4_1, Q5_0, Q5_1, MXFP4, IQ4_NL, NVFP4, IQ2_XXS,
Q8_0, Q6_K, Q2_K, Q3_K, Q4_K, Q5_K, F16, F32, BF16
```

**IQ2_XXS が追加されている。** ただし:

- **`IQ2_XS` / `IQ2_S` / `IQ3_XXS` / `IQ3_S` / `IQ4_XS` は依然として無い。**
- 27B の構成は IQ3_XXS 82・IQ3_S 64・IQ2_S 62・IQ2_XXS 59・IQ2_XS 46（Phase 64 §15 実測）。
- したがって **上流が tiled で扱える IQ 系は 59 テンソル分だけで、
  残り 254 テンソルは CPU フォールバックのまま。**

**`htp-repack.h` は `3090e45` 以降変更されていない**（`git log -- htp-repack.h` が 1 件のみ）。
Phase 63 の evidence 測定（splits 262）は `3090e45` で取っているので、
**IQ2_XXS の追加はその測定に既に反映されている。**

**結論: 「大きく動いた」とは言えないため、第2測定点の再測定は不要と判断した。**
根拠: 型表が測定点から変わっておらず、変わり得た IQ 系のうち追加されたのは
1 型（59 テンソル）のみ。splits 262 の主因である IQ3_XXS/IQ3_S/IQ2_S は未対応のまま。

**ただし Phase 65 で上流 CLI ビルド（`cli65`）を用意したので、
IQ3 系が入った時点で即測れる状態にはなった。**

### §2.3 #31055（KV-head team split）- **未実施**

Phase 64 §18.2 で `kv_head_min` / `kv_head_max` を全範囲宣言にして
#30731 のループを成立させた。**#31055 でその宣言を本物のチーム分割に置き換える作業**は
今回は行っていない。理由: 目標1（残置バッチの ABBA）と目標3（Bonsai）を優先したため。

**未実施であることを明記する。**

---

## §3 目標5-1: `build/c59-cpu` の dotprod 確認 - **完了（1 コマンド）**

Phase 64 §22.8 の仮説「Phase 59 の『dotprod は 10-15% 遅い』も dotprod 喪失が原因では」
を検証した。

```
build/c59-cpu/CMakeCache.txt
  GGML_CPU_ARM_ARCH:STRING=armv8.6-a+dotprod+i8mm
```

**dotprod は正しく入っていた。** したがって:

- **Phase 59 の測定は正しい。** dotprod 喪失の産物ではない。
- Phase 59 が「dotprod は遅い」と結論したのは、**当時 ARM で tiled が無効だった**ため。
  tiled が無い状態では dotprod は plain 経路にしか影響せず、そこで 10-15% 遅かった。
- **Phase 64 の発見（tiled は SDOT 必須、dotprod があると 3.5 倍速い）と矛盾しない。**
  2 つは別の経路の話であり、両立する。

**説明が成立した。** Phase 59 の結論を訂正する必要はない。
