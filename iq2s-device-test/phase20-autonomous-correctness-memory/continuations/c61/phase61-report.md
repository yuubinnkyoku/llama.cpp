# Phase 61 - Inkling 2 Mini を Hexagon HTP で動かす: 事前調査

対象: 2026-11-21 23:10 〜
作業 worktree: `D:\ghq\github.com\yuubinnkyoku\llama.cpp-c61`
ブランチ: `codex/c61-inkling2`
基準: `codex/c59-cpufloor` (`1b7c83d37`)
上流 merge: `upstream/master` (`c622f623f`)

この報告書は作業中に逐次更新する。数値はすべて実測で、未測定は「未測定」と書く。

---

## §0 目標0: 上流取り込み調査 - PASS

### §0.1 3 本の PR の実測

| PR | 状態 | files | + | - | 内容 |
| --- | --- | --- | --- | --- | --- |
| #32694 `model: Inkling 2 Mini` | **merged** (2026-11-20 22:01 UTC, `29b52d398`) | 13 | 631 | 123 | `inkling2_loop` アーキを追加。24 物理層を 48 キャッシュ slot に展開 |
| #32695 `model: Inkling-2-Mini - looped arch` | **open** | 16 | 489 | 24 | 汎用 looped layers を hparams/model/context に入れる別実装。未 merge |
| #32704 `inkling2_loop re-share the loop slots after the scale pass` | **merged** (2026-11-21 11:10 UTC, `690b87d88`) | 4 | 7 | 0 | scale 付与後に slot を共有し直す。NVFP4 の per-expert scale 欠落を修正 |

**判断: 取り込むべきは #32695 ではなく、merge 済みの #32694 + #32704。**
#32695 は今も open で、`src/llama-context.cpp` / `src/llama-hparams.*` / `src/llama-model-saver.cpp` に汎用 loop 機構を足す別設計である。#32694 は `LLM_ARCH_INKLING2_LOOP` と専用 `src/models/inkling2-loop.cpp` を使う設計で、mainline はこちらを採用した。#32704 はその上での必須修正である。

### §0.2 #32694 の変更箇所

```
conversion/__init__.py
conversion/inkling2.py
gguf-py/gguf/constants.py
gguf-py/gguf/gguf_writer.py
gguf-py/gguf/tensor_mapping.py
src/llama-arch.cpp
src/llama-arch.h
src/llama-model.cpp
src/llama-quant.cpp
src/models/inkling2-loop.cpp   (新規)
src/models/inkling2.cpp
src/models/models.h
tests/test-llama-archs.cpp
```

指示書が想定した `src/llama-graph.cpp` と `src/llama-memory` 系は**触っていない**。48 slot は `hparams.n_layer_all` と `n_layer_phys()` で表現し、KV と再帰状態は既存の hybrid ISWA メモリが `n_layer()` = 48 として確保する。graph ノード予算は `llama_model_inkling2::graph_nodes_per_tensor()` = 32 で、`llama-context.cpp` の `32 * n_tensors` がそのまま効く。

`inkling2-loop.cpp` の実装:

- `n_slot = n_phys + (n_passes - 1) * (core_end - core_start)` = 24 + 2 * 12 = 48。
- `core_start=6`, `core_end=18`, `n_passes=3`, `gate_rank=64`。
- `layers[24..47]` は物理 `6..17` への共有ポインタ。重みは 24 層分だけ。
- 各 pass は prelude 出力から始まり、前 pass の block 出力と最終出力を `mix()` で混ぜる。
- #32704 の `post_arch_tensors` で、scale 付与後に slot を共有する。

### §0.3 #32695 との違い

#32695 は `n_loops` / `skip_loop_final_norm` を `llama_hparams` に足し、`src/llama-context.cpp` で graph サイズを増やし、`src/llama-model-saver.cpp` も変える。汎用機構としては大きいが、mainline の #32694 と競合する。未 merge のため fork に取る理由がない。コメントでは、unsloth の `block_count=48` で `blk.24` 以降が無い GGUF を #32694 が読めない問題が報告されているが、今回の AtomicChat GGUF は `inkling2_loop` / `block_count=24` なので該当しない。

### §0.4 取り込み規模（実測）

c59 は master より **1190 commit 遅れ、245 commit 先行**していた。Inkling 2 の土台 #32217 (text+vision)、#32294、#32381、#32429、#32430、#32657 も c59 には入っていない。したがって #32694 の単純 cherry-pick では済まず、`upstream/master` の merge が必要だった。

merge の実測:

| 項目 | 値 |
| --- | --- |
| merge commit | `0b17926f2` |
| 変更ファイル | **3,120** |
| 行数 | **+430,136 / -150,802** |
| 衝突ファイル | 15（うち HTP 系 11） |

衝突の解決方針:

- `ggml/src/ggml-hexagon/**` は**丸ごと c59 版に戻した**。上流は Hexagon backend を大規模に書き換えており（61 commit、新規 op 群）、c58x4 skel と ABI が合わない。fork の host 実装と skel を対で保つため、104 ファイルを c59 に復元した（`53b91008d`）。
- `ggml/src/ggml-backend.cpp` は fork の split timing / MoE expert copy と、上流の prefetch / `hv_tensor_requires_copy` の両方を残した。fork 側の MoE copy ループを使い、`ids` / `used_ids` / `prev_ids_tensor` の宣言を戻した。
- `tests/test-backend-ops.cpp`、`tests/CMakeLists.txt`、`scripts/snapdragon/run.py` は両側の追加を残した。

### §0.5 fork の HTP 被覆（c59 時点）

`ggml/src/ggml-hexagon/ggml-hexagon.cpp` が明示的に扱う op には `MUL_MAT_ID`、`MUL_MAT`、`FLASH_ATTN_EXT`、`SSM_CONV`、`GET_ROWS`、`SET_ROWS`、`CONCAT`、`SOFT_MAX`、`ARGSORT`、`TOP_K`、`GLU` などがある。量子化型は IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS / IQ4_NL / Q4_0 / Q8_0 / Q6_K などを被覆する。**Q2_0 / IQ1 系 / NF4 は未対応**のままである。

今回の GGUF の型はこの被覆内に収まっている（§2）。

### §0.6 判定

**PASS。** 変更ファイル、行数、依存 PR、fork への取り込み方法を確定した。最大の作業は想定どおりフロントエンドだが、#32695 ではなく merge 済み #32694 + #32704 を上流 merge で取るのが正解だった。

---

## §1 目標1: PC の CPU ビルドで動くこと - PASS

### §1.1 ビルド

Windows x86_64、MSVC 19.44、CMake + Ninja、`GGML_NATIVE=ON`、`GGML_HEXAGON=OFF`。

```
cmake -B build-c61-host -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF -DGGML_NATIVE=ON
cmake --build build-c61-host --target llama-cli -j 12
```

`llama-cli` のビルドは exit 0。`build-c61-host/bin/llama-cli.exe` が生成された。

### §1.2 モデル読み込み（実測ログ）

```
print_info: arch                  = inkling2_loop
print_info: n_ctx_train           = 1048576
print_info: n_embd                = 2304
print_info: n_layer               = 48
print_info: n_layer_all           = 48
print_info: n_head                = 24
print_info: n_head_kv             = 8
print_info: n_swa                 = 1024
print_info: n_ff                  = 1152
print_info: n_expert              = 128
```

GGUF の `block_count=24`、`loop.count=3`、`loop.start=6`、`loop.end=18`、`gate_rank=64` も読み込めた。**48 slot / 24 物理ブロック**の構成が起動ログで確認できた。

### §1.3 生成（3 例）

| プロンプト | 出力 | 結果 |
| --- | --- | --- |
| `What is 17 + 26? Answer with the number only.` | thinking のあと `43` | 正しい |
| `Hello, who are you?` | `The user asks "who are you?"... I should answer honestly...` | 自然な英語の思考 |
| `こんにちは。あなたの名前は何ですか？` | 日本語の質問を正しく解釈し、`I am Inkling, an AI assistant by Thinking Machines Lab` と日本語で答える方針を思考 | 自然 |

CPU のみの速度は prompt 4.3-5.8 t/s、generation 1.6-2.2 t/s（Ryzen 7 5800HS、12 threads）。HTP の数値ではない。

### §1.4 判定

**PASS。** 読み込み、48 slot の構成、ループ出力の自然さを確認した。`--reasoning-effort none` は思考を止めなかったが、これは後続 Phase の課題とする。

---

## §2 目標2: GGUF 選定と取得 - PASS

### §2.1 取得

| 項目 | 値 |
| --- | --- |
| ファイル | `Inkling-2-Mini-AD-IQ2_XXS.gguf` |
| サイズ | **10,425,495,136 B (9.71 GiB)** |
| sha256 | **`200bd7664a7574616ec5ef8407e43243e2b309ea639bbf755797437abbe9a0c1`** |
| HF ETag | 同上（一致） |
| 取得元 | `AtomicChat/Inkling-2-Mini-GGUF` |

### §2.2 テンソル型（gguf-py で実測）

| 型 | 数 |
| --- | --- |
| F32 | 334 |
| Q8_0 | 321 |
| IQ2_XXS | 24 |
| IQ4_NL | 24 |
| Q6_K | 1 |
| 合計 | **704** |

- `blk.N.ffn_gate_up_exps.weight` は IQ2_XXS（24 層）。
- `blk.N.ffn_down_exps.weight` は **IQ4_NL**（24 層、shape `[1152, 2304, 128]`）。
- **Q2_0 / IQ1 系 / NF4 は 0 個。** fork の未対応型は混ざっていない。

指示書の「expert down は Q4_0/Q8_0 系」という想定とは違い、このファイルは IQ4_NL だった。fork は IQ4_NL を被覆しているので問題ない。

### §2.3 端末の空きと収支

| 項目 | 実測 |
| --- | --- |
| 端末 | NX741J (PQ85A01)、Snapdragon 8 Elite Gen 5 |
| `/data` 空き（モデル push 前） | **16 GB** |
| `/data` 空き（モデル配置後） | **6.7 GB** |
| MemAvailable（開始時） | 9.94 GB |
| SwapFree | 10.19 GB |
| 温度 / 画面 | 43.4 ℃ / Awake |

モデル 9.71 GiB + KV（ctx 512 で約 0.1 GiB、ctx 2048 で約 0.23 GiB）+ conv state 約 4 MiB + compute buffer。RAM は 16 GB で、27B の 9.25 GiB 実績と同程度だが余裕は 0.5 GiB 少ない。**MemAvailable 3 GB を中止線とする。**

### §2.4 端末への転送

`adb push` は TCP/mDNS とも **2-3 MB/s** しか出ず、10.4 GB の転送が 1 時間近くかかる見込みだった。途中で切り替え、PC 側に Range 対応の HTTP server を立て、`adb reverse tcp:8765` 経由で端末の `curl -C -` に落とした。実測 **3-15 MB/s**（最初のバーストは 15 MB/s、後半は 3 MB/s）。`/data` が 97% 使用で、UFS の書き込みが時間とともに鈍る。

```
# PC
python serve_range.py D:\models\inkling2 8765
adb reverse tcp:8765 tcp:8765
# device
curl -L --fail -C - -o /data/local/tmp/c61-inkling2/models/Inkling-2-Mini-AD-IQ2_XXS.gguf \
  http://127.0.0.1:8765/Inkling-2-Mini-AD-IQ2_XXS.gguf
```

### §2.5 判定

**PASS。** sha256 一致、未対応型なし、端末の空きも確保できる見込み。転送は遅いが完了する。

---

## §3 目標3: 端末で最初の HTP ゲート - 未測定

### §3.1 重要な訂正: この build の `-ngl` は「最後の N 層」

`src/llama-model.cpp` の実装:

```
const int i_gpu_start = std::max(n_layer_all + 1 - n_gpu_layers, 0);
```

`n_layer_all = 48` なので、**NGL=24 では slot 25..47 だけが HTP に割り当てられる**。しかし物理層のテンソルは slot 0..23 で作られ、slot 24..47 は `layers[24..47] = layers[6..17]` の共有ポインタである。つまり **NGL=24 では物理層の重みは 1 枚も HTP に載らず、output 層だけが HTP になる**。これは「NGL=24 = 全物理層」という指示の前提（first-N）が、この build の last-N 仕様と合わないためである。

全物理層を HTP に載せるには:

| NGL | `i_gpu_start` | HTP に載る物理層 |
| --- | --- | --- |
| 24 | 25 | なし（output のみ） |
| 32 | 17 | 17..23（7 層） |
| 40 | 9 | 9..23（15 層） |
| **48** | **1** | **1..23（23 層）** |
| 49 | 0 | 0..23（24 層、output も HTP） |

**ゲートは NGL=48 で取る。** 物理層 0 だけ CPU に残るが、MoE の `MUL_MAT_ID` は 23 層分 HTP で実行される。NGL=49 はメモリ余裕が 0.4 GiB 少なくなるため、まず 48 で試す。

この last-N 仕様は tiny model（6 層）で実測確認した: NGL=3 で layers 0-3 が CPU、4-6 が HTP。NGL=6 で layer 0 が CPU、1-6 が HTP。

### §3.2 端末側の準備

| 項目 | 値 |
| --- | --- |
| 端末 | NX741J (PQ85A01) |
| skel | c58x4 `76dd89e3bf2af1576101eba7bb10e5b9d750191fdc92b6a1fc019967166782ef` |
| host libs | `53b91008d` からクロスビルド（NDK 29、Hexagon SDK 6.6.0.0、`PREBUILT_LIB_DIR=toolv19_v81`） |
| probe | `decode-probe` を NDK clang で再ビルド（公開 llama API） |
| guard | `guard-v2`（swap 1 GiB）+ 自作 MemAvailable 3 GB 監視 |
| 配置 | `/data/local/tmp/c61-inkling2/` |

tiny model（stories15M Q4_0）で CPU / HTP の両方を先に通し、HTP の session 生成、`libggml-htp-v81.so` の load、CPU と HTP の生成 token 一致（263, 2217, 7826）を確認した。

### §3.3 HTP run の実測

モデルは NGL=48 で HTP に載った。

```
load_tensors: offloaded 48/49 layers to GPU
ggml-hex: HTP0:0 hwinfo: threads 8, hvx 8, hmx 1, vtcm 8 MB
ggml-hex: HTP0:0 new session : session-id 0 domain-id 100000
load_ms 2.4-3.6 s (mmap)
```

graph は **11,826 node**。`sched_reserve` は 28-166 ms。CPU fallback を増やすと split は 612-2,317 まで増えた。

#### 3.3.1 skel が返した NO-SUPPORT

c58x4 skel は Inkling の graph の以下の形を拒否した（host の `supports_op` は「対応」と判定していた）。

| op | 形 | 発生箇所 |
| --- | --- | --- |
| `CONT` | `2:2304:5:1` (transposed 4D) | `mix()` の `ggml_cont(ggml_transpose(states))` |
| `ADD` | `1024:5` (3D) | conv の residual |
| `REPEAT` | `2304:1:5` (3D) | `mix()` の query repeat |
| `MUL_MAT` | F32 `2:2304:5` (3D) | `mix()` の 2 番目の matmul |
| `MUL_MAT` | Q8_0 `2304:384` (m=384) | `blk.N.attn_rel.weight` |

host 側に以下の guard を足した（`ggml-hexagon.cpp`）。

- `CONT`: 非連続または 3D/4D は CPU（`ggml_is_contiguous` + `ne[2]/ne[3]`）。
- F32 `MUL_MAT`: `ne[2]/ne[3] > 1` は CPU。
- `GGML_HEXAGON_2D_ONLY=1`: 3D/4D の op をまとめて CPU（診断用、既定 off）。
- `GGML_HEXAGON_SMALL_MM_CPU=1`: m < 512 の matmul を CPU（診断用、既定 off）。

しかし、これで split が増え、swap が増える。Q8_0 m=384 の `attn_rel` は全層にあるため、個別 guard だけでは 32-step に届かない。

#### 3.3.2 最良の run（`mm3`）

state 系 op（`NONE/RESHAPE/VIEW/PERMUTE/TRANSPOSE/FILL/CPY/SET_ROWS/GET_ROWS`）と `MUL_MAT` / `MUL_MAT_ID` だけ HTP に残し、それ以外を CPU に落とした run。

```
PHASE load_end ... load_ms 2431.654
PHASE prefill_end ... prefill_ms 5865.659 status 0
PREFILL seg 0 tokens 5 ... tokens_s 0.852
TOKEN step 0 id 182416 logit 27.2450581 finite 200058
guard stopped=1
```

- prefill 5 token は **status 0** で完了。
- 最初の decode token は **182416**。host の `llama-tokenize " Tokyo"` も **182416** で一致した。PC の raw completion（`The capital of Japan is Tokyo. ...`）とも一致する。
- その直後に guard が停止。swap 増分は **263,001 pages = 1.003 GiB**（baseline 180,959,817 -> 181,222,818）。**1 GiB の閾値をわずかに超えた。**

#### 3.3.3 CPU-only の対照（`cpu4`）

`CPU` device、NGL=0、4 step。prefill 5 token の途中で guard 停止（swap 増分は同じく約 1.003 GiB）。**HTP を使わなくても、この 9.7 GiB モデルの prefill で 1 GiB の swap が出る。** これは HTP 固有の問題ではなく、端末のメモリ余裕（MemAvailable 8.3-10.0 GB、他アプリが 4-5 GB）と mmap の page cache によるものである。

#### 3.3.4 判定

**FAIL（32-step 未達）。** ただし、以下の部分は PASS に近い。

- モデルは NGL=48 で HTP にロードできた（OOM なし、crash なし）。
- `MUL_MAT_ID` を含む graph が走り、最初の token は CPU と一致した。
- 失敗は **skel の形状被覆不足**と、**1 GiB swap guard** の 2 点。

32-step を完走するには、次 Phase で (a) skel の 3D/4D・Q8_0 m=384 カーネル対応、または host 側の分割を増やさない配置 policy、(b) 端末のメモリ余裕を増やす（他アプリを止めない前提で、より小さいモデル/context/ubatch、あるいは guard の運用見直し）が必要である。

---

## §4 目標4: 速度とメモリの基準値 - PARTIAL

### §4.1 実測

| 項目 | 値 | 条件 |
| --- | --- | --- |
| モデル load | **2.4-3.6 s** | NGL=48、mmap、HTP session 生成込み |
| prefill 5 token | **5.87 s (0.852 t/s)** | `mm3`、CPU fallback 多用、status 0 |
| decode 1 step | 未測定（step 0 の token のみ） | guard 停止のため step 時間は取れず |
| graph nodes | 11,826 | `sched_reserve` |
| HTP compute buffer | 25-50 MiB | split 数により変動 |
| CPU compute buffer | 8-49 MiB | |
| splits | 612-2,317 | CPU fallback の量により変動 |

27B 実績（NGL=44、812-830 ms/step）との比較は、Inkling の 32-step decode が取れていないため**不能**。CPU fallback の prefill 0.852 t/s は HTP の性能ではない。

### §4.2 メモリ

| 項目 | 値 |
| --- | --- |
| MemAvailable（run 前） | 8.3-10.0 GB |
| MemAvailable（run 後） | 10.1 GB |
| swap 増分（`mm3`） | **263,001 pages = 1.003 GiB** |
| swap 増分（`cpu4`） | 約 1.003 GiB |
| `/data` 空き（モデル配置後） | **6.7 GB** |

モデル 9.71 GiB は mmap で載り、HTP 側も fastrpc mmap で共有されるため、HTP model buffer の二重コピーは起きていない（load が 2-3 s で完了し、MemAvailable が 10 GB 台を保っている）。それでも page cache と他アプリで swap が 1 GiB 出る。

### §4.3 判定

**PARTIAL。** モデル load と prefill の初回値は取れたが、32-step decode の基準値は取れていない。CPU fallback の数値は性能基準に使わない。

---

## §5 目標5: 品質基準の整備 - PARTIAL

### §5.1 最初の token 一致

| 経路 | 最初の token |
| --- | --- |
| HTP 混在 `mm3`（prefill status 0） | **182416**（logit 27.245） |
| host CPU tokenize `" Tokyo"` | **182416** |
| PC raw completion | `The capital of Japan is Tokyo. ...` |

**最初の token は一致した。** loop の 48 slot と MoE の `MUL_MAT_ID` を含む graph が、少なくとも 1 token 分は数値的に破綻していないことを示す。

### §5.2 replay hash

4-token replay の logits hash は**未取得**。32-step が guard で止まるため、`P19_REPLAY_TOKENS` を使った 4-step の完走もできていない。次 Phase で、guard を満たす run から最初の 4 token を `replay.txt` に固定し、`sha256(logits.f32)` を基準値にする。

### §5.3 AD-IQ2_S / Q8_0 との比較

**未実施。** AD-IQ2_S (10.76 GB) の追加取得と PC 側の比較は行っていない。今回の目的は HTP ゲートまでで、比較基準は次 Phase に回す。

### §5.4 判定

**PARTIAL。** 最初の token 一致は確認。replay hash と GGUF 間比較は未実施。

---

## §6 安全・測定規則の遵守

| 条件 | 状態 |
| --- | --- |
| server の長時間常駐 | 不使用 |
| decode は probe + guard | 使用（`guard-v2`、swap 1 GiB、MemAvailable 3 GB の二重 guard） |
| 1 run の swap-out 増分 1 GiB 超は比較に使わない | **遵守**。`mm3` と `cpu4` は 1.003 GiB で比較に使わない |
| 画面点灯維持 | Awake を確認 |
| 前面アプリ停止 0 | 停止していない |
| 熱・vendor 電源・ZRAM へ介入しない | 読むだけ |
| 端末再起動 | していない（uptime 2 日 22 時間） |
| メイン worktree の未コミット作業 | 保護（隔離 worktree を使用） |

---

## §7 成果物

| 種別 | パス |
| --- | --- |
| 報告書 | `continuations/c61/phase61-report.md` |
| probe runner | `continuations/c61/p61_probe.sh`（端末配置版） |
| host ログ | `build-c61-host/cli-*.log` |
| 端末 run | `/data/local/tmp/c61-inkling2/runs/` |

### §7.1 コミット

| commit | 内容 |
| --- | --- |
| `0b17926f2` | upstream/master の merge（inkling2-loop、#32694 + #32704） |
| `53b91008d` | HTP backend を c59 版に復元（c58x4 skel と対にする） |
