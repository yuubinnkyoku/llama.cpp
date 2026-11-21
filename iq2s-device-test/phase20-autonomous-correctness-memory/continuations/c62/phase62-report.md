# Phase 62 - Inkling 2 Mini HTP ゲート: 32-step 完走と HMX flash attention の数値不良

## §0 判定

**目標3の 32-step ゲートは PASS。**

| 条件 | 結果 |
| --- | --- |
| NGL=48（全物理層） | `offloaded 48/49 layers to GPU`、load 2.8 s |
| 32-step 完走 | **PASS**（RC 0、`GENERATED n 32`） |
| 出力の自然さ | **PASS**（`" Tokyo. The capital of the United States is Washington, D.C. The capital of France is Paris. ..."`） |
| 最初の token | **182416**（CPU/PC の `" Tokyo"` と一致、logit 27.259） |
| crash / 再起動 | なし |
| swap guard（1.5 GiB） | **0.60 GiB**（158,532 pages）で超過なし |
| MemAvailable（2.8 GB 中止線） | 11.0〜11.25 GB で余裕 |
| MUL_MAT_ID | down 専門家（IQ4_NL）は HTP。gate/up（IQ2_XXS）は HTP カーネルが無く CPU |

確定した run 条件（`gate8`）:

```
NGL=48, CTX=512, NB=32, NUB=32, mmap, P20_EARLY_SOURCE_RELEASE=1
GGML_HEXAGON_DMA64=0          # PAD の extended mapping 回避
GGML_HEXAGON_FA_SELECT=1      # flash attention を HVX のみに固定（HMX は数値不良）
GGML_HEXAGON_SMALL_MM_CPU=0   # 既定のまま
OPFILTER なし（flash attention も HTP）
guard-v2 /proc/vmstat 393216 20 + MemAvailable 2,800,000 kB
```

実測:

| 項目 | gate8 | gate7（SMALLMM=1） |
| --- | --- | --- |
| graph nodes | 11,826 | 11,826 |
| splits（pp / tg） | 1,305 / 652 | 1,211 / 984 |
| prefill 5 token | 2,092 ms（2.39 t/s） | 2,201 ms（2.27 t/s） |
| **decode 32-step 平均** | **709.6 ms/step（1.41 t/s）** | 720.4 ms/step（1.39 t/s） |
| decode 最小 / 最大 | 363 / 693 ms | 363 / 693 ms |
| IO（decode 1 step） | 57〜353 MB | 同程度 |
| swap 増分 | **0.605 GiB**（158,532 pages） | 0.574 GiB |
| MemAvailable | 11.25 → 11.01 GB | 11.12 → 11.02 GB |

出力 token（gate8）:

```
182416 13 149917 151451 149879 149876 151465 151480 149884 156229 11 149950 ...
" Tokyo" "." " The" " capital" " of" " the" " United" " States" " is" " Washington" "," " D" ...
```

4-token replay hash（`FA_SELECT=1`、2 回走らせて完全一致）:

```
tokens 182416 13 68014 371
sha256(logits.f32) be30a788c131537de787d97557f5367aae66daeefd699c5f6a7cc29dba6bc723
```

## §1 経緯: 何が通らせ、何が壊していたか

Phase 61 の最良 run（`mm3`）は「state 系 + MUL_MAT/MUL_MAT_ID だけ HTP」で、splits 2,078/2,366、decode 2.1〜4.1 s/step、swap 1.63 GiB/6 step で guard 停止だった。Phase 62 は配置を広げ、正解 token を判定に使って壊れる op を 1 つに絞った。

| 配置 | splits (pp/tg) | 最初の token | decode | swap（32 step） |
| --- | --- | --- | --- | --- |
| mm3（matmul のみ HTP） | 2,078 / 2,366 | 182416（正） | 2.1〜4.1 s/step | 1.63 GiB / 6 step で停止 |
| 全 op HTP（HMX FA、gate5） | 1,211 / 984 | **266（誤）** | 0.79 s/step | 0.49 GiB |
| FA だけ CPU（gate6） | 1,305 / 1,078 | 182416（正） | 1.03 s/step | 0.40 GiB |
| HVX FA、SMALLMM=1（gate7） | 1,211 / 984 | 182416（正） | 0.72 s/step | 0.57 GiB |
| **HVX FA、SMALLMM=0（gate8）** | **1,305 / 652** | **182416（正）** | **0.71 s/step** | **0.60 GiB** |

**HMX flash attention だけが数値不良**で、HVX flash attention は正しい。HMX を避けるだけで、flash attention を含む全 op を HTP に置いたままゲートを通せる。

## §2 HMX flash attention の数値不良

最初の token（正解 182416 / logit 27.24）を判定に使い、`GGML_HEXAGON_OPFILTER`（HTP に載せない op の regex）で 1 群ずつ CPU に落とした。DMA64=0、ctx 512、5-token prompt。

| 除外した op | 最初の token | logit | 判定 |
| --- | --- | --- | --- |
| なし（全 HTP、HMX FA） | 266 | 17.459 | 誤 |
| PAD のみ | 266 | 17.459 | 誤（PAD は無関係） |
| SSM_CONV のみ | 220 | 19.784 | 誤 |
| SOFT_MAX のみ | 196975 | 19.519 | 誤 |
| ARGSORT のみ | 266 | 17.459 | 誤 |
| 新規 HTP op 全部（FILL 以外） | 198 | 19.368 | 誤 |
| 上記 + FLASH_ATTN_EXT | 182416 | 27.244 | 正 |
| FLASH_ATTN_EXT のみ | 182416 | 27.238 | 正 |
| **全 HTP + FA_SELECT=1（HVX FA）** | **182416** | **27.243** | **正** |

- FLASH_ATTN_EXT を CPU に落とすと、残りの op（SSM_CONV、SOFT_MAX、ARGSORT、RMS_NORM、ADD/MUL 系、CONCAT、PAD、clip 系）はすべて HTP で正しい値になる。
- その FLASH_ATTN_EXT も、**HMX ではなく HVX カーネルを使えば正しい**（`GGML_HEXAGON_FA_SELECT=1`）。
- Inkling の attention は head_dim 128、n_head 24、n_kv 256、mask は相対バイアスを焼き込んだ 4D `[256,5,24,1]`、sinks はゼロ [24]。HMX FA のチャンク計算か mask/sinks の扱いがこの形で壊れている可能性が高い。HVX 経路は同じ入力を正しく処理する。

## §3 PAD と DMA64

`pad-ops.c` の `op_pad` は、`ne00 >= 512` かつ VTCM が足りるときだけ DMA 経路を使い、それ以外で extended mapping（DMA64）の tensor を拒否する。

```
use_dma = nb[0]==4 && ne00 >= 512 && vtcm >= needed
if (!use_dma && (extended(src0) || extended(dst))) return NO_SUPPORT;
```

`blk.N.attn_rel_bias.weight` は `[16, 1024]` の weight で、`ne00=16 < 512`、weight は extended mapping なので `NO-SUPPORT` になる。`GGML_HEXAGON_DMA64=0` で回避できる。Phase 61 が「skel 未対応の 5 形状」と考えたもののうち、少なくとも PAD はこの DMA64 の問題であり、Q8_0 m=384 は PAD の誤帰属だった可能性が高い（`SMALLMM=0` でも 384 の MUL_MAT は HTP で動き、最初の token は正しい）。

## §4 目標4の基準値

最終構成（`FA_SELECT=1`、`DMA64=0`、`SMALLMM=0`、ctx 512、ubatch 32）で prefill 512 と decode 32-step を取った。

| 項目 | 値 | 備考 |
| --- | --- | --- |
| prefill 512 token | **78,997 ms（6.48 t/s）** | `P53_PROMPT_TOKENS=512`、16 ubatch |
| decode 32-step（prefill 512 の後） | **822.2 ms/step（1.22 t/s）** | step 0 は 2,159 ms、以降 0.4〜0.8 s |
| decode 32-step（5-token prefill） | **709.6 ms/step（1.41 t/s）** | gate8 |
| swap（prefill 512 run 全体） | 1.285 GiB（336,974 pages） | うち decode 区間 0.121 GiB |
| MemAvailable | 10.87〜11.04 GB | 中止線 2.8 GB には触れない |
| splits（pp / tg） | 1,305 / 652 | |

27B 実績（NGL=44/50、812〜830 ms/step）と比べると、Inkling の decode は **709〜822 ms/step で同程度**。prefill は 6.48 t/s（5-token prefill は 2.39 t/s）。熱条件は 27B の測定時と同一ではないため、比は参考値。

## §5 メモリと swap の内訳

- gate8 は開始時 MemAvailable 11.25 GB、Cached 約 9.9 GB、SwapFree 10.4 GB。32 step の swap 増分は 0.605 GiB で、全 step で MemAvailable 9.7 GB 以上。
- gate4（同じ全 HTP 配置だが HMX FA）は MemAvailable 10.28 GB で prefill の 5 token 中に 1.54 GiB を swap し、guard 停止した。**同じ配置でも空き RAM が 1 GB 違うだけで prefill の swap が 1.5 GiB を超える。**
- gate8 の IO は 1 step あたり 57〜353 MB（gate6 は 0.15〜1.5 GB）。expert 重み（gate/up IQ2_XXS 約 4.4 GB + down IQ4_NL 約 4.0 GB）は mmap の page cache にあり、step ごとに一部が読み直される。cache が温まっている間は IO が減る。
- **測定前の MemAvailable 確認が必須。** 10 GB を切る状態では prefill だけで 1.5 GiB を超えることがある。11 GB 級の空きがあれば 512-token prefill でも合計 1.285 GiB に収まる。

## §6 実装メモ

- ホスト側の shape guard（`85e152967`）: `CONT` の 3D/4D・非連続を CPU、F32 `MUL_MAT` の 3D を CPU。最終構成でも有効（mix() の一部が CPU）。
- `GGML_HEXAGON_FA_SELECT=1`: HMX flash attention を無効化し HVX を使う。数値は正しく、HMX より遅いが HTP 上で動く。
- `GGML_HEXAGON_DMA64=0`: PAD の extended mapping 回避。数値は DMA64=1 の CPU PAD と同一（最初の token / logit 一致）。
- `GGML_HEXAGON_SMALL_MM_CPU=0`（既定）: `attn_rel`（384）、`gate_inp`（128）、`sink_proj`（32）も HTP。gate7（=1）との差は decode 1.5% 程度で、既定のままでよい。
- `p62_probe.sh`: 1.5 GiB swap guard + MemAvailable 2.8 GB 監視、10 秒ごとの memlog、`TWO_PHASE=1` で prefill 後 halt を使った decode 区間だけの guard も可能。

## §7 残作業（次 Phase）

1. **HMX flash attention カーネルの修正**（bit-exact 必須）。HVX が正しいので、HMX の mask/sinks/チャンク計算を HVX と比較して切り分ける。直れば decode はさらに速くなる。
2. **IQ2_XXS の MUL_MAT_ID カーネル**。現状 IQ2 系は `hvx_mm_init_vec_dot` に vec_dot が無く、gate/up 専門家は CPU。HTP 化できれば expert streaming が半分になる。
3. PAD の extended mapping 対応（DMA64=0 を外せる）。
4. CONT 3D / F32 3D MUL_MAT の host guard を外せるかの再確認（mix() の HTP 化）。
5. 27B との同一条件比較、AD-IQ2_S / Q8_0 の品質比較（今回は未実施）。

## §8 成果物

- ブランチ `codex/c61-inkling2`、HEAD `18e357914`（c62 の report / probe / override を追加、push 済み）
- `continuations/c62/p62_probe.sh`、`override-experts-htp.txt`、本 report
- device: `/data/local/tmp/c61-inkling2/runs/gate8`（最終 32-step）、`gate7`（SMALLMM=1）、`gate6`（FA CPU）、`gate5`（HMX FA の誤り）、`prefill512c`（目標4基準値）、`replay3`/`replay4`（4-token hash）、`bisect*`（原因特定）
