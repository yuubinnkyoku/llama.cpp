# Phase 62 - Inkling 2 Mini HTP ゲート: 32-step 完走と数値破壊の原因特定

## §0 判定

**目標3の 32-step ゲートは PASS。** ただし、`FLASH_ATTN_EXT` を CPU に置く条件付きである。

| 条件 | 結果 |
| --- | --- |
| NGL=48（全物理層） | `offloaded 48/49 layers to GPU` |
| 32-step 完走 | **PASS**（RC 0、`GENERATED n 32`） |
| 出力の自然さ | **PASS**（`" Tokyo.\n\nThe capital of Japan is Tokyo.\n\nA. True..."`） |
| 最初の token | **182416**（CPU/PC の `" Tokyo"` と一致） |
| crash / 再起動 | なし |
| swap guard（1.5 GiB） | **0.396 GiB**（103,759 pages）で超過なし |
| MemAvailable（2.8 GB 中止線） | 9.9〜11.3 GB で余裕 |
| MUL_MAT_ID | down 専門家（IQ4_NL）は HTP で計算。gate/up（IQ2_XXS）は HTP カーネルが無く CPU |

確定した run 条件:

```
NGL=48, CTX=512, NB=32, NUB=32, mmap, P20_EARLY_SOURCE_RELEASE=1
GGML_HEXAGON_DMA64=0
GGML_HEXAGON_SMALL_MM_CPU=1
GGML_HEXAGON_OPFILTER=^FLASH_ATTN_EXT$     (flash attention のみ CPU)
guard-v2 /proc/vmstat 393216 20 + MemAvailable 2,800,000 kB
```

実測（gate6）:

| 項目 | 値 |
| --- | --- |
| graph nodes | 11,826 |
| splits（decode） | 1,304 |
| load | 約 2.5 s |
| prefill 5 token | 1,857 ms（2.69 t/s） |
| decode 32-step 平均 | **1,028.2 ms/step（0.973 t/s）** |
| decode 最小 / 最大 | 371 ms / 2,371 ms（約 5 step ごとに 2.2〜2.4 s の山） |
| IO（decode 1 step） | 0.15〜1.5 GB |
| swap 増分 | 0.396 GiB |
| MemAvailable | 開始 11.28 GB、終了 11.15 GB、最小 9.90 GB |

4-token replay hash（同一条件で 2 回走らせ、完全一致）:

```
tokens 182416 13 68014 371
sha256(logits.f32) a1d490abb2abb3953ab6b843b9cd6146b303c5b90c2f92d4d75ca68a70c850c6
```

## §1 何が変わって通ったか

Phase 61 の最良 run（`mm3`）は「state 系 + MUL_MAT/MUL_MAT_ID だけ HTP」で、splits 2,078/2,366、decode 2.1〜4.1 s/step、swap 1.63 GiB/6 step で guard 停止だった。Phase 62 は配置を広げ、数値破壊の原因を 1 つに絞った。

| 配置 | splits (pp/tg) | 最初の token | decode | swap |
| --- | --- | --- | --- | --- |
| mm3（matmul のみ HTP） | 2,078 / 2,366 | 182416（正） | 2.1〜4.1 s/step | 1.63 GiB / 6 step で停止 |
| 全 op HTP（gate5） | 1,211 / 1,304 | **266（誤）** | 0.79 s/step | 0.40 GiB / 32 step |
| **flash attention のみ CPU（gate6）** | 1,304 | **182416（正）** | **1.03 s/step** | **0.40 GiB / 32 step** |

全 op HTP は速いが数値が壊れる。flash attention だけ CPU に戻すと、正解に戻って 1.03 s/step を保つ。**HTP の FLASH_ATTN_EXT が Inkling の attention で誤値を出す**のが唯一の数値ブロッカーである。

## §2 数値破壊のバイセクト

最初の token（正解 182416 / logit 27.24）を判定に使い、`GGML_HEXAGON_OPFILTER`（HTP に載せない op の regex）で 1 群ずつ CPU に落とした。DMA64=0、SMALLMM=1、ctx 512、5-token prompt。

| 除外した op | 最初の token | logit | 判定 |
| --- | --- | --- | --- |
| なし（全 HTP） | 266 | 17.459 | 誤 |
| PAD のみ | 266 | 17.459 | 誤（PAD は無関係） |
| SSM_CONV のみ | 220 | 19.784 | 誤 |
| SOFT_MAX のみ | 196975 | 19.519 | 誤 |
| ARGSORT のみ | 266 | 17.459 | 誤 |
| 新規 HTP op 全部（FILL 以外） | 198 | 19.368 | 誤 |
| 上記 + FLASH_ATTN_EXT | **182416** | 27.244 | 正 |
| FLASH_ATTN_EXT + PAD | **182416** | 27.238 | 正 |
| FLASH_ATTN_EXT のみ | **182416** | 27.238 | 正 |

結論:

- **FLASH_ATTN_EXT が主因。** これを CPU に置けば、残りの op（SSM_CONV、SOFT_MAX、ARGSORT、RMS_NORM、ADD/MUL 系、CONCAT、PAD、clip 系）はすべて HTP で正しい値になる。
- PAD は DMA64=0 なら HTP で正しい（DMA64=1 では `dsp-rsp NO-SUPPORT`）。
- Phase 61 が「skel 未対応の 5 形状」と考えたもののうち、少なくとも PAD は DMA64 の extended mapping 問題、Q8_0 m=384 は PAD の誤帰属だった可能性が高い。実際、`SMALLMM=0` でも 384 の MUL_MAT は HTP で動く（今回は安全側で CPU に残した）。

## §3 PAD と DMA64

`pad-ops.c` の `op_pad` は、`ne00 >= 512` かつ VTCM が足りるときだけ DMA 経路を使い、それ以外で extended mapping（DMA64）の tensor を拒否する。

```
use_dma = nb[0]==4 && ne00 >= 512 && vtcm >= needed
if (!use_dma && (extended(src0) || extended(dst))) return NO_SUPPORT;
```

`blk.N.attn_rel_bias.weight` は `[16, 1024]` の weight で、`ne00=16 < 512`、weight は extended mapping なので `NO-SUPPORT` になる。`GGML_HEXAGON_DMA64=0` で回避できる。恒久修正は「extended mapping でも非 DMA 経路でコピーする」skel 変更（次 Phase）。

## §4 目標4の基準値

同一構成（NGL=48、DMA64=0、SMALLMM=1、flash attention のみ CPU、ctx 512、ubatch 32）で、prefill 512 と decode 32-step を取った。

| 項目 | 値 | 備考 |
| --- | --- | --- |
| prefill 512 token | **86,762 ms（5.90 t/s）** | `P53_PROMPT_TOKENS=512`、16 ubatch |
| decode 32-step（prefill 512 の後） | **933.3 ms/step（1.07 t/s）** | step 0 は 2,159 ms、以降は 0.4〜2.2 s |
| decode 32-step（5-token prefill、gate6） | **1,028.2 ms/step（0.97 t/s）** | 参考 |
| swap（prefill 512 run 全体） | 3.03 GiB（794,870 pages） | うち prefill 区間 2.63 GiB、decode 区間 0.396 GiB |
| MemAvailable | 10.6〜11.2 GB | 中止線 2.8 GB には触れない |

27B 実績（NGL=44/50、812〜830 ms/step）と比べると、Inkling の decode は 933〜1,028 ms/step で **同程度〜1.2 倍**。prefill は 5.90 t/s（5-token prefill は 2.69 t/s）で、これは CPU 側の flash attention と 1,304 splits の同期コストを含む。熱条件は 27B の測定時と同一ではないため、比は参考値。

4-token replay の logits hash は `a1d490abb2abb3953ab6b843b9cd6146b303c5b90c2f92d4d75ca68a70c850c6`（2 回一致）。

## §5 メモリと swap の内訳

- gate6 は開始時 MemAvailable 11.28 GB、Cached 約 9.9 GB、SwapFree 10.0 GB の状態で、32 step の swap 増分は 0.396 GiB。全 step で MemAvailable 9.9 GB 以上。
- gate4（同じ全 HTP 配置、DMA64=0、SMALLMM=1）は MemAvailable 10.28 GB で prefill の 5 token 中に 1.54 GiB を swap し、guard 停止した。**同じ配置でも空き RAM が 1 GB 違うだけで prefill の swap が 1.5 GiB を超える。**
- gate6 の IO は 1 step あたり 0.15〜1.5 GB。expert 重み（gate/up IQ2_XXS 約 4.4 GB + down IQ4_NL 約 4.0 GB）は mmap の page cache にあり、step ごとに一部が読み直される。cache が温まっている間は IO が減るが、完全には 0 にならない。
- 0.396 GiB / 32 step は 1.5 GiB の範囲に収まったが、**空き RAM が 10 GB を切る状態では prefill だけで 1.5 GiB を超える**（gate4）。測定前の MemAvailable 確認が必須。

## §6 実装メモ

- ホスト側の shape guard（`85e152967`）: `CONT` の 3D/4D・非連続を CPU、F32 `MUL_MAT` の 3D を CPU。両方とも今回の構成で有効。
- `GGML_HEXAGON_SMALL_MM_CPU=1`: `src0->ne[1] < 512` の MUL_MAT を CPU。`attn_rel`（384）、`gate_inp`（128）、`sink_proj`（32）が該当。
- `GGML_HEXAGON_DMA64=0`: PAD の extended mapping 回避。weight の mapping flag が変わるが、数値は gate6 と同一（最初の token / logit 一致）。
- `p62_probe.sh`: 1.5 GiB swap guard + MemAvailable 2.8 GB 監視、10 秒ごとの memlog、`TWO_PHASE=1` で prefill 後 halt を使った decode 区間だけの guard も可能。

## §7 残作業（次 Phase）

1. **FLASH_ATTN_EXT の HTP カーネル修正**（bit-exact 必須）。`kq_b`（相対バイアス）を fused mask として渡す Inkling の形、sinks=0、SWA/full 混在のどこで壊れるかを切り分ける。これが直れば全 op HTP で 0.79 s/step に戻る。
2. **IQ2_XXS の MUL_MAT_ID カーネル**。現状 IQ2 系は `hvx_mm_init_vec_dot` に vec_dot が無く、gate/up 専門家は CPU。HTP 化できれば expert streaming が半分になる。
3. PAD の extended mapping 対応（DMA64=0 を外せる）。
4. Q8_0 m=384 / CONT 3D / F32 3D の guard を外せるかの再確認（flash 修正後）。
5. 27B との速度比較、AD-IQ2_S / Q8_0 の品質比較（今回は未実施）。

## §8 成果物

- ブランチ `codex/c61-inkling2`（c62 の成果物を追加）
- `continuations/c62/p62_probe.sh`、`override-experts-htp.txt`、本 report
- device: `/data/local/tmp/c61-inkling2/runs/gate6`（32-step）、`replay1`/`replay2`（4-token hash）、`gate4`/`gate5`（比較）、`bisect*`（原因特定）
