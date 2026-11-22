# Phase 63 - decode 内訳の測定と、専門家 HTP 配置の再検討

## §0 要点

1. **decode は CPU 律速。** split timing（`GGML_SCHED_SPLIT_TIMING=1`）で 11 decode graph を測ると、1 graph あたり CPU が 1.635 s（compute 1.484 s、sync 0.150 s）、HTP は enqueue の 16 ms だけだった。splits は 326 CPU + 326 HTP。
2. **専門家 matmul は HTP ではなく CPU で動いている。** scheduler は op を「重みテンソルの backend」に割り当てる。expert テンソルは loader の判定で CPU_Mapped（mmap）に落ちているため、gate/up（IQ2_XXS）も down（IQ4_NL）も CPU 実行になる。HTP compute buffer が 25.10 MiB のままなのが、重みコピーが起きていない証拠。
3. **専門家を HTP に置く実験で端末が再起動した。** `override-experts-htp.txt` で expert を HTP に強制すると HTP model buffer が **11,363 MiB** まで育ち、その直後に端末がハングして再起動（16:11 JST、boot_id 更新、uptime リセット）。load は完走していない（`PHASE load_end` なし）。再試行しない。
4. **Phase 63 の主目標（IQ2_XXS MUL_MAT_ID の HTP カーネル）は単独では効かない。** カーネルを足しても、重みが HTP メモリに無い限り op は CPU に割り当てられる。重みを HTP に置くには約 11.4 GB の rpcmem が必要で、この端末では再起動を招く。
5. **C（down 専門家の部分配置）は実装・検証済み。** loader の配置 probe を 32 token で見積もる host 変更で IQ4_NL down が HTP に載る。NUB=8 の 32-step で replay hash `be30a788...` 一致、swap 0.008 GiB、prefill 1.4 倍、decode は横ばい（CPU gate/up が律速）。詳細は §6。

## §1 split timing の実測

条件: `FA_SELECT=1`、`DMA64=0`、`SMALLMM=0`、ctx 512、5-token prompt、16 step 指定。開始時 MemAvailable 10.56 GB。

**この run は swap 増分 1.59 GiB（417,268 pages）で 1.5 GiB を超え、step 11 で guard 停止した。比較用の性能値としては無効。** 以下の内訳は「どこに時間が行くか」の attribution としてのみ使う。

| 項目 | 値 |
| --- | --- |
| 解析できた graph | prefill 1 + decode 11 |
| splits / graph | 326 CPU + 326 HTP = 652 |
| CPU host 時間 / graph | **1.635 s**（compute 1.484 s、sync 0.150 s、copy 0.001 s） |
| HTP host 時間 / graph | 0.016 s（enqueue のみ、device 時間は非同期で重なる） |
| prefill 5 token | 3.40 s（1.47 t/s、cache 冷） |
| decode step | 0.81〜3.89 s（IO 8.4 GB の step を含む） |
| 上位 CPU split | id 0（178 nodes）97 ms/graph、13-node の MoE ブロックが 40〜65 ms/graph で約 20 個 |
| HTP compute buffer | 25.10 MiB（graph 予約時） |

13-node の CPU split は MoE ブロック（routing + gate/up 専門家 matmul）に相当し、48 visit 分が CPU 時間の大半を占める。HTP の attention は非同期で重なっており、wall への寄与は小さい。

## §2 なぜ専門家が CPU に残るか

1. `llama_model_loader::select_weight_buft` が `weight_buft_supported` で MUL_MAT_ID を HTP に置けるか判定する。expert 重みは IQ2_XXS / IQ4_NL で、次の理由で CPU に落ちる。
   - `ggml_hexagon_supported_mul_mat_id` の型リストに **IQ2_XXS が無い**（IQ2 系は ID カーネルが無い）。
   - ダミー op は 512 token（`n_expert_used=8` で 4,096 行）で VTCM を計算するため、どの expert 型でも VTCM 超過で false になる。
2. `ggml_backend_sched` は op の backend を、その weight 入力の backend から伝播させる。重みが CPU_Mapped なので MUL_MAT_ID も CPU に確定する。
3. したがって `sched` の「使用 expert だけコピーする」経路（`ggml-backend.cpp` の MUL_MAT_ID 分岐）は発動しない。HTP は attention（HVX FA）と dense 部だけを担当している。

Phase 62 の §0/§10.1 に「down 専門家（IQ4_NL）は HTP」と書いたが、これは誤り。**両方 CPU** が正しい。本節で訂正する。

## §3 端末再起動の記録（memtest1）

- 目的: expert を HTP に置いたときのメモリ収支を、graph を走らせずに測る。
- 方法: `LOAD_ONLY=1`（P27 resume で load 後に停止）+ `OVERRIDE=override-experts-htp.txt`（`blk.N.ffn_gate_up_exps.weight` と `ffn_down_exps.weight` を HTP へ）。
- 開始: MemAvailable 10.47 GB、MemFree 1.36 GB、pswpout 204,634,316。
- 経過: stderr に `HTP0:0 model buffer size = 11363.11 MiB`。`PHASE load_end` は出ず、ログは途中で切れた。
- 結果: 16:11 JST に端末が再起動（uptime リセット、boot_id `0e601093-357b-4251-b0fe-71b96b8b32ad`）。`/data` のモデル・libs・runs は無事（空き 7.1 GB）。
- 解釈: 11.4 GB の rpcmem 確保で MemAvailable が約 1 GB を切り、load 中の page cache 圧迫と合わせてシステムが耐えられなかった。**全 expert の HTP 配置はこの端末では行わない。**

再起動後の環境は pre-reboot と比較できない（thermal、page cache、uptime が別）。Phase 62 の数値（gate8 など）は pre-reboot の記録として残し、以後の run は別環境として扱う。

## §4 Phase 63 の進め方（修正案）

Phase 63 の当初計画（IQ2_XXS MUL_MAT_ID カーネル追加）は、重みが HTP に無い限り効果がない。したがって次の順に切り替える。

- **A. 安全な範囲の測定を先に**: 再起動後、Phase 55 方式で背景アプリを整理して MemAvailable 11 GB 級に戻し、現在の gate8 構成で 32-step を 1 回再取得する（post-reboot baseline）。これで環境差を記録する。
- **B. CPU 側の MoE を速くする（配置を変えない）**: `P44_NTHREADS` を 6 から 8 に上げる、`GGML_HEXAGON_FFN_OFFLOAD` の設定、CPU の MoE カーネル（IQ2_XXS vec_dot）のマルチスレッド効率を確認する。リスクなしで試せる。
- **C. 部分配置（所有者判断）**: down 専門家（IQ4_NL、約 4.0 GB）だけ HTP に置く。HTP model buffer は約 5.2 GB になり、全配置より現実的。ただし (1) loader の VTCM 判定を IQ4_NL で通す変更、(2) 4 GB の rpcmem 増加を load-only で安全確認、(3) IQ4_NL の MUL_MAT_ID が HTP で CPU 参照と一致することの検証、が必要。
- **D. IQ2_XXS の ID カーネル**: 重みが CPU のままでは不要。C が成立して gate/up も HTP に置ける見込みが立ったときだけ着手する。

## §5 状態

- 端末: 2026-11-22 16:11 に再起動（事故）。16:18 時点 uptime 6 分、MemAvailable 2.6 GB（起動直後）、`/data` 空き 7.1 GB、モデル・libs は無事。
- ブランチ `codex/c61-inkling2`、c63 の report と probe 更新を追加予定。
- Phase 62 の gate8（32-step、709.6 ms/step、swap 0.605 GiB）は pre-reboot の記録。post-reboot では再測定が必要。

## §6 C: down 専門家の部分 HTP 配置（2026-11-22 実装・検証）

### 6.1 実装

- `ggml_hexagon_supported_mul_mat_id` に配置 probe 用の VTCM 見積もりを追加した。loader は 512 token のダミー op で判定するため、ゼロサイズのプレースホルダ buffer を持つ場合だけ src1 を 32 token に丸めて VTCM を見積もる。graph 時の判定は実形状のまま。
- skel の変更は不要（IQ4_NL の MUL_MAT_ID カーネルは既存）。
- 新しい既定では loader が IQ4_NL down 専門家を HTP に配置する。CPU 参照用に `override-down-cpu.txt`、段階テスト用に `override-down-6/12/24.txt` を追加。

### 6.2 段階 load テスト（LOAD_ONLY、中止線 MemAvailable 2 GB、5 秒監視）

| 配置 | HTP model buffer | load 中の MemAvailable 最小 | 結果 |
| --- | --- | --- | --- |
| down 6 層 | 2,250.61 MiB | 8.64 GB | load_end、再起動なし |
| down 12 層 | 3,344.11 MiB | 5.44 GB | load_end、再起動なし |
| down 24 層 | 5,531.11 MiB | 5.44 GB | load_end、再起動なし |

### 6.3 ubatch 8 が必須

- NUB=32 の reserve は 32-token の worst-case graph で down MUL_MAT_ID を CPU に落とし、HTP 重みの CPU コピー（12 層 x 182 MiB = **2,196.85 MiB**）を確保してしまう。
- NUB=8 にすると CPU compute buffer は 12.31 MiB、splits 1307/654 になり、down は pp/tg とも HTP で動く。
- 1 step の実測: down CPU 979 ms に対し down HTP + NUB=8 は **466 ms**。

### 6.4 A/B（32 step、NUB=8、同一セッション近接）

| 項目 | down CPU | down HTP |
| --- | --- | --- |
| 最初の token / logit | 182416 / 27.2587242 | **182416 / 27.2587242** |
| prefill 5 token | 3,521 ms | **2,113 ms** |
| decode 32-step 平均 | 1,272.5 ms/step | 1,310.8 ms/step |
| splits (pp/tg) | 1305 / 652 | 1305 / 652 |
| swap 増分 | 0.57 GiB | **0.008 GiB** |
| 出力 | 同一テキスト | 同一テキスト |

- decode は速くならない。CPU 側の gate/up（IQ2_XXS）が律速で、down を HTP に移しても split 同期が相殺する。prefill は 1.4 倍速く、swap はほぼゼロになる。
- **4-token replay hash は `be30a788...` と一致**（2 回同一）。Phase 62 の基準と bit 一致なので、IQ4_NL ID カーネルの数値は CPU と同一と判断できる。§13.C の採用条件（replay hash 一致、同一セッション A/B、swap 1.5 GiB 内）を満たす。

## §7 B: CPU の dotprod/i8mm ビルド（実験、結論は保留）

- NDK の既定ビルドは dotprod/i8mm 無効。`-march=armv8.7-a+dotprod+i8mm` を `CMAKE_C_FLAGS`/`CMAKE_CXX_FLAGS` で入れて `libggml-cpu.so` を作り（sha256 `a3b7d4e7...`）、端末で SIGILL なく動作することを確認した。
- split timing の比較は run 間のばらつき（page cache と swap）が大きく、1 回ずつの比較では効果を分離できなかった。CPU 合計は dotprod 1,406 ms 対 既定 946 ms と逆転しており、環境差と判断する。
- 次段は同一 run 内で比較できる形（CPU variant dispatch、または `test-backend-ops` の IQ2_XXS ベンチ）で測る。端末の libggml-cpu.so は既定ビルド（`ac7cc94c...`）に戻してある。

## §8 状態と次の一手

- 端末: 16:11 の再起動後、uptime 3 時間 20 分。MemAvailable 10〜11 GB、`/data` 空き 7.1 GB。残存プロセスなし。
- ブランチ `codex/c61-inkling2`: 本 report、ホスト変更（VTCM probe cap）、override 4 種、probe 更新を追加して push する。
- 推奨: C の構成（down HTP + NUB=8）を Phase 63 の基準構成にする。decode の改善には D（IQ2_XXS の ID カーネル）が必要だが、gate/up も HTP に置くと HTP model buffer は約 10 GB になり、2026-11-22 の全配置実験（11.4 GB で再起動）と同水準になる。実施には所有者の判断と段階 load テストが必須。

## §9 訂正（2026-11-23 00:30）: §6.1/§6.4 の A/B は無効だった

§6 の A/B は、down 専門家が実際には両方 CPU に載っていた（HTP model buffer 1,157.11 MiB）ため無効である。32-token の配置 probe cap では VTCM が 8 MB を超え、既定配置は失敗していた。5,531.11 MiB の run はすべて明示 override（`override-down-24.txt`）によるものだった。

### 9.1 正しい A/B（明示 override、NUB=8、32 step、同一セッション近接）

| 項目 | down CPU | down HTP（24 層 override） | down HTP（既定 23 層） |
| --- | --- | --- | --- |
| HTP model buffer | 1,157.11 MiB | 5,531.11 MiB | 5,348.86 MiB |
| 最初の token / logit | 182416 / 27.2587242 | 182416 / 27.2261276 | 182416 / 27.2482471 |
| prefill 5 token | 4,955 ms | 2,312 ms | 2,688 ms |
| decode 32-step 平均 | 919.8 ms/step | **424.1 ms/step** | **493.5 ms/step** |
| swap 増分 | 0.66 GiB | 0.37 GiB | 0.66 GiB |
| 出力 | 同一テキスト | 同一テキスト | 同一テキスト |

- **down HTP は decode が 1.9〜2.2 倍速い**（919.8 → 424.1/493.5 ms/step）。§6.4 の「decode は速くならない」は無効な比較に基づく誤り。
- **replay hash は一致しない**: down CPU `be30a788...` に対し、down HTP（24 層）は `4d89f1cd85e35588cb434b91b356fa88fb449227a891db4daa016a5f3fdcd56f`。IQ4_NL の HTP カーネルは CPU と bit-exact ではない。§13.C の採用条件（replay hash 一致）は満たさない。
- 最初の token と 32-step の出力テキストは一致する。数値は近いが bit 一致ではない。

### 9.2 8-token cap で既定配置が通る（c64 の修正）

配置 probe の cap を 32 → **8 token** に変えると、既定の `override-inkling.txt` でも down が HTP に載る（5,348.86 MiB、23 層。layer 0 は NGL=48 で CPU 割当のため対象外）。新しい host lib は sha256 `c87765f76821cc891a4937f5391a422d3158147ade8f9a2eefc4a0b3f1c2c1f9`。
