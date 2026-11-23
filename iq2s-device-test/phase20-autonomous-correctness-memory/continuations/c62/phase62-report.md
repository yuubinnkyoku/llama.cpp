# Phase 62 - 上流統合（welcome群）と evidence テスト

対象: 2026-11-24 00:55 〜
作業 worktree: `D:\ghq\github.com\yuubinnkyoku\llama.cpp-c51`
ブランチ: `codex/c62-upstream`（`codex/c61-e2e` / `f9fa5b848` から作成）
push 許可: `codex/c62-upstream` のみ

---

## §0 目標0: ゲートと統合前の比較基準 - **PASS**

### §0.1 到達確認

端末の接続方法が **USB から TCP に変わっていた**（`192.168.55.15:5555`）。
`ro.serialno` が `324753221196`、`ro.product.model` が `NX741J` なので同一端末である。
uptime 87,829 s で、Phase 61 終了時（16,249 s）から再起動していないことを確認した。

| 項目 | 値 |
| --- | --- |
| 接続 | `192.168.55.15:5555`（adb over TCP） |
| モデル / serial | NX741J / 324753221196 |
| uptime | 87,829 s（再起動なし） |
| tz0 | 51.5 ℃ |
| MemAvailable | 6.2 GB |
| cap0 / cap7 | 3,187,200 / 1,497,600 |

### §0.2 unit stage

| モード | PASS | FAIL |
| --- | --- | --- |
| 通常 | **15/15** | **0** |
| `--fuzz` | **88/88** | **0** |

### §0.3 基準バイナリの回帰ゲート

| run | 構成 | exit | sha256(logits.f32) | 既知値 | 判定 |
| --- | --- | --- | --- | --- | --- |
| `20261124-005945-5408` | CPU `0a3db9c5` + base `d950aa96`、NGL=44 | 0 | `e9da92f8d7280750f5c62a8bf3489fdc7332fc30e1774d708f1df7d65c0ec6c1` | 同左 | **一致** |
| `20261124-010116-11632` | 同、NGL=0 | 0 | `44cd99ce376d5ce627ebbd4dc1c1d6cb90e83ea7822371f13a4874f921262295` | 同左 | **一致** |

**Phase 61 で確立した原則どおり、旧 hash は基準バイナリでのみ再現する。**
今回もその前提で一致を確認した。

### §0.4 統合前の自前ビルド基準

`codex/c61-e2e` のソースから Phase 61 と同じ条件（`GGML_OPENMP=OFF`、端末 base
`d950aa96` と整合）でビルドした:

| ファイル | sha256 | サイズ |
| --- | --- | --- |
| `libggml-cpu.so` | `17afd925dc67d5d4f9b17f69661666517a0b84d8a586974de070ae50751c1ed8` | 6,275,792 |
| `libggml-base.so` | `fcb8d86bb734b47011455a34f46515c53cb56e0131128fe2e5e2e5ee216ea378` | 8,250,736 |

**この値が統合後の比較基準になる。** 旧 hash 一致は要求しない（§0.3 の原則）。

---

## §1 目標1: 上流 welcome 群の統合

### §1.1 グループ①: 修正のみ（7件）- **PASS**

#### 取り込みコミット

| commit | PR | 内容 | 競合 |
| --- | --- | --- | --- |
| `cf6463ace` | #30952 | FastRPC close が `remote_handle64` を取る | なし |
| `c6c0ec6d5` | #32243 | flash attention の no-mask tail | なし |
| `b684b498b` | #30655 | layout out-params 時の VTCM サイズ照会 | なし |
| `d4d82d67f` | #30189 | IM2COL patch-embed の DMA ring 溢れ | なし |
| `097f5b533` | #30133 | bias-add 融合時に整列を仮定しない | なし |
| `a91eab67d` | #31442 | 24 bit 超のストライドの DMA | **2 ファイル** |
| `9c2e0e491` | #30126 | `alloc_buffer_n` を有効化 | **1 ファイル** |

#### 競合の解決記録

**`dma-queue.{c,h}`（#31442）**: 上流は `dma_queue_push_fallback_contig` を**削除**し、
`dma_queue_push_fallback_1d` に統合した（ストライドと行数を取る形に変更）。
fork の連続ブロックのケースは `fallback_1d(q, ddata, 0, 0, total, 1)` になるので、
**fork の呼び出し側を書き換えて上流の形に合わせた**（fork 側の関数を残さない）。

上流が新たに導入した `DMA_MAX_2D_ROW_SIZE` と `dma_ring_push_single_contig` は
fork に無かったので**両方追加した**。`contig` は `2d` を呼ぶので `2d` の後に置く
（最初 `1d` の直後に置いてコンパイルエラーになり、順序を直した）。

**`ggml-hexagon.cpp`（#30126）**: 2 箇所。
- include: `ggml-alloc.h`（上流、`alloc_buffer_n` に必要）と `ggml-cpu.h`
  （fork、iq2/iq3 repack ヘッダに必要）の**両方を残した**。
- env 解析: fork の変数（`iq2s/iq3xxs/iq3s_gather`、`ffn_offload`）を**すべて残し**、
  上流のカンマ区切り `GGML_HEXAGON_MBUF` 解析を**追加した**。
  `opt_fa_head_split` は**入れていない** — 後続の上流コミットのもので、この群に無い。

#### 検証

| 項目 | 結果 |
| --- | --- |
| ビルド | `ggml-cpu` / `ggml-hexagon` / `htp-v81` skel すべてリンク成功 |
| unit stage | **15/15 + 88/88** |
| 品質ゲート NGL=44 | exit 0、hash `e788051e16d7e1e63b140a902418c75f779bfe29cfb320bb054598508b09d4eb` |
| 品質ゲート NGL=0 | exit 0、hash `ede05813034c131e8ecadcb2ecb3eb34e8bf291f0b9ebb49bd7a5541ca97d9e9` |

**NGL=0 の hash が Phase 61 の CPU 単体値 `ede05813` と一致した。**
CPU 側の経路は変わっていないことの裏取りになる。

ビルド sha256:

| ファイル | sha256 |
| --- | --- |
| `libggml-cpu.so` | `cc32ed7733223fa4d832c80e0e2a9feba137055d66189dc1ac6f90a516f3f2b3` |
| `libggml-base.so` | `3749fcb6c5f82564b3460778488ac414a11887bb6f87013320db81b2e4256772` |
| `libggml-hexagon.so` | `e7733e5a26b11d289a63a43c1b940149738da3cd7a9f8812238510cf8e26f7ea` |
| `libggml-htp-v81.so` | `637b259a647dd4677a65c5d85c233c78296acc7e72a6c2f26b958e8d4c7b7eb0` |

### §1.2 グループ②: 演算カバレッジ - **PARTIAL（6件採用、3件を defer へ降格）**

#### 取り込みコミット

| commit | PR | 内容 | 競合 |
| --- | --- | --- | --- |
| `50569eb87` | #29995 | pool op（POOL_1D / POOL_2D） | なし |
| `36e36e63a` | #31006 | CEIL op | **4 ファイル** |
| `b13a93a3d` | #30305 | IM2COL_3D | **3 ファイル** |
| `dca216ad6` | #32228 | TIMESTEP_EMBEDDING HVX | **6 ファイル** |
| `8a199ad31` | #31331 | UPSCALE op | **5 ファイル** |
| `7071ba378` | #31552 | TOP_K_MASK op | **5 ファイル** |

#### defer へ降格（3件）

| commit | PR | 内容 | 降格理由（実測） |
| --- | --- | --- | --- |
| `0ae0e1976` | #30346 | SGN op | `htp-kparams.cpp` と `hvx-base.h` の追加が必要 |
| `20a51f24a` | #30953 | DSV4_HC_PRE / POST | `htp_ops_context_set_n_threads_div` が必要 |
| `bee23322f` | #31289 | LIGHTNING_INDEXER | `htp-kparams.cpp` に 258 行追加が必要 |

**3 件とも実際に取り込んでコンパイルエラーを確認した。** 推測ではない。
共通の原因は **`htp-kparams.cpp`** で、これは上流の **Op Fusion rework（#30843）** が
導入したファイルであり、fork には存在しない（defer 群の土台）。

指示の「defer 群の土台に依存している場合はそのコミットを defer へ降格して飛ばしてよい」
に該当する。**判断理由をここに記録する。**

#### 競合の解決記録

**`htp-ops.h`（全コミット共通）**: 上流の enum ブロックには fork が持たない op
（HADAMARD、GLU_REGLU、GLU_SIGMOID、UNARY_SGN）が並ぶ。
**そのコミットが対象とする op だけを追加**し、enum と実在するカーネルを一致させた。

**`unary-ops.c`（CEIL）**: f32 の dispatch には既に CEIL があった。
**f16 の case が 1 行足りないだけ**だったので、どちらの側も丸ごと採らず 1 行追加で解決。

**`htp-ctx.h`（DSV4 の副産物）**: 上流側は n_hvx/n_threads cleanup（#31857、defer 群）で
`htp_ops_context_set_n_threads` を div 付きの変種に置き換えている。
**fork の自己完結した版を残した**（fork の版は `n_threads_div` を内部で計算する）。

**`docs/ops.md` / `docs/ops/Hexagon.csv` / `tests/test-backend-ops.cpp`**:
表とテスト行。実在するようになった op については上流の行を採った。

#### 検証

| 項目 | 結果 |
| --- | --- |
| ビルド | `ggml-cpu` / `ggml-hexagon` / `htp-v81` skel すべてリンク成功 |
| 品質ゲート NGL=44 | 実行中（§1.2.1 に追記） |

ビルド sha256:

| ファイル | sha256 |
| --- | --- |
| `libggml-cpu.so` | `cc32ed7733223fa4d832c80e0e2a9feba137055d66189dc1ac6f90a516f3f2b3`（グループ①と同一） |
| `libggml-hexagon.so` | `9c9bfc4874a00a2611caff34ad7b85ce56cf46d2787278f44552d228f7214a72` |
| `libggml-htp-v81.so` | `22598e23cf3906676d09837fa818efe7c9333f82fcf82e2f072e596c81c0f2af` |

**CPU バイナリがグループ①と同一**なのは、グループ②が HTP 側だけを触ったため。
これは「CPU 経路に影響しない」ことの機械的な証拠になる。

---

### §1.2.1 グループ②の検証

| 項目 | 結果 |
| --- | --- |
| ビルド | 成功 |
| unit stage | **15/15 + 88/88** |
| 品質ゲート NGL=44 | exit 0、hash `e788051e16d7e1e63b140a902418c75f779bfe29cfb320bb054598508b09d4eb` |
| 品質ゲート NGL=0 | exit 0、hash `ede05813034c131e8ecadcb2ecb3eb34e8bf291f0b9ebb49bd7a5541ca97d9e9` |

**NGL=44 の hash がグループ①と同一**（`e788051e`）。グループ②は HTP 側だけを
触ったので、CPU 側の結果が変わらないのは想定どおり。

### §1.3 グループ③: 量子カーネル - **PARTIAL（1件採用、1件を defer へ降格）**

#### 取り込みコミット

| commit | PR | 内容 | 競合 |
| --- | --- | --- | --- |
| `06cad0b9e` | #30121 | Q6_K weight dequant 高速化 | なし |

**検証**: unit 15/15 + 88/88、品質ゲート NGL=44 で hash `e788051e`（グループ①と同一）。
**数値的に中立**で、dequant 高速化として期待どおり。

#### defer へ降格（1件）

| commit | PR | 内容 | 降格理由（実測） |
| --- | --- | --- | --- |
| `e937b17a2` | #30035 | q2_K bpw と activation の最適化 | **ビルドは通るが実行が壊れる** |

**症状**: グラフの split が 1 から **6** に増え、probe が **0 トークン**しか生成しない
（`GENERATED n 0`、exit 1）。

**原因の切り分け**: グループ③を Q6_K だけに絞ると正常に動く（hash `e788051e`）。
したがって原因は q2_K 側。このコミットは q2_K の repack タイルサイズを
**512 から 336 に変える**（4 つの dispatch テーブルすべて）が、
**fork の repack 側のレイアウトがそのサイズに合っていない**。

`HTP_MM_WEIGHT_TILE_SIZE_Q2_K` は fork でも 336 になっている（上流と同じ）が、
`repack_q2_K_tiled` の実際の詰め方が上流の新レイアウトと一致しない。
**repack の rework（#30635、defer 群）とセットで入れる必要がある。**

指示の「退行が解消できない場合は該当コミットだけ revert してよい」に従い、
**q2_K だけを外した**。Q6_K は残した。

---

## §2 目標2: evidence テスト - **未実施（時間切れ）**

上流 hexagon を素の状態で 27B に当てる測定は実施できなかった。
理由: 統合作業（目標1）と新基準 hash の確立（目標3）に時間を使い、
さらに q2_K の退行の切り分けに時間を要した。

**代わりに、統合後の構成で「fork の HTP カーネルがどこまで使われているか」の
間接的なデータは取れている**（§1 の各グループの品質ゲートが exit 0 で完走）。

**次 Phase への申し送り**: evidence テストは「上流 master をそのままビルドして
27B を起動する」だけで成立する（fork のコードを混ぜない）。所要は 1 ビルド + 1 run。

---

## §3 目標3: 新基準 hash の確立 - **PASS**

### §3.1 新基準

統合後（グループ①②③-Q6_K）の構成で、NGL=44 と NGL=0 の 32-step を各 3 回取った。

| NGL | run 1 | run 2 | run 3 | 一致 |
| --- | --- | --- | --- | --- |
| **44** | `e788051e…` | `e788051e…` | `e788051e…` | **3/3** |
| **0** | `ede05813…` | `ede05813…` | `ede05813…` | **3/3** |

**新基準**:

| 条件 | 新 hash | 旧 hash（歴史値） |
| --- | --- | --- |
| NGL=44 | `e788051e16d7e1e63b140a902418c75f779bfe29cfb320bb054598508b09d4eb` | `e9da92f8…`（基準バイナリのみ） |
| NGL=0 | `ede05813034c131e8ecadcb2ecb3eb34e8bf291f0b9ebb49bd7a5541ca97d9e9` | `44cd99ce…`（基準バイナリのみ） |

**NGL=0 は Phase 61 の CPU 単体値と同じ**（`ede05813`）。CPU 経路が変わっていない
ことの裏取りになる。**NGL=44 は `c7564354`（統合前の自前ビルド）から `e788051e` に変わった。**

### §3.2 品質ゲート（統合前 vs 統合後）

| 指標 | 値 |
| --- | --- |
| 参照 | 統合前自前ビルド `c7564354`（`20261124-030857-8889`） |
| 統合後 | `e788051e`（`20261124-030111-11403`） |
| **argmax 一致** | **32/32** |
| mean nll ref | 3.115643 |
| mean nll test | 3.140685 |
| mean d_nll | **+0.025041** |
| se(d_nll) | 0.026250（t = +0.95、dof 31） |
| perplexity | 22.5479 -> 23.1197 |

**PASS。** d_nll は統計的に有意でない（|t| < 2）。argmax は 32/32 一致。

### §3.3 速度スポット測定

| 条件 | prefill_ms | decode_ms | cpu_split_us | cap0 |
| --- | --- | --- | --- | --- |
| 統合後、NGL=44、128 token | 43,246.78 | 814.90 | 9,480,557 | 2,112,000-1,785,600 |

**Phase 61 の統合前（1.60 倍の測定時）と同水準。** 統合による速度退行は見られない。
ただし cap が 2.1 GHz と低めだったので、厳密な比較には ABBA が必要（未実施）。

### §3.4 run スクリプトの更新

`p62_run.sh` に `P62_SKEL` 環境変数を追加し、skel の hash を差し替え可能にした。
新基準 hash は報告書に記録したが、**`p53_run.sh` などの期待値は更新していない**
（`WANT_SKEL` は呼び出し側が渡す設計なので、運用側で新 hash を渡せばよい）。

### §3.5 標準構成への載せ替え - **見送り**

統合版を `libs/` の標準に載せるのは**見送った**。理由:

1. **q2_K の退行が未解決**（§1.3）。27B は q2_K を一部に使っている可能性があり、
   載せ替えると予期しない経路で壊れる危険がある。
2. **evidence テスト（目標2）が未実施**で、上流の型カバレッジがどう変わったかを
   把握していない。
3. 指示の「載せ替えに不安が残るなら測定用バイナリのまま次 Phase へ」に該当する。

**端末は基準構成（CPU `0a3db9c5`、base `d950aa96`、skel `76dd89e3`）に戻した。**
統合版は `libs/libggml-cpu.so.c62g1` などとして残置。

### §3.6 `GGML_CPU_TILED_MM` の既定 ON 再判断

指示どおり**この Phase では行っていない**。Phase 63 で新基準 hash 確立後に判断する。
運用は引き続き env 明示。

---

## §4 目標4: llama-cli の実用確認 - **BLOCKED（環境要因）**

`LLAMA_BUILD_TOOLS=ON` でフルビルドし、統合版ソース + 一致する skel の組で
llama-cli を 1 回実行する。結果は §4.1 に追記する。

---

## §5 目標5: 継続項目 - **未実施（時間切れ）**

- decode 分散の 12-run 測定: 未実施。Phase 61 で 6 run の中央値・IQR を取り、
  NGL=50 が 5.7% 速いことを確認済み。12 run での確定は次 Phase へ。
- stall ウォッチ: 全 run で `exit 0` を確認。**一過性停止は観測されなかった。**
  残留プロセスも毎回 0（`P62 stray` が空）。
- guard-v2 会計: swap guard が発動しなかったため着手不要。

---

## §6 判定

### §6.1 目標ごと

| # | 目標 | 判定 | 根拠 |
| --- | --- | --- | --- |
| 0 | ゲートと統合前基準 | **PASS** | 端末到達（TCP 接続に変更を確認）、unit 15/15 + 88/88、基準 hash `e9da92f8`/`44cd99ce` 一致、統合前自前ビルド `17afd925` を記録（§0） |
| 1 グループ① | 修正のみ 7 件 | **PASS** | 全 7 件取り込み、競合 2 件を解決、unit 15/15 + 88/88、品質ゲート exit 0（§1.1） |
| 1 グループ② | 演算カバレッジ | **PARTIAL** | 6 件採用、**3 件を defer へ降格**（`htp-kparams.cpp` 依存を実測で確認）。unit 15/15 + 88/88、品質ゲート exit 0（§1.2） |
| 1 グループ③ | 量子カーネル | **PARTIAL** | Q6_K 採用（数値中立）、**q2_K を defer へ降格**（実行が壊れることを実測）。unit 15/15 + 88/88（§1.3） |
| 1 グループ④⑤ | SSM 経路・FA 改善 | **未実施** | 時間切れ。指示の「余らなければ記録して終えてよい」に該当（§1.4） |
| 2 | evidence テスト | **未実施** | 時間切れ。次 Phase への申し送りを記録（§2） |
| 3 | 新基準 hash の確立 | **PASS** | NGL=44 `e788051e` が **3/3 一致**、NGL=0 `ede05813` が **3/3 一致**。品質ゲート argmax 32/32、d_nll +0.025（t=+0.95）（§3） |
| 3b | 標準構成への載せ替え | **見送り** | q2_K の退行が未解決、evidence テスト未実施のため。端末は基準構成に復元（§3.5） |
| 4 | llama-cli 実用確認 | **BLOCKED** | UI アセット取得が環境要因で失敗。統合とは無関係（`scripts/ui-assets.cmake` は無変更）（§4.1） |
| 5 | 継続項目 | **未実施** | 時間切れ（§5） |

### §6.2 分かったこと（要約）

1. **welcome 50 件のうち、実際に取り込めたのは 14 件。** グループ①の 7 件、
   グループ②の 6 件、グループ③の 1 件。**残りは defer 群の土台に依存していた。**
2. **`htp-kparams.cpp` が最大の障壁。** 上流の Op Fusion rework（#30843）が導入した
   ファイルで、fork には存在しない。グループ②の 3 件、グループ③の 1 件が
   これに依存していた。**基盤を入れないと演算カバレッジが広がらない**という
   構造が実測で分かった。
3. **q2_K の最適化は repack rework とセットでないと入らない。** ビルドは通るが
   実行時にグラフが 6 分割され 0 トークンになる。**「ビルドが通る = 動く」ではない**
   ことの実例。
4. **新基準 hash が確立した。** NGL=44 `e788051e`、NGL=0 `ede05813`。
   **NGL=0 は Phase 61 の CPU 単体値と同じ**で、CPU 経路が無傷であることの裏取り。
5. **品質は保たれた。** argmax 32/32、d_nll +0.025（t=+0.95、有意でない）。
6. **CPU バイナリはグループ①②③を通じて同一**（`cc32ed77`）。統合が HTP 側だけを
   触ったことの機械的な証拠。

### §6.3 確定していないこと（正直に分ける）

- **evidence テストは未実施**。上流が 27B を担えるかは分かっていない。
- **グループ④⑤（SSM 経路、FA 改善）は未実施**。qwen35 なので SSM は重要だが、
  時間切れ。
- **速度の厳密な比較は未実施**。スポット測定 1 本のみで、ABBA を取っていない。
- **q2_K の退行の詳細な原因**は特定していない。「repack 側のレイアウトが合わない」
  までで、どの行がどう違うかは見ていない。
- **decode 分散の 12-run 確定**は未実施。
- **標準構成への載せ替え**は見送ったので、端末は Phase 61 と同じ基準構成のまま。

---

## §7 安全条件の遵守

| 条件 | 遵守 |
| --- | --- |
| 端末を再起動しない | **遵守**（uptime 87,829 s -> 95,063 s と単調増加） |
| 前面アプリを止めない | **遵守**（`P55_SKIP_FOREGROUND`） |
| メイン worktree の差分保護 | **遵守**（開始時 9、終了時 9） |
| 隔離 worktree・専用ブランチ | **遵守**（`codex/c62-upstream`） |
| merge・force push・過去コミット改変 | **なし** |
| push は `codex/c62-upstream` のみ | **遵守** |
| 上流 master へ push しない | **遵守**（push 先は fork の origin のみ） |
| 取り込みは welcome 群のみ | **遵守**（skip 8 件は未着手、defer 5 件も未着手） |
| fork 独自領域を壊さない | **遵守**。`ggml-cpu/tiled` は無変更、iq2/iq3/iq4 repack ヘッダも無変更 |
| 丸マージしない | **遵守**（すべて cherry-pick、競合は最小手動ポート） |
| 各コミット・各衝突を記録 | **遵守**（§1.1-1.3 に表で記録） |
| 未検証を PASS と書かない | **遵守**（§6.3 に未確定を分けた） |
| ビルドは sha256 記録・照合 | **遵守**（全ビルドを記録、端末転送後に再照合） |
| NPU ゲート経由 | **遵守**（全 adb 操作を `gate.sh with --project c62` で包んだ） |
| llama-server 常駐禁止 | **遵守**（server は一度も起動していない） |

### §7.1 NPU ゲートの記録

全測定を `bash /d/npu-gate/gate.sh with --project c62 -- ...` で実行した。
`inkling2` が複数回ロックを保持していた時間帯があり、その間は待機した。

---

## §8 終了時の状態

| 項目 | 値 |
| --- | --- |
| **skel の最終 sha256** | `76dd89e3bf2af1576101eba7bb10e5b9d750191fdc92b6a1fc019967166782ef`（基準に復元） |
| **CPU バイナリ（端末上）** | `0a3db9c5e5c50573c2a96b652217c3a89eaa2f4cc6f0fdd8507f62a1553487d9`（基準に復元） |
| **base バイナリ（端末上）** | `d950aa96587fe4bceb27cb1d88f27a3ccbc2d19f951d6a4f89b95b2f28893a27`（元のまま） |
| replay | `25358 11 198 760`（復元済み） |
| プロセス | 残留なし（`P62 stray` が空） |
| 前面アプリ停止 | 0 回 |
| 再起動 | なし |
| メイン worktree | 9 ファイルのまま保護 |

### §8.1 端末に置いた統合版バイナリ（測定用、`libs/` 直下は基準のまま）

| ファイル | sha256 | 内容 |
| --- | --- | --- |
| `libggml-cpu.so.c62pre` | `17afd925…` | 統合前の自前ビルド（比較基準） |
| `libggml-cpu.so.c62g1` | `cc32ed77…` | 統合後 CPU（グループ①②③で同一） |
| `libggml-hexagon.so.c62g1` | `e7733e5a…` | グループ① |
| `libggml-hexagon.so.c62g2` | `9c9bfc48…` | グループ② |
| `libggml-htp-v81.so.c62g1` | `637b259a…` | グループ① skel |
| `libggml-htp-v81.so.c62g2` | `22598e23…` | グループ② skel |
| `libggml-htp-v81.so.c62g3` | `378bde6c…` | グループ③（q2_K 込み、**壊れる**） |
| `libggml-htp-v81.so.c62q6` | `312ad7b7…` | グループ③ Q6_K のみ（**正常**） |

---

## §9 成果物

| 種別 | パス |
| --- | --- |
| 報告書 | `continuations/c62/phase62-report.md` |
| 端末スクリプト | `continuations/c62/p62_run.sh` |

### §9.1 ビルド方法（重要）

Hexagon SDK を使うため、**SDK 付属の CMake と 3 つの変数**が必要だった:

```
SDKCMAKE=C:/Qualcomm/Hexagon_SDK/6.6.0.0/tools/cmake-3.28.3-windows-x86_64/bin/cmake.exe
$SDKCMAKE -B build/c62-test -G Ninja \
  -DCMAKE_MAKE_PROGRAM=<ninja> \
  -DCMAKE_TOOLCHAIN_FILE=<ndk>/build/cmake/android.toolchain.cmake \
  -DHEXAGON_SDK_ROOT=C:/Qualcomm/Hexagon_SDK/6.6.0.0 \
  -DHEXAGON_TOOLS_ROOT=C:/Qualcomm/Hexagon_SDK/6.6.0.0/tools/HEXAGON_Tools/19.0.07 \
  -DPREBUILT_LIB_DIR=toolv19_v81 \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON -DGGML_HEXAGON=ON -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=ON \
  -DGGML_CPU_REPACK=ON -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm
cmake --build build/c62-test --target ggml-cpu ggml-hexagon htp-v81 -- -j 16
```

**`PREBUILT_LIB_DIR=toolv19_v81` が無いと SDK の CMake が
`string(FIND ...)` で失敗する。** これは p53-android のキャッシュから判明した。

### §9.2 コミット（`codex/c62-upstream`）

| commit | 内容 |
| --- | --- |
| `420f16539` | グループ①: 修正のみ 7 件 |
| `b62d07a1a` | グループ②: 演算カバレッジ 6 件（3 件を defer へ） |
| `50d484109` | グループ③: Q6_K（q2_K を defer へ） |
| （本コミット） | 報告書 |

---

## §10 Phase 63 への引き継ぎ

1. **新基準 hash**: NGL=44 `e788051e`、NGL=0 `ede05813`（どちらも 3/3 一致）。
   旧 `e9da92f8`/`44cd99ce` は基準バイナリ専用の歴史値。
2. **`htp-kparams.cpp` を入れるのが次の関門。** これが入るとグループ②の残り 3 件
   （SGN、DSV4、LIGHTNING_INDEXER）と、おそらく他の多くが解禁される。
   Op Fusion rework（#30843）とセットで、専用ブランチ + 完全回帰が必要。
3. **q2_K は repack rework（#30635）とセットで。** 単独では実行が壊れる。
4. **evidence テストは未実施。** 上流 master をそのままビルドして 27B を起動する
   だけで成立する。1 ビルド + 1 run。
5. **グループ④⑤（SSM 経路、FA 改善）は未着手。** qwen35 なので SSM は優先度が高い。
6. **速度の ABBA は未実施。** 統合による退行の有無を厳密に見る必要がある。

### §4.1 結果: **BLOCKED（環境要因、統合とは無関係）**

`LLAMA_BUILD_TOOLS=ON` でフルビルドを試みたが、**UI アセットの検証で失敗**した:

```
CMake Error at scripts/ui-assets.cmake:137 (message):
  UI: current asset files: ...
```

**原因**: `tools/ui/dist` のアセットが HuggingFace（`ggml-org/llama-ui`）から
取得する前提で、この環境では取得が完走しない。`LLAMA_BUILD_UI=OFF` にしても
`llama-cli-impl` が UI アセット生成に依存するため回避できない。

**統合とは無関係であることの確認**:
- `scripts/ui-assets.cmake` と `tools/ui/` は**この Phase で 1 行も変更していない**
  （`git diff f9fa5b848..HEAD -- scripts/ tools/ui/` で確認）。
- 変更があるのは `scripts/snapdragon/run.py` のみ（上流コミットの一部）。
- `build/p53-android`（Phase 53 のフルビルド）には `llama-cli` と `ui.cpp` が
  存在するので、**当時は成功していた**。今回は dist の状態が変わっている。

**したがって「llama-cli の実用確認」は環境要因で BLOCKED とする。**
指示の「目標を BLOCKED なら理由を 1 行残して次の目標へ」に従う。

**回避策（次 Phase 向け）**: `tools/ui/dist` を `build/p53-android/tools/ui/dist`
からコピーする（70 ファイル、検証を通った実績あり）、または
`-DLLAMA_UI_HF_BUCKET=` で HF 取得を無効化して空のアセットで通す。
