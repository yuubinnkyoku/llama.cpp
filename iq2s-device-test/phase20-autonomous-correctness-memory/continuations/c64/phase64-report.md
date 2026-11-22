# Phase 64 - D（gate/up の HTP 配置）の段階 load とメモリ曲線

## §0 結論

- **D（gate/up 全 24 層の HTP 配置）は不採用。** down 24 + up 6 で HTP model buffer **6,989.11 MiB**、up 6 層あたり約 243 MiB（repack 込み）。24 層に外挿すると **11,363 MiB** で、2026-11-22 に端末を再起動させた全配置実験の値と一致する。
- 段階テスト中に危険なメモリ状態を作った。up-6 の load は開始 MemAvailable **3.26 GB**（前 run の page cache が残っていた）で走り、load 中に **0.10 GB** まで落ちた。LOAD_MEM_GUARD が作動し、端末の再起動はなかった。監視を 1 秒間隔に強化した。
- **既定配置は低メモリで静かに CPU へフォールバックする。** MemAvailable 9.4 GB で `override-inkling.txt`（既定）を使うと down が CPU に落ち、HTP buffer は 1,157.11 MiB のまま。同じメモリで明示 override（`override-down-24.txt`）なら 5,531.11 MiB を確保できる。既定配置には約 10.5 GB 以上の空きが必要。
- 次の主目標は §15.2 の M タイル化。D はここで打ち切る。

## §1 実測

| 構成 | HTP model buffer | 開始 MemAvailable | load 中最小 | 結果 |
| --- | --- | --- | --- | --- |
| C（down 24、19:14） | 5,531.11 MiB | 10.53 GB | 5.44 GB | 成功 |
| down 24 + up 6（00:04） | **6,989.11 MiB** | 3.26 GB | **0.10 GB** | load_end、guard 作動、再起動なし |
| 全 expert（参考、16:10 の事故） | 11,363.11 MiB | 10.47 GB | 約 1 GB | 再起動 |

- up-6 の増分は 6,989.11 - 5,531.11 = **1,458 MiB / 6 層 = 243 MiB/層**（IQ2_XXS 167 MiB の repack 込み）。
- 外挿: up 12 = 8,447 MiB、up 24 = **11,363 MiB**（全配置事故と一致）。
- したがって「24 層で 10 GB 超え・2 GB 割れ」が確定し、D の全配置は不採用。

## §2 メモリ安全の教訓

- LOAD_ONLY の監視が 5 秒間隔で、数秒で数 GB 進む確保バーストに間に合わず、2 GB の中止線を大きく割った（0.10 GB まで低下）。
- 監視を **1 秒間隔**に変更した。中止線は所有者指定の 2 GB のまま。
- `timeout` の子プロセスが残ると HTP buffer が解放されず、見かけ上メモリが戻らない。LOAD_ONLY の終了時に `decode-probe` を明示的に kill する処理を追加済み（今回の 0.10 GB 状態からの復旧もこれで行った）。
- 配置実験の前提: 開始 MemAvailable 10.5 GB 以上、監視 1 秒、終了時の孤児確認。

## §3 既定配置のフォールバック

- 9.4 GB で既定の `override-inkling.txt`（`^$=CPU` のみ）を使うと、down（IQ4_NL）が CPU に落ちる。loader の `weight_buft_supported` の HTP 判定が false になり、CPU へ move される。
- 明示 override（`blk\.[0-9]+\.ffn_down_exps\.weight=HTP`）では同じメモリで 5,531.11 MiB を確保できる。override は support 判定をバイパスするため。
- 原因候補: probe 用の zero-size HTP buffer 確保が低メモリで失敗し、判定が false になる。確定には `weight_buft_supported` の失敗理由ログが必要。
- 運用: C 構成の run では **`override-down-24.txt` を明示的に使う**（決定的）。既定配置は 10.5 GB 以上の余裕があるときだけ。

## §4 次の一手

1. **M タイル化（§15.2）**: down の MUL_MAT_ID を ubatch 32 でも HTP で動かす。VTCM の src1 バッファを M タイルに分割し、worker をタイルループにする。NUB=32 の worst-case reserve（2,196.85 MiB の CPU コピー）をなくす。bit-exact は replay hash で検証。
2. 既定配置フォールバックの原因ログを足す（低メモリ時の再現性）。
3. D は不採用。up の部分配置（6 層）は理論上可能だが、開始 10.5 GB でも残り 3.5 GB で decode 中の churn に耐えられない見込み。費用対効果から見送り。

## §5 状態

- 端末: uptime 7 時間 55 分（16:11 の再起動後、追加の再起動なし）。孤児 kill 後 MemAvailable 10.6 GB、`/data` 空き 7.1 GB。
- ブランチ `codex/c61-inkling2`: 本 report と probe の監視 1 秒化を追加して push する。

## §6 追加結果（2026-11-23 00:30）: 8-token cap と最終 A/B

### 6.1 8-token cap で既定配置が通った

配置 probe の cap を 32 → 8 token に変更した（libggml-hexagon.so sha256 `c87765f76821cc891a4937f5391a422d3158147ade8f9a2eefc4a0b3f1c2c1f9`）。既定の `override-inkling.txt` のままでも down が HTP に載り、HTP model buffer は **5,348.86 MiB**（23 層。layer 0 は NGL=48 で CPU 割当のため対象外）。明示 override（24 層）は 5,531.11 MiB。

### 6.2 最終 A/B（NUB=8、32 step、同一セッション近接）

| 項目 | down CPU | down HTP 24 層 | down HTP 既定 23 層 |
| --- | --- | --- | --- |
| prefill 5 token | 4,955 ms | 2,312 ms | 2,688 ms |
| decode 32-step 平均 | 919.8 ms/step | **424.1 ms/step** | **493.5 ms/step** |
| swap 増分 | 0.66 GiB | 0.37 GiB | 0.66 GiB |
| 最初の token | 182416 | 182416 | 182416 |
| 出力テキスト | 同一 | 同一 | 同一 |

- **down HTP は decode が 1.9〜2.2 倍速い。** c63 §6.4 の「速くならない」は、down が両方 CPU だった無効な比較による誤り（c63 §9 で訂正）。
- **bit-exact ではない**: replay hash は down CPU `be30a788...`、down HTP 24 層 `4d89f1cd85e35588cb434b91b356fa88fb449227a891db4daa016a5f3fdcd56f`。§13.C の「CPU 参照と bit 一致」は満たさない。最初の token と 32-step 出力は一致し、logit 差は約 0.03。
- したがって C は「性能は大きいが bit-exact でない」ため、採用には所有者の判断が必要。bit-exact を維持するなら IQ4_NL ID カーネルの CPU 一致実装（タイル順序の見直し）か、C を見送って M タイル化/CPU 高速化に進む。

### 6.3 D の結論（変更なし）

全 gate/up（24 層）の HTP 配置は 11,363 MiB で不採用（§0）。up 6 層でも 6,989 MiB で、開始 10.5 GB でも残り 3.5 GB。decode 中の churn を考えると実用性は低い。

### 6.4 状態

- 端末: uptime 8 時間 16 分（16:11 再起動後、追加再起動なし）。MemAvailable 10〜12 GB、`/data` 7.1 GB。
- ブランチ: 本節、c63 §9 訂正、probe の 1 秒監視、c64 override 群を追加して push する。
