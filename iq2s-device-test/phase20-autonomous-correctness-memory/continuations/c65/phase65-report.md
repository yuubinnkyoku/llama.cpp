# Phase 65 - 基準 hash の凍結、品質ゲート、M タイル化（試作）と再起動

## §0 結論

- **基準 hash を凍結**: down HTP 既定 23 層（NUB=8、override なし、host lib `c87765f7…`）の replay hash `65dd1c563da7f1fb24dc53a362d4acff4afb184bd98f04df96f4d513d5ca661d` を複数回の run で確認した（c64replay_def1/2、c65hash2、c65m_rep1/2、いずれも一致）。
- **品質ゲート**: argmax 一致 **32/32**（基準 31/32 以上）。NLL 差は平均 -0.0035（HTP がわずかに低い）、t 値 **-1.31** で ±1 の帯をわずかに外れるが、95% CI は 0 を含み有意差なし。logit 差は mean_abs 0.154、相関 0.9991。
- **M タイル化は試作したが、最初の実タイル実行（prefill 512、ubatch 32）で端末が再起動した。** 非タイルの範囲（5-token prefill、32-step decode）では hash `65dd1c56` を保ったまま NUB=32 で CPU buffer 49.24 MiB（従来 2,196.85 MiB）を達成したが、実タイル経路にメモリ安全性の欠陥がある。**リバート済み。再試行しない。**
- 端末は 2 回目の再起動（00:32 JST）。安全な lib（skel `76dd89e3`、host `c87765f7`）に戻し、モデル・データは無事。

## §1 品質ゲート（down CPU 参照 vs down HTP 既定 23 層、32 step）

| 指標 | 値 | 基準 | 判定 |
| --- | --- | --- | --- |
| argmax 一致 | **32 / 32** | 31/32 以上 | PASS |
| NLL 平均差（HTP - CPU） | **-0.0035** | 有意差なし | 実質 PASS |
| NLL 差の t 値 | -1.31 | ±1 | わずかに外れる（CI は 0 を含む） |
| logit 差 mean_abs / max_abs | 0.154 / 1.781 | - | 小さい |
| 32 step の相関（最小 / 平均） | 0.99687 / 0.99911 | - | 高い |
| CPU argmax が HTP top10 に入る率 | 32 / 32 | - | PASS |

解釈: HTP down の数値は CPU 参照と bit 一致しないが、argmax と出力テキストは一致し、NLL 差は有意でない。§17 の品質ゲートは「argmax 31/32 以上」を満たし、NLL は平均 -0.0035 nat で実用上等価。t 値のみ ±1 の外（-1.31）なので、厳密運用では追加 32 step で再確認するのが望ましい。

## §2 M タイル化（試作）

### 2.1 実装

- host: `ggml_hexagon_precompute_hvx_mm_params` の ID 分岐で、VTCM に収まる最大の M チャンクを探索し `kparams->m_chunk` に設定（収まらなければ従来どおり UNSUPPORTED）。graph 時の実形状で判定。
- DSP: `hvx_mm_matmul_id` をチャンクループに変更（チャンクごとに activation 転送・量子化・worker 実行）。`hvx_mm_id` は絶対行 `ir1 + rm2*ne11` でチャンク範囲をフィルタし、VTCM の src1 を `row - m0` で引く。
- ビルド: skel `cced2257…`、host `728bf58a…`。

### 2.2 非タイル範囲の結果（NUB=32）

| 項目 | 値 |
| --- | --- |
| 5-token prefill | 2,143 ms（2.33 t/s） |
| decode 32-step | **472.8 ms/step（2.12 t/s）** |
| CPU compute buffer | **49.24 MiB**（タイル化前は 2,196.85 MiB） |
| replay hash（NUB=32） | **`65dd1c56…` 一致**（2 回） |
| 最初の token | 182416 |

5-token（40 行）と decode（8 行）は VTCM に収まるためタイル経路を通らず、従来と同一の数値・hash を保った。NUB=32 でも CPU コピーが消えることを確認した。

### 2.3 実タイル経路のクラッシュ

prefill 512（ubatch 32 → 256 行、m_chunk により 2 チャンク）を実行した直後、`PHASE prefill_begin` の後に端末が再起動（00:32 JST、uptime リセット、boot_id 更新）。DSP のエラーはホストに届かず、そのままシステムが落ちた。

- 実タイル経路はこれが初回実行だった（5-token 40 行、decode 8 行はタイル不要）。
- 原因候補: チャンクあたりの DMA/VTCM レイアウトの不整合（チャンクサイズと `L` の再計算、`quant_*` のインデックス、`src1_stride` の扱い）。境界外アクセスが DSP フォルトを起こし、端末再起動に至った可能性が高い。
- 対応: 変更を **リバート**（`git checkout`）し、skel `76dd89e3`・host `c87765f7` に戻した。M タイル化は再試行しない。

## §3 端末状態（2026-11-23 00:38）

- 再起動 2 回目（16:11、00:32）。今回は M タイル化の prefill 512 実行が引き金。
- 復旧後: uptime 5 分、MemAvailable 6.9 GB（起動直後の app churn）、SwapFree 10.7 GB、`/data` のモデル 10,425,495,136 B、libs は安全版に復元。
- 再起動直後のため MemAvailable が 11 GB に届かず、C 構成（5.35 GB rpcmem）の run はメモリ監視で停止した。測定は控える。

## §4 次の一手

1. **M タイル化を再開するなら DSP 側の安全性レビューから**。チャンク境界の VTCM/DMA レイアウトを 1 チャンク実行で検証できる診断（小さい m_chunk を明示指定、canary、境界行の一致）を先に作り、prefill 512 を直接走らせない。
2. 常用構成は §17 の C 構成（NUB=8、down HTP 既定 23 層、hash `65dd1c56`）。NUB=32 は M タイル化が安全になるまで使わない。
3. 品質ゲートの NLL t 値が ±1 の外なので、余裕のある時間に 64 step で再確認する。
4. dotprod/tiled 制御ベンチ（Phase 60 統合）はデバイスに依存しない形で進められる。

## §5 成果物

- 本 report（c65）。ソースは HEAD `65f13b421` の状態（M タイル化はリバート済み）。
- 端末の lib: skel `76dd89e3…`、host `c87765f7…`（8-token cap、タイル化なし）。
- 再起動の記録: 00:32 JST、prefill 512 + M タイル経路が引き金。
