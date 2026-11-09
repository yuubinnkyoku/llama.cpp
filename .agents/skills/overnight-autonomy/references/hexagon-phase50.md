# Hexagon / llama.cpp：Phase 50 の基準

Phase 49 で確定した数値と、Phase 50 で守る測定条件をまとめる。
規則の本体は `../SKILL.md` と `hexagon.md`。Phase 49 の作業契約は `hexagon-phase49.md`。
このファイルは Phase 50 以降の /goal で最初に読む。

## 確定ベースライン（Phase 49、同一ビルド `533b03638`）

標準構成は **NGL=16 / `GGML_HEXAGON_SPLIT=0` / bf16 カーネル有効 / IQ3 gather 有効**。
CPU 側の比較対象は **NGL=0**（正規の CPU 経路）。NGL=24/32 は追加評価のみ。

熱制限下（cap 2112000 / 1862400、8 step、step 1-7 の中央値）:

| 構成 | ms/token |
|---|---:|
| CPU 単独 NGL=0 | 約 735 |
| NGL=9 SPLIT=0 | 約 660 |
| NGL=16 SPLIT=0 | 約 645 |

fast cap（cap 3628800 / 4608000）:

| 構成 | ms/token |
|---|---:|
| CPU 単独 NGL=0 | 467-482 |
| NGL=9 SPLIT=0 | 467-474 |
| NGL=16 SPLIT=0 | 476 |

fast cap は同等であり、性能 PASS ではない。**熱制限下のみ PASS**。
条件の違う数値を直接比べない。Phase 50 では同じ手順で取り直した数値を優先する。

DSP 実時間は 1 token あたり約 59 ms（IQ3 gather 最適化後）。CPU split は 394-401 ms（NGL=9、fast cap）。
この 2 つは並列に走るので、足し算を token 時間とみなさない。

## 周波数と画面状態（最重要）

- 画面 OFF（Dozing）では CPU が **883200 / 768000** に固定される。この状態の数値は測定に使わない。
- 画面 ON で fast cap 3628800 / 4608000。負荷が続くと温度制御が **2112000 / 1862400** へ落とす。
- 充電していない端末では `svc power stayon true` は効かない。`input keyevent KEYCODE_WAKEUP` を
  20 秒ごとに送る以外に画面を保つ方法は確認できていない（2026-11-10 実測）。
- **試験の前後で `scaling_max_freq` を記録し、変わった run は捨てる。** 開始時だけの記録では足りない。
  実際に 8 step の途中で 3628800 -> 2112000 へ落ちた例がある。
- DSP クロックは 2112 MHz 固定。熱制限で DSP が遅くなるとは書かない。

## メモリ前提

- `run49.sh` の前提は MemAvailable 9500000 kB 以上、SwapFree 8000000 kB 以上、residency 95% 以上。
- 背景アプリが anon page を持つと、モデルの page cache を広げるために swap-out が増える。
  1024 MiB の swap-out ガードは**緩めない**。ガードに掛かった run は捨てて環境を整えてから取り直す。
- 実測（2026-11-10）: 背景アプリを止めないと NGL=0 の 1 run で 1073-1139 MiB の swap-out が出て
  ガードに掛かる。止めれば通る。Phase 50 でも試験前に止める。

## 測定手順（固定）

1. 背景アプリを止める（`am force-stop` の対象は下のスクリプトと同じ）。
2. 画面を起こし、cap が fast か throttled かを記録する。どちらの系列かは run ごとに記録する。
3. `GGML_SCHED_SPLIT_TIMING=1 GGML_SCHED_DEBUG=1` で 8 step 以上を回す。
4. ABBA で交互に最低 4 組。同じ cap の組だけを比較する。
5. step 0 を含む中央値と step 1-7 の中央値を両方出す。ばらつき（最小-最大）も出す。
6. 差が 5% 未満なら、その組では「差なし」と書く。1-2% の差を PASS にしない。
7. 試験は 1 回 180 秒以内。swap-out 1024 MiB、DSP 異常、OOM 兆候、強い熱制限で即停止。

再現スクリプトは `iq2s-device-test/phase20-autonomous-correctness-memory/continuations/c49/` の
`p495_run.sh`（端末側ラッパー）と `p495_abba.sh`（交互比較）を使う。

## Phase 50 の合格条件（提案）

性能:

- fast cap で、同一条件の NGL=0 より NGL=16 が再現性をもって速いこと。目標は 5% 以上。
- 5% 未満しか出ない場合は、その時点のばらつきを測り、閾値を上げ下げするのではなく
  「未達」として記録する。熱制限下の優位（約 12%）を壊さないこと。

正当性:

- bf16 単独試験 PASS（65536 パターンのビット一致、M=1/2/3、K=5120/5152/96、非有限 0）。
- IQ3 単独試験 PASS（4 step logits が基準 `5585d4ed...` とバイト一致）。
- 2 / 4 / 16 step が exit 0、非有限 logits なし、CPU split の logits が基準 `577417e9...` と一致。
- 数値契約を緩めない。既存の 33 テンソル分割のビット一致を壊さない。

安全:

- メモリガード違反なし、DSP 異常なし、CPU 単独経路の回帰なし。

比較:

- 最適化の前後で同じモデル・NGL・cap・入力（同じ replay token 列）を使う。
- 変更前後の比較と CPU 単独との比較を両方残す。

## Phase 50 で優先する作業（Phase 49.5 の実測から）

1. **CPU split の削減**。CPU 側は 1 token で 9152.8 MiB を読み、その 89% が MUL_MAT の重み。
   1 個の遅い演算を直す話ではなく、重みを読む量か、担当する層の数かを減らすしかない。
2. **fast cap での CPU/DSP 重複**。CPU split と DSP 実行は一部しか重ならない。
3. **IQ2_S / IQ2_XS の gather 最適化**。合計約 6 ms/token なので上限は小さい。優先度は低い。
4. **32x2 の IQ3 実装**。prefill 専用。decode の改善として計上しない。
5. **キャッシュ dirty 区間と flush**。実費用が小さいなら後回し。整合性の別ゲートが要る。
