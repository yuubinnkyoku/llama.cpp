# Hexagon / llama.cpp：夜間自律開発の追加規則

## 現在地（Phase 48）
- 最新のPhase 48引き継ぎ (`continuations/c48/phase48-round1-dsp-analysis.md`、
  `phase48-round2-boundary-and-ratio.md`、`phase48-morning-report.md`)、現行ソース・実測・Git状態を
  最初に照合する。古い資料より新しい実装と検証結果を優先する。
- 検証済み基準は `codex/c48-perf`（base は Phase 47 の `5673eaa41`）。行分割は
  `blk.57.ffn_up.weight` ほか **33テンソル**（IQ3_S / IQ3_XXS）、CPU比率の既定は **0.5**。
  34本目 `blk.63.attn_output.weight` は単独では動作するが、34本・4stepで不安定なので既定にしない。
- **DSP のクロックは 2112 MHz 固定**。CPU が 3628 -> 2112 -> 1785 MHz と絞られても DSP は変わらない。
  「熱制限で DSP が遅くなる」と書かない。
- DSP の 1 decode step の実行時間は **95 ms**（NGL=9）。ホストが応答待ちした実時間で測る。
  MUL_MAT が 97 %。型ごとの実効速度（DSP、n=1）: iq4_xs 67 / q4_K 63 / q2_K 46 GB/s に対し、
  **iq3_s 14 / iq3_xxs 13 / iq2_s 10 / iq2_xs 8 GB/s**。LUT gather 型は nibble unpack 型の 1/4-1/8。

## 熱制限下の損失の読み方
- 1 step = CPU split（layer 0-54）約 717 ms + DSP 実行 95 ms + **HTP split の往復 約 163 ms**。
  往復の内訳は sched が 1 token に HTP 13 + CPU 13 の split を作るため。原因は gated delta net の
  `blk.NN.ssm_alpha.weight` / `ssm_beta.weight`（bf16, 5120x48）が HTP 型ゲートで落ちること。
  1 layer あたり HTP -> CPU -> HTP -> CPU -> HTP と 4 回往復する。
- 往復 1 回は 7-12 ms。NGL sweep の傾き（+15 - +35 ms/layer）と整合する。
- この往復を消すには BF16 の MUL_MAT を HTP で受ける必要がある（DSP に bf16 の vec_dot は無い。
  bf16 -> f32 は 16 bit 左シフトで厳密）。未実装。SOFTPLUS を CPU へ落とす案は split が増えて逆効果。
- したがって現状で最速の構成は **NGL=0（正規の CPU 実行）**: cap 1785 で 828 ms、cap 2112 で 684 ms。
  NGL=9 は同じ cap で 1018 / 864 ms。**HTP を使うと 20-26 % 遅い**。

## 検証・測定の契約
- 正当性はCPU担当行＝同一CPU参照のビット一致、HTP担当行＝同一入力フルHTP参照のビット一致、
  全行有限・センチネル0・境界行・2->4ステップを検証する。可能なら16ステップの安定性も確認する。
  未比較を `nonbit=0` と表現しない。新しい分割行数は32行単位と出力境界の整列を検証する。
- split の CPU シェアと DSP シェアは 1 op の中で重なる（`wait_us` は 2-200 us）。`wait_us` が
  大きいときだけ DSP シェアが遅いと判断する。`GGML_HEXAGON_SPLIT_TRACE=1` で per op に出る。
- CPU シェアの実効速度は 6.7-7.5 GB/s。スレッド数依存が強く、8 にすると 2.5 倍遅くなる（既定 6）。
- 非同期投入の復帰・未回収キュー・短い待ち時間だけでDSP内部の演算重複を断定しない。同期前に
  DSP出力をCPUで読まない。DSP側時刻がなければ重複は未確定と記録する。
- 既知の実験条件は `GGML_HEXAGON_DMA64=0`、`GGML_HEXAGON_OPPOLL=0`。`GGML_HEXAGON_PROFILE=1`
  （per op の usec/mhz）、`GGML_HEXAGON_TIMELINE=1`（split ごとの submit/wait/prep）、
  `GGML_SCHED_DEBUG=1`（split 構成）を最初に使う。
- 生成速度は**同じNGL・入力・周波数/温度条件**で複数組比較し、分散を含め評価する。cap は必ず記録し、
  途中で cap が変わった run は別 tier として扱う。1 ブロック 5 step では 1 % 差は読めない。
  CPU 単独の値は NGL=0 で測る。`CPU_FRAC=1.0` を完全 CPU の代用にしない。
- **安全ガードは解除しない。** 1試験180秒以内、Swap-out増分1024 MiB到達、DSP異常、OOM兆候、
  強い熱制限では当該試験を停止。温度保護・ZRAM・重要サービスには介入しない。ADBがTCP接続だけなら
  端末再起動は禁止。復旧不能ならホスト側作業に切り替える。
- 終了時はPASS/PARTIAL/FAIL/BLOCKED、実測と未検証、変更内容、コミット、再現コマンド、次の一手を
  簡潔に残す。安全な代替策が残る限り確認待ちで止まらない。
