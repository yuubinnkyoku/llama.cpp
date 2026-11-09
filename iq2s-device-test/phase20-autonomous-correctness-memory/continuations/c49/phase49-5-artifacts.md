# Phase 49.5 - 保存済みの成果物と、Git に入れない原本

`continuations/c49/` は `.gitignore` の対象 (`/iq2s-device-test/`)。このフェーズの報告と
再現スクリプトだけを `git add -f` で追跡し、巨大な原本は端末とローカルに置いたままにする。
原本の場所とハッシュをここに残す。

## 追跡するファイル

| ファイル | 内容 |
|---|---|
| `phase49-5-report.md` | Phase 49.5 の報告 (判定: READY WITH RISKS) |
| `p495_run.sh` | 端末側 1 run ラッパー。背景アプリ停止、画面維持、cap の前後記録 |
| `p495_abba.sh` | 端末側 交互比較。cap が変わった run を CAP_CHANGED として捨てる |
| `p495_sched_timing_test.cpp` | sched 計測の独立試験 (mix / async モード) |
| `p495_cpu_matvec_test.cpp` | CPU 側 matvec 帯域の実測 |
| `p495_logits_cmp.py` | logits 比較 (最大/平均/相対 L2/RMS/順位/softmax 距離) |
| `p495_graph_ops.py` | グラフダンプから演算別のノード数とバイト数 |
| `p495_mulmat_classes.py` | MUL_MAT を重み種別に分類し、1 token の読み出し量を出す |
| `p495-bf16-test.log` | bf16 回帰試験の生出力 (2026-11-10 00:56) |
| `p495-mix.err` | sched 計測 mix モードの生ログ |
| `p495-mixoff.err` | 計測無効時のログ (sched 行が出ないことの証拠) |

## Git に入れない原本 (ローカルのみ)

| ファイル | サイズ | sha256 / 備考 |
|---|---:|---|
| `lg-ngl0.f32` | 3973120 | `577417e9fd33912282f964f8c99e250fadc1b5605453f04fcd5bbea23ef3560b` |
| `lg-c495-ngl0.f32` | 3973120 | 同じ (2026-11-10 00:20 に再取得) |
| `lg-ngl9.f32` | 3973120 | `5585d4edc47ee7505c558aeb1b929a7eb7089e238d1b2d81a3ffcfa4958bd480` |
| `lg-c495-ngl9.f32` | 3973120 | 同じ (2026-11-10 00:23 に再取得、IQ3 回帰) |
| `lg-g2.f32` | 3973120 | 同じ (Phase 49 の IQ3 変更後) |
| `lg-c48.f32` | 3973120 | `4f2bd63cf6821bdc66f0e8a2038d55c50b92962c6326c56010f9891c68ef2c0b` (Phase 48 HTP) |
| `p495-d2cpu-stderr.txt` | 3328246 | NGL=0 2 step のグラフダンプ (2024 ノード x 8 グラフ) |
| `p49-prof-g1.log` / `p49-prof-g2.log` | 306272 / 305303 | Phase 49 の IQ3 前後のカーネルプロファイル |
| `p49-abba-last.log` | 18596 | Phase 49 の ABBA 生ログ |

logits は 4 step x 248320 vocab x 4 byte。比較は `p495_logits_cmp.py` で再現できる。

## 端末上の原本

`/data/local/tmp/c46-27b/` (serial は `adb devices` のもの):

- `runs/<runid>/logits.f32`, `stdout.txt`, `stderr.txt`, `manifest.txt`
- 2026-11-10 の run: `20261110-002043-27093` (NGL=0 4 step)、`20261110-002308-628` (NGL=9 4 step)、
  `20261110-002515-8512` (NGL=0 2 step グラフダンプ)
- ABBA の生ログ: `p495-abba.log`
