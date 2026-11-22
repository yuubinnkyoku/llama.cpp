# Phase 66 - 開始記録: CPU variant A/B の試行（無効）とゲート運用

## §0 状態

- npu-gate 運用を開始。端末操作はすべて `gate with` 経由。
- 02:14 の `gate status`: lock free、queue 0、token none、MemAvailable 6 GB。
- 02:21 の `gate status`: **lock HELD project=c61**（heartbeat 1 秒、正常保持）。他プロジェクトが使用中のため、以後の端末操作は待ち。
- Phase 66 のタスク状況:
  - タスク 2（64-step 品質ゲート）: **BLOCKED**。再起動後のアイドル時 MemAvailable が 9.3〜10.0 GB で、C 構成の前提 11 GB に届かない。
  - タスク 1（M タイル化の DSP 安全性診断）: 未着手。11 GB と診断ビルドが必要。
  - タスク 3（CPU variant の同一セッション A/B）: ハーネスを作成して試行したが、**この試行は無効**（下記）。

## §1 タスク 3 の試行（無効）

目的: NDK 既定 CPU バックエンド（`ac7cc94c…`）と `-march=armv8.7-a+dotprod+i8mm` ビルド（`a3b7d4e7…`）の decode 性能を、同一セッション内の交互 run（A/B/B/A）で分離する。

実施: `gate with --project inkling2` の 1 回で、variant ディレクトリ作成 → CPU lib 2 種を配置 → down CPU 構成（override-down-cpu、NUB=8、16 step）を 4 回。

無効だった理由:

1. **MSYS のパス変換で push が失敗**。`adb push <local> /data/local/tmp/...` の宛先が `C:/Program Files/Git/data/local/...` に変換され、`libggml-cpu-base.so` / `-dot.so` が端末に置かれなかった。結果、variant-base と variant-dot の両方が既定 CPU lib（`ac7cc94c`）になり、A/B が成立していない。
2. **全 run が swap guard 停止**。開始 MemAvailable 8.6 GB で、モデル load の page cache churn により 1 run あたり swap 増分 393,833〜419,029 pages（1.50〜1.60 GiB）となり、1.5 GiB の上限を超えて全 4 run が 16 step 完走前に停止した。RESULT 行なし。
3. したがってこのセッションの decode 数値は破棄する（記録もしない）。

端末への副作用: 各 run 後にモデルは解放され、MemAvailable 10.0〜10.5 GB に回復。追加のバイナリは端末に残っていない（variant ディレクトリは baseline のコピー）。

## §2 次回の修正

- push は `MSYS_NO_PATHCONV=1` を付ける（または `//data/...` 形式）。ゲートスクリプト内で `export MSYS_NO_PATHCONV=1`。
- A/B の前提: 開始 MemAvailable 10.5 GB 以上（CPU-only 構成でも load churn で 1.5 GiB 近く出るため）。11 GB 未満なら実施しない。
- run 数は A/B/B/A、step は 16。1 run の swap が 1.5 GiB を超えたらその run を破棄し、A/B を成立させない。
- C 構成の測定（タスク 2）は 11 GB 必須のまま。再起動後の現状では BLOCKED。

## §3 ゲート状態（報告用）

```
== NPU gate status 2026-11-23 02:21:15 ==
lock:   HELD project=c61 pid=1279 since=2026-11-23 02:21:06 heartbeat_age=1s
queue:  0 jobs
token:  c61 pid=1279 at=2026-11-23 02:21:06
mem:    MemAvailable=10 GB (10818244 kB)
```

他プロジェクト（c61）がロックを保持している。取得は無期限待機のため、次回は `gate with` をそのまま打てば順番に実行される。

## §4 成果物

- `c66/ab_dotprod.sh`: CPU variant A/B ハーネス（次回修正版の基礎）
- `c66/p62_probe.sh`: `LIBDIR` 対応（ライブラリディレクトリを切り替えて同一 probe を実行）
- CPU lib 2 種の sha256: base `ac7cc94c1aa5d2516f5f7dd9a5829c552b940b07edb228c82f9883cda2bd29ce`、dot `a3b7d4e7d330d0bebe19ff3b5aa2deae62e47592f26875b2c09ce61f2f58bf0f`
- 端末状態（02:21）: uptime 1 時間 49 分（00:32 の再起動後、追加再起動なし）、MemAvailable 10 GB、swap 残 11.8 GB、`/data` 空き 7.1 GB、残存 probe なし。

## §5 追記（02:40）: A/B ハーネス成立、データは示唆的だが未確定

修正点:
- MSYS のパス変換は `export MSYS_NO_PATHCONV=1` で解決（Windows 形式のローカルパス + リモートは `/data/...` のまま）。
- probe script の CRLF を LF に正規化（Android の sh は `set -u\r` を解釈できない）。
- variant lib の sha256 を run 前に検証（base `ac7cc94c`、dot `a3b7d4e7`）。

結果（down CPU 構成、NUB=8、16 step、A/B/B/A、warmup 1 回、開始 MemAvailable 11.5 GB）:

| run | variant | decode ms/step | swap (pages) |
| --- | --- | --- | --- |
| warm | base | 888.1 | 232,427 |
| a1 | base | 687.9 | 146,492 |
| b1 | dot | 707.6 | 32,818 |
| b2 | dot | 837.8 | 197 |
| a2 | base | 1,129.3 | 2,479 |

- base 平均 908.6、dot 平均 772.7（-15%）だが、base の幅 688〜1,129 と重なり、run 間の熱/cache ドリフトを分離できていない。**この 4 run では採用判断しない。**
- 全 run で最初の token は 182416。logit は base 27.2587242、dot 27.2549171 と差がある（SIMD コード生成による数値差。hash が変わるため、CPU variant の採用時も基準 hash の再凍結が必要）。
- ハーネス自体は成立（variant の実体を sha256 で確認、同一セッション近接、swap 記録）。次は `test-backend-ops perf` 相当の IQ2_XXS マイクロベンチか、反復回数を増やした A/B/B/A/A/B/B/A で判断する。

## §6 端末・ゲート状態（02:40）

- 02:21〜02:40 の間、別プロジェクト c61（`llama.cpp-c51` の 27B e2e）がゲートを使用。こちらの `gate with` は待機後に実行された（キューとして機能）。
- 端末: uptime 1 時間 48 分（00:32 の再起動後、追加再起動なし）、MemAvailable 11.5 GB（run 前）、`/data` 空き 7.1 GB、残存 probe なし。
- 変更はコミット `4af0c62f1`（ハーネス初版）に続き、本追記と LF 修正を追加コミットする。

## §7 追記（02:52）: 64-step 品質ゲートは測定不成立（swap guard）

タスク 2 を `gate with --project inkling2 --min-mem 11` で試行（ゲートの 11 GB チェックは OK、開始 MemAvailable 11.5 GB）。

| run | HTP model buffer | 完走 step | swap 増分 | 判定 |
| --- | --- | --- | --- | --- |
| cpu64（down CPU 参照） | 1,157.11 MiB | 24 / 64 で guard 停止 | 415,677 pages = 1.59 GiB | 無効 |
| htp64（既定 23 層） | 5,348.86 MiB | 1 / 64 で guard 停止 | 426,588 pages = 1.63 GiB | 無効 |

- どちらも最初の token は 182416（cpu logit 27.2587242、htp 27.2482471）で配置は正しい（buffer サイズで確認）。しかし 64 step の途中で swap 増分が 1.5 GiB を超え、guard が停止した。
- 原因: 冷えた page cache でのモデル load churn。32-step の基準 run が warmup 後の 0.008 GiB だったのに対し、今回は冷間で 1.5 GiB を超えた。
- 対策（次回）: 各構成で warmup run（8 step）を先に走らせ、その直後に測定 run を置く。warmup の swap は測定 run に数えない（測定 run の guard はそのまま）。
- 32-step の品質ゲート PASS は不変。64-step は任意タスクのため、測定成立を優先してここで打ち切る。

端末（02:51、ゲート経由で確認）: uptime 2 時間 18 分（00:32 再起動後、追加再起動なし）、MemAvailable 11.3 GB、SwapFree 10.9 GB、`/data` 空き 4.9 GB、残存 probe なし。

## §8 追記（04:30）: 64-step 品質ゲート PASS、凍結 hash 再確認、CPU variant A/B v3

### 8.1 64-step 品質ゲート（warmup 付きで成立）

`gate with --project inkling2 --min-mem 11`（04:15 にロック取得、MemAvailable 12.16 GB で開始）。

| run | HTP model buffer | decode | swap 増分 | 判定 |
| --- | --- | --- | --- | --- |
| warm_cpu（8 step） | 1,157.11 MiB | 2,056.6 ms/step | 0.92 GiB | warmup（判定外） |
| **cpu64** | 1,157.11 MiB | **1,376.0 ms/step** | **0.43 GiB** | **有効** |
| warm_htp（8 step） | 5,348.86 MiB | 1,112.8 ms/step | 0 GiB | warmup（判定外） |
| **htp64** | 5,348.86 MiB | **696.0 ms/step** | **0.48 GiB** | **有効** |

- warmup を各構成の直前に置くことで、64-step の swap を 1.5 GiB 未満に収められた（前回の冷間 1.59/1.63 GiB から改善）。
- 品質ゲート（64 step、down CPU 参照 vs down HTP 既定 23 層）:

| 指標 | 値 | 基準 | 判定 |
| --- | --- | --- | --- |
| argmax 一致 | **64 / 64** | 31/32 相当 | **PASS** |
| NLL 平均差（HTP - CPU） | **+0.001747** | 有意差なし | PASS |
| NLL 差の t 値 | **+0.8330** | ±1 | **PASS**（32 step の -1.31 も解消） |
| logit 差 mean_abs / max_abs | 0.1794 / 2.4251 | - | 小さい |
| 相関（最小 / 平均） | 0.99152 / 0.99868 | - | 高い |
| CPU argmax が HTP top10 内 | 64 / 64 | - | PASS |

- 同じセッションでの decode は cpu64 1,376.0 ms/step に対し htp64 696.0 ms/step（約 2.0 倍）。

### 8.2 凍結 hash の再確認

`gate with --min-mem 11` で 2 run:
- h1: `65dd1c563da7f1fb24dc53a362d4acff4afb184bd98f04df96f4d513d5ca661d`（HTP buffer 5,348.86 MiB、swap 18,726 pages）
- h2: `65dd1c563da7f1fb24dc53a362d4acff4afb184bd98f04df96f4d513d5ca661d`（5,348.86 MiB、swap 0）
- 最初の token は 182416、logit 27.2482471。**凍結基準は c66 の変更後も不変。**

### 8.3 CPU variant A/B v3（8 run、参考）

variant の sha256（base `ac7cc94c`、dot `a3b7d4e7`）を確認した同一セッション 8 run。warmup 込み。b3 はホスト adb デーモン再起動で欠測、b4 は 249.4 ms/step の外れ値。

- base: a1 905.9 / a2 809.4 / a3 1,055.7 / a4 1,337.4 ms/step
- dot: b1 602.8 / b2 737.1 ms/step（有効 2 点）
- dot が速い方向だが、ドリフトと外れ値で確定不能。**採用判断は保留。** 次はマイクロベンチ。

### 8.4 端末・ゲート状態（04:26〜04:29）

- 端末: uptime 3 時間 56 分（00:32 の再起動後、追加再起動なし）、MemAvailable 12.2 GB、SwapFree 10.9 GB、`/data` 空き 4.8 GB、残存 probe なし。
- gate status（04:26:31）: lock HELD project=c61（pid 13000、heartbeat 1s）。他プロジェクトの run が継続中。こちらは待機後に実行。
