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
