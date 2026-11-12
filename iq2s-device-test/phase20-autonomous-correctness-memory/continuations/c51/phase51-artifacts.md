# Phase 51 成果物の置き場所とハッシュ

大きなログ・バイナリは git に入れない。ここに場所とハッシュだけを残す。

## 1. 端末側 (残したもの)

一次資料は `/data/local/tmp/c46-27b/` にある。作業で読んだものだけ挙げる。

| ファイル | 内容 | sha256 |
|---|---|---|
| `A.log` `A2.log` `A3.log` `A4.log` `B.log` | 単一テンソル分割、DMA64=0 | - |
| `C.log` `D.log` | 同、2 iteration | - |
| `E1.log` `E2.log` | 同、DMA64=1 で `dspqueue_read failed: 0x0000002e` | - |
| `p47-f33-*.log` `p47-f34-*.log` `p47-f35-*.log` `p47-f5-*.log` `p47-f6-*.log` | Phase 47 の ABBA (ngl 9、5 step) | tar: f33logs / f34 / f35 / f5logs / f56logs |
| `p47-a34.out` `p47-t33.out` `p47-f33.out` `p47-f34.out` | 同、集計側 | - |
| `all34` `all34.ref` | 34 op の分割ダンプ | 5923915810a71cb7d7f5e8a3c2d8ef77 / 11fa33b4a36736ceb4d17e7a2b41cadc |
| `split-async.bin` `split-seq.bin` | 合成ハーネスの dump (両者同一) | 82d68f598f262e01944f17b01cc1bf37 (両方) |
| `split-ngl9.bin` | 実モデルの dump | f422767601c889b4002d78af70d3a5c4 |
| `split33.txt` / `split34.txt` | 分割リスト (34 本目 = `blk.63.attn_output.weight`) | `libs/` 配下 |
| `c46_split_repeat` | Phase 46 の単一テンソル合成ハーネス (source は repo に無い) | 326408b5a95f0226579d0e98f53b30e2 |

`c47` / `c48` / `c49` の主報告書は、このリポジトリのどのブランチにも存在しない。
Phase 47 の判断根拠は上記の生ログだけである。

## 2. 手元に落としたログ (git 管理外)

`continuations/c51/air/` に置いてある (22 MB)。`.gitignore` で除外。

- `A.log` から `E2.log`、`p47-*.out`、`gate5*.log`、`timing.log`、`run_abcd.sh`、`run_repeat.sh`
- `ex-f33logs/` `ex-f34/` `ex-f35/` `ex-f56logs/` `ex-f5logs/` `ex-f6logs/`: Phase 47 の生ログ
- `s33t1.stderr.txt` (frac 0.5 trace)、`f25t1.stderr.txt` (0.25)、`f125t1.stderr.txt` (0.125)、
  `ax2.stderr.txt` (両修正 0.25)、`prof9d.stderr.txt` (SPLIT=0 profile)
- `lg-nosplit-ngl16.f32`、`lg-f25-oldlib.f32`、`lg-f25-c51base.f32`、
  `lg-f25-c51base-r2.f32`、`lg-f25-cachefix.f32`

## 3. logits のハッシュ (NGL=9 は 4 step replay、明記なければ)

| sha256 (先頭 32) | 条件 |
|---|---|
| `f3a0d235c0da32d0c40ed901b9236026` | SPLIT=0、NGL=16。Phase 50 の基準と一致 |
| `5585d4edc47ee7505c558aeb1b929a7e` | SPLIT=0、NGL=9。Phase 50 の ngl=9 基準と一致 |
| `2d4693c43c0cc307e0cafbeb60e6029c` | 33 本、frac 0.25、`seq`。修正後の `async` も同じ値 |
| `276f689efe42c8d74bb831f7c5cba014` | 33 本、frac 0.5、修正前 async (再現せず) |
| `788611da9d15a5fb8cca827a10c1049f` | 33 本、frac 0.25、Phase 50 配備 lib の async |
| `9456dc7458643944d6b0acbe35663377` | 33 本、frac 0.25、c51 base lib の async |
| `019160106423dc12778369da578d3265b` | 同上を再実行 (値が違う = 再現しない) |
| `b1f2ab9892aa9180bc49b795d368eeb3` | 33 本、frac 0.125、修正前 async |
| `c38fcdc0237db202a36cfb4b56b07513` | 33 本、frac 0.25、cache 修正のみ |
| `4d4c311786f735f8f2eb874c3735bc05` | 34 本、CPU_ROWS=32、NGL=16、修正前 async |

## 4. ライブラリのハッシュ

| 名前 | sha256 | 内容 |
|---|---|---|
| skel (DSP) | `288e6712df9c8201ef05c12ec5704968d4f71407992cff8e0f03aa002ee0861d` | Phase 50 の D。c50 のビルドと一致 |
| host 配備 (Phase 50) | `d6f60f65e2dc3ca18eadd95c7f6839758d8d82f716ef577a3bba15af68f3ec08` | 11-09 02:02 のビルド。c50 の build-c50 とは別物 |
| c51 base | `5c55b585dd7378316d59da8245c9f87dd72c5eea250a7f313462349c281b4da7` | c51 HEAD 相当、修正なし |
| c51 async 修正のみ | `67b867f87cb109622adfdcfc4e92af428a0042e6a6e4e8a08cd23bc7f819badf` | bb2607d4f |
| c51 両修正 | `c646b4db3817f6eacbdb99617289ece59cee4358f8f4b28a624b5cd9a96a6aaf` | bb2607d4f + 302685f2d |
| 副産物 | `libggml-hexagon.so.c51` `.c51base` `.c51async` `.c51both` `.c51t3` `.c51final` `.p50deployed` | 端末 `libs/` に保存済み |
| c51 3 スレッド既定 | `90e62d01ff8649e085464aaec203c2fde267a3e95db4eee4c45a615f41df9ab3` | 36b407da5 相当 |
| c51 最終 (コメント整理後) | `44baedd81e5d7498b89430c2b9a60c612a0f14903be74de267780ab9b2c8c7e0` | 1cad3f770、端末に配備済み |

skel は c51 でビルドすると `61e598ea...` になる (commit 文字列の埋め込みで変わる)。
数値の基準を動かさないため、端末の skel は Phase 50 の D のまま固定した。
