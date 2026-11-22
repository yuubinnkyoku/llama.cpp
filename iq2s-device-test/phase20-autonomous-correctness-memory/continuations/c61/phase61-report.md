# Phase 61 - tiled NEON の end-to-end 効果と NGL 再探索

対象: 2026-11-23 02:13 〜
作業 worktree: `D:\ghq\github.com\yuubinnkyoku\llama.cpp-c51`
ブランチ: `codex/c61-e2e`（`codex/c60-tiledarm` / `842dbb917` から作成）
push 許可: `codex/c61-e2e` のみ

---

## §0 目標0: 到達確認と回帰ゲート - **PASS（ただし重大な発見あり）**

### §0.1 到達確認

端末は 02:13 の時点で到達済み（`adb devices` に `324753221196`）。
uptime 6,105 s、tz0 37.3 ℃、MemAvailable 9.9 GB、cap0 **3,628,800 Hz（フル）**。
Phase 60 の測定時は 1.78 GHz に落ちていたが、今回は復帰直後で冷えていた。

### §0.2 unit stage（最初に実行）

| モード | PASS | FAIL |
| --- | --- | --- |
| 通常（13 型 + MUL_MAT_ID 2 件） | **15/15** | **0** |
| `--fuzz` | **88/88** | **0** |

Phase 60 で修正した NEON カーネルは端末で健在。**端末環境に変化なし。**

### §0.3 回帰ゲート

| run | 構成 | exit | sha256(logits.f32) | 既知値 | 判定 |
| --- | --- | --- | --- | --- | --- |
| `20261123-022119-21160` | 基準 `0a3db9c5`、NGL=44 | 0 | `e9da92f8d7280750f5c62a8bf3489fdc7332fc30e1774d708f1df7d65c0ec6c1` | 同左 | **一致** |
| `20261123-022233-26812` | 基準 `0a3db9c5`、NGL=0 | 0 | `44cd99ce376d5ce627ebbd4dc1c1d6cb90e83ea7822371f13a4874f921262295` | 同左 | **一致** |
| `20261123-025223-15834` | 基準 `0a3db9c5`、NGL=44（再確認） | 0 | `e9da92f8…` | 同左 | **一致** |

**目標0 の回帰は PASS。** 初回の swap guard 発動もなかった（冷キャッシュだったが
Phase 60 より MemAvailable が多く、swap に落ちなかった）。

### §0.4 重大な発見: 基準バイナリは tiled が完全にコンパイルアウトされている

指示の目標0-4「c60-neon + `GGML_CPU_TILED_MM=0` でも `e9da92f8` が一致することの裏取り」
を実行したところ、**一致しなかった**（`c7564354af84c751912a665853df42e484edf7aea9f40a8016df17bec19a2f6a`）。

原因を追い込んだ結果、**基準 `0a3db9c5` は `ggml_compute_forward_mul_mat_tiled` が
`mov w0, wzr; ret`（常に false）にコンパイルされている**ことが分かった:

```
$ llvm-objdump -d build/p52/devlibs/libggml-cpu.so --start-address=0x59e20
0000000000059e20 <ggml_compute_forward_mul_mat_tiled>:
   59e20:  mov  w0, wzr
   59e24:  ret
```

これは Phase 60 で ARM gate を開放する**前**のソース（`__ARM_NEON` が gate 条件に入って
いない版）でビルドされているため。つまり:

- **基準 `0a3db9c5` では tiled は絶対に走らない。** `e9da92f8` は純粋な stock パスの hash。
- Phase 60 以降のビルドは gate が入っているので、`GGML_CPU_TILED_MM=0` でも
  `ggml_tiled_wdata_size()` が 0 以外を返しうる（`ggml_tiled_supported` が型を見るため）。

### §0.5 なぜ hash が変わるのか（機構）

`ggml-cpu.c:1605` で tiled の作業バッファは `params->wdata` から取られる:

```c
char * tiled_scratch = incr_ptr_aligned(&wdata_cur, ggml_tiled_wdata_size(nth, dst), 64);
```

`ggml_tiled_wdata_size()` は `ggml_tiled_supported()` が真なら
`64 + n_tasks * 512KB` を返す。**この予約が 0 か非 0 かで、後続の全バッファの
オフセットがずれる。** オフセットがずれるとキャッシュライン境界が変わり、
スレッドの分割と浮動小数の加算順序が変わって hash が変わる。

**したがって「tiled を OFF にすれば基準と同じ hash」は成立しない。**
Phase 60 の報告書 §2.4.3 が FORCE 経路で品質ゲートを取ったのは正しかったが、
「`GGML_CPU_TILED_MM=0` で `44cd99ce` に戻る」という前提は**この fork では誤り**だった。

### §0.6 切り分けの記録（すべて実測、すべて `c7564354`）

| 仮説 | 検証 | 結果 |
| --- | --- | --- |
| dotprod が原因 | `c60scalar`（armv8-a、`sdot` 0 個）+ tiled OFF | `c7564354` → **否定** |
| OpenMP が原因 | `c61-neon`/`c61-scalar` を `GGML_OPENMP=ON` でビルド | `c7564354` → **否定** |
| `-march` が原因 | `c61-noarch`（`GGML_CPU_ARM_ARCH` 空、`-march` なし） | `c7564354` → **否定** |
| ARM gate が原因 | gate を Phase 60 前の形に戻してビルド（`c61-nogate`） | `c7564354` → **否定** |
| 基準バイナリ自体 | `0a3db9c5` を再実行 | `e9da92f8` → **環境は正常** |

**結論: 差は「tiled のコードがバイナリに入っているかどうか」そのもの。**
gate を戻しても消えないのは、`ggml_tiled_wdata_size` の呼び出しが
`ggml-cpu.c` に残っており、`ggml_tiled_supported` が型を見て真を返すため。

### §0.7 この発見の意味

1. **Phase 60 の「`GGML_CPU_TILED_MM=0` で `44cd99ce` に戻る」という記述は誤り。**
   Phase 60 報告書 §1.3 の表で「tiled-on と tiled-off が同一ハッシュ `ede05813`」と
   書いたのは、**両方とも tiled パスに入っていなかった**（1 token < 8 行）ため。
   あの比較は「tiled が効いていない」ことの証拠にはなるが、「OFF で基準に戻る」ことの
   証拠にはならない。
2. **今後の A/B は「基準バイナリ vs 新バイナリ」で取るしかない。** 同一バイナリ内の
   env 切り替えでは、wdata レイアウトが同じなので比較には使える（tiled ON vs OFF は
   同じバイナリなので wdata サイズは同じ）。**ただし基準 hash とは一致しない。**
3. **既定 ON の判断（目標2）はこの影響を受ける。** 既定 ON にすると
   `e9da92f8` は二度と出なくなる。これは「退行」ではなく「tiled が入ったことによる
   正当な変化」だが、**回帰ゲートの基準を取り直す必要がある。**

---
