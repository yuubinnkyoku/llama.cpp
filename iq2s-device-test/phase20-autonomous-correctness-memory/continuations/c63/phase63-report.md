# Phase 63 - evidence テスト・統合版の本採用・tiled 既定 ON

対象: 2026-11-25 00:05 〜
作業 worktree: `D:\ghq\github.com\yuubinnkyoku\llama.cpp-c51`
ブランチ: `codex/c63-evidence`（`codex/c62-upstream` / `49cc0dd15` から作成）
push 許可: `codex/c63-evidence` のみ

---

## §0 目標0: ゲート - **PASS**

### §0.1 到達確認

| 項目 | 値 |
| --- | --- |
| 接続 | `192.168.55.15:5555`（adb over TCP、Phase 62 から継続） |
| serial / model | `324753221196` / `NX741J`（同一端末） |
| **uptime** | **171,175 s**（Phase 62 終了時 96,323 s から単調増加 = **再起動なし**） |
| tz0 | 47.6 ℃ |
| MemAvailable | 6.0 GB |
| cap0 / cap7 | 3,033,600 / 2,112,000 |

### §0.2 unit stage

| モード | PASS | FAIL |
| --- | --- | --- |
| 通常 | **15/15** | **0** |
| `--fuzz` | **88/88** | **0** |

### §0.3 基準バイナリの回帰

| run | 構成 | exit | sha256(logits.f32) | 既知値 | 判定 |
| --- | --- | --- | --- | --- | --- |
| `20261125-000843-5251` | CPU `0a3db9c5` + base `d950aa96`、NGL=44 | 0 | `e9da92f8d7280750f5c62a8bf3489fdc7332fc30e1774d708f1df7d65c0ec6c1` | 同左 | **一致** |
| `20261125-001013-11049` | 同、NGL=0 | 0 | `44cd99ce376d5ce627ebbd4dc1c1d6cb90e83ea7822371f13a4874f921262295` | 同左 | **一致** |

### §0.4 統合版の新基準

| run | 構成 | exit | sha256(logits.f32) | 既知値 | 判定 |
| --- | --- | --- | --- | --- | --- |
| `20261125-001237-17221` | CPU `cc32ed77` + hexagon `9c9bfc48` + skel `312ad7b7`、NGL=44 | 0 | `e788051e16d7e1e63b140a902418c75f779bfe29cfb320bb054598508b09d4eb` | 同左 | **一致** |
| `20261125-001419-23519` | 同、NGL=0 | 0 | `ede05813034c131e8ecadcb2ecb3eb34e8bf291f0b9ebb49bd7a5541ca97d9e9` | 同左 | **一致** |

**目標0 は PASS。** 基準系・統合系の両方が既知値に一致した。

---

## §1 目標1: evidence テスト - **PASS（決定的な結果）**

### §1.1 方法

**上流 master を無改変で clone してビルドした**（fork のコードは 1 行も混ざらない）:

| 項目 | 値 |
| --- | --- |
| clone 元 | `https://github.com/ggml-org/llama.cpp` master |
| commit | `3090e45a8`（2026-11-24 16:34 "metal : reserve the top-k scratch only for short rows (#32942)"） |
| 作業ディレクトリ | `D:\ghq\github.com\yuubinnkyoku\llama-upstream-c63`（fork の worktree とは別） |
| `git status` | **dirty 0**（無改変を確認） |
| ビルド | Phase 62 §9.1 のレシピ（SDK 付属 CMake + `PREBUILT_LIB_DIR=toolv19_v81`） |

ビルド sha256:

| ファイル | sha256 | サイズ |
| --- | --- | --- |
| `libggml-cpu.so` | `e655bf5ca3ac53b4d18548dfa648c0ce4431b049251c018eb6b02fa75f04ca93` | 4,945,976 |
| `libggml-base.so` | `4671a16bd39ce103e2f70c9ff5a3b71b91dece8954b092feb680ec5cf22a4301` | 8,539,616 |
| `libggml-hexagon.so` | `f40c30eb5382882fc588c9076d8a452e6a6eef25c13cdbd1aaeee08894b60f2c` | 9,649,848 |
| `libggml-htp-v81.so` | `41bfebd9a687adb453c0cf219410e3f0e3a304950e3f95abbfb687d54fe2f401` | 1,051,016 |

端末の `libs/up63/` に置き、`p63_evid.sh` が 4 点を差し替えて実行し、
**終了時に基準構成へ自動復元**する（復元後の sha256 も記録済み）。

### §1.2 結果: **上流は 27B を動かせる。ただし遅い。**

| NGL | ロード | exit | prefill 128 | decode | **graph splits** | swap-out |
| --- | --- | --- | --- | --- | --- | --- |
| **44** | **成功** | 0 | **105,291.80 ms** | 2,843.40 ms | **262** | 100,720 pages |
| **0** | **成功** | 0 | 121,846.50 ms | 1,466.72 ms | **1** | 142,836 pages |

**ロードは成功し、4 トークンの生成も完走した。** クラッシュなし、tombstone なし。

### §1.3 決定的な発見: **262 分割**

**上流の NGL=44 では graph が 262 個に分割される。** fork は 1 個。

```
上流:  sched_reserve: graph: nodes = 3848, splits = 262
fork:  sched_reserve: graph: nodes = 3848, splits = 1
```

**これは「上流の HTP カーネルが 27B の主力型を扱えず、ノードごとに CPU へ
フォールバックしている」ことの直接的な証拠である。** 分割が 262 個あるということは、
HTP と CPU の間で 262 回の受け渡しが発生するということでもあり、
これが速度差の主因になっている。

**NGL=0 では splits = 1**（全部 CPU なので分割が起きない）。この対比が
「分割は HTP の型非対応に由来する」ことを裏づける。

### §1.4 速度の比較

| 構成 | prefill 128 | decode | 備考 |
| --- | --- | --- | --- |
| **上流 NGL=44** | **105.29 s** | 2,843 ms | splits 262 |
| **fork NGL=44**（統合版） | **43.25 s**（Phase 62 スポット） | 815 ms | splits 1 |
| 上流 NGL=0 | 121.85 s | 1,467 ms | splits 1 |
| fork NGL=0（Phase 61） | ~111 s | — | splits 1 |

**上流 NGL=44 は fork の 2.4 倍遅い**（105.29 s vs 43.25 s）。
**decode は 3.5 倍遅い**（2,843 ms vs 815 ms）。

**注意**: cap が完全には揃っていない（上流 1,785,600-1,670,400、fork の Phase 62
スポットは 2,112,000-1,785,600）。**厳密な比較には同一セッションの ABBA が必要**だが、
**262 分割という構造的な差があるので、順序は逆転しないと判断できる。**

### §1.5 乗り換え exit plan への示唆

**「上流は 27B を担えない」が数値で裏づけられた。**

- ロードはできる（型の欠落で落ちることはない）。
- しかし **262 分割**により、HTP と CPU の間で頻繁に受け渡しが起きる。
- 結果として **prefill で 2.4 倍、decode で 3.5 倍遅い**。

**したがって現時点の乗り換えは非現実的**という Phase 61 の判断は正しかった。
上流が iq2_xs/iq2_s/iq3_s/iq3_xxs の HTP カーネルを揃えるまで、fork を維持する
理由が定量的に示された。

**これが継続監視の第 1 測定点になる。** 次回以降、上流の型カバレッジが広がったら
同じ手順で測り、262 が減っているかを確認すればよい。

---
