# marine-plus

日本語テキストのアクセント (IP 境界・AP 境界・アクセント核位置) を推定するニューラルネットワークツールキット。
[marine](https://github.com/6gsn/marine) のフォークで、JSUT に加えて ROHAN4600 コーパスも使った学習や、
pyopenjtalk-plus 対応、Windows/Python 3.13 対応などを拡張している。

## コーディングルール

- Python 3.10 以上。Type Hints には `list[int]` / `dict[str, int]` / `int | None` を使う (`typing.List` 等は使わない)
- フォーマッタは `uv run task format` (Ruff)。**ダブルクォート** を使う (シングルクォートではない)
- リンタは `uv run task lint` (Ruff + Pyright)
- テストは `uv run task test` (pytest)

## パッケージマネージャ

**uv** を使用。`pip` は使わない。

```bash
uv sync              # 依存関係のインストール
uv run task <name>   # taskipy タスクの実行
```

## プロジェクト構造

```
marine/
  bin/           # CLI スクリプト (学習パイプラインの各 stage)
  models/        # Bi-LSTM + CRF + Attention モデル定義
  data/          # Dataset, FeatureSet, 特徴量テーブル
  utils/         # ユーティリティ (g2p, mora 分割, OpenJTalk 連携)
  predict.py     # 推論 API
recipe/
  20220912_release/              # 本家互換 (JSUT のみ, jsut2corpus 経由)
  20250122_marine-plus/          # JSUT のみ (corpus.yaml 経由)
  20260308_marine-plus_mora-based/  # JSUT + ROHAN, モーラベース 
  20260308_marine-plus_ap-based/    # JSUT + ROHAN, AP ベース
  common/        # 共通スクリプト, 固定 val/test ID
tests/           # pytest テスト
```

## 学習パイプライン (2026 レシピ)

```
corpus.yaml (中間 yaml)
    ↓ stage 1: make-raw-corpus
raw_corpus.json
    ↓ stage 2: prepare-features-pyopenjtalk
features.json
    ↓ stage 3: build-vocab
vocab.pkl
    ↓ stage 4: pack-corpus
feature_pack/ (ids.pkl, features.pkl, labels.pkl, wrong_mora_info.csv)
    ↓ stage 5: train-model
model checkpoint
```

### 実行方法

```bash
# 全 stage 実行
bash recipe/20260308_marine-plus_mora-based/run.sh --stage 1 --stop-stage 5

# 特定 stage のみ
bash recipe/20260308_marine-plus_mora-based/run.sh --stage 5 --stop-stage 5
```

### 主要な taskipy タスク

| タスク | 説明 |
|---|---|
| `vvproj2yaml` | .vvproj → corpus.yaml 変換 |
| `jsut2corpus` | jsut-label-fixed → raw_corpus.json 直接変換 (2022 レシピ用) |
| `make-raw-corpus` | corpus.yaml → raw_corpus.json 変換 (2025/2026 レシピ用) |
| `validate-yaml-corpus` | corpus.yaml のバリデーション |
| `pack-corpus` | features + vocab → 学習用 pkl 生成 (wrong_mora_info.csv もここで生成) |
| `train-model` | Hydra 設定でモデル学習 |

## データの流れ: 2 つの変換パス

### パス 1: 2022 レシピ (JSUT のみ)
```
/.../jsut-label-fixed/ (text_kana/ + e2e_symbol/) → jsut2corpus → raw_corpus.json
```

### パス 2: 2025/2026 レシピ (JSUT + ROHAN)
```
/.../jsut-label-fixed/ ──────────────────────────────┐
                                                     ├→ data/corpus.yaml → make-raw-corpus → raw_corpus.json
/../Zundamon_ROHAN_label/vvproj_data/ → vvproj2yaml ─┘
```

corpus.yaml は `{text, annotation}` の 2 フィールドのみの中間フォーマット。
JSUT と ROHAN の統合は corpus.yaml レベルで行う (Python で yaml を merge)。

## corpus.yaml のフォーマット

```yaml
BASIC5000_0001:
  text: 水をマレーシアから買わなくてはならないのです。
  annotation: ^ミ[ズヲ#マ[レ]ーシアカラ#カ[ワナ]クテワ#ナ[ラ]ナイノデス$
ROHAN4600_0001:
  text: 流し斬りが完全に入れば,デバフの効果が付与される.
  annotation: ^ナ[ガシギリガ#カ[ンゼンニ#ハ]イレバ_デ[バフノ#コ]オカガ#フ[ヨサ]レル$
```

annotation は JSUT 互換形式: `^` 開始, `$` 終了, `]` アクセント核, `#` AP 境界, `_` IP 境界。

## pack_corpus.py の重要な概念

このファイルはパイプラインの中核で、dataset 側 annotation と OpenJTalk 形態素解析のアライメントを取る。

### アライメントの仕組み

1. **完全一致**: extracted == expected → そのまま採用
2. **ソフトマッチ** (`is_softmatch_mora_sequence`): 長音揺れ・四つ仮名等のモーラ数不変の差異を吸収
3. **Surface アライメント** (`build_surface_aligned_nodes`): surface から expected pronunciation を説明できる場合、pronunciation を expected 側へ書き換え
4. **編集距離アライメント** (`_remap_labels_by_mora_alignment`): DP でラベルを写像
5. **ドロップ**: 上記すべて失敗 → wrong_mora_info.csv に記録

## 注意事項

- **wrong_mora_info.csv**: pack_corpus の出力。ドロップされたエントリの expected / extracted の読みが記録される。件数はデータ品質の指標
- **stage 1-4 は再実行可能**: データを変更したら stage 1 からやり直す。stage 5 のみの再実行で学習だけやり直すことも可能
- **val/test ID の固定**: `recipe/common/database/` に固定 ID がある。データ変更時に ID が corpus に含まれない場合は自動再生成される (WARNING ログ)
- **Hydra 設定**: 学習のハイパーパラメータは `recipe/*/conf/train/` 以下の yaml で管理
