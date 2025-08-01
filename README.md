# OWL-ViT Streamlit Application

このアプリケーションは、OWL-ViT（Vision Transformer for Open-World Localization）を使用して物体検出を行うStreamlitアプリケーションです。単一責任原則に従って設計されており、再利用性とテストしやすさを重視しています。

## 機能

- **🎯 テキストガイド検出**: テキストクエリで物体を検出
- **🖼️ 画像ガイド検出**: 類似画像で物体を検出
- **🔧 カスタマイズ可能な設定**: 信頼度閾値、NMS閾値の調整
- **📊 詳細な検出結果表示**: バウンディングボックス、信頼度スコア
- **💾 結果のダウンロード**: 検出結果画像をPNG形式でダウンロード
- **🔄 複数の入力方法**: サンプル画像、ファイルアップロード、URL

## アーキテクチャ

単一責任原則に従い、以下のモジュールに分離されています：

### モジュール構成

```
owl_vit_app/
├── app.py                 # メインアプリケーション
├── image_processor.py     # 画像処理モジュール
├── model_manager.py       # モデル管理モジュール
├── ui_components.py       # UIコンポーネントモジュール
├── pyproject.toml         # プロジェクト設定
└── README.md             # このファイル
```

### 各モジュールの責任

#### `image_processor.py`
- **ImageLoader**: 画像の読み込み（URL、ファイルアップロード）
- **ImageValidator**: 画像の検証と情報取得
- **ImageVisualizer**: 画像の表示と可視化
- **ImagePreprocessor**: 画像の前処理（リサイズ、モード変換）

#### `model_manager.py`
- **ModelLoader**: OWL-ViTモデルとプロセッサーの読み込み
- **TextQueryProcessor**: テキストクエリの処理と検証
- **DetectionProcessor**: テキストガイド検出の処理
- **ImageGuidedDetectionProcessor**: 画像ガイド検出の処理

#### `ui_components.py`
- **SidebarManager**: サイドバーの設定UI
- **InputManager**: 入力セクションのUI
- **ResultsManager**: 結果表示のUI

#### `app.py`
- **OWLViTApp**: メインアプリケーションクラス
- 各モジュールの統合とアプリケーションの制御

## インストール

### uvを使用する場合（推奨）

1. uvをインストール（まだインストールしていない場合）:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. 依存関係をインストール:
```bash
uv sync
```

3. アプリケーションを起動:
```bash
uv run streamlit run app.py
```

### pipを使用する場合

1. 依存関係をインストール:
```bash
pip install -e .
```

2. アプリケーションを起動:
```bash
streamlit run app.py
```

## 使用方法

### 1. アプリケーションの起動

上記のインストール手順に従ってアプリケーションを起動し、ブラウザで `http://localhost:8501` にアクセスします。

### 2. 設定の調整

サイドバーで以下の設定を調整できます：

- **モデル選択**: 使用するOWL-ViTモデルを選択
- **検出モード**: テキストガイド検出または画像ガイド検出
- **信頼度閾値**: 検出結果の信頼度閾値（0.0-1.0）
- **NMS閾値**: Non-Maximum Suppressionの閾値（0.0-1.0）

### 3. 画像の入力

以下の方法で画像を入力できます：

- **サンプル画像を使用**: デフォルトのサンプル画像を使用
- **画像をアップロード**: ローカルファイルをアップロード
- **URLから画像を取得**: 画像URLを指定

### 4. 検出の実行

#### テキストガイド検出の場合

1. テキストクエリの入力方法を選択：
   - **単一クエリ**: 1つの物体を検出
   - **複数クエリ**: 複数の物体を同時に検出
   - **プリセットクエリ**: 事前定義されたクエリセットを使用

2. テキストクエリを入力（例: "cat", "dog", "car"）

3. 「物体検出を実行」ボタンをクリック

#### 画像ガイド検出の場合

1. クエリ画像を入力（検出したい物体の類似画像）

2. 「類似物体検出を実行」ボタンをクリック

### 5. 結果の確認

- 検出結果が画像上にバウンディングボックスで表示されます
- 各検出結果の信頼度スコアが表示されます
- 検出結果画像をダウンロードできます

## 技術詳細

### OWL-ViTについて

OWL-ViTは、オープンボキャブラリ物体検出のためのVision Transformerモデルです。

**主な特徴:**
- **ゼロショット検出**: 事前に学習した物体クラス以外も検出可能
- **マルチモーダル理解**: テキストと画像の両方を理解
- **CLIPベース**: 強力なマルチモーダル表現学習

**技術仕様:**
- アーキテクチャ: Vision Transformer + CLIP
- 入力: 画像 + テキスト/画像クエリ
- 出力: バウンディングボックス + 信頼度スコア

### 利用可能なモデル

- `google/owlvit-base-patch32`: 基本モデル（推奨）
- `google/owlvit-base-patch16`: 高解像度対応モデル
- `google/owlvit-large-patch14`: 大規模モデル（高精度）

## 開発

### テスト

```bash
# テストの実行
uv run pytest

# カバレッジ付きテスト
uv run pytest --cov=.
```

### コードフォーマット

```bash
# コードのフォーマット
uv run black .

# リント
uv run flake8 .

# 型チェック
uv run mypy .
```

### アーキテクチャの利点

1. **単一責任原則**: 各クラスが明確な責任を持つ
2. **再利用性**: モジュール間の依存関係が最小限
3. **テストしやすさ**: 各機能が独立してテスト可能
4. **保守性**: 機能の追加・変更が容易
5. **拡張性**: 新しい機能の追加が簡単

## トラブルシューティング

### モデルの読み込みに失敗する場合

- インターネット接続を確認してください
- 十分なディスク容量があることを確認してください（モデルサイズ: ~1GB）
- GPUメモリが不足している場合は、CPUモードで実行されます

### 検出結果が得られない場合

- 信頼度閾値を下げてみてください
- テキストクエリをより具体的にしてください
- 画像の品質を確認してください

### メモリ不足エラーが発生する場合

- 画像サイズを小さくしてください
- より軽量なモデルを使用してください
- バッチサイズを小さくしてください

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 参考資料

- [OWL-ViT Paper](https://arxiv.org/abs/2205.06230)
- [Hugging Face Documentation](https://huggingface.co/docs/transformers/model_doc/owlvit)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OWL-ViT GitHub Repository](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit) 