#!/bin/bash

# OWL-ViT Streamlit Application Launcher (uv版)

echo "🦉 OWL-ViT Streamlit Application を起動中 (uv版)..."

# uvがインストールされているかチェック
if ! command -v uv &> /dev/null; then
    echo "❌ uvがインストールされていません。"
    echo "以下のコマンドでuvをインストールしてください："
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✅ uvが見つかりました"

# 仮想環境を作成し、依存関係をインストール
echo "📦 依存関係をインストール中..."
uv sync

# アプリケーションを起動
echo "🚀 Streamlitアプリケーションを起動中..."
echo "ブラウザで http://localhost:8501 にアクセスしてください"
echo ""
echo "アプリケーションを停止するには Ctrl+C を押してください"
echo ""

uv run streamlit run app.py 