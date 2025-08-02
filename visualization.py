"""
OWL-ViTの処理フロー可視化モジュール
画像→ViTトークン→スコアマップ→検出の流れを視覚的に表現
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from typing import List, Tuple, Optional, Dict, Any
import seaborn as sns


class OWLViTVisualizer:
    """OWL-ViTの処理フローを可視化するクラス"""
    
    def __init__(self):
        """可視化クラスの初期化"""
        # 日本語フォントの設定
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
    
    @staticmethod
    def create_processing_flow_diagram() -> plt.Figure:
        """
        OWL-ViTの処理フロー図を作成します
        
        Returns:
            処理フロー図のmatplotlib Figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis('off')
        
        # 背景色
        ax.set_facecolor('#f8f9fa')
        
        # ステップ1: 画像入力
        self._draw_image_input_step(ax, 1, 6)
        
        # ステップ2: ViTトークン化
        self._draw_vit_tokenization_step(ax, 3, 6)
        
        # ステップ3: スコアマップ生成
        self._draw_score_map_step(ax, 5, 6)
        
        # ステップ4: 検出結果
        self._draw_detection_step(ax, 7, 6)
        
        # 矢印で接続
        self._draw_arrows(ax)
        
        # タイトル
        ax.text(5, 7.5, '🦉 OWL-ViT 処理フロー', 
                fontsize=20, fontweight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        return fig
    
    @staticmethod
    def _draw_image_input_step(ax, x: float, y: float):
        """画像入力ステップを描画"""
        # 画像の枠
        rect = patches.Rectangle((x-0.8, y-0.6), 1.6, 1.2, 
                               linewidth=2, edgecolor='#007bff', facecolor='#e3f2fd')
        ax.add_patch(rect)
        
        # 画像アイコン
        ax.text(x, y+0.1, '📷', fontsize=24, ha='center')
        
        # ラベル
        ax.text(x, y-0.8, '画像入力', fontsize=12, fontweight='bold', ha='center')
        ax.text(x, y-1.0, 'RGB画像', fontsize=10, ha='center', color='#666')
    
    @staticmethod
    def _draw_vit_tokenization_step(ax, x: float, y: float):
        """ViTトークン化ステップを描画"""
        # トークンの枠
        rect = patches.Rectangle((x-0.8, y-0.6), 1.6, 1.2, 
                               linewidth=2, edgecolor='#28a745', facecolor='#e8f5e8')
        ax.add_patch(rect)
        
        # トークンアイコン
        ax.text(x, y+0.1, '🔲', fontsize=24, ha='center')
        
        # ラベル
        ax.text(x, y-0.8, 'ViTトークン化', fontsize=12, fontweight='bold', ha='center')
        ax.text(x, y-1.0, 'パッチ分割', fontsize=10, ha='center', color='#666')
    
    @staticmethod
    def _draw_score_map_step(ax, x: float, y: float):
        """スコアマップ生成ステップを描画"""
        # スコアマップの枠
        rect = patches.Rectangle((x-0.8, y-0.6), 1.6, 1.2, 
                               linewidth=2, edgecolor='#ffc107', facecolor='#fff8e1')
        ax.add_patch(rect)
        
        # スコアマップアイコン
        ax.text(x, y+0.1, '🎯', fontsize=24, ha='center')
        
        # ラベル
        ax.text(x, y-0.8, 'スコアマップ', fontsize=12, fontweight='bold', ha='center')
        ax.text(x, y-1.0, '信頼度スコア', fontsize=10, ha='center', color='#666')
    
    @staticmethod
    def _draw_detection_step(ax, x: float, y: float):
        """検出結果ステップを描画"""
        # 検出結果の枠
        rect = patches.Rectangle((x-0.8, y-0.6), 1.6, 1.2, 
                               linewidth=2, edgecolor='#dc3545', facecolor='#fce4ec')
        ax.add_patch(rect)
        
        # 検出アイコン
        ax.text(x, y+0.1, '📍', fontsize=24, ha='center')
        
        # ラベル
        ax.text(x, y-0.8, '検出結果', fontsize=12, fontweight='bold', ha='center')
        ax.text(x, y-1.0, 'バウンディングボックス', fontsize=10, ha='center', color='#666')
    
    @staticmethod
    def _draw_arrows(ax):
        """矢印を描画"""
        # ステップ間の矢印
        arrow_props = dict(arrowstyle='->', lw=2, color='#333')
        
        # 画像→ViT
        ax.annotate('', xy=(2.2, 6), xytext=(1.8, 6), arrowprops=arrow_props)
        ax.text(2.0, 6.3, 'Vision\nTransformer', fontsize=9, ha='center', 
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # ViT→スコアマップ
        ax.annotate('', xy=(4.2, 6), xytext=(3.8, 6), arrowprops=arrow_props)
        ax.text(4.0, 6.3, 'CLIP\nマッチング', fontsize=9, ha='center',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # スコアマップ→検出
        ax.annotate('', xy=(6.2, 6), xytext=(5.8, 6), arrowprops=arrow_props)
        ax.text(6.0, 6.3, '後処理\nNMS', fontsize=9, ha='center',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    @staticmethod
    def create_detailed_flow_diagram() -> plt.Figure:
        """
        詳細な処理フロー図を作成します
        
        Returns:
            詳細な処理フロー図のmatplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🦉 OWL-ViT 詳細処理フロー', fontsize=20, fontweight='bold')
        
        # サブプロット1: 画像前処理
        ax1 = axes[0, 0]
        OWLViTVisualizer._draw_image_preprocessing(ax1)
        
        # サブプロット2: ViT処理
        ax2 = axes[0, 1]
        OWLViTVisualizer._draw_vit_processing(ax2)
        
        # サブプロット3: CLIPマッチング
        ax3 = axes[1, 0]
        OWLViTVisualizer._draw_clip_matching(ax3)
        
        # サブプロット4: 検出後処理
        ax4 = axes[1, 1]
        OWLViTVisualizer._draw_detection_postprocessing(ax4)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def _draw_image_preprocessing(ax):
        """画像前処理の詳細図を描画"""
        ax.set_title('📷 画像前処理', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis('off')
        
        # 元画像
        ax.text(2, 6, '元画像', fontsize=12, ha='center')
        rect1 = patches.Rectangle((1, 4.5), 2, 1.5, linewidth=2, edgecolor='blue', facecolor='lightblue')
        ax.add_patch(rect1)
        
        # リサイズ
        ax.annotate('', xy=(4.5, 5.25), xytext=(3.5, 5.25), arrowprops=dict(arrowstyle='->', lw=2))
        ax.text(4, 5.5, 'リサイズ', fontsize=10, ha='center')
        
        # 正規化画像
        ax.text(7, 6, '正規化画像', fontsize=12, ha='center')
        rect2 = patches.Rectangle((6, 4.5), 2, 1.5, linewidth=2, edgecolor='green', facecolor='lightgreen')
        ax.add_patch(rect2)
        
        # パッチ分割
        ax.text(4.5, 3, 'パッチ分割', fontsize=12, ha='center')
        for i in range(4):
            for j in range(4):
                rect = patches.Rectangle((3+i*0.5, 1.5+j*0.3), 0.4, 0.25, 
                                       linewidth=1, edgecolor='red', facecolor='lightcoral')
                ax.add_patch(rect)
    
    @staticmethod
    def _draw_vit_processing(ax):
        """ViT処理の詳細図を描画"""
        ax.set_title('🔲 Vision Transformer', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis('off')
        
        # パッチ埋め込み
        ax.text(2, 6, 'パッチ埋め込み', fontsize=12, ha='center')
        for i in range(6):
            rect = patches.Rectangle((1, 4-i*0.5), 2, 0.4, linewidth=1, edgecolor='purple', facecolor='lavender')
            ax.add_patch(rect)
        
        # 位置エンコーディング
        ax.annotate('', xy=(4.5, 2), xytext=(3.5, 2), arrowprops=dict(arrowstyle='->', lw=2))
        ax.text(4, 2.3, '+位置エンコーディング', fontsize=10, ha='center')
        
        # Transformer層
        ax.text(7, 6, 'Transformer層', fontsize=12, ha='center')
        for i in range(4):
            rect = patches.Rectangle((6, 4.5-i*0.8), 2, 0.6, linewidth=1, edgecolor='orange', facecolor='moccasin')
            ax.add_patch(rect)
            ax.text(7, 4.2-i*0.8, f'Layer {i+1}', fontsize=8, ha='center')
    
    @staticmethod
    def _draw_clip_matching(ax):
        """CLIPマッチングの詳細図を描画"""
        ax.set_title('🎯 CLIP マッチング', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis('off')
        
        # テキストエンコーダー
        ax.text(2, 6, 'テキスト\nエンコーダー', fontsize=12, ha='center')
        rect1 = patches.Rectangle((1, 4), 2, 1.5, linewidth=2, edgecolor='blue', facecolor='lightblue')
        ax.add_patch(rect1)
        ax.text(2, 4.5, '"a photo of a cat"', fontsize=8, ha='center')
        
        # 画像エンコーダー
        ax.text(7, 6, '画像\nエンコーダー', fontsize=12, ha='center')
        rect2 = patches.Rectangle((6, 4), 2, 1.5, linewidth=2, edgecolor='green', facecolor='lightgreen')
        ax.add_patch(rect2)
        ax.text(7, 4.5, 'ViT特徴量', fontsize=8, ha='center')
        
        # 類似度計算
        ax.text(4.5, 2, '類似度計算\n(コサイン類似度)', fontsize=12, ha='center')
        ax.annotate('', xy=(4.5, 3), xytext=(3, 4.75), arrowprops=dict(arrowstyle='->', lw=2))
        ax.annotate('', xy=(4.5, 3), xytext=(7, 4.75), arrowprops=dict(arrowstyle='->', lw=2))
        
        # スコアマップ
        ax.text(4.5, 0.5, 'スコアマップ', fontsize=12, ha='center')
        rect3 = patches.Rectangle((3.5, -0.5), 2, 1, linewidth=2, edgecolor='red', facecolor='lightcoral')
        ax.add_patch(rect3)
    
    @staticmethod
    def _draw_detection_postprocessing(ax):
        """検出後処理の詳細図を描画"""
        ax.set_title('📍 検出後処理', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis('off')
        
        # スコアマップ
        ax.text(2, 6, 'スコアマップ', fontsize=12, ha='center')
        rect1 = patches.Rectangle((1, 4), 2, 1.5, linewidth=2, edgecolor='red', facecolor='lightcoral')
        ax.add_patch(rect1)
        
        # 閾値処理
        ax.annotate('', xy=(4.5, 4.75), xytext=(3.5, 4.75), arrowprops=dict(arrowstyle='->', lw=2))
        ax.text(4, 5, '閾値処理', fontsize=10, ha='center')
        
        # 候補領域
        ax.text(7, 6, '候補領域', fontsize=12, ha='center')
        rect2 = patches.Rectangle((6, 4), 2, 1.5, linewidth=2, edgecolor='orange', facecolor='moccasin')
        ax.add_patch(rect2)
        
        # NMS
        ax.annotate('', xy=(4.5, 2), xytext=(7, 4.75), arrowprops=dict(arrowstyle='->', lw=2))
        ax.text(4.5, 2.3, 'NMS\n(Non-Maximum Suppression)', fontsize=10, ha='center')
        
        # 最終結果
        ax.text(4.5, 0.5, '検出結果', fontsize=12, ha='center')
        rect3 = patches.Rectangle((3.5, -0.5), 2, 1, linewidth=2, edgecolor='green', facecolor='lightgreen')
        ax.add_patch(rect3)
        ax.text(4.5, 0, 'バウンディングボックス', fontsize=8, ha='center')
    
    @staticmethod
    def create_interactive_flow_diagram() -> plt.Figure:
        """
        インタラクティブな処理フロー図を作成します
        
        Returns:
            インタラクティブな処理フロー図のmatplotlib Figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # 背景
        ax.set_facecolor('#f8f9fa')
        
        # メインタイトル
        ax.text(6, 9.5, '🦉 OWL-ViT インタラクティブ処理フロー', 
                fontsize=18, fontweight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
        
        # 各ステップの詳細説明付き図
        steps = [
            {'pos': (2, 7), 'title': '📷 画像入力', 'desc': 'RGB画像\nサイズ調整\n正規化'},
            {'pos': (4, 7), 'title': '🔲 ViT処理', 'desc': 'パッチ分割\n位置エンコーディング\nTransformer層'},
            {'pos': (6, 7), 'title': '🎯 CLIPマッチング', 'desc': 'テキスト特徴量\n画像特徴量\n類似度計算'},
            {'pos': (8, 7), 'title': '📍 検出結果', 'desc': 'スコアマップ\n閾値処理\nNMS処理'}
        ]
        
        colors = ['#007bff', '#28a745', '#ffc107', '#dc3545']
        
        for i, step in enumerate(steps):
            x, y = step['pos']
            
            # ステップボックス
            rect = patches.Rectangle((x-0.8, y-0.8), 1.6, 1.6, 
                                   linewidth=3, edgecolor=colors[i], facecolor='white', alpha=0.8)
            ax.add_patch(rect)
            
            # タイトル
            ax.text(x, y+0.3, step['title'], fontsize=12, fontweight='bold', ha='center')
            
            # 説明
            ax.text(x, y-0.2, step['desc'], fontsize=9, ha='center', va='top')
        
        # 矢印と詳細説明
        arrows = [
            {'start': (2.8, 7), 'end': (3.2, 7), 'desc': 'Vision\nTransformer'},
            {'start': (4.8, 7), 'end': (5.2, 7), 'desc': 'CLIP\nマッチング'},
            {'start': (6.8, 7), 'end': (7.2, 7), 'desc': '後処理\nNMS'}
        ]
        
        for arrow in arrows:
            ax.annotate('', xy=arrow['end'], xytext=arrow['start'], 
                       arrowprops=dict(arrowstyle='->', lw=3, color='#333'))
            ax.text((arrow['start'][0] + arrow['end'][0])/2, arrow['start'][1] + 0.5, 
                   arrow['desc'], fontsize=9, ha='center',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # 技術詳細
        details = [
            '• パッチサイズ: 32x32 (Base-32)',
            '• 埋め込み次元: 768',
            '• Transformer層数: 12',
            '• アテンションヘッド数: 12',
            '• 最大シーケンス長: 512'
        ]
        
        for i, detail in enumerate(details):
            ax.text(1, 4-i*0.5, detail, fontsize=10, ha='left',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='#e9ecef', alpha=0.8))
        
        # 処理時間目安
        timing_info = [
            '⏱️ 処理時間目安:',
            '• 画像読み込み: ~100ms',
            '• ViT処理: ~500ms',
            '• CLIPマッチング: ~200ms',
            '• 後処理: ~50ms',
            '• 合計: ~850ms'
        ]
        
        for i, info in enumerate(timing_info):
            ax.text(8, 4-i*0.4, info, fontsize=10, ha='left',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='#e9ecef', alpha=0.8))
        
        return fig


class FlowVisualizationManager:
    """処理フローの可視化を管理するクラス"""
    
    @staticmethod
    def display_flow_visualization() -> None:
        """処理フローの可視化を表示します"""
        st.header("🦉 OWL-ViT 処理フロー可視化")
        
        # 可視化タイプの選択
        viz_type = st.selectbox(
            "可視化タイプを選択",
            ["基本フロー図", "詳細フロー図", "インタラクティブフロー図"],
            help="OWL-ViTの処理フローを異なる詳細レベルで表示"
        )
        
        if viz_type == "基本フロー図":
            fig = OWLViTVisualizer.create_processing_flow_diagram()
            st.pyplot(fig)
            
            st.markdown("""
            **基本フロー図の説明:**
            - 📷 **画像入力**: RGB画像を読み込み、前処理を実行
            - 🔲 **ViTトークン化**: 画像をパッチに分割し、Vision Transformerで処理
            - 🎯 **スコアマップ**: CLIPマッチングにより信頼度スコアを生成
            - 📍 **検出結果**: 後処理によりバウンディングボックスを出力
            """)
            
        elif viz_type == "詳細フロー図":
            fig = OWLViTVisualizer.create_detailed_flow_diagram()
            st.pyplot(fig)
            
            st.markdown("""
            **詳細フロー図の説明:**
            
            **📷 画像前処理:**
            - 画像のリサイズと正規化
            - パッチ分割（32x32ピクセル）
            
            **🔲 Vision Transformer:**
            - パッチ埋め込み
            - 位置エンコーディング
            - 12層のTransformer処理
            
            **🎯 CLIPマッチング:**
            - テキストエンコーダーによるクエリ処理
            - 画像エンコーダーによる特徴抽出
            - コサイン類似度によるマッチング
            
            **📍 検出後処理:**
            - スコアマップの生成
            - 閾値処理
            - Non-Maximum Suppression (NMS)
            """)
            
        elif viz_type == "インタラクティブフロー図":
            fig = OWLViTVisualizer.create_interactive_flow_diagram()
            st.pyplot(fig)
            
            st.markdown("""
            **インタラクティブフロー図の説明:**
            
            この図は、OWL-ViTの処理フローを技術的な詳細と共に表示しています。
            各ステップの処理内容と技術仕様、処理時間の目安を確認できます。
            
            **技術仕様:**
            - パッチサイズ: 32x32ピクセル
            - 埋め込み次元: 768
            - Transformer層数: 12
            - アテンションヘッド数: 12
            - 最大シーケンス長: 512
            """)
        
        # 技術的な詳細情報
        with st.expander("🔧 技術的な詳細"):
            st.markdown("""
            **OWL-ViTのアーキテクチャ詳細:**
            
            **1. Vision Transformer (ViT)**
            - 画像を固定サイズのパッチに分割
            - 各パッチを線形埋め込みでベクトル化
            - 位置エンコーディングを追加
            - Transformerエンコーダーで処理
            
            **2. CLIP (Contrastive Language-Image Pre-training)**
            - テキストと画像の対比学習
            - 共通の埋め込み空間で表現
            - ゼロショット転移学習に対応
            
            **3. 物体検出ヘッド**
            - スコアマップの生成
            - バウンディングボックスの回帰
            - Non-Maximum Suppression
            
            **4. 後処理**
            - 信頼度閾値によるフィルタリング
            - NMSによる重複除去
            - 座標の正規化
            """)
        
        # パフォーマンス情報
        with st.expander("⚡ パフォーマンス情報"):
            st.markdown("""
            **処理時間の目安 (GPU使用時):**
            
            | 処理ステップ | 時間 |
            |-------------|------|
            | 画像読み込み | ~100ms |
            | ViT処理 | ~500ms |
            | CLIPマッチング | ~200ms |
            | 後処理 | ~50ms |
            | **合計** | **~850ms** |
            
            **メモリ使用量:**
            - モデルサイズ: ~1GB
            - 推論時のメモリ: ~2-4GB
            - 画像サイズによる変動: あり
            
            **精度:**
            - COCOデータセットでのmAP: ~35%
            - ゼロショット性能: 良好
            - 日本語クエリ対応: 翻訳機能付き
            """) 