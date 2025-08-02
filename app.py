"""
OWL-ViT Streamlit Application
単一責任原則に従い、各モジュールの機能を統合したメインアプリケーション
"""

import streamlit as st
import torch
from PIL import Image
from typing import Optional, List

# カスタムモジュールのインポート
from image_processor import ImageVisualizer
from model_manager import (
    ModelLoader, TextQueryProcessor, DetectionProcessor, 
    ImageGuidedDetectionProcessor
)
from ui_components import SidebarManager, InputManager, ResultsManager
from visualization import FlowVisualizationManager


class OWLViTApp:
    """OWL-ViTアプリケーションのメインクラス"""
    
    def __init__(self):
        """アプリケーションの初期化"""
        self.setup_page_config()
        self.model = None
        self.processor = None
    
    def setup_page_config(self) -> None:
        """ページ設定を行います"""
        st.set_page_config(
            page_title="OWL-ViT 物体検出",
            page_icon="🦉",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 日本語フォントの設定
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;500;700&display=swap');
        
        .main .block-container {
            font-family: 'Noto Sans JP', sans-serif;
        }
        
        .stButton > button {
            font-family: 'Noto Sans JP', sans-serif;
        }
        
        .stSelectbox > div > div > div {
            font-family: 'Noto Sans JP', sans-serif;
        }
        
        .stTextInput > div > div > div > input {
            font-family: 'Noto Sans JP', sans-serif;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def display_header(self) -> None:
        """ヘッダーを表示します"""
        st.title("🦉 OWL-ViT Object Detection")
        st.markdown("""
        このアプリケーションは、OWL-ViT（Vision Transformer for Open-World Localization）を使用して
        テキストクエリまたは画像クエリによる物体検出を行います。
        
        **特徴:**
        - 🎯 **シンプル検出**: 基本的な物体検出
        - 🔤 **テキストガイド検出**: テキストクエリで物体を検出
        - 🖼️ **画像ガイド検出**: 類似画像で物体を検出
        - 📷 **多様な画像入力**: アップロード、URL、カメラ撮影対応
        - 🔍 **テキスト検索**: 検索機能とクエリ提案
        - 🔧 **カスタマイズ可能な設定**
        - 📊 **詳細な検出結果の表示**
        """)
    
    def initialize_model(self, model_name: str) -> bool:
        """
        モデルを初期化します
        
        Args:
            model_name: 使用するモデル名
            
        Returns:
            初期化が成功した場合True
        """
        if self.model is None or self.processor is None:
            self.model, self.processor = ModelLoader.load_model_and_processor(model_name)
            
        if self.model is None or self.processor is None:
            st.error("モデルの初期化に失敗しました")
            return False
        
        return True
    
    def run_text_guided_detection(
        self,
        image: Image.Image,
        text_queries: List[str],
        confidence_threshold: float,
        translation_method: str = "辞書翻訳のみ"
    ) -> Optional[Image.Image]:
        """
        テキストガイド検出を実行します
        
        Args:
            image: 入力画像
            text_queries: テキストクエリのリスト
            confidence_threshold: 信頼度閾値
            
        Returns:
            可視化された結果画像、失敗時はNone
        """
        try:
            from translator import JapaneseTranslator
            
            # デバッグ情報の表示
            st.subheader("🔍 検出プロセス")
            
            # テキストクエリのフォーマット
            formatted_queries = TextQueryProcessor.format_text_queries(text_queries, translation_method)
            
            # 使用されるクエリを表示
            st.write("**使用されるクエリ:**")
            for i, query_list in enumerate(formatted_queries):
                st.write(f"クエリ {i+1}: {', '.join(query_list)}")
            
            # 入力の準備
            inputs = DetectionProcessor.prepare_inputs(
                self.processor, image, formatted_queries
            )
            if inputs is None:
                return None
            
            # 推論の実行
            outputs = DetectionProcessor.run_inference(self.model, inputs)
            if outputs is None:
                return None
            
            # 結果の後処理（より低い信頼度閾値で試行）
            target_sizes = torch.tensor([image.size[::-1]])
            
            # より低い信頼度閾値で試行
            thresholds_to_try = [
                confidence_threshold, 
                confidence_threshold * 0.7, 
                confidence_threshold * 0.5,
                confidence_threshold * 0.3,
                confidence_threshold * 0.1,  # 非常に低い閾値も試行
                0.05,  # さらに低い閾値
                0.01   # 最も低い閾値
            ]
            results = None
            successful_threshold = None
            
            st.write("**🎯 信頼度閾値の調整を試行中...**")
            
            for threshold in thresholds_to_try:
                st.write(f"- 閾値 {threshold:.3f} を試行中...")
                results = DetectionProcessor.post_process_results(
                    self.processor, outputs, target_sizes, threshold
                )
                if results and any(len(result["boxes"]) > 0 for result in results):
                    successful_threshold = threshold
                    st.success(f"✅ 信頼度閾値 {threshold:.3f} で検出に成功しました")
                    break
                else:
                    st.write(f"  ❌ 閾値 {threshold:.3f} では検出なし")
            
            if results is None or not any(len(result["boxes"]) > 0 for result in results):
                st.warning("⚠️ すべての閾値で検出結果がありませんでした。")
                st.write("**🔍 デバッグ情報:**")
                st.write(f"- 試行した閾値: {thresholds_to_try}")
                st.write(f"- 元の画像サイズ: {image.size}")
                st.write(f"- テキストクエリ数: {len(text_queries)}")
                st.write("**💡 改善のヒント:**")
                st.write("- 信頼度閾値をさらに下げてみてください")
                st.write("- より具体的なテキストクエリを試してください")
                st.write("- 画像の品質や物体の大きさを確認してください")
                return None
            
            # 結果の表示
            ResultsManager.display_detection_results(
                results, text_queries, confidence_threshold
            )
            
            # 可視化画像の作成
            if results and len(results) > 0:
                result = results[0]  # 最初の結果を使用
                boxes = result["boxes"].tolist()
                scores = result["scores"].tolist()
                
                # ラベルの取得（翻訳情報を含む）
                labels = []
                for label_idx in result["labels"].tolist():
                    if label_idx < len(text_queries):
                        original_query = text_queries[label_idx]
                        # 翻訳情報を追加
                        if JapaneseTranslator.is_japanese(original_query):
                            english_translation = JapaneseTranslator.translate_japanese_to_english(original_query)
                            if english_translation != original_query:
                                labels.append(f"{original_query}({english_translation})")
                            else:
                                labels.append(original_query)
                        else:
                            labels.append(original_query)
                    else:
                        labels.append("unknown")
                
                # 検出結果がある場合のみ可視化
                if len(boxes) > 0:
                    visualized_image = ImageVisualizer.create_detection_visualization(
                        image, boxes, scores, labels, confidence_threshold
                    )
                    return visualized_image
                else:
                    st.info("検出結果がありませんでした")
                    return None
            
            return None
            
        except Exception as e:
            st.error(f"テキストガイド検出の実行中にエラーが発生しました: {e}")
            if st.session_state.get('debug_mode', True):
                st.write("**🔍 エラーの詳細情報:**")
                st.write(f"- エラータイプ: {type(e).__name__}")
                st.write(f"- エラーメッセージ: {str(e)}")
                st.write(f"- テキストクエリ数: {len(text_queries)}")
                st.write(f"- 画像サイズ: {image.size}")
                st.write("**💡 解決策のヒント:**")
                st.write("- テキストクエリの数を減らしてみてください")
                st.write("- より簡単なクエリを試してください")
                st.write("- 画像のサイズを小さくしてみてください")
            return None
    
    def run_image_guided_detection(
        self,
        image: Image.Image,
        query_image: Image.Image,
        confidence_threshold: float,
        nms_threshold: float
    ) -> Optional[Image.Image]:
        """
        画像ガイド検出を実行します
        
        Args:
            image: ターゲット画像
            query_image: クエリ画像
            confidence_threshold: 信頼度閾値
            nms_threshold: NMS閾値
            
        Returns:
            可視化された結果画像、失敗時はNone
        """
        try:
            # 入力の準備
            inputs = ImageGuidedDetectionProcessor.prepare_image_guided_inputs(
                self.processor, image, query_image
            )
            if inputs is None:
                return None
            
            # 推論の実行
            outputs = ImageGuidedDetectionProcessor.run_image_guided_inference(
                self.model, inputs
            )
            if outputs is None:
                return None
            
            # 結果の後処理
            target_sizes = torch.tensor([image.size[::-1]])
            results = ImageGuidedDetectionProcessor.post_process_image_guided_results(
                self.processor, outputs, target_sizes, confidence_threshold, nms_threshold
            )
            if results is None:
                return None
            
            # 結果の表示
            ResultsManager.display_image_guided_results(results, confidence_threshold)
            
            # 可視化画像の作成
            if results and len(results) > 0:
                result = results[0]  # 最初の結果を使用
                boxes = result["boxes"].tolist()
                scores = result["scores"].tolist()
                labels = ["similar object"] * len(boxes)  # 画像ガイド検出ではラベルは固定
                
                visualized_image = ImageVisualizer.create_detection_visualization(
                    image, boxes, scores, labels, confidence_threshold
                )
                return visualized_image
            
            return None
            
        except Exception as e:
            st.error(f"画像ガイド検出の実行中にエラーが発生しました: {e}")
            return None
    
    def run(self) -> None:
        """アプリケーションのメイン実行ループ"""
        # ヘッダーの表示
        self.display_header()
        
        # サイドバーの設定
        selected_model = SidebarManager.create_model_selection()
        detection_mode = SidebarManager.create_detection_mode_selection()
        confidence_threshold, nms_threshold = SidebarManager.create_confidence_settings()
        visualization_type = SidebarManager.create_visualization_options()
        
        # 可視化の表示
        if st.session_state.get('show_flow_diagram', False):
            FlowVisualizationManager.display_flow_visualization()
            st.markdown("---")  # 区切り線
        
        # モデルの初期化
        if not self.initialize_model(selected_model):
            st.stop()
        
        # 入力セクション
        image = InputManager.create_image_input_section()
        if image is None:
            st.info("画像を入力してください")
            return
        
        # 検出モードに応じた処理
        if detection_mode == "シンプル検出":
            # シンプル検出モード
            st.subheader("🎯 シンプル検出")
            st.write("基本的な物体検出を実行します。")
            
            # 検出実行ボタン
            if st.button("🔍 シンプル検出を実行", type="primary"):
                with st.spinner("シンプル検出を実行中..."):
                    # 基本的なクエリを使用（数を制限）
                    basic_queries = ["person", "car", "dog"]
                    visualized_image = self.run_text_guided_detection(
                        image, basic_queries, confidence_threshold, "辞書翻訳のみ"
                    )
                    
                    if visualized_image:
                        st.image(visualized_image, caption="検出結果", use_container_width=True)
                        ResultsManager.create_download_section(visualized_image)
                    else:
                        st.warning("シンプル検出で物体が見つかりませんでした。")
        
        elif detection_mode == "テキストガイド検出":
            result = InputManager.create_text_query_section()
            if result is None:
                st.info("テキストクエリを入力してください")
                return
            
            text_queries, translation_method = result
            
            # 検出実行ボタン
            if st.button("🔍 物体検出を実行", type="primary"):
                with st.spinner("物体検出を実行中..."):
                    visualized_image = self.run_text_guided_detection(
                        image, text_queries, confidence_threshold, translation_method
                    )
                    
                    if visualized_image:
                        st.image(visualized_image, caption="検出結果", use_container_width=True)
                        ResultsManager.create_download_section(visualized_image)
        
        elif detection_mode == "画像ガイド検出":
            query_image = InputManager.create_query_image_section()
            if query_image is None:
                st.info("クエリ画像を入力してください")
                return
            
            # 検出実行ボタン
            if st.button("🔍 類似物体検出を実行", type="primary"):
                with st.spinner("類似物体検出を実行中..."):
                    visualized_image = self.run_image_guided_detection(
                        image, query_image, confidence_threshold, nms_threshold
                    )
                    
                    if visualized_image:
                        st.image(visualized_image, caption="検出結果", use_container_width=True)
                        ResultsManager.create_download_section(visualized_image)
        
        # 情報セクション
        self.display_info_section()
    
    def display_info_section(self) -> None:
        """情報セクションを表示します"""
        with st.expander("ℹ️ OWL-ViTについて"):
            st.markdown("""
            **OWL-ViT（Vision Transformer for Open-World Localization）**は、
            オープンボキャブラリ物体検出のためのVision Transformerモデルです。
            
            **主な特徴:**
            - 🎯 **ゼロショット検出**: 事前に学習した物体クラス以外も検出可能
            - 🔤 **テキストガイド検出**: テキストクエリで物体を検出
            - 🖼️ **画像ガイド検出**: 類似画像で物体を検出
            - 🧠 **CLIPベース**: マルチモーダルな理解能力
            
            **技術詳細:**
            - アーキテクチャ: Vision Transformer + CLIP
            - 入力: 画像 + テキスト/画像クエリ
            - 出力: バウンディングボックス + 信頼度スコア
            
            **参考資料:**
            - [OWL-ViT Paper](https://arxiv.org/abs/2205.06230)
            - [Hugging Face Documentation](https://huggingface.co/docs/transformers/model_doc/owlvit)
            """)


def main():
    """メイン関数"""
    app = OWLViTApp()
    app.run()


if __name__ == "__main__":
    main() 