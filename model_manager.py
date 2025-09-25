"""
OWL-ViTモデルの管理を担当するモジュール
単一責任原則に従い、モデルの読み込み、推論、後処理機能を提供
"""

import torch
from typing import Optional, List, Dict, Any, Tuple
from transformers import AutoProcessor, OwlViTForObjectDetection
import streamlit as st


class ModelLoader:
    """モデルの読み込みを担当するクラス"""
    
    DEFAULT_MODEL_NAME = "google/owlvit-base-patch32"
    
    @staticmethod
    @st.cache_resource
    def load_model_and_processor(model_name: str = DEFAULT_MODEL_NAME) -> Tuple[Optional[OwlViTForObjectDetection], Optional[Any]]:
        """
        モデルとプロセッサーを読み込みます
        
        Args:
            model_name: モデル名
            
        Returns:
            (モデル, プロセッサー)のタプル、失敗時は(None, None)
        """
        try:
            with st.spinner(f"モデル '{model_name}' を読み込み中..."):
                processor = AutoProcessor.from_pretrained(model_name)
                model = OwlViTForObjectDetection.from_pretrained(model_name)
                return model, processor
        except Exception as e:
            st.error(f"モデルの読み込みに失敗しました: {e}")
            return None, None
    
    @staticmethod
    def get_available_models() -> List[str]:
        """
        利用可能なモデルのリストを取得します
        
        Returns:
            利用可能なモデル名のリスト
        """
        return [
            "google/owlvit-base-patch32",
            "google/owlvit-base-patch16",
            "google/owlvit-large-patch14"
        ]


class TextQueryProcessor:
    """テキストクエリの処理を担当するクラス"""
    
    @staticmethod
    def validate_text_queries(text_queries: List[str]) -> bool:
        """
        テキストクエリが適切かどうかを検証します
        
        Args:
            text_queries: 検証するテキストクエリのリスト
            
        Returns:
            テキストクエリが適切な場合True
        """
        if not text_queries:
            st.warning("テキストクエリを入力してください")
            return False
        
        for query in text_queries:
            if not query.strip():
                st.warning("空のテキストクエリは使用できません")
                return False
            
            if len(query.strip()) < 2:
                st.warning("テキストクエリは2文字以上である必要があります")
                return False
        
        return True
    
    @staticmethod
    def format_text_queries(text_queries: List[str], translation_method: str = "辞書翻訳のみ") -> List[List[str]]:
        """
        テキストクエリをOWL-ViTの入力形式に変換します
        
        Args:
            text_queries: 元のテキストクエリのリスト
            translation_method: 翻訳方法
            
        Returns:
            フォーマットされたテキストクエリのリスト
        """
        from translator import JapaneseTranslator
        
        use_api = translation_method == "辞書翻訳 + API翻訳"
        
        # 各元クエリに対して、基本的なクエリ形式のみを生成
        formatted_queries = []
        
        for query in text_queries:
            query = query.strip()
            
            # クエリが長すぎる場合は短縮
            if len(query) > 50:
                query = query[:50]
                st.warning(f"クエリが長すぎるため、最初の50文字に短縮しました: {query}")
            
            # 基本的なクエリ形式のみを生成
            if JapaneseTranslator.is_japanese(query):
                # 日本語の場合、英語に翻訳
                english_translation = JapaneseTranslator.translate_japanese_to_english(query, use_api=use_api)
                if english_translation != query:
                    formatted_queries.append(f"a photo of a {english_translation}")
                else:
                    formatted_queries.append(f"a photo of a {query}")
            else:
                # 英語の場合、基本的な形式
                formatted_queries.append(f"a photo of a {query}")
        
        # クエリの数を制限（エラー回避のため）
        if len(formatted_queries) > 5:
            formatted_queries = formatted_queries[:5]
            st.warning(f"クエリの数が多すぎるため、最初の5個に制限しました")
            
        # デバッグ情報を表示
        if st.session_state.get('debug_mode', True):
            st.write(f"**📝 生成されたクエリバリエーション:**")
            st.write(f"- 元のクエリ数: {len(text_queries)}")
            st.write(f"- 生成されたクエリ数: {len(formatted_queries)}")
            for i, query in enumerate(formatted_queries):
                st.write(f"  {i+1}. '{query}'")
        
        # OWL-ViTの期待する形式: 各クエリを個別のリストとして返す
        return [[query] for query in formatted_queries]


class DetectionProcessor:
    """物体検出の処理を担当するクラス"""
    
    @staticmethod
    def prepare_inputs(
        processor: Any,
        image: Any,
        text_queries: List[List[str]]
    ) -> Optional[Dict[str, Any]]:
        """
        モデルへの入力を準備します
        
        Args:
            processor: OWL-ViTプロセッサー
            image: 入力画像
            text_queries: テキストクエリのリスト
            
        Returns:
            準備された入力、失敗時はNone
        """
        try:
            # デバッグ情報の表示（条件付き）
            if st.session_state.get('debug_mode', True):
                st.write("**🔍 入力準備のデバッグ情報:**")
                st.write(f"- 元のテキストクエリ数: {len(text_queries)}")
            
            # テキストクエリの検証
            if not text_queries or not isinstance(text_queries, list):
                st.error("テキストクエリが無効です")
                return None
            
            # 各クエリリストから最初のクエリを取得
            normalized_queries = []
            for i, query_list in enumerate(text_queries):
                if isinstance(query_list, list) and len(query_list) > 0:
                    query = query_list[0]
                    if len(query) > 100:  # 最大100文字に制限
                        query = query[:100]
                    normalized_queries.append(query)
                    if st.session_state.get('debug_mode', True):
                        st.write(f"- クエリ {i+1}: '{query}'")
                else:
                    st.error("テキストクエリの形式が不正です")
                    return None
            
            if st.session_state.get('debug_mode', True):
                st.write(f"- 正規化されたクエリ数: {len(normalized_queries)}")
            
            # OWL-ViTの正しい入力形式で処理
            # 各クエリに対して同じ画像を使用
            inputs = processor(
                text=normalized_queries,  # 文字列のリスト
                images=image,  # 単一の画像
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            
            if st.session_state.get('debug_mode', True):
                st.write("✅ 入力の準備が完了しました")
                if 'input_ids' in inputs:
                    st.write(f"- 入力テンソルの形状: {inputs['input_ids'].shape}")
            
            return inputs
        except Exception as e:
            st.error(f"入力の準備に失敗しました: {e}")
            # デバッグ情報を表示
            st.write(f"テキストクエリの数: {len(text_queries)}")
            for i, query_list in enumerate(text_queries):
                st.write(f"クエリ {i+1} の長さ: {len(query_list)}")
                st.write(f"クエリ {i+1} の内容: {query_list}")
            return None
    
    @staticmethod
    def run_inference(
        model: OwlViTForObjectDetection,
        inputs: Dict[str, Any]
    ) -> Optional[Any]:
        """
        推論を実行します
        
        Args:
            model: OWL-ViTモデル
            inputs: モデルへの入力
            
        Returns:
            推論結果、失敗時はNone
        """
        try:
            if st.session_state.get('debug_mode', True):
                st.write("**🚀 推論実行中...**")
            
            # モデルを評価モードに設定
            model.eval()
            
            with torch.no_grad():
                # シンプルな推論実行
                outputs = model(**inputs)
            
            # 推論結果のデバッグ情報を表示
            if st.session_state.get('debug_mode', True) and hasattr(outputs, 'logits'):
                logits_shape = outputs.logits.shape
                st.write(f"- ロジットの形状: {logits_shape}")
                
                # 最大スコアを表示
                max_scores = torch.max(outputs.logits, dim=-1)[0]
                st.write(f"- 最大スコア: {torch.max(max_scores).item():.4f}")
                st.write(f"- 平均スコア: {torch.mean(max_scores).item():.4f}")
            
            if st.session_state.get('debug_mode', True):
                st.write("✅ 推論が完了しました")
            return outputs
        except Exception as e:
            st.error(f"推論の実行に失敗しました: {e}")
            if st.session_state.get('debug_mode', True):
                st.write(f"エラーの詳細: {str(e)}")
            return None
    
    @staticmethod
    def post_process_results(
        processor: Any,
        outputs: Any,
        target_sizes: torch.Tensor,
        confidence_threshold: float = 0.5
    ) -> Optional[List[Dict[str, Any]]]:
        """
        推論結果を後処理します
        
        Args:
            processor: OWL-ViTプロセッサー
            outputs: 推論結果
            target_sizes: ターゲットサイズ
            confidence_threshold: 信頼度の閾値
            
        Returns:
            後処理された結果、失敗時はNone
        """
        try:
            if st.session_state.get('debug_mode', True):
                st.write(f"**🔧 後処理実行中 (閾値: {confidence_threshold:.3f})...**")
            
            # シンプルな後処理
            results = processor.post_process_object_detection(
                outputs=outputs,
                threshold=confidence_threshold,
                target_sizes=target_sizes
            )
            
            # 後処理結果のデバッグ情報を表示
            if st.session_state.get('debug_mode', True) and results and len(results) > 0:
                total_boxes = sum(len(result.get("boxes", [])) for result in results)
                st.write(f"- 検出されたボックス数: {total_boxes}")
                
                for i, result in enumerate(results):
                    boxes = result.get("boxes", [])
                    scores = result.get("scores", [])
                    if len(boxes) > 0:
                        st.write(f"- クエリ {i+1}: {len(boxes)}個のボックス")
                        st.write(f"  - 最高スコア: {max(scores):.4f}")
                        st.write(f"  - 平均スコア: {sum(scores)/len(scores):.4f}")
                    else:
                        st.write(f"- クエリ {i+1}: 検出なし")
            elif st.session_state.get('debug_mode', True):
                st.write("- 検出されたボックス: 0個")
            
            if st.session_state.get('debug_mode', True):
                st.write("✅ 後処理が完了しました")
            return results
        except Exception as e:
            st.error(f"結果の後処理に失敗しました: {e}")
            if st.session_state.get('debug_mode', True):
                st.write(f"エラーの詳細: {str(e)}")
            return None


class ImageGuidedDetectionProcessor:
    """画像ガイド検出の処理を担当するクラス"""
    
    @staticmethod
    def prepare_image_guided_inputs(
        processor: Any,
        image: Any,
        query_image: Any
    ) -> Optional[Dict[str, Any]]:
        """
        画像ガイド検出の入力を準備します
        
        Args:
            processor: OWL-ViTプロセッサー
            image: ターゲット画像
            query_image: クエリ画像
            
        Returns:
            準備された入力、失敗時はNone
        """
        try:
            inputs = processor(
                images=image,
                query_images=query_image,
                return_tensors="pt"
            )
            return inputs
        except Exception as e:
            st.error(f"画像ガイド検出の入力準備に失敗しました: {e}")
            return None
    
    @staticmethod
    def run_image_guided_inference(
        model: OwlViTForObjectDetection,
        inputs: Dict[str, Any]
    ) -> Optional[Any]:
        """
        画像ガイド検出の推論を実行します
        
        Args:
            model: OWL-ViTモデル
            inputs: モデルへの入力
            
        Returns:
            推論結果、失敗時はNone
        """
        try:
            with torch.no_grad():
                outputs = model.image_guided_detection(**inputs)
            return outputs
        except Exception as e:
            st.error(f"画像ガイド検出の推論に失敗しました: {e}")
            return None
    
    @staticmethod
    def post_process_image_guided_results(
        processor: Any,
        outputs: Any,
        target_sizes: torch.Tensor,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.3
    ) -> Optional[List[Dict[str, Any]]]:
        """
        画像ガイド検出の結果を後処理します
        
        Args:
            processor: OWL-ViTプロセッサー
            outputs: 推論結果
            target_sizes: ターゲットサイズ
            confidence_threshold: 信頼度の閾値
            nms_threshold: NMSの閾値
            
        Returns:
            後処理された結果、失敗時はNone
        """
        try:
            results = processor.post_process_image_guided_detection(
                outputs=outputs,
                threshold=confidence_threshold,
                nms_threshold=nms_threshold,
                target_sizes=target_sizes
            )
            return results
        except Exception as e:
            st.error(f"画像ガイド検出の結果後処理に失敗しました: {e}")
            return None 