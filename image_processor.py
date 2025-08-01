"""
画像処理を担当するモジュール
単一責任原則に従い、画像の読み込み、前処理、可視化機能を提供
"""

import requests
from typing import Optional, Tuple, List
from PIL import Image
import numpy as np
import streamlit as st


class ImageLoader:
    """画像の読み込みを担当するクラス"""
    
    @staticmethod
    def load_from_url(url: str) -> Optional[Image.Image]:
        """
        URLから画像を読み込みます
        
        Args:
            url: 画像のURL
            
        Returns:
            読み込まれた画像、失敗時はNone
        """
        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()
            return Image.open(response.raw)
        except Exception as e:
            st.error(f"URLからの画像読み込みに失敗しました: {e}")
            return None
    
    @staticmethod
    def load_from_upload(uploaded_file) -> Optional[Image.Image]:
        """
        アップロードされたファイルから画像を読み込みます
        
        Args:
            uploaded_file: Streamlitのアップロードファイルオブジェクト
            
        Returns:
            読み込まれた画像、失敗時はNone
        """
        try:
            return Image.open(uploaded_file)
        except Exception as e:
            st.error(f"アップロードファイルの読み込みに失敗しました: {e}")
            return None
    
    @staticmethod
    def load_from_camera() -> Optional[Image.Image]:
        """
        カメラで撮影した画像を読み込みます
        
        Returns:
            読み込まれた画像、失敗時はNone
        """
        try:
            # Streamlitのカメラ入力を使用
            camera_input = st.camera_input("カメラで撮影")
            if camera_input is not None:
                return Image.open(camera_input)
            return None
        except Exception as e:
            st.error(f"カメラからの画像読み込みに失敗しました: {e}")
            return None


class ImageValidator:
    """画像の検証を担当するクラス"""
    
    @staticmethod
    def validate_image(image: Image.Image) -> bool:
        """
        画像が適切かどうかを検証します
        
        Args:
            image: 検証する画像
            
        Returns:
            画像が適切な場合True
        """
        if image is None:
            return False
        
        # 画像サイズの最小値チェック
        min_size = 100
        if image.width < min_size or image.height < min_size:
            st.warning(f"画像サイズが小さすぎます。最小サイズ: {min_size}x{min_size}")
            return False
        
        # 画像モードのチェック
        if image.mode not in ['RGB', 'RGBA']:
            st.warning("RGBまたはRGBAモードの画像を使用してください")
            return False
        
        return True
    
    @staticmethod
    def get_image_info(image: Image.Image) -> dict:
        """
        画像の基本情報を取得します
        
        Args:
            image: 情報を取得する画像
            
        Returns:
            画像の基本情報を含む辞書
        """
        return {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": image.format,
            "size_mb": len(image.tobytes()) / (1024 * 1024)
        }


class ImageVisualizer:
    """画像の可視化を担当するクラス"""
    
    @staticmethod
    def display_image_with_info(image: Image.Image, caption: str = "画像") -> None:
        """
        画像とその情報を表示します
        
        Args:
            image: 表示する画像
            caption: 画像のキャプション
        """
        st.image(image, caption=caption, use_container_width=True)
        
        # 画像情報を表示
        info = ImageValidator.get_image_info(image)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("幅", f"{info['width']}px")
        with col2:
            st.metric("高さ", f"{info['height']}px")
        with col3:
            st.metric("サイズ", f"{info['size_mb']:.2f}MB")
    
    @staticmethod
    def create_detection_visualization(
        image: Image.Image,
        boxes: List[List[float]],
        scores: List[float],
        labels: List[str],
        confidence_threshold: float = 0.5
    ) -> Image.Image:
        """
        検出結果を可視化した画像を作成します
        
        Args:
            image: 元画像
            boxes: バウンディングボックスの座標リスト
            scores: 信頼度スコアのリスト
            labels: ラベルのリスト
            confidence_threshold: 信頼度の閾値
            
        Returns:
            検出結果を描画した画像
        """
        import cv2
        
        # PIL画像をOpenCV形式に変換
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 閾値以上の検出結果のみを描画
        for box, score, label in zip(boxes, scores, labels):
            if score >= confidence_threshold:
                x1, y1, x2, y2 = [int(coord) for coord in box]
                
                # バウンディングボックスを描画
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # ラベルとスコアを描画
                label_text = f"{label}: {score:.3f}"
                cv2.putText(
                    img_cv, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
        
        # OpenCV形式をPIL形式に戻す
        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


class ImagePreprocessor:
    """画像の前処理を担当するクラス"""
    
    @staticmethod
    def resize_image_if_needed(
        image: Image.Image,
        max_size: Tuple[int, int] = (800, 800)
    ) -> Image.Image:
        """
        必要に応じて画像をリサイズします
        
        Args:
            image: リサイズする画像
            max_size: 最大サイズ (width, height)
            
        Returns:
            リサイズされた画像
        """
        if image.width <= max_size[0] and image.height <= max_size[1]:
            return image
        
        # アスペクト比を保持してリサイズ
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        return image
    
    @staticmethod
    def convert_to_rgb(image: Image.Image) -> Image.Image:
        """
        画像をRGBモードに変換します
        
        Args:
            image: 変換する画像
            
        Returns:
            RGBモードの画像
        """
        if image.mode != 'RGB':
            return image.convert('RGB')
        return image 