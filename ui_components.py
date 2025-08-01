"""
UIコンポーネントを管理するモジュール
単一責任原則に従い、StreamlitのUI要素を提供
"""

import streamlit as st
from typing import List, Optional, Tuple
from PIL import Image


class SidebarManager:
    """サイドバーの管理を担当するクラス"""
    
    @staticmethod
    def create_model_selection() -> str:
        """
        モデル選択のUIを作成します
        
        Returns:
            選択されたモデル名
        """
        st.sidebar.header("🔧 設定")
        
        from model_manager import ModelLoader
        available_models = ModelLoader.get_available_models()
        
        selected_model = st.sidebar.selectbox(
            "モデルを選択",
            available_models,
            index=0,
            help="使用するOWL-ViTモデルを選択してください（Base-32が推奨）"
        )
        
        return selected_model
    
    @staticmethod
    def create_detection_mode_selection() -> str:
        """
        検出モード選択のUIを作成します
        
        Returns:
            選択された検出モード
        """
        detection_mode = st.sidebar.selectbox(
            "検出モード",
            ["シンプル検出", "テキストガイド検出", "画像ガイド検出"],
            help="シンプル検出: 基本的な検出、テキストガイド検出: テキストクエリで物体を検出、画像ガイド検出: 類似画像で物体を検出"
        )
        
        return detection_mode
    
    @staticmethod
    def create_confidence_settings() -> Tuple[float, float]:
        """
        信頼度設定のUIを作成します
        
        Returns:
            (信頼度閾値, NMS閾値)のタプル
        """
        st.sidebar.subheader("信頼度設定")
        
        confidence_threshold = st.sidebar.slider(
            "信頼度閾値",
            min_value=0.0,
            max_value=1.0,
            value=0.1,  # デフォルト値を0.3から0.1に下げる
            step=0.05,
            help="検出結果の信頼度閾値を設定（低い値ほど多くの物体を検出）"
        )
        
        nms_threshold = st.sidebar.slider(
            "NMS閾値",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Non-Maximum Suppressionの閾値を設定"
        )
        
        # デバッグモードの追加
        debug_mode = st.sidebar.checkbox(
            "デバッグモード",
            value=True,  # デフォルトで有効
            help="詳細なデバッグ情報を表示"
        )
        
        # デバッグモードをセッション状態に保存
        st.session_state.debug_mode = debug_mode
        
        return confidence_threshold, nms_threshold


class InputManager:
    """入力管理を担当するクラス"""
    
    @staticmethod
    def create_image_input_section() -> Optional[Image.Image]:
        """
        画像入力セクションのUIを作成します
        
        Returns:
            読み込まれた画像、失敗時はNone
        """
        st.header("📷 画像入力")
        
        input_method = st.selectbox(
            "画像入力方法を選択",
            ["サンプル画像を使用", "画像をアップロード", "URLから画像を取得", "カメラで撮影"]
        )
        
        image = None
        
        if input_method == "サンプル画像を使用":
            image = InputManager._load_sample_image()
        elif input_method == "画像をアップロード":
            image = InputManager._load_uploaded_image()
        elif input_method == "URLから画像を取得":
            image = InputManager._load_url_image()
        elif input_method == "カメラで撮影":
            image = InputManager._load_camera_image()
        
        if image:
            st.success("画像が正常に読み込まれました")
            st.image(image, caption="入力画像", use_container_width=True)
            
            # 画像情報の表示
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("幅", f"{image.width}px")
            with col2:
                st.metric("高さ", f"{image.height}px")
            with col3:
                st.metric("サイズ", f"{image.width * image.height:,}px²")
            
            return image
        else:
            st.warning("画像を入力してください")
            return None
    
    @staticmethod
    def _load_sample_image() -> Optional[Image.Image]:
        """サンプル画像を読み込みます"""
        import requests
        from io import BytesIO
        
        # サンプル画像の選択
        sample_images = {
            "オフィスシーン": "https://images.unsplash.com/photo-1497366216548-37526070297c?w=800&h=600&fit=crop",
            "キッチンシーン": "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=800&h=600&fit=crop",
            "リビングルーム": "https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=800&h=600&fit=crop",
            "街並み": "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=800&h=600&fit=crop",
            "自然風景": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&h=600&fit=crop"
        }
        
        selected_image = st.selectbox(
            "サンプル画像を選択",
            list(sample_images.keys())
        )
        
        if selected_image:
            try:
                url = sample_images[selected_image]
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                return image
            except Exception as e:
                st.error(f"サンプル画像の読み込みに失敗しました: {e}")
                return None
        
        return None
    
    @staticmethod
    def _load_uploaded_image() -> Optional[Image.Image]:
        """アップロードされた画像を読み込みます"""
        uploaded_file = st.file_uploader(
            "画像ファイルを選択してください",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff']
        )
        
        if uploaded_file is not None:
            from image_processor import ImageLoader
            return ImageLoader.load_from_upload(uploaded_file)
        
        return None
    
    @staticmethod
    def _load_url_image() -> Optional[Image.Image]:
        """URLから画像を読み込みます"""
        image_url = st.text_input(
            "画像のURLを入力してください",
            value="http://images.cocodataset.org/val2017/000000039769.jpg"
        )
        
        if image_url:
            from image_processor import ImageLoader
            return ImageLoader.load_from_url(image_url)
        
        return None
    
    @staticmethod
    def _load_camera_image() -> Optional[Image.Image]:
        """カメラで撮影した画像を読み込みます"""
        try:
            camera_input = st.camera_input("カメラで撮影")
            if camera_input is not None:
                return Image.open(camera_input)
            return None
        except Exception as e:
            st.error(f"カメラからの画像読み込みに失敗しました: {e}")
            return None
    
    @staticmethod
    def _get_query_suggestions(query: str) -> List[str]:
        """クエリの提案を取得します"""
        # クエリ辞書
        query_dict = {
            "猫": ["cat", "kitten", "feline", "ペット"],
            "犬": ["dog", "puppy", "canine", "ペット"],
            "車": ["car", "automobile", "vehicle", "乗り物"],
            "人": ["person", "human", "people", "人物"],
            "椅子": ["chair", "seat", "furniture", "家具"],
            "テーブル": ["table", "desk", "furniture", "家具"],
            "テレビ": ["television", "tv", "electronics", "電子機器"],
            "スマートフォン": ["smartphone", "phone", "mobile", "電子機器"],
            "りんご": ["apple", "fruit", "food", "食べ物"],
            "ピザ": ["pizza", "food", "meal", "食べ物"],
            "本": ["book", "reading", "literature", "書籍"],
            "花": ["flower", "plant", "nature", "植物"],
            "木": ["tree", "plant", "nature", "植物"],
            "建物": ["building", "house", "architecture", "建築"],
            "空": ["sky", "cloud", "weather", "天気"],
            "海": ["sea", "ocean", "water", "自然"],
            "山": ["mountain", "hill", "nature", "自然"]
        }
        
        suggestions = []
        query_lower = query.lower()
        
        # 完全一致
        for key, values in query_dict.items():
            if query_lower in key.lower() or any(query_lower in v.lower() for v in values):
                suggestions.extend(values)
        
        # 部分一致
        for key, values in query_dict.items():
            if any(query_lower in v.lower() for v in values) or any(v.lower() in query_lower for v in values):
                suggestions.extend(values)
        
        # 重複を除去して返す
        return list(set(suggestions))
    
    @staticmethod
    def create_text_query_section() -> Optional[Tuple[List[str], str]]:
        """
        テキストクエリ入力セクションのUIを作成します
        
        Returns:
            (テキストクエリのリスト, 翻訳方法)のタプル、失敗時はNone
        """
        st.header("🔤 テキストクエリ入力")
        
        # 入力方法の選択
        input_method = st.selectbox(
            "テキストクエリの入力方法を選択",
            ["手動入力", "プリセットクエリ", "テキスト検索"]
        )
        
        text_queries = []
        translation_method = "辞書翻訳のみ"
        
        if input_method == "手動入力":
            # 手動入力
            st.subheader("手動でテキストクエリを入力")
            
            # 複数のテキストクエリを入力
            num_queries = st.number_input(
                "クエリの数",
                min_value=1,
                max_value=5,
                value=1,
                help="検出したい物体のクエリ数を設定"
            )
            
            for i in range(num_queries):
                query = st.text_input(
                    f"テキストクエリ {i+1}",
                    placeholder="例: 猫, car, 椅子",
                    help="検出したい物体を入力してください（日本語・英語対応）"
                )
                if query.strip():
                    text_queries.append(query.strip())
            
            # 翻訳方法の選択
            translation_method = st.selectbox(
                "翻訳方法",
                ["辞書翻訳のみ", "辞書翻訳 + API翻訳"],
                help="日本語クエリの翻訳方法を選択"
            )
        
        elif input_method == "プリセットクエリ":
            # プリセットクエリ
            st.subheader("プリセットクエリから選択")
            
            # カテゴリ別のプリセットクエリ
            preset_categories = {
                "動物": ["cat", "dog", "bird", "horse", "fish", "猫", "犬", "鳥", "馬", "魚"],
                "乗り物": ["car", "bicycle", "motorcycle", "bus", "train", "車", "自転車", "バイク", "バス", "電車"],
                "家具": ["chair", "table", "sofa", "bed", "desk", "椅子", "テーブル", "ソファ", "ベッド", "机"],
                "食べ物": ["apple", "banana", "pizza", "cake", "bread", "りんご", "バナナ", "ピザ", "ケーキ", "パン"],
                "電子機器": ["television", "computer", "smartphone", "camera", "テレビ", "パソコン", "スマートフォン", "カメラ"]
            }
            
            selected_category = st.selectbox(
                "カテゴリを選択",
                list(preset_categories.keys())
            )
            
            if selected_category:
                preset_queries = preset_categories[selected_category]
                
                # 複数選択可能
                selected_queries = st.multiselect(
                    f"{selected_category}からクエリを選択",
                    preset_queries,
                    default=preset_queries[:3],  # デフォルトで最初の3つを選択
                    help="検出したい物体を複数選択できます"
                )
                
                text_queries = selected_queries
        
        elif input_method == "テキスト検索":
            # テキスト検索機能
            st.subheader("テキスト検索")
            
            # 検索ボックス
            search_query = st.text_input(
                "検索したい物体を入力",
                placeholder="例: 猫, car, 椅子",
                help="検出したい物体を入力してください（日本語・英語対応）"
            )
            
            # クエリ提案機能
            if search_query.strip():
                # 検索クエリを分割（カンマ区切り）
                queries = [q.strip() for q in search_query.split(',') if q.strip()]
                
                # 検索結果の表示
                st.write("**検索結果:**")
                for i, query in enumerate(queries):
                    st.write(f"{i+1}. {query}")
                
                # クエリ提案
                suggestions = InputManager._get_query_suggestions(search_query)
                if suggestions:
                    st.write("**💡 関連するクエリの提案:**")
                    for suggestion in suggestions[:5]:  # 最大5つの提案
                        if st.button(f"➕ {suggestion}", key=f"suggestion_{suggestion}"):
                            if suggestion not in queries:
                                queries.append(suggestion)
                                st.rerun()
                
                # 検索結果から選択
                if len(queries) > 1:
                    selected_indices = st.multiselect(
                        "検出に使用するクエリを選択",
                        range(len(queries)),
                        default=range(min(3, len(queries))),
                        format_func=lambda x: queries[x]
                    )
                    text_queries = [queries[i] for i in selected_indices]
                else:
                    text_queries = queries
                
                # 翻訳方法の選択
                translation_method = st.selectbox(
                    "翻訳方法",
                    ["辞書翻訳のみ", "辞書翻訳 + API翻訳"],
                    help="日本語クエリの翻訳方法を選択"
                )
        
        # 入力されたクエリの表示
        if text_queries:
            st.subheader("📝 入力されたクエリ")
            for i, query in enumerate(text_queries):
                st.write(f"{i+1}. **{query}**")
            
            # 検証
            from model_manager import TextQueryProcessor
            if TextQueryProcessor.validate_text_queries(text_queries):
                return text_queries, translation_method
            else:
                return None
        else:
            st.warning("テキストクエリを入力してください")
            return None
    
    @staticmethod
    def create_query_image_section() -> Optional[Image.Image]:
        """
        クエリ画像入力セクションのUIを作成します
        
        Returns:
            クエリ画像、失敗時はNone
        """
        st.header("🖼️ クエリ画像入力")
        
        input_method = st.selectbox(
            "クエリ画像の入力方法",
            ["サンプル画像を使用", "画像をアップロード", "URLから画像を取得"]
        )
        
        from image_processor import ImageLoader, ImageValidator, ImagePreprocessor
        
        query_image = None
        
        if input_method == "サンプル画像を使用":
            sample_url = "http://images.cocodataset.org/val2017/000000001675.jpg"
            query_image = ImageLoader.load_from_url(sample_url)
        elif input_method == "画像をアップロード":
            uploaded_file = st.file_uploader(
                "クエリ画像ファイルを選択してください",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff']
            )
            if uploaded_file is not None:
                query_image = ImageLoader.load_from_upload(uploaded_file)
        elif input_method == "URLから画像を取得":
            image_url = st.text_input(
                "クエリ画像のURLを入力してください",
                value="http://images.cocodataset.org/val2017/000000001675.jpg"
            )
            if image_url:
                query_image = ImageLoader.load_from_url(image_url)
        
        if query_image is not None:
            # 画像の前処理
            query_image = ImagePreprocessor.convert_to_rgb(query_image)
            query_image = ImagePreprocessor.resize_image_if_needed(query_image)
            
            # 画像の検証
            if ImageValidator.validate_image(query_image):
                from image_processor import ImageVisualizer
                ImageVisualizer.display_image_with_info(query_image, "クエリ画像")
                return query_image
        
        return None


class ResultsManager:
    """結果表示を担当するクラス"""
    
    @staticmethod
    def display_detection_results(
        results: List[dict],
        text_queries: List[str],
        confidence_threshold: float
    ) -> None:
        """
        検出結果を表示します
        
        Args:
            results: 検出結果のリスト
            text_queries: テキストクエリのリスト
            confidence_threshold: 信頼度閾値
        """
        st.header("🎯 検出結果")
        
        if not results:
            st.warning("検出結果がありません")
            return
        
        for i, result in enumerate(results):
            boxes = result["boxes"]
            scores = result["scores"]
            labels = result["labels"]
            
            if len(boxes) == 0:
                st.info(f"クエリ '{text_queries[i]}' に対する検出結果がありません")
                continue
            
            st.subheader(f"クエリ: {text_queries[i]}")
            
            # 検出結果の詳細表示
            for box, score, label in zip(boxes, scores, labels):
                if score >= confidence_threshold:
                    box_coords = [round(coord, 2) for coord in box.tolist()]
                    st.write(f"📍 検出位置: {box_coords}, 信頼度: {score:.3f}")
            
            # 統計情報
            valid_detections = sum(1 for score in scores if score >= confidence_threshold)
            st.metric("検出数", valid_detections)
    
    @staticmethod
    def display_image_guided_results(
        results: List[dict],
        confidence_threshold: float
    ) -> None:
        """
        画像ガイド検出結果を表示します
        
        Args:
            results: 検出結果のリスト
            confidence_threshold: 信頼度閾値
        """
        st.header("🎯 画像ガイド検出結果")
        
        if not results:
            st.warning("検出結果がありません")
            return
        
        for i, result in enumerate(results):
            boxes = result["boxes"]
            scores = result["scores"]
            
            if len(boxes) == 0:
                st.info("類似オブジェクトの検出結果がありません")
                continue
            
            st.subheader("類似オブジェクトの検出結果")
            
            # 検出結果の詳細表示
            for box, score in zip(boxes, scores):
                if score >= confidence_threshold:
                    box_coords = [round(coord, 2) for coord in box.tolist()]
                    st.write(f"📍 検出位置: {box_coords}, 類似度: {score:.3f}")
            
            # 統計情報
            valid_detections = sum(1 for score in scores if score >= confidence_threshold)
            st.metric("検出数", valid_detections)
    
    @staticmethod
    def create_download_section(visualized_image: Image.Image) -> None:
        """
        ダウンロードセクションのUIを作成します
        
        Args:
            visualized_image: 可視化された画像
        """
        st.header("💾 結果のダウンロード")
        
        # 画像をバイト形式に変換
        import io
        img_buffer = io.BytesIO()
        visualized_image.save(img_buffer, format='PNG')
        
        st.download_button(
            label="検出結果画像をダウンロード",
            data=img_buffer.getvalue(),
            file_name="owl_vit_detection_result.png",
            mime="image/png"
        ) 