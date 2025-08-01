"""
日本語クエリの翻訳を担当するモジュール
OWL-ViTの検出精度向上のため、日本語クエリを英語に翻訳
"""

import streamlit as st
import requests
import json
from typing import Dict, List, Optional


class JapaneseTranslator:
    """日本語クエリの翻訳を担当するクラス"""
    
    # 日本語→英語の翻訳辞書
    TRANSLATION_DICT = {
        # 動物
        "猫": "cat",
        "犬": "dog", 
        "鳥": "bird",
        "馬": "horse",
        "牛": "cow",
        "豚": "pig",
        "羊": "sheep",
        "鶏": "chicken",
        "魚": "fish",
        "ねこ": "cat",
        "いぬ": "dog",
        "とり": "bird",
        "うま": "horse",
        "うし": "cow",
        "ぶた": "pig",
        "ひつじ": "sheep",
        "にわとり": "chicken",
        "さかな": "fish",
        
        # 乗り物
        "車": "car",
        "自動車": "car",
        "自転車": "bicycle",
        "バイク": "motorcycle",
        "オートバイ": "motorcycle",
        "バス": "bus",
        "電車": "train",
        "飛行機": "airplane",
        "船": "ship",
        "ボート": "boat",
        "トラック": "truck",
        "くるま": "car",
        "じてんしゃ": "bicycle",
        "でんしゃ": "train",
        "ひこうき": "airplane",
        "ふね": "ship",
        
        # 家具
        "椅子": "chair",
        "テーブル": "table",
        "机": "desk",
        "ソファ": "sofa",
        "ベッド": "bed",
        "棚": "shelf",
        "本棚": "bookshelf",
        "タンス": "dresser",
        "いす": "chair",
        "つくえ": "desk",
        "たな": "shelf",
        "ほんだな": "bookshelf",
        
        # 食べ物
        "りんご": "apple",
        "バナナ": "banana",
        "オレンジ": "orange",
        "ピザ": "pizza",
        "ケーキ": "cake",
        "パン": "bread",
        "ご飯": "rice",
        "パスタ": "pasta",
        "ハンバーガー": "hamburger",
        "サンドイッチ": "sandwich",
        "ごはん": "rice",
        
        # 電子機器
        "リモコン": "remote control",
        "テレビのリモコン": "television remote control",
        "テレビリモコン": "television remote control",
        "テレビ": "television",
        "TV": "television",
        "パソコン": "computer",
        "コンピューター": "computer",
        "スマートフォン": "smartphone",
        "スマホ": "smartphone",
        "携帯電話": "mobile phone",
        "携帯": "mobile phone",
        "カメラ": "camera",
        "ラジオ": "radio",
        "スピーカー": "speaker",
        "ヘッドフォン": "headphones",
        "イヤホン": "earphones",
        "タブレット": "tablet",
        "プリンター": "printer",
        "キーボード": "keyboard",
        "マウス": "mouse",
        "モニター": "monitor",
        "ディスプレイ": "display",
        "ケータイ": "mobile phone",
        
        # その他の一般的な物体
        "本": "book",
        "ペン": "pen",
        "鉛筆": "pencil",
        "ノート": "notebook",
        "紙": "paper",
        "時計": "clock",
        "腕時計": "watch",
        "財布": "wallet",
        "かばん": "bag",
        "バッグ": "bag",
        "靴": "shoes",
        "くつ": "shoes",
        "帽子": "hat",
        "ぼうし": "hat",
        "眼鏡": "glasses",
        "めがね": "glasses",
        "傘": "umbrella",
        "かさ": "umbrella",
        "ドア": "door",
        "窓": "window",
        "まど": "window",
        "鍵": "key",
        "かぎ": "key",
        "電話": "phone",
        "でんわ": "phone",
        "花瓶": "vase",
        "かびん": "vase",
        "花": "flower",
        "はな": "flower",
        "植物": "plant",
        "しょくぶつ": "plant",
        "木": "tree",
        "き": "tree",
        "草": "grass",
        "くさ": "grass",
    }
    
    @staticmethod
    def translate_japanese_to_english(japanese_query: str, use_api: bool = False) -> str:
        """
        日本語クエリを英語に翻訳します
        
        Args:
            japanese_query: 日本語クエリ
            use_api: 外部APIを使用するかどうか
            
        Returns:
            英語に翻訳されたクエリ
        """
        # 翻訳辞書から検索（完全一致）
        if japanese_query in JapaneseTranslator.TRANSLATION_DICT:
            return JapaneseTranslator.TRANSLATION_DICT[japanese_query]
        
        # 長い語句から順に部分一致で検索（複合語を優先）
        sorted_items = sorted(JapaneseTranslator.TRANSLATION_DICT.items(), key=lambda x: len(x[0]), reverse=True)
        for jp, en in sorted_items:
            if jp in japanese_query:
                return en
        
        # 外部APIを使用する場合
        if use_api:
            api_translation = JapaneseTranslator._translate_with_api(japanese_query)
            if api_translation:
                return api_translation
        
        # 翻訳できない場合は元のクエリを返す
        return japanese_query
    
    @staticmethod
    def _translate_with_api(japanese_query: str) -> Optional[str]:
        """
        外部APIを使用して翻訳します
        
        Args:
            japanese_query: 日本語クエリ
            
        Returns:
            翻訳結果、失敗時はNone
        """
        try:
            # より安定した翻訳APIを使用
            url = "https://translate.googleapis.com/translate_a/single"
            params = {
                "client": "gtx",
                "sl": "ja",
                "tl": "en",
                "dt": "t",
                "q": japanese_query
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                try:
                    # Google Translate APIの応答形式を解析
                    result = response.json()
                    if result and len(result) > 0 and len(result[0]) > 0:
                        translated_text = result[0][0][0].strip()
                        if translated_text and translated_text != japanese_query:
                            return translated_text
                except (json.JSONDecodeError, IndexError, KeyError):
                    st.warning("翻訳APIの応答形式が不正です")
            else:
                st.warning(f"翻訳APIの応答エラー: {response.status_code}")
            
        except requests.exceptions.Timeout:
            st.warning("翻訳APIのタイムアウトが発生しました")
        except requests.exceptions.RequestException as e:
            st.warning(f"翻訳APIの通信エラー: {e}")
        except Exception as e:
            st.warning(f"外部翻訳APIの使用に失敗しました: {e}")
        
        return None
    
    @staticmethod
    def translate_with_multiple_methods(japanese_query: str) -> List[str]:
        """
        複数の方法で翻訳を試行します
        
        Args:
            japanese_query: 日本語クエリ
            
        Returns:
            翻訳結果のリスト
        """
        translations = []
        
        # 1. 辞書翻訳
        dict_translation = JapaneseTranslator.translate_japanese_to_english(japanese_query, use_api=False)
        if dict_translation != japanese_query:
            translations.append(dict_translation)
        
        # 2. API翻訳（エラーが発生した場合はスキップ）
        try:
            api_translation = JapaneseTranslator._translate_with_api(japanese_query)
            if api_translation and api_translation not in translations:
                translations.append(api_translation)
        except Exception:
            # API翻訳でエラーが発生した場合は無視
            pass
        
        # 3. 元のクエリも含める
        translations.append(japanese_query)
        
        return translations
    
    @staticmethod
    def is_japanese(text: str) -> bool:
        """
        テキストが日本語かどうかを判定します
        
        Args:
            text: 判定するテキスト
            
        Returns:
            日本語の場合True
        """
        # ひらがな、カタカナ、漢字が含まれているかチェック
        for char in text:
            if ord(char) > 127:
                return True
        return False
    
    @staticmethod
    def get_multiple_query_variations(query: str, use_api: bool = True) -> List[str]:
        """
        クエリの複数のバリエーションを生成します
        
        Args:
            query: 元のクエリ
            use_api: 外部APIを使用するかどうか
            
        Returns:
            クエリのバリエーションリスト
        """
        variations = [query]
        
        # 日本語の場合、複数の翻訳方法を試す
        if JapaneseTranslator.is_japanese(query):
            if use_api:
                # 複数の翻訳方法を使用
                translations = JapaneseTranslator.translate_with_multiple_methods(query)
                for translation in translations:
                    if translation not in variations:
                        variations.append(translation)
            else:
                # 辞書翻訳のみ
                english_translation = JapaneseTranslator.translate_japanese_to_english(query, use_api=False)
                if english_translation != query:
                    variations.append(english_translation)
        
        # 英語の場合、複数の形式を試す
        else:
            # 単数形・複数形のバリエーション
            if query.endswith('s'):
                variations.append(query[:-1])  # 複数形→単数形
            else:
                variations.append(query + 's')  # 単数形→複数形
            
            # より多くの英語バリエーションを追加
            if not query.startswith('a ') and not query.startswith('the '):
                variations.extend([
                    f"a {query}",
                    f"the {query}",
                    f"a photo of {query}",
                    f"a photo of a {query}",
                    f"an image of {query}",
                    f"an image of a {query}"
                ])
        
        # 重複を除去
        unique_variations = list(dict.fromkeys(variations))
        
        return unique_variations 