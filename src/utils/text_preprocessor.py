#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
텍스트 전처리 유틸리티 모듈
"""

import re
import html
import json as json_module
import unicodedata
import logging
from typing import Optional


class TextPreprocessor:
    """텍스트 전처리를 담당하는 클래스"""
    
    def __init__(self):
        self.MAX_TEXT_LENGTH = 8000
    
    def preprocess_text(self, text: str) -> str:
        """입력 텍스트를 AI 처리에 적합하게 전처리하는 메서드"""
        logging.info(f"전처리 시작: 입력 길이={len(text) if text else 0}")
        logging.info(f"전처리 입력 미리보기: {text[:100] if text else 'None'}...")

        # null 체크
        if not text:
            logging.info("전처리: 빈 텍스트 입력")
            return ""
        
        # 문자열로 변환 및 HTML 엔티티 디코딩
        text = str(text)
        text = html.unescape(text)  # &amp; → &, &lt; → < 등
        logging.info(f"HTML 디코딩 후 길이: {len(text)}")
        
        # HTML 태그 제거 및 텍스트 형태로 변환 (구조 유지)
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)      # <br> → 줄바꿈
        text = re.sub(r'</p>', '\n\n', text, flags=re.IGNORECASE)         # </p> → 단락 구분
        text = re.sub(r'<p[^>]*>', '\n', text, flags=re.IGNORECASE)       # <p> → 줄바꿈
        text = re.sub(r'<li[^>]*>', '\n• ', text, flags=re.IGNORECASE)    # <li> → 불릿포인트
        text = re.sub(r'</li>', '', text, flags=re.IGNORECASE)            # </li> 제거
        text = re.sub(r'<[^>]+>', '', text)                               # 나머지 HTML 태그 모두 제거
        logging.info(f"HTML 태그 제거 후 길이: {len(text)}")
        
        # 구 앱 이름을 바이블 애플로 교체 (전처리 단계에서)
        text = re.sub(r'바이블\s*애플\s*\(구\)\s*다번역\s*성경\s*찬송', '바이블 애플', text, flags=re.IGNORECASE)
        text = re.sub(r'바이블\s*애플\s*\(구\)\s*다번역성경찬송', '바이블 애플', text, flags=re.IGNORECASE)
        text = re.sub(r'\(구\)\s*다번역\s*성경\s*찬송', '바이블 애플', text, flags=re.IGNORECASE)
        text = re.sub(r'\(구\)\s*다번역성경찬송', '바이블 애플', text, flags=re.IGNORECASE)
        text = re.sub(r'다번역\s*성경\s*찬송', '바이블 애플', text, flags=re.IGNORECASE)
        text = re.sub(r'다번역성경찬송', '바이블 애플', text, flags=re.IGNORECASE)
        
        # 공백 및 줄바꿈 정규화 - 일관된 형태로 변환
        text = re.sub(r'\n{3,}', '\n\n', text)    # 3개 이상 줄바꿈 → 2개로 제한
        text = re.sub(r'[ \t]+', ' ', text)       # 연속 공백/탭 → 단일 공백
        text = text.strip()                       # 앞뒤 공백 제거
        
        logging.info(f"전처리 완료: 최종 길이={len(text)}")
        logging.info(f"전처리 결과 미리보기: {text[:100]}...")
        
        return text

    def preprocess_text_for_metadata(self, text: str, for_metadata: bool = False) -> str:
        """메타데이터용 텍스트 전처리"""
        if not text or text == 'None':
            return ""
        
        text = str(text)
        text = html.unescape(text)
        
        # HTML 태그 제거
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</p>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<p[^>]*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<[^>]+>', '', text)
        
        # 유니코드 정규화
        text = unicodedata.normalize('NFC', text)
        
        # 공백 정리
        if for_metadata:
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r'[ \t]+', ' ', text)
        else:
            text = re.sub(r'\s+', ' ', text)
        
        text = text.strip()
        
        # 길이 제한
        max_length = 1000 if for_metadata else self.MAX_TEXT_LENGTH
        if len(text) > max_length:
            text = text[:max_length-3] + "..."
        
        return text

    def escape_json_string(self, text: str) -> str:
        """JSON 문자열 이스케이프 처리"""
        if not text:
            return ""
        escaped = json_module.dumps(text, ensure_ascii=False) # ensure_ascii=False: 한글 깨짐 방지
        return escaped[1:-1]  # 앞뒤 따옴표 제거

    def remove_old_app_name(self, text: str) -> str:
        """이전 앱 이름을 제거하는 메서드"""
        patterns_to_remove = [
            r'\s*\(구\)\s*다번역성경찬송',
            r'\s*\(구\)다번역성경찬송',
            r'바이블\s*애플\s*\(구\)\s*다번역성경찬송',
            r'바이블애플\s*\(구\)다번역성경찬송',
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        text = re.sub(r'(GOODTV\s+바이블\s*애플)\s+', r'\1', text)
        
        return text

    def clean_generated_text(self, text: str) -> str:
        """생성된 텍스트를 정리하고 검증하는 메서드"""
        if not text:
            return ""
        # 제어 문자 제거
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        text = re.sub(r'[\b\r\f\v]', '', text)

        # 영어 약어 제거
        text = re.sub(r'\b[a-z]{1,2}\b(?:\s+[a-z]{1,2}\b)*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[а-я]+', '', text)
        text = re.sub(r'[α-ω]+', '', text)

        # 한글 문자 제거
        text = re.sub(r'[^\w\s가-힣.,!?()"\'-]{3,}', '', text)
        text = re.sub(r'[.,;:!?]{3,}', '.', text)

        # 공백 정리
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text

    def clean_answer_text(self, text: str) -> str:
        """답변 텍스트를 정리하고 포맷팅하는 메서드 (Quill 에디터용)"""
        if not text:
            return ""
        
        # 제어 문자만 제거하고 HTML 태그는 유지
        text = re.sub(r'[\b\r\f\v]', '', text)
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)

        # HTML 태그 제거하지 않음 (Quill 에디터용)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        
        # HTML 태그 내부의 공백만 정리 (태그 자체는 유지)
        text = re.sub(r'>\s+<', '><', text)  # 태그 사이 공백 제거
        text = re.sub(r'<p>\s+', '<p>', text)  # <p> 태그 내부 앞 공백 제거
        text = re.sub(r'\s+</p>', '</p>', text)  # </p> 태그 앞 공백 제거
        
        text = self.remove_old_app_name(text)
        
        return text

    def extract_keywords(self, text: str) -> list:
        """텍스트에서 핵심 키워드 추출"""
        # 불용어 제거용 리스트
        stop_words = {'는', '은', '이', '가', '을', '를', '에', '에서', '로', '으로', '와', '과', '의', '도', '만', '까지', '부터', '께서', '에게', '한테', '로부터', '으로부터'}
        
        # 특수문자 제거 및 단어 분리
        words = re.findall(r'[가-힣a-zA-Z0-9]+', text)
        
        # 불용어 제거 및 2글자 이상 단어만 선택
        keywords = [word for word in words if len(word) >= 2 and word not in stop_words]
        
        return keywords

    def extract_key_concepts(self, text: str) -> list:
        """텍스트에서 핵심 개념을 추출"""        
        # 2글자 이상의 한글 명사 추출
        korean_nouns = re.findall(r'[가-힣]{2,}', text)
        
        # 영어 단어 추출
        english_words = re.findall(r'[a-zA-Z]{3,}', text)
        
        # 중복 제거 및 정리
        concepts = []
        for word in korean_nouns + english_words:
            word = word.lower().strip()
            if len(word) >= 2 and word not in ['있나요', '해주세요', '도와주세요', '문의', '질문']:
                concepts.append(word)
        
        return list(set(concepts))  # 중복 제거

    def extract_translations_from_text(self, text: str) -> list:
        """텍스트에서 성경 번역본명을 추출"""
        
        translation_patterns = [
            r'NIV',
            r'KJV', 
            r'ESV',
            r'개역개정',
            r'개역한글',
            r'개역\s*개정',
            r'개역\s*한글',
            r'영어\s*번역본',
            r'영문\s*성경',
            r'한글\s*번역본',
            r'한국어\s*성경'
        ]
        
        found_translations = []
        for pattern in translation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            found_translations.extend(matches)
        
        # 중복 제거 및 정규화
        normalized = []
        for trans in found_translations:
            trans = re.sub(r'\s+', '', trans)  # 공백 제거
            if trans not in normalized:
                normalized.append(trans)
        
        return normalized
