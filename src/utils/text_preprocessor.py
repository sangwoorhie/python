#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
텍스트 전처리 유틸리티 모듈
- AI API 호출 및 검색을 위한 텍스트 정제 및 표준화
- HTML 태그 제거, 앱 이름 통일, 공백 정규화
- 키워드 추출, 성경 번역본 인식, JSON 이스케이프 처리
- Quill 에디터 호환 텍스트 포맷팅 및 메타데이터 최적화
"""

import re
import html
import json as json_module
import unicodedata
import logging
from typing import Optional


# ===== 텍스트 전처리를 담당하는 메인 클래스 =====
class TextPreprocessor:
    
    # TextPreprocessor 초기화 - 텍스트 처리 설정
    def __init__(self):
        self.MAX_TEXT_LENGTH = 8000  # 최대 텍스트 길이 제한 (AI API 호환성)
    
    # 입력 텍스트를 AI 처리에 적합하게 전처리하는 메서드
    # ☆ 실제 전처리 메서드
    # HTML 태그 제거, 앱 이름 통일, 공백 정규화
    def preprocess_text(self, text: str) -> str:
        # 1단계: 입력 텍스트 유효성 검사 및 로깅
        logging.info(f"전처리 시작: 입력 길이={len(text) if text else 0}")
        # logging.info(f"전처리 입력 미리보기: {text[:100] if text else 'None'}...")

        # 2단계: null 체크 - 빈 텍스트 처리
        if not text:
            logging.info("전처리: 빈 텍스트 입력")
            return ""
        
        # 3단계: 문자열로 변환 및 HTML 엔티티 디코딩
        text = str(text)  # 안전한 문자열 변환
        text = html.unescape(text)  # &amp; → &, &lt; → < 등 HTML 엔티티 복원
        logging.info(f"HTML 디코딩 후 길이: {len(text)}")
        
        # 4단계: HTML 태그 제거 및 텍스트 형태로 변환 (구조 유지)
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)      # <br> → 줄바꿈
        text = re.sub(r'</p>', '\n\n', text, flags=re.IGNORECASE)         # </p> → 단락 구분
        text = re.sub(r'<p[^>]*>', '\n', text, flags=re.IGNORECASE)       # <p> → 줄바꿈
        text = re.sub(r'<li[^>]*>', '\n• ', text, flags=re.IGNORECASE)    # <li> → 불릿포인트
        text = re.sub(r'</li>', '', text, flags=re.IGNORECASE)            # </li> 제거
        text = re.sub(r'<[^>]+>', '', text)                               # 나머지 HTML 태그 모두 제거
        logging.info(f"HTML 태그 제거 후 길이: {len(text)}")
        
        # 5단계: 구 앱 이름을 바이블 애플로 통일 (브랜드 일관성 유지)
        text = re.sub(r'바이블\s*애플\s*\(구\)\s*다번역\s*성경\s*찬송', '바이블 애플', text, flags=re.IGNORECASE)
        text = re.sub(r'바이블\s*애플\s*\(구\)\s*다번역성경찬송', '바이블 애플', text, flags=re.IGNORECASE)
        text = re.sub(r'\(구\)\s*다번역\s*성경\s*찬송', '바이블 애플', text, flags=re.IGNORECASE)
        text = re.sub(r'\(구\)\s*다번역성경찬송', '바이블 애플', text, flags=re.IGNORECASE)
        text = re.sub(r'다번역\s*성경\s*찬송', '바이블 애플', text, flags=re.IGNORECASE)
        text = re.sub(r'다번역성경찬송', '바이블 애플', text, flags=re.IGNORECASE)
        
        # 6단계: 공백 및 줄바꿈 정규화 - AI 처리에 최적화된 형태로 변환
        text = re.sub(r'\n{3,}', '\n\n', text)    # 3개 이상 줄바꿈 → 2개로 제한 (가독성)
        text = re.sub(r'[ \t]+', ' ', text)       # 연속 공백/탭 → 단일 공백 (토큰 절약)
        text = text.strip()                       # 앞뒤 공백 제거 (깔끔한 처리)
        
        # 7단계: 전처리 완료 로깅
        logging.info(f"전처리 완료: 최종 길이={len(text)}")
        # logging.info(f"전처리 결과 미리보기: {text[:100]}...")
        
        return text

    # 메타데이터용 텍스트 전처리 (길이 제한 및 최적화)
    def preprocess_text_for_metadata(self, text: str, for_metadata: bool = False) -> str:
        # 1단계: 입력 텍스트 유효성 검사
        if not text or text == 'None':
            return ""
        
        # 2단계: 기본 텍스트 정제
        text = str(text)  # 안전한 문자열 변환
        text = html.unescape(text)  # HTML 엔티티 디코딩
        
        # 3단계: HTML 태그 제거 (메타데이터용 간소화)
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)  # <br> → 줄바꿈
        text = re.sub(r'</p>', '\n', text, flags=re.IGNORECASE)       # </p> → 줄바꿈
        text = re.sub(r'<p[^>]*>', '', text, flags=re.IGNORECASE)     # <p> 제거
        text = re.sub(r'<[^>]+>', '', text)                           # 모든 HTML 태그 제거
        
        # 4단계: 유니코드 정규화 (NFC: 정규 결합)
        text = unicodedata.normalize('NFC', text)
        
        # 5단계: 공백 정리 (메타데이터 용도에 따라 분기)
        if for_metadata:
            # 메타데이터용: 구조 유지하며 정리
            text = re.sub(r'\n{3,}', '\n\n', text)  # 과도한 줄바꿈 제한
            text = re.sub(r'[ \t]+', ' ', text)     # 연속 공백 정리
        else:
            # 일반용: 모든 공백을 단일 공백으로 통일
            text = re.sub(r'\s+', ' ', text)
        
        text = text.strip()  # 앞뒤 공백 제거
        
        # 6단계: 길이 제한 (메타데이터와 일반 처리 분기)
        max_length = 1000 if for_metadata else self.MAX_TEXT_LENGTH
        if len(text) > max_length:
            text = text[:max_length-3] + "..."  # 안전한 자르기 (말줄임표 추가)
        
        return text

    # JSON 문자열 이스케이프 처리 (API 호출용)
    def escape_json_string(self, text: str) -> str:
        # 1단계: 빈 텍스트 검사
        if not text:
            return ""
        
        # 2단계: JSON 직렬화로 특수문자 이스케이프 (한글 보존)
        escaped = json_module.dumps(text, ensure_ascii=False) # ensure_ascii=False: 한글 깨짐 방지
        
        # 3단계: 앞뒤 따옴표 제거하여 순수 이스케이프된 문자열 반환
        return escaped[1:-1]

    # 이전 앱 이름을 제거하는 메서드 (브랜드 통일성)
    def remove_old_app_name(self, text: str) -> str:
        # 1단계: 제거할 구 앱 이름 패턴 정의
        patterns_to_remove = [
            r'\s*\(구\)\s*다번역성경찬송',
            r'\s*\(구\)다번역성경찬송',
            r'바이블\s*애플\s*\(구\)\s*다번역성경찬송',
            r'바이블애플\s*\(구\)다번역성경찬송',
        ]
        
        # 2단계: 각 패턴을 순차적으로 제거 (대소문자 무시)
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # 3단계: GOODTV 바이블 애플 뒤 불필요한 공백 정리
        text = re.sub(r'(GOODTV\s+바이블\s*애플)\s+', r'\1', text)
        
        return text

    # 생성된 텍스트를 정리하고 검증하는 메서드 (AI 출력 정제)
    def clean_generated_text(self, text: str) -> str:
        # 1단계: 빈 텍스트 검사
        if not text:
            return ""
        
        # 2단계: 제어 문자 제거 (ASCII 제어 문자)
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)  # NULL, 백스페이스 등
        text = re.sub(r'[\b\r\f\v]', '', text)  # 백스페이스, 캐리지 리턴, 폼 피드, 세로 탭

        # 3단계: 불필요한 언어 문자 제거 (한국어 앱용 정제)
        text = re.sub(r'\b[a-z]{1,2}\b(?:\s+[a-z]{1,2}\b)*', '', text, flags=re.IGNORECASE)  # 영어 약어
        text = re.sub(r'[а-я]+', '', text)  # 키릴 문자 (러시아어)
        text = re.sub(r'[α-ω]+', '', text)  # 그리스 문자

        # 4단계: 특수 문자 및 과도한 구두점 정리
        text = re.sub(r'[^\w\s가-힣.,!?()"\'-]{3,}', '', text)  # 3개 이상 연속 특수문자 제거
        text = re.sub(r'[.,;:!?]{3,}', '.', text)  # 과도한 구두점을 마침표로 통일

        # 5단계: 공백 정리 및 최종 정제
        text = re.sub(r'\s+', ' ', text)  # 연속 공백 → 단일 공백
        text = text.strip()  # 앞뒤 공백 제거
        
        return text

    # 답변 텍스트를 정리하고 포맷팅하는 메서드 (Quill 에디터용)
    def clean_answer_text(self, text: str) -> str:
        # 1단계: 빈 텍스트 검사
        if not text:
            return ""
        
        # 2단계: 제어 문자만 선별 제거 (HTML 태그 보존)
        text = re.sub(r'[\b\r\f\v]', '', text)  # 백스페이스, 캐리지 리턴, 폼 피드, 세로 탭
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)  # ASCII 제어 문자

        # 3단계: 마크다운 스타일 제거 (Quill 에디터 호환성)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **굵게** → 굵게
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *기울임* → 기울임
        
        # 4단계: HTML 태그 내부 공백만 정리 (태그 자체는 유지)
        text = re.sub(r'>\s+<', '><', text)      # 태그 사이 공백 제거
        text = re.sub(r'<p>\s+', '<p>', text)    # <p> 태그 내부 앞 공백 제거
        text = re.sub(r'\s+</p>', '</p>', text)  # </p> 태그 앞 공백 제거
        
        # 5단계: 구 앱 이름 제거 (브랜드 통일)
        text = self.remove_old_app_name(text)
        
        return text

    # 텍스트에서 핵심 키워드 추출 (검색 최적화용)
    def extract_keywords(self, text: str) -> list:
        # 1단계: 한국어 불용어 정의 (조사, 어미 등)
        stop_words = {'는', '은', '이', '가', '을', '를', '에', '에서', '로', '으로', '와', '과', '의', '도', '만', '까지', '부터', '께서', '에게', '한테', '로부터', '으로부터'}
        
        # 2단계: 정규식으로 의미있는 단어 추출 (한글, 영어, 숫자)
        words = re.findall(r'[가-힣a-zA-Z0-9]+', text)
        
        # 3단계: 불용어 제거 및 길이 필터링 (2글자 이상)
        keywords = [word for word in words if len(word) >= 2 and word not in stop_words]
        
        return keywords

    # 텍스트에서 핵심 개념을 추출 (의미 분석용)
    def extract_key_concepts(self, text: str) -> list:
        # 1단계: 한글 명사 추출 (2글자 이상)
        korean_nouns = re.findall(r'[가-힣]{2,}', text)
        
        # 2단계: 영어 단어 추출 (3글자 이상)
        english_words = re.findall(r'[a-zA-Z]{3,}', text)
        
        # 3단계: 모든 단어 통합 및 정제
        concepts = []
        for word in korean_nouns + english_words:
            word = word.lower().strip()  # 소문자 변환 및 공백 제거
            # 4단계: 일반적인 질문 표현 제외
            if len(word) >= 2 and word not in ['있나요', '해주세요', '도와주세요', '문의', '질문']:
                concepts.append(word)
        
        # 5단계: 중복 제거하여 유니크한 개념 반환
        return list(set(concepts))

    # 텍스트에서 성경 번역본명을 추출 (성경 앱 특화)
    def extract_translations_from_text(self, text: str) -> list:
        # 1단계: 성경 번역본 패턴 정의 (영어 + 한국어)
        translation_patterns = [
            r'NIV',                # New International Version
            r'KJV',                # King James Version
            r'ESV',                # English Standard Version
            r'개역개정',            # 개역개정판
            r'개역한글',            # 개역한글판
            r'개역\s*개정',        # 개역 개정 (공백 허용)
            r'개역\s*한글',        # 개역 한글 (공백 허용)
            r'영어\s*번역본',      # 영어 번역본
            r'영문\s*성경',        # 영문 성경
            r'한글\s*번역본',      # 한글 번역본
            r'한국어\s*성경'       # 한국어 성경
        ]
        
        # 2단계: 각 패턴으로 텍스트에서 매칭되는 번역본명 찾기
        found_translations = []
        for pattern in translation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)  # 대소문자 무시
            found_translations.extend(matches)
        
        # 3단계: 중복 제거 및 정규화 (공백 제거 및 통일)
        normalized = []
        for trans in found_translations:
            trans = re.sub(r'\s+', '', trans)  # 공백 제거로 정규화
            if trans not in normalized:
                normalized.append(trans)
        
        return normalized
