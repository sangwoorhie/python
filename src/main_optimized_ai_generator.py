#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
최적화된 AI 답변 생성 메인 클래스
캐싱, 배치 처리, 지능형 API 관리를 통합한 고성능 AI 시스템
"""

import re
import logging
import time
from memory_profiler import profile
from typing import Dict, List, Optional
import numpy as np
from langdetect import detect, LangDetectException
import json

# 기존 모듈들
from src.utils.text_preprocessor import TextPreprocessor
from src.utils.memory_manager import memory_cleanup
from src.models.answer_generator import AnswerGenerator
from src.services.quality_validator import QualityValidator
from src.services.sync_service import SyncService
from src.utils.unified_text_analyzer import UnifiedTextAnalyzer

# 최적화 모듈들
from src.utils.cache_manager import CacheManager
from src.utils.batch_processor import BatchProcessor
from src.utils.intelligent_api_manager import (
    IntelligentAPIManager, APICallRequest, APICallStrategy
)
from src.services.optimized_search_service import OptimizedSearchService


class OptimizedAIAnswerGenerator:
    """최적화된 AI 답변 생성 클래스 - 기존 인터페이스 완전 호환"""

    def __init__(self, pinecone_index, openai_client, connection_string=None, 
                 category_mapping=None, redis_config=None):
        # 1단계. 기본 컴포넌트 초기화
        self.index = pinecone_index
        self.openai_client = openai_client
        
        # 2단계. 유틸리티 컴포넌트 생성
        self.text_processor = TextPreprocessor()
        
        # 3단계. 최적화 시스템 초기화
        self._initialize_optimization_system(redis_config) # ← REDIS_CONFIG 사용
        
        # 4단계: AI 모델 컴포넌트 초기화
        self.unified_analyzer = UnifiedTextAnalyzer(openai_client)
        self.answer_generator = AnswerGenerator(openai_client)
        
        # 5단계: 서비스 컴포넌트 초기화 (최적화 적용)
        self.search_service = OptimizedSearchService(pinecone_index, self.api_manager)
        self.quality_validator = QualityValidator(openai_client)
        
        # 6단계: 동기화 서비스 초기화 (조건부)
        if connection_string and category_mapping:
            self.sync_service = SyncService(
                pinecone_index, 
                openai_client, 
                connection_string, 
                category_mapping
                )
        
        # 7단계: 성능 모니터링 초기화
        self.performance_stats = {
            'total_requests': 0, # 총 요청 수
            'cache_hit_rate': 0.0, # 캐시 히트율
            'avg_processing_time': 0.0, # 평균 처리 시간
            'api_calls_saved': 0 # 절약된 API 호출 수
        }
        
        logging.info("최적화된 AI 답변 생성기 초기화 완료")

    def _initialize_optimization_system(self, redis_config: Optional[Dict]):
        """최적화 시스템 초기화"""
        # Redis 설정
        if redis_config:
            self.cache_manager = CacheManager(
                redis_host=redis_config.get('host', 'localhost'),
                redis_port=redis_config.get('port', 6379),
                redis_db=redis_config.get('db', 0),
                redis_password=redis_config.get('password')
            )
        else:
            # 기본 설정 (로컬 Redis 또는 메모리 캐시)
            self.cache_manager = CacheManager()
        
        # 배치 프로세서 초기화
        self.batch_processor = BatchProcessor(
            max_workers=5,
            batch_size=10,
            batch_timeout=2.0
        )
        
        # 지능형 API 관리자 초기화
        self.api_manager = IntelligentAPIManager(
            cache_manager=self.cache_manager,
            batch_processor=self.batch_processor,
            openai_client=self.openai_client
        )
        
        # 배치 프로세서 시작
        self.batch_processor.start()
        
        logging.info("최적화 시스템 초기화 완료")

    # ================================
    # 기존 인터페이스 호환성 메서드들 (최적화 적용)
    # ================================

    def detect_language(self, text: str) -> str:
        """언어 감지 (langdetect 기반 - 정확한 언어 패턴 분석)"""
        try:
            # ===== 1단계: langdetect 라이브러리를 사용한 자동 언어 감지 =====
            detected = detect(text)
            
            # ===== 2단계: 지원 언어 검증 (한국어/영어만 지원) =====
            if detected == 'ko':
                return 'ko'                                   # 한국어로 감지됨
            elif detected == 'en':
                return 'en'                                   # 영어로 감지됨
            else:
                # 기타 언어는 기본값(한국어)으로 처리
                return 'ko'
                
        except LangDetectException as e:
            logging.warning(f"langdetect 언어 감지 실패: {e}, 폴백 로직 사용")
            
            # ===== 3단계: 감지 실패시 개선된 문자 비율 기반 폴백 로직 =====
            # 기본 문자 카운트
            korean_chars = len(re.findall(r'[가-힣]', text))
            english_chars = len(re.findall(r'[a-zA-Z]', text))
            
            # 한국어 문법 패턴 가중치 (조사, 어미 등)
            korean_particles = len(re.findall(r'[을를이가에서로과와의도만까지부터께서에게한테]', text))
            korean_endings = len(re.findall(r'습니다|세요|어요|겠어요|았어요|었어요|하게|주세요', text))
            
            # 가중치 적용한 점수 계산
            korean_score = korean_chars + (korean_particles * 2) + (korean_endings * 3)
            english_score = english_chars
            
            # 문자 수 비교로 언어 판단 (개선된 버전)
            if korean_score > english_score:
                return 'ko'                                   # 한국어로 판단
            else:
                return 'en'                                   # 영어로 판단

    def preprocess_text(self, text: str) -> str:
        """텍스트 전처리 (기존 호환)"""
        return self.text_processor.preprocess_text(text)

    def create_embedding(self, text: str):
        """임베딩 생성 (캐싱 적용)"""
        request = APICallRequest(
            operation='embedding',
            data={'text': text},
            priority=3,
            strategy=APICallStrategy.CACHE_FIRST
        )
        
        response = self.api_manager.process_request(request)
        return response.data if response.success else None

    # analyze_question_intent 메서드 제거됨 - unified_analyzer.analyze_and_correct()로 통합
    # 영어 질문의 경우에만 개별 호출이 필요하므로 기본값 반환 메서드로 대체
    def _get_default_intent_analysis(self, query: str) -> dict:
        """기본 의도 분석 결과 반환 (영어 질문용)"""
        return {
            "core_intent": "general_inquiry",
            "intent_category": "일반문의",
            "primary_action": "기타",
            "target_object": "기타",
            "constraint_conditions": [],
            "standardized_query": query,
            "semantic_keywords": [query[:20]],
            # 기존 호환성 필드
            "intent_type": "일반문의",
            "main_topic": "기타",
            "specific_request": query[:100],
            "keywords": [query[:20]],
            "urgency": "medium",
            "action_type": "기타"
        }

    def search_similar_answers(self, query: str, top_k: int = 5, similarity_threshold: float = 0.7, lang: str = 'ko') -> list:
        """유사 답변 검색 (최적화 적용)"""
        return self.search_service.search_similar_answers_optimized(query, top_k, lang)
    
    def search_similar_answers_with_cached_intent(self, query: str, cached_intent: Dict, top_k: int = 5, lang: str = 'ko') -> list:
        """캐시된 의도 분석을 활용한 유사 답변 검색 (API 호출 절약)"""
        return self.search_service.search_similar_answers_with_cached_intent(query, cached_intent, top_k, lang)

    def analyze_context_quality(self, similar_answers: list, query: str) -> dict:
        """컨텍스트 품질 분석 (기존 호환)"""
        return self.search_service.analyze_context_quality(similar_answers, query)

    def get_best_fallback_answer(self, similar_answers: list, lang: str = 'ko') -> str:
        """최적 폴백 답변 선택 (최적화 적용)"""
        return self.search_service.get_best_fallback_answer(similar_answers, lang)

    def generate_with_enhanced_gpt(self, query: str, similar_answers: list, context_analysis: dict, lang: str = 'ko') -> str:
        """향상된 GPT 생성 (기존 호환)"""
        return self.answer_generator.generate_with_enhanced_gpt(query, similar_answers, context_analysis, lang)

    def is_valid_text(self, text: str, lang: str = 'ko') -> bool:
        """텍스트 유효성 검증 (기존 호환)"""
        return self.quality_validator.is_valid_text(text, lang)

    def check_answer_completeness(self, answer: str, query: str, lang: str = 'ko') -> float:
        """답변 완성도 검증 (기존 호환)"""
        return self.quality_validator.check_answer_completeness(answer, query, lang)

    def detect_empty_promises(self, answer: str, lang: str = 'ko') -> float:
        """빈 약속 패턴 감지 (기존 호환)"""
        return self.quality_validator.detect_empty_promises(answer, lang)

    def detect_hallucination_and_inconsistency(self, answer: str, query: str, lang: str = 'ko') -> dict:
        """할루시네이션 및 일관성 검증 (기존 호환)"""
        return self.quality_validator.detect_hallucination_and_inconsistency(answer, query, lang)

    def validate_answer_relevance_ai(self, answer: str, query: str, question_analysis: dict) -> bool:
        """AI 기반 답변 관련성 검증 (기존 호환)"""
        return self.quality_validator.validate_answer_relevance_ai(answer, query, question_analysis)

    # fix_korean_typos_with_ai 메서드 제거됨 - unified_analyzer.analyze_and_correct()로 통합

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """번역 (캐싱 적용)"""
        request = APICallRequest(
            operation='translation',
            data={'text': text, 'source_lang': source_lang, 'target_lang': target_lang},
            priority=4,
            strategy=APICallStrategy.CACHE_FIRST
        )
        
        response = self.api_manager.process_request(request)
        return response.data if response.success else text

    def remove_greeting_and_closing(self, text: str, lang: str = 'ko') -> str:
        """인사말/끝맺음말 제거 (기존 호환)"""
        return self.answer_generator.remove_greeting_and_closing(text, lang)

    def format_answer_with_html_paragraphs(self, text: str, lang: str = 'ko') -> str:
        """HTML 단락 포맷팅 (기존 호환)"""
        return self._format_answer_with_html_paragraphs(text, lang)

    def _format_answer_with_html_paragraphs(self, text: str, lang: str = 'ko') -> str:
        """답변 텍스트를 HTML 단락 형식으로 포맷팅하는 메서드"""
        if not text:
            return ""

        text = self.text_processor.remove_old_app_name(text)

        # 문장을 마침표, 느낌표, 물음표로 분리
        sentences = re.split(r'(?<=[.!?])\s+', text)

        paragraphs = []
        current_paragraph = []

        # 단락 분리 트리거 키워드들
        if lang == 'ko':
            paragraph_triggers = [
                '안녕하세요', '감사합니다', '감사드립니다', '바이블 애플을',
                '따라서', '그러므로', '또한', '그리고', '또는', '하지만', '그런데',
                '현재', '지금', '만약', '혹시', '성도님', '고객님',
                '기능', '스피커', '버튼', '메뉴', '화면', '설정'
            ]
        else:  # 영어
            paragraph_triggers = [
                'Hello', 'Thank', 'Therefore', 'However', 'Additionally',
                'Currently', 'If', 'Please', 'Feature', 'Function'
            ]

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue

            # 첫 번째 문장은 항상 별도 단락
            if i == 0:
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                paragraphs.append(sentence)
                continue

            should_break = False

            # 트리거 키워드로 시작하는 문장은 새 단락
            for trigger in paragraph_triggers:
                if sentence.startswith(trigger):
                    should_break = True
                    break

            # 현재 단락에 2개 이상 문장이 있으면 새 단락
            if current_paragraph and len(current_paragraph) >= 2:
                should_break = True

            # 새 단락 분리
            if should_break and current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = [sentence]
            else:
                current_paragraph.append(sentence)

        # 마지막 단락 추가
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))

        # HTML 단락으로 변환
        html_paragraphs = []
        for i, paragraph in enumerate(paragraphs):
            html_paragraphs.append(f"<p>{paragraph}</p>")

            # 단락 사이에 빈 줄 추가 (마지막 단락 제외)
            if i < len(paragraphs) - 1:
                html_paragraphs.append("<p><br></p>")

        return ''.join(html_paragraphs)

    # ================================
    # 메인 비즈니스 로직 (최적화 적용)
    # ================================

    @profile
    def generate_ai_answer(self, query: str, similar_answers: list, lang: str) -> str:
        """최적화된 AI 답변 생성"""
        
        # 1. 언어 감지 (빠른 룰 기반)
        if not lang or lang == 'auto':
            detected_lang = self.detect_language(query)
            lang = detected_lang
            logging.info(f"감지된 언어: {lang}")

        # 2. 유사 답변이 없는 경우
        if not similar_answers:
            logging.error("유사 답변이 전혀 없음")
            if lang == 'en':
                default_msg = "<p>We need more detailed information to provide an accurate answer to your inquiry.</p><p><br></p><p>Please contact our customer service center for prompt assistance.</p>"
            else:
                default_msg = "<p>안녕하세요, GOODTV 바이블 애플입니다.</p><p><br></p><p>바이블 애플을 이용해 주셔서 진심으로 감사드립니다.</p><p><br></p><p>남겨주신 문의는 현재 담당자가 직접 확인하고 있습니다.</p><p><br></p><p>성도님께 도움이 될 수 있도록 내용을 꼼꼼히 살펴보고 정확하고 구체적인 답변을 준비하겠습니다.</p><p><br></p><p>답변은 최대 하루 이내에 드릴 예정이오니 조금만 기다려 주시면 감사하겠습니다.</p><p><br></p><p>항상 주님 안에서 평안하세요, 감사합니다.</p>"
            return default_msg

        # 3. 컨텍스트 분석
        context_analysis = self.analyze_context_quality(similar_answers, query)

        try:
            approach = context_analysis['recommended_approach']
            logging.info(f"선택된 접근 방식: {approach}, 언어: {lang}")

            base_answer = ""

            if approach == 'direct_use':
                base_answer = self.get_best_fallback_answer(similar_answers[:3], lang)

            elif approach in ['gpt_with_strong_context', 'gpt_with_weak_context']:
                base_answer = self.generate_with_enhanced_gpt(query, similar_answers, context_analysis, lang)

                if not base_answer or not self.is_valid_text(base_answer, lang):
                    logging.warning("GPT 생성 실패, 폴백 답변 사용")
                    base_answer = self.get_best_fallback_answer(similar_answers, lang)

            else:
                base_answer = self.get_best_fallback_answer(similar_answers, lang)

            # 최종 검증 및 품질 향상
            if not base_answer:
                if lang == 'en':
                    return "<p>We need more detailed information to provide an accurate answer to your inquiry.</p><p><br></p><p>Please contact our customer service center for prompt assistance.</p>"
                else:
                    return "<p>안녕하세요, GOODTV 바이블 애플입니다.</p><p><br></p><p>바이블 애플을 이용해 주셔서 진심으로 감사드립니다.</p><p><br></p><p>남겨주신 문의는 현재 담당자가 직접 확인하고 있습니다.</p><p><br></p><p>성도님께 도움이 될 수 있도록 내용을 꼼꼼히 살펴보고 정확하고 구체적인 답변을 준비하겠습니다.</p><p><br></p><p>답변은 최대 하루 이내에 드릴 예정이오니 조금만 기다려 주시면 감사하겠습니다.</p><p><br></p><p>항상 주님 안에서 평안하세요, 감사합니다.</p>"

            # 자가 평가 로직 추가
            if base_answer:
                coherence_score = self._evaluate_semantic_coherence(query, base_answer)
                
                # 임계값을 0.3으로 낮추고, 재생성 실패시 원본 유지
                if coherence_score < 0.3:  # 0.5 → 0.3으로 완화
                    logging.info(f"낮은 일관성 점수 ({coherence_score:.2f}), 답변 재생성 시도")
                    
                    # 원본 답변 백업
                    original_answer = base_answer
                    
                    # 관련성 낮은 답변 필터링 후 재생성
                    filtered_answers = self._filter_by_coherence(query, similar_answers)
                    if filtered_answers and len(filtered_answers) >= 2:
                        new_answer = self.generate_with_enhanced_gpt(
                            query, filtered_answers[:3], context_analysis, lang
                        )
                        
                        # 재생성된 답변이 유효한 경우에만 사용
                        if new_answer and len(new_answer) > 50:  # 최소 길이 체크
                            base_answer = new_answer
                            logging.info(f"재생성 성공, 필터링된 답변 {len(filtered_answers)}개 사용")
                        else:
                            # 재생성 실패시 원본 유지
                            base_answer = original_answer
                            logging.warning("재생성 실패, 원본 답변 유지")
                    else:
                        logging.warning(f"필터링된 답변 부족 ({len(filtered_answers)}개), 원본 유지")

            # 강화된 답변 완성도 검증 및 재생성 로직
            base_completeness = self.check_answer_completeness(base_answer, query, lang)
            empty_promise_score = self.detect_empty_promises(base_answer, lang)
            final_hallucination_check = self.detect_hallucination_and_inconsistency(base_answer, query, lang)
            final_hallucination_score = final_hallucination_check['overall_score']

            # 할루시네이션이 치명적이면 즉시 폴백으로 변경
            if final_hallucination_score < 0.3:
                logging.error("최종 답변에서 치명적 할루시네이션 감지! 폴백 답변으로 강제 변경")
                base_answer = self.get_best_fallback_answer(similar_answers, lang)

            # 재생성 조건 검사
            needs_regeneration = (
                base_completeness < 0.6 or
                empty_promise_score < 0.3 or
                final_hallucination_score < 0.5
            )

            if needs_regeneration and approach in ['gpt_with_strong_context', 'gpt_with_weak_context']:
                # 재생성 시도 (최대 2회)
                for attempt in range(2):
                    retry_analysis = context_analysis.copy()
                    retry_analysis['recommended_approach'] = 'gpt_with_strong_context'

                    retry_answer = self.generate_with_enhanced_gpt(query, similar_answers, retry_analysis, lang)
                    if retry_answer:
                        retry_completeness = self.check_answer_completeness(retry_answer, query, lang)
                        retry_empty_promise = self.detect_empty_promises(retry_answer, lang)
                        retry_hallucination_check = self.detect_hallucination_and_inconsistency(retry_answer, query, lang)
                        retry_hallucination_score = retry_hallucination_check['overall_score']

                        # 재생성 답변에 치명적 할루시네이션이 있으면 사용하지 않음
                        if retry_hallucination_score < 0.3:
                            continue

                        # 재생성 답변이 더 나은지 확인
                        is_better = (
                            retry_completeness > base_completeness and
                            retry_empty_promise > empty_promise_score and
                            retry_hallucination_score > final_hallucination_score
                        )

                        if is_better:
                            base_answer = retry_answer
                            break

            # 언어별 포맷팅
            if lang == 'en':
                # 영어 답변 포맷팅
                base_answer = self.text_processor.remove_old_app_name(base_answer)

                # 기존 인사말/끝맺음말 제거
                base_answer = re.sub(r'^Hello[^.]*\.\s*', '', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'^This is GOODTV Bible App[^.]*\.\s*', '', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'\s*Thank you[^.]*\.?\s*$', '', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'\s*Best regards[^.]*\.?\s*$', '', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'\s*God bless[^.]*\.?\s*$', '', base_answer, flags=re.IGNORECASE)

                formatted_body = self.format_answer_with_html_paragraphs(base_answer.strip(), 'en')

                final_answer = "<p>Hello, this is GOODTV Bible Apple App customer service team.</p><p><br></p><p>Thank you very much for using our app and for taking the time to contact us.</p><p><br></p>"
                final_answer += formatted_body
                final_answer += "<p><br></p><p>Thank you once again for sharing your thoughts with us!</p><p><br></p><p>May God's peace and grace always be with you.</p>"

            else:  # 한국어
                # 한국어 답변 최종 포맷팅
                base_answer = self.text_processor.remove_old_app_name(base_answer)
                base_answer = re.sub(r'고객님', '성도님', base_answer)

                # 기존 인사말/끝맺음말 제거
                base_answer = re.sub(r'^안녕하세요[^.]*\.\s*', '', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'^GOODTV\s+바이블\s*애플[^.]*\.\s*', '', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'^바이블\s*애플[^.]*\.\s*', '', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'고객센터[^.]*\.\s*', '', base_answer, flags=re.IGNORECASE)

                # 끝맺음말 제거
                base_answer = re.sub(r'\s*감사합니다[^.]*\.?\s*$', '', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'\s*평안하세요[^.]*\.?\s*$', '', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'\s*주님\s*안에서[^.]*\.?\s*$', '', base_answer, flags=re.IGNORECASE)

                # 구 앱 이름을 바이블 애플로 완전 교체
                base_answer = re.sub(r'바이블\s*애플\s*\(구\)\s*다번역\s*성경\s*찬송', '바이블 애플', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'다번역\s*성경\s*찬송', '바이블 애플', base_answer, flags=re.IGNORECASE)

                # 중복 끝맺음말 제거
                base_answer = re.sub(r'항상\s*성도님들?께\s*좋은\s*(서비스|성경앱)을?\s*제공하기\s*위해\s*노력하는\s*바이블\s*애플이\s*되겠습니다\.?\s*', '', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'\s*항상\s*$', '', base_answer, flags=re.IGNORECASE)

                # 본문을 HTML 단락 형식으로 포맷팅
                formatted_body = self.format_answer_with_html_paragraphs(base_answer.strip(), 'ko')

                # 한국어 고정 인사말 및 끝맺음말
                final_answer = "<p>안녕하세요. GOODTV 바이블 애플입니다.</p><p><br></p><p>바이블 애플을 이용해주셔서 감사드립니다.</p><p><br></p>"

                # HTML 포맷팅 후 완전한 정리 작업
                formatted_body = re.sub(r'<p>\s*항상\s*성도님들?께\s*좋은\s*(서비스|성경앱)을?\s*제공하기\s*위해\s*노력하는\s*바이블\s*애플이\s*되겠습니다\.?\s*</p>', '', formatted_body, flags=re.IGNORECASE)
                formatted_body = re.sub(r'<p>\s*감사합니다\.?\s*(주님\s*안에서\s*평안하세요\.?)?\s*</p>', '', formatted_body, flags=re.IGNORECASE)
                formatted_body = re.sub(r'<p>\s*항상\s*</p>', '', formatted_body, flags=re.IGNORECASE)
                formatted_body = re.sub(r'(<p><br></p>\s*){3,}', '<p><br></p><p><br></p>', formatted_body)
                formatted_body = re.sub(r'(<p><br></p>\s*)+$', '', formatted_body)

                final_answer += formatted_body
                final_answer += "<p><br></p><p>항상 성도님께 좋은 성경앱을 제공하기 위해 노력하는 바이블 애플이 되겠습니다.</p><p><br></p><p>감사합니다. 주님 안에서 평안하세요.</p>"

            logging.info(f"최종 답변 생성 완료: {len(final_answer)}자, 접근방식: {approach}, 언어: {lang}")
            return final_answer

        except Exception as e:
            logging.error(f"답변 생성 중 오류: {e}")
            if lang == 'en':
                return "<p>Sorry, we cannot generate an answer at this moment.</p><p><br></p><p>Please contact our customer service center.</p>"
            else:
                return "<p>안녕하세요, GOODTV 바이블 애플입니다.</p><p><br></p><p>바이블 애플을 이용해 주셔서 진심으로 감사드립니다.</p><p><br></p><p>남겨주신 문의는 현재 담당자가 직접 확인하고 있습니다.</p><p><br></p><p>성도님께 도움이 될 수 있도록 내용을 꼼꼼히 살펴보고 정확하고 구체적인 답변을 준비하겠습니다.</p><p><br></p><p>답변은 최대 하루 이내에 드릴 예정이오니 조금만 기다려 주시면 감사하겠습니다.</p><p><br></p><p>항상 주님 안에서 평안하세요, 감사합니다.</p>"

    # ☆ 2. 기본 전처리 메서드
    def process(self, seq: int, question: str, lang: str) -> dict:
        """최적화된 메인 처리 메서드"""
        start_time = time.time() # 현재 시간을 타임스탬프로 기록하는 코드
        
        try:
            with memory_cleanup():
                # 성능 통계 업데이트
                self.performance_stats['total_requests'] += 1
                
                # 1단계. 전처리
                processed_question = self.preprocess_text(question)

                # 2단계. 통합 분석 (오타 수정 + 의도 분석) - API 호출 1회로 통합
                if lang == 'ko' or lang == 'auto':
                    corrected_text, intent_analysis = self.unified_analyzer.analyze_and_correct(processed_question)
                    processed_question = corrected_text
                    
                    print("="*80)
                    print("🔍 [통합 분석 결과]")
                    print(f"원본 질문: {processed_question}")
                    print(f"수정된 질문: {corrected_text}")
                    print(f"의도 분석 JSON: {json.dumps(intent_analysis, ensure_ascii=False, indent=2)}")
                    print("="*80)
                    
                    # 로그 파일에도 기록
                    logging.info(f"통합 분석 - 원본: {processed_question}")
                    logging.info(f"통합 분석 - 수정: {corrected_text}")
                    logging.info(f"통합 분석 - 의도: {json.dumps(intent_analysis, ensure_ascii=False)}")

                    # 의도 분석 결과를 임시 저장 (검색 단계에서 재사용)
                    self._cached_intent_analysis = intent_analysis
                    if processed_question != question:
                        logging.info(f"통합 분석 - 오타 수정: {question[:50]} → {processed_question[:50]}")
                        logging.info(f"통합 분석 - 의도: {intent_analysis.get('core_intent', 'N/A')}")
                else:
                    # 영어인 경우 기본 의도 분석 사용 (GPT 호출 생략)
                    self._cached_intent_analysis = self._get_default_intent_analysis(processed_question)

                if not processed_question:
                    return {"success": False, "error": "질문이 비어있습니다."}

                # 언어 자동 감지
                if not lang or lang == 'auto':
                    lang = self.detect_language(processed_question)
                    logging.info(f"자동 감지된 언어: {lang}")

                logging.info(f"처리 시작 - SEQ: {seq}, 언어: {lang}, 질문: {processed_question[:50]}...")

                # 3단계. 유사 답변 검색 (캐시된 의도 분석 활용)
                similar_answers = self.search_similar_answers_with_cached_intent(
                    processed_question, self._cached_intent_analysis, lang=lang)

                # AI 답변 생성
                ai_answer = self.generate_ai_answer(processed_question, similar_answers, lang)

                # 특수문자 정리
                ai_answer = ai_answer.replace('"', '"').replace('"', '"')
                ai_answer = ai_answer.replace(''', "'").replace(''', "'")

                # 성능 통계 업데이트
                processing_time = time.time() - start_time
                self._update_performance_stats(processing_time)

                result = {
                    "success": True,
                    "answer": ai_answer,
                    "similar_count": len(similar_answers),
                    "embedding_model": "text-embedding-3-small",
                    "generation_model": "gpt-5-mini",
                    "detected_language": lang,
                    "processing_time": processing_time,
                    "optimization_stats": self.get_optimization_summary()
                }

                logging.info(f"처리 완료 - SEQ: {seq}, 언어: {lang}, 시간: {processing_time:.2f}s")
                return result

        except Exception as e:
            logging.error(f"처리 중 오류 - SEQ: {seq}, 오류: {str(e)}")
            return {"success": False, "error": str(e)}

    def _evaluate_semantic_coherence(self, query: str, answer: str) -> float:
        """의미적 일관성 평가"""
        try:
            query_embedding = self.create_embedding(query)
            
            # HTML 태그 제거 후 임베딩 생성
            clean_answer = self.text_processor.preprocess_text(answer)
            answer_embedding = self.create_embedding(clean_answer)
            
            if query_embedding and answer_embedding:
                # numpy 배열로 변환
                q_vec = np.array(query_embedding)
                a_vec = np.array(answer_embedding)
                
                # 코사인 유사도 계산
                similarity = np.dot(q_vec, a_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(a_vec))
                return float(similarity)
        except Exception as e:
            logging.error(f"일관성 평가 실패: {e}")
        
        return 0.5  # 기본값

    def _filter_by_coherence(self, query: str, similar_answers: list) -> list:
        """일관성 기반 필터링 - 더 관대한 버전"""
        try:
            query_embedding = self.create_embedding(query)
            if not query_embedding:
                return similar_answers
            
            q_vec = np.array(query_embedding)
            filtered = []
            
            for answer in similar_answers:
                try:
                    answer_embedding = self.create_embedding(answer['question'])
                    if answer_embedding:
                        a_vec = np.array(answer_embedding)
                        similarity = np.dot(q_vec, a_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(a_vec))
                        
                        # 임계값을 0.4로 낮춤 (0.6 → 0.4)
                        if similarity > 0.4:
                            filtered.append(answer)
                except Exception as e:
                    logging.error(f"답변 필터링 중 오류: {e}")
                    continue
            
            # 필터링된 답변이 너무 적으면 상위 답변 추가
            if len(filtered) < 3:
                # 원본 답변 중 필터링되지 않은 것들 추가
                for answer in similar_answers:
                    if answer not in filtered:
                        filtered.append(answer)
                    if len(filtered) >= 5:  # 최대 5개까지
                        break
            
            logging.info(f"일관성 필터링: {len(similar_answers)}개 → {len(filtered)}개")
            return filtered
        
        except Exception as e:
            logging.error(f"일관성 필터링 실패: {e}")
            return similar_answers

    def _update_performance_stats(self, processing_time: float):
        """성능 통계 업데이트"""
        api_stats = self.api_manager.get_performance_stats()
        
        self.performance_stats['cache_hit_rate'] = api_stats['cache_hit_rate']
        self.performance_stats['api_calls_saved'] = api_stats['api_calls_saved']
        
        # 평균 처리 시간 계산
        total_requests = self.performance_stats['total_requests']
        current_avg = self.performance_stats['avg_processing_time']
        self.performance_stats['avg_processing_time'] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
        )

    def get_optimization_summary(self) -> Dict:
        """최적화 요약 정보"""
        api_stats = self.api_manager.get_performance_stats()
        search_stats = self.search_service.get_optimization_stats()
        
        return {
            'cache_hit_rate': api_stats['cache_hit_rate'],
            'api_calls_saved': api_stats['api_calls_saved'],
            'batch_processed': api_stats['batch_processed'],
            'avg_processing_time': self.performance_stats['avg_processing_time'],
            'embedding_cache_size': search_stats['embedding_cache_size']
        }

    def get_detailed_performance_stats(self) -> Dict:
        """상세 성능 통계"""
        return {
            'performance_stats': self.performance_stats,
            'api_manager_stats': self.api_manager.get_performance_stats(),
            'search_stats': self.search_service.get_optimization_stats(),
            'cache_stats': self.cache_manager.get_cache_stats(),
            'batch_stats': self.batch_processor.get_stats(),
            'system_health': self.api_manager.health_check()
        }

    def optimize_for_production(self):
        """프로덕션 환경 최적화 설정"""
        # API 관리자 최적화
        self.api_manager.optimize_settings(
            enable_smart_caching=True, # 캐싱 활성화
            enable_batch_processing=True, # 배치 처리 활성화
            min_batch_size=3, # 최소 배치 크기
            max_wait_time=1.5, # 최대 대기 시간
            cache_hit_bonus=0.9 # 캐시 히트 보너스
        )
        
        # 검색 서비스 최적화
        self.search_service.update_search_config(
            adaptive_layer_count=True, # 동적 레이어 카운트 활성화
            early_termination=True, # 조기 종료 활성화
            similarity_threshold=0.8, # 유사도 임계값
            enable_result_caching=True # 결과 캐싱 활성화
        )
        
        logging.info("프로덕션 최적화 설정 적용 완료")

    def cleanup(self):
        """리소스 정리"""
        try:
            if hasattr(self, 'batch_processor'):
                self.batch_processor.stop()
            
            if hasattr(self, 'search_service'):
                self.search_service.clear_caches()
            
            logging.info("최적화된 AI 생성기 리소스 정리 완료")
        except Exception as e:
            logging.error(f"리소스 정리 중 오류: {e}")

    def __del__(self):
        """소멸자"""
        self.cleanup()
