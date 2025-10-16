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
from src.services.ai_answer_generator import AIAnswerGenerator

# 최적화 모듈들
from src.utils.cache_manager import CacheManager
from src.utils.batch_processor import BatchProcessor
from src.utils.intelligent_api_manager import (
    IntelligentAPIManager, APICallRequest, APICallStrategy
)
# from src.services.optimized_search_service import OptimizedSearchService
from src.services.enhanced_search_service import EnhancedPineconeSearchService

class OptimizedAIAnswerGenerator:
    """최적화된 AI 답변 생성 클래스 - 기존 인터페이스 완전 호환"""

    def __init__(self, 
                 pinecone_index, # Pinecone Index
                 openai_client, # OpenAI Client
                 connection_string, # Connection String
                 category_mapping, # Category Mapping
                 redis_config # Redis Config
                 ):
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
        self.ai_answer_generator = AIAnswerGenerator(openai_client)
        
        # 5단계: 서비스 컴포넌트 초기화 (최적화 적용)
        # self.search_service = OptimizedSearchService(pinecone_index, self.api_manager)
        self.quality_validator = QualityValidator(openai_client)
        
        # Enhanced Search Service 초기화 (기존 코드에 추가)
        self.search_service = EnhancedPineconeSearchService(
            openai_client=openai_client,
            pinecone_index=pinecone_index
        )
        logging.info("Enhanced Pinecone Search Service 초기화 완료")

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
        
        # logging.info("최적화된 AI 답변 생성기 초기화 완료")

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
        
        logging.info("최적화 시스템 초기화 완료")                               # 영어로 판단

    def preprocess_text(self, text: str) -> str:
        """텍스트 전처리 (기존 호환)"""
        return self.text_processor.preprocess_text(text)

    def _get_default_intent_analysis(self, query: str) -> dict:
        """기본 의도 분석 결과 반환 (영어 질문용)"""
        return {
            "core_intent": "general_inquiry",
            "intent_category": "일반문의",
            "primary_action": "기타",
            "semantic_keywords": [query[:20]],
        }

    def search_similar_answers(self, query: str, top_k: int = 5, similarity_threshold: float = 0.7, lang: str = 'ko') -> list:
        """유사 답변 검색 (최적화 적용)"""
        return self.search_service.search_similar_answers_optimized(query, top_k, lang)
    
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
    # ☆ 1. 기본 전처리 메서드
    def process(self, seq: int, question: str, lang: str) -> dict:
        start_time = time.time() # 현재 시간을 타임스탬프로 기록하는 코드
        
        try:
            with memory_cleanup():

                # 성능 통계 업데이트
                self.performance_stats['total_requests'] += 1
                
                # 1단계: POST 요청 수신 로그
                logging.info(f"1. POST /generate_answer: seq={seq}, question='{question}', lang='{lang}'")
                
                # 2단계: 전처리 (HTML 태그 제거, 앱 이름 통일, 공백 정규화)
                preprocess_start = time.time()
                processed_question = self.preprocess_text(question)
                preprocess_time = time.time() - preprocess_start
                logging.info(f"2. HTML 태그 제거, 앱 이름 통일, 공백 정규화 전처리: '{question}' → '{processed_question}', 시간={preprocess_time:.3f}s")

                # 3단계: 통합 분석 (오타 수정 + 의도 분석) API 호출
                analysis_start = time.time()    
                corrected_text, intent_analysis = self.unified_analyzer.analyze_and_correct(processed_question)
                core_intent = intent_analysis.get('core_intent', 'general_inquiry')
                processed_question = corrected_text # corrected_text를 쿼리로 사용
                semantic_keywords = intent_analysis.get('semantic_keywords', [])
                result = ', '.join(semantic_keywords)

                analysis_time = time.time() - analysis_start 
                logging.info(f"3. 통합 분석 완료: corrected='{corrected_text}', intent={{'core_intent': '{intent_analysis.get('core_intent', 'N/A')}'}}, 시간={analysis_time:.2f}s")
                    
                if not processed_question:
                    return {"success": False, "error": "질문이 비어있습니다."}

                # 4단계: 언어 한국어로 고정
                lang = 'ko'
                search_start = time.time()
                
                # 5단계: 의도 기반 검색 (Pinecone 검색) API 호출
                similar_answers = self.search_service.search_by_enhanced_intent(
                    intent_analysis=intent_analysis,  # 전체 의도 분석 결과 전달
                    original_query=corrected_text, # corrected_text를 쿼리로 사용 (원래 이거였음)
                    # original_query=core_intent,
                    # original_query= result,
                    lang=lang,
                    top_k=3  # 상위 3개 결과만 반환
                )
                
                search_time = time.time() - search_start
                logging.info(f"검색 완료: {len(similar_answers)}개 결과, 시간={search_time:.2f}s")

                # 검색 결과 상세 로깅 (디버깅용)
                for i, result in enumerate(similar_answers[:3], 1):
                    logging.info(
                        f"검색결과 상세 #{i}: "
                        f"score={result.get('score', 0):.4f}, "
                        f"category='{result.get('category', '')}', "
                        f"question='{result.get('question', '')}', "
                        f"answer_length={len(result.get('answer', ''))}자"
                    )

                # 검색 결과 확인 - 결과가 없으면 폴백 답변 사용
                if not similar_answers:
                    logging.warning(f"⚠️ 검색 결과 없음 - 폴백 답변 사용")
                    
                    # 폴백 답변 사용
                    generation_start = time.time()
                    ai_answer = self.ai_answer_generator._get_fallback_answer()
                    generation_time = time.time() - generation_start
                    logging.info(f"폴백 답변 생성 완료: 길이={len(ai_answer)}자, 시간={generation_time:.2f}s")
                else:
                    
                    # 6단계: AI 답변 생성 (AIAnswerGenerator 사용)
                    generation_start = time.time()
                    logging.info("6. AI 답변 생성 시작")
                    
                    ai_answer = self.ai_answer_generator.generate_answer(
                        corrected_text=corrected_text,
                        intent_analysis=intent_analysis,
                        similar_answers=similar_answers,
                        lang=lang
                    )
                    
                    generation_time = time.time() - generation_start
                    logging.info(f"AI 답변 생성 완료: 길이={len(ai_answer)}자, 시간={generation_time:.2f}s")

                # 특수문자 정리
                ai_answer = ai_answer.replace('"', '"').replace('"', '"')
                ai_answer = ai_answer.replace(''', "'").replace(''', "'")

                # 성능 통계 업데이트
                total_time = time.time() - start_time
                self._update_performance_stats(total_time)

                result = {
                    "success": True,
                    "answer": ai_answer,
                    "similar_count": len(similar_answers),
                    "embedding_model": "text-embedding-3-small",
                    "generation_model": "gpt-5-mini",
                    "detected_language": lang,
                    "processing_time": total_time,
                    "optimization_stats": self.get_optimization_summary()
                }

                logging.info(f"처리 완료 - SEQ: {seq}, 총 시간: {total_time:.2f}s")
                logging.info(f"답변: {ai_answer}")
                return result

        except Exception as e:
            logging.error(f"처리 중 오류 - SEQ: {seq}, 오류: {str(e)}")
            return {"success": False, "error": str(e)}

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

    def get_optimization_summary(self) -> dict:
        """최적화 통계 요약 반환 (안전한 버전)"""
        try:
            summary = {
                'total_requests': self.performance_stats.get('total_requests', 0),
                'cache_hit_rate': self.performance_stats.get('cache_hit_rate', 0.0),
                'avg_processing_time': self.performance_stats.get('avg_processing_time', 0.0),
                'api_calls_saved': self.performance_stats.get('api_calls_saved', 0)
            }
            
            # search_service의 통계 메서드가 있는 경우에만 호출
            if hasattr(self.search_service, 'get_optimization_stats'):
                summary['search_stats'] = self.search_service.get_optimization_stats()
            elif hasattr(self.search_service, 'get_search_statistics'):
                summary['search_stats'] = self.search_service.get_search_statistics()
            else:
                summary['search_stats'] = {
                    'method': 'enhanced_pinecone_search',
                    'status': 'active'
                }
            
            return summary
            
        except Exception as e:
            logging.warning(f"최적화 통계 생성 실패: {e}")
            return {
                'total_requests': 0,
                'cache_hit_rate': 0.0,
                'avg_processing_time': 0.0,
                'api_calls_saved': 0,
                'error': str(e)
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
        
        # 검색 서비스 최적화 (조건부 호출)
        if hasattr(self.search_service, 'update_search_config'):
            self.search_service.update_search_config(
                adaptive_layer_count=True, # 동적 레이어 카운트 활성화
                early_termination=True, # 조기 종료 활성화
                similarity_threshold=0.8, # 유사도 임계값
                enable_result_caching=True # 결과 캐싱 활성화
            )
            logging.info("검색 서비스 최적화 설정 완료")
        else:
            logging.info("검색 서비스 최적화 건너뛰기 (EnhancedPineconeSearchService에서는 지원되지 않음)")
        
        logging.info("프로덕션 최적화 설정 적용 완료")

    def cleanup(self):
        """리소스 정리"""
        try:
            if hasattr(self, 'batch_processor'):
                self.batch_processor.stop()
                logging.info("배치 처리 시스템 중지됨")
            
            # 검색 서비스 캐시 정리 (조건부 호출)
            if hasattr(self, 'search_service') and hasattr(self.search_service, 'clear_caches'):
                self.search_service.clear_caches()
                logging.info("검색 서비스 캐시 지워짐")
            else:
                logging.info("검색 서비스 캐시 정리 건너뛰기 (메서드 없음)")
                
            logging.info("최적화된 AI 생성기 리소스 정리 완료")
            
        except Exception as e:
            logging.error(f"리소스 정리 중 오류: {str(e)}")

    def __del__(self):
        """소멸자"""
        self.cleanup()
