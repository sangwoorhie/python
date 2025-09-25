#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
지능형 API 호출 관리 시스템
- 캐싱, 배치 처리, 조건부 호출을 통합 관리하여 API 비용과 지연시간 최적화
- 4가지 API 호출 전략 (캐시 우선, 배치 전용, 즉시 호출, 스킵)
- 중복 요청 감지 및 통합, 우선순위 기반 처리
- 실시간 성능 모니터링 및 동적 최적화
"""

import uuid
import time
import logging
import hashlib
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .cache_manager import CacheManager
from .batch_processor import BatchProcessor, BatchRequest, BatchResult


# ===== API 호출 전략 열거형 =====
class APICallStrategy(Enum):
    # API 호출 최적화를 위한 4가지 전략
    CACHE_FIRST = "cache_first"          # 캐시 우선: 캐시 확인 후 없으면 API 호출
    BATCH_ONLY = "batch_only"            # 배치 처리만: 배치 시스템을 통한 효율적 처리
    IMMEDIATE = "immediate"              # 즉시 호출: 바로 API 호출 (높은 우선순위)
    SKIP = "skip"                        # 호출 생략: 불필요한 요청 건너뛰기


# ===== API 호출 요청 데이터 구조 =====
@dataclass
class APICallRequest:
    # API 호출 요청을 나타내는 데이터 클래스
    # - 작업 유형, 데이터, 우선순위, 전략 등을 포함
    operation: str                       # 작업 유형: 'embedding', 'intent_analysis', 'typo_correction', 'translation'
    data: Dict[str, Any]                # 요청 데이터 (텍스트, 언어 설정 등)
    priority: int = 5                    # 우선순위 (1=최고, 10=최저, 낮은 숫자일수록 높은 우선순위)
    strategy: APICallStrategy = APICallStrategy.CACHE_FIRST  # 사용할 API 호출 전략
    timeout: float = 30.0               # 타임아웃 시간 (초)
    require_fresh: bool = False         # 신선한 데이터 요구 여부 (캐시 무시하고 최신 데이터)


# ===== API 호출 응답 데이터 구조 =====
@dataclass
class APICallResponse:
    # API 호출 결과를 나타내는 데이터 클래스
    # - 성공/실패, 결과 데이터, 성능 정보 등을 포함
    success: bool                        # 호출 성공 여부
    data: Any = None                    # 응답 데이터 (임베딩, 분석 결과 등)
    error: Optional[str] = None          # 오류 메시지 (실패시)
    cache_hit: bool = False             # 캐시 히트 여부
    processing_time: float = 0.0        # 처리 소요 시간 (초)
    strategy_used: Optional[APICallStrategy] = None  # 실제 사용된 전략


# ===== 지능형 API 호출 관리 시스템 =====
class IntelligentAPIManager:
    
    # IntelligentAPIManager 초기화 - API 호출 최적화 시스템 설정
    # Args:
    #     cache_manager: 캐시 관리자 (Redis/메모리 캐시)
    #     batch_processor: 배치 처리 시스템
    #     openai_client: OpenAI API 클라이언트 (선택적)
    def __init__(self, cache_manager: CacheManager, batch_processor: BatchProcessor, 
                 openai_client=None):
        # ===== 1단계: 핵심 컴포넌트 설정 =====
        self.cache_manager = cache_manager              # 캐시 시스템 (임베딩, 의도분석 등 캐싱)
        self.batch_processor = batch_processor          # 배치 처리 시스템 (여러 요청 통합 처리)
        self.openai_client = openai_client              # OpenAI API 클라이언트
        
        # ===== 2단계: 배치 프로세서 OpenAI 클라이언트 연결 =====
        if openai_client:
            self.batch_processor.set_openai_client(openai_client)
        
        # ===== 3단계: 최적화 설정 구성 =====
        self.optimization_config = {
            'enable_smart_caching': True,               # 지능형 캐싱 활성화
            'enable_batch_processing': True,            # 배치 처리 활성화
            'min_batch_size': 2,                        # 최소 배치 크기
            'max_wait_time': 2.0,                       # 최대 대기 시간 (초)
            'similarity_threshold': 0.95,               # 유사 요청 통합 임계값
            'cache_hit_bonus': 0.8                      # 캐시 히트시 우선순위 보너스
        }
        
        # ===== 4단계: 요청 추적 시스템 초기화 =====
        self.pending_requests = {}                      # 현재 처리 중인 요청 추적
        self.recent_requests = {}                       # 최근 요청 캐시 (중복 방지용)
        
        # ===== 5단계: 성능 통계 시스템 초기화 =====
        self.stats = {
            'total_requests': 0,                        # 총 요청 수
            'cache_hits': 0,                            # 캐시 히트 수
            'batch_processed': 0,                       # 배치 처리 수
            'immediate_calls': 0,                       # 즉시 호출 수
            'skipped_calls': 0,                         # 스킵된 호출 수
            'api_calls_saved': 0,                       # 절약된 API 호출 수
            'total_processing_time': 0.0                # 총 처리 시간
        }
        
        # ===== 6단계: 초기화 완료 로깅 =====
        # logging.info("지능형 API 관리자 초기화 완료")

    # API 요청 지능형 처리 메인 메서드
    # Args:
    #     request: 처리할 API 호출 요청
    # Returns:
    #     APICallResponse: 처리 결과 (성공/실패, 데이터, 성능 정보)
    def process_request(self, request: APICallRequest) -> APICallResponse:
        # ===== 처리 시작 시간 기록 =====
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            # ===== 1단계: 요청 중복 검사 및 통합 =====
            # 동일한 요청이 이미 처리 중이거나 최근에 처리된 경우 통합
            deduplicated_request = self._deduplicate_request(request)
            if deduplicated_request != request:
                logging.debug(f"중복 요청 감지 및 통합: {request.operation}")
                request = deduplicated_request
            
            # ===== 2단계: 최적 전략 결정 =====
            # 요청 특성, 우선순위, 캐시 상태 등을 고려하여 최적의 API 호출 전략 선택
            strategy = self._determine_optimal_strategy(request)
            
            # ===== 3단계: 전략에 따른 처리 실행 =====
            # 선택된 전략(CACHE_FIRST, BATCH_ONLY, IMMEDIATE, SKIP)에 따라 실제 API 호출 수행
            response = self._execute_strategy(request, strategy)
            
            # ===== 4단계: 응답 후처리 및 캐싱 =====
            # 결과 캐싱, 최근 요청 기록 업데이트, 대기 중인 요청 정리
            response = self._post_process_response(request, response)
            
            # ===== 5단계: 성능 통계 업데이트 =====
            processing_time = time.time() - start_time
            self._update_stats(response, processing_time)
            
            # ===== 6단계: 최종 응답 반환 =====
            return response
            
        except Exception as e:
            # ===== 예외 처리: 전체 처리 실패 =====
            logging.error(f"API 요청 처리 실패: {e}")
            return APICallResponse(
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )

    # 중복 요청 감지 및 통합 메서드
    # Args:
    #     request: 중복 검사할 API 요청
    # Returns:
    #     APICallRequest: 중복 제거된 요청 (기존 요청과 통합되거나 새 요청)
    def _deduplicate_request(self, request: APICallRequest) -> APICallRequest:
        # ===== 1단계: 요청 해시 생성 =====
        request_hash = self._generate_request_hash(request)
        
        # ===== 2단계: 최근 동일 요청 확인 (5분 내) =====
        current_time = time.time()
        if request_hash in self.recent_requests:
            last_time, last_response = self.recent_requests[request_hash]
            if current_time - last_time < 300:  # 5분 내 동일 요청
                logging.debug(f"최근 동일 요청 발견, 캐시된 응답 사용: {request.operation}")
                self.stats['api_calls_saved'] += 1
                return request
        
        # ===== 3단계: 현재 처리 중인 동일 요청 확인 =====
        if request_hash in self.pending_requests:
            pending_request = self.pending_requests[request_hash]
            # 3-1: 우선순위가 더 높으면 기존 요청 업데이트
            if request.priority < pending_request.priority:
                self.pending_requests[request_hash] = request
                logging.debug(f"더 높은 우선순위로 요청 업데이트: {request.operation}")
            # 3-2: 기존 대기 중인 요청과 통합
            return self.pending_requests[request_hash]
        
        # ===== 4단계: 새 요청 등록 =====
        self.pending_requests[request_hash] = request
        return request

    # 최적 API 호출 전략 결정 메서드
    # Args:
    #     request: 전략을 결정할 API 요청
    # Returns:
    #     APICallStrategy: 최적화된 API 호출 전략
    def _determine_optimal_strategy(self, request: APICallRequest) -> APICallStrategy:
        # ===== 1단계: 사용자 지정 전략 우선 고려 =====
        # 사용자가 명시적으로 전략을 지정한 경우 우선 적용
        if request.strategy != APICallStrategy.CACHE_FIRST:
            return request.strategy
        
        # ===== 2단계: 신선한 데이터 요구시 캐시 건너뛰기 =====
        # require_fresh=True인 경우 캐시를 무시하고 최신 데이터 요청
        if request.require_fresh:
            if self._should_use_batch(request):
                return APICallStrategy.BATCH_ONLY
            else:
                return APICallStrategy.IMMEDIATE
        
        # ===== 3단계: 캐시 사용 가능성 확인 =====
        # 캐시에 데이터가 있으면 캐시 우선 전략 사용
        cache_available = self._check_cache_availability(request)
        if cache_available:
            return APICallStrategy.CACHE_FIRST
        
        # ===== 4단계: 배치 처리 적합성 확인 =====
        # 배치 처리가 효율적인 경우 배치 전용 전략 사용
        if self._should_use_batch(request):
            return APICallStrategy.BATCH_ONLY
        
        # ===== 5단계: 조건부 스킵 확인 =====
        # 불필요한 요청인 경우 스킵 전략 사용
        if self._should_skip_call(request):
            return APICallStrategy.SKIP
        
        # ===== 6단계: 기본 전략 (즉시 호출) =====
        return APICallStrategy.IMMEDIATE

    # 선택된 전략에 따른 API 호출 실행 메서드
    # Args:
    #     request: 실행할 API 요청
    #     strategy: 사용할 API 호출 전략
    # Returns:
    #     APICallResponse: 전략 실행 결과
    def _execute_strategy(self, request: APICallRequest, strategy: APICallStrategy) -> APICallResponse:
        
        # ===== 캐시 우선 전략 실행 =====
        if strategy == APICallStrategy.CACHE_FIRST:
            # 1단계: 캐시에서 데이터 조회 시도
            cache_result = self._try_cache_first(request)
            if cache_result:
                return cache_result
            
            # 2단계: 캐시 미스시 폴백 전략 선택
            # 배치 처리가 적합하면 배치 전용, 아니면 즉시 호출
            fallback_strategy = (APICallStrategy.BATCH_ONLY 
                               if self._should_use_batch(request) 
                               else APICallStrategy.IMMEDIATE)
            return self._execute_strategy(request, fallback_strategy)
        
        # ===== 배치 전용 전략 실행 =====
        elif strategy == APICallStrategy.BATCH_ONLY:
            return self._execute_batch_call(request)
        
        # ===== 즉시 호출 전략 실행 =====
        elif strategy == APICallStrategy.IMMEDIATE:
            return self._execute_immediate_call(request)
        
        # ===== 스킵 전략 실행 =====
        elif strategy == APICallStrategy.SKIP:
            return APICallResponse(
                success=True,
                data=None,
                strategy_used=strategy
            )
        
        # ===== 알 수 없는 전략 처리 =====
        else:
            return APICallResponse(
                success=False,
                error=f"알 수 없는 전략: {strategy}"
            )

    # 캐시 우선 조회 메서드
    # Args:
    #     request: 캐시에서 조회할 API 요청
    # Returns:
    #     Optional[APICallResponse]: 캐시 히트시 응답, 미스시 None
    def _try_cache_first(self, request: APICallRequest) -> Optional[APICallResponse]:
        operation = request.operation
        data = request.data
        
        try:
            # ===== 임베딩 캐시 조회 =====
            if operation == 'embedding':
                text = data.get('text', '')
                cached_embedding = self.cache_manager.get_embedding_cache(text)
                if cached_embedding:
                    self.stats['cache_hits'] += 1
                    return APICallResponse(
                        success=True,
                        data=cached_embedding,
                        cache_hit=True,
                        strategy_used=APICallStrategy.CACHE_FIRST
                    )
            
            # ===== 의도 분석 캐시 조회 =====
            elif operation == 'intent_analysis':
                query = data.get('query', '')
                cached_intent = self.cache_manager.get_intent_analysis_cache(query)
                if cached_intent:
                    self.stats['cache_hits'] += 1
                    return APICallResponse(
                        success=True,
                        data=cached_intent,
                        cache_hit=True,
                        strategy_used=APICallStrategy.CACHE_FIRST
                    )
            
            # ===== 오타 수정 캐시 조회 =====
            elif operation == 'typo_correction':
                text = data.get('text', '')
                cached_correction = self.cache_manager.get_typo_correction_cache(text)
                if cached_correction:
                    self.stats['cache_hits'] += 1
                    return APICallResponse(
                        success=True,
                        data=cached_correction,
                        cache_hit=True,
                        strategy_used=APICallStrategy.CACHE_FIRST
                    )
            
            # ===== 번역 캐시 조회 =====
            elif operation == 'translation':
                text = data.get('text', '')
                source_lang = data.get('source_lang', 'ko')
                target_lang = data.get('target_lang', 'en')
                cached_translation = self.cache_manager.get_translation_cache(
                    text, source_lang, target_lang
                )
                if cached_translation:
                    self.stats['cache_hits'] += 1
                    return APICallResponse(
                        success=True,
                        data=cached_translation,
                        cache_hit=True,
                        strategy_used=APICallStrategy.CACHE_FIRST
                    )
            
            # ===== 캐시 미스 =====
            return None
            
        except Exception as e:
            # ===== 예외 처리: 캐시 조회 실패 =====
            logging.error(f"캐시 조회 실패: {e}")
            return None

    # 배치 호출 실행 메서드
    # Args:
    #     request: 배치 처리할 API 요청
    # Returns:
    #     APICallResponse: 배치 처리 결과
    def _execute_batch_call(self, request: APICallRequest) -> APICallResponse:
        try:
            # ===== 1단계: 배치 요청 생성 =====
            batch_request = BatchRequest(
                id=str(uuid.uuid4()),              # 고유 요청 ID 생성
                operation_type=request.operation,  # 작업 유형 설정
                data=request.data,                 # 요청 데이터 전달
                priority=request.priority          # 우선순위 설정
            )
            
            # ===== 2단계: 배치 처리 시스템에 제출 =====
            request_id = self.batch_processor.submit_request(batch_request)
            
            # ===== 3단계: 배치 처리 결과 대기 =====
            batch_result = self.batch_processor.get_result(request_id, request.timeout)
            
            # ===== 4단계: 배치 처리 성공시 =====
            if batch_result and batch_result.success:
                self.stats['batch_processed'] += 1
                
                # 4-1: 결과 캐싱 (향후 요청을 위한 캐시 저장)
                self._cache_result(request, batch_result.result)
                
                return APICallResponse(
                    success=True,
                    data=batch_result.result,
                    processing_time=batch_result.processing_time,
                    strategy_used=APICallStrategy.BATCH_ONLY
                )
            else:
                # ===== 5단계: 배치 처리 실패시 =====
                error_msg = batch_result.error if batch_result else "배치 처리 타임아웃"
                return APICallResponse(
                    success=False,
                    error=error_msg,
                    strategy_used=APICallStrategy.BATCH_ONLY
                )
                
        except Exception as e:
            # ===== 예외 처리: 배치 호출 실패 =====
            logging.error(f"배치 호출 실패: {e}")
            return APICallResponse(
                success=False,
                error=str(e),
                strategy_used=APICallStrategy.BATCH_ONLY
            )

    # 즉시 호출 실행 메서드
    # Args:
    #     request: 즉시 처리할 API 요청
    # Returns:
    #     APICallResponse: 즉시 호출 결과
    def _execute_immediate_call(self, request: APICallRequest) -> APICallResponse:
        # ===== 1단계: OpenAI 클라이언트 확인 =====
        if not self.openai_client:
            return APICallResponse(
                success=False,
                error="OpenAI 클라이언트가 설정되지 않음",
                strategy_used=APICallStrategy.IMMEDIATE
            )
        
        try:
            # ===== 2단계: 처리 시작 시간 기록 =====
            start_time = time.time()
            operation = request.operation
            data = request.data
            
            # ===== 3단계: 작업 유형별 API 호출 =====
            if operation == 'embedding':
                result = self._call_embedding_api(data)
            elif operation == 'intent_analysis':
                result = self._call_intent_analysis_api(data)
            elif operation == 'typo_correction':
                result = self._call_typo_correction_api(data)
            elif operation == 'translation':
                result = self._call_translation_api(data)
            else:
                return APICallResponse(
                    success=False,
                    error=f"지원하지 않는 작업: {operation}",
                    strategy_used=APICallStrategy.IMMEDIATE
                )
            
            # ===== 4단계: 처리 시간 계산 및 통계 업데이트 =====
            processing_time = time.time() - start_time
            self.stats['immediate_calls'] += 1
            
            # ===== 5단계: 결과 캐싱 =====
            self._cache_result(request, result)
            
            # ===== 6단계: 성공 응답 반환 =====
            return APICallResponse(
                success=True,
                data=result,
                processing_time=processing_time,
                strategy_used=APICallStrategy.IMMEDIATE
            )
            
        except Exception as e:
            # ===== 예외 처리: 즉시 호출 실패 =====
            logging.error(f"즉시 호출 실패: {e}")
            return APICallResponse(
                success=False,
                error=str(e),
                strategy_used=APICallStrategy.IMMEDIATE
            )

    # 임베딩 API 호출 메서드
    # Args:
    #     data: 임베딩 요청 데이터 (text 포함)
    # Returns:
    #     List[float]: 1536차원 임베딩 벡터
    def _call_embedding_api(self, data: Dict[str, Any]) -> List[float]:
        text = data.get('text', '')
        response = self.openai_client.embeddings.create(
            model='text-embedding-3-small',  # 경제적이고 성능 좋은 임베딩 모델
            input=text
        )
        return response.data[0].embedding

    # 의도 분석 API 호출 메서드
    # Args:
    #     data: 의도 분석 요청 데이터 (query 포함)
    # Returns:
    #     Dict[str, Any]: 의도 분석 결과 딕셔너리
    # _call_intent_analysis_api 메서드 제거됨 - unified_analyzer.analyze_and_correct()로 통합

    # 오타 수정 API 호출 메서드
    # Args:
    #     data: 오타 수정 요청 데이터 (text 포함)
    # Returns:
    #     str: 수정된 텍스트
    # _call_typo_correction_api 메서드 제거됨 - unified_analyzer.analyze_and_correct()로 통합

    # 번역 API 호출 메서드
    # Args:
    #     data: 번역 요청 데이터 (text, source_lang, target_lang 포함)
    # Returns:
    #     str: 번역된 텍스트
    def _call_translation_api(self, data: Dict[str, Any]) -> str:
        text = data.get('text', '')
        source_lang = data.get('source_lang', 'ko')
        target_lang = data.get('target_lang', 'en')
        
        # ===== 언어 코드를 영어 이름으로 매핑 =====
        lang_map = {'ko': 'Korean', 'en': 'English'}
        system_prompt = f"Translate from {lang_map.get(source_lang, source_lang)} to {lang_map.get(target_lang, target_lang)}. Only provide the translation."
        
        # ===== GPT API 호출 =====
        response = self.openai_client.chat.completions.create(
            model='gpt-5-mini',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            max_completion_tokens=60000,              # 충분한 번역 결과 길이
            # temperature=0.3              # 일관성 있는 번역 (낮은 창의성)
        )
        
        return response.choices[0].message.content.strip()

    # API 호출 결과 캐싱 메서드
    # Args:
    #     request: 원본 API 요청
    #     result: 캐싱할 결과 데이터
    def _cache_result(self, request: APICallRequest, result: Any):
        try:
            operation = request.operation
            data = request.data
            
            # ===== 작업 유형별 캐싱 =====
            if operation == 'embedding':
                text = data.get('text', '')
                self.cache_manager.set_embedding_cache(text, result)
            elif operation == 'intent_analysis':
                query = data.get('query', '')
                self.cache_manager.set_intent_analysis_cache(query, result)
            elif operation == 'typo_correction':
                text = data.get('text', '')
                self.cache_manager.set_typo_correction_cache(text, result)
            elif operation == 'translation':
                text = data.get('text', '')
                source_lang = data.get('source_lang', 'ko')
                target_lang = data.get('target_lang', 'en')
                self.cache_manager.set_translation_cache(text, result, source_lang, target_lang)
                
        except Exception as e:
            # ===== 예외 처리: 캐싱 실패 =====
            logging.error(f"결과 캐싱 실패: {e}")

    # 캐시 사용 가능 여부 확인 메서드
    # Args:
    #     request: 캐시 확인할 API 요청
    # Returns:
    #     bool: 캐시에 데이터가 있으면 True, 없으면 False
    def _check_cache_availability(self, request: APICallRequest) -> bool:
        # ===== 1단계: 지능형 캐싱 활성화 확인 =====
        if not self.optimization_config['enable_smart_caching']:
            return False
        
        operation = request.operation
        data = request.data
        
        try:
            # ===== 2단계: 작업 유형별 캐시 존재 여부 확인 =====
            if operation == 'embedding':
                text = data.get('text', '')
                return self.cache_manager.get_embedding_cache(text) is not None
            elif operation == 'intent_analysis':
                query = data.get('query', '')
                return self.cache_manager.get_intent_analysis_cache(query) is not None
            elif operation == 'typo_correction':
                text = data.get('text', '')
                return self.cache_manager.get_typo_correction_cache(text) is not None
            elif operation == 'translation':
                text = data.get('text', '')
                source_lang = data.get('source_lang', 'ko')
                target_lang = data.get('target_lang', 'en')
                return self.cache_manager.get_translation_cache(text, source_lang, target_lang) is not None
            
            return False
            
        except Exception as e:
            # ===== 예외 처리: 캐시 확인 실패 =====
            logging.error(f"캐시 확인 실패: {e}")
            return False

    # 배치 처리 사용 여부 결정 메서드
    # Args:
    #     request: 배치 처리 여부를 결정할 API 요청
    # Returns:
    #     bool: 배치 처리가 적합하면 True, 아니면 False
    def _should_use_batch(self, request: APICallRequest) -> bool:
        # ===== 1단계: 배치 처리 활성화 확인 =====
        if not self.optimization_config['enable_batch_processing']:
            return False
        
        # ===== 2단계: 높은 우선순위 요청은 즉시 처리 =====
        # 우선순위 1-2는 즉시 처리 (긴급 요청)
        if request.priority <= 2:
            return False
        
        # ===== 3단계: 배치 처리 가능한 작업 유형 확인 =====
        batchable_operations = ['embedding', 'translation', 'intent_analysis', 'typo_correction']
        return request.operation in batchable_operations

    # API 호출 스킵 여부 결정 메서드
    # Args:
    #     request: 스킵 여부를 결정할 API 요청
    # Returns:
    #     bool: 스킵해야 하면 True, 아니면 False
    def _should_skip_call(self, request: APICallRequest) -> bool:
        # ===== 1단계: 낮은 우선순위 요청 확인 =====
        # 우선순위 8-10은 매우 낮은 우선순위
        if request.priority >= 8:
            request_hash = self._generate_request_hash(request)
            
            # ===== 2단계: 최근 동일 요청 확인 =====
            if request_hash in self.recent_requests:
                last_time, _ = self.recent_requests[request_hash]
                # 1분 내 동일 요청이 있으면 스킵
                if time.time() - last_time < 60:
                    self.stats['skipped_calls'] += 1
                    return True
        
        return False

    # 요청 해시 생성 메서드 (중복 요청 감지용)
    # Args:
    #     request: 해시를 생성할 API 요청
    # Returns:
    #     str: MD5 해시 문자열
    def _generate_request_hash(self, request: APICallRequest) -> str:
        # ===== 1단계: 해시 데이터 구성 =====
        hash_data = {
            'operation': request.operation,  # 작업 유형
            'data': request.data             # 요청 데이터
        }
        
        # ===== 2단계: JSON 직렬화 및 MD5 해시 생성 =====
        hash_string = json.dumps(hash_data, sort_keys=True)  # 정렬로 일관성 보장
        return hashlib.md5(hash_string.encode()).hexdigest()

    # 응답 후처리 메서드
    # Args:
    #     request: 원본 API 요청
    #     response: 처리된 응답
    # Returns:
    #     APICallResponse: 후처리된 응답
    def _post_process_response(self, request: APICallRequest, response: APICallResponse) -> APICallResponse:
        # ===== 1단계: 최근 요청 기록 업데이트 =====
        # 중복 요청 감지를 위한 최근 요청 캐시 업데이트
        request_hash = self._generate_request_hash(request)
        self.recent_requests[request_hash] = (time.time(), response)
        
        # ===== 2단계: 보류 중인 요청에서 제거 =====
        # 처리 완료된 요청을 대기 목록에서 제거
        if request_hash in self.pending_requests:
            del self.pending_requests[request_hash]
        
        return response

    # 성능 통계 업데이트 메서드
    # Args:
    #     response: 처리된 API 응답
    #     processing_time: 처리 소요 시간
    def _update_stats(self, response: APICallResponse, processing_time: float):
        # ===== 1단계: 총 처리 시간 누적 =====
        self.stats['total_processing_time'] += processing_time
        
        # ===== 2단계: 캐시 히트 통계 업데이트 =====
        if response.cache_hit:
            self.stats['cache_hits'] += 1
        
        # ===== 3단계: 전략별 통계 업데이트 =====
        if response.strategy_used == APICallStrategy.BATCH_ONLY:
            self.stats['batch_processed'] += 1
        elif response.strategy_used == APICallStrategy.IMMEDIATE:
            self.stats['immediate_calls'] += 1
        elif response.strategy_used == APICallStrategy.SKIP:
            self.stats['skipped_calls'] += 1

    # 성능 통계 조회 메서드
    # Returns:
    #     Dict[str, Any]: 종합 성능 통계 정보
    def get_performance_stats(self) -> Dict[str, Any]:
        # ===== 1단계: 캐시 히트 비율 계산 =====
        cache_hit_rate = (self.stats['cache_hits'] / max(self.stats['total_requests'], 1)) * 100
        
        # ===== 2단계: 평균 처리 시간 계산 =====
        avg_processing_time = self.stats['total_processing_time'] / max(self.stats['total_requests'], 1)
        
        # ===== 3단계: 종합 통계 정보 구성 =====
        return {
            'total_requests': self.stats['total_requests'],           # 총 요청 수
            'cache_hit_rate': round(cache_hit_rate, 2),               # 캐시 히트 비율 (%)
            'api_calls_saved': self.stats['api_calls_saved'],         # 절약된 API 호출 수
            'batch_processed': self.stats['batch_processed'],         # 배치 처리 수
            'immediate_calls': self.stats['immediate_calls'],         # 즉시 호출 수
            'skipped_calls': self.stats['skipped_calls'],             # 스킵된 호출 수
            'average_processing_time': round(avg_processing_time, 3), # 평균 처리 시간 (초)
            'cache_stats': self.cache_manager.get_cache_stats(),      # 캐시 시스템 통계
            'batch_stats': self.batch_processor.get_stats()           # 배치 처리 시스템 통계
        }

    # 최적화 설정 업데이트 메서드
    # Args:
    #     **kwargs: 업데이트할 설정 키-값 쌍
    def optimize_settings(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.optimization_config:
                self.optimization_config[key] = value
                # logging.info(f"최적화 설정 업데이트: {key} = {value}")

    # 최근 요청 캐시 지우기 메서드
    def clear_recent_requests(self):
        self.recent_requests.clear()
        logging.info("최근 요청 캐시 지워짐")

    # 시스템 상태 확인 메서드
    # Returns:
    #     Dict[str, Any]: 시스템 상태 정보
    def health_check(self) -> Dict[str, Any]:
        return {
            'cache_available': self.cache_manager.is_cache_available(),        # 캐시 시스템 가용성
            'batch_processor_running': self.batch_processor.running,           # 배치 프로세서 실행 상태
            'pending_requests': len(self.pending_requests),                    # 대기 중인 요청 수
            'recent_requests': len(self.recent_requests),                      # 최근 요청 캐시 크기
            'openai_client_available': self.openai_client is not None          # OpenAI 클라이언트 가용성
        }
