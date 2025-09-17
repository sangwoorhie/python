#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
지능형 API 호출 관리 시스템
캐싱, 배치 처리, 조건부 호출을 통합 관리하여 API 비용과 지연시간 최적화
"""

import uuid
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .cache_manager import CacheManager
from .batch_processor import BatchProcessor, BatchRequest, BatchResult


class APICallStrategy(Enum):
    """API 호출 전략"""
    CACHE_FIRST = "cache_first"          # 캐시 우선
    BATCH_ONLY = "batch_only"            # 배치 처리만
    IMMEDIATE = "immediate"              # 즉시 호출
    SKIP = "skip"                        # 호출 생략


@dataclass
class APICallRequest:
    """API 호출 요청 정보"""
    operation: str                       # 'embedding', 'intent_analysis', etc.
    data: Dict[str, Any]                # 요청 데이터
    priority: int = 5                    # 우선순위 (1=최고, 10=최저)
    strategy: APICallStrategy = APICallStrategy.CACHE_FIRST
    timeout: float = 30.0               # 타임아웃 (초)
    require_fresh: bool = False         # 신선한 데이터 요구 (캐시 무시)


@dataclass
class APICallResponse:
    """API 호출 응답 정보"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    cache_hit: bool = False
    processing_time: float = 0.0
    strategy_used: Optional[APICallStrategy] = None


class IntelligentAPIManager:
    """지능형 API 호출 관리 시스템"""
    
    def __init__(self, cache_manager: CacheManager, batch_processor: BatchProcessor, 
                 openai_client=None):
        self.cache_manager = cache_manager
        self.batch_processor = batch_processor
        self.openai_client = openai_client
        
        # 배치 프로세서에 OpenAI 클라이언트 설정
        if openai_client:
            self.batch_processor.set_openai_client(openai_client)
        
        # 최적화 설정
        self.optimization_config = {
            'enable_smart_caching': True,
            'enable_batch_processing': True,
            'min_batch_size': 2,
            'max_wait_time': 2.0,
            'similarity_threshold': 0.95,  # 유사 요청 통합 임계값
            'cache_hit_bonus': 0.8         # 캐시 히트시 우선순위 보너스
        }
        
        # 중복 요청 추적
        self.pending_requests = {}
        self.recent_requests = {}  # 최근 요청 캐시 (중복 방지)
        
        # 통계 정보
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'batch_processed': 0,
            'immediate_calls': 0,
            'skipped_calls': 0,
            'api_calls_saved': 0,
            'total_processing_time': 0.0
        }
        
        logging.info("지능형 API 관리자 초기화 완료")

    def process_request(self, request: APICallRequest) -> APICallResponse:
        """API 요청 지능형 처리"""
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            # 1. 요청 중복 검사 및 통합
            deduplicated_request = self._deduplicate_request(request)
            if deduplicated_request != request:
                logging.debug(f"중복 요청 감지 및 통합: {request.operation}")
                request = deduplicated_request
            
            # 2. 최적 전략 결정
            strategy = self._determine_optimal_strategy(request)
            
            # 3. 전략에 따른 처리
            response = self._execute_strategy(request, strategy)
            
            # 4. 응답 후처리 및 캐싱
            response = self._post_process_response(request, response)
            
            # 5. 통계 업데이트
            processing_time = time.time() - start_time
            self._update_stats(response, processing_time)
            
            return response
            
        except Exception as e:
            logging.error(f"API 요청 처리 실패: {e}")
            return APICallResponse(
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )

    def _deduplicate_request(self, request: APICallRequest) -> APICallRequest:
        """중복 요청 감지 및 통합"""
        request_hash = self._generate_request_hash(request)
        
        # 최근 동일 요청이 있는지 확인 (5분 내)
        current_time = time.time()
        if request_hash in self.recent_requests:
            last_time, last_response = self.recent_requests[request_hash]
            if current_time - last_time < 300:  # 5분
                logging.debug(f"최근 동일 요청 발견, 캐시된 응답 사용: {request.operation}")
                self.stats['api_calls_saved'] += 1
                return request
        
        # 현재 처리 중인 동일 요청이 있는지 확인
        if request_hash in self.pending_requests:
            pending_request = self.pending_requests[request_hash]
            # 우선순위가 더 높으면 기존 요청 업데이트
            if request.priority < pending_request.priority:
                self.pending_requests[request_hash] = request
                logging.debug(f"더 높은 우선순위로 요청 업데이트: {request.operation}")
            return self.pending_requests[request_hash]
        
        # 새 요청 등록
        self.pending_requests[request_hash] = request
        return request

    def _determine_optimal_strategy(self, request: APICallRequest) -> APICallStrategy:
        """최적 API 호출 전략 결정"""
        # 사용자 지정 전략이 있으면 우선 고려
        if request.strategy != APICallStrategy.CACHE_FIRST:
            return request.strategy
        
        # 신선한 데이터가 필요하면 캐시 건너뛰기
        if request.require_fresh:
            if self._should_use_batch(request):
                return APICallStrategy.BATCH_ONLY
            else:
                return APICallStrategy.IMMEDIATE
        
        # 캐시 확인
        cache_available = self._check_cache_availability(request)
        if cache_available:
            return APICallStrategy.CACHE_FIRST
        
        # 배치 처리 여부 결정
        if self._should_use_batch(request):
            return APICallStrategy.BATCH_ONLY
        
        # 조건부 스킵 확인
        if self._should_skip_call(request):
            return APICallStrategy.SKIP
        
        # 기본: 즉시 호출
        return APICallStrategy.IMMEDIATE

    def _execute_strategy(self, request: APICallRequest, strategy: APICallStrategy) -> APICallResponse:
        """선택된 전략에 따른 API 호출 실행"""
        
        if strategy == APICallStrategy.CACHE_FIRST:
            # 캐시 우선 전략
            cache_result = self._try_cache_first(request)
            if cache_result:
                return cache_result
            
            # 캐시 미스시 다음 전략 선택
            fallback_strategy = (APICallStrategy.BATCH_ONLY 
                               if self._should_use_batch(request) 
                               else APICallStrategy.IMMEDIATE)
            return self._execute_strategy(request, fallback_strategy)
        
        elif strategy == APICallStrategy.BATCH_ONLY:
            return self._execute_batch_call(request)
        
        elif strategy == APICallStrategy.IMMEDIATE:
            return self._execute_immediate_call(request)
        
        elif strategy == APICallStrategy.SKIP:
            return APICallResponse(
                success=True,
                data=None,
                strategy_used=strategy
            )
        
        else:
            return APICallResponse(
                success=False,
                error=f"알 수 없는 전략: {strategy}"
            )

    def _try_cache_first(self, request: APICallRequest) -> Optional[APICallResponse]:
        """캐시 우선 조회"""
        operation = request.operation
        data = request.data
        
        try:
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
            
            return None
            
        except Exception as e:
            logging.error(f"캐시 조회 실패: {e}")
            return None

    def _execute_batch_call(self, request: APICallRequest) -> APICallResponse:
        """배치 호출 실행"""
        try:
            # 배치 요청 생성
            batch_request = BatchRequest(
                id=str(uuid.uuid4()),
                operation_type=request.operation,
                data=request.data,
                priority=request.priority
            )
            
            # 배치 처리 시스템에 제출
            request_id = self.batch_processor.submit_request(batch_request)
            
            # 결과 대기
            batch_result = self.batch_processor.get_result(request_id, request.timeout)
            
            if batch_result and batch_result.success:
                self.stats['batch_processed'] += 1
                
                # 결과 캐싱
                self._cache_result(request, batch_result.result)
                
                return APICallResponse(
                    success=True,
                    data=batch_result.result,
                    processing_time=batch_result.processing_time,
                    strategy_used=APICallStrategy.BATCH_ONLY
                )
            else:
                error_msg = batch_result.error if batch_result else "배치 처리 타임아웃"
                return APICallResponse(
                    success=False,
                    error=error_msg,
                    strategy_used=APICallStrategy.BATCH_ONLY
                )
                
        except Exception as e:
            logging.error(f"배치 호출 실패: {e}")
            return APICallResponse(
                success=False,
                error=str(e),
                strategy_used=APICallStrategy.BATCH_ONLY
            )

    def _execute_immediate_call(self, request: APICallRequest) -> APICallResponse:
        """즉시 호출 실행"""
        if not self.openai_client:
            return APICallResponse(
                success=False,
                error="OpenAI 클라이언트가 설정되지 않음",
                strategy_used=APICallStrategy.IMMEDIATE
            )
        
        try:
            start_time = time.time()
            operation = request.operation
            data = request.data
            
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
            
            processing_time = time.time() - start_time
            self.stats['immediate_calls'] += 1
            
            # 결과 캐싱
            self._cache_result(request, result)
            
            return APICallResponse(
                success=True,
                data=result,
                processing_time=processing_time,
                strategy_used=APICallStrategy.IMMEDIATE
            )
            
        except Exception as e:
            logging.error(f"즉시 호출 실패: {e}")
            return APICallResponse(
                success=False,
                error=str(e),
                strategy_used=APICallStrategy.IMMEDIATE
            )

    def _call_embedding_api(self, data: Dict[str, Any]) -> List[float]:
        """임베딩 API 호출"""
        text = data.get('text', '')
        response = self.openai_client.embeddings.create(
            model='text-embedding-3-small',
            input=text
        )
        return response.data[0].embedding

    def _call_intent_analysis_api(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """의도 분석 API 호출"""
        query = data.get('query', '')
        
        system_prompt = """당신은 바이블 앱 문의 분석 전문가입니다.
질문의 본질적 의도를 파악하여 JSON 형태로 응답하세요:
{"core_intent": "...", "intent_category": "...", "primary_action": "...", "target_object": "..."}"""
        
        response = self.openai_client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"질문: {query}"}
            ],
            max_tokens=200,
            temperature=0.2
        )
        
        result_text = response.choices[0].message.content.strip()
        
        try:
            import json
            intent_data = json.loads(result_text)
            # 기존 호환성 필드 추가
            intent_data.update({
                'intent_type': intent_data.get('intent_category', '일반문의'),
                'main_topic': intent_data.get('target_object', '기타'),
                'specific_request': query[:100],
                'keywords': [query[:20]],
                'urgency': 'medium',
                'action_type': intent_data.get('primary_action', '기타')
            })
            return intent_data
        except json.JSONDecodeError:
            # 파싱 실패시 기본값 반환
            return {
                "core_intent": "general_inquiry",
                "intent_category": "일반문의",
                "primary_action": "기타",
                "target_object": "기타",
                "intent_type": "일반문의",
                "main_topic": "기타",
                "specific_request": query[:100],
                "keywords": [query[:20]],
                "urgency": "medium",
                "action_type": "기타"
            }

    def _call_typo_correction_api(self, data: Dict[str, Any]) -> str:
        """오타 수정 API 호출"""
        text = data.get('text', '')
        
        system_prompt = """한국어 맞춤법 및 오타 교정 전문가입니다.
입력된 텍스트의 맞춤법과 오타만 수정하세요. 의미와 어조는 변경하지 마세요.
수정된 텍스트만 반환하세요."""
        
        response = self.openai_client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            max_tokens=600,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()

    def _call_translation_api(self, data: Dict[str, Any]) -> str:
        """번역 API 호출"""
        text = data.get('text', '')
        source_lang = data.get('source_lang', 'ko')
        target_lang = data.get('target_lang', 'en')
        
        lang_map = {'ko': 'Korean', 'en': 'English'}
        system_prompt = f"Translate from {lang_map.get(source_lang, source_lang)} to {lang_map.get(target_lang, target_lang)}. Only provide the translation."
        
        response = self.openai_client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            max_tokens=600,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()

    def _cache_result(self, request: APICallRequest, result: Any):
        """결과 캐싱"""
        try:
            operation = request.operation
            data = request.data
            
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
            logging.error(f"결과 캐싱 실패: {e}")

    def _check_cache_availability(self, request: APICallRequest) -> bool:
        """캐시 사용 가능 여부 확인"""
        if not self.optimization_config['enable_smart_caching']:
            return False
        
        operation = request.operation
        data = request.data
        
        try:
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
            logging.error(f"캐시 확인 실패: {e}")
            return False

    def _should_use_batch(self, request: APICallRequest) -> bool:
        """배치 처리 사용 여부 결정"""
        if not self.optimization_config['enable_batch_processing']:
            return False
        
        # 높은 우선순위 요청은 즉시 처리
        if request.priority <= 2:
            return False
        
        # 배치 처리 가능한 작업 유형 확인
        batchable_operations = ['embedding', 'translation', 'intent_analysis', 'typo_correction']
        return request.operation in batchable_operations

    def _should_skip_call(self, request: APICallRequest) -> bool:
        """API 호출 스킵 여부 결정"""
        # 우선순위가 매우 낮고 최근에 유사한 요청이 있었다면 스킵
        if request.priority >= 8:
            request_hash = self._generate_request_hash(request)
            if request_hash in self.recent_requests:
                last_time, _ = self.recent_requests[request_hash]
                if time.time() - last_time < 60:  # 1분 내
                    self.stats['skipped_calls'] += 1
                    return True
        
        return False

    def _generate_request_hash(self, request: APICallRequest) -> str:
        """요청 해시 생성"""
        import hashlib
        import json
        
        hash_data = {
            'operation': request.operation,
            'data': request.data
        }
        
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.md5(hash_string.encode()).hexdigest()

    def _post_process_response(self, request: APICallRequest, response: APICallResponse) -> APICallResponse:
        """응답 후처리"""
        # 최근 요청 기록 업데이트
        request_hash = self._generate_request_hash(request)
        self.recent_requests[request_hash] = (time.time(), response)
        
        # 보류 중인 요청에서 제거
        if request_hash in self.pending_requests:
            del self.pending_requests[request_hash]
        
        return response

    def _update_stats(self, response: APICallResponse, processing_time: float):
        """통계 업데이트"""
        self.stats['total_processing_time'] += processing_time
        
        if response.cache_hit:
            self.stats['cache_hits'] += 1
        
        if response.strategy_used == APICallStrategy.BATCH_ONLY:
            self.stats['batch_processed'] += 1
        elif response.strategy_used == APICallStrategy.IMMEDIATE:
            self.stats['immediate_calls'] += 1
        elif response.strategy_used == APICallStrategy.SKIP:
            self.stats['skipped_calls'] += 1

    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 조회"""
        cache_hit_rate = (self.stats['cache_hits'] / max(self.stats['total_requests'], 1)) * 100
        avg_processing_time = self.stats['total_processing_time'] / max(self.stats['total_requests'], 1)
        
        return {
            'total_requests': self.stats['total_requests'],
            'cache_hit_rate': round(cache_hit_rate, 2),
            'api_calls_saved': self.stats['api_calls_saved'],
            'batch_processed': self.stats['batch_processed'],
            'immediate_calls': self.stats['immediate_calls'],
            'skipped_calls': self.stats['skipped_calls'],
            'average_processing_time': round(avg_processing_time, 3),
            'cache_stats': self.cache_manager.get_cache_stats(),
            'batch_stats': self.batch_processor.get_stats()
        }

    def optimize_settings(self, **kwargs):
        """최적화 설정 업데이트"""
        for key, value in kwargs.items():
            if key in self.optimization_config:
                self.optimization_config[key] = value
                logging.info(f"최적화 설정 업데이트: {key} = {value}")

    def clear_recent_requests(self):
        """최근 요청 캐시 지우기"""
        self.recent_requests.clear()
        logging.info("최근 요청 캐시 지워짐")

    def health_check(self) -> Dict[str, Any]:
        """시스템 상태 확인"""
        return {
            'cache_available': self.cache_manager.is_cache_available(),
            'batch_processor_running': self.batch_processor.running,
            'pending_requests': len(self.pending_requests),
            'recent_requests': len(self.recent_requests),
            'openai_client_available': self.openai_client is not None
        }
