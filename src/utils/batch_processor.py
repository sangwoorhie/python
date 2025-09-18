#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI API 배치 처리 시스템
- 여러 AI API 호출을 효율적으로 배치 처리하여 성능 최적화
- 임베딩, 번역, 의도분석, 오타수정 등의 배치 작업 지원
- 우선순위 기반 큐 시스템 및 비동기 처리
- API 호출 비용 절약 및 응답 시간 최적화
"""

import asyncio
import logging
import time
import threading
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from queue import Queue, Empty


# ===== 배치 요청 데이터 구조 =====
@dataclass
class BatchRequest:
    # 배치 처리 요청을 나타내는 데이터 클래스
    # - 각 요청은 고유 ID, 작업 유형, 데이터, 우선순위를 가짐
    id: str                                          # 요청 고유 식별자
    operation_type: str                              # 작업 유형: 'embedding', 'translation', 'intent_analysis', 'typo_correction'
    data: Dict[str, Any]                             # 처리할 데이터 (텍스트, 언어 등)
    callback: Optional[Callable] = None              # 처리 완료 후 콜백 함수 (선택적)
    priority: int = 5                                # 우선순위 (1=최고, 10=최저)


# ===== 배치 처리 결과 데이터 구조 =====
@dataclass
class BatchResult:
    # 배치 처리 결과를 나타내는 데이터 클래스
    # - 성공/실패 여부, 결과 데이터, 오류 정보, 처리 시간 포함
    request_id: str                                  # 요청 ID (BatchRequest.id와 매칭)
    success: bool                                    # 처리 성공 여부
    result: Any = None                               # 처리 결과 데이터 (성공시)
    error: Optional[str] = None                      # 오류 메시지 (실패시)
    processing_time: float = 0.0                     # 처리 소요 시간 (초)


# ===== AI API 호출을 위한 지능형 배치 처리 시스템 =====
class BatchProcessor:
    
    # BatchProcessor 초기화 - 배치 처리 시스템 설정
    # Args:
    #     max_workers: 최대 작업자 스레드 수 (동시 처리 수준)
    #     batch_size: 배치당 최대 요청 수 (API 효율성)
    #     batch_timeout: 배치 수집 대기 시간 (초)
    def __init__(self, max_workers: int = 5, batch_size: int = 10, batch_timeout: float = 2.0):
        # ===== 기본 설정 =====
        self.max_workers = max_workers                # 동시 작업자 수
        self.batch_size = batch_size                  # 배치 크기
        self.batch_timeout = batch_timeout            # 배치 대기 시간
        
        # ===== 작업 유형별 큐 시스템 =====
        # 각 AI 작업 유형별로 별도 큐 운영 (우선순위 및 효율성)
        self.request_queues = {
            'embedding': Queue(),                     # 텍스트 임베딩 요청
            'translation': Queue(),                   # 텍스트 번역 요청
            'intent_analysis': Queue(),               # 의도 분석 요청
            'typo_correction': Queue()                # 오타 수정 요청
        }
        
        # ===== 결과 저장 및 추적 시스템 =====
        self.results = {}                             # 완료된 결과 저장소 {request_id: BatchResult}
        self.pending_requests = {}                    # 처리 중인 요청 추적 {request_id: BatchRequest}
        
        # ===== 스레드 풀 시스템 =====
        self.executor = ThreadPoolExecutor(max_workers=max_workers)  # 비동기 작업 실행기
        
        # ===== 배치 처리 스레드 관리 =====
        self.batch_threads = {}                       # 작업 유형별 배치 처리 스레드
        self.running = False                          # 시스템 실행 상태 플래그
        
        # ===== 성능 통계 시스템 =====
        self.stats = {
            'total_requests': 0,                      # 총 요청 수
            'batch_processed': 0,                     # 처리된 배치 수
            'cache_hits': 0,                          # 캐시 히트 수
            'api_calls_saved': 0                      # 절약된 API 호출 수
        }
        
        # 초기화 완료 로깅
        logging.info(f"배치 프로세서 초기화: workers={max_workers}, batch_size={batch_size}, timeout={batch_timeout}s")

    # 배치 처리 시스템 시작 메서드
    # - 각 작업 유형별로 별도 배치 처리 스레드 생성 및 시작
    # - 이미 실행 중인 경우 중복 시작 방지
    def start(self):
        # ===== 1단계: 중복 시작 방지 =====
        if self.running:
            return
            
        # ===== 2단계: 시스템 상태 활성화 =====
        self.running = True
        
        # ===== 3단계: 작업 유형별 배치 처리 스레드 생성 =====
        # 각 AI 작업 유형(embedding, translation, intent_analysis, typo_correction)별로
        # 독립적인 배치 처리 스레드를 생성하여 병렬 처리
        for operation_type in self.request_queues.keys():
            thread = threading.Thread(
                target=self._batch_worker,               # 배치 작업자 함수
                args=(operation_type,),                  # 작업 유형 전달
                daemon=True                              # 메인 스레드 종료시 함께 종료
            )
            thread.start()                               # 스레드 시작
            self.batch_threads[operation_type] = thread  # 스레드 추적용 저장
            
        # ===== 4단계: 시작 완료 로깅 =====
        logging.info("배치 처리 시스템 시작됨")

    # 배치 처리 시스템 중지 메서드
    # - 모든 배치 처리 스레드 안전하게 종료
    # - 진행 중인 작업 완료 대기 후 정리
    def stop(self):
        # ===== 1단계: 시스템 종료 신호 =====
        self.running = False
        
        # ===== 2단계: 모든 배치 처리 스레드 종료 대기 =====
        # 각 작업 유형별 스레드가 안전하게 종료될 때까지 대기
        for thread in self.batch_threads.values():
            if thread.is_alive():                        # 스레드가 살아있는 경우
                thread.join(timeout=5)                   # 최대 5초 대기
                
        # ===== 3단계: 스레드 풀 정리 =====
        # 모든 작업자 스레드 종료 대기
        self.executor.shutdown(wait=True)
        
        # ===== 4단계: 종료 완료 로깅 =====
        logging.info("배치 처리 시스템 중지됨")

    # 배치 요청 제출 메서드
    # Args:
    #     request: 처리할 배치 요청 객체 (BatchRequest)
    # Returns:
    #     str: 요청 ID (결과 조회시 사용)
    def submit_request(self, request: BatchRequest) -> str:
        # ===== 1단계: 시스템 자동 시작 =====
        # 시스템이 아직 시작되지 않은 경우 자동으로 시작
        if not self.running:
            self.start()
            
        # ===== 2단계: 통계 업데이트 =====
        self.stats['total_requests'] += 1            # 총 요청 수 증가
        self.pending_requests[request.id] = request   # 처리 중인 요청 추가
        
        # ===== 3단계: 작업 유형 검증 및 큐 배치 =====
        if request.operation_type in self.request_queues:
            # 지원하는 작업 유형인 경우 해당 큐에 추가
            self.request_queues[request.operation_type].put(request)
            logging.debug(f"배치 요청 제출: {request.id} ({request.operation_type})")
        else:
            # 지원하지 않는 작업 유형인 경우 오류 로깅
            logging.error(f"지원하지 않는 작업 유형: {request.operation_type}")
            
        # ===== 4단계: 요청 ID 반환 =====
        return request.id

    # 배치 처리 결과 조회 메서드 (동기식 대기)
    # Args:
    #     request_id: 조회할 요청의 ID
    #     timeout: 최대 대기 시간 (초)
    # Returns:
    #     Optional[BatchResult]: 처리 결과 (타임아웃시 None)
    def get_result(self, request_id: str, timeout: float = 30.0) -> Optional[BatchResult]:
        # ===== 1단계: 대기 시작 시간 기록 =====
        start_time = time.time()
        
        # ===== 2단계: 결과 조회 루프 (폴링 방식) =====
        while time.time() - start_time < timeout:
            # 2-1: 결과 존재 여부 확인
            if request_id in self.results:
                # 2-2: 결과 추출 및 정리
                result = self.results.pop(request_id)     # 결과 가져오기 및 삭제
                if request_id in self.pending_requests:
                    del self.pending_requests[request_id] # 대기 목록에서 제거
                return result
                
            # 2-3: 짧은 대기 후 재시도 (CPU 효율성)
            time.sleep(0.1)  # 100ms 대기
            
        # ===== 3단계: 타임아웃 처리 =====
        logging.warning(f"배치 결과 타임아웃: {request_id}")
        return None

    # 배치 작업자 스레드 (특정 작업 유형 전담)
    # Args:
    #     operation_type: 처리할 작업 유형 ('embedding', 'translation', 등)
    # 무한 루프로 실행되며 시스템 종료시까지 배치 처리 수행
    def _batch_worker(self, operation_type: str):
        # ===== 해당 작업 유형의 큐 참조 =====
        queue = self.request_queues[operation_type]
        
        # ===== 배치 처리 메인 루프 =====
        while self.running:
            try:
                # ===== 1단계: 배치 수집 =====
                # 큐에서 여러 요청을 모아서 배치 구성
                batch_requests = self._collect_batch(queue, operation_type)
                
                # ===== 2단계: 빈 배치 처리 =====
                # 수집된 요청이 없으면 짧은 대기 후 재시도
                if not batch_requests:
                    time.sleep(0.1)
                    continue
                
                # ===== 3단계: 배치 처리 실행 =====
                # 수집된 배치를 실제 AI API로 처리
                batch_results = self._process_batch(batch_requests, operation_type)
                
                # ===== 4단계: 결과 저장 =====
                # 처리 결과를 결과 저장소에 저장 (클라이언트가 조회 가능)
                for result in batch_results:
                    self.results[result.request_id] = result
                    
                # ===== 5단계: 통계 업데이트 및 로깅 =====
                self.stats['batch_processed'] += 1
                logging.debug(f"배치 처리 완료: {operation_type}, {len(batch_requests)}개 요청")
                
            except Exception as e:
                # ===== 예외 처리: 오류 시 1초 대기 후 재시도 =====
                logging.error(f"배치 작업자 오류 ({operation_type}): {e}")
                time.sleep(1)

    # 배치 요청 수집 메서드 (지능형 배치 구성)
    # Args:
    #     queue: 요청 큐
    #     operation_type: 작업 유형 (로깅용)
    # Returns:
    #     List[BatchRequest]: 수집된 배치 요청 목록
    def _collect_batch(self, queue: Queue, operation_type: str) -> List[BatchRequest]:
        # ===== 배치 초기화 =====
        batch = []
        start_time = time.time()
        
        # ===== 1단계: 첫 번째 요청 대기 (블로킹) =====
        # 최소 1개 요청이 올 때까지 대기 (배치 시작 조건)
        try:
            first_request = queue.get(timeout=1.0)      # 최대 1초 대기
            batch.append(first_request)
        except Empty:
            # 타임아웃시 빈 배치 반환
            return batch
        
        # ===== 2단계: 추가 요청 수집 (논블로킹) =====
        # 배치 크기 또는 타임아웃 조건까지 추가 요청 수집
        while (len(batch) < self.batch_size and 
               time.time() - start_time < self.batch_timeout):
            try:
                # 논블로킹으로 추가 요청 시도
                request = queue.get_nowait()
                batch.append(request)
            except Empty:
                # 큐가 비어있으면 짧은 대기 후 재시도
                time.sleep(0.05)  # 50ms 대기
                
        # ===== 3단계: 우선순위 정렬 =====
        # 낮은 숫자 = 높은 우선순위로 정렬 (중요한 요청 우선 처리)
        batch.sort(key=lambda x: x.priority)
        
        # ===== 4단계: 배치 반환 =====
        return batch

    # 배치 처리 라우터 메서드 (작업 유형별 배치 처리 실행)
    # Args:
    #     batch_requests: 처리할 배치 요청 목록
    #     operation_type: 작업 유형
    # Returns:
    #     List[BatchResult]: 처리 결과 목록
    def _process_batch(self, batch_requests: List[BatchRequest], operation_type: str) -> List[BatchResult]:
        # ===== 결과 리스트 초기화 =====
        results = []
        
        # ===== 작업 유형별 배치 처리 라우팅 =====
        if operation_type == 'embedding':
            # 텍스트 임베딩 배치 처리 (OpenAI Embeddings API)
            results = self._process_embedding_batch(batch_requests)
        elif operation_type == 'translation':
            # 텍스트 번역 배치 처리 (GPT 기반)
            results = self._process_translation_batch(batch_requests)
        elif operation_type == 'intent_analysis':
            # 의도 분석 배치 처리 (GPT 기반)
            results = self._process_intent_analysis_batch(batch_requests)
        elif operation_type == 'typo_correction':
            # 오타 수정 배치 처리 (GPT 기반)
            results = self._process_typo_correction_batch(batch_requests)
        else:
            # ===== 지원하지 않는 작업 유형 처리 =====
            # 모든 요청을 실패로 처리
            for request in batch_requests:
                results.append(BatchResult(
                    request_id=request.id,
                    success=False,
                    error=f"지원하지 않는 작업 유형: {operation_type}"
                ))
                
        # ===== 처리 결과 반환 =====
        return results

    # 텍스트 임베딩 배치 처리 메서드 (OpenAI Embeddings API 활용)
    # Args:
    #     batch_requests: 임베딩 요청 배치 목록
    # Returns:
    #     List[BatchResult]: 임베딩 결과 목록
    def _process_embedding_batch(self, batch_requests: List[BatchRequest]) -> List[BatchResult]:
        # ===== 결과 리스트 초기화 =====
        results = []
        
        # ===== 1단계: 배치 데이터 준비 =====
        # 여러 텍스트를 한 번의 API 호출로 처리하기 위한 준비
        texts = []                                    # API에 전송할 텍스트 리스트
        request_map = {}                              # 인덱스 → 요청 객체 매핑
        
        # 1-1: 각 요청에서 텍스트 추출 및 매핑 구성
        for i, request in enumerate(batch_requests):
            text = request.data.get('text', '')       # 요청에서 텍스트 추출
            texts.append(text)                        # 배치 리스트에 추가
            request_map[i] = request                  # 인덱스 매핑 저장
        
        try:
            # ===== 2단계: 처리 시간 측정 시작 =====
            start_time = time.time()
            
            # ===== 3단계: OpenAI Embeddings API 배치 호출 =====
            if hasattr(self, 'openai_client'):
                # 3-1: 여러 텍스트를 한 번의 API 호출로 처리 (비용 및 시간 효율성)
                response = self.openai_client.embeddings.create(
                    model='text-embedding-3-small',  # 경제적이고 성능 좋은 모델
                    input=texts                       # 전체 텍스트 배치
                )
                
                # 3-2: 응답에서 임베딩 벡터 추출
                embeddings = [item.embedding for item in response.data]
                processing_time = time.time() - start_time
                
                # ===== 4단계: 결과 매핑 및 생성 =====
                # API 응답 순서와 요청 순서가 일치하므로 인덱스로 매핑
                for i, embedding in enumerate(embeddings):
                    request = request_map[i]
                    results.append(BatchResult(
                        request_id=request.id,
                        success=True,
                        result=embedding,             # 1536차원 임베딩 벡터
                        processing_time=processing_time / len(embeddings)  # 평균 처리 시간
                    ))
                    
                # ===== 5단계: 성공 로깅 =====
                logging.info(f"임베딩 배치 처리 완료: {len(texts)}개, {processing_time:.2f}s")
                
            else:
                # ===== OpenAI 클라이언트 미설정 오류 처리 =====
                for request in batch_requests:
                    results.append(BatchResult(
                        request_id=request.id,
                        success=False,
                        error="OpenAI 클라이언트가 설정되지 않음"
                    ))
                    
        except Exception as e:
            # ===== 예외 처리: API 호출 실패 =====
            logging.error(f"임베딩 배치 처리 실패: {e}")
            # 모든 요청을 실패로 처리
            for request in batch_requests:
                results.append(BatchResult(
                    request_id=request.id,
                    success=False,
                    error=str(e)
                ))
                
        # ===== 6단계: 결과 반환 =====
        return results

    # 텍스트 번역 배치 처리 메서드 (GPT 기반 다중 언어 번역)
    # Args:
    #     batch_requests: 번역 요청 배치 목록
    # Returns:
    #     List[BatchResult]: 번역 결과 목록
    def _process_translation_batch(self, batch_requests: List[BatchRequest]) -> List[BatchResult]:
        # ===== 결과 리스트 초기화 =====
        results = []
        
        # ===== 1단계: 언어쌍별 그룹화 =====
        # 같은 언어쌍(예: ko-en, en-ko)끼리 그룹화하여 효율적 배치 처리
        language_groups = {}
        
        # 1-1: 각 요청을 언어쌍별로 분류
        for request in batch_requests:
            source_lang = request.data.get('source_lang', 'ko')  # 기본값: 한국어
            target_lang = request.data.get('target_lang', 'en')  # 기본값: 영어
            lang_pair = f"{source_lang}-{target_lang}"           # 언어쌍 키 생성
            
            if lang_pair not in language_groups:
                language_groups[lang_pair] = []
            language_groups[lang_pair].append(request)
        
        # ===== 2단계: 언어쌍별 배치 번역 처리 =====
        for lang_pair, group_requests in language_groups.items():
            try:
                # 2-1: 언어쌍 분해 및 텍스트 추출
                source_lang, target_lang = lang_pair.split('-')
                texts = [req.data.get('text', '') for req in group_requests]
                
                # 2-2: 처리 시간 측정 시작
                start_time = time.time()
                
                # ===== 3단계: GPT 기반 배치 번역 =====
                if hasattr(self, 'openai_client'):
                    # 3-1: 여러 텍스트를 구분자로 연결 (한 번의 API 호출)
                    combined_text = '\n---SEPARATOR---\n'.join(texts)
                    
                    # 3-2: 번역 시스템 프롬프트 구성
                    system_prompt = f"Translate the following texts from {source_lang} to {target_lang}. Separate each translation with '---SEPARATOR---'."
                    
                    # 3-3: GPT API 호출 (배치 번역)
                    response = self.openai_client.chat.completions.create(
                        model='gpt-3.5-turbo',
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": combined_text}
                        ],
                        max_tokens=1000,                      # 충분한 번역 길이 허용
                        temperature=0.3                       # 일관성 있는 번역 (낮은 창의성)
                    )
                    
                    # 3-4: 번역 결과 분리 및 파싱
                    translated_combined = response.choices[0].message.content
                    translated_texts = translated_combined.split('---SEPARATOR---')
                    
                    processing_time = time.time() - start_time
                    
                    # ===== 4단계: 결과 매핑 및 검증 =====
                    for i, request in enumerate(group_requests):
                        if i < len(translated_texts):
                            # 4-1: 성공적 번역 결과 저장
                            results.append(BatchResult(
                                request_id=request.id,
                                success=True,
                                result=translated_texts[i].strip(),  # 앞뒤 공백 제거
                                processing_time=processing_time / len(group_requests)
                            ))
                        else:
                            # 4-2: 매핑 실패 처리 (번역 결과 부족)
                            results.append(BatchResult(
                                request_id=request.id,
                                success=False,
                                error="번역 결과 매핑 실패"
                            ))
                            
                    # ===== 5단계: 성공 로깅 =====
                    logging.info(f"번역 배치 처리 완료: {lang_pair}, {len(texts)}개, {processing_time:.2f}s")
                    
                else:
                    # ===== OpenAI 클라이언트 미설정 오류 처리 =====
                    for request in group_requests:
                        results.append(BatchResult(
                            request_id=request.id,
                            success=False,
                            error="OpenAI 클라이언트가 설정되지 않음"
                        ))
                        
            except Exception as e:
                # ===== 예외 처리: 언어쌍별 번역 실패 =====
                logging.error(f"번역 배치 처리 실패 ({lang_pair}): {e}")
                for request in group_requests:
                    results.append(BatchResult(
                        request_id=request.id,
                        success=False,
                        error=str(e)
                    ))
                    
        # ===== 6단계: 전체 번역 결과 반환 =====
        return results

    # 사용자 질문 의도 분석 배치 처리 메서드 (GPT 기반 질문 이해)
    # Args:
    #     batch_requests: 의도 분석 요청 배치 목록
    # Returns:
    #     List[BatchResult]: 의도 분석 결과 목록
    def _process_intent_analysis_batch(self, batch_requests: List[BatchRequest]) -> List[BatchResult]:
        # ===== 결과 리스트 초기화 =====
        results = []
        
        try:
            # ===== 1단계: 질문 텍스트 추출 =====
            queries = [req.data.get('query', '') for req in batch_requests]
            start_time = time.time()
            
            # ===== 2단계: GPT 기반 배치 의도 분석 =====
            if hasattr(self, 'openai_client'):
                # 2-1: 여러 질문을 구분자로 연결 (한 번의 API 호출)
                combined_queries = '\n---QUERY---\n'.join([f"Q{i+1}: {q}" for i, q in enumerate(queries)])
                
                # 2-2: 바이블 앱 전용 의도 분석 시스템 프롬프트
                system_prompt = """당신은 바이블 앱 문의 분석 전문가입니다. 
다음 질문들을 각각 분석하여 의도를 파악하세요.

각 질문에 대해 다음 JSON 형태로 응답하세요:
Q1: {"core_intent": "...", "intent_category": "...", "primary_action": "...", "target_object": "..."}
Q2: {"core_intent": "...", "intent_category": "...", "primary_action": "...", "target_object": "..."}
...

---QUERY--- 로 구분된 각 질문을 순서대로 분석해주세요."""

                # 2-3: GPT API 호출 (배치 의도 분석)
                response = self.openai_client.chat.completions.create(
                    model='gpt-3.5-turbo',
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": combined_queries}
                    ],
                    max_tokens=800,                       # 충분한 분석 결과 길이
                    temperature=0.2                       # 일관성 있는 분석 (낮은 창의성)
                )
                
                # 2-4: 분석 결과 추출 및 처리 시간 계산
                result_text = response.choices[0].message.content
                processing_time = time.time() - start_time
                
                # ===== 3단계: 결과 파싱 및 구조화 =====
                # 3-1: 정규식으로 각 질문의 JSON 결과 추출
                pattern = r'Q(\d+):\s*(\{[^}]+\})'
                matches = re.findall(pattern, result_text)
                
                # 3-2: 각 요청에 대한 결과 매핑
                for i, request in enumerate(batch_requests):
                    try:
                        if i < len(matches):
                            # 3-2-1: JSON 파싱 및 의도 데이터 구성
                            intent_data = json.loads(matches[i][1])
                            
                            # 3-2-2: 기존 시스템 호환성을 위한 필드 추가
                            intent_data.update({
                                'intent_type': intent_data.get('intent_category', '일반문의'),
                                'main_topic': intent_data.get('target_object', '기타'),
                                'specific_request': request.data.get('query', '')[:100],  # 요약
                                'keywords': [request.data.get('query', '')[:20]],         # 키워드 추출
                                'urgency': 'medium',                                      # 기본 우선순위
                                'action_type': intent_data.get('primary_action', '기타')  # 액션 타입
                            })
                            
                            # 3-2-3: 성공 결과 저장
                            results.append(BatchResult(
                                request_id=request.id,
                                success=True,
                                result=intent_data,
                                processing_time=processing_time / len(batch_requests)
                            ))
                        else:
                            # 3-2-4: 매칭되지 않은 경우 기본 의도 데이터 생성
                            default_intent = {
                                "core_intent": "general_inquiry",
                                "intent_category": "일반문의",
                                "primary_action": "기타",
                                "target_object": "기타",
                                "intent_type": "일반문의",
                                "main_topic": "기타",
                                "specific_request": request.data.get('query', '')[:100],
                                "keywords": [request.data.get('query', '')[:20]],
                                "urgency": "medium",
                                "action_type": "기타"
                            }
                            results.append(BatchResult(
                                request_id=request.id,
                                success=True,
                                result=default_intent,
                                processing_time=processing_time / len(batch_requests)
                            ))
                            
                    except json.JSONDecodeError as e:
                        # 3-2-5: JSON 파싱 실패 처리
                        logging.error(f"의도 분석 결과 파싱 실패: {e}")
                        results.append(BatchResult(
                            request_id=request.id,
                            success=False,
                            error=f"결과 파싱 실패: {e}"
                        ))
                        
                # ===== 4단계: 성공 로깅 =====
                logging.info(f"의도 분석 배치 처리 완료: {len(queries)}개, {processing_time:.2f}s")
                
            else:
                # ===== OpenAI 클라이언트 미설정 오류 처리 =====
                for request in batch_requests:
                    results.append(BatchResult(
                        request_id=request.id,
                        success=False,
                        error="OpenAI 클라이언트가 설정되지 않음"
                    ))
                    
        except Exception as e:
            # ===== 예외 처리: 의도 분석 실패 =====
            logging.error(f"의도 분석 배치 처리 실패: {e}")
            for request in batch_requests:
                results.append(BatchResult(
                    request_id=request.id,
                    success=False,
                    error=str(e)
                ))
                
        # ===== 5단계: 의도 분석 결과 반환 =====
        return results

    # 텍스트 오타 수정 배치 처리 메서드 (GPT 기반 맞춤법 교정)
    # Args:
    #     batch_requests: 오타 수정 요청 배치 목록
    # Returns:
    #     List[BatchResult]: 오타 수정 결과 목록
    def _process_typo_correction_batch(self, batch_requests: List[BatchRequest]) -> List[BatchResult]:
        # ===== 결과 리스트 초기화 =====
        results = []
        
        try:
            # ===== 1단계: 텍스트 추출 및 준비 =====
            texts = [req.data.get('text', '') for req in batch_requests]
            start_time = time.time()
            
            # ===== 2단계: GPT 기반 배치 오타 수정 =====
            if hasattr(self, 'openai_client'):
                # 2-1: 여러 텍스트를 구분자로 연결 (한 번의 API 호출)
                combined_text = '\n---TEXT---\n'.join([f"T{i+1}: {t}" for i, t in enumerate(texts)])
                
                # 2-2: 한국어 맞춤법 교정 전문가 시스템 프롬프트
                system_prompt = """한국어 맞춤법 및 오타 교정 전문가입니다.
각 텍스트의 맞춤법과 오타만 수정하세요. 의미와 어조는 변경하지 마세요.

다음 형식으로 응답하세요:
T1: [수정된 텍스트1]
T2: [수정된 텍스트2]
...

---TEXT--- 로 구분된 각 텍스트를 순서대로 수정해주세요."""

                # 2-3: GPT API 호출 (배치 오타 수정)
                response = self.openai_client.chat.completions.create(
                    model='gpt-3.5-turbo',
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": combined_text}
                    ],
                    max_tokens=800,                       # 충분한 수정 결과 길이
                    temperature=0.1                       # 매우 보수적 (일관성 중시)
                )
                
                # 2-4: 수정 결과 추출 및 처리 시간 계산
                result_text = response.choices[0].message.content
                processing_time = time.time() - start_time
                
                # ===== 3단계: 결과 파싱 및 매핑 =====
                # 3-1: 정규식으로 각 텍스트의 수정 결과 추출
                pattern = r'T(\d+):\s*(.+?)(?=T\d+:|$)'
                matches = re.findall(pattern, result_text, re.DOTALL)
                
                # 3-2: 각 요청에 대한 결과 매핑
                for i, request in enumerate(batch_requests):
                    if i < len(matches):
                        # 3-2-1: 성공적 수정 결과 저장
                        corrected_text = matches[i][1].strip()
                        results.append(BatchResult(
                            request_id=request.id,
                            success=True,
                            result=corrected_text,
                            processing_time=processing_time / len(batch_requests)
                        ))
                    else:
                        # 3-2-2: 매핑되지 않은 경우 원본 반환 (안전장치)
                        results.append(BatchResult(
                            request_id=request.id,
                            success=True,
                            result=request.data.get('text', ''),
                            processing_time=processing_time / len(batch_requests)
                        ))
                        
                # ===== 4단계: 성공 로깅 =====
                logging.info(f"오타 수정 배치 처리 완료: {len(texts)}개, {processing_time:.2f}s")
                
            else:
                # ===== OpenAI 클라이언트 미설정 오류 처리 =====
                for request in batch_requests:
                    results.append(BatchResult(
                        request_id=request.id,
                        success=False,
                        error="OpenAI 클라이언트가 설정되지 않음"
                    ))
                    
        except Exception as e:
            # ===== 예외 처리: 오타 수정 실패 =====
            logging.error(f"오타 수정 배치 처리 실패: {e}")
            for request in batch_requests:
                results.append(BatchResult(
                    request_id=request.id,
                    success=False,
                    error=str(e)
                ))
                
        # ===== 5단계: 오타 수정 결과 반환 =====
        return results

    # 배치 처리 시스템 통계 조회 메서드
    # Returns:
    #     Dict[str, Any]: 상세 성능 통계 정보
    def get_stats(self) -> Dict[str, Any]:
        # ===== 종합 성능 통계 생성 =====
        return {
            'total_requests': self.stats['total_requests'],           # 총 처리 요청 수
            'batch_processed': self.stats['batch_processed'],         # 처리된 배치 수
            'cache_hits': self.stats['cache_hits'],                   # 캐시 히트 횟수
            'api_calls_saved': self.stats['api_calls_saved'],         # 절약된 API 호출 수
            'pending_requests': len(self.pending_requests),           # 현재 대기 중인 요청 수
            'queue_sizes': {                                          # 작업 유형별 큐 크기
                op_type: queue.qsize() 
                for op_type, queue in self.request_queues.items()
            },
            'average_batch_size': (                                   # 평균 배치 크기
                self.stats['total_requests'] / max(self.stats['batch_processed'], 1)
            )
        }

    # OpenAI 클라이언트 설정 메서드
    # Args:
    #     openai_client: OpenAI API 클라이언트 객체
    def set_openai_client(self, openai_client):
        # OpenAI 클라이언트를 배치 프로세서에 등록
        self.openai_client = openai_client
        logging.info("OpenAI 클라이언트가 배치 프로세서에 설정됨")
