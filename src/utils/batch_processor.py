#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
배치 처리 시스템
여러 API 호출을 효율적으로 배치 처리하여 성능 최적화
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from queue import Queue, Empty
import threading


@dataclass
class BatchRequest:
    """배치 요청 데이터 클래스"""
    id: str
    operation_type: str  # 'embedding', 'translation', 'intent_analysis' 등
    data: Dict[str, Any]
    callback: Optional[Callable] = None
    priority: int = 5  # 1(최고) ~ 10(최저)


@dataclass
class BatchResult:
    """배치 결과 데이터 클래스"""
    request_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    processing_time: float = 0.0


class BatchProcessor:
    """AI API 호출을 위한 지능형 배치 처리 시스템"""
    
    def __init__(self, max_workers: int = 5, batch_size: int = 10, batch_timeout: float = 2.0):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        # 작업 큐 (우선순위별)
        self.request_queues = {
            'embedding': Queue(),
            'translation': Queue(), 
            'intent_analysis': Queue(),
            'typo_correction': Queue()
        }
        
        # 결과 저장소
        self.results = {}
        self.pending_requests = {}
        
        # 스레드 풀
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 배치 처리 스레드
        self.batch_threads = {}
        self.running = False
        
        # 통계
        self.stats = {
            'total_requests': 0,
            'batch_processed': 0,
            'cache_hits': 0,
            'api_calls_saved': 0
        }
        
        logging.info(f"배치 프로세서 초기화: workers={max_workers}, batch_size={batch_size}, timeout={batch_timeout}s")

    def start(self):
        """배치 처리 시스템 시작"""
        if self.running:
            return
            
        self.running = True
        
        # 각 작업 유형별로 배치 처리 스레드 시작
        for operation_type in self.request_queues.keys():
            thread = threading.Thread(
                target=self._batch_worker,
                args=(operation_type,),
                daemon=True
            )
            thread.start()
            self.batch_threads[operation_type] = thread
            
        logging.info("배치 처리 시스템 시작됨")

    def stop(self):
        """배치 처리 시스템 중지"""
        self.running = False
        
        # 모든 스레드 종료 대기
        for thread in self.batch_threads.values():
            if thread.is_alive():
                thread.join(timeout=5)
                
        self.executor.shutdown(wait=True)
        logging.info("배치 처리 시스템 중지됨")

    def submit_request(self, request: BatchRequest) -> str:
        """배치 요청 제출"""
        if not self.running:
            self.start()
            
        self.stats['total_requests'] += 1
        self.pending_requests[request.id] = request
        
        # 우선순위에 따라 큐에 추가
        if request.operation_type in self.request_queues:
            self.request_queues[request.operation_type].put(request)
            logging.debug(f"배치 요청 제출: {request.id} ({request.operation_type})")
        else:
            logging.error(f"지원하지 않는 작업 유형: {request.operation_type}")
            
        return request.id

    def get_result(self, request_id: str, timeout: float = 30.0) -> Optional[BatchResult]:
        """배치 처리 결과 조회 (동기식)"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if request_id in self.results:
                result = self.results.pop(request_id)
                if request_id in self.pending_requests:
                    del self.pending_requests[request_id]
                return result
                
            time.sleep(0.1)  # 100ms 대기
            
        logging.warning(f"배치 결과 타임아웃: {request_id}")
        return None

    def _batch_worker(self, operation_type: str):
        """배치 작업자 스레드"""
        queue = self.request_queues[operation_type]
        
        while self.running:
            try:
                # 배치 수집
                batch_requests = self._collect_batch(queue, operation_type)
                
                if not batch_requests:
                    time.sleep(0.1)
                    continue
                
                # 배치 처리
                batch_results = self._process_batch(batch_requests, operation_type)
                
                # 결과 저장
                for result in batch_results:
                    self.results[result.request_id] = result
                    
                self.stats['batch_processed'] += 1
                logging.debug(f"배치 처리 완료: {operation_type}, {len(batch_requests)}개 요청")
                
            except Exception as e:
                logging.error(f"배치 작업자 오류 ({operation_type}): {e}")
                time.sleep(1)

    def _collect_batch(self, queue: Queue, operation_type: str) -> List[BatchRequest]:
        """배치 요청 수집"""
        batch = []
        start_time = time.time()
        
        # 첫 번째 요청 대기 (블로킹)
        try:
            first_request = queue.get(timeout=1.0)
            batch.append(first_request)
        except Empty:
            return batch
        
        # 추가 요청 수집 (논블로킹)
        while (len(batch) < self.batch_size and 
               time.time() - start_time < self.batch_timeout):
            try:
                request = queue.get_nowait()
                batch.append(request)
            except Empty:
                time.sleep(0.05)  # 50ms 대기
                
        # 우선순위 정렬 (낮은 숫자 = 높은 우선순위)
        batch.sort(key=lambda x: x.priority)
        
        return batch

    def _process_batch(self, batch_requests: List[BatchRequest], operation_type: str) -> List[BatchResult]:
        """배치 처리 실행"""
        results = []
        
        if operation_type == 'embedding':
            results = self._process_embedding_batch(batch_requests)
        elif operation_type == 'translation':
            results = self._process_translation_batch(batch_requests)
        elif operation_type == 'intent_analysis':
            results = self._process_intent_analysis_batch(batch_requests)
        elif operation_type == 'typo_correction':
            results = self._process_typo_correction_batch(batch_requests)
        else:
            # 지원하지 않는 작업 유형
            for request in batch_requests:
                results.append(BatchResult(
                    request_id=request.id,
                    success=False,
                    error=f"지원하지 않는 작업 유형: {operation_type}"
                ))
                
        return results

    def _process_embedding_batch(self, batch_requests: List[BatchRequest]) -> List[BatchResult]:
        """임베딩 배치 처리"""
        results = []
        
        # 텍스트 리스트 준비
        texts = []
        request_map = {}
        
        for i, request in enumerate(batch_requests):
            text = request.data.get('text', '')
            texts.append(text)
            request_map[i] = request
        
        try:
            start_time = time.time()
            
            # OpenAI API 배치 호출 (실제 구현에서는 OpenAI 클라이언트 주입)
            if hasattr(self, 'openai_client'):
                response = self.openai_client.embeddings.create(
                    model='text-embedding-3-small',
                    input=texts
                )
                
                embeddings = [item.embedding for item in response.data]
                processing_time = time.time() - start_time
                
                # 결과 매핑
                for i, embedding in enumerate(embeddings):
                    request = request_map[i]
                    results.append(BatchResult(
                        request_id=request.id,
                        success=True,
                        result=embedding,
                        processing_time=processing_time / len(embeddings)
                    ))
                    
                logging.info(f"임베딩 배치 처리 완료: {len(texts)}개, {processing_time:.2f}s")
                
            else:
                # OpenAI 클라이언트가 없는 경우 개별 처리
                for request in batch_requests:
                    results.append(BatchResult(
                        request_id=request.id,
                        success=False,
                        error="OpenAI 클라이언트가 설정되지 않음"
                    ))
                    
        except Exception as e:
            logging.error(f"임베딩 배치 처리 실패: {e}")
            # 모든 요청을 실패로 처리
            for request in batch_requests:
                results.append(BatchResult(
                    request_id=request.id,
                    success=False,
                    error=str(e)
                ))
                
        return results

    def _process_translation_batch(self, batch_requests: List[BatchRequest]) -> List[BatchResult]:
        """번역 배치 처리"""
        results = []
        
        # 언어쌍별로 그룹화
        language_groups = {}
        
        for request in batch_requests:
            source_lang = request.data.get('source_lang', 'ko')
            target_lang = request.data.get('target_lang', 'en')
            lang_pair = f"{source_lang}-{target_lang}"
            
            if lang_pair not in language_groups:
                language_groups[lang_pair] = []
            language_groups[lang_pair].append(request)
        
        # 언어쌍별로 배치 처리
        for lang_pair, group_requests in language_groups.items():
            try:
                source_lang, target_lang = lang_pair.split('-')
                texts = [req.data.get('text', '') for req in group_requests]
                
                start_time = time.time()
                
                # 실제 번역 API 호출 (GPT 활용)
                if hasattr(self, 'openai_client'):
                    # 여러 텍스트를 한 번에 번역
                    combined_text = '\n---SEPARATOR---\n'.join(texts)
                    
                    system_prompt = f"Translate the following texts from {source_lang} to {target_lang}. Separate each translation with '---SEPARATOR---'."
                    
                    response = self.openai_client.chat.completions.create(
                        model='gpt-3.5-turbo',
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": combined_text}
                        ],
                        max_tokens=1000,
                        temperature=0.3
                    )
                    
                    translated_combined = response.choices[0].message.content
                    translated_texts = translated_combined.split('---SEPARATOR---')
                    
                    processing_time = time.time() - start_time
                    
                    # 결과 매핑
                    for i, request in enumerate(group_requests):
                        if i < len(translated_texts):
                            results.append(BatchResult(
                                request_id=request.id,
                                success=True,
                                result=translated_texts[i].strip(),
                                processing_time=processing_time / len(group_requests)
                            ))
                        else:
                            results.append(BatchResult(
                                request_id=request.id,
                                success=False,
                                error="번역 결과 매핑 실패"
                            ))
                            
                    logging.info(f"번역 배치 처리 완료: {lang_pair}, {len(texts)}개, {processing_time:.2f}s")
                    
                else:
                    # OpenAI 클라이언트가 없는 경우
                    for request in group_requests:
                        results.append(BatchResult(
                            request_id=request.id,
                            success=False,
                            error="OpenAI 클라이언트가 설정되지 않음"
                        ))
                        
            except Exception as e:
                logging.error(f"번역 배치 처리 실패 ({lang_pair}): {e}")
                for request in group_requests:
                    results.append(BatchResult(
                        request_id=request.id,
                        success=False,
                        error=str(e)
                    ))
                    
        return results

    def _process_intent_analysis_batch(self, batch_requests: List[BatchRequest]) -> List[BatchResult]:
        """의도 분석 배치 처리"""
        results = []
        
        try:
            queries = [req.data.get('query', '') for req in batch_requests]
            start_time = time.time()
            
            if hasattr(self, 'openai_client'):
                # 여러 질문을 한 번에 분석
                combined_queries = '\n---QUERY---\n'.join([f"Q{i+1}: {q}" for i, q in enumerate(queries)])
                
                system_prompt = """당신은 바이블 앱 문의 분석 전문가입니다. 
다음 질문들을 각각 분석하여 의도를 파악하세요.

각 질문에 대해 다음 JSON 형태로 응답하세요:
Q1: {"core_intent": "...", "intent_category": "...", "primary_action": "...", "target_object": "..."}
Q2: {"core_intent": "...", "intent_category": "...", "primary_action": "...", "target_object": "..."}
...

---QUERY--- 로 구분된 각 질문을 순서대로 분석해주세요."""

                response = self.openai_client.chat.completions.create(
                    model='gpt-3.5-turbo',
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": combined_queries}
                    ],
                    max_tokens=800,
                    temperature=0.2
                )
                
                result_text = response.choices[0].message.content
                processing_time = time.time() - start_time
                
                # 결과 파싱
                import re
                import json
                
                pattern = r'Q(\d+):\s*(\{[^}]+\})'
                matches = re.findall(pattern, result_text)
                
                for i, request in enumerate(batch_requests):
                    try:
                        if i < len(matches):
                            intent_data = json.loads(matches[i][1])
                            # 기존 호환성 필드 추가
                            intent_data.update({
                                'intent_type': intent_data.get('intent_category', '일반문의'),
                                'main_topic': intent_data.get('target_object', '기타'),
                                'specific_request': request.data.get('query', '')[:100],
                                'keywords': [request.data.get('query', '')[:20]],
                                'urgency': 'medium',
                                'action_type': intent_data.get('primary_action', '기타')
                            })
                            
                            results.append(BatchResult(
                                request_id=request.id,
                                success=True,
                                result=intent_data,
                                processing_time=processing_time / len(batch_requests)
                            ))
                        else:
                            # 매칭되지 않은 경우 기본값
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
                        logging.error(f"의도 분석 결과 파싱 실패: {e}")
                        results.append(BatchResult(
                            request_id=request.id,
                            success=False,
                            error=f"결과 파싱 실패: {e}"
                        ))
                        
                logging.info(f"의도 분석 배치 처리 완료: {len(queries)}개, {processing_time:.2f}s")
                
            else:
                for request in batch_requests:
                    results.append(BatchResult(
                        request_id=request.id,
                        success=False,
                        error="OpenAI 클라이언트가 설정되지 않음"
                    ))
                    
        except Exception as e:
            logging.error(f"의도 분석 배치 처리 실패: {e}")
            for request in batch_requests:
                results.append(BatchResult(
                    request_id=request.id,
                    success=False,
                    error=str(e)
                ))
                
        return results

    def _process_typo_correction_batch(self, batch_requests: List[BatchRequest]) -> List[BatchResult]:
        """오타 수정 배치 처리"""
        results = []
        
        try:
            texts = [req.data.get('text', '') for req in batch_requests]
            start_time = time.time()
            
            if hasattr(self, 'openai_client'):
                # 여러 텍스트를 한 번에 오타 수정
                combined_text = '\n---TEXT---\n'.join([f"T{i+1}: {t}" for i, t in enumerate(texts)])
                
                system_prompt = """한국어 맞춤법 및 오타 교정 전문가입니다.
각 텍스트의 맞춤법과 오타만 수정하세요. 의미와 어조는 변경하지 마세요.

다음 형식으로 응답하세요:
T1: [수정된 텍스트1]
T2: [수정된 텍스트2]
...

---TEXT--- 로 구분된 각 텍스트를 순서대로 수정해주세요."""

                response = self.openai_client.chat.completions.create(
                    model='gpt-3.5-turbo',
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": combined_text}
                    ],
                    max_tokens=800,
                    temperature=0.1
                )
                
                result_text = response.choices[0].message.content
                processing_time = time.time() - start_time
                
                # 결과 파싱
                import re
                pattern = r'T(\d+):\s*(.+?)(?=T\d+:|$)'
                matches = re.findall(pattern, result_text, re.DOTALL)
                
                for i, request in enumerate(batch_requests):
                    if i < len(matches):
                        corrected_text = matches[i][1].strip()
                        results.append(BatchResult(
                            request_id=request.id,
                            success=True,
                            result=corrected_text,
                            processing_time=processing_time / len(batch_requests)
                        ))
                    else:
                        # 매칭되지 않은 경우 원본 반환
                        results.append(BatchResult(
                            request_id=request.id,
                            success=True,
                            result=request.data.get('text', ''),
                            processing_time=processing_time / len(batch_requests)
                        ))
                        
                logging.info(f"오타 수정 배치 처리 완료: {len(texts)}개, {processing_time:.2f}s")
                
            else:
                for request in batch_requests:
                    results.append(BatchResult(
                        request_id=request.id,
                        success=False,
                        error="OpenAI 클라이언트가 설정되지 않음"
                    ))
                    
        except Exception as e:
            logging.error(f"오타 수정 배치 처리 실패: {e}")
            for request in batch_requests:
                results.append(BatchResult(
                    request_id=request.id,
                    success=False,
                    error=str(e)
                ))
                
        return results

    def get_stats(self) -> Dict[str, Any]:
        """배치 처리 통계 조회"""
        return {
            'total_requests': self.stats['total_requests'],
            'batch_processed': self.stats['batch_processed'],
            'cache_hits': self.stats['cache_hits'],
            'api_calls_saved': self.stats['api_calls_saved'],
            'pending_requests': len(self.pending_requests),
            'queue_sizes': {
                op_type: queue.qsize() 
                for op_type, queue in self.request_queues.items()
            },
            'average_batch_size': (
                self.stats['total_requests'] / max(self.stats['batch_processed'], 1)
            )
        }

    def set_openai_client(self, openai_client):
        """OpenAI 클라이언트 설정"""
        self.openai_client = openai_client
        logging.info("OpenAI 클라이언트가 배치 프로세서에 설정됨")
