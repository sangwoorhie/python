#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
최적화된 검색 서비스 모듈
- 캐싱, 배치 처리, 지능형 API 관리를 통합한 고성능 검색 시스템
- 다층 검색 및 의미론적 유사성 계산
- 실시간 성능 최적화 및 조기 종료 메커니즘
"""

import logging
import time
from typing import List, Dict, Set, Optional
from src.utils.memory_manager import memory_cleanup
from src.utils.text_preprocessor import TextPreprocessor
from src.utils.intelligent_api_manager import (
    IntelligentAPIManager, APICallRequest, APICallStrategy
)
from src.models.question_analyzer import QuestionAnalyzer
import numpy as np

# ===== 최적화된 Pinecone 벡터 검색을 담당하는 메인 클래스 =====
class OptimizedSearchService:
    
    # OptimizedSearchService 초기화
    # Args:
    #     pinecone_index: Pinecone 벡터 인덱스
    #     api_manager: 지능형 API 관리자
    def __init__(self, pinecone_index, api_manager: IntelligentAPIManager):
        self.index = pinecone_index                           # Pinecone 벡터 검색 인덱스
        self.api_manager = api_manager                        # 캐싱/배치 API 관리자
        self.text_processor = TextPreprocessor()              # 텍스트 전처리 도구
        
        # QuestionAnalyzer 초기화 (의도 관련성 계산용)
        self._question_analyzer = QuestionAnalyzer(api_manager.openai_client)
        
        # ===== 검색 최적화 설정 =====
        self.search_config = {
            'max_layers': 5,                                  # 최대 검색 레이어 수
            'adaptive_layer_count': True,                     # 동적 레이어 수 조정
            'early_termination': True,                        # 조기 종료 활성화
            'similarity_threshold': 0.75,                     # 유사도 임계값
            'min_results_threshold': 3,                       # 최소 결과 수 임계값
            'enable_result_caching': True,                    # 검색 결과 캐싱 활성화
            'cache_ttl_hours': 24                            # 캐시 유효시간 (시간)
        }
        
        # ===== 성능 최적화를 위한 캐시 시스템 =====
        self.embedding_cache = {}                            # 임베딩 재사용 캐시
        self.search_history = {}                             # 검색 기록 캐시
        
        logging.info("최적화된 검색 서비스 초기화 완료")

    # 최적화된 의미론적 다층 검색 - 메인 검색 메서드
    # Args:
    #     query: 검색할 사용자 질문
    #     top_k: 반환할 최대 결과 수
    #     lang: 언어 코드
    # Returns:
    #     List[Dict]: 검색된 유사 답변 리스트
    def search_similar_answers_optimized(self, query: str, top_k: int = 8, lang: str = 'ko') -> List[Dict]:
        try:
            # ===== 메모리 최적화 컨텍스트 시작 =====
            with memory_cleanup():
                search_start = time.time()
                logging.info(f"=== 최적화된 다층 검색 시작 ===")
                logging.info(f"원본 질문: {query}")
                
                # ===== 1단계: 검색 결과 캐시 확인 =====
                if self.search_config['enable_result_caching']:
                    cached_results = self._check_search_cache(query, {'top_k': top_k, 'lang': lang})
                    if cached_results:
                        logging.info(f"검색 결과 캐시 히트: {len(cached_results)}개 결과")
                        return cached_results
                
                # ===== 2단계: 기본 전처리 (캐싱 적용) =====
                processed_query = self._preprocess_with_caching(query, lang)
                
                # ===== 3단계: 핵심 의도 분석 (캐싱 적용) =====
                # 통합 분석기에서 이미 처리된 의도 분석을 사용하므로 개별 호출 불필요
                intent_analysis = self._get_default_intent_analysis(processed_query)
                
                # ===== 4단계: 검색 레이어 계획 수립 =====
                search_plan = self._create_search_plan(processed_query, intent_analysis)
                
                # ===== 5단계: 최적화된 다층 검색 실행 =====
                search_results = self._execute_optimized_search(search_plan, top_k)
                
                # ===== 6단계: 결과 후처리 및 점수 계산 =====
                final_results = self._postprocess_results(
                    search_results, processed_query, intent_analysis, top_k
                )
                
                # ===== 7단계: 검색 결과 캐싱 =====
                if self.search_config['enable_result_caching']:
                    self._cache_search_results(query, {'top_k': top_k, 'lang': lang}, final_results)
                
                # ===== 8단계: 검색 완료 및 성능 로깅 =====
                search_time = time.time() - search_start
                logging.info(f"✅ 캐시된 의도 분석 활용 검색 완료: {len(final_results)}개 결과, {search_time:.2f}s")
                
                # 🔍 최종 검색 결과 상세 로깅
                if final_results:
                    logging.info(f"🎯 캐시된 의도 분석 검색 결과:")
                    for i, result in enumerate(final_results[:3]):
                        score = result.get('adjusted_score', 0.0)
                        question = result.get('question', 'N/A')[:80]
                        answer = result.get('answer', 'N/A')[:80]
                        logging.info(f"   └── 결과 {i+1}: 점수={score:.3f}, 질문='{question}...', 답변='{answer}...'")
                
                return final_results
                
        except Exception as e:
            # ===== 예외 처리: 검색 실패시 빈 리스트 반환 =====
            logging.error(f"최적화된 검색 실패: {str(e)}")
            return []

    # 캐시된 의도 분석을 활용한 최적화된 검색 - API 호출 절약
    def search_similar_answers_with_cached_intent(self, query: str, cached_intent: Dict, top_k: int = 8, lang: str = 'ko') -> List[Dict]:
        try:
            # ===== 메모리 최적화 컨텍스트 시작 =====
            with memory_cleanup():
                search_start = time.time()
                logging.info(f"=== 캐시된 의도 분석 활용 검색 시작 ===")
                logging.info(f"원본 질문: {query}")
                logging.info(f"캐시된 의도: {cached_intent.get('core_intent', 'N/A')}")
                
                # ===== 1단계: 검색 결과 캐시 확인 =====
                if self.search_config['enable_result_caching']:
                    cached_results = self._check_search_cache(query, {'top_k': top_k, 'lang': lang})
                    if cached_results:
                        logging.info(f"검색 결과 캐시 히트: {len(cached_results)}개 결과")
                        return cached_results
                
                # ===== 2단계: 기본 전처리 (오타는 이미 수정됨) =====
                processed_query = self.text_processor.preprocess_text(query)
                
                # ===== 3단계: 캐시된 의도 분석 활용 (API 호출 생략!) =====
                intent_analysis = cached_intent  # 이미 수정된 의도 분석 사용
                logging.info(f"🔍 캐시된 의도 분석 활용: {intent_analysis.get('core_intent', 'N/A')}")
                
                # ===== 4단계: 검색 레이어 계획 수립 =====
                search_plan = self._create_search_plan(processed_query, intent_analysis)
                
                # ===== 5단계: 최적화된 다층 검색 실행 =====
                search_results = self._execute_optimized_search(search_plan, top_k)
                
                # ===== 6단계: 결과 후처리 및 점수 계산 =====
                final_results = self._postprocess_results(
                    search_results, processed_query, intent_analysis, top_k
                )
                
                # ===== 7단계: 검색 결과 캐싱 =====
                if self.search_config['enable_result_caching']:
                    self._cache_search_results(query, {'top_k': top_k, 'lang': lang}, final_results)
                
                # ===== 8단계: 검색 완료 및 성능 로깅 =====
                search_time = time.time() - search_start
                logging.info(f"🔍 캐시된 의도 활용 검색 완료: {len(final_results)}개 결과, {search_time:.2f}s")
                
                return final_results
                
        except Exception as e:
            # ===== 예외 처리: 검색 실패시 빈 리스트 반환 =====
            logging.error(f"캐시된 의도 활용 검색 실패: {str(e)}")
            return []

    # 캐싱 기반 텍스트 전처리 메서드
    # Args:
    #     query: 전처리할 질문 텍스트
    #     lang: 언어 코드
    # Returns:
    #     str: 전처리된 질문 텍스트
    def _preprocess_with_caching(self, query: str, lang: str) -> str:
        # ===== 기본 텍스트 전처리만 수행 =====
        # 오타 수정은 main_optimized_ai_generator.py의 통합 분석기에서 처리됨
        processed_query = self.text_processor.preprocess_text(query)
        return processed_query

    # 캐싱 기반 질문 의도 분석 메서드
    # Args:
    #     query: 의도를 분석할 질문
    # Returns:
    #     Dict: 분석된 의도 정보 (core_intent, 카테고리 등)
    def _get_default_intent_analysis(self, query: str) -> Dict:
        """기본 의도 분석 결과 반환 (통합 분석기 사용시 폴백용)"""
        return {
            "core_intent": "general_inquiry",
            "intent_category": "일반문의",
            "primary_action": "기타",
            "target_object": "기타",
            "standardized_query": query,
            "semantic_keywords": [query[:20]]
        }

    # 지능형 다층 검색 계획 수립 메서드
    # Args:
    #     query: 검색할 질문
    #     intent_analysis: 의도 분석 결과
    # Returns:
    #     Dict: 검색 계획 (레이어 구성, 타겟 결과 수 등)
    def _create_search_plan(self, query: str, intent_analysis: Dict) -> Dict:
        # ===== 1단계: 의도 분석 결과에서 핵심 정보 추출 =====
        core_intent = intent_analysis.get('core_intent', '')
        standardized_query = intent_analysis.get('standardized_query', query)
        semantic_keywords = intent_analysis.get('semantic_keywords', [])
        
        # ===== 2단계: 기존 개념 추출 (추가 분석) =====
        key_concepts = self.text_processor.extract_key_concepts(query)
        
        # ===== 3단계: 동적 레이어 개수 결정 =====
        if self.search_config['adaptive_layer_count']:
            layer_count = self._determine_optimal_layer_count(intent_analysis, key_concepts)
        else:
            layer_count = self.search_config['max_layers']
        
        # ===== 4단계: 검색 레이어 구성 (우선순위별) =====
        search_layers = []
        
        # Layer 1: 원본 질문 (필수 레이어 - 가장 높은 가중치)
        search_layers.append({
            'query': query,
            'weight': 1.0,  # 최고 가중치
            'type': 'original',
            'priority': 1
        })
        
        # Layer 2: 표준화된 의도 기반 질문 (GPT 분석 결과)
        if standardized_query and standardized_query != query:
            search_layers.append({
                'query': standardized_query,
                'weight': 0.95,                               # 높은 가중치
                'type': 'intent_based',
                'priority': 2
            })
        
        # Layer 3: 핵심 의도만 (추상화된 검색)
        if core_intent and layer_count >= 3:
            search_layers.append({
                'query': core_intent.replace('_', ' '),       # 언더스코어 제거
                'weight': 0.9,
                'type': 'core_intent',
                'priority': 3
            })
        
        # Layer 4: 의미론적 키워드 조합 (GPT 추출 키워드)
        if semantic_keywords and len(semantic_keywords) >= 2 and layer_count >= 4:
            semantic_query = ' '.join(semantic_keywords[:3]) # 상위 3개 키워드
            search_layers.append({
                'query': semantic_query,
                'weight': 0.8,
                'type': 'semantic_keywords',
                'priority': 4
            })
        
        # Layer 5: 기존 개념 기반 검색 (규칙 기반 키워드)
        if key_concepts and len(key_concepts) >= 2 and layer_count >= 5:
            concept_query = ' '.join(key_concepts[:3])       # 상위 3개 개념
            search_layers.append({
                'query': concept_query,
                'weight': 0.7,                               # 낮은 가중치
                'type': 'concept_based',
                'priority': 5
            })
        
        # ===== 5단계: 검색 계획 반환 =====
        return {
            'layers': search_layers,
            'target_results': self._calculate_target_results(len(search_layers)),
            'early_termination_enabled': self.search_config['early_termination']
        }

    # 질문 복잡도에 따른 최적 레이어 수 결정 메서드
    # Args:
    #     intent_analysis: 의도 분석 결과
    #     key_concepts: 추출된 핵심 개념 리스트
    # Returns:
    #     int: 최적 레이어 수
    def _determine_optimal_layer_count(self, intent_analysis: Dict, key_concepts: List) -> int:
        base_count = 2  # 기본 레이어 (원본 + 의도 기반)
        
        # ===== 복잡성 기반 추가 레이어 계산 =====
        complexity_score = 0
        
        # 의미론적 키워드 개수 평가
        semantic_keywords = intent_analysis.get('semantic_keywords', [])
        if len(semantic_keywords) >= 2:
            complexity_score += 1
        
        # 핵심 개념 개수 평가
        if len(key_concepts) >= 2:
            complexity_score += 1
        
        # 의도 카테고리 복잡도 평가 (복잡한 문의 유형)
        intent_category = intent_analysis.get('intent_category', '')
        if intent_category in ['개선/제안', '오류/장애']:
            complexity_score += 1
        
        # ===== 최종 레이어 수 결정 (최대값 제한) =====
        final_count = min(base_count + complexity_score, self.search_config['max_layers'])
        
        logging.debug(f"동적 레이어 계산: 기본={base_count}, 복잡도={complexity_score}, 최종={final_count}")
        
        return final_count

    # 레이어별 타겟 결과 수 계산 메서드
    # Args:
    #     layer_count: 검색 레이어 수
    # Returns:
    #     Dict[str, int]: 레이어별 타겟 결과 수
    def _calculate_target_results(self, layer_count: int) -> Dict[str, int]:
        base_results = 8
        
        return {
            'first_layer': base_results * 2,  # 첫 번째 레이어는 더 많이 검색
            'other_layers': base_results,     # 나머지 레이어는 기본 수량
            'total_unique': base_results * layer_count  # 전체 유니크 결과 목표
        }

    # 최적화된 다층 검색 실행 메서드 (핵심 검색 로직)
    # Args:
    #     search_plan: 검색 계획 (레이어 구성 정보)
    #     top_k: 최대 반환 결과 수
    # Returns:
    #     List[Dict]: 검색된 결과 리스트
    def _execute_optimized_search(self, search_plan: Dict, top_k: int) -> List[Dict]:
        layers = search_plan['layers']
        target_results = search_plan['target_results']
        
        all_results = []
        seen_ids = set()
        sufficient_results = False
        
        # 임베딩 요청 배치 준비
        embedding_requests = []
        
        for i, layer in enumerate(layers):
            search_query = layer['query']
            if not search_query or len(search_query.strip()) < 2:
                continue
            
            # 임베딩 캐시 확인
            if search_query in self.embedding_cache:
                layer['embedding'] = self.embedding_cache[search_query]
                logging.debug(f"임베딩 캐시 히트: 레이어 {i+1}")
            else:
                # 배치 요청에 추가
                embedding_requests.append({
                    'layer_index': i,
                    'query': search_query,
                    'priority': layer['priority']
                })
        
        # 필요한 임베딩 배치 생성
        if embedding_requests:
            self._generate_embeddings_batch(embedding_requests, layers)
        
        # 각 레이어별 검색 실행
        for i, layer in enumerate(layers):
            if 'embedding' not in layer:
                continue
            
            layer_type = layer['type']
            weight = layer['weight']
            
            logging.debug(f"레이어 {i+1} ({layer_type}) 검색 실행: {layer['query'][:50]}...")
            
            # 타겟 결과 수 결정
            if i == 0:
                search_top_k = target_results['first_layer']
            else:
                search_top_k = target_results['other_layers']
            
            # Pinecone 검색 실행
            try:
                results = self.index.query(
                    vector=layer['embedding'],
                    top_k=search_top_k,
                    include_metadata=True
                )
                
                # 🔍 벡터 DB 검색 결과 상세 로깅
                logging.info(f"🔍 벡터 DB 검색 결과 (레이어 {i+1}):")
                logging.info(f"   └── 검색 쿼리: '{layer['query'][:100]}...'")
                logging.info(f"   └── 검색 타입: {layer_type}")
                logging.info(f"   └── 가중치: {weight}")
                logging.info(f"   └── 요청된 결과 수: {search_top_k}")
                logging.info(f"   └── 실제 반환된 결과 수: {len(results.matches) if hasattr(results, 'matches') else 0}")
                
                # 상위 3개 결과 상세 로깅
                if hasattr(results, 'matches') and results.matches:
                    for j, match in enumerate(results.matches[:3]):
                        similarity_score = getattr(match, 'score', 0.0)
                        metadata = getattr(match, 'metadata', {})
                        question = metadata.get('question', 'N/A')[:100]
                        answer = metadata.get('answer', 'N/A')[:100]
                        logging.info(f"   └── 결과 {j+1}: 유사도={similarity_score:.3f}, 질문='{question}...', 답변='{answer}...'")
                
                # 결과 처리 및 가중치 적용
                layer_results = self._process_layer_results(
                    results, weight, layer_type, seen_ids
                )
                
                all_results.extend(layer_results)
                
                # 처리된 결과 로깅
                logging.info(f"   └── 처리된 결과 수: {len(layer_results)}")
                if layer_results:
                    for j, result in enumerate(layer_results[:3]):
                        logging.info(f"   └── 처리된 결과 {j+1}: 조정된 점수={result.get('adjusted_score', 0.0):.3f}, 질문='{result.get('question', 'N/A')[:50]}...'")
                
                # 조기 종료 조건 확인
                if (search_plan['early_termination_enabled'] and 
                    len(all_results) >= self.search_config['min_results_threshold'] and
                    self._check_early_termination_condition(all_results)):
                    logging.info(f"조기 종료: 레이어 {i+1}에서 충분한 결과 획득")
                    sufficient_results = True
                    break
                
            except Exception as e:
                logging.error(f"레이어 {i+1} 검색 실패: {e}")
                continue
        
        # 결과 정렬
        all_results.sort(key=lambda x: x['adjusted_score'], reverse=True)
        
        # 🔍 최종 검색 결과 상세 로깅
        logging.info(f"🎯 최종 검색 결과 요약:")
        logging.info(f"   └── 실행된 레이어 수: {len(layers)}")
        logging.info(f"   └── 총 검색된 결과 수: {len(all_results)}")
        logging.info(f"   └── 조기 종료 여부: {'예' if sufficient_results else '아니오'}")
        
        if all_results:
            logging.info(f"   └── 상위 3개 결과:")
            for i, result in enumerate(all_results[:3]):
                score = result.get('adjusted_score', 0.0)
                question = result.get('question', 'N/A')[:80]
                answer = result.get('answer', 'N/A')[:80]
                logging.info(f"      {i+1}. 점수={score:.3f}, 질문='{question}...', 답변='{answer}...'")
        
        logging.info(f"검색 실행 완료: {len(layers)}개 레이어, {len(all_results)}개 결과"
                    f"{', 조기종료' if sufficient_results else ''}")
        
        return all_results

    # 임베딩 배치 생성 메서드
    # Args:
    #     embedding_requests: 임베딩 요청 리스트
    #     layers: 검색 레이어 리스트
    def _generate_embeddings_batch(self, embedding_requests: List[Dict], layers: List[Dict]):
        if not embedding_requests:
            return
        
        # 배치 요청 생성
        batch_requests = []
        for req in embedding_requests:
            api_request = APICallRequest(
                operation='embedding',
                data={'text': req['query']},
                priority=req['priority'],
                strategy=APICallStrategy.BATCH_ONLY if len(embedding_requests) > 1 else APICallStrategy.CACHE_FIRST
            )
            batch_requests.append((req, api_request))
        
        # 배치 처리 또는 개별 처리
        if len(batch_requests) > 1:
            # 배치 처리
            logging.info(f"임베딩 배치 처리: {len(batch_requests)}개 요청")
            for req_info, api_request in batch_requests:
                response = self.api_manager.process_request(api_request)
                if response.success:
                    layer_index = req_info['layer_index']
                    embedding = response.data
                    layers[layer_index]['embedding'] = embedding
                    # 메모리 캐시에도 저장
                    self.embedding_cache[req_info['query']] = embedding
        else:
            # 단일 요청
            req_info, api_request = batch_requests[0]
            response = self.api_manager.process_request(api_request)
            if response.success:
                layer_index = req_info['layer_index']
                embedding = response.data
                layers[layer_index]['embedding'] = embedding
                self.embedding_cache[req_info['query']] = embedding

    # 레이어 검색 결과 처리 메서드
    # Args:
    #     results: Pinecone 검색 결과
    #     weight: 레이어 가중치
    #     layer_type: 레이어 타입
    #     seen_ids: 이미 본 ID 집합
    # Returns:
    #     List[Dict]: 처리된 결과 리스트
    def _process_layer_results(self, results: Dict, weight: float, layer_type: str, seen_ids: Set) -> List[Dict]:
        layer_results = []
        
        for match in results.get('matches', []):
            match_id = match['id']
            if match_id not in seen_ids:
                seen_ids.add(match_id)
                
                # 가중치 적용 점수 계산
                adjusted_score = match['score'] * weight
                
                processed_match = {
                    'id': match_id,
                    'score': match['score'],
                    'adjusted_score': adjusted_score,
                    'search_type': layer_type,
                    'layer_weight': weight,
                    'metadata': match.get('metadata', {})
                }
                
                layer_results.append(processed_match)
        
        return layer_results

    # 조기 종료 조건 확인 메서드
    # Args:
    #     results: 현재까지의 검색 결과
    # Returns:
    #     bool: 조기 종료 가능 여부
    def _check_early_termination_condition(self, results: List[Dict]) -> bool:
        if not results:
            return False
        
        # 상위 결과의 품질 확인
        top_results = results[:3]
        high_quality_count = sum(1 for r in top_results if r['adjusted_score'] >= self.search_config['similarity_threshold'])
        
        # 상위 3개 중 2개 이상이 고품질이면 조기 종료
        return high_quality_count >= 2

    # 검색 결과 후처리 및 최종 점수 계산 메서드
    # Args:
    #     search_results: 검색된 결과 리스트
    #     query: 원본 질문
    #     intent_analysis: 의도 분석 결과
    #     top_k: 최대 반환 결과 수
    # Returns:
    #     List[Dict]: 후처리된 최종 결과
    def _postprocess_results(self, search_results: List[Dict], query: str, 
                         intent_analysis: Dict, top_k: int) -> List[Dict]:
        # ===== 🔍 검색 결과 디버그 출력 =====
        print("\n" + "="*80)
        print(f"🔍 [SEARCH DEBUG] 검색 결과 후처리 시작: {len(search_results)}개 결과")
        print("="*80)
        for i, result in enumerate(search_results[:5]):  # 상위 5개만 출력
            question = result.get('metadata', {}).get('question', 'N/A')[:100]
            print(f"검색결과 #{i+1}: 점수={result['score']:.3f}, 질문={question}...")
        print("="*80)
        
        if not search_results:
            logging.info("🔍 [SEARCH DEBUG] 검색 결과가 없어서 빈 리스트 반환")
            return []
        
        # 벡터 유사도 기반 동적 임계값 계산
        scores = [r['score'] for r in search_results[:top_k*2]]
        if scores:
            # 상위 20%와 하위 20% 점수 차이로 동적 임계값 설정
            top_percentile = np.percentile(scores, 80)
            bottom_percentile = np.percentile(scores, 20)
            dynamic_threshold = (top_percentile + bottom_percentile) / 2
        else:
            dynamic_threshold = 0.5
        
        # 개념 관련성 계산을 위한 키워드 추출
        key_concepts = self.text_processor.extract_key_concepts(query)
        
        final_results = []
        
        logging.info(f"🔍 검색 결과 후처리 시작: {len(search_results)}개 결과")
        
        for i, match in enumerate(search_results[:top_k*2]):
            metadata = match.get('metadata', {})
            question = metadata.get('question', '')
            answer = metadata.get('answer', '')
            category = metadata.get('category', '일반')
            
            logging.info(f"🔍 결과 #{i+1} 처리 시작: 질문='{question[:50]}...'")
            
            # 기본 점수
            vector_score = match['score']
            adjusted_score = match['adjusted_score']
            
            # 의도 관련성 계산 (캐싱 적용)
            logging.info(f"🔍 의도 관련성 계산 시작: 질문='{question[:50]}...'")
            logging.info(f"🔍 사용자 의도 분석: {intent_analysis.get('core_intent', 'N/A')}")
            intent_relevance = self._calculate_intent_relevance_cached(
                intent_analysis, question, answer
            )
            logging.info(f"🔍 의도 관련성 계산 완료: {intent_relevance:.3f}")
            
            # 🔍 의도 관련성 계산 상세 로그
            logging.info(f"🔍 의도 관련성 계산:")
            logging.info(f"   └── 사용자 의도: {intent_analysis.get('core_intent', 'N/A')}")
            logging.info(f"   └── 기존 질문: {question[:80]}...")
            logging.info(f"   └── 의도 관련성 점수: {intent_relevance:.3f}")
            
            # 개념 관련성 계산
            concept_relevance = self._calculate_concept_relevance(
                query, key_concepts, question, answer
            )
            
            # 최종 점수 = 벡터 유사도(60%) + 의도 관련성(25%) + 개념 관련성(15%)
            final_score = (adjusted_score * 0.6 + 
                        intent_relevance * 0.25 + 
                        concept_relevance * 0.15)
            
            # 🔍 최종 점수 계산 상세 로그
            logging.info(f"🔍 최종 점수 계산:")
            logging.info(f"   └── 벡터 유사도: {adjusted_score:.3f} × 0.6 = {adjusted_score * 0.6:.3f}")
            logging.info(f"   └── 의도 관련성: {intent_relevance:.3f} × 0.25 = {intent_relevance * 0.25:.3f}")
            logging.info(f"   └── 개념 관련성: {concept_relevance:.3f} × 0.15 = {concept_relevance * 0.15:.3f}")
            logging.info(f"   └── 최종 점수: {final_score:.3f}")
            
            # === dynamic_threshold 활용 부분 추가 ===
            # 동적 임계값 사용: final_score가 아닌 vector_score에 적용
            min_threshold = dynamic_threshold if i >= 3 else 0.3  # 상위 3개는 더 낮은 임계값
            
            # 최소 임계값 또는 상위 순위 확인
            if final_score >= min_threshold or i < 3:
                print(f"✅ [SEARCH DEBUG] 결과 #{i+1} 선택됨: 최종점수={final_score:.3f} (임계값={min_threshold:.3f})")
                final_results.append({
                    'score': final_score,
                    'vector_score': vector_score,
                    'intent_relevance': intent_relevance,
                    'concept_relevance': concept_relevance,
                    'question': question,
                    'answer': answer,
                    'category': category,
                    'rank': i + 1,
                    'search_type': match['search_type'],
                    'layer_weight': match.get('layer_weight', 1.0),
                    'lang': 'ko'
                })
                
                logging.debug(f"선택: #{i+1} 최종점수={final_score:.3f} "
                            f"(벡터={vector_score:.3f}, 의도={intent_relevance:.3f}, "
                            f"개념={concept_relevance:.3f}) 타입={match['search_type']}")
            else:
                print(f"❌ [SEARCH DEBUG] 결과 #{i+1} 제외됨: 최종점수={final_score:.3f} < 임계값={min_threshold:.3f}")
            
            if len(final_results) >= top_k:
                break
    
        # ===== 🔍 최종 결과 요약 =====
        print("\n" + "="*80)
        print(f"🔍 [SEARCH DEBUG] 최종 선택된 결과: {len(final_results)}개")
        print("="*80)
        for i, result in enumerate(final_results):
            print(f"최종결과 #{i+1}: 점수={result['score']:.3f}, 질문={result['question'][:80]}...")
        print("="*80)
        
        # 디버그 파일에도 저장
        try:
            with open('/home/ec2-user/python/debug_search_results.txt', 'w', encoding='utf-8') as f:
                f.write(f"검색 질문: {query}\n")
                f.write("="*80 + "\n")
                f.write(f"최종 선택된 결과: {len(final_results)}개\n")
                f.write("="*80 + "\n")
                for i, result in enumerate(final_results):
                    f.write(f"\n결과 #{i+1}:\n")
                    f.write(f"점수: {result['score']:.3f}\n")
                    f.write(f"질문: {result['question']}\n")
                    f.write(f"답변: {result['answer'][:200]}...\n")
                    f.write("-" * 40 + "\n")
        except Exception as e:
            print(f"🔍 [DEBUG] 검색결과 파일 저장 실패: {e}")
    
        return final_results

    # 캐싱 기반 의도 관련성 계산 메서드
    # Args:
    #     query_intent: 질문의 의도 정보
    #     ref_question: 참조 질문
    #     ref_answer: 참조 답변
    # Returns:
    #     float: 의도 관련성 점수
    def _calculate_intent_relevance_cached(self, query_intent: Dict, ref_question: str, ref_answer: str) -> float:
        # 간단한 메모리 캐시 사용 (빠른 액세스)
        cache_key = f"{query_intent.get('core_intent', '')[:20]}:{ref_question[:30]}"
        
        if cache_key in self.search_history:
            return self.search_history[cache_key]
        
        # 의도 관련성 계산 (기존 로직)
        relevance = self._calculate_intent_similarity(query_intent, ref_question, ref_answer)
        
        # 캐시 저장 (최대 1000개)
        if len(self.search_history) > 1000:
            self.search_history.clear()
        self.search_history[cache_key] = relevance
        
        return relevance

    # 의도 유사성 계산 메서드 (기존 로직 유지)
    # Args:
    #     query_intent_analysis: 질문 의도 분석 결과
    #     ref_question: 참조 질문
    #     ref_answer: 참조 답변
    # Returns:
    #     float: 의도 유사성 점수
    def _calculate_intent_similarity(self, query_intent_analysis: dict, ref_question: str, ref_answer: str) -> float:
        try:
            # QuestionAnalyzer를 사용하되 캐싱 적용
            if not hasattr(self, '_question_analyzer'):
                logging.error("QuestionAnalyzer가 초기화되지 않음!")
                return 0.5
            
            logging.info(f"🔍 QuestionAnalyzer를 통한 의도 유사성 계산 시작")
            result = self._question_analyzer.calculate_intent_similarity(
                query_intent_analysis, ref_question, ref_answer
            )
            logging.info(f"🔍 QuestionAnalyzer 계산 결과: {result:.3f}")
            return result
        except Exception as e:
            logging.error(f"의도 유사성 계산 실패: {e}")
            return 0.3

    # 개념 관련성 계산 메서드 (기존 로직 유지)
    # Args:
    #     query: 원본 질문
    #     query_concepts: 질문에서 추출된 개념들
    #     ref_question: 참조 질문
    #     ref_answer: 참조 답변
    # Returns:
    #     float: 개념 관련성 점수
    def _calculate_concept_relevance(self, query: str, query_concepts: List, ref_question: str, ref_answer: str) -> float:
        if not query_concepts:
            return 0.5
        
        ref_concepts = self.text_processor.extract_key_concepts(ref_question + ' ' + ref_answer)
        
        if not ref_concepts:
            return 0.3
        
        matched_concepts = 0
        total_weight = 0
        
        for query_concept in query_concepts:
            concept_weight = len(query_concept) / 10.0
            total_weight += concept_weight
            
            if query_concept in ref_concepts:
                matched_concepts += concept_weight
                continue
            
            # 부분 일치 검사
            for ref_concept in ref_concepts:
                if len(query_concept) >= 3 and len(ref_concept) >= 3:
                    common_chars = set(query_concept) & set(ref_concept)
                    similarity = len(common_chars) / max(len(set(query_concept)), len(set(ref_concept)))
                    
                    if similarity >= 0.7:
                        matched_concepts += concept_weight * similarity
                        break
        
        relevance = matched_concepts / total_weight if total_weight > 0 else 0
        return min(relevance, 1.0)

    # 검색 결과 캐시 확인 메서드
    def _check_search_cache(self, query: str, search_params: Dict) -> Optional[List[Dict]]:
        return self.api_manager.cache_manager.get_search_results_cache(query, search_params)

    # 검색 결과 캐싱 메서드
    def _cache_search_results(self, query: str, search_params: Dict, results: List[Dict]):
        self.api_manager.cache_manager.set_search_results_cache(
            query, search_params, results, self.search_config['cache_ttl_hours']
        )

    # 컨텍스트 품질 분석 메서드 (기존 호환성 유지)
    def analyze_context_quality(self, similar_answers: list, query: str) -> dict:
        if not similar_answers:
            return {
                'has_good_context': False,
                'best_score': 0.0,
                'recommended_approach': 'fallback',
                'quality_level': 'none'
            }
        
        best_answer = similar_answers[0]
        best_score = best_answer['score']
        relevance_score = best_answer.get('intent_relevance', 0.5)
        
        high_quality_count = len([ans for ans in similar_answers if ans['score'] >= 0.7])
        good_relevance_count = len([ans for ans in similar_answers if ans.get('intent_relevance', 0) >= 0.6])
        
        # 접근 방식 결정
        if best_score >= 0.9 and relevance_score >= 0.7:
            approach = 'direct_use'
            quality_level = 'excellent'
        elif best_score >= 0.8 and relevance_score >= 0.6:
            approach = 'direct_use'
            quality_level = 'very_high'
        elif best_score >= 0.7 and relevance_score >= 0.5:
            approach = 'gpt_with_strong_context'
            quality_level = 'high'
        elif best_score >= 0.6 and (high_quality_count + good_relevance_count) >= 2:
            approach = 'gpt_with_strong_context'
            quality_level = 'medium'
        elif best_score >= 0.4 and relevance_score >= 0.4:
            approach = 'gpt_with_weak_context'
            quality_level = 'low'
        else:
            approach = 'fallback'
            quality_level = 'very_low'
        
        return {
            'has_good_context': quality_level in ['excellent', 'very_high', 'high', 'medium'],
            'best_score': best_score,
            'relevance_score': relevance_score,
            'high_quality_count': high_quality_count,
            'good_relevance_count': good_relevance_count,
            'recommended_approach': approach,
            'quality_level': quality_level,
            'context_summary': f"품질: {quality_level}, 점수: {best_score:.3f}, 관련성: {relevance_score:.3f}"
        }

    # 최적 폴백 답변 선택 메서드 (기존 호환성 유지)
    def get_best_fallback_answer(self, similar_answers: list, lang: str = 'ko') -> str:
        logging.info(f"=== get_best_fallback_answer 시작 ===")
        logging.info(f"입력된 similar_answers 개수: {len(similar_answers)}")
        
        if not similar_answers:
            logging.warning("similar_answers가 비어있음")
            return ""
        
        # 점수와 텍스트 품질을 종합 평가
        best_answer = ""
        best_score = 0
        
        for i, ans in enumerate(similar_answers[:3]):
            score = ans['score']
            answer_text = ans['answer']
            
            # 매우 높은 유사도면 바로 반환
            if score >= 0.9:
                logging.info(f"매우 높은 유사도({score:.3f}) - 원본 답변 바로 반환")
                return answer_text.strip()
            
            # 기본 정리
            answer_text = self.text_processor.preprocess_text(answer_text)
            
            # 영어 번역 (필요시 캐싱 적용)
            if lang == 'en' and ans.get('lang', 'ko') == 'ko':
                translation_request = APICallRequest(
                    operation='translation',
                    data={'text': answer_text, 'source_lang': 'ko', 'target_lang': 'en'},
                    priority=4,
                    strategy=APICallStrategy.CACHE_FIRST
                )
                translation_response = self.api_manager.process_request(translation_request)
                if translation_response.success:
                    answer_text = translation_response.data
            
            # 높은 유사도인 경우 첫 번째 답변 선택
            if score >= 0.8:
                logging.info(f"높은 유사도({score:.3f})로 답변 #{i+1} 직접 선택")
                return answer_text if answer_text else ans['answer'].strip()
            
            # 종합 점수 계산
            length_score = min(len(answer_text) / 200, 1.0)
            completeness_score = 1.0 if answer_text.endswith(('.', '!', '?')) else 0.8
            total_score = score * 0.8 + length_score * 0.1 + completeness_score * 0.1
            
            if total_score > best_score:
                best_score = total_score
                best_answer = answer_text
        
        # 안전장치
        if not best_answer and similar_answers:
            logging.error("최종 답변이 비어있음! 첫 번째 원본 답변 강제 반환")
            return similar_answers[0]['answer'].strip()
        
        return best_answer

    # 최적화 통계 조회 메서드
    def get_optimization_stats(self) -> Dict:
        return {
            'search_config': self.search_config,
            'embedding_cache_size': len(self.embedding_cache),
            'search_history_size': len(self.search_history),
            'api_manager_stats': self.api_manager.get_performance_stats()
        }

    # 캐시 지우기 메서드
    def clear_caches(self):
        self.embedding_cache.clear()
        self.search_history.clear()
        logging.info("검색 서비스 캐시 지워짐")

    # 검색 설정 업데이트 메서드
    def update_search_config(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.search_config:
                self.search_config[key] = value
                logging.info(f"검색 설정 업데이트: {key} = {value}")
