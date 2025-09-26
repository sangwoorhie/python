#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
최적화된 Pinecone 검색 서비스
- 단일 API 호출로 멀티 구성요소 통합 검색
"""

import logging
import time
import json
from typing import List, Dict, Optional
from openai import OpenAI
import pinecone
from src.utils.memory_manager import memory_cleanup

# 퍼센테이지 기반 통합 검색 쿼리 생성
# 각 구성요소를 퍼센테이지 가중치에 따라 반복하여 단일 텍스트로 결합

# 가중치 분배:
# - Original Query: 60% (6번 반복) - 수정된 원본 텍스트가 가장 중요
# - Core Intent: 20% (2번 반복)
# - Semantic Keywords: 10% (1번 반복)
# - Intent Category: 5% (1번 반복)
# - Primary Action: 5% (1번 반복)

# Returns:
#     str: 퍼센테이지 가중치가 반영된 통합 검색 쿼리
class EnhancedPineconeSearchService:
    """퍼센테이지 기반 가중치를 사용한 최적화된 Pinecone 벡터 검색 서비스"""
    
    def __init__(self, 
                 openai_client: OpenAI,
                 pinecone_index
                 ):
        """
        Args:
            openai_client: OpenAI 클라이언트
            pinecone_index: 이미 초기화된 Pinecone Index 객체
        """
        self.openai_client = openai_client
        self.embedding_model = "text-embedding-3-small"
        self.index = pinecone_index  # 기존 index 재사용
        self.pinecone_index_name = "bible-app-support-1536-openai"
        
        # 검색 가중치 설정 (퍼센테이지 기반)
        self.text_weights = {
            'core_intent': 20,        # 핵심 의도 : 20% 가중치
            'intent_category': 5,    # 카테고리 : 5% 가중치
            'primary_action': 5,     # 주요 행동 : 5% 가중치  
            'semantic_keywords': 10,  # 의미론적 키워드 : 10% 가중치
            'original_query': 60       # 원본 쿼리 : 60% 가중치
        }
        
        # 반복 횟수 계산을 위한 스케일 팩터 (전체 반복 횟수 조절)
        self.repetition_scale = 10  # 60%→6번, 20%→2번, 10%→1번, 5%→0.5번(→1번) 반복
        
    # 퍼센테이지 기반 고도화된 의도 분석 검색 (단일 API 호출)
        
    # 가중치 분배:
    # - Core Intent: 70% (압도적 중요도)
    # - Intent Category: 10%
    # - Primary Action: 10%  
    # - Semantic Keywords: 10%
    # - Original Query: 0% (제외됨)
        
    # Args:
    #     intent_analysis: 통합 분석기에서 생성한 의도 분석 결과
    #     original_query: 원본 쿼리 (백업용, 실제로는 제외됨)
    #     lang: 언어 코드
    #     top_k: 반환할 상위 결과 수
            
    # Returns:
    #     List[Dict]: 유사도 순으로 정렬된 검색 결과 
    def search_by_enhanced_intent(self, 
                                intent_analysis: Dict, 
                                original_query: str,
                                lang: str = 'ko',
                                top_k: int = 3) -> List[Dict]:
        try:
            with memory_cleanup():
                logging.info(f"==================== Percentage-Based Weighted Search 시작 ====================")
                
                # 의도 분석 결과 추출
                core_intent = intent_analysis.get('core_intent', '').strip()
                intent_category = intent_analysis.get('intent_category', '').strip()
                primary_action = intent_analysis.get('primary_action', '').strip()
                semantic_keywords = intent_analysis.get('semantic_keywords', [])
                
                # 검색 구성요소 및 가중치 로깅
                logging.info(f"검색 구성요소 (퍼센테이지 가중치):")
                logging.info(f"  - Original Query (60%): '{original_query}'")  # 순서와 퍼센테이지 변경
                logging.info(f"  - Core Intent (20%): '{core_intent}'")        # 퍼센테이지 변경
                logging.info(f"  - Semantic Keywords (10%): {semantic_keywords}")
                logging.info(f"  - Intent Category (5%): '{intent_category}'") # 퍼센테이지 변경
                logging.info(f"  - Primary Action (5%): '{primary_action}'")   # 퍼센테이지 변경
                
                # 통합 검색 쿼리 생성
                unified_query = self._build_unified_search_query(
                    core_intent, intent_category, primary_action, 
                    semantic_keywords, original_query
                )
                
                logging.info(f"통합 검색 쿼리: '{unified_query}'")
                
                # 단일 검색 수행 (1번 임베딩 + 1번 Pinecone 검색)
                search_results = self._perform_single_optimized_search(
                    unified_query, top_k
                )
                
                # 결과 후처리
                final_results = self._enhance_search_results(
                    search_results, intent_analysis, unified_query
                )
                
                logging.info(f"Percentage-Based Weighted Search 완료: {len(final_results)}개 결과 반환, 가중치 최적화됨")
                return final_results
                
        except Exception as e:
            logging.error(f"Optimized search 실패: {str(e)}")
            # Fallback으로 원본 쿼리만 사용
            return self._fallback_search(original_query, top_k)
    
    #     퍼센테이지 기반 통합 검색 쿼리 생성
    #     각 구성요소를 퍼센테이지 가중치에 따라 반복하여 단일 텍스트로 결합
        
    #     가중치 분배:
    #     - Core Intent: 70% (7번 반복)
    #     - Intent Category: 10% (1번 반복) 
    #     - Primary Action: 10% (1번 반복)
    #     - Semantic Keywords: 10% (1번 반복)
    #     - Original Query: 0% (제외)
        
    #     Returns:
    #         str: 퍼센테이지 가중치가 반영된 통합 검색 쿼리
    def _build_unified_search_query(self, 
                                   core_intent: str,
                                   intent_category: str, 
                                   primary_action: str,
                                   semantic_keywords: List[str], 
                                   original_query: str) -> str:
        query_parts = []
        
        try:
            # 1. Original Query (60% 가중치 - 6번 반복) - 가장 중요하게 변경
            if original_query:
                repetitions = int(self.text_weights['original_query'] * self.repetition_scale / 100)
                query_parts.extend([original_query] * repetitions)
                logging.debug(f"Original Query 추가: '{original_query}' x{repetitions} (60%)")
            
            # 2. Core Intent (20% 가중치 - 2번 반복)
            if core_intent:
                repetitions = int(self.text_weights['core_intent'] * self.repetition_scale / 100)
                query_parts.extend([core_intent] * repetitions)
                logging.debug(f"Core Intent 추가: '{core_intent}' x{repetitions} (20%)")
            
            # 3. Semantic Keywords (10% 가중치 - 1번 반복)
            if semantic_keywords:
                keywords_text = " ".join(semantic_keywords)
                if keywords_text.strip():
                    repetitions = int(self.text_weights['semantic_keywords'] * self.repetition_scale / 100)
                    query_parts.extend([keywords_text] * repetitions)
                    logging.debug(f"Semantic Keywords 추가: '{keywords_text}' x{repetitions} (10%)")
            
            # 4. Intent Category (5% 가중치 - 1번 반복, 0.5→1로 반올림)
            if intent_category:
                repetitions = max(1, int(self.text_weights['intent_category'] * self.repetition_scale / 100))
                query_parts.extend([intent_category] * repetitions)
                logging.debug(f"Intent Category 추가: '{intent_category}' x{repetitions} (5%)")
            
            # 5. Primary Action (5% 가중치 - 1번 반복, 0.5→1로 반올림)
            if primary_action:
                repetitions = max(1, int(self.text_weights['primary_action'] * self.repetition_scale / 100))
                query_parts.extend([primary_action] * repetitions)
                logging.debug(f"Primary Action 추가: '{primary_action}' x{repetitions} (5%)")
                    
            # 통합 쿼리 생성 (공백으로 구분)
            unified_query = " ".join(query_parts)
            
            # 쿼리 길이 제한 (임베딩 모델의 토큰 한계 고려)
            if len(unified_query) > 2000:  # 대략적인 길이 제한
                unified_query = unified_query[:2000]
                logging.warning(f"쿼리 길이 초과로 잘림: {len(unified_query)} 글자")
            
            return unified_query
            
        except Exception as e:
            logging.error(f"통합 쿼리 생성 실패: {str(e)}")
            # 실패시 core_intent 우선, 없으면 원본 쿼리 반환
            if core_intent:
                return core_intent
            return original_query or "일반 문의"
    
    #     최적화된 단일 검색 수행 (1번 임베딩 + 1번 Pinecone 검색)
        
    #     Args:
    #         unified_query: 통합된 검색 쿼리
    #         top_k: 반환할 결과 수
            
    #     Returns:
    #         List[Dict]: 검색 결과
    def _perform_single_optimized_search(self, 
                                       unified_query: str,
                                       top_k: int) -> List[Dict]:
        try:
            if not unified_query or not unified_query.strip():
                logging.warning("빈 통합 쿼리")
                return []
                
            # 1단계: 단일 임베딩 생성
            embedding_start = time.time()
            embedding_response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=unified_query  # 통합 쿼리를 한 번에 임베딩
            )
            query_embedding = embedding_response.data[0].embedding
            embedding_time = time.time() - embedding_start
            
            # 2단계: 단일 Pinecone 검색
            search_start = time.time()
            search_response = self.index.query(
                vector=query_embedding,
                top_k=top_k * 2,  # 여유분 확보 (후처리에서 필터링)
                include_metadata=True,
                include_values=False
            )
            search_time = time.time() - search_start
            
            # 3단계: 결과 처리
            results = []
            if search_response.matches:
                for match in search_response.matches:
                    result = {
                        'id': match.id,
                        'score': float(match.score),
                        'answer': match.metadata.get('answer', ''),
                        'category': match.metadata.get('category', ''),
                        'question': match.metadata.get('question', ''),
                        'name': match.metadata.get('name', ''),
                        'regdate': match.metadata.get('regdate', ''),
                        'seq': match.metadata.get('seq', ''),
                        'search_method': 'unified_weighted'
                    }
                    results.append(result)
            
            logging.info(f"최적화된 검색 완료: {len(results)}개 결과, "
                        f"임베딩={embedding_time:.3f}s, 검색={search_time:.3f}s, "
                        f"총 API 호출=2회 (임베딩 1회 + 검색 1회)")
            
            return results
            
        except Exception as e:
            logging.error(f"최적화된 검색 실패: {str(e)}")
            return []
    
    # 검색 결과 후처리 및 품질 향상
        
    # Args:
    #     search_results: 원본 검색 결과
    #     intent_analysis: 의도 분석 결과
    #     unified_query: 사용된 통합 쿼리
            
    # Returns:
    #     List[Dict]: 후처리된 검색 결과
    def _enhance_search_results(self, 
                              search_results: List[Dict], 
                              intent_analysis: Dict,
                              unified_query: str) -> List[Dict]:
        try:
            if not search_results:
                return []
            
            # 결과에 추가 정보 부여
            enhanced_results = []
            for result in search_results:
                # 검색 품질 메타데이터 추가
                result['unified_query'] = unified_query[:100] + "..." if len(unified_query) > 100 else unified_query
                result['intent_components'] = {
                    'core_intent': intent_analysis.get('core_intent', ''),
                    'intent_category': intent_analysis.get('intent_category', ''),
                    'primary_action': intent_analysis.get('primary_action', ''),
                    'semantic_keywords_count': len(intent_analysis.get('semantic_keywords', []))
                }
                
                enhanced_results.append(result)
            
            # 유사도 점수 기준 정렬 (이미 Pinecone에서 정렬되어 오지만 확실히)
            enhanced_results.sort(key=lambda x: x['score'], reverse=True)
            
            # 상위 결과 로깅
            for i, result in enumerate(enhanced_results[:3], 1):
                logging.info(f"최종결과 #{i}: score={result['score']:.4f}, "
                           f"answer='{result['answer']}...'")
            
            return enhanced_results
            
        except Exception as e:
            logging.error(f"결과 후처리 실패: {str(e)}")
            return search_results  # 실패시 원본 결과 반환
    
    
    # Fallback용 단순 검색
        
    # Args:
    #     query: 검색 쿼리
    #     top_k: 반환할 결과 수
            
    # Returns:
    #     List[Dict]: 검색 결과
    def _fallback_search(self, query: str, top_k: int) -> List[Dict]:

        try:
            logging.warning("Enhanced search 실패, fallback 단순 검색 수행")
            
            if not query or not query.strip():
                return []
            
            # 단순 임베딩 + 검색
            embedding_response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=query
            )
            query_embedding = embedding_response.data[0].embedding
            
            search_response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                include_values=False
            )
            
            results = []
            if search_response.matches:
                for match in search_response.matches:
                    result = {
                        'id': match.id,
                        'score': float(match.score),
                        'answer': match.metadata.get('answer', ''),
                        'category': match.metadata.get('category', ''),
                        'question': match.metadata.get('question', ''),
                        'name': match.metadata.get('name', ''),
                        'regdate': match.metadata.get('regdate', ''),
                        'seq': match.metadata.get('seq', ''),
                        'search_method': 'fallback_simple'
                    }
                    results.append(result)
            
            logging.info(f"Fallback 검색 완료: {len(results)}개 결과")
            return results
            
        except Exception as e:
            logging.error(f"Fallback 검색도 실패: {str(e)}")
            return []
    
    def get_search_statistics(self) -> Dict:
        """검색 통계 정보 반환"""
        try:
            index_stats = self.index.describe_index_stats()
            return {
                'index_name': self.pinecone_index_name,
                'total_vectors': index_stats.get('total_vector_count', 0),
                'dimension': index_stats.get('dimension', 0),
                'embedding_model': self.embedding_model,
                'optimization_method': 'text_focused_search',
                'weight_percentages': self.text_weights,
                'repetition_scale': self.repetition_scale,
                'api_calls_per_search': 2,
                'weight_distribution': 'corrected_text:60%, core_intent:20%, keywords:10%, category/action:5% each'  # 분배 설명 업데이트
            }
        except Exception as e:
            logging.error(f"검색 통계 조회 실패: {str(e)}")
            return {}