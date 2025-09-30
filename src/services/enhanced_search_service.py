#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
단순화된 Pinecone 검색 서비스
- Original Query만 사용하여 검색
- 가중치 개념 완전 제거
- 단일 임베딩 + 단일 검색 방식
"""

import logging
import time
from typing import List, Dict, Optional
from openai import OpenAI
import pinecone
from src.utils.memory_manager import memory_cleanup


class EnhancedPineconeSearchService:
    """Original Query 중심의 단순화된 Pinecone 벡터 검색 서비스"""
    
    def __init__(self, 
                 openai_client: OpenAI,
                 pinecone_index):
        """
        Args:
            openai_client: OpenAI 클라이언트
            pinecone_index: 이미 초기화된 Pinecone Index 객체
        """
        self.openai_client = openai_client
        self.embedding_model = "text-embedding-3-small"
        self.index = pinecone_index  # 기존 index 재사용
        self.pinecone_index_name = "bible-app-support-1536-openai"
        
    def search_by_enhanced_intent(self, 
                                   intent_analysis: Dict, 
                                   original_query: str,
                                   lang: str = 'ko',
                                   top_k: int = 3) -> List[Dict]:
        """
        Original Query만 사용한 단순화된 검색
        
        Args:
            intent_analysis: 통합 분석기에서 생성한 의도 분석 결과 (사용하지 않음)
            original_query: 오타 수정된 원본 쿼리 (이것만 사용)
            lang: 언어 코드
            top_k: 반환할 상위 결과 수
            
        Returns:
            List[Dict]: 유사도 순으로 정렬된 검색 결과
        """
        try:
            with memory_cleanup():
                logging.info(f"==================== Original Query Only Search 시작 ====================")
                logging.info(f"검색 쿼리: '{original_query}'")
                logging.info(f"언어: {lang}, 상위 결과 수: {top_k}")
                
                # 빈 쿼리 체크
                if not original_query or not original_query.strip():
                    logging.warning("검색 쿼리가 비어있음")
                    return []
                
                # 단일 검색 수행 (Original Query만 사용)
                search_results = self._perform_simple_search(
                    query=original_query,
                    top_k=top_k
                )
                
                # 결과에 메타데이터 추가
                final_results = self._add_metadata_to_results(
                    search_results, 
                    original_query,
                    intent_analysis  # 로깅 목적으로만 사용
                )
                
                logging.info(f"Original Query Only Search 완료: {len(final_results)}개 결과 반환")
                return final_results
                
        except Exception as e:
            logging.error(f"검색 실패: {str(e)}")
            logging.error(f"실패 상세 - 쿼리: '{original_query}', 오류 타입: {type(e).__name__}")
            return []
    
    def _perform_simple_search(self, 
                               query: str,
                               top_k: int) -> List[Dict]:
        """
        단순 검색 수행 (1번 임베딩 + 1번 Pinecone 검색)
        
        Args:
            query: 검색할 텍스트
            top_k: 반환할 결과 수
            
        Returns:
            List[Dict]: 검색 결과
        """
        try:
            # 1단계: 임베딩 생성
            embedding_start = time.time()
            embedding_response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=query
            )
            query_embedding = embedding_response.data[0].embedding
            embedding_time = time.time() - embedding_start
            logging.info(f"임베딩 생성 완료: {embedding_time:.3f}초")
            
            # 2단계: Pinecone 검색
            search_start = time.time()
            search_response = self.index.query(
                vector=query_embedding,
                top_k=top_k,  # 요청한 개수만큼만 검색
                include_metadata=True,
                include_values=False
            )
            search_time = time.time() - search_start
            logging.info(f"Pinecone 검색 완료: {search_time:.3f}초")
            
            # 3단계: 결과 처리
            results = []
            if search_response.matches:
                for i, match in enumerate(search_response.matches, 1):
                    result = {
                        'id': match.id,
                        'score': float(match.score),
                        'answer': match.metadata.get('answer', ''),
                        'category': match.metadata.get('category', ''),
                        'question': match.metadata.get('question', ''),
                        'name': match.metadata.get('name', ''),
                        'regdate': match.metadata.get('regdate', ''),
                        'seq': match.metadata.get('seq', ''),
                        'search_method': 'original_query_only',
                        'rank': i  # 순위 추가
                    }
                    results.append(result)
                    
                    # 상위 3개 결과 로깅
                    if i <= 3:
                        logging.info(f"검색결과 #{i}: id={result['id']}, "
                                f"score={result['score']:.4f}, "
                                f"category='{result['category']}', "
                                f"answer='{result['answer']}'") 
            
            logging.info(f"검색 완료 통계: "
                        f"결과 수={len(results)}, "
                        f"임베딩 시간={embedding_time:.3f}s, "
                        f"검색 시간={search_time:.3f}s, "
                        f"총 API 호출=2회 (임베딩 1회 + Pinecone 1회)")
            
            return results
            
        except Exception as e:
            logging.error(f"검색 수행 실패: {str(e)}")
            logging.error(f"쿼리: '{query}', 오류 타입: {type(e).__name__}")
            return []
    
    def _add_metadata_to_results(self, 
                                 search_results: List[Dict], 
                                 original_query: str,
                                 intent_analysis: Dict) -> List[Dict]:
        """
        검색 결과에 메타데이터 추가 (디버깅 및 로깅 목적)
        
        Args:
            search_results: 원본 검색 결과
            original_query: 사용된 검색 쿼리
            intent_analysis: 의도 분석 결과 (참조용)
            
        Returns:
            List[Dict]: 메타데이터가 추가된 검색 결과
        """
        try:
            if not search_results:
                return []
            
            # 결과에 추가 정보 부여
            enhanced_results = []
            for result in search_results:
                # 검색 컨텍스트 메타데이터 추가
                result['search_context'] = {
                    'used_query': original_query,
                    'query_length': len(original_query),
                    'search_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    # 의도 분석 정보는 참조용으로만 포함
                    'intent_info': {
                        'core_intent': intent_analysis.get('core_intent', 'N/A'),
                        'category': intent_analysis.get('intent_category', 'N/A')
                    } if intent_analysis else None
                }
                
                enhanced_results.append(result)
            
            # 유사도 점수 기준 정렬 (확실히 하기 위해)
            enhanced_results.sort(key=lambda x: x['score'], reverse=True)
            
            return enhanced_results
            
        except Exception as e:
            logging.error(f"메타데이터 추가 실패: {str(e)}")
            return search_results  # 실패시 원본 결과 반환
    
    def get_search_statistics(self) -> Dict:
        """검색 통계 정보 반환"""
        try:
            index_stats = self.index.describe_index_stats()
            return {
                'index_name': self.pinecone_index_name,
                'total_vectors': index_stats.get('total_vector_count', 0),
                'dimension': index_stats.get('dimension', 0),
                'embedding_model': self.embedding_model,
                'optimization_method': 'original_query_only',
                'api_calls_per_search': 2,
                'search_strategy': 'Single query vector search without weighting',
                'description': 'Simplified search using only the corrected original query'
            }
        except Exception as e:
            logging.error(f"검색 통계 조회 실패: {str(e)}")
            return {
                'error': str(e),
                'index_name': self.pinecone_index_name,
                'embedding_model': self.embedding_model
            }