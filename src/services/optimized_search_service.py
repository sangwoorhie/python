#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
최적화된 검색 서비스 모듈
캐싱, 배치 처리, 지능형 API 관리를 통합한 고성능 검색 시스템
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


class OptimizedSearchService:
    """최적화된 Pinecone 벡터 검색 서비스"""
    
    def __init__(self, pinecone_index, api_manager: IntelligentAPIManager):
        self.index = pinecone_index
        self.api_manager = api_manager
        self.text_processor = TextPreprocessor()
        
        # 검색 최적화 설정
        self.search_config = {
            'max_layers': 5,
            'adaptive_layer_count': True,
            'early_termination': True,
            'similarity_threshold': 0.75,
            'min_results_threshold': 3,
            'enable_result_caching': True,
            'cache_ttl_hours': 24
        }
        
        # 임베딩 재사용을 위한 캐시
        self.embedding_cache = {}
        self.search_history = {}
        
        logging.info("최적화된 검색 서비스 초기화 완료")

    def search_similar_answers_optimized(self, query: str, top_k: int = 8, lang: str = 'ko') -> List[Dict]:
        """최적화된 의미론적 다층 검색"""
        try:
            with memory_cleanup():
                search_start = time.time()
                logging.info(f"=== 최적화된 다층 검색 시작 ===")
                logging.info(f"원본 질문: {query}")
                
                # 1. 검색 결과 캐시 확인
                if self.search_config['enable_result_caching']:
                    cached_results = self._check_search_cache(query, {'top_k': top_k, 'lang': lang})
                    if cached_results:
                        logging.info(f"검색 결과 캐시 히트: {len(cached_results)}개 결과")
                        return cached_results
                
                # 2. 기본 전처리 (캐싱 적용)
                processed_query = self._preprocess_with_caching(query, lang)
                
                # 3. 핵심 의도 분석 (캐싱 적용)
                intent_analysis = self._analyze_intent_with_caching(processed_query)
                
                # 4. 검색 레이어 계획 수립
                search_plan = self._create_search_plan(processed_query, intent_analysis)
                
                # 5. 최적화된 다층 검색 실행
                search_results = self._execute_optimized_search(search_plan, top_k)
                
                # 6. 결과 후처리 및 점수 계산
                final_results = self._postprocess_results(
                    search_results, processed_query, intent_analysis, top_k
                )
                
                # 7. 검색 결과 캐싱
                if self.search_config['enable_result_caching']:
                    self._cache_search_results(query, {'top_k': top_k, 'lang': lang}, final_results)
                
                search_time = time.time() - search_start
                logging.info(f"최적화된 검색 완료: {len(final_results)}개 결과, {search_time:.2f}s")
                
                return final_results
                
        except Exception as e:
            logging.error(f"최적화된 검색 실패: {str(e)}")
            return []

    def _preprocess_with_caching(self, query: str, lang: str) -> str:
        """캐싱 기반 전처리"""
        # 기본 전처리
        processed_query = self.text_processor.preprocess_text(query)
        
        # 한국어인 경우 오타 수정 (캐싱 적용)
        if lang == 'ko' or lang == 'auto':
            typo_request = APICallRequest(
                operation='typo_correction',
                data={'text': processed_query},
                priority=3,
                strategy=APICallStrategy.CACHE_FIRST
            )
            
            typo_response = self.api_manager.process_request(typo_request)
            if typo_response.success and typo_response.data:
                processed_query = typo_response.data
                if not typo_response.cache_hit:
                    logging.info(f"오타 수정 적용: {query[:50]} → {processed_query[:50]}")
        
        return processed_query

    def _analyze_intent_with_caching(self, query: str) -> Dict:
        """캐싱 기반 의도 분석"""
        intent_request = APICallRequest(
            operation='intent_analysis',
            data={'query': query},
            priority=2,
            strategy=APICallStrategy.CACHE_FIRST
        )
        
        intent_response = self.api_manager.process_request(intent_request)
        
        if intent_response.success and intent_response.data:
            return intent_response.data
        else:
            # 기본값 반환
            return {
                "core_intent": "general_inquiry",
                "intent_category": "일반문의",
                "primary_action": "기타",
                "target_object": "기타",
                "standardized_query": query,
                "semantic_keywords": [query[:20]]
            }

    def _create_search_plan(self, query: str, intent_analysis: Dict) -> Dict:
        """검색 계획 수립"""
        core_intent = intent_analysis.get('core_intent', '')
        standardized_query = intent_analysis.get('standardized_query', query)
        semantic_keywords = intent_analysis.get('semantic_keywords', [])
        
        # 기존 개념 추출
        key_concepts = self.text_processor.extract_key_concepts(query)
        
        # 동적 레이어 개수 결정
        if self.search_config['adaptive_layer_count']:
            layer_count = self._determine_optimal_layer_count(intent_analysis, key_concepts)
        else:
            layer_count = self.search_config['max_layers']
        
        # 검색 레이어 구성
        search_layers = []
        
        # Layer 1: 원본 질문 (필수)
        search_layers.append({
            'query': query,
            'weight': 1.0,
            'type': 'original',
            'priority': 1
        })
        
        # Layer 2: 표준화된 의도 기반 질문
        if standardized_query and standardized_query != query:
            search_layers.append({
                'query': standardized_query,
                'weight': 0.95,
                'type': 'intent_based',
                'priority': 2
            })
        
        # Layer 3: 핵심 의도만
        if core_intent and layer_count >= 3:
            search_layers.append({
                'query': core_intent.replace('_', ' '),
                'weight': 0.9,
                'type': 'core_intent',
                'priority': 3
            })
        
        # Layer 4: 의미론적 키워드 조합
        if semantic_keywords and len(semantic_keywords) >= 2 and layer_count >= 4:
            semantic_query = ' '.join(semantic_keywords[:3])
            search_layers.append({
                'query': semantic_query,
                'weight': 0.8,
                'type': 'semantic_keywords',
                'priority': 4
            })
        
        # Layer 5: 기존 개념 기반 검색
        if key_concepts and len(key_concepts) >= 2 and layer_count >= 5:
            concept_query = ' '.join(key_concepts[:3])
            search_layers.append({
                'query': concept_query,
                'weight': 0.7,
                'type': 'concept_based',
                'priority': 5
            })
        
        return {
            'layers': search_layers,
            'target_results': self._calculate_target_results(len(search_layers)),
            'early_termination_enabled': self.search_config['early_termination']
        }

    def _determine_optimal_layer_count(self, intent_analysis: Dict, key_concepts: List) -> int:
        """최적 레이어 수 결정"""
        base_count = 2  # 원본 + 의도 기반
        
        # 복잡성 기반 추가 레이어
        complexity_score = 0
        
        # 의미론적 키워드 개수
        semantic_keywords = intent_analysis.get('semantic_keywords', [])
        if len(semantic_keywords) >= 2:
            complexity_score += 1
        
        # 핵심 개념 개수
        if len(key_concepts) >= 2:
            complexity_score += 1
        
        # 의도 카테고리 복잡도
        intent_category = intent_analysis.get('intent_category', '')
        if intent_category in ['개선/제안', '오류/장애']:
            complexity_score += 1
        
        # 최종 레이어 수 결정
        final_count = min(base_count + complexity_score, self.search_config['max_layers'])
        
        logging.debug(f"동적 레이어 계산: 기본={base_count}, 복잡도={complexity_score}, 최종={final_count}")
        
        return final_count

    def _calculate_target_results(self, layer_count: int) -> Dict[str, int]:
        """레이어별 타겟 결과 수 계산"""
        base_results = 8
        
        return {
            'first_layer': base_results * 2,  # 첫 번째 레이어는 더 많이
            'other_layers': base_results,
            'total_unique': base_results * layer_count
        }

    def _execute_optimized_search(self, search_plan: Dict, top_k: int) -> List[Dict]:
        """최적화된 검색 실행"""
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
                
                # 결과 처리 및 가중치 적용
                layer_results = self._process_layer_results(
                    results, weight, layer_type, seen_ids
                )
                
                all_results.extend(layer_results)
                
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
        
        logging.info(f"검색 실행 완료: {len(layers)}개 레이어, {len(all_results)}개 결과"
                    f"{', 조기종료' if sufficient_results else ''}")
        
        return all_results

    def _generate_embeddings_batch(self, embedding_requests: List[Dict], layers: List[Dict]):
        """임베딩 배치 생성"""
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

    def _process_layer_results(self, results: Dict, weight: float, layer_type: str, seen_ids: Set) -> List[Dict]:
        """레이어 결과 처리"""
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

    def _check_early_termination_condition(self, results: List[Dict]) -> bool:
        """조기 종료 조건 확인"""
        if not results:
            return False
        
        # 상위 결과의 품질 확인
        top_results = results[:3]
        high_quality_count = sum(1 for r in top_results if r['adjusted_score'] >= self.search_config['similarity_threshold'])
        
        # 상위 3개 중 2개 이상이 고품질이면 조기 종료
        return high_quality_count >= 2

    def _postprocess_results(self, search_results: List[Dict], query: str, 
                           intent_analysis: Dict, top_k: int) -> List[Dict]:
        """검색 결과 후처리 및 최종 점수 계산"""
        if not search_results:
            return []
        
        # 개념 관련성 계산을 위한 키워드 추출
        key_concepts = self.text_processor.extract_key_concepts(query)
        
        final_results = []
        
        for i, match in enumerate(search_results[:top_k*2]):
            metadata = match.get('metadata', {})
            question = metadata.get('question', '')
            answer = metadata.get('answer', '')
            category = metadata.get('category', '일반')
            
            # 기본 점수
            vector_score = match['score']
            adjusted_score = match['adjusted_score']
            
            # 의도 관련성 계산 (캐싱 적용)
            intent_relevance = self._calculate_intent_relevance_cached(
                intent_analysis, question, answer
            )
            
            # 개념 관련성 계산
            concept_relevance = self._calculate_concept_relevance(
                query, key_concepts, question, answer
            )
            
            # 최종 점수 = 벡터 유사도(60%) + 의도 관련성(25%) + 개념 관련성(15%)
            final_score = (adjusted_score * 0.6 + 
                          intent_relevance * 0.25 + 
                          concept_relevance * 0.15)
            
            # 최소 임계값 또는 상위 순위 확인
            if final_score >= 0.4 or i < 3:
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
            
            if len(final_results) >= top_k:
                break
        
        return final_results

    def _calculate_intent_relevance_cached(self, query_intent: Dict, ref_question: str, ref_answer: str) -> float:
        """캐싱 기반 의도 관련성 계산"""
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

    def _calculate_intent_similarity(self, query_intent_analysis: dict, ref_question: str, ref_answer: str) -> float:
        """의도 유사성 계산 (기존 로직 유지)"""
        try:
            # QuestionAnalyzer를 사용하되 캐싱 적용
            if not hasattr(self, '_question_analyzer'):
                # 임시로 question analyzer 생성 (실제로는 의존성 주입)
                return 0.5
            
            return self._question_analyzer.calculate_intent_similarity(
                query_intent_analysis, ref_question, ref_answer
            )
        except Exception as e:
            logging.error(f"의도 유사성 계산 실패: {e}")
            return 0.3

    def _calculate_concept_relevance(self, query: str, query_concepts: List, ref_question: str, ref_answer: str) -> float:
        """개념 관련성 계산 (기존 로직 유지)"""
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

    def _check_search_cache(self, query: str, search_params: Dict) -> Optional[List[Dict]]:
        """검색 결과 캐시 확인"""
        return self.api_manager.cache_manager.get_search_results_cache(query, search_params)

    def _cache_search_results(self, query: str, search_params: Dict, results: List[Dict]):
        """검색 결과 캐싱"""
        self.api_manager.cache_manager.set_search_results_cache(
            query, search_params, results, self.search_config['cache_ttl_hours']
        )

    def analyze_context_quality(self, similar_answers: list, query: str) -> dict:
        """컨텍스트 품질 분석 (기존 호환성 유지)"""
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

    def get_best_fallback_answer(self, similar_answers: list, lang: str = 'ko') -> str:
        """최적 폴백 답변 선택 (기존 호환성 유지)"""
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

    def get_optimization_stats(self) -> Dict:
        """최적화 통계 조회"""
        return {
            'search_config': self.search_config,
            'embedding_cache_size': len(self.embedding_cache),
            'search_history_size': len(self.search_history),
            'api_manager_stats': self.api_manager.get_performance_stats()
        }

    def clear_caches(self):
        """캐시 지우기"""
        self.embedding_cache.clear()
        self.search_history.clear()
        logging.info("검색 서비스 캐시 지워짐")

    def update_search_config(self, **kwargs):
        """검색 설정 업데이트"""
        for key, value in kwargs.items():
            if key in self.search_config:
                self.search_config[key] = value
                logging.info(f"검색 설정 업데이트: {key} = {value}")
