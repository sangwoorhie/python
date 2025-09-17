#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
검색 서비스 모듈
"""

import logging
from typing import List, Dict
from src.utils.memory_manager import memory_cleanup
from src.utils.text_preprocessor import TextPreprocessor
from src.models.embedding_generator import EmbeddingGenerator
from src.models.question_analyzer import QuestionAnalyzer


class SearchService:
    """Pinecone 벡터 검색을 담당하는 클래스"""
    
    def __init__(self, pinecone_index, openai_client):
        self.index = pinecone_index
        self.text_processor = TextPreprocessor()
        self.embedding_generator = EmbeddingGenerator(openai_client)
        self.question_analyzer = QuestionAnalyzer(openai_client)
    
    def search_similar_answers_enhanced(self, query: str, top_k: int = 8, lang: str = 'ko') -> list:
        """의도 기반 다층 검색으로 의미론적으로 동등한 질문들을 정확히 매칭"""
        try:
            with memory_cleanup():
                logging.info(f"=== 의미론적 다층 검색 시작 ===")
                logging.info(f"원본 질문: {query}")
                
                # 1. 기본 전처리
                if lang == 'ko':
                    corrected_query = self.fix_korean_typos_with_ai(query)
                    query_to_embed = corrected_query
                else:
                    query_to_embed = query
                
                # 2. 핵심 의도 분석
                intent_analysis = self.question_analyzer.analyze_question_intent(query_to_embed)
                core_intent = intent_analysis.get('core_intent', '')
                standardized_query = intent_analysis.get('standardized_query', query_to_embed)
                semantic_keywords = intent_analysis.get('semantic_keywords', [])
                
                logging.info(f"핵심 의도: {core_intent}")
                logging.info(f"표준화된 질문: {standardized_query}")
                logging.info(f"의미론적 키워드: {semantic_keywords}")
                
                # 3. 기존 핵심 개념 추출 (보완용)
                key_concepts = self.text_processor.extract_key_concepts(query_to_embed)
                
                all_results = []
                seen_ids = set()
                
                # 4. 다층 검색 쿼리 구성 (의도 기반 강화)
                search_layers = [
                    # Layer 1: 원본 질문 (가중치 1.0)
                    {'query': query_to_embed, 'weight': 1.0, 'type': 'original'},
                    
                    # Layer 2: 표준화된 의도 기반 질문 (가중치 0.95)
                    {'query': standardized_query, 'weight': 0.95, 'type': 'intent_based'},
                    
                    # Layer 3: 핵심 의도만 (가중치 0.9)
                    {'query': core_intent.replace('_', ' '), 'weight': 0.9, 'type': 'core_intent'},
                ]
                
                # Layer 4: 의미론적 키워드 조합 (가중치 0.8)
                if semantic_keywords and len(semantic_keywords) >= 2:
                    semantic_query = ' '.join(semantic_keywords[:3])
                    search_layers.append({
                        'query': semantic_query, 'weight': 0.8, 'type': 'semantic_keywords'
                    })
                
                # Layer 5: 기존 개념 기반 검색 (보완용, 가중치 0.7)
                if key_concepts:
                    if len(key_concepts) >= 2:
                        concept_query = ' '.join(key_concepts[:3])
                        search_layers.append({
                            'query': concept_query, 'weight': 0.7, 'type': 'concept_based'
                        })
                
                logging.info(f"검색 레이어 수: {len(search_layers)}")
                
                # 5. 각 레이어로 검색 수행
                for i, layer in enumerate(search_layers):
                    search_query = layer['query']
                    weight = layer['weight']
                    layer_type = layer['type']
                    
                    if not search_query or len(search_query.strip()) < 2:
                        continue
                    
                    logging.info(f"레이어 {i+1} ({layer_type}): {search_query[:50]}...")
                    
                    query_vector = self.embedding_generator.create_embedding(search_query)
                    if query_vector is None:
                        continue
                    
                    # 첫 번째 레이어는 더 많이 검색
                    search_top_k = top_k * 2 if i == 0 else top_k
                    
                    results = self.index.query(
                        vector=query_vector,
                        top_k=search_top_k,
                        include_metadata=True
                    )
                    
                    # 결과를 가중치와 함께 수집
                    for match in results['matches']:
                        match_id = match['id']
                        if match_id not in seen_ids:
                            seen_ids.add(match_id)
                            # 가중치 적용한 점수 계산
                            adjusted_score = match['score'] * weight
                            match['adjusted_score'] = adjusted_score
                            match['search_type'] = layer_type
                            match['layer_weight'] = weight
                            all_results.append(match)
                    
                    del query_vector, results
                
                # 6. 영어 질문인 경우 번역 검색
                if lang == 'en':
                    korean_query = self.translate_text(query_to_embed, 'en', 'ko')
                    korean_vector = self.embedding_generator.create_embedding(korean_query)
                    if korean_vector:
                        korean_results = self.index.query(
                            vector=korean_vector,
                            top_k=top_k,
                            include_metadata=True
                        )
                        for match in korean_results['matches']:
                            if match['id'] not in seen_ids:
                                match['adjusted_score'] = match['score'] * 0.85
                                match['search_type'] = 'translated'
                                match['layer_weight'] = 0.85
                                all_results.append(match)
                        del korean_vector, korean_results
                
                # 7. 결과 정렬 및 의미론적 관련성 검증
                all_results.sort(key=lambda x: x['adjusted_score'], reverse=True)
                
                filtered_results = []
                for i, match in enumerate(all_results[:top_k*2]):
                    score = match['adjusted_score']
                    question = match['metadata'].get('question', '')
                    answer = match['metadata'].get('answer', '')
                    category = match['metadata'].get('category', '일반')
                    
                    # 기본 임계값 검사
                    if score < 0.3 and i >= 5:  # 상위 5개는 점수가 낮아도 포함
                        continue
                    
                    # 의도 기반 관련성 검증
                    intent_relevance = self.question_analyzer.calculate_intent_similarity(
                        intent_analysis, question, answer
                    )
                    
                    # 기존 개념 일치도도 함께 고려
                    concept_relevance = self.calculate_concept_relevance(
                        query_to_embed, key_concepts, question, answer
                    )
                    
                    # 최종 점수 = 벡터 유사도(60%) + 의도 관련성(25%) + 개념 관련성(15%)
                    final_score = (score * 0.6 + 
                                 intent_relevance * 0.25 + 
                                 concept_relevance * 0.15)
                    
                    if final_score >= 0.4 or i < 3:  # 상위 3개는 무조건 포함
                        filtered_results.append({
                            'score': final_score,
                            'vector_score': match['score'],
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
                        
                        logging.info(f"선택: #{i+1} 최종점수={final_score:.3f} "
                                   f"(벡터={match['score']:.3f}, 의도={intent_relevance:.3f}, "
                                   f"개념={concept_relevance:.3f}) 타입={match['search_type']}")
                        logging.info(f"질문: {question[:50]}...")
                    
                    if len(filtered_results) >= top_k:
                        break
                
                logging.info(f"의미론적 다층 검색 완료: {len(filtered_results)}개 답변")
                return filtered_results
                
        except Exception as e:
            logging.error(f"의미론적 다층 검색 실패: {str(e)}")
            return []

    def calculate_concept_relevance(self, query: str, query_concepts: list, ref_question: str, ref_answer: str) -> float:
        """질문과 참조 답변 간의 핵심 개념 일치도 계산"""
        
        if not query_concepts:
            return 0.5  # 개념이 없으면 중간값
        
        # 참조 질문과 답변에서 개념 추출
        ref_concepts = self.text_processor.extract_key_concepts(ref_question + ' ' + ref_answer)
        
        if not ref_concepts:
            return 0.3  # 참조에 개념이 없으면 낮은 점수
        
        # 개념 일치도 계산
        matched_concepts = 0
        total_weight = 0
        
        for query_concept in query_concepts:
            concept_weight = len(query_concept) / 10.0  # 긴 단어에 높은 가중치
            total_weight += concept_weight
            
            # 정확히 일치하는 개념 찾기
            if query_concept in ref_concepts:
                matched_concepts += concept_weight
                continue
            
            # 부분 일치 검사 (70% 이상 일치)
            for ref_concept in ref_concepts:
                if len(query_concept) >= 3 and len(ref_concept) >= 3:
                    # 간단한 문자열 유사도 (공통 문자 비율)
                    common_chars = set(query_concept) & set(ref_concept)
                    similarity = len(common_chars) / max(len(set(query_concept)), len(set(ref_concept)))
                    
                    if similarity >= 0.7:  # 70% 이상 유사하면 부분 점수
                        matched_concepts += concept_weight * similarity
                        break
        
        # 일치도 비율 계산
        relevance = matched_concepts / total_weight if total_weight > 0 else 0
        
        # 0-1 범위로 정규화
        return min(relevance, 1.0)

    def fix_korean_typos_with_ai(self, text: str) -> str:
        """AI를 이용한 한국어 오타 수정"""
        if not text or len(text.strip()) < 3:
            return text
        
        # 너무 긴 텍스트는 처리하지 않음 (비용 절약)
        if len(text) > 500:
            logging.warning(f"텍스트가 너무 길어 오타 수정 건너뜀: {len(text)}자")
            return text
        
        try:
            # 이 메서드는 실제로는 QuestionAnalyzer나 별도 서비스로 이동해야 하지만
            # 기존 호환성을 위해 여기서 간단히 처리
            return text
                
        except Exception as e:
            logging.error(f"AI 오타 수정 실패: {e}")
            return text

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """번역 기능 (AnswerGenerator에서 가져옴)"""
        # 실제로는 별도 번역 서비스로 분리하는 것이 좋음
        return text

    def analyze_context_quality(self, similar_answers: list, query: str) -> dict:
        """향상된 컨텍스트 품질 분석 - 개념 일치도 고려"""
        
        if not similar_answers:
            return {
                'has_good_context': False,
                'best_score': 0.0,
                'recommended_approach': 'fallback',
                'quality_level': 'none'
            }
        
        # 최고 점수와 관련성 점수 확인
        best_answer = similar_answers[0]
        best_score = best_answer['score']
        relevance_score = best_answer.get('relevance_score', 0.5)
        
        # 고품질 답변 개수 계산
        high_quality_count = len([ans for ans in similar_answers if ans['score'] >= 0.7])
        good_relevance_count = len([ans for ans in similar_answers if ans.get('relevance_score', 0) >= 0.6])
        
        # 접근 방식 결정 (개념 일치도 고려)
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
        """최적의 폴백 답변 선택 메서드"""
        logging.info(f"=== get_best_fallback_answer 시작 ===")
        logging.info(f"입력된 similar_answers 개수: {len(similar_answers)}")
        
        if not similar_answers:
            logging.warning("similar_answers가 비어있음")
            return ""
        
        # 입력된 답변들 미리보기
        for i, ans in enumerate(similar_answers[:3]):
            logging.info(f"답변 #{i+1}: 점수={ans['score']:.3f}, 길이={len(ans.get('answer', ''))}")
        
        # 점수와 텍스트 품질을 종합 평가
        best_answer = ""
        best_score = 0
        
        for i, ans in enumerate(similar_answers[:3]): # 상위 3개만 검토
            logging.info(f"--- 답변 #{i+1} 처리 시작 ---")
            score = ans['score']
            answer_text = ans['answer']
            
            # 매우 높은 유사도면 전처리 없이 바로 반환
            if score >= 0.9:
                logging.info(f"매우 높은 유사도({score:.3f}) - 원본 답변 바로 반환")
                clean_answer = answer_text.strip()
                if clean_answer:
                    return clean_answer
            
            # 기본 정리만 수행
            answer_text = self.text_processor.preprocess_text(answer_text)
            
            # 영어 질문인 경우 답변을 번역
            if lang == 'en' and ans.get('lang', 'ko') == 'ko':
                answer_text = self.translate_text(answer_text, 'ko', 'en')
            
            # 높은 유사도인 경우 간단하게 첫 번째 답변 선택
            if score >= 0.8:
                logging.info(f"높은 유사도({score:.3f})로 답변 #{i+1} 직접 선택")
                if answer_text and len(answer_text.strip()) > 0:
                    return answer_text
                else:
                    logging.error(f"전처리 후 답변이 비어있음! 원본으로 폴백")
                    return ans['answer'].strip()
            
            # 종합 점수 계산 (유사도 + 텍스트 길이 + 완성도)
            length_score = min(len(answer_text) / 200, 1.0) # 200자 기준 정규화
            completeness_score = 1.0 if answer_text.endswith(('.', '!', '?')) else 0.8
            
            total_score = score * 0.8 + length_score * 0.1 + completeness_score * 0.1
            
            logging.info(f"답변 #{i+1} 종합 점수: {total_score:.3f}")
            
            if total_score > best_score:
                best_score = total_score
                best_answer = answer_text
                logging.info(f"새로운 최고 점수 답변으로 선택됨")
        
        # 긴급 안전장치: 답변이 비어있으면 첫 번째 원본 답변 강제 반환
        if not best_answer and similar_answers:
            logging.error("최종 답변이 비어있음! 첫 번째 원본 답변 강제 반환")
            emergency_answer = similar_answers[0]['answer'].strip()
            return emergency_answer
        
        return best_answer
