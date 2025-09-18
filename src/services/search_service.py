#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
검색 서비스 모듈
- Pinecone 벡터 검색을 통한 의미론적 유사 답변 검색
- 다층 검색 전략으로 검색 정확도 향상
- 의도 기반 검색으로 사용자 질문의 진정한 의미 파악
"""

import logging
from typing import List, Dict
from src.utils.memory_manager import memory_cleanup
from src.utils.text_preprocessor import TextPreprocessor
from src.models.embedding_generator import EmbeddingGenerator
from src.models.question_analyzer import QuestionAnalyzer

# ===== Pinecone 벡터 검색을 담당하는 메인 클래스 =====
class SearchService:
    
    # SearchService 초기화 - 검색에 필요한 모든 구성 요소 설정
    # Args:
    #     pinecone_index: Pinecone 벡터 데이터베이스 인덱스
    #     openai_client: OpenAI API 클라이언트 (임베딩/분석용)
    def __init__(self, pinecone_index, openai_client):
        self.index = pinecone_index                               # Pinecone 벡터 검색 인덱스
        self.text_processor = TextPreprocessor()                  # 텍스트 전처리 도구
        self.embedding_generator = EmbeddingGenerator(openai_client)  # 임베딩 생성기
        self.question_analyzer = QuestionAnalyzer(openai_client)      # 질문 의도 분석기
    
    # 의도 기반 다층 검색으로 의미론적으로 동등한 질문들을 정확히 매칭하는 메서드
    # Args:
    #     query: 검색할 사용자 질문
    #     top_k: 반환할 최대 결과 수
    #     lang: 언어 코드 ('ko' 또는 'en')
    # Returns:
    #     list: 검색된 유사 답변 리스트
    def search_similar_answers_enhanced(self, query: str, top_k: int = 8, lang: str = 'ko') -> list:
        try:
            # ===== 메모리 최적화 컨텍스트 시작 =====
            with memory_cleanup():
                logging.info(f"=== 의미론적 다층 검색 시작 ===")
                logging.info(f"원본 질문: {query}")
                
                # ===== 1단계: 기본 전처리 =====
                if lang == 'ko':
                    # 한국어인 경우 AI 기반 오타 수정 적용
                    corrected_query = self.fix_korean_typos_with_ai(query)
                    query_to_embed = corrected_query
                else:
                    # 영어인 경우 원본 그대로 사용
                    query_to_embed = query
                
                # ===== 2단계: 핵심 의도 분석 =====
                # GPT를 활용해 사용자 질문의 진정한 의도와 목적 파악
                intent_analysis = self.question_analyzer.analyze_question_intent(query_to_embed)
                core_intent = intent_analysis.get('core_intent', '')                    # 핵심 의도
                standardized_query = intent_analysis.get('standardized_query', query_to_embed)  # 표준화된 질문
                semantic_keywords = intent_analysis.get('semantic_keywords', [])        # 의미론적 키워드
                
                logging.info(f"핵심 의도: {core_intent}")
                logging.info(f"표준화된 질문: {standardized_query}")
                logging.info(f"의미론적 키워드: {semantic_keywords}")
                
                # ===== 3단계: 기존 핵심 개념 추출 (보완용) =====
                # 규칙 기반으로 추출한 키워드로 의도 분석 결과 보완
                key_concepts = self.text_processor.extract_key_concepts(query_to_embed)
                
                # ===== 4단계: 검색 결과 수집 준비 =====
                all_results = []                                              # 전체 검색 결과
                seen_ids = set()                                              # 중복 제거용 ID 집합
                
                # ===== 5단계: 다층 검색 쿼리 구성 (의도 기반 강화) =====
                search_layers = [
                    # Layer 1: 원본 질문 (가중치 1.0 - 최고 우선순위)
                    {'query': query_to_embed, 'weight': 1.0, 'type': 'original'},
                    
                    # Layer 2: 표준화된 의도 기반 질문 (가중치 0.95 - GPT 분석 결과)
                    {'query': standardized_query, 'weight': 0.95, 'type': 'intent_based'},
                    
                    # Layer 3: 핵심 의도만 (가중치 0.9 - 추상화된 검색)
                    {'query': core_intent.replace('_', ' '), 'weight': 0.9, 'type': 'core_intent'},
                ]
                
                # Layer 4: 의미론적 키워드 조합 (가중치 0.8 - GPT 추출 키워드)
                if semantic_keywords and len(semantic_keywords) >= 2:
                    semantic_query = ' '.join(semantic_keywords[:3])          # 상위 3개 키워드 조합
                    search_layers.append({
                        'query': semantic_query, 'weight': 0.8, 'type': 'semantic_keywords'
                    })
                
                # Layer 5: 기존 개념 기반 검색 (가중치 0.7 - 규칙 기반 보완)
                if key_concepts:
                    if len(key_concepts) >= 2:
                        concept_query = ' '.join(key_concepts[:3])            # 상위 3개 개념 조합
                        search_layers.append({
                            'query': concept_query, 'weight': 0.7, 'type': 'concept_based'
                        })
                
                logging.info(f"검색 레이어 수: {len(search_layers)}")
                
                # ===== 6단계: 각 레이어별 검색 수행 =====
                for i, layer in enumerate(search_layers):
                    search_query = layer['query']
                    weight = layer['weight']
                    layer_type = layer['type']
                    
                    # 유효하지 않은 검색어는 건너뛰기
                    if not search_query or len(search_query.strip()) < 2:
                        continue
                    
                    logging.info(f"레이어 {i+1} ({layer_type}): {search_query[:50]}...")
                    
                    # ===== 6-1: 임베딩 벡터 생성 =====
                    query_vector = self.embedding_generator.create_embedding(search_query)
                    if query_vector is None:
                        continue
                    
                    # ===== 6-2: 검색 범위 설정 =====
                    # 첫 번째 레이어는 더 많이 검색하여 후보 확보
                    search_top_k = top_k * 2 if i == 0 else top_k
                    
                    # ===== 6-3: Pinecone 벡터 검색 실행 =====
                    results = self.index.query(
                        vector=query_vector,
                        top_k=search_top_k,
                        include_metadata=True
                    )
                    
                    # ===== 6-4: 검색 결과 처리 및 가중치 적용 =====
                    for match in results['matches']:
                        match_id = match['id']
                        if match_id not in seen_ids:                         # 중복 제거
                            seen_ids.add(match_id)
                            # 가중치 적용한 조정 점수 계산
                            adjusted_score = match['score'] * weight
                            match['adjusted_score'] = adjusted_score
                            match['search_type'] = layer_type
                            match['layer_weight'] = weight
                            all_results.append(match)
                    
                    # ===== 6-5: 메모리 정리 =====
                    del query_vector, results
                
                # ===== 7단계: 영어 질문인 경우 번역 검색 (다국어 지원) =====
                if lang == 'en':
                    # 영어 질문을 한국어로 번역하여 추가 검색
                    korean_query = self.translate_text(query_to_embed, 'en', 'ko')
                    korean_vector = self.embedding_generator.create_embedding(korean_query)
                    if korean_vector:
                        korean_results = self.index.query(
                            vector=korean_vector,
                            top_k=top_k,
                            include_metadata=True
                        )
                        # 번역 검색 결과 추가 (가중치 0.85 적용)
                        for match in korean_results['matches']:
                            if match['id'] not in seen_ids:
                                match['adjusted_score'] = match['score'] * 0.85  # 번역 페널티
                                match['search_type'] = 'translated'
                                match['layer_weight'] = 0.85
                                all_results.append(match)
                        del korean_vector, korean_results
                
                # ===== 8단계: 결과 정렬 및 의미론적 관련성 검증 =====
                # 조정된 점수 기준으로 정렬
                all_results.sort(key=lambda x: x['adjusted_score'], reverse=True)
                
                # ===== 9단계: 최종 결과 필터링 및 점수 재계산 =====
                filtered_results = []
                for i, match in enumerate(all_results[:top_k*2]):           # 후보의 2배까지 검토
                    score = match['adjusted_score']
                    question = match['metadata'].get('question', '')
                    answer = match['metadata'].get('answer', '')
                    category = match['metadata'].get('category', '일반')
                    
                    # ===== 9-1: 기본 임계값 검사 =====
                    if score < 0.3 and i >= 5:  # 상위 5개는 점수가 낮아도 포함
                        continue
                    
                    # ===== 9-2: 의도 기반 관련성 검증 =====
                    # GPT 분석 결과와 참조 답변 간의 의미적 유사성 계산
                    intent_relevance = self.question_analyzer.calculate_intent_similarity(
                        intent_analysis, question, answer
                    )
                    
                    # ===== 9-3: 개념 일치도 계산 =====
                    # 규칙 기반 키워드와 참조 답변 간의 개념적 연관성
                    concept_relevance = self.calculate_concept_relevance(
                        query_to_embed, key_concepts, question, answer
                    )
                    
                    # ===== 9-4: 최종 점수 계산 (가중 평균) =====
                    # 벡터 유사도(60%) + 의도 관련성(25%) + 개념 관련성(15%)
                    final_score = (score * 0.6 + 
                                 intent_relevance * 0.25 + 
                                 concept_relevance * 0.15)
                    
                    # ===== 9-5: 결과 선택 기준 =====
                    if final_score >= 0.4 or i < 3:  # 최소 점수 또는 상위 3개 무조건 포함
                        filtered_results.append({
                            'score': final_score,                          # 최종 종합 점수
                            'vector_score': match['score'],               # 원본 벡터 유사도
                            'intent_relevance': intent_relevance,         # 의도 관련성 점수
                            'concept_relevance': concept_relevance,       # 개념 관련성 점수
                            'question': question,                         # 참조 질문
                            'answer': answer,                             # 참조 답변
                            'category': category,                         # 카테고리
                            'rank': i + 1,                               # 순위
                            'search_type': match['search_type'],          # 검색 유형
                            'layer_weight': match.get('layer_weight', 1.0), # 레이어 가중치
                            'lang': 'ko'                                  # 언어
                        })
                        
                        # ===== 9-6: 상세 로깅 =====
                        logging.info(f"선택: #{i+1} 최종점수={final_score:.3f} "
                                   f"(벡터={match['score']:.3f}, 의도={intent_relevance:.3f}, "
                                   f"개념={concept_relevance:.3f}) 타입={match['search_type']}")
                        logging.info(f"질문: {question[:50]}...")
                    
                    # ===== 9-7: 목표 개수 달성시 종료 =====
                    if len(filtered_results) >= top_k:
                        break
                
                # ===== 10단계: 검색 완료 =====
                logging.info(f"의미론적 다층 검색 완료: {len(filtered_results)}개 답변")
                return filtered_results
                
        except Exception as e:
            # ===== 예외 처리: 검색 실패시 빈 리스트 반환 =====
            logging.error(f"의미론적 다층 검색 실패: {str(e)}")
            return []

    # 질문과 참조 답변 간의 핵심 개념 일치도를 계산하는 메서드
    # Args:
    #     query: 원본 사용자 질문
    #     query_concepts: 질문에서 추출된 핵심 개념 리스트
    #     ref_question: 참조 질문
    #     ref_answer: 참조 답변
    # Returns:
    #     float: 개념 일치도 점수 (0.0 ~ 1.0)
    def calculate_concept_relevance(self, query: str, query_concepts: list, ref_question: str, ref_answer: str) -> float:
        # ===== 1단계: 질문 개념 유효성 검사 =====
        if not query_concepts:
            return 0.5  # 개념이 없으면 중간값 반환
        
        # ===== 2단계: 참조 질문과 답변에서 개념 추출 =====
        ref_concepts = self.text_processor.extract_key_concepts(ref_question + ' ' + ref_answer)
        
        if not ref_concepts:
            return 0.3  # 참조에 개념이 없으면 낮은 점수
        
        # ===== 3단계: 개념 일치도 계산 준비 =====
        matched_concepts = 0                                    # 일치한 개념의 가중치 합
        total_weight = 0                                        # 전체 개념의 가중치 합
        
        # ===== 4단계: 각 질문 개념별 일치도 검사 =====
        for query_concept in query_concepts:
            # ===== 4-1: 개념 가중치 계산 (긴 단어에 높은 가중치) =====
            concept_weight = len(query_concept) / 10.0
            total_weight += concept_weight
            
            # ===== 4-2: 정확 일치 검사 =====
            if query_concept in ref_concepts:
                matched_concepts += concept_weight
                continue
            
            # ===== 4-3: 부분 일치 검사 (70% 이상 유사성) =====
            for ref_concept in ref_concepts:
                if len(query_concept) >= 3 and len(ref_concept) >= 3:
                    # 간단한 문자열 유사도 계산 (공통 문자 비율)
                    common_chars = set(query_concept) & set(ref_concept)
                    total_chars = len(set(query_concept)) + len(set(ref_concept))
                    similarity = len(common_chars) / max(len(set(query_concept)), len(set(ref_concept)))
                    
                    # 70% 이상 유사하면 부분 점수 부여
                    if similarity >= 0.7:
                        matched_concepts += concept_weight * similarity
                        break
        
        # ===== 5단계: 일치도 비율 계산 =====
        relevance = matched_concepts / total_weight if total_weight > 0 else 0
        
        # ===== 6단계: 0-1 범위로 정규화하여 반환 =====
        return min(relevance, 1.0)

    # AI를 이용한 한국어 오타 수정 메서드
    # Args:
    #     text: 오타 수정할 한국어 텍스트
    # Returns:
    #     str: 오타가 수정된 텍스트 (수정 실패시 원본 반환)
    def fix_korean_typos_with_ai(self, text: str) -> str:
        # ===== 1단계: 기본 유효성 검사 =====
        if not text or len(text.strip()) < 3:
            return text
        
        # ===== 2단계: 텍스트 길이 제한 (비용 최적화) =====
        if len(text) > 500:
            logging.warning(f"텍스트가 너무 길어 오타 수정 건너뜀: {len(text)}자")
            return text
        
        try:
            # ===== 3단계: AI 오타 수정 (향후 구현 예정) =====
            # 이 메서드는 실제로는 QuestionAnalyzer나 별도 서비스로 이동해야 하지만
            # 기존 호환성을 위해 여기서 간단히 처리
            # TODO: GPT 기반 한국어 오타 수정 로직 구현
            return text
                
        except Exception as e:
            # ===== 예외 처리: 오타 수정 실패시 원본 반환 =====
            logging.error(f"AI 오타 수정 실패: {e}")
            return text

    # 다국어 번역 기능 메서드 (AnswerGenerator에서 가져온 기능)
    # Args:
    #     text: 번역할 텍스트
    #     source_lang: 원본 언어 코드
    #     target_lang: 목적 언어 코드
    # Returns:
    #     str: 번역된 텍스트 (현재는 원본 반환)
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        # ===== 향후 개선 사항 =====
        # 실제로는 별도 번역 서비스로 분리하는 것이 좋음
        # TODO: GPT 기반 번역 로직 구현 또는 전용 번역 서비스 연동
        return text

    # 향상된 컨텍스트 품질 분석 메서드 - 개념 일치도 고려
    # Args:
    #     similar_answers: 유사 답변 리스트
    #     query: 원본 사용자 질문
    # Returns:
    #     dict: 컨텍스트 품질 분석 결과
    def analyze_context_quality(self, similar_answers: list, query: str) -> dict:
        # ===== 1단계: 기본 유효성 검사 =====
        if not similar_answers:
            return {
                'has_good_context': False,
                'best_score': 0.0,
                'recommended_approach': 'fallback',
                'quality_level': 'none'
            }
        
        # ===== 2단계: 최고 점수와 관련성 점수 확인 =====
        best_answer = similar_answers[0]
        best_score = best_answer['score']
        relevance_score = best_answer.get('relevance_score', 0.5)
        
        # ===== 3단계: 고품질 답변 개수 계산 =====
        high_quality_count = len([ans for ans in similar_answers if ans['score'] >= 0.7])          # 유사도 70% 이상
        good_relevance_count = len([ans for ans in similar_answers if ans.get('relevance_score', 0) >= 0.6])  # 관련성 60% 이상
        
        # ===== 4단계: 접근 방식 결정 (개념 일치도 고려) =====
        if best_score >= 0.9 and relevance_score >= 0.7:
            # 최고 품질: 직접 사용 가능
            approach = 'direct_use'
            quality_level = 'excellent'
        elif best_score >= 0.8 and relevance_score >= 0.6:
            # 매우 높은 품질: 직접 사용 가능
            approach = 'direct_use' 
            quality_level = 'very_high'
        elif best_score >= 0.7 and relevance_score >= 0.5:
            # 높은 품질: GPT로 강한 컨텍스트 활용
            approach = 'gpt_with_strong_context'
            quality_level = 'high'
        elif best_score >= 0.6 and (high_quality_count + good_relevance_count) >= 2:
            # 중간 품질: 여러 답변 조합하여 GPT 활용
            approach = 'gpt_with_strong_context'
            quality_level = 'medium'
        elif best_score >= 0.4 and relevance_score >= 0.4:
            # 낮은 품질: 약한 컨텍스트로 GPT 활용
            approach = 'gpt_with_weak_context'
            quality_level = 'low'
        else:
            # 매우 낮은 품질: 폴백 답변 사용
            approach = 'fallback'
            quality_level = 'very_low'
        
        # ===== 5단계: 분석 결과 반환 =====
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

    # 최적의 폴백 답변 선택 메서드 (검색 실패시 대안)
    # Args:
    #     similar_answers: 유사 답변 리스트
    #     lang: 언어 코드 ('ko' 또는 'en')
    # Returns:
    #     str: 선택된 최적 폴백 답변
    def get_best_fallback_answer(self, similar_answers: list, lang: str = 'ko') -> str:
        logging.info(f"=== get_best_fallback_answer 시작 ===")
        logging.info(f"입력된 similar_answers 개수: {len(similar_answers)}")
        
        # ===== 1단계: 기본 유효성 검사 =====
        if not similar_answers:
            logging.warning("similar_answers가 비어있음")
            return ""
        
        # ===== 2단계: 입력된 답변들 미리보기 =====
        for i, ans in enumerate(similar_answers[:3]):
            logging.info(f"답변 #{i+1}: 점수={ans['score']:.3f}, 길이={len(ans.get('answer', ''))}")
        
        # ===== 3단계: 점수와 텍스트 품질을 종합 평가 =====
        best_answer = ""
        best_score = 0
        
        # ===== 4단계: 상위 3개 답변만 검토 (효율성) =====
        for i, ans in enumerate(similar_answers[:3]):
            logging.info(f"--- 답변 #{i+1} 처리 시작 ---")
            score = ans['score']
            answer_text = ans['answer']
            
            # ===== 4-1: 매우 높은 유사도 확인 (즉시 반환) =====
            if score >= 0.9:
                logging.info(f"매우 높은 유사도({score:.3f}) - 원본 답변 바로 반환")
                clean_answer = answer_text.strip()
                if clean_answer:
                    return clean_answer
            
            # ===== 4-2: 기본 텍스트 전처리 =====
            answer_text = self.text_processor.preprocess_text(answer_text)
            
            # ===== 4-3: 다국어 지원 (필요시 번역) =====
            if lang == 'en' and ans.get('lang', 'ko') == 'ko':
                answer_text = self.translate_text(answer_text, 'ko', 'en')
            
            # ===== 4-4: 높은 유사도 확인 (간단 선택) =====
            if score >= 0.8:
                logging.info(f"높은 유사도({score:.3f})로 답변 #{i+1} 직접 선택")
                if answer_text and len(answer_text.strip()) > 0:
                    return answer_text
                else:
                    logging.error(f"전처리 후 답변이 비어있음! 원본으로 폴백")
                    return ans['answer'].strip()
            
            # ===== 4-5: 종합 점수 계산 =====
            # 유사도(80%) + 텍스트 길이(10%) + 완성도(10%)
            length_score = min(len(answer_text) / 200, 1.0)         # 200자 기준 정규화
            completeness_score = 1.0 if answer_text.endswith(('.', '!', '?')) else 0.8  # 문장 완성도
            
            total_score = score * 0.8 + length_score * 0.1 + completeness_score * 0.1
            
            logging.info(f"답변 #{i+1} 종합 점수: {total_score:.3f}")
            
            # ===== 4-6: 최고 점수 답변 업데이트 =====
            if total_score > best_score:
                best_score = total_score
                best_answer = answer_text
                logging.info(f"새로운 최고 점수 답변으로 선택됨")
        
        # ===== 5단계: 긴급 안전장치 =====
        # 답변이 비어있으면 첫 번째 원본 답변 강제 반환
        if not best_answer and similar_answers:
            logging.error("최종 답변이 비어있음! 첫 번째 원본 답변 강제 반환")
            emergency_answer = similar_answers[0]['answer'].strip()
            return emergency_answer
        
        # ===== 6단계: 최종 답변 반환 =====
        return best_answer
