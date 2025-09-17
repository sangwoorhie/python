#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
질문 분석 모델 모듈
"""

import json
import logging
import re
from typing import Dict
from langdetect import detect, LangDetectException
from src.utils.memory_manager import memory_cleanup


class QuestionAnalyzer:
    """질문 분석을 담당하는 클래스"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
    
    def detect_language(self, text: str) -> str:
        """텍스트의 언어를 감지하는 메서드"""
        try:
            # langdetect 라이브러리 사용
            detected = detect(text)
            
            # 영어와 한국어만 지원
            if detected == 'en':
                return 'en'
            elif detected == 'ko':
                return 'ko'
            else:
                # 기본값은 한국어
                return 'ko'
        except LangDetectException:
            # 감지 실패시 텍스트 내 한글 비율로 판단
            korean_chars = len(re.findall(r'[가-힣]', text))
            english_chars = len(re.findall(r'[a-zA-Z]', text))
            
            if korean_chars > english_chars:
                return 'ko'
            else:
                return 'en'

    def analyze_question_intent(self, query: str) -> dict:
        """AI를 이용해 질문의 본질적 의도와 핵심 목적을 정확히 분석"""
        try:
            with memory_cleanup():
                system_prompt = """당신은 바이블 앱 문의 분석 전문가입니다. 
고객 질문의 본질적 의도를 파악하여 의미론적으로 동등한 질문들이 같은 결과를 얻도록 분석하세요.

분석 결과를 JSON 형태로 반환:

{
  "core_intent": "핵심 의도 (표준화된 형태)",
  "intent_category": "의도 카테고리",
  "primary_action": "주요 행동",
  "target_object": "대상 객체",
  "constraint_conditions": ["제약 조건들"],
  "standardized_query": "표준화된 질문 형태",
  "semantic_keywords": ["의미론적 핵심 키워드들"]
}

🎯 의미론적 동등성 분석 기준:

1. **핵심 의도 파악**: 질문의 본질적 목적이 무엇인지 파악
   - "두 번역본을 동시에 보고 싶다" → core_intent: "multiple_translations_view"
   - "텍스트를 복사하고 싶다" → core_intent: "text_copy"
   - "연속으로 듣고 싶다" → core_intent: "continuous_audio_play"

2. **표준화된 형태로 변환**: 구체적 예시를 제거하고 일반화
   - "요한복음 3장 16절 NIV와 KJV 동시에" → "서로 다른 번역본 동시 보기"
   - "개역한글과 개역개정 동시에" → "서로 다른 번역본 동시 보기"

3. **의미론적 키워드 추출**: 표면적 단어가 아닌 의미적 개념
   - "동시에", "함께", "비교하여", "나란히" → "simultaneous_view"
   - "NIV", "KJV", "개역한글", "번역본" → "translation_version"

4. **제약 조건 식별**: 요청의 구체적 조건들
   - "영어 번역본만", "한글 번역본만", "특정 장절" 등

예시 분석:
질문1: "요한복음 3장 16절 영어 번역본 NIV와 KJV 동시에 보려면?"
질문2: "개역한글과 개역개정을 동시에 보려면?"
질문3: "두 개의 번역본을 어떻게 동시에 볼 수 있죠?"

→ 모두 core_intent: "multiple_translations_simultaneous_view"
→ 모두 standardized_query: "서로 다른 번역본을 동시에 보는 방법"
"""

                user_prompt = f"""다음 질문을 의미론적으로 분석하여 본질적 의도를 파악해주세요:

질문: {query}

특히 다음 사항에 집중하세요:
1. 이 질문이 정말로 묻고자 하는 바가 무엇인가?
2. 구체적 예시(성경 구절, 번역본명 등)를 제거하고 일반화하면?
3. 비슷한 의도의 다른 질문들과 어떻게 통합할 수 있는가?"""

                response = self.openai_client.chat.completions.create(
                    model='gpt-3.5-turbo',
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=400,
                    temperature=0.2  # 더 일관성 있는 분석을 위해 낮춤
                )
                
                result_text = response.choices[0].message.content.strip()
                
                # JSON 파싱 시도
                try:
                    result = json.loads(result_text)
                    logging.info(f"강화된 의도 분석 결과: {result}")
                    
                    # 기존 형식과의 호환성을 위해 추가 필드 생성
                    result['intent_type'] = result.get('intent_category', '일반문의')
                    result['main_topic'] = result.get('target_object', '기타')
                    result['specific_request'] = result.get('standardized_query', query[:100])
                    result['keywords'] = result.get('semantic_keywords', [query[:20]])
                    result['urgency'] = 'medium'
                    result['action_type'] = result.get('primary_action', '기타')
                    
                    return result
                except json.JSONDecodeError:
                    logging.warning(f"JSON 파싱 실패, 기본값 반환: {result_text}")
                    return {
                        "core_intent": "general_inquiry",
                        "intent_category": "일반문의",
                        "primary_action": "기타",
                        "target_object": "기타",
                        "constraint_conditions": [],
                        "standardized_query": query,
                        "semantic_keywords": [query[:20]],
                        # 기존 호환성 필드
                        "intent_type": "일반문의",
                        "main_topic": "기타",
                        "specific_request": query[:100],
                        "keywords": [query[:20]],
                        "urgency": "medium",
                        "action_type": "기타"
                    }
                
        except Exception as e:
            logging.error(f"강화된 의도 분석 실패: {e}")
            return {
                "core_intent": "general_inquiry",
                "intent_category": "일반문의", 
                "primary_action": "기타",
                "target_object": "기타",
                "constraint_conditions": [],
                "standardized_query": query,
                "semantic_keywords": [query[:20]],
                # 기존 호환성 필드
                "intent_type": "일반문의",
                "main_topic": "기타",
                "specific_request": query[:100],
                "keywords": [query[:20]],
                "urgency": "medium",
                "action_type": "기타"
            }

    def calculate_intent_similarity(self, query_intent_analysis: dict, ref_question: str, ref_answer: str) -> float:
        """질문의 의도와 참조 답변 간의 의미론적 유사성 계산"""
        
        try:
            # 1. 질문 의도 정보 추출
            query_core_intent = query_intent_analysis.get('core_intent', '')
            query_primary_action = query_intent_analysis.get('primary_action', '')
            query_target_object = query_intent_analysis.get('target_object', '')
            query_semantic_keywords = query_intent_analysis.get('semantic_keywords', [])
            
            if not query_core_intent:
                return 0.5  # 의도 정보가 없으면 중간값
            
            # 2. 참조 질문과 답변에서 의도 분석
            ref_text = ref_question + ' ' + ref_answer
            ref_intent_analysis = self.analyze_question_intent(ref_question)
            
            ref_core_intent = ref_intent_analysis.get('core_intent', '')
            ref_primary_action = ref_intent_analysis.get('primary_action', '')
            ref_target_object = ref_intent_analysis.get('target_object', '')
            ref_semantic_keywords = ref_intent_analysis.get('semantic_keywords', [])
            
            # 3. 핵심 의도 일치도 계산 (가장 중요)
            intent_match_score = 0.0
            if query_core_intent == ref_core_intent:
                intent_match_score = 1.0
            elif query_core_intent and ref_core_intent:
                # 의도 이름의 유사성 검사 (부분 일치)
                query_intent_words = set(query_core_intent.split('_'))
                ref_intent_words = set(ref_core_intent.split('_'))
                
                if query_intent_words & ref_intent_words:  # 공통 단어가 있으면
                    overlap_ratio = len(query_intent_words & ref_intent_words) / len(query_intent_words | ref_intent_words)
                    intent_match_score = overlap_ratio * 0.8  # 완전 일치보다는 낮게
            
            # 4. 행동 유형 일치도 계산
            action_match_score = 0.0
            if query_primary_action == ref_primary_action:
                action_match_score = 1.0
            elif query_primary_action and ref_primary_action:
                # 행동 유형 유사성 검사
                action_similarity_map = {
                    ('보기', '확인'): 0.8,
                    ('복사', '저장'): 0.7,
                    ('듣기', '재생'): 0.9,
                    ('검색', '찾기'): 0.8,
                    ('설정', '변경'): 0.7
                }
                
                action_key = (query_primary_action, ref_primary_action)
                reverse_key = (ref_primary_action, query_primary_action)
                
                if action_key in action_similarity_map:
                    action_match_score = action_similarity_map[action_key]
                elif reverse_key in action_similarity_map:
                    action_match_score = action_similarity_map[reverse_key]
            
            # 5. 대상 객체 일치도 계산
            object_match_score = 0.0
            if query_target_object == ref_target_object:
                object_match_score = 1.0
            elif query_target_object and ref_target_object:
                # 객체 유사성 검사
                object_similarity_map = {
                    ('번역본', '성경'): 0.8,
                    ('텍스트', '내용'): 0.7,
                    ('음성', '오디오'): 0.9,
                    ('화면', '디스플레이'): 0.7
                }
                
                object_key = (query_target_object, ref_target_object)
                reverse_key = (ref_target_object, query_target_object)
                
                if object_key in object_similarity_map:
                    object_match_score = object_similarity_map[object_key]
                elif reverse_key in object_similarity_map:
                    object_match_score = object_similarity_map[reverse_key]
            
            # 6. 의미론적 키워드 일치도 계산
            keyword_match_score = 0.0
            if query_semantic_keywords and ref_semantic_keywords:
                query_keyword_set = set(query_semantic_keywords)
                ref_keyword_set = set(ref_semantic_keywords)
                
                common_keywords = query_keyword_set & ref_keyword_set
                total_keywords = query_keyword_set | ref_keyword_set
                
                if total_keywords:
                    keyword_match_score = len(common_keywords) / len(total_keywords)
            
            # 7. 전체 점수 계산 (가중 평균)
            total_score = (
                intent_match_score * 0.4 +      # 핵심 의도 일치 (40%)
                action_match_score * 0.25 +     # 행동 유형 일치 (25%)
                object_match_score * 0.2 +      # 대상 객체 일치 (20%)
                keyword_match_score * 0.15      # 키워드 일치 (15%)
            )
            
            logging.debug(f"의도 유사성 분석: 의도={intent_match_score:.2f}, "
                         f"행동={action_match_score:.2f}, 객체={object_match_score:.2f}, "
                         f"키워드={keyword_match_score:.2f}, 전체={total_score:.2f}")
            
            return min(total_score, 1.0)
            
        except Exception as e:
            logging.error(f"의도 유사성 계산 실패: {e}")
            return 0.3  # 오류시 낮은 기본값
