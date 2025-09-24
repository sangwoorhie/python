#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
질문 분석 모델 모듈
- 사용자 질문의 언어 감지 및 의도 분석
- 의미론적 유사성 계산을 통한 답변 매칭 최적화
- GPT 기반 지능형 질문 분석 시스템
"""

import json
import logging
import re
from typing import Dict
from langdetect import detect, LangDetectException
from src.utils.memory_manager import memory_cleanup

# ===== 질문 분석 및 의도 파악을 담당하는 메인 클래스 =====
class QuestionAnalyzer:
    
    # QuestionAnalyzer 초기화
    # Args:
    #     openai_client: OpenAI API 클라이언트 인스턴스
    def __init__(self, openai_client):
        self.openai_client = openai_client                    # GPT 분석을 위한 OpenAI 클라이언트
    
    # 텍스트의 언어를 자동 감지하는 메서드
    # Args:
    #     text: 언어를 감지할 텍스트
    # Returns:
    #     str: 감지된 언어 코드 ('ko' 또는 'en')
    def detect_language(self, text: str) -> str:
        try:
            # ===== 1단계: langdetect 라이브러리를 사용한 자동 언어 감지 =====
            detected = detect(text)
            
            # ===== 2단계: 지원 언어 검증 (한국어/영어만 지원) =====
            if detected == 'en':
                return 'en'                                   # 영어로 감지됨
            elif detected == 'ko':
                return 'ko'                                   # 한국어로 감지됨
            else:
                # 기타 언어는 기본값(한국어)으로 처리
                return 'ko'
                
        except LangDetectException:
            # ===== 3단계: 감지 실패시 문자 비율 기반 폴백 로직 =====
            # 텍스트 내 한글과 영문 문자 수를 직접 카운트
            korean_chars = len(re.findall(r'[가-힣]', text))  # 한글 문자 수
            english_chars = len(re.findall(r'[a-zA-Z]', text)) # 영문 문자 수
            
            # 문자 수 비교로 언어 판단
            if korean_chars > english_chars:
                return 'ko'                                   # 한글이 더 많으면 한국어
            else:
                return 'en'                                   # 영문이 더 많으면 영어

    # GPT를 이용해 질문의 본질적 의도와 핵심 목적을 정확히 분석하는 메서드
    # Args:
    #     query: 분석할 사용자 질문
    # Returns:
    #     dict: 의도 분석 결과 (core_intent, 카테고리, 키워드 등)
    def analyze_question_intent(self, query: str) -> dict:
        try:
            # ===== 메모리 최적화 컨텍스트 시작 =====
            with memory_cleanup():
                # ===== 1단계: GPT 의도 분석을 위한 시스템 프롬프트 구성 =====
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

                # ===== 2단계: 사용자 질문 분석을 위한 프롬프트 생성 =====
                user_prompt = f"""다음 질문을 의미론적으로 분석하여 본질적 의도를 파악해주세요:

질문: {query}

특히 다음 사항에 집중하세요:
1. 이 질문이 정말로 묻고자 하는 바가 무엇인가?
2. 구체적 예시(성경 구절, 번역본명 등)를 제거하고 일반화하면?
3. 비슷한 의도의 다른 질문들과 어떻게 통합할 수 있는가?"""

                # ===== 3단계: GPT API 호출로 의도 분석 실행 =====
                response = self.openai_client.chat.completions.create(
                    model='gpt-5-mini',
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_completion_tokens=400,                               # 충분한 분석 결과 길이
                    # temperature=0.2                               # 일관성 있는 분석을 위해 낮은 값
                    response_format={"type": "json_object"}                  # JSON 형식으로 응답
                )
                
                # ===== 4단계: GPT 응답 텍스트 추출 =====
                result_text = response.choices[0].message.content.strip()
                
                # ===== 5단계: JSON 파싱 및 결과 구조화 =====
                try:
                    # JSON 형태로 응답 파싱
                    content = response.choices[0].message.content
                    result = json.loads(content) 
                    logging.info(f"강화된 의도 분석 결과: {result}")
                    
                    # ===== 6단계: 기존 시스템과의 호환성을 위한 필드 추가 =====
                    result['intent_type'] = result.get('intent_category', '일반문의')
                    result['main_topic'] = result.get('target_object', '기타')
                    result['specific_request'] = result.get('standardized_query', query[:100])
                    result['keywords'] = result.get('semantic_keywords', [query[:20]])
                    result['urgency'] = 'medium'
                    result['action_type'] = result.get('primary_action', '기타')
                    
                    return result
                except json.JSONDecodeError:
                    # ===== JSON 파싱 실패시 기본값 반환 =====
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
            # ===== 전체 의도 분석 프로세스 실패시 기본값 반환 =====
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

    # 질문의 의도와 참조 답변 간의 의미론적 유사성을 계산하는 메서드
    # Args:
    #     query_intent_analysis: 분석된 질문 의도 정보
    #     ref_question: 참조 질문
    #     ref_answer: 참조 답변
    # Returns:
    #     float: 유사성 점수 (0.0 ~ 1.0)
    def calculate_intent_similarity(self, query_intent_analysis: dict, ref_question: str, ref_answer: str) -> float:
        
        try:
            # ===== 1단계: 사용자 질문의 의도 정보 추출 =====
            query_core_intent = query_intent_analysis.get('core_intent', '')
            query_primary_action = query_intent_analysis.get('primary_action', '')
            query_target_object = query_intent_analysis.get('target_object', '')
            query_semantic_keywords = query_intent_analysis.get('semantic_keywords', [])
            
            # 의도 정보가 없으면 중간값 반환
            if not query_core_intent:
                return 0.5
            
            # ===== 2단계: 참조 질문의 의도 분석 실행 =====
            ref_text = ref_question + ' ' + ref_answer
                
            # 🔍 실시간 의도 분석 시작 로그
            logging.info(f"🔍 기존 답변 실시간 의도 분석 시작:")
            logging.info(f"   └── 기존 질문: {ref_question[:80]}...")
                
            ref_intent_analysis = self.analyze_question_intent(ref_question)
                
                # 🔍 실시간 의도 분석 결과 로그
            logging.info(f"🔍 기존 답변 의도 분석 결과:")
            logging.info(f"   └── 핵심 의도: {ref_intent_analysis.get('core_intent', 'N/A')}")
            logging.info(f"   └── 주요 행동: {ref_intent_analysis.get('primary_action', 'N/A')}")
            logging.info(f"   └── 대상 객체: {ref_intent_analysis.get('target_object', 'N/A')}")
            logging.info(f"   └── 키워드: {ref_intent_analysis.get('semantic_keywords', [])}")
            
            ref_core_intent = ref_intent_analysis.get('core_intent', '')
            ref_primary_action = ref_intent_analysis.get('primary_action', '')
            ref_target_object = ref_intent_analysis.get('target_object', '')
            ref_semantic_keywords = ref_intent_analysis.get('semantic_keywords', [])
            
            # ===== 3단계: 핵심 의도 일치도 계산 (가장 중요한 지표) =====
            intent_match_score = 0.0
            if query_core_intent == ref_core_intent:
                # 완전 일치: 최고 점수
                intent_match_score = 1.0
            elif query_core_intent and ref_core_intent:
                # 부분 일치: 의도 이름의 단어 유사성 검사
                query_intent_words = set(query_core_intent.split('_'))
                ref_intent_words = set(ref_core_intent.split('_'))
                
                if query_intent_words & ref_intent_words:  # 공통 단어가 있으면
                    overlap_ratio = len(query_intent_words & ref_intent_words) / len(query_intent_words | ref_intent_words)
                    intent_match_score = overlap_ratio * 0.8  # 완전 일치보다는 낮게 설정
            
            # ===== 4단계: 행동 유형 일치도 계산 =====
            action_match_score = 0.0
            if query_primary_action == ref_primary_action:
                # 완전 일치: 최고 점수
                action_match_score = 1.0
            elif query_primary_action and ref_primary_action:
                # 유사한 행동 유형 매핑 테이블
                action_similarity_map = {
                    ('보기', '확인'): 0.8,
                    ('복사', '저장'): 0.7,
                    ('듣기', '재생'): 0.9,
                    ('검색', '찾기'): 0.8,
                    ('설정', '변경'): 0.7
                }
                
                action_key = (query_primary_action, ref_primary_action)
                reverse_key = (ref_primary_action, query_primary_action)
                
                # 양방향 매핑 검사
                if action_key in action_similarity_map:
                    action_match_score = action_similarity_map[action_key]
                elif reverse_key in action_similarity_map:
                    action_match_score = action_similarity_map[reverse_key]
            
            # ===== 5단계: 대상 객체 일치도 계산 =====
            object_match_score = 0.0
            if query_target_object == ref_target_object:
                # 완전 일치: 최고 점수
                object_match_score = 1.0
            elif query_target_object and ref_target_object:
                # 유사한 객체 유형 매핑 테이블
                object_similarity_map = {
                    ('번역본', '성경'): 0.8,
                    ('텍스트', '내용'): 0.7,
                    ('음성', '오디오'): 0.9,
                    ('화면', '디스플레이'): 0.7
                }
                
                object_key = (query_target_object, ref_target_object)
                reverse_key = (ref_target_object, query_target_object)
                
                # 양방향 매핑 검사
                if object_key in object_similarity_map:
                    object_match_score = object_similarity_map[object_key]
                elif reverse_key in object_similarity_map:
                    object_match_score = object_similarity_map[reverse_key]
            
            # ===== 6단계: 의미론적 키워드 일치도 계산 =====
            keyword_match_score = 0.0
            if query_semantic_keywords and ref_semantic_keywords:
                query_keyword_set = set(query_semantic_keywords)
                ref_keyword_set = set(ref_semantic_keywords)
                
                # 교집합과 합집합을 이용한 Jaccard 유사도 계산
                common_keywords = query_keyword_set & ref_keyword_set
                total_keywords = query_keyword_set | ref_keyword_set
                
                if total_keywords:
                    keyword_match_score = len(common_keywords) / len(total_keywords)
            
            # ===== 7단계: 전체 유사성 점수 계산 (가중 평균) =====
            total_score = (
                intent_match_score * 0.4 +      # 핵심 의도 일치 (40% - 가장 중요)
                action_match_score * 0.25 +     # 행동 유형 일치 (25%)
                object_match_score * 0.2 +      # 대상 객체 일치 (20%)
                keyword_match_score * 0.15      # 키워드 일치 (15%)
            )
            
            # ===== 8단계: 디버그 로깅 및 결과 반환 =====
            logging.debug(f"의도 유사성 분석: 의도={intent_match_score:.2f}, "
                         f"행동={action_match_score:.2f}, 객체={object_match_score:.2f}, "
                         f"키워드={keyword_match_score:.2f}, 전체={total_score:.2f}")
            
            # 🔍 의도 유사성 계산 상세 로그
            logging.info(f"🔍 의도 유사성 계산 상세:")
            logging.info(f"   └── 사용자 의도: {query_core_intent}")
            logging.info(f"   └── 기존 답변 의도: {ref_core_intent}")
            logging.info(f"   └── 의도 일치도: {intent_match_score:.3f} (40%)")
            logging.info(f"   └── 행동 일치도: {action_match_score:.3f} (25%)")
            logging.info(f"   └── 객체 일치도: {object_match_score:.3f} (20%)")
            logging.info(f"   └── 키워드 일치도: {keyword_match_score:.3f} (15%)")
            logging.info(f"   └── 최종 의도 관련성: {total_score:.3f}")
            
            return min(total_score, 1.0)  # 1.0을 초과하지 않도록 제한
            
        except Exception as e:
            # ===== 예외 처리: 유사성 계산 실패시 기본값 반환 =====
            logging.error(f"의도 유사성 계산 실패: {e}")
            return 0.3  # 오류시 낮은 기본값 반환
