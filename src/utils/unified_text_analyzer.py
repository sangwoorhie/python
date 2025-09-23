#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
통합 텍스트 분석기 모듈
- 오타 수정과 의도 분석을 한 번의 GPT 호출로 처리
- API 비용 절감 및 처리 성능 최적화
"""

import logging
import json
from typing import Dict, Tuple
from src.utils.memory_manager import memory_cleanup

class UnifiedTextAnalyzer:
    """오타 수정 + 의도 분석을 통합한 분석기"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.model = 'gpt-5-mini'
    
    # 한 번의 GPT 호출로 오타 수정과 의도 분석을 동시에 수행    
    # Args:
    #     text: 분석할 텍스트
    # Returns:
    #     Tuple[str, Dict]: (수정된_텍스트, 의도_분석_결과)
    def analyze_and_correct(self, text: str) -> Tuple[str, Dict]:
        try:
            with memory_cleanup():
                logging.info(f"통합 분석기 시작: 입력 텍스트='{text}'")
                
                # 통합 시스템 프롬프트
                system_prompt = """당신은 바이블 앱 문의 전문 분석가입니다.
    사용자의 질문에 대해 다음 두 가지 작업을 동시에 수행하세요:

    1. 맞춤법 및 오타 수정 (의미와 어조는 유지)
    2. 질문의 본질적 의도 분석

    응답 형식 (JSON):
    {
        "corrected_text": "오타가 수정된 텍스트",
        "intent_analysis": {
            "core_intent": "핵심 의도 (표준화된 형태)",
            "intent_category": "의도 카테고리",
            "primary_action": "주요 행동",
            "target_object": "대상 객체",
            "constraint_conditions": ["제약 조건들"],
            "standardized_query": "표준화된 질문 형태",
            "semantic_keywords": ["의미론적 핵심 키워드들"]
        }
    }

    오타 수정 지침:
    - 앱/어플리케이션 → 앱으로 통일
    - 띄어쓰기, 맞춤법, 조사 사용법 정확히 교정
    - 원문의 의미와 어조는 절대 변경 금지

    의도 분석 지침:
    - 질문의 본질적 목적 파악
    - 구체적 예시 제거하고 일반화
    - 의미론적으로 동등한 질문들이 같은 결과 도출하도록 분석

    중요: 반드시 유효한 JSON 형식으로만 응답하고, 다른 설명은 포함하지 마세요."""

                user_prompt = f"다음 텍스트를 분석해주세요:\n\n{text}"
                
                # GPT API 호출 (gpt-5-mini 모델에 맞는 파라미터 사용)
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_completion_tokens=600
                    # temperature 파라미터 제거 (gpt-5-mini에서 지원하지 않음)
                )
                
                result_text = response.choices[0].message.content.strip()
                
                # 🔍 GPT 응답 검증 및 로깅 강화
                logging.info(f"통합 분석 - GPT 원본 응답: {result_text}")
                
                # 빈 응답 체크
                if not result_text or result_text.isspace():
                    logging.error("GPT 응답이 비어있음 - 기본값 반환")
                    return text, self._get_default_intent_analysis(text)
                
                # JSON 파싱 시도
                try:
                    result = json.loads(result_text)
                    corrected_text = result.get('corrected_text', text)
                    intent_analysis = result.get('intent_analysis', {})
                    
                    # 기존 호환성을 위한 필드 추가
                    intent_analysis.update({
                        'intent_type': intent_analysis.get('intent_category', '일반문의'),
                        'main_topic': intent_analysis.get('target_object', '기타'),
                        'specific_request': intent_analysis.get('standardized_query', text[:100]),
                        'keywords': intent_analysis.get('semantic_keywords', [text[:20]]),
                        'urgency': 'medium',
                        'action_type': intent_analysis.get('primary_action', '기타')
                    })
                    
                    # 상세 결과 로그
                    logging.info(f"통합 분석 완료 - 수정된 텍스트: '{corrected_text}'")
                    logging.info(f"통합 분석 완료 - 의도 분석: {json.dumps(intent_analysis, ensure_ascii=False)}")
                    
                    # 로깅
                    if corrected_text != text:
                        logging.info(f"통합 분석 - 오타 수정: '{text[:50]}...' → '{corrected_text[:50]}...'")
                    
                    logging.info(f"통합 분석 - 의도: {intent_analysis.get('core_intent', 'N/A')}")
                    
                    # 🔍 디버그: 파싱된 결과 출력
                    logging.info("�� [사용자 질문 의도 분석 결과]")
                    logging.info(f"입력 텍스트: {text}")
                    logging.info(f"수정된 텍스트: {corrected_text}")
                    logging.info(f"핵심 의도: {intent_analysis.get('core_intent', 'N/A')}")
                    logging.info(f"주요 행동: {intent_analysis.get('primary_action', 'N/A')}")
                    logging.info(f"대상 객체: {intent_analysis.get('target_object', 'N/A')}")
                    logging.info(f"의미론적 키워드: {intent_analysis.get('semantic_keywords', [])}")
                    logging.info(f"전체 의도 분석: {json.dumps(intent_analysis, ensure_ascii=False, indent=2)}")
                    logging.info("="*60)

                    return corrected_text, intent_analysis
                    
                except json.JSONDecodeError as e:
                    logging.error(f"통합 분석 JSON 파싱 실패: {e}")
                    logging.error(f"파싱 실패한 응답: {result_text}")
                    
                    # JSON 파싱 실패시 텍스트 기반 파싱 시도
                    corrected_text, intent_analysis = self._parse_text_response(result_text, text)
                    
                    if not intent_analysis:
                        logging.warning("텍스트 파싱도 실패, 기본값 반환")
                        return text, self._get_default_intent_analysis(text)
                    
        except Exception as e:
            logging.error(f"통합 텍스트 분석 실패: {e}")
            return text, self._get_default_intent_analysis(text)

def _parse_text_response(self, response_text: str, original_text: str) -> Tuple[str, Dict]:
    """텍스트 응답을 파싱하여 의도 분석 결과 추출"""
    try:
        # 기본값 설정
        corrected_text = original_text
        intent_analysis = {
            "core_intent": "general_inquiry",
            "intent_category": "일반문의",
            "primary_action": "기타",
            "target_object": "기타",
            "constraint_conditions": [],
            "standardized_query": original_text,
            "semantic_keywords": [original_text[:20]],
            "intent_type": "일반문의",
            "main_topic": "기타",
            "specific_request": original_text[:100],
            "keywords": [original_text[:20]],
            "urgency": "medium",
            "action_type": "기타"
        }
        
        # 텍스트에서 키워드 추출 시도
        if "오타" in response_text or "수정" in response_text:
            # 오타 수정이 있는 경우
            lines = response_text.split('\n')
            for line in lines:
                if "수정된" in line or "corrected" in line.lower():
                    corrected_text = line.split(':')[-1].strip() if ':' in line else line.strip()
                    break
        
        # 의도 분석 키워드 추출
        if "의도" in response_text or "intent" in response_text.lower():
            lines = response_text.split('\n')
            for line in lines:
                if "핵심" in line or "core" in line.lower():
                    intent_analysis["core_intent"] = line.split(':')[-1].strip() if ':' in line else "general_inquiry"
                elif "행동" in line or "action" in line.lower():
                    intent_analysis["primary_action"] = line.split(':')[-1].strip() if ':' in line else "기타"
                elif "대상" in line or "target" in line.lower():
                    intent_analysis["target_object"] = line.split(':')[-1].strip() if ':' in line else "기타"
        
        logging.info(f"텍스트 파싱 결과 - 수정된 텍스트: '{corrected_text}'")
        logging.info(f"텍스트 파싱 결과 - 의도 분석: {json.dumps(intent_analysis, ensure_ascii=False)}")
        
        return corrected_text, intent_analysis
        
    except Exception as e:
        logging.error(f"텍스트 파싱 실패: {e}")
        return original_text, self._get_default_intent_analysis(original_text)
    
    def _get_default_intent_analysis(self, text: str) -> Dict:
        """분석 실패시 기본 의도 분석 결과"""
        return {
            "core_intent": "general_inquiry",
            "intent_category": "일반문의",
            "primary_action": "기타",
            "target_object": "기타",
            "constraint_conditions": [],
            "standardized_query": text,
            "semantic_keywords": [text[:20]],
            # 기존 호환성 필드
            "intent_type": "일반문의",
            "main_topic": "기타",
            "specific_request": text[:100],
            "keywords": [text[:20]],
            "urgency": "medium",
            "action_type": "기타"
        }
