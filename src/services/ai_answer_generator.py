#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
간소화된 AI 답변 생성 서비스
- 복잡한 로직 제거
- 통합 분석 결과와 검색 결과를 활용한 간단한 답변 생성
"""

import logging
import re
from typing import Dict, List
from src.utils.memory_manager import memory_cleanup


class AIAnswerGenerator:
    """ AI 답변 생성 클래스"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.model = 'gpt-5-mini'
    
    def generate_answer(self, 
                       corrected_text: str, 
                       intent_analysis: Dict, 
                       similar_answers: List[Dict], 
                       lang: str = 'ko') -> str:
        try:
            with memory_cleanup():
                logging.info("간소화된 AI 답변 생성 시작")
                
                # 1. 참고답변 컨텍스트 구성
                context = self._build_context(similar_answers)
                
                # 2. 프롬프트 생성
                system_prompt, user_prompt = self._create_prompts(
                    corrected_text, 
                    intent_analysis, 
                    context
                )
                
                # 3. GPT API 호출
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_completion_tokens=2000
                )
                
                # 4. 답변 추출
                ai_answer = response.choices[0].message.content.strip()
                
                if not ai_answer:
                    logging.warning("GPT 응답이 비어있음")
                    return self._get_fallback_answer()
                
                # 5. HTML 포맷팅 적용
                formatted_answer = self._apply_html_formatting(ai_answer, lang)
                
                logging.info(f"답변 생성 완료: {len(formatted_answer)}자")
                return formatted_answer
                
        except Exception as e:
            logging.error(f"AI 답변 생성 실패: {str(e)}")
            return self._get_fallback_answer()
    
    def _create_prompts(self, corrected_text: str, intent_analysis: Dict, context: str) -> tuple:
        """프롬프트 생성 (한국어 전용)"""
        
        system_prompt = """당신은 GOODTV 바이블 앱 고객센터 상담원입니다.

주요 역할:
1. 고객의 문의에 대해 정확하고 친절한 답변 제공
2. 제공된 참고답변을 활용하여 일관성 있는 답변 작성
3. 오탈자 신고는 개발팀 확인 후 업데이트로 처리됨을 안내

답변 원칙:
- 참고답변의 해결 방법을 우선적으로 활용
- 바이블 앱의 실제 기능 범위 내에서만 답변
- 구체적이고 실행 가능한 해결책 제시
- 인사말/끝맺음말 없이 본문만 작성"""

        user_prompt = f"""고객 문의 분석 결과:
- 수정된 질문: {corrected_text}
- 핵심 의도: {intent_analysis.get('core_intent', '일반 문의')}
- 의도 카테고리: {intent_analysis.get('intent_category', '일반')}
- 키워드: {', '.join(intent_analysis.get('semantic_keywords', [])[:5])}

참고답변:
{context}

위 분석 결과와 참고답변을 바탕으로 고객의 문의에 대한 답변을 작성해주세요."""

        return system_prompt, user_prompt
    
    def _build_context(self, similar_answers: List[Dict]) -> str:
        """검색 결과를 기반으로 간단한 컨텍스트 구성"""
        if not similar_answers:
            return "참고할 유사 답변이 없습니다."
        
        context_parts = []
        for i, ans in enumerate(similar_answers[:3], 1):
            answer_text = ans.get('answer', '')
            answer_text = self._remove_greetings(answer_text)
            
            if answer_text and len(answer_text) > 20:
                context_parts.append(
                    f"[참고답변 {i}] (유사도: {ans.get('score', 0):.3f})\n"
                    f"카테고리: {ans.get('category', '기타')}\n"
                    f"답변: {answer_text[:400]}..."
                )
        
        if not context_parts:
            return "참고할 유사 답변이 없습니다."
            
        return "\n\n".join(context_parts)
    
    def _remove_greetings(self, text: str) -> str:
        """인사말과 끝맺음말 제거"""
        greeting_patterns = [
            r'^안녕하세요[^.]*\.\s*',
            r'^GOODTV\s+바이블\s*애플[^.]*\.\s*',
            r'^바이블\s*애플[^.]*\.\s*',
            r'^감사합니다[^.]*\.\s*',
        ]
        
        closing_patterns = [
            r'\s*감사합니다[^.]*\.?\s*$',
            r'\s*평안하세요[^.]*\.?\s*$',
            r'\s*주님\s*안에서[^.]*\.?\s*$',
        ]
        
        for pattern in greeting_patterns + closing_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _apply_html_formatting(self, text: str, lang: str) -> str:
        """HTML 포맷팅 적용"""
        # 줄바꿈 처리
        text = text.replace('\n', '<br>')
        
        # 번호 목록 처리
        text = re.sub(r'^(\d+)\.\s+', r'<strong>\1.</strong> ', text, flags=re.MULTILINE)
        
        # 기본 HTML 래퍼
        formatted_text = f'<div class="ai-answer">{text}</div>'
        
        return formatted_text
    
    def _get_fallback_answer(self) -> str:
        """오류 시 기본 답변"""
        fallback_text = """남겨주신 문의는 현재 담당자가 직접 확인하고 있습니다.
성도님께 도움이 될 수 있도록 내용을 꼼꼼히 살펴
정확하고 구체적인 답변을 준비하겠습니다.
답변은 최대 하루 이내에 드릴 예정이오니
조금만 기다려 주시면 감사하겠습니다."""
        
        return self._apply_html_formatting(fallback_text, 'ko')