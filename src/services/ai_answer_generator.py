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
    """간소화된 AI 답변 생성 클래스"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.model = 'gpt-5-mini'
    
    def generate_answer(self, 
                       corrected_text: str, 
                       intent_analysis: Dict, 
                       similar_answers: List[Dict], 
                       lang: str = 'ko') -> str:
        """
        간단한 AI 답변 생성
        
        Args:
            corrected_text: 오타 수정된 텍스트
            intent_analysis: 의도 분석 결과
            similar_answers: 검색된 유사 답변들
            lang: 언어 코드 (한국어 고정)
            
        Returns:
            str: 생성된 답변 (HTML 포맷 적용)
        """
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
        
        system_prompt = """You are a customer service representative for the GOODTV Bible Apple App support center.

Key Roles:
1. Provide accurate and friendly responses to customer inquiries.
2. Utilize the provided reference answers to create consistent replies.
3. For typo reports, inform that they will be reviewed by the development team and addressed in updates.

Response Principles:
- Prioritize solutions from the reference answers.
- Limit responses to the actual features and scope of the Bible app.
- Offer specific and actionable solutions.
- Write only the main content, without greetings or closing remarks.

Prohibitions:
- Do not recommend any apps other than the GOODTV Bible Apple App.
- Avoid providing uncertain information or speculative answers.
- Do not invent new features not present in the reference answers."""

        user_prompt = f"""Customer Inquiry Analysis Results:
- Corrected Question: {corrected_text}
- Core Intent: {intent_analysis.get('core_intent', 'General Inquiry')}
- Intent Category: {intent_analysis.get('intent_category', 'General')}
- Primary Action: {intent_analysis.get('primary_action', 'Needs Confirmation')}
- Keywords: {', '.join(intent_analysis.get('semantic_keywords', [])[:5])}

Reference Answers (Sorted by Similarity):
{context}

Based on the above analysis results and reference answers, please create a clear and helpful response to the customer's inquiry.
For typo reports, include guidance on development team review and updates. For usage inquiries, provide detailed step-by-step guides."""

        return system_prompt, user_prompt
    
    def _build_context(self, similar_answers: List[Dict]) -> str:
        """검색 결과를 기반으로 간단한 컨텍스트 구성"""
        if not similar_answers:
            return "참고할 유사 답변이 없습니다."
        
        context_parts = []
        for i, ans in enumerate(similar_answers[:3], 1):  # 상위 3개만 사용
            # 답변 텍스트 전처리
            answer_text = ans.get('answer', '')
            answer_text = self._remove_greetings(answer_text)
            
            if answer_text and len(answer_text) > 20:
                context_parts.append(
                    f"[참고답변 {i}] (유사도: {ans.get('score', 0):.3f})\n"
                    f"카테고리: {ans.get('category', '기타')}\n"
                    f"답변: {answer_text[:400]}..."
                )
                
                # 디버깅용 로그
                logging.debug(f"참고답변 {i} 추가: 유사도={ans.get('score', 0):.3f}, "
                            f"카테고리={ans.get('category', '기타')}")
        
        if not context_parts:
            return "참고할 유사 답변이 없습니다."
            
        return "\n\n".join(context_parts)
    
    def _remove_greetings(self, text: str) -> str:
        """인사말과 끝맺음말 제거"""
        # 인사말 패턴
        greeting_patterns = [
            r'^안녕하세요[^.]*\.\s*',
            r'^GOODTV\s+바이블\s*애플[^.]*\.\s*',
            r'^바이블\s*애플[^.]*\.\s*',
            r'^성도님[^.]*\.\s*',
            r'^고객님[^.]*\.\s*',
            r'^감사합니다[^.]*\.\s*',
            r'^감사드립니다[^.]*\.\s*',
        ]
        
        # 끝맺음말 패턴
        closing_patterns = [
            r'\s*감사합니다[^.]*\.?\s*$',
            r'\s*감사드립니다[^.]*\.?\s*$',
            r'\s*평안하세요[^.]*\.?\s*$',
            r'\s*주님\s*안에서[^.]*\.?\s*$',
            r'\s*항상[^.]*바이블\s*애플[^.]*\.?\s*$',
        ]
        
        # 패턴 적용
        for pattern in greeting_patterns + closing_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _apply_html_formatting(self, text: str, lang: str) -> str:
        """HTML 포맷팅 적용"""
        # 기본 줄바꿈 처리
        text = text.replace('\n', '<br>')
        
        # 번호 목록 처리 (1. 2. 3. 형식)
        text = re.sub(r'^(\d+)\.\s+', r'<strong>\1.</strong> ', text, flags=re.MULTILINE)
        
        # 강조 표시 처리
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        
        # 기본 HTML 래퍼
        formatted_text = f'<div class="ai-answer">{text}</div>'
        
        return formatted_text
    
    def _get_fallback_answer(self) -> str:
        """오류 시 기본 답변"""
        # 인사말과 끝맺음말은 자동 제거되므로 본문만 작성
        fallback_text = """남겨주신 문의는 현재 담당자가 직접 확인하고 있습니다.
    성도님께 도움이 될 수 있도록 내용을 꼼꼼히 살펴
    정확하고 구체적인 답변을 준비하겠습니다.
    답변은 최대 하루 이내에 드릴 예정이오니
    조금만 기다려 주시면 감사하겠습니다."""
        
        # HTML 포맷팅 적용
        return self._apply_html_formatting(fallback_text, 'ko')