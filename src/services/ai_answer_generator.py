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
    """AI 답변 생성 클래스"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.model = 'gpt-5-mini'
        logging.info("AIAnswerGenerator 초기화 완료")
    
    def generate_answer(self, 
                       corrected_text: str, 
                       intent_analysis: Dict, 
                       similar_answers: List[Dict], 
                       lang: str = 'ko') -> str:
        """AI 답변 생성 메인 메서드"""
        try:
            with memory_cleanup():
                logging.info("=" * 80)
                logging.info("AI 답변 생성 프로세스 시작")
                logging.info("=" * 80)
                
                # 1단계: 입력 데이터 로깅
                logging.info(f"[1단계] 입력 데이터 확인")
                logging.info(f"  - 수정된 질문: '{corrected_text}'")
                logging.info(f"  - 핵심 의도: {intent_analysis.get('core_intent', 'N/A')}")
                logging.info(f"  - 의도 카테고리: {intent_analysis.get('intent_category', 'N/A')}")
                logging.info(f"  - 검색된 참고답변 수: {len(similar_answers)}개")
                logging.info(f"  - 언어: {lang}")
                
                # 2단계: 참고답변 컨텍스트 구성
                logging.info(f"\n[2단계] 참고답변 컨텍스트 구성 시작")
                context = self._build_context(similar_answers)
                logging.info(f"  - 컨텍스트 길이: {len(context)}자")
                logging.info(f"  - 컨텍스트 미리보기:\n{context[:300]}...")
                
                # 3단계: 프롬프트 생성
                logging.info(f"\n[3단계] GPT 프롬프트 생성")
                system_prompt, user_prompt = self._create_prompts(
                    corrected_text, 
                    intent_analysis, 
                    context
                )
                logging.info(f"  - System 프롬프트 길이: {len(system_prompt)}자")
                logging.info(f"  - User 프롬프트 길이: {len(user_prompt)}자")
                
                # 4단계: GPT API 호출
                # logging.info(f"\n[4단계] GPT-5-mini API 호출 시작")
                # logging.info(f"  - 모델: {self.model}")
                # logging.info(f"  - Max tokens: 2000")
                
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_completion_tokens=2000
                )
                
                logging.info(f"  - GPT API 호출 완료")
                logging.info(f"  - 사용된 토큰: {response.usage.total_tokens if hasattr(response, 'usage') else 'N/A'}")
                
                # 5단계: GPT 원본 답변 추출
                ai_answer_raw = response.choices[0].message.content.strip()
                # logging.info(f"\n[5단계] GPT 원본 답변 추출")
                # logging.info(f"  - 원본 답변 길이: {len(ai_answer_raw)}자")
                # logging.info(f"  - 원본 답변 미리보기:\n{ai_answer_raw[:200]}...")
                
                if not ai_answer_raw:
                    logging.warning("  ⚠️ GPT 응답이 비어있음 - 폴백 답변 사용")
                    return self._get_fallback_answer()
                
                # 6단계: 인사말/끝맺음말 추가 및 HTML 포맷팅
                logging.info(f"\n[6단계] 최종 답변 포맷팅")
                final_answer = self._format_final_answer(ai_answer_raw, lang)
                
                logging.info(f"  - 최종 답변 길이: {len(final_answer)}자")
                logging.info(f"  - 인사말 포함 여부: {'안녕하세요' in final_answer}")
                logging.info(f"  - 끝맺음말 포함 여부: {'주님 안에서 평안하세요' in final_answer}")
                
                # 7단계: 완료
                logging.info("=" * 80)
                logging.info("✅ AI 답변 생성 완료")
                logging.info("=" * 80)
                
                return final_answer
                
        except Exception as e:
            logging.error("=" * 80)
            logging.error(f"❌ AI 답변 생성 실패: {str(e)}")
            logging.error(f"  - 에러 타입: {type(e).__name__}")
            logging.error("=" * 80)
            return self._get_fallback_answer()
    
    def _create_prompts(self, corrected_text: str, intent_analysis: Dict, context: str) -> tuple:
        """프롬프트 생성 (한국어 전용)"""
        
        system_prompt = """You are an AI customer service agent for GOODTV Bible Apple (바이블 애플), a Korean Christian Bible app.

    YOUR ROLE:
    1. Provide accurate, helpful answers to customer inquiries
    2. Use the provided reference answers as your PRIMARY guidance
    3. Only discuss features and functions that actually exist in the Bible Apple app

    ANSWER CONSTRUCTION RULES:
    1. PRIMARY PRINCIPLE: Stay faithful to the reference answers' content and solutions
    2. ANALYZE CAREFULLY: Understand the customer's corrected question and core intent deeply
    3. ADAPTATION: If the customer's specific situation differs from the reference answers:
    - Adapt the solution appropriately while maintaining the same tone and style
    - Keep the fundamental approach and problem-solving structure from references
    - Ensure your answer addresses the customer's actual intent
    4. TONE CONSISTENCY: Match the tone, formality level, and speaking style of the reference answers
    5. CREATIVITY CONSTRAINT: Be helpful but NOT overly creative - stick close to established patterns
    6. STEP-BY-STEP: Present solutions in clear, sequential steps when applicable
    7. TYPO REPORTS: Always explain that text corrections require app updates and take time

    SPECIAL INSTRUCTIONS FOR TYPO/TEXT ERROR REPORTS:
    - Acknowledge the report and thank them
    - Explain it requires an app update (takes time to implement)
    - DO NOT promise immediate fixes

    CRITICAL OUTPUT REQUIREMENTS:
    ⚠️ Write ONLY the main content body
    ⚠️ NO greetings (안녕하세요, etc.)
    ⚠️ NO closings (감사합니다, 평안하세요, etc.)
    ⚠️ The system will automatically add standard greetings and closings
    ⚠️ Your response MUST be in KOREAN (한국어)"""

        user_prompt = f"""CUSTOMER INQUIRY ANALYSIS:
    - Corrected Question: {corrected_text}
    - Core Intent: {intent_analysis.get('core_intent', '일반 문의')}
    - Intent Category: {intent_analysis.get('intent_category', '일반')}
    - Key Keywords: {', '.join(intent_analysis.get('semantic_keywords', [])[:5])}

    REFERENCE ANSWERS (use these as your primary guidance):
    {context}

    TASK:
    Based on the analysis above and the reference answers, create a response that:

    1. UNDERSTANDS the customer's actual problem from their corrected question and core intent
    2. USES the reference answers' solutions as your foundation
    3. ADAPTS appropriately if the customer's specific situation requires it
    4. MAINTAINS the same tone, style, and level of formality as the references
    5. PROVIDES step-by-step guidance when solving technical issues

    OUTPUT REQUIREMENTS:
    ✓ Write in Korean (한국어)
    ✓ Body content only - NO greetings or closings
    ✓ Be specific and actionable
    ✓ Use numbered steps for troubleshooting guides
    ✓ Stay within the app's actual capabilities

    ❌ DO NOT include: 안녕하세요, 감사합니다, 평안하세요, etc.
    ❌ DO NOT be overly creative - follow reference patterns
    ❌ DO NOT promise features that don't exist"""

        return system_prompt, user_prompt
    
    def _build_context(self, similar_answers: List[Dict]) -> str:
        """검색 결과를 기반으로 컨텍스트 구성"""
        if not similar_answers:
            logging.warning("  ⚠️ 참고할 유사 답변이 없음")
            return "참고할 유사 답변이 없습니다."
        
        context_parts = []
        for i, ans in enumerate(similar_answers[:3], 1):
            answer_text = ans.get('answer', '')
            
            # 참고답변에서만 인사말/끝맺음말 제거 (컨텍스트용)
            answer_text = self._remove_greetings_from_reference(answer_text)
            
            if answer_text and len(answer_text) > 20:
                score = ans.get('score', 0)
                category = ans.get('category', '기타')
                
                context_parts.append(
                    f"[참고답변 {i}] (유사도: {score:.3f}, 카테고리: {category})\n"
                    f"{answer_text[:500]}..."
                )
                
                logging.info(f"  - 참고답변 {i}: 유사도={score:.3f}, 길이={len(answer_text)}자, 카테고리={category}")
        
        if not context_parts:
            logging.warning("  ⚠️ 유효한 참고답변 없음")
            return "참고할 유사 답변이 없습니다."
        
        return "\n\n".join(context_parts)
    
    def _remove_greetings_from_reference(self, text: str) -> str:
        """참고답변에서만 인사말과 끝맺음말 제거 (컨텍스트 구성용)"""
        greeting_patterns = [
            r'^안녕하세요[^.]*\.\s*',
            r'^GOODTV\s+바이블\s*애플[^.]*\.\s*',
            r'^바이블\s*애플[^.]*\.\s*',
            r'바이블\s*애플을\s*이용해주셔서\s*감사드립니다\.\s*',
        ]
        
        closing_patterns = [
            r'\s*감사합니다[^.]*\.?\s*$',
            r'\s*평안하세요[^.]*\.?\s*$',
            r'\s*주님\s*안에서[^.]*\.?\s*$',
            r'\s*항상\s*성도님[^.]*\.?\s*$',
        ]
        
        for pattern in greeting_patterns + closing_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _format_final_answer(self, ai_content: str, lang: str) -> str:
        """
        최종 답변 포맷팅 - 인사말 + 본문 + 끝맺음말
        
        Args:
            ai_content: GPT가 생성한 본문 내용
            lang: 언어 코드
            
        Returns:
            인사말과 끝맺음말이 포함된 최종 HTML 답변
        """
        # logging.info("  [포맷팅] 최종 답변 구성 시작")
        
        # 1. GPT 답변에서 혹시 모를 인사말/끝맺음말 제거 (안전장치)
        ai_content = self._remove_greetings_from_reference(ai_content)
        # logging.info(f"  [포맷팅] 인사말/끝맺음말 제거 후 길이: {len(ai_content)}자")
        
        # 2. 간단한 HTML 변환 (줄바꿈만 처리)
        ai_content = ai_content.replace('\n\n', '<br><br>')
        ai_content = ai_content.replace('\n', '<br>')
        
        # 3. 번호 목록 강조
        ai_content = re.sub(r'(\d+)\.\s+', r'<strong>\1.</strong> ', ai_content)
        
        # 4. 고정된 인사말
        greeting = (
            "<p>안녕하세요, 바이블 애플입니다. "
            "바이블 애플을 이용해 주셔서 감사합니다.</p>"
            "<p><br></p>"
        )
        
        # 5. 본문
        body = f"<p>{ai_content}</p>"
        
        # 6. 고정된 끝맺음말
        closing = (
            "<p><br></p>"
            "<p>항상 성도님께 좋은 성경앱을 제공하기 위해 노력하는 바이블 애플이 되겠습니다.</p>"
            "<p><br></p>"
            "<p>감사합니다. 주님 안에서 평안하세요.</p>"
        )
        
        # 7. 최종 조합
        final_answer = greeting + body + closing
        
        # logging.info(f"  [포맷팅] 인사말 추가됨: {len(greeting)}자")
        # logging.info(f"  [포맷팅] 본문 길이: {len(body)}자")
        # logging.info(f"  [포맷팅] 끝맺음말 추가됨: {len(closing)}자")
        logging.info(f"  [포맷팅] 최종 답변 총 길이: {len(final_answer)}자")
        
        return final_answer
    
    def _get_fallback_answer(self) -> str:
        """오류 시 기본 답변 (인사말/끝맺음말 포함)"""
        logging.warning("폴백 답변 생성")
        
        greeting = (
            "<p>안녕하세요, 바이블 애플입니다. "
            "바이블 애플을 이용해 주셔서 감사합니다.</p>"
            "<p><br></p>"
        )
        
        body = (
            "<p>남겨주신 문의는 현재 담당자가 직접 확인하고 있습니다.</p>"
            "<p><br></p>"
            "<p>성도님께 도움이 될 수 있도록 내용을 꼼꼼히 살펴보고 "
            "정확하고 구체적인 답변을 준비하겠습니다.</p>"
            "<p><br></p>"
            "<p>답변은 최대 하루 이내에 드릴 예정이오니 "
            "조금만 기다려 주시면 감사하겠습니다.</p>"
        )
        
        closing = (
            "<p><br></p>"
            "<p>항상 성도님께 좋은 성경앱을 제공하기 위해 노력하는 바이블 애플이 되겠습니다.</p>"
            "<p><br></p>"
            "<p>감사합니다. 주님 안에서 평안하세요.</p>"
        )
        
        return greeting + body + closing