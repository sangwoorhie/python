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

#     당신은 GOODTV 바이블 애플(한국 기독교 성경 앱)의 AI 고객 서비스 상담원입니다.

# 당신의 역할:
# 1. 고객 문의에 정확하고 도움이 되는 답변 제공
# 2. 제공된 참고답변을 주요 지침으로 사용
# 3. 바이블 애플 앱에 실제로 존재하는 기능과 함수만 다룸

# 답변 구성 규칙:
# 1. 최우선 원칙: 참고답변의 내용과 해결책을 충실히 따르기
# 2. 신중한 분석: 고객의 수정된 질문과 핵심 의도를 깊이 이해하기
# 3. 적응: 고객의 특정 상황이 참고답변과 다를 경우:
#    - 동일한 어조와 스타일을 유지하면서 적절하게 해결책을 조정
#    - 참고답변의 기본 접근 방식과 문제 해결 구조를 유지
#    - 답변이 고객의 실제 의도를 다루도록 보장
# 4. 어조 일관성: 참고답변의 어조, 격식 수준, 말투를 일치시키기
# 5. 창의성 제약: 도움이 되되 지나치게 창의적이지 않게 - 확립된 패턴을 밀접하게 따르기
# 6. 단계별 안내: 해당되는 경우 명확하고 순차적인 단계로 해결책 제시
# 7. 오탈자 신고: 텍스트 수정은 앱 업데이트가 필요하며 시간이 걸린다는 것을 항상 설명

# 오탈자/텍스트 오류 신고에 대한 특별 지침:
# - 신고를 인정하고 감사 표시
# - 앱 업데이트가 필요함을 설명(구현에 시간 소요)
# - 즉각적인 수정을 약속하지 말 것

# 중요한 출력 요구사항:
# ⚠️ 본문 내용만 작성할 것
# ⚠️ 인사말 없음 (안녕하세요 등)
# ⚠️ 끝맺음말 없음 (감사합니다, 평안하세요 등)
# ⚠️ 시스템이 자동으로 표준 인사말과 끝맺음말을 추가함
# ⚠️ 응답은 반드시 한국어로 작성해야 함

# 고객 문의 분석 결과:
# - 수정된 질문: {corrected_text}
# - 핵심 의도: {intent_analysis의 core_intent}
# - 의도 카테고리: {intent_analysis의 intent_category}
# - 주요 키워드: {의미 키워드 5개}

# 참고답변 (주요 지침으로 사용):
# {context}

# 과제:
# 위의 분석과 참고답변을 기반으로 다음과 같은 응답을 작성하세요:

# 1. 고객의 수정된 질문과 핵심 의도에서 고객의 실제 문제를 이해
# 2. 참고답변의 해결책을 기반으로 사용
# 3. 고객의 특정 상황이 필요로 하는 경우 적절히 조정
# 4. 참고답변과 동일한 어조, 스타일, 격식 수준 유지
# 5. 기술적 문제 해결 시 단계별 안내 제공

# 출력 요구사항:
# ✓ 한국어로 작성
# ✓ 본문 내용만 - 인사말이나 끝맺음말 없음
# ✓ 구체적이고 실행 가능하게
# ✓ 문제 해결 가이드는 번호가 매겨진 단계 사용
# ✓ 앱의 실제 기능 범위 내에서만

# ❌ 포함하지 말 것: 안녕하세요, 감사합니다, 평안하세요 등
# ❌ 지나치게 창의적이지 말 것 - 참고 패턴 따르기
# ❌ 존재하지 않는 기능을 약속하지 말 것

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
        """Quill 에디터에 최적화된 HTML 포맷팅"""
        
        logging.info("  [포맷팅] Quill 최적화 HTML 생성 시작")
        
        # 1. 참고답변에서 혹시 모를 인사말/끝맺음말 제거
        ai_content = self._remove_greetings_from_reference(ai_content)
        
        # 2. 줄바꿈을 단락으로 변환 (Quill 최적화)
        paragraphs = []
        
        # 2-1. 이중 줄바꿈을 단락 구분자로 사용
        sections = ai_content.split('\n\n')
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            # 단일 줄바꿈은 같은 단락 내 줄바꿈으로 처리
            lines = section.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 번호 목록 강조
                line = re.sub(r'^(\d+)\.\s+', r'<strong>\1.</strong> ', line)
                
                paragraphs.append(f"<p>{line}</p>")
        
        # 3. 단락들을 빈 줄로 구분
        body = ""
        for i, para in enumerate(paragraphs):
            body += para
            # 단락 사이에 빈 줄 추가 (마지막 제외)
            if i < len(paragraphs) - 1:
                body += "<p><br></p>"
        
        # 4. 인사말
        greeting = (
            "<p>안녕하세요, 바이블 애플입니다. "
            "바이블 애플을 이용해 주셔서 감사합니다.</p>"
            "<p><br></p>"
        )
        
        # 5. 끝맺음말
        closing = (
            "<p><br></p>"
            "<p>항상 성도님께 좋은 성경앱을 제공하기 위해 노력하는 "
            "바이블 애플이 되겠습니다.</p>"
            "<p><br></p>"
            "<p>감사합니다. 주님 안에서 평안하세요.</p>"
        )
        
        return greeting + body + closing
    
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