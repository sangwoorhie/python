#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
답변 생성 모델 모듈
"""

import logging
import re
from typing import Dict, List
from src.utils.memory_manager import memory_cleanup
from src.utils.text_preprocessor import TextPreprocessor

# GPT 기반 답변 생성을 담당하는 클래스
class AnswerGenerator:
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.text_processor = TextPreprocessor()
        self.gpt_model = 'gpt-3.5-turbo'
    
    def get_gpt_prompts(self, query: str, context: str, lang: str = 'ko') -> tuple:
        """언어별 GPT 프롬프트 생성"""
        if lang == 'en': # 영어
            system_prompt = """You are a GOODTV Bible App customer service representative.

Guidelines:
1. Follow the style and content of the provided reference answers faithfully
2. Find and apply solutions from similar situations in the reference answers
3. Adapt to the customer's specific situation while maintaining the tone and style of the reference answers

⚠️ Absolute Prohibitions:
- Do not guide non-existent features or menus
- Do not create specific settings methods or button locations
- If a feature is not in the reference answers, say "Sorry, this feature is currently not available"
- If uncertain, respond with "We will review this internally"

4. For feature requests or improvement suggestions, use:
   - "Thank you for your valuable feedback"
   - "We will discuss/review this internally"
   - "We will forward this as an improvement"

5. Address customers as 'Dear user' or similar polite forms
6. Use 'GOODTV Bible App' or 'Bible App' as the app name

🚫 Do NOT generate greetings or closings:
- Do not use "Hello", "Thank you", "Best regards", etc.
- Do not use "God bless", "In Christ", etc.
- Only write the main content

7. Do not use HTML tags, write in natural sentences"""

            user_prompt = f"""Customer inquiry: {query}

Reference answers (main content only, greetings and closings removed):
{context}

Based on the reference answers' solution methods and tone, write a specific answer to the customer's problem.
Important: Do not include greetings or closings. Only write the main content."""

        else:  # 한국어
            system_prompt = """당신은 GOODTV 바이블 애플 고객센터 상담원입니다.

🏆 바이블 애플 핵심 기능 (절대 준수):
- 바이블 애플은 **자체적으로 여러 번역본을 동시에 볼 수 있는 기능을 제공**합니다
- NIV, KJV, 개역개정, 개역한글 등 다양한 번역본을 **한 화면에서 비교 가능**합니다
- 다른 앱 다운로드나 외부 서비스 이용은 **절대 안내하지 마세요**
- 바이블 애플 내부 기능만으로 모든 번역본 비교가 가능합니다

🚨 절대 금지사항 (할루시네이션 방지):
- ❌ 바이블 애플 앱 외 다른 앱 다운로드 추천 금지
- ❌ 바이블 애플에 없는 기능이나 메뉴 언급 금지  
- ❌ 확실하지 않은 정보나 추측성 답변 금지
- ❌ 답변 중간에 다른 번역본이나 언어로 내용 변경 금지
- ❌ 참고답변에 없는 새로운 해결책 창작 금지

🎯 핵심 원칙 (참고답변 절대 준수):
1. **참고답변 100% 활용**: 제공된 참고답변의 해결 방법을 그대로 사용하세요
2. **질문 내용 고정**: 질문에서 언급한 번역본/기능을 절대 바꾸지 마세요
3. **일관성 철저 유지**: 답변 처음부터 끝까지 동일한 내용과 번역본 유지
4. **도메인 지식 준수**: 바이블 애플의 실제 기능 범위 내에서만 답변

📋 참고답변 활용 지침:

✅ 참고답변 분석 우선 순위:
1. 고객 질문과 의미적으로 가장 유사한 참고답변 식별
2. 해당 참고답변의 핵심 해결 단계와 방법 추출  
3. 참고답변에 명시된 구체적 기능명, 메뉴명, 버튼명 파악
4. 참고답변의 톤앤매너와 설명 스타일 학습

🔍 참고답변 기반 답변 작성:
- **핵심 해결책 유지**: 참고답변의 주요 해결 방법을 그대로 활용
- **구체적 정보 보존**: 참고답변에 나온 설정 위치, 버튼명, 메뉴 경로를 정확히 반영
- **단계별 순서 준수**: 참고답변의 해결 단계 순서를 유지하거나 개선
- **전문 용어 일치**: 참고답변에 사용된 앱 전문 용어와 표현 방식 따르기

⚠️ 참고답변 충실성 검증:
- 참고답변에 없는 새로운 기능이나 방법 추가 금지
- 참고답변과 상충되는 해결책 제시 금지
- 참고답변의 핵심 내용을 누락하거나 변형하지 말 것
- 불확실한 정보보다는 참고답변에서 확인된 내용만 활용

🚫 절대 금지사항:
- 인사말("안녕하세요", "감사합니다" 등) 사용 금지
- 끝맺음말("평안하세요", "주님 안에서" 등) 사용 금지  
- 본문 내용만 작성하고 격식적 표현 생략

🚨 빈 약속 금지 (매우 중요):
- "안내해드리겠습니다", "도움드리겠습니다", "설명드리겠습니다" 등의 약속 표현 사용 시 
  반드시 구체적인 실행 내용이 바로 뒤따라야 합니다
- 약속만 하고 실제 안내/도움/설명 내용이 없으면 절대 안됩니다
- 예시: ❌ "방법을 안내해드리겠습니다." (끝) 
         ✅ "방법을 안내해드리겠습니다. 1. 화면 상단의 설정 메뉴를 터치하세요..."

💡 참고답변 기반 구체적 작성법:
- **참고답변 단계 재현**: 참고답변의 해결 단계를 순서대로 설명
- **참고답변 용어 사용**: 참고답변에 나온 정확한 기능명과 위치 표현 활용
- **참고답변 스타일 반영**: 참고답변의 설명 방식과 구체성 수준 유지
- **검증된 정보 우선**: 참고답변에서 검증된 정보를 창의적 추측보다 우선

💡 참고답변 부족시 대응:
- 참고답변이 부족해도 그 범위 내에서만 확장하여 답변
- 참고답변의 핵심 원리를 고객 상황에 맞게 적용
- 바이블 애플의 실제 서비스 범위 내에서만 현실적인 답변 제공"""

            user_prompt = f"""고객 문의: {query}

참고 답변들 (핵심 정보):
{context}

🎯 참고답변 우선 활용 지시사항:
위 참고 답변들을 면밀히 분석하고 다음 원칙에 따라 답변하세요:

1. **참고답변 최우선 분석**: 
   - 고객 질문과 의미적으로 가장 일치하는 참고답변을 식별
   - 해당 참고답변의 해결 방법, 단계, 기능명을 정확히 파악
   - 참고답변에 나온 구체적 용어와 설명 방식을 학습

2. **참고답변 충실한 활용**:
   - 참고답변의 핵심 해결책을 그대로 활용하여 답변 작성
   - 참고답변에 명시된 설정 위치, 버튼명, 메뉴 경로를 정확히 반영
   - 참고답변의 단계별 순서와 설명 스타일을 따라 답변 구성
   - 참고답변에 사용된 전문 용어와 표현 방식을 동일하게 사용

3. **참고답변 기반 확장**:
   - 참고답변의 범위 내에서만 고객 상황에 맞게 내용 조정
   - 참고답변에 없는 새로운 기능이나 방법 추가 절대 금지
   - 참고답변과 상충되는 해결책 제시 금지

🚨 필수 요구사항:
1. **참고답변 우선**: 창의적 해결책보다 참고답변의 검증된 방법 우선 활용
2. **구체적 실행**: "안내해드리겠습니다" 등의 약속 후 반드시 구체적 내용 제시
3. **정확한 용어**: 참고답변의 정확한 기능명, 메뉴명, 버튼명 사용
4. **단계별 설명**: 참고답변의 해결 단계를 순서대로 명확히 설명
5. **본문만 작성**: 인사말이나 끝맺음말 없이 핵심 내용만 작성

🔒 할루시네이션 엄격 금지:
- 질문에서 언급한 번역본이나 기능을 절대 바꾸지 마세요
- 답변 중간에 다른 내용으로 변경하는 것을 절대 금지합니다
- 바이블 애플 외부 앱이나 서비스 추천을 절대 하지 마세요
- 참고답변에 없는 기능이나 방법을 창작하지 마세요
- 확실하지 않은 정보는 절대 언급하지 마세요

✅ 일관성 검증:
- 답변 전체에서 동일한 번역본/기능 유지
- 질문의 핵심 요구사항에서 절대 벗어나지 않기
- 바이블 애플 자체 기능만으로 해결책 제시

❌ 절대 금지: 참고답변 무시, 외부 앱 추천, 내용 변경
✅ 반드시 준수: 참고답변 방법을 질문에 정확히 적용, 일관성 유지

지금 즉시 참고답변에 100% 충실하면서 질문 내용을 절대 바꾸지 않고 답변하세요."""

        return system_prompt, user_prompt

    def generate_with_enhanced_gpt(self, query: str, similar_answers: list, context_analysis: dict, lang: str = 'ko') -> str:
        """향상된 GPT 생성 - 일관성과 품질 보장"""
        try:
            with memory_cleanup():
                approach = context_analysis['recommended_approach']
                context = self.create_enhanced_context(similar_answers, target_lang=lang)
                
                if not context:
                    logging.warning("유효한 컨텍스트가 없어 GPT 생성 중단")
                    return ""
                
                # 통일된 프롬프트 생성
                system_prompt, user_prompt = self.get_gpt_prompts(query, context, lang)
                
                # 일관성을 위한 보수적 temperature 설정
                if approach == 'gpt_with_strong_context':
                    temperature = 0.3 if context_analysis.get('context_relevance') == 'high' else 0.4
                    max_tokens = 700
                elif approach == 'gpt_with_weak_context':
                    temperature = 0.4
                    max_tokens = 650
                else: # fallback이나 기타
                    return ""
                
                # 답변 품질 보장을 위한 3회 재시도 메커니즘
                max_attempts = 3
                for attempt in range(max_attempts):
                    # GPT API 호출
                    response = self.openai_client.chat.completions.create(
                        model=self.gpt_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=0.9,
                        frequency_penalty=0.1,
                        presence_penalty=0.1
                    )
                    
                    generated = response.choices[0].message.content.strip()
                    del response
                    
                    # 생성된 텍스트 정리
                    generated = self.text_processor.clean_generated_text(generated)
                    
                    # 최소 길이 검증
                    if len(generated.strip()) >= 20:
                        logging.info(f"GPT 생성 성공 (시도 #{attempt+1}, {approach}): {len(generated)}자")
                        return generated
                    
                    # 마지막 시도가 아니면 temperature 조정
                    if attempt < max_attempts - 1:
                        temperature = min(temperature + 0.1, 0.6)
                
                logging.warning("모든 GPT 생성 시도 실패")
                return ""
                
        except Exception as e:
            logging.error(f"향상된 GPT 생성 실패: {e}")
            return ""

    def create_enhanced_context(self, similar_answers: list, max_answers: int = 7, target_lang: str = 'ko') -> str:
        """GPT용 향상된 컨텍스트 생성"""
        if not similar_answers:
            return ""
        
        context_parts = []
        used_answers = 0
        
        # 유사도 점수에 따른 답변 그룹핑
        high_score = [ans for ans in similar_answers if ans['score'] >= 0.7]
        medium_score = [ans for ans in similar_answers if 0.5 <= ans['score'] < 0.7]
        medium_low_score = [ans for ans in similar_answers if 0.5 <= ans['score'] < 0.6]

        # 1단계: 고품질 답변 우선 포함 (최대 4개)
        for ans in high_score[:4]:
            if used_answers >= max_answers:
                break
            
            clean_answer = self.text_processor.preprocess_text(ans['answer'])
            clean_answer = self.remove_greeting_and_closing(clean_answer, 'ko')
            
            # 영어 질문인 경우 답변을 번역
            if target_lang == 'en' and ans.get('lang', 'ko') == 'ko':
                clean_answer = self.translate_text(clean_answer, 'ko', 'en')
            
            if len(clean_answer.strip()) > 20:
                context_parts.append(f"[참고답변 {used_answers+1} - 점수: {ans['score']:.2f}]\n{clean_answer[:400]}")
                used_answers += 1
        
        # 2단계: 중품질 답변으로 보완 (최대 3개)
        for ans in medium_score[:3]:
            if used_answers >= max_answers:
                break
            
            clean_answer = self.text_processor.preprocess_text(ans['answer'])
            clean_answer = self.remove_greeting_and_closing(clean_answer, 'ko')
            
            if target_lang == 'en' and ans.get('lang', 'ko') == 'ko':
                clean_answer = self.translate_text(clean_answer, 'ko', 'en')
            
            if len(clean_answer.strip()) > 20:
                context_parts.append(f"[참고답변 {used_answers+1} - 점수: {ans['score']:.2f}]\n{clean_answer[:300]}")
                used_answers += 1

        # 3단계: 답변이 부족한 경우 중간 품질 답변 추가
        if used_answers < 3:
            for ans in medium_low_score[:2]:
                if used_answers >= max_answers:
                    break
                
                clean_answer = self.text_processor.preprocess_text(ans['answer'])
                clean_answer = self.remove_greeting_and_closing(clean_answer, 'ko')
                
                if target_lang == 'en' and ans.get('lang', 'ko') == 'ko':
                    clean_answer = self.translate_text(clean_answer, 'ko', 'en')
                
                if len(clean_answer.strip()) > 20:
                    context_parts.append(f"[참고답변 {used_answers+1} - 점수: {ans['score']:.2f}]\n{clean_answer[:250]}")
                    used_answers += 1
        
        logging.info(f"컨텍스트 생성: {used_answers}개의 답변 포함 (언어: {target_lang})")
        
        return "\n\n" + "="*50 + "\n\n".join(context_parts)

    def remove_greeting_and_closing(self, text: str, lang: str = 'ko') -> str:
        """참고 답변에서 인사말과 끝맺음말을 제거하는 메서드"""
        if not text:
            return ""
        
        if lang == 'ko':
            greeting_patterns = [
                r'^안녕하세요[^.]*\.\s*',
                r'^GOODTV\s+바이블\s*애플[^.]*\.\s*',
                r'^바이블\s*애플[^.]*\.\s*',
                r'^성도님[^.]*\.\s*',
                r'^고객님[^.]*\.\s*',
                r'^감사합니다[^.]*\.\s*',
                r'^감사드립니다[^.]*\.\s*',
                r'^바이블\s*애플을\s*이용해주셔서[^.]*\.\s*',
                r'^바이블\s*애플을\s*애용해\s*주셔서[^.]*\.\s*'
            ]
            
            closing_patterns = [
                r'\s*감사합니다[^.]*\.?\s*$',
                r'\s*감사드립니다[^.]*\.?\s*$',
                r'\s*평안하세요[^.]*\.?\s*$',
                r'\s*주님\s*안에서[^.]*\.?\s*$',
                r'\s*함께\s*기도하며[^.]*\.?\s*$',
                r'\s*항상[^.]*바이블\s*애플[^.]*\.?\s*$',
                r'\s*항상\s*주님\s*안에서[^.]*\.?\s*$',
                r'\s*주님\s*안에서\s*평안하세요[^.]*\.?\s*$',
                r'\s*주님의\s*은총이[^.]*\.?\s*$',
                r'\s*기도드리겠습니다[^.]*\.?\s*$'
            ]
        else:  # 영어
            greeting_patterns = [
                r'^Hello[^.]*\.\s*',
                r'^Hi[^.]*\.\s*',
                r'^Dear[^.]*\.\s*',
                r'^Thank you[^.]*\.\s*',
                r'^Thanks[^.]*\.\s*',
                r'^This is GOODTV Bible App[^.]*\.\s*',
            ]
            
            closing_patterns = [
                r'\s*Thank you[^.]*\.?\s*$',
                r'\s*Thanks[^.]*\.?\s*$',
                r'\s*Best regards[^.]*\.?\s*$',
                r'\s*Sincerely[^.]*\.?\s*$',
                r'\s*God bless[^.]*\.?\s*$',
                r'\s*May God[^.]*\.?\s*$',
            ]
        
        # 인사말 제거
        for pattern in greeting_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # 끝맺음말 제거
        for pattern in closing_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        text = text.strip()
        return text

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """GPT를 사용한 번역"""
        try:
            lang_map = {
                'ko': 'Korean',
                'en': 'English'
            }
            
            system_prompt = f"You are a professional translator. Translate the following text from {lang_map[source_lang]} to {lang_map[target_lang]}. Keep the same tone and style. Only provide the translation without any explanation."
            
            response = self.openai_client.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                max_tokens=600,
                temperature=0.5
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logging.error(f"번역 실패: {e}")
            return text
