#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
답변 생성 모델 모듈
- GPT 기반 AI 답변 생성
- 다국어 지원 (한국어/영어)
- 컨텍스트 기반 답변 품질 최적화
"""

import logging
import re
from typing import Dict, List
from src.utils.memory_manager import memory_cleanup
from src.utils.text_preprocessor import TextPreprocessor

# ===== GPT 기반 답변 생성을 담당하는 메인 클래스 =====
class AnswerGenerator:
    
    # AnswerGenerator 초기화
    # Args:
    #     openai_client: OpenAI API 클라이언트 인스턴스
    def __init__(self, openai_client):
        self.openai_client = openai_client                # OpenAI API 클라이언트
        self.text_processor = TextPreprocessor()          # 텍스트 전처리 도구
        self.gpt_model = 'gpt-4o'                        # 사용할 GPT 모델
    
    # 언어별 GPT 프롬프트 생성 - 한국어/영어 지원
    # Args:
    #     query: 사용자 질문
    #     context: 참고답변 컨텍스트
    #     lang: 언어 코드 ('ko' 또는 'en')
    # Returns:
    #     tuple: (시스템 프롬프트, 사용자 프롬프트)
    def get_gpt_prompts(self, query: str, context: str, lang: str = 'ko') -> tuple:
        # ===== 언어별 프롬프트 생성 =====
        if lang == 'en': # 영어 프롬프트
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

        else:  # 한국어 프롬프트 (기본값)
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

🔍 질문 유형 정확한 이해 (매우 중요):
- "오탈자가 있어요", "수정해주세요", "잘못되어 있어요" → **앱 개발팀 신고 사안**
- "어떻게 바꾸나요", "설정 방법", "사용법" → **사용자 가이드 제공**
- **오탈자 신고는 개발팀 확인 후 업데이트로 해결되는 사안입니다**

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
1. **질문 유형 먼저 판단**: 
   - 오탈자/오류 신고 → 개발팀 확인 및 업데이트 안내
   - 사용법 문의 → 구체적 가이드 제공
2. **참고답변 우선**: 창의적 해결책보다 참고답변의 검증된 방법 우선 활용
3. **구체적 실행**: "안내해드리겠습니다" 등의 약속 후 반드시 구체적 내용 제시
4. **정확한 용어**: 참고답변의 정확한 기능명, 메뉴명, 버튼명 사용
5. **단계별 설명**: 참고답변의 해결 단계를 순서대로 명확히 설명
6. **본문만 작성**: 인사말이나 끝맺음말 없이 핵심 내용만 작성

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

🆘 오탈자/오류 신고 질문 처리 가이드:
현재 질문이 "오탈자가 있어요", "수정해주세요", "잘못되어 있어요" 등의 신고성 문의라면:
1. 사용자가 설정을 바꾸려는 것이 아님을 이해하세요
2. 앱 개발팀이 확인하고 업데이트로 해결할 사안임을 안내하세요  
3. 번역본 비교나 설정 변경 방법을 안내하지 마세요
4. 참고답변의 "확인하였습니다", "수정 적용에 시간이 소요됩니다" 등의 표현을 활용하세요

지금 즉시 참고답변에 100% 충실하면서 질문 내용을 절대 바꾸지 않고 답변하세요."""

        # ===== 프롬프트 반환 =====
        return system_prompt, user_prompt

    # 향상된 GPT 생성 - 일관성과 품질 보장
    # Args:
    #     query: 사용자 질문
    #     similar_answers: 유사한 참고답변 리스트
    #     context_analysis: 컨텍스트 분석 결과
    #     lang: 언어 코드
    # Returns:
    #     str: 생성된 답변 텍스트
    def generate_with_enhanced_gpt(self, query: str, similar_answers: list, context_analysis: dict, lang: str = 'ko') -> str:
        try:
            # 메모리 최적화 컨텍스트 시작
            with memory_cleanup():
                # 1단계: 컨텍스트 분석 및 생성
                approach = context_analysis['recommended_approach']
                context = self.create_enhanced_context(similar_answers, target_lang=lang)
                
                # ===== 🔍 참고답변 컨텍스트 디버그 출력 =====
                print("="*80)
                print("🔍 [DEBUG] GPT에 전달되는 참고답변 컨텍스트:")
                print("="*80)
                print(context)
                print("="*80)
                
                # 디버그 파일에도 저장 (EC2에서 쉽게 확인 가능)
                try:
                    with open('/home/ec2-user/python/debug_context.txt', 'w', encoding='utf-8') as f:
                        f.write("GPT에 전달되는 참고답변 컨텍스트:\n")
                        f.write("="*80 + "\n")
                        f.write(f"질문: {query}\n")
                        f.write("="*80 + "\n")
                        f.write(context)
                        f.write("\n" + "="*80 + "\n")
                    print("🔍 [DEBUG] 컨텍스트가 /home/ec2-user/python/debug_context.txt 파일에 저장되었습니다.")
                except Exception as e:
                    print(f"🔍 [DEBUG] 파일 저장 실패: {e}")
                
                # 컨텍스트 유효성 검증
                if not context:
                    logging.warning("유효한 컨텍스트가 없어 GPT 생성 중단")
                    return ""
                
                # 2단계: 언어별 프롬프트 생성
                system_prompt, user_prompt = self.get_gpt_prompts(query, context, lang)
                
                # ===== 🔍 전체 프롬프트 디버그 출력 =====
                print("\n" + "="*80)
                print("🔍 [DEBUG] GPT에 전달되는 전체 프롬프트:")
                print("="*80)
                print("📋 [SYSTEM PROMPT]:")
                print(system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt)
                print("\n📝 [USER PROMPT]:")
                print(user_prompt)
                print("="*80)
                
                # 프롬프트도 파일에 추가 저장
                try:
                    with open('/home/ec2-user/python/debug_context.txt', 'a', encoding='utf-8') as f:
                        f.write("\n\n전체 프롬프트 정보:\n")
                        f.write("="*80 + "\n")
                        f.write("SYSTEM PROMPT:\n")
                        f.write(system_prompt + "\n\n")
                        f.write("USER PROMPT:\n")
                        f.write(user_prompt + "\n")
                        f.write("="*80 + "\n")
                except Exception as e:
                    print(f"🔍 [DEBUG] 프롬프트 파일 저장 실패: {e}")
                
                # 3단계: 접근 방식에 따른 GPT 파라미터 설정
                if approach == 'gpt_with_strong_context':
                    # 강한 컨텍스트: 낮은 temperature로 일관성 확보
                    temperature = 0.3 if context_analysis.get('context_relevance') == 'high' else 0.4
                    max_tokens = 700
                elif approach == 'gpt_with_weak_context':
                    # 약한 컨텍스트: 적당한 창의성 허용
                    temperature = 0.4
                    max_tokens = 650
                else: # fallback이나 기타 - 생성 중단
                    return ""
                
                # 4단계: 답변 품질 보장을 위한 3회 재시도 메커니즘
                max_attempts = 3
                for attempt in range(max_attempts):
                    # GPT API 호출 (핵심 생성 로직)
                    response = self.openai_client.chat.completions.create(
                        model=self.gpt_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=0.9,                    # 다양성 제어
                        frequency_penalty=0.1,        # 반복 방지
                        presence_penalty=0.1          # 주제 일관성
                    )
                    
                    # 5단계: 응답 추출 및 정리
                    original_response = response.choices[0].message.content.strip()
                    generated = original_response
                    del response  # 메모리 해제
                    
                    # ===== 🔍 GPT 응답 디버그 출력 =====
                    print("\n" + "="*80)
                    print("🤖 [DEBUG] GPT 원본 응답:")
                    print("="*80)
                    print(original_response)
                    print("="*80)
                    
                    # 텍스트 후처리 (불필요한 문구 제거 등)
                    generated = self.text_processor.clean_generated_text(generated)
                    
                    # ===== 🔍 후처리된 응답 디버그 출력 =====
                    print("\n" + "="*80)
                    print("✨ [DEBUG] 후처리된 최종 응답:")
                    print("="*80)
                    print(generated)
                    print("="*80)
                    
                    # GPT 응답도 파일에 저장
                    try:
                        with open('/home/ec2-user/python/debug_context.txt', 'a', encoding='utf-8') as f:
                            f.write(f"\n\nGPT 원본 응답 (시도 #{attempt+1}):\n")
                            f.write("="*80 + "\n")
                            f.write(original_response)
                            f.write(f"\n\n후처리된 최종 응답:\n")
                            f.write("="*80 + "\n")
                            f.write(generated)
                            f.write("\n" + "="*80 + "\n")
                    except Exception as e:
                        print(f"🔍 [DEBUG] GPT 응답 파일 저장 실패: {e}")
                    
                    # 6단계: 품질 검증 (최소 길이 체크)
                    if len(generated.strip()) >= 20:
                        logging.info(f"GPT 생성 성공 (시도 #{attempt+1}, {approach}): {len(generated)}자")
                        return generated
                    
                    # 7단계: 재시도를 위한 파라미터 조정
                    if attempt < max_attempts - 1:
                        temperature = min(temperature + 0.1, 0.6)  # 창의성 증가
                
                # 모든 시도 실패시
                logging.warning("모든 GPT 생성 시도 실패")
                return ""
                
        except Exception as e:
            logging.error(f"향상된 GPT 생성 실패: {e}")
            return ""

    # GPT용 향상된 컨텍스트 생성 - 품질별 우선순위 적용
    # Args:
    #     similar_answers: 유사한 답변 리스트
    #     max_answers: 최대 포함할 답변 수
    #     target_lang: 대상 언어
    # Returns:
    #     str: 구성된 컨텍스트 문자열
    def create_enhanced_context(self, similar_answers: list, max_answers: int = 7, target_lang: str = 'ko') -> str:
        # ===== 🔍 컨텍스트 생성 디버그 출력 =====
        print("\n" + "="*80)
        print(f"🔍 [CONTEXT DEBUG] 컨텍스트 생성 시작: {len(similar_answers) if similar_answers else 0}개 유사답변")
        print("="*80)
        
        if not similar_answers:
            print("🔍 [CONTEXT DEBUG] 유사답변이 없어서 빈 컨텍스트 반환")
            return ""
        
        # ===== 1단계: 초기화 및 품질별 답변 분류 =====
        context_parts = []
        used_answers = 0
        
        # 유사도 점수에 따른 답변 그룹핑 (품질별 분류)
        high_score = [ans for ans in similar_answers if ans['score'] >= 0.7]      # 고품질
        medium_score = [ans for ans in similar_answers if 0.5 <= ans['score'] < 0.7]  # 중품질
        medium_low_score = [ans for ans in similar_answers if 0.4 <= ans['score'] < 0.5]  # 중하품질 (0.4-0.5로 조정)
        low_score = [ans for ans in similar_answers if 0.3 <= ans['score'] < 0.4]  # 저품질 (새로 추가)
        
        # ===== 🔍 품질별 분류 결과 출력 =====
        print(f"🔍 [CONTEXT DEBUG] 품질별 분류: 고품질({len(high_score)}개), 중품질({len(medium_score)}개), 중하품질({len(medium_low_score)}개), 저품질({len(low_score)}개)")
        
        # 유사답변 상세 정보 출력
        for i, ans in enumerate(similar_answers[:5]):  # 상위 5개만
            print(f"유사답변 #{i+1}: 점수={ans['score']:.3f}, 질문={ans.get('question', 'N/A')[:60]}...")
        print("="*40)

        # ===== 2단계: 고품질 답변 우선 포함 (최대 4개) =====
        for ans in high_score[:4]:
            if used_answers >= max_answers:
                break
            
            # 텍스트 전처리 및 인사말/끝맺음말 제거
            clean_answer = self.text_processor.preprocess_text(ans['answer'])
            clean_answer = self.remove_greeting_and_closing(clean_answer, 'ko')
            
            # 다국어 지원: 영어 질문인 경우 답변을 번역
            if target_lang == 'en' and ans.get('lang', 'ko') == 'ko':
                clean_answer = self.translate_text(clean_answer, 'ko', 'en')
            
            # 품질 검증 및 컨텍스트 추가
            if len(clean_answer.strip()) > 20:
                print(f"✅ [CONTEXT DEBUG] 고품질 답변 #{used_answers+1} 추가: 점수={ans['score']:.3f}")
                context_parts.append(f"[참고답변 {used_answers+1} - 점수: {ans['score']:.2f}]\n{clean_answer[:400]}")
                used_answers += 1
            else:
                print(f"❌ [CONTEXT DEBUG] 고품질 답변 제외: 정제 후 길이={len(clean_answer.strip())}")
        
        # ===== 3단계: 중품질 답변으로 보완 (최대 3개) =====
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

        # ===== 4단계: 중하품질 답변 추가 (최대 3개) =====
        for ans in medium_low_score[:3]:
            if used_answers >= max_answers:
                break
            
            clean_answer = self.text_processor.preprocess_text(ans['answer'])
            clean_answer = self.remove_greeting_and_closing(clean_answer, 'ko')
            
            if target_lang == 'en' and ans.get('lang', 'ko') == 'ko':
                clean_answer = self.translate_text(clean_answer, 'ko', 'en')
            
            if len(clean_answer.strip()) > 20:
                print(f"✅ [CONTEXT DEBUG] 중하품질 답변 #{used_answers+1} 추가: 점수={ans['score']:.3f}")
                context_parts.append(f"[참고답변 {used_answers+1} - 점수: {ans['score']:.2f}]\n{clean_answer[:250]}")
                used_answers += 1
            else:
                print(f"❌ [CONTEXT DEBUG] 중하품질 답변 제외: 정제 후 길이={len(clean_answer.strip())}")

        # ===== 5단계: 답변 부족시 저품질 답변도 추가 =====
        if used_answers < 2:  # 최소 2개는 확보하도록
            for ans in low_score[:2]:
                if used_answers >= max_answers:
                    break
                
                clean_answer = self.text_processor.preprocess_text(ans['answer'])
                clean_answer = self.remove_greeting_and_closing(clean_answer, 'ko')
                
                if target_lang == 'en' and ans.get('lang', 'ko') == 'ko':
                    clean_answer = self.translate_text(clean_answer, 'ko', 'en')
                
                if len(clean_answer.strip()) > 20:
                    print(f"✅ [CONTEXT DEBUG] 저품질 답변 #{used_answers+1} 추가: 점수={ans['score']:.3f}")
                    context_parts.append(f"[참고답변 {used_answers+1} - 점수: {ans['score']:.2f}]\n{clean_answer[:200]}")
                    used_answers += 1
                else:
                    print(f"❌ [CONTEXT DEBUG] 저품질 답변 제외: 정제 후 길이={len(clean_answer.strip())}")
        
        # ===== 6단계: 최종 컨텍스트 구성 및 반환 =====
        print(f"🔍 [CONTEXT DEBUG] 최종 컨텍스트: {used_answers}개 답변 포함")
        logging.info(f"컨텍스트 생성: {used_answers}개의 답변 포함 (언어: {target_lang})")
        
        if used_answers == 0:
            print("❌ [CONTEXT DEBUG] 컨텍스트에 포함된 답변이 없음!")
            return ""
        
        final_context = "\n\n" + "="*50 + "\n\n".join(context_parts)
        print(f"🔍 [CONTEXT DEBUG] 생성된 컨텍스트 길이: {len(final_context)}자")
        
        return final_context

    # 참고 답변에서 인사말과 끝맺음말을 제거하는 메서드
    # Args:
    #     text: 처리할 텍스트
    #     lang: 언어 코드 ('ko' 또는 'en')
    # Returns:
    #     str: 인사말/끝맺음말이 제거된 텍스트
    def remove_greeting_and_closing(self, text: str, lang: str = 'ko') -> str:
        if not text:
            return ""
        
        # ===== 언어별 패턴 정의 =====
        if lang == 'ko':
            # 한국어 인사말 패턴
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
            
            # 한국어 끝맺음말 패턴
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
        else:  # 영어 패턴
            # 영어 인사말 패턴
            greeting_patterns = [
                r'^Hello[^.]*\.\s*',
                r'^Hi[^.]*\.\s*',
                r'^Dear[^.]*\.\s*',
                r'^Thank you[^.]*\.\s*',
                r'^Thanks[^.]*\.\s*',
                r'^This is GOODTV Bible App[^.]*\.\s*',
            ]
            
            # 영어 끝맺음말 패턴
            closing_patterns = [
                r'\s*Thank you[^.]*\.?\s*$',
                r'\s*Thanks[^.]*\.?\s*$',
                r'\s*Best regards[^.]*\.?\s*$',
                r'\s*Sincerely[^.]*\.?\s*$',
                r'\s*God bless[^.]*\.?\s*$',
                r'\s*May God[^.]*\.?\s*$',
            ]
        
        # ===== 패턴 적용하여 텍스트 정리 =====
        # 1단계: 인사말 제거
        for pattern in greeting_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # 2단계: 끝맺음말 제거
        for pattern in closing_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # 3단계: 공백 정리 및 반환
        text = text.strip()
        return text

    # GPT를 사용한 다국어 번역 - 원문 톤앤매너 유지
    # Args:
    #     text: 번역할 텍스트
    #     source_lang: 원본 언어 코드
    #     target_lang: 목적 언어 코드
    # Returns:
    #     str: 번역된 텍스트 (실패시 원문 반환)
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        try:
            # ===== 1단계: 언어 매핑 =====
            lang_map = {
                'ko': 'Korean',
                'en': 'English'
            }
            
            # ===== 2단계: 번역 프롬프트 생성 =====
            system_prompt = f"You are a professional translator. Translate the following text from {lang_map[source_lang]} to {lang_map[target_lang]}. Keep the same tone and style. Only provide the translation without any explanation."
            
            # ===== 3단계: GPT API 호출로 번역 실행 =====
            response = self.openai_client.chat.completions.create(
                model='gpt-4o',
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                max_tokens=600,              # 충분한 번역 길이 허용
                temperature=0.5              # 일관성과 창의성 균형
            )
            
            # ===== 4단계: 번역 결과 반환 =====
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            # 번역 실패시 원문 반환
            logging.error(f"번역 실패: {e}")
            return text
