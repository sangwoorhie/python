#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
품질 검증 서비스 모듈
"""

import re
import logging
from typing import Dict, List
from src.utils.memory_manager import memory_cleanup
from src.utils.text_preprocessor import TextPreprocessor


class QualityValidator:
    """답변 품질 검증을 담당하는 클래스"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.text_processor = TextPreprocessor()
    
    def is_valid_text(self, text: str, lang: str = 'ko') -> bool:
        """텍스트 유효성 검증 메서드"""
        if not text or len(text.strip()) < 3:
            return False
        
        if lang == 'ko':
            return self.is_valid_korean_text(text)
        else:  # 영어
            return self.is_valid_english_text(text)

    def is_valid_korean_text(self, text: str) -> bool:
        """한국어 텍스트의 유효성을 검증하는 메서드"""
        if not text or len(text.strip()) < 3:
            logging.info(f"한국어 검증 실패: 텍스트가 너무 짧음 (길이: {len(text.strip()) if text else 0})")
            return False
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'\s', '', text))
        
        if total_chars == 0:
            logging.info("한국어 검증 실패: 총 글자 수가 0")
            return False
            
        korean_ratio = korean_chars / total_chars
        logging.info(f"한국어 비율: {korean_ratio:.3f} (한국어: {korean_chars}, 전체: {total_chars})")
        
        # 한국어 비율 기준을 완화 (0.2 → 0.1)
        if korean_ratio < 0.1:
            logging.info(f"한국어 검증 실패: 한국어 비율 부족 ({korean_ratio:.3f} < 0.1)")
            return False
        
        # 무의미한 패턴 감지 (GPT 할루시네이션 방지)
        meaningless_patterns = [
            r'^[a-z\s\.,;:\(\)\[\]\-_&\/\'"]+$', # 영어
            r'^[A-Z\s\.,;:\(\)\[\]\-_&\/\'"]+$', # 영어 대문자
            r'^[\s\.,;:\(\)\[\]\-_&\/\'"]+$',    # 공백
            r'^[0-9\s\.,;:\(\)\[\]\-_&\/\'"]+$', # 숫자
            r'.*[а-я].*',                        # 러시아어
            r'.*[α-ω].*',                        # 그리스어
        ]
        
        for pattern in meaningless_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                logging.info(f"한국어 검증 실패: 무의미한 패턴 감지")
                return False
        
        # 반복 문자 감지: 같은 문자가 5번 이상 연속으로 나타나면 비정상 텍스트로 간주
        if re.search(r'(.)\1{5,}', text):
            logging.info("한국어 검증 실패: 반복 문자 감지")
            return False
        
        # 영어 비율 검사를 완화 (0.5 → 0.7)
        random_pattern = r'[a-zA-Z]{8,}'
        if re.search(random_pattern, text) and korean_ratio < 0.3:
            logging.info(f"한국어 검증 실패: 긴 영어 단어와 낮은 한국어 비율")
            return False
        
        logging.info("한국어 검증 성공")
        return True

    def is_valid_english_text(self, text: str) -> bool:
        """영어 텍스트의 유효성을 검증하는 메서드"""
        if not text or len(text.strip()) < 3:
            return False
        
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(re.sub(r'\s', '', text))
        
        if total_chars == 0:
            return False
            
        english_ratio = english_chars / total_chars
        
        if english_ratio < 0.7:  # 영어 비율이 70% 미만이면 무효
            return False
        
        # 반복 문자 감지
        if re.search(r'(.)\1{5,}', text):
            return False
        
        return True

    def check_answer_completeness(self, answer: str, query: str, lang: str = 'ko') -> float:
        """생성된 답변의 완성도와 유용성을 평가"""
        
        try:
            # 1. 기본 길이 검사
            if len(answer.strip()) < 10:
                return 0.0
                
            # 2. 실질적 내용 비율 검사
            meaningful_content_ratio = self.calculate_meaningful_content_ratio(answer, lang)
            
            # 3. 질문-답변 관련성 키워드 매칭
            query_keywords = set(self.text_processor.extract_keywords(query.lower()))
            answer_keywords = set(self.text_processor.extract_keywords(answer.lower()))
            keyword_overlap = len(query_keywords & answer_keywords)
            keyword_relevance = keyword_overlap / max(len(query_keywords), 1) if query_keywords else 0.5
            
            # 4. 답변 완결성 검사 (문장이 완성되어 있는지)
            completeness_score = self.check_sentence_completeness(answer, lang)
            
            # 5. 구체성 검사 (구체적인 정보가 포함되어 있는지)
            specificity_score = self.check_answer_specificity(answer, query, lang)
            
            # 6. 종합 점수 계산 (가중 평균)
            final_score = (
                meaningful_content_ratio * 0.3 +    # 의미있는 내용 비율
                keyword_relevance * 0.25 +          # 키워드 관련성
                completeness_score * 0.25 +         # 문장 완결성
                specificity_score * 0.2             # 구체성
            )
            
            logging.info(f"답변 완성도 분석: 내용비율={meaningful_content_ratio:.2f}, "
                        f"키워드관련성={keyword_relevance:.2f}, 완결성={completeness_score:.2f}, "
                        f"구체성={specificity_score:.2f}, 최종점수={final_score:.2f}")
            
            return min(final_score, 1.0)
            
        except Exception as e:
            logging.error(f"답변 완성도 검증 실패: {e}")
            return 0.5  # 오류시 중간값 반환

    def calculate_meaningful_content_ratio(self, text: str, lang: str = 'ko') -> float:
        """텍스트에서 의미있는 내용의 비율을 계산"""
        
        if not text:
            return 0.0
            
        # HTML 태그 제거
        clean_text = re.sub(r'<[^>]+>', '', text)
        
        if lang == 'ko':
            # 한국어 불용구 제거
            filler_patterns = [
                r'안녕하세요[^.]*\.',
                r'감사[드립]*니다[^.]*\.',
                r'평안하세요[^.]*\.',
                r'주님\s*안에서[^.]*\.',
                r'바이블\s*애플[^.]*\.',
                r'GOODTV[^.]*\.',
                r'문의[해주셔서]*\s*감사[^.]*\.',
                r'안내[해]*드리겠습니다[^.]*\.',
                r'도움이\s*[되]*[시]*[길]*[바라]*[며]*[^.]*\.',
                r'항상[^.]*바이블\s*애플[^.]*\.'
            ]
        else:
            # 영어 불용구 제거
            filler_patterns = [
                r'Hello[^.]*\.',
                r'Thank you[^.]*\.',
                r'Best regards[^.]*\.',
                r'God bless[^.]*\.',
                r'Bible App[^.]*\.',
                r'GOODTV[^.]*\.',
                r'We will[^.]*\.',
                r'Please contact[^.]*\.'
            ]
        
        # 불용구 제거
        for pattern in filler_patterns:
            clean_text = re.sub(pattern, '', clean_text, flags=re.IGNORECASE)
        
        # 공백 정리
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        # 의미있는 내용 비율 계산
        original_length = len(re.sub(r'<[^>]+>', '', text).strip())
        meaningful_length = len(clean_text)
        
        if original_length == 0:
            return 0.0
            
        ratio = meaningful_length / original_length
        return min(ratio, 1.0)

    def check_sentence_completeness(self, text: str, lang: str = 'ko') -> float:
        """문장이 완성되어 있는지 검사"""
        
        if not text:
            return 0.0
            
        # HTML 태그 제거
        clean_text = re.sub(r'<[^>]+>', '', text).strip()
        
        if len(clean_text) < 5:
            return 0.0
        
        # 문장 끝 표시 확인
        if lang == 'ko':
            sentence_endings = r'[.!?니다요음됩다음까다하세요습니다니까]'
        else:
            sentence_endings = r'[.!?]'
        
        # 마지막 문장이 완성되어 있는지 확인
        if re.search(sentence_endings + r'\s*$', clean_text):
            return 1.0
        
        # 중간에 완성된 문장이 있는지 확인
        sentences = re.split(sentence_endings, clean_text)
        if len(sentences) > 1:
            return 0.7  # 부분적으로 완성됨
        
        # 문장이 불완전한 경우
        return 0.3

    def check_answer_specificity(self, answer: str, query: str, lang: str = 'ko') -> float:
        """답변이 구체적인 정보를 포함하는지 검사 (빈 약속 패턴 엄격 감지)"""
        
        if not answer:
            return 0.0
        
        # 빈 약속 패턴 엄격 감지
        empty_promise_score = self.detect_empty_promises(answer, lang)
        if empty_promise_score < 0.3:  # 빈 약속이 감지되면 매우 낮은 점수
            logging.warning(f"빈 약속 패턴 감지! 점수: {empty_promise_score:.2f}")
            return empty_promise_score
            
        specificity_score = 0.0
        
        if lang == 'ko':
            # 구체적 정보 패턴 (한국어) - 더 엄격하게 강화
            specific_patterns = [
                r'\d+[가지개단계번째차례]',  # 숫자가 포함된 단계
                r'[메뉴설정화면버튼탭]에서',    # 구체적 위치
                r'다음과\s*같[은이]',         # 구체적 방법 제시
                r'[클릭선택터치누르]',         # 구체적 동작
                r'[방법단계절차과정]',         # 구체적 프로세스
                r'\w+\s*버튼',               # 버튼명
                r'\w+\s*메뉴',               # 메뉴명
                r'NIV|KJV|ESV|번역본',       # 구체적 번역본
                r'[상하좌우]단[에의]',         # 구체적 위치
                r'설정[에서으로]',            # 설정 관련
                r'화면\s*[상하좌우중앙]',      # 화면 위치
                r'탭하여|클릭하여|터치하여',    # 구체적 행동
                r'다음\s*순서',              # 순서 안내
                r'먼저|그다음|마지막으로'       # 단계별 안내
            ]
            
            # 빈 약속/모호한 표현 패턴 (더 엄격하게)
            vague_patterns = [
                r'안내[해]*드리겠습니다',
                r'도움[을이]\s*드리겠습니다',
                r'확인[하고하여해서]',
                r'검토[하고하여]',
                r'준비[하고하겠습니다]',
                r'전달[하고하겠드리겠]',
                r'제공[하고하겠드리겠]',
                r'노력[하고하겠]',
                r'살펴[보고보겠]',
                r'방법[을이]\s*찾아[드리겠보겠]'
            ]
        else:
            # 구체적 정보 패턴 (영어)
            specific_patterns = [
                r'\d+\s*steps?',
                r'follow\s+these',
                r'click\s+on',
                r'go\s+to',
                r'select\s+\w+',
                r'settings?\s+menu',
                r'NIV|KJV|ESV|translation',
                r'top\s+of\s+screen',
                r'button\s+\w+'
            ]
            
            vague_patterns = [
                r'we\s+will\s+review',
                r'we\s+are\s+working',
                r'please\s+contact',
                r'will\s+be\s+available'
            ]
        
        # 구체성 점수 계산
        specific_count = 0
        for pattern in specific_patterns:
            specific_count += len(re.findall(pattern, answer, re.IGNORECASE))
        
        vague_count = 0
        for pattern in vague_patterns:
            vague_count += len(re.findall(pattern, answer, re.IGNORECASE))
        
        # 구체적 정보가 많고 모호한 표현이 적을수록 높은 점수
        if specific_count > 0:
            specificity_score = specific_count / (specific_count + vague_count + 1)
        else:
            specificity_score = 0.1 if vague_count == 0 else 0.0
        
        return min(specificity_score, 1.0)

    def detect_empty_promises(self, answer: str, lang: str = 'ko') -> float:
        """약속만 하고 실제 내용이 없는 빈 약속 패턴을 감지"""
        
        if not answer:
            return 0.0
        
        # HTML 태그 제거하여 순수 텍스트로 분석
        clean_text = re.sub(r'<[^>]+>', '', answer)
        
        if lang == 'ko':
            # 위험한 약속 표현들 (이후 실제 내용이 와야 함)
            promise_patterns = [
                r'안내[해]*드리겠습니다',
                r'도움[을이]?\s*드리겠습니다',
                r'방법[을이]?\s*안내[해]*드리겠습니다',
                r'설명[해]*드리겠습니다',
                r'알려[드리겠드릴]',
                r'제공[해]*드리겠습니다',
                r'도와[드리겠드릴]',
                r'찾아[드리겠드릴]'
            ]
            
            # 실제 내용을 나타내는 패턴들
            content_patterns = [
                r'\d+\.\s*',                    # 번호 매기기 (1., 2., ...)
                r'먼저',                       # 단계별 설명 시작
                r'다음과?\s*같[은이]',           # 구체적 방법 제시
                r'[메뉴설정화면버튼]',           # 구체적 UI 요소
                r'클릭|터치|선택|이동',          # 구체적 행동
                r'NIV|KJV|ESV',               # 구체적 번역본
                r'상단|하단|좌측|우측',         # 구체적 위치
                r'설정에서|메뉴에서',           # 구체적 경로
                r'다음\s*[순서단계방법절차]',    # 단계별 안내
                r'[0-9]+[번째단계]',           # 순서 표시
                r'화면\s*[상하좌우중앙]'        # 위치 설명
            ]
        else:  # 영어
            promise_patterns = [
                r'will\s+guide\s+you',
                r'will\s+help\s+you',
                r'will\s+show\s+you',
                r'will\s+provide',
                r'let\s+me\s+help',
                r'here[\'\"]s\s+how'
            ]
            
            content_patterns = [
                r'\d+\.\s*',
                r'first|second|third',
                r'step\s+\d+',
                r'click|tap|select',
                r'menu|setting|screen',
                r'NIV|KJV|ESV',
                r'top|bottom|left|right'
            ]
        
        # 약속 표현 찾기
        promise_count = 0
        promise_positions = []
        
        for pattern in promise_patterns:
            matches = list(re.finditer(pattern, clean_text, re.IGNORECASE))
            promise_count += len(matches)
            promise_positions.extend([match.start() for match in matches])
        
        if promise_count == 0:
            return 0.8  # 약속 표현이 없으면 중간 점수
        
        # 약속 이후에 실제 내용이 있는지 확인
        content_after_promise = 0
        total_text_after_promises = 0
        
        for pos in promise_positions:
            # 약속 표현 이후의 텍스트 추출
            text_after = clean_text[pos:]
            
            # 끝맺음말 제거하여 실제 내용만 검사
            text_after = re.sub(r'항상\s*성도님께[^.]*\.', '', text_after, flags=re.IGNORECASE)
            text_after = re.sub(r'감사합니다[^.]*\.', '', text_after, flags=re.IGNORECASE)
            text_after = re.sub(r'주님\s*안에서[^.]*\.', '', text_after, flags=re.IGNORECASE)
            text_after = re.sub(r'평안하세요[^.]*\.', '', text_after, flags=re.IGNORECASE)
            
            total_text_after_promises += len(text_after.strip())
            
            # 실제 내용 패턴이 있는지 확인
            for content_pattern in content_patterns:
                if re.search(content_pattern, text_after, re.IGNORECASE):
                    content_after_promise += 1
                    break
        
        # 점수 계산
        if promise_count > 0:
            # 약속 대비 실제 내용 비율
            content_ratio = content_after_promise / promise_count
            
            # 약속 이후 텍스트 길이 비율 (평균)
            avg_length_after = total_text_after_promises / len(promise_positions) if promise_positions else 0
            length_score = min(avg_length_after / 100, 1.0)  # 100자 기준으로 정규화
            
            # 최종 점수 (내용 비율과 길이를 고려)
            final_score = content_ratio * 0.7 + length_score * 0.3
            
            logging.info(f"빈 약속 분석: 약속={promise_count}개, 실제내용={content_after_promise}개, "
                        f"내용비율={content_ratio:.2f}, 길이점수={length_score:.2f}, 최종점수={final_score:.2f}")
            
            return final_score
        
        return 0.5  # 기본값

    def detect_hallucination_and_inconsistency(self, answer: str, query: str, lang: str = 'ko') -> dict:
        """생성된 답변에서 할루시네이션과 일관성 문제를 감지"""
        
        issues = {
            'external_app_recommendation': False,
            'bible_app_domain_violation': False,
            'content_inconsistency': False,
            'translation_switching': False,
            'invalid_features': False,
            'overall_score': 1.0,
            'detected_issues': []
        }
        
        if not answer:
            return issues
        
        # HTML 태그 제거하여 순수 텍스트로 분석
        clean_answer = re.sub(r'<[^>]+>', '', answer)
        clean_query = re.sub(r'<[^>]+>', '', query)
        
        if lang == 'ko':
            # 1. 외부 앱 추천 감지 (치명적)
            external_app_patterns = [
                r'Parallel\s*Bible',
                r'병렬\s*성경\s*앱',
                r'다른\s*앱을?\s*(다운로드|설치)',
                r'앱\s*스토어에서\s*(검색|다운로드)',
                r'구글\s*플레이\s*스토어',
                r'외부\s*(앱|어플리케이션)',
                r'별도[의]*\s*(앱|어플)',
                r'추가로\s*(앱을|어플을)\s*설치'
            ]
            
            for pattern in external_app_patterns:
                if re.search(pattern, clean_answer, re.IGNORECASE):
                    issues['external_app_recommendation'] = True
                    issues['detected_issues'].append(f"외부 앱 추천 감지: {pattern}")
                    issues['overall_score'] -= 0.8  # 매우 심각한 감점
            
            # 2. 번역본 변경/교체 감지 (질문 vs 답변)
            query_translations = self.text_processor.extract_translations_from_text(clean_query)
            answer_translations = self.text_processor.extract_translations_from_text(clean_answer)
            
            if query_translations and answer_translations:
                # 질문에서 언급한 번역본이 답변에서 다른 번역본으로 바뀌었는지 확인
                query_set = set(query_translations)
                answer_set = set(answer_translations)
                
                # 질문에 없던 번역본이 답변에 추가되었는지 확인
                unexpected_translations = answer_set - query_set
                if unexpected_translations:
                    # 완전히 다른 번역본(예: 개역한글 → 영문성경)은 금지
                    problematic = False
                    for trans in unexpected_translations:
                        if any(forbidden in trans.lower() for forbidden in ['영어', 'english', 'niv', 'kjv', 'esv']) and \
                           not any(allowed in q_trans.lower() for q_trans in query_translations for allowed in ['영어', 'english', 'niv', 'kjv', 'esv']):
                            problematic = True
                            break
                        elif any(forbidden in trans.lower() for forbidden in ['한글', '개역', 'korean']) and \
                             not any(allowed in q_trans.lower() for q_trans in query_translations for allowed in ['한글', '개역', 'korean']):
                            problematic = True
                            break
                    
                    if problematic:
                        issues['translation_switching'] = True
                        issues['detected_issues'].append(f"번역본 변경: {query_translations} → {list(unexpected_translations)}")
                        issues['overall_score'] -= 0.7
        
        # 최종 점수 정규화
        issues['overall_score'] = max(issues['overall_score'], 0.0)
        
        # 심각한 문제가 하나라도 있으면 전체 점수를 매우 낮게
        critical_issues = [
            issues['external_app_recommendation'],
            issues['translation_switching'],
        ]
        
        if any(critical_issues):
            issues['overall_score'] = min(issues['overall_score'], 0.2)
        
        logging.info(f"할루시네이션 검증 결과: 점수={issues['overall_score']:.2f}, 문제={len(issues['detected_issues'])}개")
        
        return issues

    def validate_answer_relevance_ai(self, answer: str, query: str, question_analysis: dict) -> bool:
        """AI를 이용해 생성된 답변이 질문과 관련성이 있는지 엄격하게 검증"""
        
        try:
            with memory_cleanup():
                system_prompt = """당신은 답변 품질 검증 전문가입니다.
생성된 답변이 고객의 질문에 적절히 대응하는지 엄격하게 평가하세요.

⚠️ 엄격한 평가 기준:
1. 답변이 질문의 핵심 행동 요청과 일치하는가? (복사≠재생)
2. 답변이 질문의 주제 영역과 일치하는가? (텍스트≠음성)
3. 답변이 실제 문제 해결에 직접적으로 도움이 되는가?
4. 답변에서 언급하는 기능이 질문에서 요청한 기능과 같은가?

🚫 부적절한 답변 예시:
- 텍스트 복사 질문에 음성 재생 답변
- 검색 기능 질문에 설정 변경 답변  
- 오류 신고에 일반 사용법 답변
- 구체적 기능 질문에 추상적 안내 답변

결과: "relevant" 또는 "irrelevant" 중 하나만 반환하세요."""

                user_prompt = f"""질문 분석:
의도: {question_analysis.get('intent_type', 'N/A')}
주제: {question_analysis.get('main_topic', 'N/A')}
행동유형: {question_analysis.get('action_type', 'N/A')}
요청사항: {question_analysis.get('specific_request', 'N/A')}

원본 질문: {query}

생성된 답변: {answer}

⚠️ 특히 주의: 질문의 행동유형과 답변에서 다루는 행동이 다르면 "irrelevant"입니다.
이 답변이 질문에 적절한지 엄격하게 평가해주세요."""

                response = self.openai_client.chat.completions.create(
                    model='gpt-3.5-turbo',
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=30,
                    temperature=0.1
                )
                
                result = response.choices[0].message.content.strip().lower()
                
                is_relevant = 'relevant' in result and 'irrelevant' not in result
                
                logging.info(f"AI 답변 관련성 검증: {result} -> {is_relevant}")
                
                return is_relevant
                
        except Exception as e:
            logging.error(f"AI 답변 관련성 검증 실패: {e}")
            # 폴백: 기본적인 키워드 매칭
            query_keywords = set(self.text_processor.extract_keywords(query.lower()))
            answer_keywords = set(self.text_processor.extract_keywords(answer.lower()))
            
            keyword_overlap = len(query_keywords & answer_keywords)
            keyword_relevance = keyword_overlap / max(len(query_keywords), 1)
            
            return keyword_relevance >= 0.2  # 20% 이상 키워드 일치시 관련성 있음으로 판단
