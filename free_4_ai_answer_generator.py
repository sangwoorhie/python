#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI Answer Generator Flask API for ASP Classic Integration
파일명: free_4_ai_answer_generator.py
설명: Flask API로 ASP Classic에서 호출
모델: google/flan-t5-base
개선사항: 한글 자모 분리 해결, 스마트 따옴표 정리, 성경 구절 정규화, 문단 나누기 개선
"""

# 필수 라이브러리 임포트
import os # 파일 경로 처리 및 환경변수 접근
import json # JSON 데이터 처리 및 직렬화
import json as json_module # 표준 json 모듈 별칭으로 임포트 (이스케이프 처리용)
import re # 정규식을 이용한 텍스트 패턴 매칭 및 치환
import html # HTML 엔티티 디코딩 (&amp; → &)
import unicodedata # 유니코드 문자 처리
import logging # 애플리케이션 로그 기록 및 디버깅
from flask import Flask, request, jsonify # 웹 서버 프레임워크 및 HTTP 요청/응답 처리
from pinecone import Pinecone # 벡터 데이터베이스 연결 및 유사도 검색
from sentence_transformers import SentenceTransformer # 텍스트를 벡터로 변환하는 임베딩 모델
from transformers import T5ForConditionalGeneration, T5Tokenizer # Google T5 텍스트 생성 모델
from dotenv import load_dotenv # .env 파일에서 환경변수 로드
from flask_cors import CORS # 크로스 도메인 요청 허용 설정

# Flask 웹 애플리케이션 인스턴스 생성
app = Flask(__name__)
CORS(app) # 브라우저의 Same-Origin Policy를 우회하여 다른 도메인에서 API 호출 허용

# 로깅 시스템 설정 - 파일에 로그 저장
logging.basicConfig(
    filename='/home/ec2-user/python/logs/ai_generator.log', # 로그 파일 경로 (EC2 서버용)
    level=logging.INFO, # INFO 레벨 이상의 로그만 기록
    format='%(asctime)s - %(levelname)s - %(message)s', # 로그 포맷: 시간-레벨-메시지
    encoding='utf-8' # 한글 로그 지원을 위한 UTF-8 인코딩
)

# .env 파일에서 환경변수 로드 (API 키 등 민감한 정보)
load_dotenv()

# AI 모델 및 벡터 데이터베이스 초기화
# AI 모델 및 벡터 데이터베이스 초기화하는 이유는 매 요청마다 모델을 다시 로드하면 30초씩 걸리므로 사용자 경험 저하
try:
    # Pinecone 벡터 데이터베이스 연결 (유사도 검색용)
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY')) # 환경변수에서 API 키를 가져와서 Pinecone 클라이언트 초기화
    index = pc.Index("bible-app-support-768-free") # 성경 앱 지원용 인덱스
    
    # 다국어 임베딩 모델 로드 (768차원 벡터 생성)
    embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    
    # Google T5 텍스트 생성 모델 및 토크나이저 로드
    # T5ForConditionalGeneration: 실제 텍스트를 생성하는 모델
    # T5Tokenizer: 텍스트를 토큰으로 변환하고 다시 텍스트로 변환하는 도구
    text_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')
    text_tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')

except Exception as e:
    # 모델 로드 실패 시 로그 기록 및 예외 발생
    # 이 단계에서 실패하면 애플리케이션 자체가 시작되지 않음
    logging.error(f"모듈 로드 실패: {str(e)}")
    app.logger.error(f"모듈 로드 실패: {str(e)}")
    raise

# AI 기반 답변 생성 클래스
# 텍스트 전처리, 임베딩 생성, 유사도 검색, AI 답변 생성을 담당
# 사용자 질문 → preprocess_text() → create_embedding() → search_similar_answers() → generate_with_t5() → clean_answer_text() → format_answer_with_html_paragraphs() → 최종 HTML 답변
class AIAnswerGenerator:

    ### ★ 1. 사용자 입력 텍스트 전처리 함수 (AI 모델이 처리하기 적합한 형태로 텍스트 정제)
    # 웹에서 온 텍스트는 HTML태그, 제어문자, 특수 공백 등이 섞여있고, 이런 노이즈가 임베딩 품질을 떨어뜨리므로 단계별 정제 필요
    # 1) HTML 엔티티 먼저 디코딩 → 2) 제어문자 제거 → 3) 구조적 태그 변환 → 4) 나머지 태그 제거
    def preprocess_text(self, text: str) -> str:
        if not text: # 빈 텍스트 처리
            return ""
        
        text = str(text) # 문자열 타입 강제 변환
        text = html.unescape(text) # HTML 엔티티 디코딩 (&amp; → &, &lt; → <)
        
        # 1단계: 제어 문자 제거 (백스페이스, 탭, 캐리지 리턴 등)
        text = re.sub(r'[\b\r\f\v]', '', text)  # \b(백스페이스), \r, \f, \v 제거
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)  # 기타 제어 문자 제거
        
        # 2단계: HTML 태그를 적절한 구분자로 변환 (순서가 중요!)
        # 먼저 구조적 태그들을 의미있는 구분자로 변환한 후, 나머지 태그 제거
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE) # <br> → 줄바꿈
        text = re.sub(r'</p>', '\n\n', text, flags=re.IGNORECASE) # </p> → 문단 구분
        text = re.sub(r'<p[^>]*>', '\n', text, flags=re.IGNORECASE) # <p> → 줄바꿈
        text = re.sub(r'<li[^>]*>', '\n• ', text, flags=re.IGNORECASE) # <li> → 리스트
        text = re.sub(r'</li>', '', text, flags=re.IGNORECASE)
        
        # 3단계: 강조 태그 제거 (** 문제 해결)
        # 마크다운과 HTML 강조 태그를 모두 제거
        text = re.sub(r'<(strong|b)[^>]*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</(strong|b)>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        
        # 4단계: 공백 및 연속된 줄바꿈 정리
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)   
        text = text.strip()
        
        return text

    ### JSON 문자열 이스케이프 처리 함수 (ASP Classic에서 JSON 파싱 오류를 방지하기 위한 특수문자 처리)
    # ASP Classic은 특수문자를 포함한 JSON 문자열을 파싱할 때 오류가 발생하므로 이스케이프 처리가 필요
    def escape_json_string(self, text: str) -> str:
        if not text:
            return ""
        
        # Python의 json 모듈을 사용하여 안전하게 이스케이프
        # dumps로 JSON 문자열로 만든 후, 양쪽 따옴표 제거
        escaped = json_module.dumps(text, ensure_ascii=False)
        return escaped[1:-1]  # 앞뒤 따옴표 제거

    ### ★ 2. 텍스트를 768차원 벡터로 변환하는 함수 (의미적 유사도 검색을 위한 벡터 표현 생성)        
        # 📌 SentenceTransformer 작동 원리:
        # 1) 텍스트를 토큰으로 분할 (subword tokenization)
        # 2) (BERT 계열 모델(양방향 트랜스포머(Bidirectional Transformer)를 사용해 문맥을 이해하는 모델))로 각 토큰의 컨텍스트 벡터 생성
        # 3) 평균 풀링으로 문장 전체를 하나의 768차원 벡터로 압축
        # 4) 정규화하여 코사인 유사도 계산에 최적화
        
        # 📌 왜 convert_to_tensor=False?
        # - PyTorch 텐서가 아닌 NumPy 배열로 반환
        # - 이후 .tolist()로 Python 리스트로 변환하여 JSON 직렬화 가능
        
        # - NumPy 배열(ndarray)은 Python의 NumPy 라이브러리에서 제공하는 다차원 배열 객체입니다. 숫자 데이터를 효율적으로 저장하고, 수학적 연산(덧셈, 곱셈, 행렬 연산 등)을 빠르게 수행할 수 있도록 설계
        # - PyTorch 라이브러리에서 제공하는 다차원 배열 객체로, 머신러닝과 딥러닝 작업에 최적화되어 있습니다. NumPy 배열과 비슷하지만, GPU를 활용한 고속 연산과 자동 미분(gradient 계산) 기능을 지원하는 점이 다름.
    def create_embedding(self, text: str) -> list:

        try:
            # SentenceTransformer로 텍스트를 벡터로 변환 (전역에서 로드한 embedding_model을 사용해 768차원 벡터로 변환)
            embedding = embedding_model.encode(text, convert_to_tensor=False)
            return embedding.tolist() # numpy 배열을 리스트로 변환
        except Exception as e:
            logging.error(f"임베딩 생성 실패: {e}")
            return None # 실패 시 None 반환으로 안전장치

    ### ★ 3. 텍스트를 768차원 벡터로 변환하는 함수 (사용자 질문과 의미적으로 유사한 기존 답변들을 찾아 컨텍스트로 활용)   
    # 1) 사용자 질문을 768차원 벡터로 변환
    # 2) Pinecone에 저장된 모든 벡터와 코사인 유사도 계산
    # 3) 유사도가 높은 상위 10개 선택
    # 4) 임계값 0.6 이상인 것만 필터링하여 품질 보장
    def search_similar_answers(self, query: str, top_k: int = 10, similarity_threshold: float = 0.6) -> list:

        try:
            # 1단계: 질문을 벡터로 변환
            query_vector = self.create_embedding(query)
            if query_vector is None: # 임베딩 생성 실패 시
                return []
            
            # 2단계: Pinecone에서 벡터 유사도 검색 수행
            # vector=query_vector: 검색할 기준 벡터
            # top_k=10: 상위 10개 결과만 반환 (성능 최적화)
            # include_metadata=True: 질문/답변 텍스트도 함께 반환
            results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
            
            # 3단계: 유사도 임계값(0.6) 이상인 결과만 필터링
            # 이 과정에서 저품질 매칭을 제거하여 AI 생성 품질 향상
            filtered_results = [
                {
                    'score': match['score'], # 유사도 점수 (0~1)
                    'question': match['metadata'].get('question', ''), # 기존 질문
                    'answer': match['metadata'].get('answer', ''), # 기존 답변
                    'category': match['metadata'].get('category', '일반') # 카테고리
                }
                for match in results['matches'] if match['score'] >= similarity_threshold
            ]
            
            logging.info(f"유사 답변 {len(filtered_results)}개 검색 완료")
            return filtered_results
        except Exception as e:
            logging.error(f"Pinecone 검색 실패: {str(e)}")
            return [] # 검색 실패 시 빈 리스트 반환

    ### 더 이상 사용되지 않는 앱 이름 제거 함수 (다양한 패턴으로 "(구)다번역성경찬송" 제거)
    def remove_old_app_name(self, text: str) -> str:
        # 다양한 패턴으로 제거
        patterns_to_remove = [
            r'\s*\(구\)\s*다번역성경찬송',
            r'\s*\(구\)다번역성경찬송',
            r'바이블\s*애플\s*\(구\)\s*다번역성경찬송',
            r'바이블애플\s*\(구\)다번역성경찬송',
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # GOODTV 바이블 애플 뒤의 불필요한 공백 정리
        text = re.sub(r'(GOODTV\s+바이블\s*애플)\s+', r'\1', text)
        
        return text

    ### ★ HTML 태그를 사용한 문단 나누기 함수 (ASP Classic의 Quill 에디터에서 올바르게 표시되도록 HTML 태그 적용)
    # AI가 생성한 텍스트는 하나의 긴 문장일 수 있으므로, 사용자 가독성을 위해 의미 단위로 문단 분리리
    def format_answer_with_html_paragraphs(self, text: str) -> str:
        if not text:
            return ""
        
        # 먼저 "(구)다번역성경찬송" 제거
        text = self.remove_old_app_name(text)
        
        # 문장을 분리하여 처리
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # 문단을 저장할 리스트
        paragraphs = []
        current_paragraph = []
        
        # 문단 나누기 트리거 키워드
        # 한국어 고객 서비스 문체의 담화 표지어 분석 결과에 기반하여 선정
        paragraph_triggers = [
            # 안내나 설명의 시작
            '해당', '이', '만약', '혹시', '성도님', '고객님',
            # 추가 설명
            '번거로우시', '불편하시', '죄송하지만', '참고로',
            # 마무리 멘트 시작
            '항상', '늘', '앞으로도', '지속적으로',
            # 기능 설명 시작
            '스피커', '버튼', '메뉴', '화면', '설정',
        ]
        
        # 각 문장을 분석하여 문단 구성
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 특별 처리 : 인사말은 항상 별도 문단
            if i == 0 and any(greeting in sentence for greeting in ['안녕하세요', '안녕']):
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                paragraphs.append(sentence)
                continue
            
            # 문단 나누기 조건 확인
            should_break = False
            
            # 조건 1. 트리거 키워드로 시작하는 경우
            for trigger in paragraph_triggers:
                if sentence.startswith(trigger):
                    should_break = True
                    break
            
            # 조건 2. 이전 문장이 설명 완료이고 새로운 주제로 전환
            if current_paragraph and len(current_paragraph) >= 2:
                should_break = True
            
            # 조건 3. 마무리 인사 시작
            if any(closing in sentence for closing in ['감사합니다', '감사드립니다', '평안하세요']):
                should_break = True
            
            # 문단 분리 실행 (should_break가 True이고 current_paragraph가 비어있지 않은 경우)
            if should_break and current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = [sentence]
            else:
                current_paragraph.append(sentence)
        
        # 남은 문장 처리
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        # HTML 형식으로 변환
        html_paragraphs = []
        for i, paragraph in enumerate(paragraphs):
            # 각 문단을 <p> 태그로 감싸기 (Quill 에디터 호환)
            html_paragraphs.append(f"<p>{paragraph}</p>")
            
            # 문단 사이에 빈 줄 추가 (마지막 문단 제외)
            if i < len(paragraphs) - 1:
                # 특정 조건에서 추가 공백 줄
                if any(keyword in paragraph for keyword in ['감사합니다', '감사드립니다', '평안하세요']):
                    # 마무리 인사 전에는 공백 없음
                    pass
                else:
                    # 일반 문단 사이에 빈 줄 추가 (가독성 향상)
                    html_paragraphs.append("<p><br></p>")
        
        return ''.join(html_paragraphs)

    ### 답변 텍스트 최종 정리 함수 (HTML 형식)
    def clean_answer_text(self, text: str) -> str:
        if not text:
            return ""
        
        # 1단계: 제어 문자 재제거 (AI 생성 과정에서 다시 들어올 수 있음)
        text = re.sub(r'[\b\r\f\v]', '', text)  # \b(백스페이스), \r, \f, \v 제거
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)  # 기타 제어 문자 제거
        
        # 2단계: 기존 HTML 태그 초기화 (깨끗한 상태에서 시작)
        text = re.sub(r'<[^>]+>', '', text)
        
        # 3단계: 마크다운 강조 제거 (AI가 마크다운으로 생성할 수 있음
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        
        # 4단계: 중복된 인사말/마무리 인사 제거
        greetings = ['안녕하세요', '안녕하세요.', '안녕하세요,']
        closings = ['감사합니다', '감사합니다.', '감사드립니다', '감사드립니다.']
        
        # 중복 인사말 처리
        for greeting in greetings:
            pattern = rf'({re.escape(greeting)}[,\s]*)+{re.escape(greeting)}'
            text = re.sub(pattern, greeting, text, flags=re.IGNORECASE)
        
        # 중복 마무리 인사 처리
        for closing in closings:
            pattern = rf'({re.escape(closing)}[,\s]*)+{re.escape(closing)}'
            text = re.sub(pattern, closing, text, flags=re.IGNORECASE)
        
        # 5단계 : 연속된 구두점 정리 (AI 생성 시 불규칙한 구두점 문제 해결)
        text = re.sub(r'[,\s]*,[,\s]*', ', ', text)
        text = re.sub(r'[.\s]*\.[.\s]*', '. ', text)
        
        # 6단계 : 연속된 공백을 하나로 통합
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\s+([,.!?])', r'\1', text)
        text = re.sub(r'([,.!?])\s+', r'\1 ', text)
        
        # "(구)다번역성경찬송" 제거
        text = self.remove_old_app_name(text)
        
        # HTML 문단 나누기 적용
        text = self.format_answer_with_html_paragraphs(text)
        
        return text

    ### ★ 4. T5 모델을 사용하여 실제 답변 생성 함수
    # 📌 T5 모델 작동 원리:
    # 1) 입력 프롬프트를 토큰으로 변환
    # 2) 인코더에서 입력 이해
    # 3) 디코더에서 토큰별로 순차 생성
    # 4) beam search로 여러 후보 중 최적 선택
    
    # 📌 프롬프트 엔지니어링:
    # - "질문: ... 참고답변: ... 답변:" 형식으로 T5에게 태스크 명시
    # - 상위 3개 답변만 컨텍스트로 사용 (토큰 512 제한 때문)
    def generate_with_t5(self, query: str, similar_answers: list) -> str:

        try:
            # 1단계 : 상위 3개 유사 답변을 컨텍스트로 사용하되, 제어 문자 제거
            context_answers = []
            for ans in similar_answers[:3]:
                clean_ans = ans['answer']
                # 제어 문자 제거 (T5 모델이 제어문자에 민감함)
                clean_ans = re.sub(r'[\b\r\f\v]', '', clean_ans)
                clean_ans = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', clean_ans)
                # HTML 태그 제거 (T5는 순수 텍스트 처리)
                clean_ans = re.sub(r'<[^>]+>', '', clean_ans)
                context_answers.append(clean_ans)
            
            context = " ".join(context_answers)
            
            # 2단계 : T5 모델용 프롬프트 구성
            prompt = f"질문: {query}\n참고답변: {context}\n답변:"
            
            # 3단계 : 텍스트 토큰화 (최대 512 토큰)
            inputs = text_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            # 4단계 : T5 모델로 답변 생성 (최대 200 토큰)
            outputs = text_model.generate(
                **inputs, 
                max_length=200,
                num_beams=4,  # 더 나은 품질을 위한 빔 서치 : 4개 후보를 동시 탐색하여 최적 선택
                early_stopping=True,
                do_sample=True, # 약간의 랜덤성 도입 (다양성)
                temperature=0.7  # 창의성 조절 (0.7 = 적절한 자연스러움)
            )
            
            # 5단계 : 생성된 토큰을 텍스트로로 디코딩
            generated = text_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 6단계 : 프롬프트 부분 제거하여 순수 답변만 추출
            if "답변:" in generated:
                generated = generated.split("답변:")[-1].strip()
            
            # 7단계 : 생성된 답변에서도 제어 문자 제거
            generated = re.sub(r'[\b\r\f\v]', '', generated)
            generated = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', generated)
            
            return generated
            
        except Exception as e:
            logging.error(f"T5 모델 생성 실패: {e}")
            # T5 실패 시 첫 번째 유사 답변 반환
            if similar_answers:
                fallback_answer = similar_answers[0]['answer']
                # 폴백 답변에서도 제어 문자 제거
                fallback_answer = re.sub(r'[\b\r\f\v]', '', fallback_answer)
                fallback_answer = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', fallback_answer)
                return fallback_answer
            return ""

    ### T5 모델을 사용한 AI 답변 생성 함수 (HTML 형식)
    # 유사 답변들을 컨텍스트로 활용하여 자연스러운 새로운 답변 생성
    def generate_ai_answer(self, query: str, similar_answers: list, lang: str) -> str:
        # 유사 답변이 없는 경우 기본 메시지 반환
        if not similar_answers:
            default_msg = "<p>문의해주신 내용에 대해 정확한 답변을 드리기 위해 더 자세한 정보가 필요합니다.</p><p><br></p><p>고객센터로 문의해주시면 신속하게 도움을 드리겠습니다.</p>"
            return default_msg
        
        try:
            # T5 모델을 사용하여 새로운 답변 생성
            base_answer = self.generate_with_t5(query, similar_answers)
            
            # T5 생성 결과가 비어있거나 너무 짧은 경우 폴백
            if not base_answer or len(base_answer.strip()) < 10:
                base_answer = similar_answers[0]['answer']
            
            # 기존 HTML 태그 제거 후 깨끗한 텍스트로 시작
            base_answer = re.sub(r'<[^>]+>', '', base_answer)
            
            # "(구)다번역성경찬송" 제거
            base_answer = self.remove_old_app_name(base_answer)
            
            # 기존 인사말이 있는지 확인
            has_greeting = any(greeting in base_answer.lower() for greeting in ['안녕하세요', '안녕'])
            has_closing = any(closing in base_answer.lower() for closing in ['감사합니다', '감사드립니다'])
            
            # 적절한 형태로 답변 구성
            final_answer = ""
            
            if not has_greeting:
                final_answer += "안녕하세요, GOODTV 바이블 애플입니다. "
            
            final_answer += base_answer
            
            if not has_closing:
                final_answer += " 항상 주님 안에서 평안하세요. 감사합니다."
            
            # 최종 HTML 정리 (문단 나누기 포함)
            final_answer = self.clean_answer_text(final_answer)
            
            return final_answer
            
        except Exception as e:
            logging.error(f"답변 생성 실패: {e}")
            # 모델 실패 시 폴백 전략
            if similar_answers:
                base_answer = similar_answers[0]['answer']
                base_answer = re.sub(r'<[^>]+>', '', base_answer)
                base_answer = self.remove_old_app_name(base_answer)
                base_answer = self.clean_answer_text(base_answer)
                
                # 인사말이 없으면 추가
                if not any(greeting in base_answer.lower() for greeting in ['안녕하세요', '안녕']):
                    base_answer = f"<p>안녕하세요, GOODTV 바이블 애플입니다.</p><p><br></p>{base_answer}"
                
                # 마무리 인사가 없으면 추가
                if not any(closing in base_answer.lower() for closing in ['감사합니다', '감사드립니다']):
                    base_answer = f"{base_answer}<p><br></p><p>감사합니다.</p>"
                
                return base_answer
            
            return "<p>죄송합니다. 현재 답변을 생성할 수 없습니다.</p><p><br></p><p>고객센터로 문의해주세요.</p>"

    # 전체 AI 답변 생성 파이프라인을 관리하는 메인 함수
    # 목적: 텍스트 전처리 → 유사도 검색 → AI 답변 생성의 전체 흐름 오케스트레이션
    # 데이터 흐름 순서: 질문 → 전처리 → 임베딩 → 검색 → AI생성 → 후처리 → HTML답변
    def process(self, seq: int, question: str, lang: str) -> dict:

        try:
            # 1단계: 사용자 질문 전처리
            processed_question = self.preprocess_text(question)
            if not processed_question: # 빈 질문 검증
                return {"success": False, "error": "질문이 비어있습니다."}
            
            # 처리 시작 로그 (질문 앞 50자만 기록)
            logging.info(f"처리 시작 - SEQ: {seq}, 질문: {processed_question[:50]}...")
            
            # 2단계: 유사한 기존 답변 검색
            similar_answers = self.search_similar_answers(processed_question)
            
            # 3단계: AI 답변 생성 (HTML 형식)
            ai_answer = self.generate_ai_answer(processed_question, similar_answers, lang)
            
            # 4단계: 답변 텍스트 최종 정리 (JSON 파싱 오류 방지)
            # 스마트 따옴표 정규화
            ai_answer = ai_answer.replace('"', '"').replace('"', '"')
            ai_answer = ai_answer.replace(''', "'").replace(''', "'")
            
            # HTML 태그 내부의 줄바꿈은 유지하되, 태그 외부 줄바꿈은 제거
            # (Quill 에디터에서 올바르게 렌더링되도록)
            
            # 5단계: 결과 구조화
            result = {
                "success": True,
                "answer": ai_answer,  # HTML 형식의 답변
                "similar_count": len(similar_answers), # 찾은 유사 답변 개수
                "embedding_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", # 사용된 임베딩 모델
                "generation_model": "google/flan-t5-base" # 사용된 생성 모델
            }
            
            logging.info(f"처리 완료 - SEQ: {seq}, HTML 답변 생성됨")
            return result
            
        except Exception as e:
            # 전체 프로세스 실패 시 로그 기록 및 에러 반환
            logging.error(f"처리 중 오류 - SEQ: {seq}, 오류: {str(e)}")
            return {"success": False, "error": str(e)}

# AI 답변 생성기 인스턴스 생성 (전역 객체)
generator = AIAnswerGenerator()

# Flask API 엔드포인트 정의
@app.route('/generate_answer', methods=['POST'])

    # AI 답변 생성 API 엔드포인트
    # 목적: HTTP POST 요청을 받아 AI 답변을 생성하고 JSON으로 응답
    # 요청 형식: {"seq": 123, "question": "질문내용", "lang": "kr"}
    # 응답 형식: {"success": true, "answer": "답변내용", ...}
def generate_answer():
    try:
        # HTTP 요청에서 JSON 데이터 파싱
        data = request.get_json()
        seq = data.get('seq', 0) # 요청 시퀀스 번호 (추적용)
        question = data.get('question', '') # 사용자 질문
        lang = data.get('lang', 'kr') # 응답 언어 (기본값: 한국어)
        
        # 필수 파라미터 검증
        if not question:
            return jsonify({"success": False, "error": "질문이 필요합니다."}), 400
        
        # AI 답변 생성 처리
        result = generator.process(seq, question, lang)
        
        # Flask jsonify가 자동으로 이스케이프 처리
        response = jsonify(result)
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response
        
    except Exception as e:
        # API 레벨 예외 처리
        logging.error(f"API 호출 오류: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

# 메인 실행 부분 (스크립트가 직접 실행될 때만)
if __name__ == "__main__":
    # 환경변수에서 포트 번호 읽기 (기본값: 8000)
    port = int(os.getenv('FLASK_PORT', 8000))
    
    print(f"Flask API starting on port {port}")
    
    # Flask 서버 시작
    # host='0.0.0.0': 모든 네트워크 인터페이스에서 접속 허용 (외부 접속 가능)
    # debug=False: 프로덕션 모드 (보안상 디버그 모드 비활성화)
    app.run(host='0.0.0.0', port=port, debug=False)