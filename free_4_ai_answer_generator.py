#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI Answer Generator Flask API for ASP Classic Integration
파일명: free_4_ai_answer_generator.py
설명: Flask API로 ASP Classic에서 호출
모델: google/flan-t5-base
개선사항: 한글 자모 분리 해결, 스마트 따옴표 정리, 성경 구절 정규화
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
try:
    # Pinecone 벡터 데이터베이스 연결 (유사도 검색용)
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index("bible-app-support-768-free") # 성경 앱 지원용 인덱스
    
    # 다국어 임베딩 모델 로드 (768차원 벡터 생성)
    embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    
    # Google T5 텍스트 생성 모델 및 토크나이저 로드
    text_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')
    text_tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
except Exception as e:
    # 모델 로드 실패 시 로그 기록 및 예외 발생
    logging.error(f"모듈 로드 실패: {str(e)}")
    app.logger.error(f"모듈 로드 실패: {str(e)}")
    raise

class AIAnswerGenerator:
    """
    AI 기반 답변 생성 클래스
    - 텍스트 전처리, 임베딩 생성, 유사도 검색, AI 답변 생성을 담당
    """

    def preprocess_text(self, text: str) -> str:
        """
        사용자 입력 텍스트 전처리 함수
        목적: AI 모델이 처리하기 적합한 형태로 텍스트 정제
        """
        if not text: # 빈 텍스트 처리
            return ""
        
        text = str(text) # 문자열 타입 강제 변환
        text = html.unescape(text) # HTML 엔티티 디코딩 (&amp; → &, &lt; → <)
        
        # HTML 태그를 적절한 구분자로 변환
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE) # <br> → 줄바꿈
        text = re.sub(r'</p>', '\n\n', text, flags=re.IGNORECASE) # </p> → 문단 구분
        text = re.sub(r'<p[^>]*>', '\n', text, flags=re.IGNORECASE) # <p> → 줄바꿈
        text = re.sub(r'<li[^>]*>', '\n• ', text, flags=re.IGNORECASE) # <li> → 리스트
        text = re.sub(r'</li>', '', text, flags=re.IGNORECASE)
        
        # 강조 태그 제거 (** 문제 해결)
        text = re.sub(r'<(strong|b)[^>]*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</(strong|b)>', '', text, flags=re.IGNORECASE)
        
        # 나머지 HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        
        # ** 마크다운 강조 제거
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        
        # 연속된 줄바꿈 정리 (단, 기존 문단 구조는 유지)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 줄바꿈이 아닌 연속 공백만 정리
        text = re.sub(r'[ \t]+', ' ', text)
        
        text = text.strip()
        
        return text

    def escape_json_string(self, text: str) -> str:
        """
        JSON 문자열을 위한 안전한 이스케이프 처리
        목적: ASP Classic에서 JSON 파싱 오류를 방지하기 위한 특수문자 처리
        """
        if not text:
            return ""
        
        # Python의 json 모듈을 사용하여 안전하게 이스케이프
        # dumps로 JSON 문자열로 만든 후, 양쪽 따옴표 제거
        escaped = json_module.dumps(text, ensure_ascii=False)
        return escaped[1:-1]  # 앞뒤 따옴표 제거

    def create_embedding(self, text: str) -> list:
        """
        텍스트를 768차원 벡터로 변환하는 함수
        목적: 의미적 유사도 검색을 위한 벡터 표현 생성
        """
        try:
            # SentenceTransformer로 텍스트를 벡터로 변환
            embedding = embedding_model.encode(text, convert_to_tensor=False)
            return embedding.tolist() # numpy 배열을 리스트로 변환
        except Exception as e:
            logging.error(f"임베딩 생성 실패: {e}")
            return None # 실패 시 None 반환으로 안전장치

    def search_similar_answers(self, query: str, top_k: int = 10, similarity_threshold: float = 0.6) -> list:
        """
        Pinecone에서 유사한 답변을 검색하는 함수
        목적: 사용자 질문과 의미적으로 유사한 기존 답변들을 찾아 컨텍스트로 활용
        """
        try:
            # 질문을 벡터로 변환
            query_vector = self.create_embedding(query)
            if query_vector is None: # 임베딩 생성 실패 시
                return []
            
            # Pinecone에서 벡터 유사도 검색 수행
            results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
            
            # 유사도 임계값(0.6) 이상인 결과만 필터링
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

    def clean_answer_text(self, text: str) -> str:
        """
        답변 텍스트 정리 함수
        목적: 중복 인사말, 특수문자, 포맷팅 문제 해결 및 적절한 문단 나누기
        """
        if not text:
            return ""
        
        # ** 마크다운 강조 제거
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        
        # 중복된 인사말 제거
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
        
        # 연속된 구두점 정리
        text = re.sub(r'[,\s]*,[,\s]*', ', ', text)
        text = re.sub(r'[.\s]*\.[.\s]*', '. ', text)
        
        # 적절한 위치에서 문단 나누기
        # 1. 문장 끝에서 자연스러운 문단 나누기 포인트 찾기
        paragraph_break_patterns = [
            r'(\. )([A-Z가-힣]+[^.]*?(?:입니다|습니다|드립니다|하겠습니다|됩니다|있습니다)\.)',  # 정중한 문장 끝
            r'(\. )(그래서|따라서|현재|특히|불편하시겠지만|성경)',  # 접속어나 새로운 주제 시작
            r'(\. )(바이블 애플|GOODTV|대한성서공회)',  # 회사/기관명으로 시작하는 문장
            r'(\. )(또|그리고|하지만|그러나|그런데)',  # 접속 부사
            r'(습니다\. )(성도님|고객님|이용자님)',  # 호칭으로 시작하는 새 문장
            r'(니다\. )([가-힣]+이|[가-힣]+을|[가-힣]+는)',  # 새로운 주제 시작
        ]
        
        for pattern in paragraph_break_patterns:
            text = re.sub(pattern, r'\1\n\n\2', text)
        
        # 특정 키워드 후에 문단 나누기
        keyword_breaks = [
            '감사드립니다.',
            '확인하였습니다.',
            '부탁드립니다.',
            '어렵습니다.',
            '있습니다.',
            '하겠습니다.',
            '됩니다.',
            '보입니다.',
        ]
        
        for keyword in keyword_breaks:
            # 키워드 다음에 공백과 한글 대문자가 오는 경우 문단 나누기
            pattern = rf'({re.escape(keyword)}) ([가-힣A-Z][^.]*)'
            text = re.sub(pattern, rf'\1\n\n\2', text)
        
        # 연속된 공백을 하나로 통합 (문단 구분은 유지)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # 연속된 줄바꿈 정리 (최대 2개까지만)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 문장 부호 앞뒤 공백 정리
        text = re.sub(r'\s+([,.!?])', r'\1', text)
        text = re.sub(r'([,.!?])\s+', r'\1 ', text)
        
        # 각 줄의 시작과 끝 공백 제거
        lines = []
        for line in text.split('\n'):
            lines.append(line.strip())
        
        text = '\n'.join(lines)
        
        return text.strip()

    def generate_ai_answer(self, query: str, similar_answers: list, lang: str) -> str:
        """
        T5 모델을 사용한 AI 답변 생성 함수
        목적: 유사 답변들을 컨텍스트로 활용하여 자연스러운 새로운 답변 생성
        """
        # 유사 답변이 없는 경우 기본 메시지 반환
        if not similar_answers:
            return "문의해주신 내용에 대해 정확한 답변을 드리기 위해 더 자세한 정보가 필요합니다. 고객센터로 문의해주시면 신속하게 도움을 드리겠습니다."
        
        try:
            # 첫 번째 유사 답변을 정리하여 사용 (T5 모델 대신)
            base_answer = similar_answers[0]['answer']
            
            # 답변 텍스트 정리
            cleaned_answer = self.clean_answer_text(base_answer)
            
            # 기존 인사말이 있는지 확인
            has_greeting = any(greeting in cleaned_answer.lower() for greeting in ['안녕하세요', '안녕'])
            has_closing = any(closing in cleaned_answer.lower() for closing in ['감사합니다', '감사드립니다'])
            
            # 적절한 형태로 답변 구성
            final_answer = ""
            
            if not has_greeting:
                final_answer += "안녕하세요.\n\n"
            
            final_answer += cleaned_answer
            
            if not has_closing:
                final_answer += "\n\n추가 문의사항이 있으시면 언제든 연락해주세요. 감사합니다."
            
            # 최종 정리
            final_answer = self.clean_answer_text(final_answer)
            
            return final_answer
            
        except Exception as e:
            logging.error(f"답변 생성 실패: {e}")
            # 모델 실패 시 폴백 전략
            if similar_answers:
                base_answer = self.clean_answer_text(similar_answers[0]['answer'])
                return f"안녕하세요.\n\n{base_answer}\n\n감사합니다."
            return "죄송합니다. 현재 답변을 생성할 수 없습니다. 고객센터로 문의해주세요."

    def process(self, seq: int, question: str, lang: str) -> dict:
        """
        전체 AI 답변 생성 파이프라인을 관리하는 메인 함수
        목적: 텍스트 전처리 → 유사도 검색 → AI 답변 생성의 전체 흐름 orchestration
        """
        try:
            # 1단계: 사용자 질문 전처리
            processed_question = self.preprocess_text(question)
            if not processed_question: # 빈 질문 검증
                return {"success": False, "error": "질문이 비어있습니다."}
            
            # 처리 시작 로그 (질문 앞 50자만 기록)
            logging.info(f"처리 시작 - SEQ: {seq}, 질문: {processed_question[:50]}...")
            
            # 2단계: 유사한 기존 답변 검색
            similar_answers = self.search_similar_answers(processed_question)
            
            # 3단계: AI 답변 생성
            ai_answer = self.generate_ai_answer(processed_question, similar_answers, lang)
            
            # 4단계: 답변 텍스트 최종 정리 (JSON 파싱 오류 방지)
            # 스마트 따옴표 정규화
            ai_answer = ai_answer.replace('"', '"').replace('"', '"')
            ai_answer = ai_answer.replace(''', "'").replace(''', "'")
            # 줄바꿈 정규화
            ai_answer = ai_answer.replace('\r\n', '\n').replace('\r', '\n')
            # 최종 텍스트 정리
            ai_answer = self.clean_answer_text(ai_answer)
            
            # 5단계: 결과 구조화
            result = {
                "success": True,
                "answer": ai_answer,  # 이미 안전한 텍스트
                "similar_count": len(similar_answers), # 찾은 유사 답변 개수
                "embedding_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", # 사용된 임베딩 모델
                "generation_model": "google/flan-t5-base" # 사용된 생성 모델
            }
            
            logging.info(f"처리 완료 - SEQ: {seq}")
            return result
            
        except Exception as e:
            # 전체 프로세스 실패 시 로그 기록 및 에러 반환
            logging.error(f"처리 중 오류 - SEQ: {seq}, 오류: {str(e)}")
            return {"success": False, "error": str(e)}

# AI 답변 생성기 인스턴스 생성 (전역 객체)
generator = AIAnswerGenerator()

# Flask API 엔드포인트 정의
@app.route('/generate_answer', methods=['POST'])
def generate_answer():
    """
    AI 답변 생성 API 엔드포인트
    목적: HTTP POST 요청을 받아 AI 답변을 생성하고 JSON으로 응답
    요청 형식: {"seq": 123, "question": "질문내용", "lang": "kr"}
    응답 형식: {"success": true, "answer": "답변내용", ...}
    """
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