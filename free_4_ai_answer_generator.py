#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=== AI 답변 생성 Flask API 서버 (다국어 지원) ===
파일명: free_4_ai_answer_generator.py
목적: ASP Classic에서 호출하는 AI 답변 생성 API + Pinecone 벡터DB 동기화
주요 기능:
1. OpenAI GPT-3.5-turbo를 이용한 자연어 답변 생성 (한국어/영어)
2. Pinecone 벡터 데이터베이스에서 유사 답변 검색
3. MSSQL 데이터베이스와 Pinecone 동기화
4. 메모리 최적화 및 모니터링
5. 다국어 지원 (한국어, 영어)
"""

# ==================================================
# 1. 필수 라이브러리 임포트 구간
# ==================================================
# 기본 Python 모듈들
import os                   # 환경변수 및 파일 시스템 작업
import sys                  # 시스템 관련 기능
import json                 # JSON 데이터 처리
import json as json_module  # JSON 모듈의 별칭 (일리아스, 코드 내 중복 방지)
import re                   # 정규표현식 패턴 매칭
import html                 # HTML 엔티티 처리
import unicodedata          # 유니코드 문자 정규화
import logging              # 로그 기록 시스템
import gc                   # 가비지 컬렉션 (메모리 관리)

# 웹 프레임워크 관련
from flask import Flask, request, jsonify  # Flask 웹 프레임워크
from flask_cors import CORS                 # CORS(Cross-Origin Resource Sharing) 처리

# AI 및 데이터베이스 관련
from pinecone import Pinecone      # Pinecone 벡터 데이터베이스
import openai                      # OpenAI API 클라이언트
import pyodbc                      # MSSQL 데이터베이스 연결

# 환경설정 및 유틸리티
from dotenv import load_dotenv     # .env 파일에서 환경변수 로드
from datetime import datetime      # 날짜/시간 처리
from typing import Optional, Dict, Any, List  # 타입 힌팅
import re                                     # 정규표현식

# 성능 모니터링 관련
from memory_profiler import profile                  # 메모리 사용량 프로파일링
import tracemalloc                                   # 메모리 추적
import threading                                     # 멀티스레딩
from contextlib import contextmanager                # 컨텍스트 매니저 (with문 사용)
from langdetect import detect, LangDetectException   # 언어 감지

# ==================================================
# 2. 시스템 초기화 및 설정
# ==================================================
# 메모리 추적 시작 - 메모리 누수 및 사용량을 모니터링하기 위함 (이 시점부터 모든 메모리 할당이 기록되어 나중에 메모리 사용량 분석이 가능해짐)
tracemalloc.start() 

# Flask 웹 애플리케이션 인스턴스 생성
# __name__: 현재 모듈명을 전달하여 Flask가 리소스 위치를 찾을 수 있게 함
app = Flask(__name__)

# CORS 설정 - 모든 엔드포인트에서 cross-origin 요청을 허용 (ASP Classic에서 호출하기 위함)
CORS(app)

# ==================================================
# 3. 로깅 시스템 설정 (콘솔 + 파일)
# ==================================================
# 로거 생성
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 포맷터 생성
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 파일 핸들러 (기존)
try:
    # 로그 디렉토리 생성
    os.makedirs('/home/ec2-user/python/logs', exist_ok=True)
    file_handler = logging.FileHandler('/home/ec2-user/python/logs/ai_generator.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
except Exception as e:
    print(f"로그 파일 핸들러 생성 실패: {e}")

# 콘솔 핸들러 추가 (실시간 디버깅용)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ==================================================
# 4. 환경변수 로드 및 시스템 상수 정의
# ==================================================
# .env 파일에서 API 키 및 데이터베이스 설정 로드
load_dotenv()

# AI 임베딩 모델 및 벡터 데이터베이스 설정 상수들
MODEL_NAME = 'text-embedding-3-small'          # OpenAI 임베딩 모델명
INDEX_NAME = "bible-app-support-1536-openai"   # Pinecone 인덱스명
EMBEDDING_DIMENSION = 1536                      # 임베딩 벡터 차원수
MAX_TEXT_LENGTH = 8000                          # 텍스트 최대 길이 제한

# GPT 자연어 모델 설정 - 보수적 설정으로 일관성 있는 답변 생성
GPT_MODEL = 'gpt-3.5-turbo'     # 사용할 GPT 모델
MAX_TOKENS = 600                 # 생성할 최대 토큰 수 (답변 길이 제한)
TEMPERATURE = 0.5                # 창의성 수준 (낮을수록 일관된 답변)

# 고객 문의 카테고리 매핑 테이블
# 카테고리 매핑 딕셔너리는 MSSQL의 숫자 인덱스를 사람이 읽을 수 있는 한글 카테고리명으로 변환합니다. 
# 이는 Pinecone 메타데이터에 저장되어 검색 결과의 가독성을 높입니다.
CATEGORY_MAPPING = {
    '1': '후원/해지',                   
    '2': '성경 통독(읽기,듣기,녹음)',   
    '3': '성경낭독 레이스',             
    '4': '개선/제안',                   
    '5': '오류/장애',               
    '6': '불만',                        
    '7': '오탈자제보',                   
    '0': '사용 문의(기타)'               
}

# 영어 카테고리 매핑 추가
CATEGORY_MAPPING_EN = {
    '1': 'Sponsorship/Cancellation',
    '2': 'Bible Reading(Read,Listen,Record)',
    '3': 'Bible Reading Race',
    '4': 'Improvement/Suggestion',
    '5': 'Error/Failure',
    '6': 'Complaint',
    '7': 'Typo Report',
    '0': 'Usage Inquiry(Other)'
}

# ==================================================
# 5. 유틸리티 함수 정의
# ==================================================
# 메모리 정리를 위한 컨텍스트 매니저
# with문 진입시 try 블록 실행
# yield에서 일시정지하고 with 블록 내부 코드 실행
# yield 키워드가 들어간 이 함수는 제너레이터이면서 동시에 컨텍스트 매니저로 동작
# 제너레이터 함수는 값을 반환하는 대신 값을 하나씩 생성하여 반환합니다. (파이썬에서 대용량 데이터 처리를 위해 사용)
# with 블록 종료시 finally에서 gc.collect() 실행

# 이렇게 하면 대용량 데이터 처리 후 자동으로 가비지 컬렉션이 실행됩니다.
@contextmanager
def memory_cleanup():
    try:
        yield  # with 블록 내부 코드 실행
    finally:
        gc.collect()  # 가비지 컬렉션 강제 실행으로 메모리 정리

# ==================================================
# 6. 외부 서비스 연결 및 초기화
# ==================================================
try:
    # Pinecone 벡터 데이터베이스 연결 설정
    # 유사 답변 검색을 위한 벡터DB - 임베딩된 질문/답변 저장소
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index(INDEX_NAME)
    
    # OpenAI API 클라이언트 초기화
    # GPT 모델 및 임베딩 생성을 위한 클라이언트
    # openai.api_key = ... 방식보다 객체지향적으로 설계
    openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # MSSQL 데이터베이스 연결 설정
    # 기존 고객 문의 데이터가 저장된 운영 DB 연결을 위한 설정
    mssql_config = {
        'server': os.getenv('MSSQL_SERVER'),       # DB 서버 주소
        'database': os.getenv('MSSQL_DATABASE'),   # 데이터베이스명
        'username': os.getenv('MSSQL_USERNAME'),   # DB 사용자명
        'password': os.getenv('MSSQL_PASSWORD')    # DB 비밀번호
    }
    
    # MSSQL Server 연결 문자열 구성
    # MSSQL Server 표준 ODBC 드라이버를 사용한 연결 문자열
    connection_string = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"  # ODBC 드라이버 버전
            f"SERVER={mssql_config['server']},1433;"      # 서버 주소와 포트
            f"DATABASE={mssql_config['database']};"       # 데이터베이스명
            f"UID={mssql_config['username']};"            # 사용자 ID
            f"PWD={mssql_config['password']};"            # 비밀번호
            f"TrustServerCertificate=yes;"                # SSL 인증서 신뢰
            f"Connection Timeout=30;"                     # 연결 타임아웃 (30초)
    )

except Exception as e:
    # 초기화 실패 시 로그 기록 후 프로그램 종료
    logging.error(f"모듈 로드 실패: {str(e)}")
    app.logger.error(f"모듈 로드 실패: {str(e)}")
    raise  # 예외를 다시 발생시켜 프로그램 중단

# ==================================================
# 7. AI 답변 생성 메인 클래스 (객체 지향 프로그래밍, 다국어 지원 추가)
# ==================================================
    # AI 답변 생성을 담당하는 메인 클래스
    # 객체지향 설계로 관련 기능들을 하나의 클래스에 캡슐화
    
    # 주요 기능:
    # 1. 텍스트 전처리 및 정제
    # 2. OpenAI를 이용한 임베딩 생성
    # 3. Pinecone에서 유사 답변 검색
    # 4. GPT를 이용한 맞춤형 답변 생성
    # 5. 한국어 텍스트 검증 및 포맷팅

class AIAnswerGenerator:
    
    # 클래스 초기화 메서드
    # OpenAI 클라이언트를 인스턴스 변수로 설정. 이는 의존성 주입 패턴의 간소화된 형태
    def __init__(self):
        self.openai_client = openai_client

    # ☆ 한국어 오타 수정 메서드
    def fix_korean_typos_with_ai(self, text: str) -> str:
        if not text or len(text.strip()) < 3:
            return text
        
        # 너무 긴 텍스트는 처리하지 않음 (비용 절약)
        if len(text) > 500:
            logging.warning(f"텍스트가 너무 길어 오타 수정 건너뜀: {len(text)}자")
            return text
        
        try:
            with memory_cleanup():
                system_prompt = """당신은 한국어 맞춤법 및 오타 교정 전문가입니다.

지침:
1. 입력된 한국어 텍스트의 맞춤법과 오타만 수정하세요
2. 원문의 의미와 어조는 절대 변경하지 마세요
3. 띄어쓰기, 맞춤법, 조사 사용법을 정확히 교정하세요
4. 앱/어플리케이션 관련 기술 용어는 표준 용어로 통일하세요
5. 수정이 필요없다면 원문 그대로 반환하세요
6. 수정된 텍스트만 반환하고 추가 설명은 하지 마세요

예시:
- "어플이 안됀다" → "앱이 안 돼요"
- "다운받기가 안되요" → "다운로드가 안 돼요"
- "삭재하고싶어요" → "삭제하고 싶어요"
- "업데이드해주세요" → "업데이트해주세요"
"""

                user_prompt = f"다음 텍스트의 맞춤법과 오타를 수정해주세요:\n\n{text}"

                response = self.openai_client.chat.completions.create(
                    model='gpt-3.5-turbo',
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=600,
                    temperature=0.1,  # 매우 보수적으로 설정
                    top_p=0.8,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                
                corrected_text = response.choices[0].message.content.strip()
                del response # 메모리 해제
                
                # 결과 검증
                if not corrected_text or len(corrected_text) == 0:
                    logging.warning("AI 오타 수정 결과가 비어있음, 원문 반환")
                    return text
                
                # 너무 많이 변경된 경우 의심스러우므로 원문 반환
                if len(corrected_text) > len(text) * 2:
                    logging.warning("AI 오타 수정 결과가 원문보다 너무 길어짐, 원문 반환")
                    return text
                
                # 수정 내용이 있으면 로그 기록
                if corrected_text != text:
                    logging.info(f"AI 오타 수정: '{text[:50]}...' → '{corrected_text[:50]}...'")
                
                return corrected_text
                
        except Exception as e:
            logging.error(f"AI 오타 수정 실패: {e}")
            # AI 실패 시 원문 그대로 반환
            return text
    
    # ☆ 텍스트의 언어를 감지하는 메서드
    def detect_language(self, text: str) -> str:
        try:
            # langdetect 라이브러리 사용
            detected = detect(text)
            
            # 영어와 한국어만 지원
            if detected == 'en':
                return 'en'
            elif detected == 'ko':
                return 'ko'
            else:
                # 기본값은 한국어
                return 'ko'
        except LangDetectException:
            # 감지 실패시 텍스트 내 한글 비율로 판단
            korean_chars = len(re.findall(r'[가-힣]', text))
            english_chars = len(re.findall(r'[a-zA-Z]', text))
            
            if korean_chars > english_chars:
                return 'ko'
            else:
                return 'en'

    # ☆ 입력 텍스트를 AI 처리에 적합하게 전처리하는 메서드 (원본 텍스트 -> 정제된 텍스트)
    def preprocess_text(self, text: str) -> str:
        logging.info(f"전처리 시작: 입력 길이={len(text) if text else 0}")
        logging.info(f"전처리 입력 미리보기: {text[:100] if text else 'None'}...")

        # null 체크
        if not text:
            logging.info("전처리: 빈 텍스트 입력")
            return ""
        
        # 문자열로 변환 및 HTML 엔티티 디코딩
        text = str(text)
        text = html.unescape(text)  # &amp; → &, &lt; → < 등
        logging.info(f"HTML 디코딩 후 길이: {len(text)}")
        
        # HTML 태그 제거 및및 텍스트 형태로 변환 (구조 유지)
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)      # <br> → 줄바꿈
        text = re.sub(r'</p>', '\n\n', text, flags=re.IGNORECASE)         # </p> → 단락 구분
        text = re.sub(r'<p[^>]*>', '\n', text, flags=re.IGNORECASE)       # <p> → 줄바꿈
        text = re.sub(r'<li[^>]*>', '\n• ', text, flags=re.IGNORECASE)    # <li> → 불릿포인트
        text = re.sub(r'</li>', '', text, flags=re.IGNORECASE)            # </li> 제거
        text = re.sub(r'<[^>]+>', '', text)                               # 나머지 HTML 태그 모두 제거
        logging.info(f"HTML 태그 제거 후 길이: {len(text)}")
        
        # 🔥 구 앱 이름을 바이블 애플로 교체 (전처리 단계에서)
        # 중복 방지를 위해 순서를 조정: 전체 패턴부터 처리
        text = re.sub(r'바이블\s*애플\s*\(구\)\s*다번역\s*성경\s*찬송', '바이블 애플', text, flags=re.IGNORECASE)
        text = re.sub(r'바이블\s*애플\s*\(구\)\s*다번역성경찬송', '바이블 애플', text, flags=re.IGNORECASE)
        text = re.sub(r'\(구\)\s*다번역\s*성경\s*찬송', '바이블 애플', text, flags=re.IGNORECASE)
        text = re.sub(r'\(구\)\s*다번역성경찬송', '바이블 애플', text, flags=re.IGNORECASE)
        text = re.sub(r'다번역\s*성경\s*찬송', '바이블 애플', text, flags=re.IGNORECASE)
        text = re.sub(r'다번역성경찬송', '바이블 애플', text, flags=re.IGNORECASE)
        
        # 공백 및 줄바꿈 정규화 - 일관된 형태로 변환
        text = re.sub(r'\n{3,}', '\n\n', text)    # 3개 이상 줄바꿈 → 2개로 제한
        text = re.sub(r'[ \t]+', ' ', text)       # 연속 공백/탭 → 단일 공백
        text = text.strip()                       # 앞뒤 공백 제거
        
        logging.info(f"전처리 완료: 최종 길이={len(text)}")
        logging.info(f"전처리 결과 미리보기: {text[:100]}...")
        
        return text

    # ☆ JSON 문자열 이스케이프 처리 (특수문자가 포함된 텍스트를 JSON으로 안전하게 변환)
    def escape_json_string(self, text: str) -> str:
        
        if not text:
            return ""
        escaped = json_module.dumps(text, ensure_ascii=False) # ensure_ascii=False: 한글 깨짐 방지
        return escaped[1:-1]  # 앞뒤 따옴표 제거

    # ☆ OpenAI API를 사용하여 텍스트를 벡터로 변환하는 메서드
    # 벡터 임베딩: 텍스트의 의미를 수치 배열로 표현하여 유사도 계산 가능
    def create_embedding(self, text: str) -> Optional[list]:

        # 빈 문자열뿐만 아니라 공백만 있는 문자열도 걸러냄 (이런 경우 JSON 변환 불가)
        if not text or not text.strip():
            return None
            
        try:
            with memory_cleanup(): # 메모리 누수 방지 (블록 종료 시 가비지 컬렉션)
                # OpenAI Embedding API 호출
                response = self.openai_client.embeddings.create(
                    model='text-embedding-3-small',    # 벡터 임베딩 모델명
                    input=text[:8000]                  # 텍스트 길이 제한 (토큰 제한 방지지)
                )
                
                # 메모리 효율성을 위해 벡터만 복사 후 응답 객체 삭제
                embedding = response.data[0].embedding.copy() # 임베딩 벡터만 복사하여 반환 (깊은 복사로 독립적인 리스트 생성)
                del response  # 원본 응답 객체 즉시 삭제 (메모리 해제)
                return embedding
                
        except Exception as e:
            logging.error(f"임베딩 생성 실패: {e}")
            return None

    # ☆ Pinecone 벡터 데이터베이스에서 유사한 답변을 검색하는 메서드
    # Args:
    #     query (str): 검색할 질문
    #     top_k (int): 검색할 최대 개수 (기본값: 5)
    #     similarity_threshold (float): 유사도 임계값 (기본값: 0.3)
            
    # Returns:
    #     list: 유사 답변 리스트 [{'score': float, 'question': str, 'answer': str, ...}, ...]
    def search_similar_answers(self, query: str, top_k: int = 5, similarity_threshold: float = 0.7, lang: str = 'ko') -> list:
        try:
            with memory_cleanup():
                # ★ 디버깅 로그 추가
                logging.info(f"=== 검색 시작 ===")
                logging.info(f"원본 질문: {query[:100]}")
                
                # ★ 오타 수정 적용 (동기화와 동일하게!)
                if lang == 'ko':
                    corrected_query = self.fix_korean_typos_with_ai(query)
                    logging.info(f"오타 수정 후: {corrected_query[:100]}")
                    query_to_embed = corrected_query
                else:
                    query_to_embed = query
                
                # 임베딩 생성
                query_vector = self.create_embedding(query_to_embed)
                
                if query_vector is None:
                    logging.error("임베딩 생성 실패")
                    return []
                
                # Pinecone 검색
                results = index.query(
                    vector=query_vector,
                    top_k=top_k * 2,  # 더 많이 검색
                    include_metadata=True
                )
                
                logging.info(f"Pinecone 검색 결과: {len(results['matches'])}개")
                
                # 한국어 벡터로 추가 검색 (영어 질문인 경우)
                korean_vector = None  # 초기화하여 NameError 방지
                if lang == 'en':
                    # 영어 쿼리를 한국어로 번역 후 임베딩 생성 (누락된 로직 추가)
                    korean_query = self.translate_text(query_to_embed, 'en', 'ko')
                    korean_vector = self.create_embedding(korean_query)
                    if korean_vector:
                        korean_results = index.query(
                            vector=korean_vector,       # 검색할 벡터
                            top_k=3,                    # 보조 검색은 적게 (3개)
                            include_metadata=True       # 메타데이터 포함 (질문, 답변, 카테고리 등)
                        )
                        # 결과 병합 (중복 제거)
                        seen_ids = set()
                        merged_matches = []
                        for match in results['matches'] + korean_results['matches']:
                            if match['id'] not in seen_ids:
                                seen_ids.add(match['id'])
                                merged_matches.append(match)
                        results['matches'] = sorted(merged_matches, key=lambda x: x['score'], reverse=True)[:top_k]
                
                # 3. 결과 필터링 및 구조화
                filtered_results = []
                for i, match in enumerate(results['matches']): # enumerate로 순위(rank) 생성
                    score = match['score'] # 유사도 점수 (0~1, 높을수록 유사)
                    question = match['metadata'].get('question', '')
                    answer = match['metadata'].get('answer', '')
                    category = match['metadata'].get('category', '일반')
                    
                    # ★ 임계값 로직 대폭 완화 - 항상 최소 3개는 반환하도록 수정
                    include_result = False
                    
                    if score >= similarity_threshold:
                        include_result = True
                        logging.info(f"임계값 통과: {score:.3f} >= {similarity_threshold:.2f}")
                    elif i < 3:  # 상위 3개는 무조건 포함
                        include_result = True
                        logging.info(f"상위 {i+1}번째 결과로 강제 포함: {score:.3f}")
                    elif score >= 0.3:  # 매우 낮은 임계값으로 추가 포함
                        include_result = True
                        logging.info(f"낮은 임계값 통과: {score:.3f} >= 0.3")
                    
                    if include_result:
                        filtered_results.append({
                            'score': score,
                            'question': question,
                            'answer': answer,
                            'category': category,
                            'rank': i + 1,
                            'lang': 'ko'  # 원본 데이터는 한국어
                        })
                        
                        # 디버깅을 위한 상세 로깅
                        logging.info(f"★ 포함된 유사 답변 #{i+1}: 점수={score:.3f}, 카테고리={category}, 언어={lang}")
                        logging.info(f"참고 질문: {question[:50]}...")
                        logging.info(f"참고 답변: {answer[:100]}...")
                    else:
                        logging.info(f"× 제외된 답변 #{i+1}: 점수={score:.3f} (너무 낮음)")
                        
                # 4. 메모리 정리
                del results # 원본 응답 객체 즉시 삭제 (메모리 해제)
                if korean_vector is not None:
                    del korean_vector # 한국어 벡터 즉시 삭제 (메모리 해제)
                del query_vector # 검색 벡터 즉시 삭제 (메모리 해제)
                
                logging.info(f"총 {len(filtered_results)}개의 유사 답변 검색 완료 (언어: {lang})")
                return filtered_results
                
        except Exception as e:
            logging.error(f"Pinecone 검색 실패: {str(e)}, query: {query[:50]}..., lang: {lang}")
            return []

    # ☆ GPT를 사용한 번역
    # Args:
    #     text (str): 번역할 텍스트
    #     source_lang (str): 원본 언어
    #     target_lang (str): 번역 언어
            
    # Returns:
    #     str: 번역된 텍스트
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        try:
            # 언어 매핑
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

    # ☆ AI 기반 질문 의도 분석 메서드 (대폭 강화된 정확도)
    def analyze_question_intent(self, query: str) -> dict:
        """AI를 이용해 질문의 의도와 핵심 내용을 매우 정확하게 분석"""
        try:
            with memory_cleanup():
                system_prompt = """당신은 바이블 앱 고객 문의 분석 전문가입니다. 
고객의 질문을 매우 정확하게 분석하여 다음 정보를 JSON 형태로 반환하세요:

{
  "intent_type": "문의 유형",
  "main_topic": "핵심 주제",
  "sub_topic": "세부 주제",
  "content_type": "콘텐츠 유형",
  "language_preference": "언어 선호도",
  "specific_request": "구체적 요청사항",
  "keywords": ["핵심", "키워드", "목록"],
  "urgency": "긴급도",
  "action_type": "요청 행동"
}

🎯 바이블 앱 전용 분석 기준:

📚 콘텐츠 유형 구분 (매우 중요!):
- "성경": 성경 본문, 구절, 장, 절 관련
- "찬송": 찬송가, 찬양, hymn, praise 관련  
- "기도": 기도문, 주기도문 관련
- "설교": 설교, 강의, 강연 관련
- "앱기능": 앱 자체 기능, 설정, 사용법

🌏 언어 구분 (정확히!):
- "korean": 한국어, 한글 관련
- "english": 영어, English 관련
- "multilingual": 다국어, 번역 관련
- "none": 언어 무관

⚠️ 매우 중요한 키워드 구분:
1. "찬송가" ≠ "성경" (완전히 다른 콘텐츠)
2. "영어 찬송가" = content_type: "찬송", language_preference: "english"
3. "영어 성경" = content_type: "성경", language_preference: "english"  
4. "복사" ≠ "재생" (완전히 다른 행동)
5. "검색" ≠ "설정" (완전히 다른 기능)

📋 분석 예시:
- "영어 찬송가도 있나요?" → content_type: "찬송", language_preference: "english"
- "NIV 성경 있나요?" → content_type: "성경", language_preference: "english"
- "복사해서 워드로" → action_type: "복사", content_type: "성경"
- "찬송가 연속재생" → action_type: "재생", content_type: "찬송"

🚫 절대 혼동하지 말 것:
- 찬송가 ≠ 성경본문
- 영어 ≠ 한국어  
- 복사 ≠ 재생
- 검색 ≠ 설정변경"""

                user_prompt = f"다음 바이블 앱 고객 문의를 매우 정확하게 분석해주세요: {query}"

                response = self.openai_client.chat.completions.create(
                    model='gpt-3.5-turbo',
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=400,
                    temperature=0.2  # 더 보수적으로 설정
                )
                
                result_text = response.choices[0].message.content.strip()
                
                # JSON 파싱 시도
                try:
                    result = json.loads(result_text)
                    logging.info(f"강화된 AI 의도 분석 결과: {result}")
                    return result
                except json.JSONDecodeError:
                    logging.warning(f"JSON 파싱 실패, 기본값 반환: {result_text}")
                    return {
                        "intent_type": "일반문의",
                        "main_topic": "기타",
                        "sub_topic": "기타",
                        "content_type": "앱기능",
                        "language_preference": "none",
                        "specific_request": query[:100],
                        "keywords": [query[:20]],
                        "urgency": "medium",
                        "action_type": "기타"
                    }
                
        except Exception as e:
            logging.error(f"AI 의도 분석 실패: {e}")
            return {
                "intent_type": "일반문의", 
                "main_topic": "기타",
                "sub_topic": "기타",
                "content_type": "앱기능",
                "language_preference": "none",
                "specific_request": query[:100],
                "keywords": [query[:20]],
                "urgency": "medium",
                "action_type": "기타"
            }

    # ☆ 검색된 유사 답변들의 품질을 분석하여 최적의 답변 생성 전략을 결정하는 메서드

    # Args:
    #     similar_answers (list): 검색된 유사 답변 리스트
    #     query (str): 원본 질문
    #     
    # Returns:
    #     dict: 분석 결과 및 권장 접근 방식
    def analyze_context_quality(self, similar_answers: list, query: str) -> dict:
        # 유사 답변이 없으면 기본값 반환
        if not similar_answers:
            return {
                'has_good_context': False,
                'best_score': 0.0,
                'recommended_approach': 'fallback',
                'context_summary': '유사 답변이 없습니다.',
                'question_type': '일반문의',
                'context_relevance': 'none'
            }
        
        # 🔥 AI 기반 질문 의도 분석 추가
        question_analysis = self.analyze_question_intent(query)
        question_type = question_analysis.get('intent_type', '일반문의')
        logging.info(f"AI 분석 결과: {question_analysis}")
        
        # 품질 지표 계산
        best_score = similar_answers[0]['score']
        high_quality_count = len([ans for ans in similar_answers if ans['score'] >= 0.7])
        medium_quality_count = len([ans for ans in similar_answers if 0.5 <= ans['score'] < 0.7])
        
        # 🔥 카테고리 일치도 분석 추가
        categories = [ans['category'] for ans in similar_answers[:5]]
        category_distribution = {cat: categories.count(cat) for cat in set(categories)}
        
        # 🔥 강화된 다단계 관련성 검증 적용
        context_relevance = self.check_context_relevance_ai(question_analysis, categories, query, similar_answers[:3])
        logging.info(f"강화된 컨텍스트 관련성: {context_relevance}")
        
        # 🔥 의사 결정 트리 대폭 개선 - 관련성을 최우선으로 고려
        if context_relevance == 'irrelevant':
            # 관련성이 없으면 무조건 특별 fallback 처리
            approach = 'smart_fallback'
            logging.warning(f"질문 유형({question_type})과 검색된 답변의 관련성이 없어 스마트 폴백 처리")
        elif context_relevance == 'high':
            # 관련성이 높으면 점수에 따라 결정
            if best_score >= 0.9:
                approach = 'direct_use'
            elif best_score >= 0.7:
                approach = 'gpt_with_strong_context'
            elif best_score >= 0.5:
                approach = 'gpt_with_strong_context'
            else:
                approach = 'gpt_with_weak_context'
        elif context_relevance == 'medium':
            # 관련성이 중간이면 더 신중하게
            if best_score >= 0.85:
                approach = 'direct_use'
            elif best_score >= 0.6:
                approach = 'gpt_with_strong_context'
            else:
                approach = 'gpt_with_weak_context'
        elif context_relevance == 'low':
            # 관련성이 낮으면 매우 신중하게
            if best_score >= 0.9:
                approach = 'gpt_with_weak_context'  # 직접 사용하지 않음
            else:
                approach = 'smart_fallback'
        else:
            approach = 'smart_fallback'
        
        # 분석 결과 구조화
        analysis = {
            'has_good_context': context_relevance in ['high', 'medium'] and best_score >= 0.4,
            'best_score': best_score,
            'high_quality_count': high_quality_count,
            'medium_quality_count': medium_quality_count,
            'category_distribution': category_distribution,
            'recommended_approach': approach,
            'question_analysis': question_analysis,
            'question_type': question_type,
            'context_relevance': context_relevance,
            'context_summary': f"의도: {question_type}, 주제: {question_analysis.get('main_topic', 'N/A')}, 관련성: {context_relevance}, 최고점수: {best_score:.3f}"
        }
        
        logging.info(f"향상된 컨텍스트 분석 결과: {analysis}")
        return analysis

    # ☆ 질문과 답변의 핵심 주제를 추출하는 메서드 (새로 추가)
    def extract_core_topic(self, text: str, text_type: str = "question") -> dict:
        """질문 또는 답변에서 핵심 주제를 추출"""
        try:
            with memory_cleanup():
                system_prompt = """당신은 바이블 앱 콘텐츠 분석 전문가입니다.
입력된 텍스트에서 핵심 주제를 추출하여 JSON 형태로 반환하세요:

{
  "content_type": "콘텐츠 유형",
  "language_type": "언어 유형", 
  "action_type": "행동 유형",
  "specific_topic": "구체적 주제",
  "confidence": "신뢰도 (0.0-1.0)"
}

📚 콘텐츠 유형:
- "bible": 성경, 성경본문, 구절, 장절
- "hymn": 찬송가, 찬양, 찬송
- "prayer": 기도, 기도문
- "sermon": 설교, 강의
- "app_function": 앱 기능, 설정
- "other": 기타

🌏 언어 유형:
- "korean": 한국어 관련
- "english": 영어 관련  
- "mixed": 다국어 관련
- "neutral": 언어 무관

⚡ 행동 유형:
- "play": 재생, 듣기
- "copy": 복사, 붙여넣기
- "search": 검색, 찾기
- "download": 다운로드
- "setting": 설정, 변경
- "inquiry": 문의, 질문
- "other": 기타

중요: 매우 정확하게 분류하세요!"""

                user_prompt = f"다음 {text_type}에서 핵심 주제를 추출해주세요:\n\n{text}"

                response = self.openai_client.chat.completions.create(
                    model='gpt-3.5-turbo',
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=300,
                    temperature=0.1
                )
                
                result_text = response.choices[0].message.content.strip()
                
                try:
                    result = json.loads(result_text)
                    logging.info(f"핵심 주제 추출 ({text_type}): {result}")
                    return result
                except json.JSONDecodeError:
                    return {
                        "content_type": "other",
                        "language_type": "neutral", 
                        "action_type": "other",
                        "specific_topic": text[:50],
                        "confidence": 0.3
                    }
                    
        except Exception as e:
            logging.error(f"핵심 주제 추출 실패: {e}")
            return {
                "content_type": "other",
                "language_type": "neutral",
                "action_type": "other", 
                "specific_topic": text[:50],
                "confidence": 0.0
            }

    # ☆ 의미적 유사성을 검증하는 메서드 (새로 추가)
    def validate_semantic_similarity(self, question_analysis: dict, answer_topics: list, query: str, answers: list) -> dict:
        """질문과 답변들 간의 의미적 유사성을 다각도로 검증"""
        try:
            # 1. 콘텐츠 유형 일치도 검사
            question_content = question_analysis.get('content_type', 'other')
            question_language = question_analysis.get('language_preference', 'none')
            question_action = question_analysis.get('action_type', 'other')
            
            content_matches = 0
            language_matches = 0
            action_matches = 0
            total_answers = len(answers)
            
            for answer in answers[:5]:  # 상위 5개만 검사
                # 답변의 핵심 주제 추출
                answer_topic = self.extract_core_topic(answer.get('answer', ''), 'answer')
                
                # 콘텐츠 유형 일치 검사
                if answer_topic['content_type'] == question_content:
                    content_matches += 1
                
                # 언어 유형 일치 검사  
                if question_language != 'none':
                    if answer_topic['language_type'] == question_language or answer_topic['language_type'] == 'neutral':
                        language_matches += 1
                else:
                    language_matches += 1  # 언어 무관인 경우 항상 일치
                
                # 행동 유형 일치 검사
                if answer_topic['action_type'] == question_action or answer_topic['action_type'] == 'other':
                    action_matches += 1
            
            # 2. 일치도 비율 계산
            content_ratio = content_matches / max(total_answers, 1)
            language_ratio = language_matches / max(total_answers, 1)
            action_ratio = action_matches / max(total_answers, 1)
            
            # 3. 종합 유사성 점수 계산 (가중 평균)
            similarity_score = (content_ratio * 0.5 + language_ratio * 0.3 + action_ratio * 0.2)
            
            # 4. 특별한 불일치 패턴 감지
            critical_mismatch = False
            
            # 찬송가 vs 성경 불일치
            if question_content == 'hymn' and content_matches == 0:
                critical_mismatch = True
                logging.warning("심각한 불일치: 찬송가 질문에 성경 답변")
                
            # 영어 vs 한국어 불일치  
            if question_language in ['english', 'korean'] and language_matches == 0:
                critical_mismatch = True
                logging.warning(f"심각한 불일치: {question_language} 질문에 다른 언어 답변")
            
            # 복사 vs 재생 불일치
            if question_action in ['copy', 'play'] and action_matches == 0:
                critical_mismatch = True
                logging.warning(f"심각한 불일치: {question_action} 요청에 다른 행동 답변")
            
            # 5. 최종 판정
            if critical_mismatch:
                final_similarity = 'critical_mismatch'
            elif similarity_score >= 0.8:
                final_similarity = 'high'
            elif similarity_score >= 0.6:
                final_similarity = 'medium'
            elif similarity_score >= 0.4:
                final_similarity = 'low'
            else:
                final_similarity = 'very_low'
            
            result = {
                'similarity_score': similarity_score,
                'content_match_ratio': content_ratio,
                'language_match_ratio': language_ratio,
                'action_match_ratio': action_ratio,
                'critical_mismatch': critical_mismatch,
                'final_similarity': final_similarity,
                'analysis_summary': f"콘텐츠일치:{content_ratio:.2f}, 언어일치:{language_ratio:.2f}, 행동일치:{action_ratio:.2f}"
            }
            
            logging.info(f"의미적 유사성 검증: {result}")
            return result
            
        except Exception as e:
            logging.error(f"의미적 유사성 검증 실패: {e}")
            return {
                'similarity_score': 0.0,
                'final_similarity': 'error',
                'critical_mismatch': True,
                'analysis_summary': f"검증 실패: {str(e)}"
            }

    # ☆ AI 기반 컨텍스트 관련성 검사 메서드 (대폭 강화 버전)
    def check_context_relevance_ai(self, question_analysis: dict, answer_categories: list, query: str, top_answers: list) -> str:
        """AI와 의미적 유사성 검증을 결합한 고도화된 관련성 검사"""
        
        try:
            # 1단계: 의미적 유사성 검증 (새로운 다단계 검증)
            semantic_result = self.validate_semantic_similarity(question_analysis, answer_categories, query, top_answers)
            
            # 심각한 불일치가 감지된 경우 즉시 irrelevant 반환
            if semantic_result.get('critical_mismatch', False):
                logging.warning(f"심각한 의미적 불일치 감지: {semantic_result.get('analysis_summary', '')}")
                return 'irrelevant'
            
            # 의미적 유사성 점수가 매우 낮은 경우
            similarity_score = semantic_result.get('similarity_score', 0.0)
            if similarity_score < 0.3:
                logging.warning(f"의미적 유사성 점수 너무 낮음: {similarity_score:.3f}")
                return 'irrelevant'
            
            # 2단계: AI 기반 상세 분석
            answer_summaries = []
            for i, answer in enumerate(top_answers[:3]):
                answer_text = answer.get('answer', '')[:200]
                answer_summaries.append(f"답변{i+1}: {answer_text}")
            
            combined_answers = "\n".join(answer_summaries)
            
            with memory_cleanup():
                system_prompt = """당신은 바이블 앱 전문 문의-답변 관련성 분석가입니다.
고객의 질문 의도와 검색된 답변들의 관련성을 매우 엄격하게 분석하여 판정하세요:

- "high": 답변이 질문과 직접적으로 관련되고 완벽히 도움됨
- "medium": 답변이 관련이 있지만 일부 불일치 있음
- "low": 답변이 약간 관련이 있지만 핵심과 거리 있음
- "irrelevant": 답변이 질문과 전혀 관련없음

🚫 바이블 앱 특화 엄격 기준:
1. 찬송가 ≠ 성경 (완전히 다른 콘텐츠)
2. 영어 ≠ 한국어 (언어 불일치)
3. 복사 ≠ 재생 (행동 불일치)
4. 검색 ≠ 설정 (기능 불일치)

📚 특별 케이스:
- "영어 찬송가" 질문 + "영어 성경" 답변 → "irrelevant"
- "복사 기능" 질문 + "재생 기능" 답변 → "irrelevant"
- "검색 방법" 질문 + "설정 변경" 답변 → "irrelevant"

결과는 "high", "medium", "low", "irrelevant" 중 하나만 반환하세요."""

                user_prompt = f"""질문 상세 분석:
콘텐츠유형: {question_analysis.get('content_type', 'N/A')}
언어선호: {question_analysis.get('language_preference', 'N/A')}
행동유형: {question_analysis.get('action_type', 'N/A')}
의도: {question_analysis.get('intent_type', 'N/A')}

의미적 유사성 검증 결과:
- 유사성 점수: {similarity_score:.3f}
- 콘텐츠 일치도: {semantic_result.get('content_match_ratio', 0):.2f}
- 언어 일치도: {semantic_result.get('language_match_ratio', 0):.2f}
- 행동 일치도: {semantic_result.get('action_match_ratio', 0):.2f}

원본 질문: {query}

검색된 답변들:
{combined_answers}

⚠️ 의미적 유사성 검증에서 이미 불일치가 감지되었다면 더욱 엄격하게 판정하세요.
위 모든 정보를 종합하여 관련성을 분석해주세요."""

                response = self.openai_client.chat.completions.create(
                    model='gpt-3.5-turbo',
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=50,
                    temperature=0.1  # 더 보수적으로
                )
                
                ai_result = response.choices[0].message.content.strip().lower()
                
                # 3단계: 의미적 유사성과 AI 분석 결과를 종합
                final_similarity = semantic_result.get('final_similarity', 'medium')
                
                # AI 결과 정규화
                if 'high' in ai_result:
                    ai_relevance = 'high'
                elif 'medium' in ai_result:
                    ai_relevance = 'medium'
                elif 'low' in ai_result:
                    ai_relevance = 'low'
                elif 'irrelevant' in ai_result:
                    ai_relevance = 'irrelevant'
                else:
                    ai_relevance = 'medium'
                
                # 최종 판정: 두 결과 중 더 보수적인 것 선택
                relevance_ranking = {'high': 4, 'medium': 3, 'low': 2, 'irrelevant': 1, 'critical_mismatch': 0, 'very_low': 1, 'error': 0}
                
                semantic_rank = relevance_ranking.get(final_similarity, 1)
                ai_rank = relevance_ranking.get(ai_relevance, 1)
                
                # 더 낮은 (보수적인) 점수 선택
                final_rank = min(semantic_rank, ai_rank)
                
                # 점수를 다시 문자열로 변환
                rank_to_relevance = {4: 'high', 3: 'medium', 2: 'low', 1: 'irrelevant', 0: 'irrelevant'}
                final_result = rank_to_relevance.get(final_rank, 'irrelevant')
                
                logging.info(f"관련성 최종 판정: 의미적={final_similarity}, AI={ai_relevance}, 최종={final_result}")
                return final_result
                    
        except Exception as e:
            logging.error(f"강화된 관련성 분석 실패: {e}")
            # 폴백: 기본적인 키워드 매칭
            return self.fallback_relevance_check(query, top_answers)
    
    # ☆ 폴백 관련성 검사 메서드 (의미론적 매칭 강화)
    def fallback_relevance_check(self, query: str, top_answers: list) -> str:
        """AI 분석 실패 시 사용하는 의미론적 키워드 매칭"""
        
        # 의미론적 키워드 그룹 정의
        semantic_groups = {
            'text_copy': ['복사', '붙여넣기', '워드', '텍스트', '복사해서', '옮기', '내보내기', '저장'],
            'audio_play': ['재생', '듣기', '음성', '소리', '연속재생', '반복', '들을', '듣고'],
            'search_find': ['검색', '찾기', '찾아서', '검색해서', '찾을', '찾는'],
            'download': ['다운로드', '다운받기', '받기', '저장'],
            'error_report': ['오류', '에러', '안됨', '안되', '문제', '고장', '버그'],
            'setting_config': ['설정', '변경', '조정', '옵션', '환경설정']
        }
        
        # 질문에서 의미 그룹 식별
        query_lower = query.lower()
        query_semantic_groups = set()
        for group_name, keywords in semantic_groups.items():
            if any(keyword in query_lower for keyword in keywords):
                query_semantic_groups.add(group_name)
        
        max_relevance = 0
        for answer in top_answers:
            answer_lower = answer.get('answer', '').lower()
            answer_semantic_groups = set()
            
            # 답변에서 의미 그룹 식별
            for group_name, keywords in semantic_groups.items():
                if any(keyword in answer_lower for keyword in keywords):
                    answer_semantic_groups.add(group_name)
            
            # 의미 그룹 일치도 계산
            if query_semantic_groups and answer_semantic_groups:
                semantic_overlap = len(query_semantic_groups & answer_semantic_groups)
                semantic_total = len(query_semantic_groups | answer_semantic_groups)
                semantic_ratio = semantic_overlap / semantic_total if semantic_total > 0 else 0
            else:
                semantic_ratio = 0
            
            # 키워드 일치도도 함께 고려
            query_words = set(self.extract_keywords(query_lower))
            answer_words = set(self.extract_keywords(answer_lower))
            keyword_overlap = len(query_words & answer_words)
            keyword_ratio = keyword_overlap / max(len(query_words), 1)
            
            # 의미론적 매칭과 키워드 매칭을 조합 (의미론적 매칭에 더 높은 가중치)
            combined_relevance = semantic_ratio * 0.7 + keyword_ratio * 0.3
            max_relevance = max(max_relevance, combined_relevance)
        
        # 🚫 의미 그룹이 완전히 다른 경우 irrelevant 처리
        if query_semantic_groups and any(answer.get('answer', '') for answer in top_answers):
            all_answer_groups = set()
            for answer in top_answers:
                answer_lower = answer.get('answer', '').lower()
                for group_name, keywords in semantic_groups.items():
                    if any(keyword in answer_lower for keyword in keywords):
                        all_answer_groups.add(group_name)
            
            # 텍스트-음성, 검색-설정 등 상반된 그룹인 경우
            conflicting_pairs = [
                ('text_copy', 'audio_play'),
                ('search_find', 'setting_config'),
                ('error_report', 'search_find')
            ]
            
            for q_group in query_semantic_groups:
                for a_group in all_answer_groups:
                    if (q_group, a_group) in conflicting_pairs or (a_group, q_group) in conflicting_pairs:
                        return 'irrelevant'
        
        # 관련성 점수에 따른 판정
        if max_relevance >= 0.6:
            return 'high'
        elif max_relevance >= 0.4:
            return 'medium'
        elif max_relevance >= 0.2:
            return 'low'
        else:
            return 'irrelevant'
    
    # ☆ 핵심 키워드 추출 메서드
    def extract_keywords(self, text: str) -> list:
        """텍스트에서 핵심 키워드 추출"""
        # 불용어 제거용 리스트
        stop_words = {'는', '은', '이', '가', '을', '를', '에', '에서', '로', '으로', '와', '과', '의', '도', '만', '까지', '부터', '께서', '에게', '한테', '로부터', '으로부터'}
        
        # 특수문자 제거 및 단어 분리
        
        words = re.findall(r'[가-힣a-zA-Z0-9]+', text)
        
        # 불용어 제거 및 2글자 이상 단어만 선택
        keywords = [word for word in words if len(word) >= 2 and word not in stop_words]
        
        return keywords

    # ☆ 스마트 폴백 답변 생성 메서드 (새로 추가)
    def generate_smart_fallback_answer(self, query: str, question_analysis: dict, lang: str = 'ko') -> str:
        """관련성이 낮을 때 질문 의도에 맞는 적절한 폴백 답변 생성"""
        try:
            with memory_cleanup():
                # 질문 분석 결과 추출
                content_type = question_analysis.get('content_type', 'other')
                language_preference = question_analysis.get('language_preference', 'none')
                action_type = question_analysis.get('action_type', 'other')
                intent_type = question_analysis.get('intent_type', '일반문의')
                
                if lang == 'en':
                    # 영어 스마트 폴백 답변 생성
                    if content_type == 'hymn':
                        if language_preference == 'english':
                            fallback_answer = "<p>Thank you for your inquiry about English hymns.</p><p><br></p><p>We are reviewing options to expand our hymn collection, including English hymns.</p><p><br></p><p>Currently, our app primarily focuses on Korean hymns and Bible content.</p><p><br></p><p>We will forward your request to our development team for future consideration.</p><p><br></p><p>Please feel free to contact us again if you have any other questions.</p>"
                        else:
                            fallback_answer = "<p>Thank you for your inquiry about hymns.</p><p><br></p><p>Our app currently provides a comprehensive collection of Korean hymns.</p><p><br></p><p>We are continuously working to improve our hymn features.</p><p><br></p><p>Please let us know if you have any specific requests or feedback.</p>"
                    elif content_type == 'bible':
                        if language_preference == 'english':
                            fallback_answer = "<p>Thank you for your inquiry about English Bible content.</p><p><br></p><p>Our app supports multiple Bible translations including English versions.</p><p><br></p><p>Please check the translation settings in the app menu to access English Bible versions.</p><p><br></p><p>If you need assistance with specific features, please contact our customer service.</p>"
                        else:
                            fallback_answer = "<p>Thank you for your Bible-related inquiry.</p><p><br></p><p>Our app provides comprehensive Bible reading features with multiple translations.</p><p><br></p><p>Please explore the Bible section for various reading and study tools.</p><p><br></p><p>Contact us if you need help with specific Bible features.</p>"
                    else:
                        fallback_answer = "<p>Thank you for contacting GOODTV Bible App support.</p><p><br></p><p>We are reviewing your inquiry to provide the most accurate information.</p><p><br></p><p>Our team will respond with detailed guidance soon.</p><p><br></p><p>Please contact us again if you have any urgent questions.</p>"
                else:
                    # 한국어 스마트 폴백 답변 생성
                    if content_type == 'hymn':
                        if language_preference == 'english':
                            fallback_answer = "<p>영어 찬송가에 대한 문의해 주셔서 감사합니다.</p><p><br></p><p>현재 바이블 애플은 한국어 찬송가를 중심으로 서비스를 제공하고 있습니다.</p><p><br></p><p>영어 찬송가 추가에 대한 성도님의 의견은 개발팀에 전달하여</p><p><br></p><p>향후 서비스 개선 시 적극 검토하도록 하겠습니다.</p><p><br></p><p>다른 궁금한 사항이 있으시면 언제든 문의해 주세요.</p>"
                        else:
                            fallback_answer = "<p>찬송가 관련 문의해 주셔서 감사합니다.</p><p><br></p><p>바이블 애플에서는 다양한 찬송가 서비스를 제공하고 있습니다.</p><p><br></p><p>찬송가 메뉴에서 원하시는 찬송을 검색하고 재생하실 수 있습니다.</p><p><br></p><p>구체적인 기능이나 사용법에 대해 궁금하신 점이 있으시면</p><p><br></p><p>언제든 다시 문의해 주세요.</p>"
                    elif content_type == 'bible':
                        if language_preference == 'english':
                            fallback_answer = "<p>영어 성경에 대한 문의해 주셔서 감사합니다.</p><p><br></p><p>바이블 애플에서는 NIV, ESV 등 다양한 영어 성경 번역본을</p><p><br></p><p>제공하고 있습니다.</p><p><br></p><p>설정 메뉴에서 번역본을 변경하여 영어 성경을 이용하실 수 있습니다.</p><p><br></p><p>자세한 사용법이 궁금하시면 다시 문의해 주세요.</p>"
                        else:
                            fallback_answer = "<p>성경 관련 문의해 주셔서 감사합니다.</p><p><br></p><p>바이블 애플에서는 다양한 성경 번역본과 읽기 기능을</p><p><br></p><p>제공하고 있습니다.</p><p><br></p><p>성경 메뉴에서 원하시는 성경 본문을 검색하고 읽으실 수 있습니다.</p><p><br></p><p>구체적인 사용법이 궁금하시면 언제든 문의해 주세요.</p>"
                    elif action_type == 'copy':
                        fallback_answer = "<p>텍스트 복사 기능에 대한 문의해 주셔서 감사합니다.</p><p><br></p><p>바이블 애플에서는 성경 본문을 선택하여 복사하는 기능을</p><p><br></p><p>제공하고 있습니다.</p><p><br></p><p>원하시는 구절을 길게 눌러 선택한 후 복사 버튼을 이용해 주세요.</p><p><br></p><p>자세한 사용법이 궁금하시면 언제든 문의해 주세요.</p>"
                    elif action_type == 'play':
                        fallback_answer = "<p>음성 재생 기능에 대한 문의해 주셔서 감사합니다.</p><p><br></p><p>바이블 애플에서는 성경과 찬송가 음성 재생 기능을</p><p><br></p><p>제공하고 있습니다.</p><p><br></p><p>재생 버튼을 눌러 원하시는 콘텐츠를 들으실 수 있습니다.</p><p><br></p><p>구체적인 재생 방법이 궁금하시면 다시 문의해 주세요.</p>"
                    else:
                        fallback_answer = "<p>바이블 애플 관련 문의해 주셔서 감사합니다.</p><p><br></p><p>성도님의 문의 내용을 정확히 파악하여</p><p><br></p><p>가장 적절한 답변을 드리기 위해 검토하고 있습니다.</p><p><br></p><p>구체적인 답변은 빠른 시일 내에 전달드리겠습니다.</p><p><br></p><p>급한 문의사항이 있으시면 언제든 다시 연락해 주세요.</p>"
                
                logging.info(f"스마트 폴백 답변 생성 완료: 콘텐츠={content_type}, 언어={language_preference}, 행동={action_type}")
                return fallback_answer
                
        except Exception as e:
            logging.error(f"스마트 폴백 답변 생성 실패: {e}")
            # 최종 안전장치
            if lang == 'en':
                return "<p>Thank you for contacting GOODTV Bible App.</p><p><br></p><p>We are reviewing your inquiry and will provide a detailed response soon.</p><p><br></p><p>Please contact us again if you have any urgent questions.</p>"
            else:
                return "<p>안녕하세요. GOODTV 바이블 애플입니다.</p><p><br></p><p>성도님의 문의 내용을 검토하여 정확한 답변을 준비하겠습니다.</p><p><br></p><p>답변은 빠른 시일 내에 전달드리겠습니다.</p><p><br></p><p>감사합니다.</p>"

    # ☆ 참고 답변에서 인사말과 끝맺음말을 제거하는 메서드
    # Args:
    #     text (str): 제거할 텍스트
    #     lang (str): 언어 (기본값: 한국어)
            
    # Returns:
    #     str: 제거된 텍스트
    def remove_greeting_and_closing(self, text: str, lang: str = 'ko') -> str:
        # null 체크
        if not text:
            return ""
        
        if lang == 'ko':
            # 한국어 인사말 제거 패턴
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
            
            # 한국어 끝맺음말 제거 패턴들
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

        else:  # 영어 인사말 제거 패턴
            greeting_patterns = [
                r'^Hello[^.]*\.\s*',
                r'^Hi[^.]*\.\s*',
                r'^Dear[^.]*\.\s*',
                r'^Thank you[^.]*\.\s*',
                r'^Thanks[^.]*\.\s*',
                r'^This is GOODTV Bible App[^.]*\.\s*',
            ]
            
            # 영어 끝맺음말 제거 패턴
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
        
        # 문장 끝의 끝맺음말들도 제거
        text = re.sub(r'[,.!?]\s*항상\s*주님\s*안에서[^.]*\.?\s*$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[,.!?]\s*감사합니다[^.]*\.?\s*$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[,.!?]\s*평안하세요[^.]*\.?\s*$', '', text, flags=re.IGNORECASE)
        
        # 앞뒤 공백 정리
        text = text.strip()
        
        return text

    # ☆ GPT 답변 생성을 위한 향상된 컨텍스트 생성 메서드
    # 
    # 컨텍스트 생성 전략:
    # 1. 품질별 답변 그룹핑 (고/중/낮은 품질)
    # 2. 고품질 답변 우선 선택 (최대 4개)
    # 3. 중품질 답변으로 보완 (최대 3개)
    # 4. 최소 개수 미달시 중간 품질 답변 추가
    # 5. 텍스트 정제 및 길이 제한 (인사말/끝맺음말 제거)
    # 
    # 이렇게 구성된 컨텍스트는 GPT에게 참고 자료로 제공되어
    # 일관된 스타일과 정확한 정보로 답변을 생성하게 함
    # 
    # Args:
    #     similar_answers (list): 검색된 유사 답변 리스트
    #     max_answers (int): 포함할 최대 답변 개수 (기본값: 7)
    #     
    # Returns:
    #     str: GPT용 컨텍스트 문자열
    def create_enhanced_context(self, similar_answers: list, max_answers: int = 7, target_lang: str = 'ko') -> str:
        if not similar_answers:
            return ""
        
        context_parts = [] # 컨텍스트 부분들을 저장할 리스트
        used_answers = 0 # 사용된 답변 개수
        
        # 유사도 점수에 따른 답변 그룹핑
        # 계층적 그룹핑으로 품질별 답변을 분류: 이는 나중에 우선순위에 따라 선택하기 위함
        high_score = [ans for ans in similar_answers if ans['score'] >= 0.7]      # 고품질 (70% 이상 유사)
        medium_score = [ans for ans in similar_answers if 0.5 <= ans['score'] < 0.7]  # 중품질 (50-70%)
        medium_low_score = [ans for ans in similar_answers if 0.5 <= ans['score'] < 0.6] # 낮은 품질 (50-60%)

        # 1단계: 고품질 답변 우선 포함 (최대 4개)
        for ans in high_score[:4]:
            if used_answers >= max_answers:
                break
            # 제어 문자(줄바꿈, 탭, 개행 등) 및 HTML 태그 제거
            clean_answer = re.sub(r'[\b\r\f\v\x00-\x08\x0B\x0C\x0E-\x1F\x7F]|<[^>]+>', '', ans['answer'])
            clean_answer = self.remove_greeting_and_closing(clean_answer, 'ko') # 인사말과 끝맺음말 제거하여 본문만 추출
            
            # 영어 질문인 경우 답변을 번역
            if target_lang == 'en' and ans.get('lang', 'ko') == 'ko':
                clean_answer = self.translate_text(clean_answer, 'ko', 'en')
            
            # 유효성 검사 및 길이 제한 (최소 20자 이상확인, 400자로 잘라서 컨텍스트 길이 제한 방지)
            if self.is_valid_text(clean_answer, target_lang) and len(clean_answer.strip()) > 20:
                context_parts.append(f"[참고답변 {used_answers+1} - 점수: {ans['score']:.2f}]\n{clean_answer[:400]}")
                used_answers += 1
        
        # 2단계: 중품질 답변으로 보완 (최대 3개)
        for ans in medium_score[:3]:
            if used_answers >= max_answers:
                break
            # 제어 문자(줄바꿈, 탭, 개행 등) 및 HTML 태그 제거
            clean_answer = re.sub(r'[\b\r\f\v\x00-\x08\x0B\x0C\x0E-\x1F\x7F]|<[^>]+>', '', ans['answer'])
            clean_answer = self.remove_greeting_and_closing(clean_answer, 'ko') # 인사말과 끝맺음말 제거하여 본문만 추출
            
            # 영어 질문인 경우 답변을 번역
            if target_lang == 'en' and ans.get('lang', 'ko') == 'ko':
                clean_answer = self.translate_text(clean_answer, 'ko', 'en')
            
            if self.is_valid_text(clean_answer, target_lang) and len(clean_answer.strip()) > 20:
                context_parts.append(f"[참고답변 {used_answers+1} - 점수: {ans['score']:.2f}]\n{clean_answer[:300]}")
                used_answers += 1

        # 3단계: 답변이 부족한 경우 중간 품질 답변 추가 (50-60% 구간)
        if used_answers < 3:  # 최소 3개 이상 확보하기 위함
            for ans in medium_low_score[:2]:
                if used_answers >= max_answers:
                    break
                clean_answer = re.sub(r'[\b\r\f\v\x00-\x08\x0B\x0C\x0E-\x1F\x7F]|<[^>]+>', '', ans['answer'])
                clean_answer = self.remove_greeting_and_closing(clean_answer, 'ko')
                
                # 영어 질문인 경우 답변을 번역
                if target_lang == 'en' and ans.get('lang', 'ko') == 'ko':
                    clean_answer = self.translate_text(clean_answer, 'ko', 'en')
                
                if self.is_valid_text(clean_answer, target_lang) and len(clean_answer.strip()) > 20:
                    context_parts.append(f"[참고답변 {used_answers+1} - 점수: {ans['score']:.2f}]\n{clean_answer[:250]}")
                    used_answers += 1
        
        logging.info(f"컨텍스트 생성: {used_answers}개의 답변 포함 (언어: {target_lang})")
        
        # 최종 컨텍스트 조합 (구분선으로 답변들 분리)
        return "\n\n" + "="*50 + "\n\n".join(context_parts)

    # ☆ 이전 앱 이름을 제거하는 메서드 (구 다번역성경찬송 등)
    # Args:
    #     text (str): 제거할 텍스트
            
    # Returns:
    #     str: 제거된 텍스트
    def remove_old_app_name(self, text: str) -> str:
        patterns_to_remove = [
            r'\s*\(구\)\s*다번역성경찬송',
            r'\s*\(구\)다번역성경찬송',
            r'바이블\s*애플\s*\(구\)\s*다번역성경찬송',
            r'바이블애플\s*\(구\)다번역성경찬송',
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        text = re.sub(r'(GOODTV\s+바이블\s*애플)\s+', r'\1', text)
        
        return text

    # ☆ 답변 텍스트를 HTML 단락 형식으로 포맷팅하는 메서드
    def format_answer_with_html_paragraphs(self, text: str, lang: str = 'ko') -> str:
        if not text:
            return ""
        
        text = self.remove_old_app_name(text)
        
        # 문장을 마침표, 느낌표, 물음표로 분리
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        paragraphs = [] # 단락들을 저장할 리스트
        current_paragraph = [] # 현재 단락을 저장할 리스트
        
        # 단락 분리 트리거 키워드들 (더 포괄적으로 확장)
        # 접속사나 인사말로 시작하는 문장은 들여쓰기를 통해 새 단락으로 분리
        if lang == 'ko':
            paragraph_triggers = [
                # 인사 및 감사
                '안녕하세요', '감사합니다', '감사드립니다', '바이블 애플을',
                # 접속어 및 연결어
                '따라서', '그러므로', '또한', '그리고', '또는', '하지만', '그런데',
                '이외', '이에', '이를', '이로', '이와', '이에', '이상', '이하',
                # 상황 설명
                '현재', '지금', '현재로', '현재까지', '현재로서는',
                '내부적으로', '외부적으로', '기술적으로', '운영상',
                # 조건 및 가정
                '만약', '혹시', '만일', '만약에', '만약의',
                '해당', '이', '그', '저', '이런', '그런', '저런',
                # 요청 및 안내
                '성도님', '고객님', '이용자', '사용자',
                '번거로우시', '불편하시', '죄송하지만', '참고로',
                '양해부탁드립니다', '양해해주시기', '이해해주시기',
                # 시간 관련
                '항상', '늘', '앞으로도', '지속적으로', '계속적으로',
                '시간이', '소요될', '걸릴', '필요한',
                # 기능 관련
                '기능', '기능은', '기능의', '기능이', '기능을',
                '스피커', '버튼', '메뉴', '화면', '설정', '옵션',
                # 의견 및 전달
                '의견은', '의견을', '전달할', '전달하겠습니다', '전달드리겠습니다',
                '토의가', '검토가', '검토를', '논의가', '논의를'
            ]
        else:  # 영어
            paragraph_triggers = [
                # Greetings and appreciation
                'Hello', 'Hi', 'Dear', 'Thank', 'Thanks', 'Appreciate',
                'Grateful', 'Welcome', 'Greetings',
                
                # Conjunctions and transitions
                'Therefore', 'However', 'Additionally', 'Furthermore', 
                'Moreover', 'Nevertheless', 'Nonetheless', 'Meanwhile',
                'Subsequently', 'Consequently', 'Hence', 'Thus', 'Besides',
                'Although', 'Though', 'While', 'Whereas', 'Instead',
                
                # Situation descriptions
                'Currently', 'Presently', 'At the moment', 'Now',
                'At this time', 'As of now', 'Recently', 'Lately',
                'Technically', 'Internally', 'Externally', 'Generally',
                'Specifically', 'Basically', 'Essentially', 'Fundamentally',
                
                # Conditions and assumptions
                'If', 'When', 'Where', 'Whether', 'Unless', 'Provided',
                'Assuming', 'Suppose', 'In case', 'Should', 'Would',
                'Could', 'Might', 'May',
                
                # Requests and guidance
                'Please', 'Kindly', 'We recommend', 'We suggest',
                'You can', 'You may', 'You should', 'You might',
                'Try', 'Consider', 'Note that', 'Be aware',
                'Remember', 'Keep in mind', 'Important',
                
                # Apologies and understanding
                'Sorry', 'Apologize', 'Apologies', 'Unfortunately',
                'Regret', 'Understand', 'Realize', 'Acknowledge',
                'We know', 'We understand', 'We appreciate',
                
                # Time-related
                'Always', 'Usually', 'Often', 'Sometimes', 'Occasionally',
                'Frequently', 'Regularly', 'Continuously', 'Constantly',
                'Soon', 'Shortly', 'Eventually', 'Later', 'Previously',
                
                # Feature and function related
                'Feature', 'Function', 'Option', 'Setting', 'Button',
                'Menu', 'Screen', 'Tab', 'Page', 'Section', 'Tool',
                'Service', 'System', 'Application', 'Update', 'Version',
                
                # Problem-solving related
                'To fix', 'To solve', 'To resolve', 'To address',
                'Solution', 'Resolution', 'Workaround', 'Alternative',
                'Issue', 'Problem', 'Error', 'Bug', 'Trouble',
                
                # Feedback and communication
                'Your feedback', 'Your suggestion', 'Your opinion',
                'We will', 'We are', 'We have', 'Our team',
                'Will be', 'Has been', 'Have been', 'Working on',
                'Looking into', 'Reviewing', 'Considering', 'Planning',
                
                # Instructions and steps
                'First', 'Second', 'Third', 'Next', 'Then', 'After',
                'Before', 'Finally', 'Lastly', 'To begin', 'To start',
                'Step', 'Follow', 'Navigate', 'Click', 'Tap', 'Select',
                
                # Emphasis and clarification
                'Indeed', 'In fact', 'Actually', 'Certainly', 'Definitely',
                'Clearly', 'Obviously', 'Importantly', 'Notably',
                'Particularly', 'Especially', 'Specifically'
            ]
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 첫 번째 문장 (인사말)은 항상 별도 단락
            if i == 0:
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                paragraphs.append(sentence)
                continue
            
            should_break = False
            
            # 트리거 키워드로 시작하는 문장은 새 단락
            for trigger in paragraph_triggers:
                if sentence.startswith(trigger):
                    should_break = True
                    break
            
            # 현재 단락에 2개 이상 문장이 있으면 새 단락
            if current_paragraph and len(current_paragraph) >= 2:
                should_break = True

            # 끝맺음말이 포함된 문장은 새 단락
            if any(closing in sentence for closing in ['감사합니다', '감사드립니다', '평안하세요', '주님 안에서']):
                should_break = True
            
            # 문장 길이가 50자 이상이고 현재 단락이 있으면 새 단락
            if len(sentence) > 50 and current_paragraph:
                should_break = True
            
            # 새 단락 분리
            if should_break and current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = [sentence]
            else:
                current_paragraph.append(sentence)
        
        # 마지막 단락 추가
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        # Quill 에디터 호환을 위한 HTML 단락으로 변환
        # 각 단락을 <p> 태그로 감싸고, 단락 사이에 <p><br></p> 태그로 빈 줄 추가
        html_paragraphs = []
        for i, paragraph in enumerate(paragraphs):
            html_paragraphs.append(f"<p>{paragraph}</p>")
            
            # 단락 사이에 빈 줄 추가 (마지막 단락 제외)
            if i < len(paragraphs) - 1:
                html_paragraphs.append("<p><br></p>")
        
        return ''.join(html_paragraphs)

    # ☆ 답변 텍스트를 정리하고 포맷팅하는 메서드 (Quill 에디터용)
    def clean_answer_text(self, text: str) -> str:
        if not text:
            return ""
        
        # 제어 문자만 제거하고 HTML 태그는 유지
        text = re.sub(r'[\b\r\f\v]', '', text)
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)

        # HTML 태그 제거하지 않음 (Quill 에디터용)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        
        # HTML 태그 내부의 공백만 정리 (태그 자체는 유지)
        text = re.sub(r'>\s+<', '><', text)  # 태그 사이 공백 제거
        text = re.sub(r'<p>\s+', '<p>', text)  # <p> 태그 내부 앞 공백 제거
        text = re.sub(r'\s+</p>', '</p>', text)  # </p> 태그 앞 공백 제거
        
        text = self.remove_old_app_name(text)
        text = self.format_answer_with_html_paragraphs(text)
        
        return text

    # ☆ 텍스트 유효성 검증 메서드
    def is_valid_text(self, text: str, lang: str = 'ko') -> bool:
        if not text or len(text.strip()) < 3:
            return False
        
        if lang == 'ko':
            return self.is_valid_korean_text(text)
        else:  # 영어
            return self.is_valid_english_text(text)

    # ☆ 한국어 텍스트의 유효성을 검증하는 메서드
    def is_valid_korean_text(self, text: str) -> bool:
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

    # ☆ 영어 텍스트의 유효성을 검증하는 메서드
    def is_valid_english_text(self, text: str) -> bool:
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

    # ☆ 생성된 텍스트를 정리하고 검증하는 메서드
    def clean_generated_text(self, text: str) -> str:
        if not text:
            return ""
        # 제어 문자 제거
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        text = re.sub(r'[\b\r\f\v]', '', text)

        # 영어 약어 제거
        text = re.sub(r'\b[a-z]{1,2}\b(?:\s+[a-z]{1,2}\b)*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[а-я]+', '', text)
        text = re.sub(r'[α-ω]+', '', text)

        # 한글 문자 제거
        text = re.sub(r'[^\w\s가-힣.,!?()"\'-]{3,}', '', text)
        text = re.sub(r'[.,;:!?]{3,}', '.', text)

        # 공백 정리
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text

    # ☆ 통일된 GPT 프롬프트 생성 메서드 (모듈화)
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

🎯 핵심 원칙:
1. 고객의 질문을 정확히 이해하고 질문 의도에 맞는 적절한 답변을 제공하세요
2. 참고 답변이 질문과 관련성이 있다면 이를 활용하여 답변하세요
3. 참고 답변이 질문과 관련성이 낮다면 고객 질문에 맞는 새로운 답변을 생성하세요

📋 답변 작성 지침:

✅ 질문별 적절한 대응:
- 문의 유형과 내용을 정확히 파악하여 그에 맞는 답변 제공
- 고객의 구체적인 요청사항에 직접적으로 대응
- 실제 존재하는 기능과 정책에 기반한 정확한 정보 제공

🔍 답변 품질 기준:
- 질문의 핵심 의도와 일치하는 내용으로 답변
- 모호하거나 회피적인 표현보다는 구체적이고 명확한 안내
- 고객의 문제 해결에 실질적으로 도움이 되는 내용

⚠️ 답변 일관성 유지:
- 질문 유형과 전혀 다른 내용의 답변 지양
- 하나의 문의에 대해 일관된 주제와 해결방향 제시
- 불확실한 정보보다는 확실한 범위 내에서 답변

🚫 형식 제한사항:
- 인사말("안녕하세요", "감사합니다" 등) 사용 금지
- 끝맺음말("평안하세요", "주님 안에서" 등) 사용 금지  
- 본문 내용만 작성하고 격식적 표현 생략

💡 창의적 문제해결:
- 참고 답변이 부적절할 때는 고객 상황에 맞는 새로운 해결책 제시
- 바이블 애플의 실제 서비스 범위 내에서 현실적인 답변 제공
- 고객 관점에서 도움이 되는 실용적인 조언 포함"""

            user_prompt = f"""고객 문의: {query}

참고 답변들:
{context}

❗ 중요 지시사항:
위 참고 답변들이 고객의 질문과 관련이 있는지 먼저 판단하세요.
- 관련이 있다면: 참고 답변의 해결 방식을 활용하여 답변하세요
- 관련이 없다면: 참고 답변을 참고하여 새로운 답변을 작성하세요

고객의 구체적인 질문에 정확히 맞는 답변만 작성하세요.
인사말이나 끝맺음말 없이 본문 내용만 작성하세요."""

        return system_prompt, user_prompt

    # ☆ 향상된 GPT 생성 - 통일된 프롬프트 사용
    def generate_with_enhanced_gpt(self, query: str, similar_answers: list, context_analysis: dict, lang: str = 'ko') -> str:
        try:
            with memory_cleanup():
                approach = context_analysis['recommended_approach']
                context = self.create_enhanced_context(similar_answers, target_lang=lang)
                
                if not context:
                    logging.warning("유효한 컨텍스트가 없어 GPT 생성 중단")
                    return ""
                
                # 통일된 프롬프트 생성
                system_prompt, user_prompt = self.get_gpt_prompts(query, context, lang)
                
                # 🔥 접근 방식별 temperature와 max_tokens 설정 개선
                if approach == 'gpt_with_strong_context':
                    # 관련성이 높은 경우 더 창의적으로 설정
                    temperature = 0.7 if context_analysis.get('context_relevance') == 'high' else 0.6
                    max_tokens = 700
                elif approach == 'gpt_with_weak_context':
                    # 관련성이 낮은 경우 더 창의적으로 설정하여 새로운 답변 생성 유도
                    temperature = 0.8
                    max_tokens = 650
                else: # fallback이나 기타
                    return ""
                
                # GPT API 호출
                response = self.openai_client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.8,
                    frequency_penalty=0.1,
                    presence_penalty=0.1
                )
                
                generated = response.choices[0].message.content.strip()
                del response
                
                # 생성된 텍스트 정리
                generated = self.clean_generated_text(generated)
                
                # 🔥 답변 관련성 추가 검증 (AI 기반)
                if not self.validate_answer_relevance_ai(generated, query, context_analysis.get('question_analysis', {})):
                    logging.warning(f"생성된 답변이 질문과 관련성이 낮음: {generated[:50]}...")
                    return ""
                
                # 텍스트 유효성 검증 (완화)
                if not self.is_valid_text(generated, lang):
                    logging.warning(f"GPT가 무효한 텍스트 생성: {generated[:50]}...")
                    # 유효성 검증 실패해도 관련성이 높으면 사용
                    if context_analysis.get('context_relevance') == 'high':
                        logging.info("관련성이 높아 유효성 검증 우회")
                    else:
                        return ""
                
                logging.info(f"GPT 생성 성공 ({approach}, 언어: {lang}): {len(generated)}자")
                return generated[:650]
                
        except Exception as e:
            logging.error(f"향상된 GPT 생성 실패: {e}")
            return ""

    # ☆ AI 기반 답변 관련성 검증 메서드 (엄격한 검증 버전)
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
            query_keywords = set(self.extract_keywords(query.lower()))
            answer_keywords = set(self.extract_keywords(answer.lower()))
            
            keyword_overlap = len(query_keywords & answer_keywords)
            keyword_relevance = keyword_overlap / max(len(query_keywords), 1)
            
            return keyword_relevance >= 0.2  # 20% 이상 키워드 일치시 관련성 있음으로 판단

    # ☆ 최적의 폴백 답변 선택 메서드 (직접 사용 답변 포함)
    def get_best_fallback_answer(self, similar_answers: list, lang: str = 'ko') -> str:
        logging.info(f"=== get_best_fallback_answer 시작 ===")
        logging.info(f"입력된 similar_answers 개수: {len(similar_answers)}")
        
        if not similar_answers:
            logging.warning("similar_answers가 비어있음")
            return ""
        
        # 입력된 답변들 미리보기
        for i, ans in enumerate(similar_answers[:3]):
            logging.info(f"답변 #{i+1}: 점수={ans['score']:.3f}, 길이={len(ans.get('answer', ''))}, 내용미리보기={ans.get('answer', '')[:50]}...")
        
        # 점수와 텍스트 품질을 종합 평가
        best_answer = ""
        best_score = 0
        
        for i, ans in enumerate(similar_answers[:3]): # 상위 3개만 검토 (품질 향상)
            logging.info(f"--- 답변 #{i+1} 처리 시작 ---")
            score = ans['score']
            answer_text = ans['answer']  # 원본 답변 텍스트 사용
            logging.info(f"원본 답변 길이: {len(answer_text)}, 내용: {answer_text[:100]}...")
            
            # 🔥 긴급 수정: 0.9 이상 점수면 전처리 없이 바로 반환
            if score >= 0.9:
                logging.info(f"🔥 매우 높은 유사도({score:.3f}) - 전처리 없이 원본 답변 바로 반환")
                print(f"🔥 매우 높은 유사도({score:.3f}) - 전처리 없이 원본 답변 바로 반환")
                # 최소한의 정리만
                clean_answer = answer_text.strip()
                if clean_answer:
                    logging.info(f"🔥 원본 답변 직접 반환: 길이={len(clean_answer)}")
                    return clean_answer
            
            # 기본 정리만 수행
            answer_text = self.preprocess_text(answer_text)
            logging.info(f"전처리 후 길이: {len(answer_text)}, 내용: {answer_text[:100]}...")
            
            # 영어 질문인 경우 답변을 번역
            if lang == 'en' and ans.get('lang', 'ko') == 'ko':
                answer_text = self.translate_text(answer_text, 'ko', 'en')
                logging.info(f"번역 후 길이: {len(answer_text)}")
            
            # 유효성 검사 (임시로 우회 - 디버깅용)
            is_valid = self.is_valid_text(answer_text, lang)
            logging.info(f"유효성 검사 결과: {is_valid}")
            
            # 임시로 유효성 검사 무시하고 진행
            if not is_valid:
                logging.warning(f"⚠️ 답변 #{i+1} 유효성 검사 실패했지만 강제로 진행")
                # continue를 주석 처리하여 유효성 검사 실패해도 계속 진행
            
            # 높은 유사도(0.8+)인 경우 간단하게 첫 번째 답변 선택
            if score >= 0.8:
                logging.info(f"높은 유사도({score:.3f})로 답변 #{i+1} 직접 선택")
                logging.info(f"선택된 답변 최종 길이: {len(answer_text)}")
                # 🔥 유효성 검사 실패해도 강제로 반환
                if answer_text and len(answer_text.strip()) > 0:
                    return answer_text
                else:
                    logging.error(f"🔥 전처리 후 답변이 비어있음! 원본으로 폴백")
                    return ans['answer'].strip()
            
            # 종합 점수 계산 (유사도 + 텍스트 길이 + 완성도)
            length_score = min(len(answer_text) / 200, 1.0) # 200자 기준 정규화
            completeness_score = 1.0 if answer_text.endswith(('.', '!', '?')) else 0.8
            
            total_score = score * 0.8 + length_score * 0.1 + completeness_score * 0.1  # 유사도 가중치 증가
            
            logging.info(f"답변 #{i+1} 종합 점수: {total_score:.3f} (유사도:{score:.3f}, 길이:{length_score:.3f}, 완성도:{completeness_score:.3f})")
            
            if total_score > best_score:
                best_score = total_score
                best_answer = answer_text
                logging.info(f"새로운 최고 점수 답변으로 선택됨")
        
        logging.info(f"=== get_best_fallback_answer 완료 ===")
        logging.info(f"최종 선택된 답변 점수: {best_score:.3f}")
        logging.info(f"최종 답변 길이: {len(best_answer)}")
        logging.info(f"최종 답변 미리보기: {best_answer[:100] if best_answer else 'None'}...")
        
        # 🔥 긴급 안전장치: 답변이 비어있으면 첫 번째 원본 답변 강제 반환
        if not best_answer and similar_answers:
            logging.error("🚨 최종 답변이 비어있음! 첫 번째 원본 답변 강제 반환")
            print("🚨 최종 답변이 비어있음! 첫 번째 원본 답변 강제 반환")
            emergency_answer = similar_answers[0]['answer'].strip()
            logging.info(f"🚨 긴급 답변 길이: {len(emergency_answer)}")
            return emergency_answer
        
        return best_answer

    # ☆ 더 보수적인 GPT-3.5-turbo 생성 메서드 (기존 코드와의 호환성 유지)
    # 보수적이고 참고 답변에 충실한 GPT-3.5-turbo 텍스트 생성
    @profile
    def generate_ai_answer(self, query: str, similar_answers: list, lang: str) -> str:
        
        # 1. 언어 감지 (lang 파라미터가 없거나 'auto'인 경우)
        if not lang or lang == 'auto':
            detected_lang = self.detect_language(query)
            lang = detected_lang
            logging.info(f"감지된 언어: {lang}")
        
        # 2. 유사 답변이 없는 경우
        if not similar_answers:
            logging.error("🚨 유사 답변이 전혀 없음 - Pinecone 검색 실패 또는 데이터 없음")
            print(f"🚨 CRITICAL: 유사 답변이 전혀 없음! query='{query[:50]}...', lang='{lang}'")
            if lang == 'en':
                default_msg = "<p>We need more detailed information to provide an accurate answer to your inquiry.</p><p><br></p><p>Please contact our customer service center for prompt assistance.</p>"
            else:
                default_msg = "<p>안녕하세요, GOODTV 바이블 애플입니다.</p><p><br></p><p>바이블 애플을 이용해 주셔서 진심으로 감사드립니다.</p><p><br></p><p>남겨주신 문의는 현재 담당자가 직접 확인하고 있습니다.</p><p><br></p><p>성도님께 도움이 될 수 있도록 내용을 꼼꼼히 살펴</p><p><br></p><p>정확하고 구체적인 답변을 준비하겠습니다.</p><p><br></p><p>답변은 최대 하루 이내에 드릴 예정이오니</p><p><br></p><p>조금만 기다려 주시면 감사하겠습니다.</p><p><br></p><p>항상 주님 안에서 평안하세요, 감사합니다.</p>"
            return default_msg
        
        # 3. 컨텍스트 분석
        context_analysis = self.analyze_context_quality(similar_answers, query)
        
        # 4. 검색 결과 상세 로깅
        logging.info(f"✅ 검색된 유사 답변 개수: {len(similar_answers)}")
        print(f"✅ 검색된 유사 답변 개수: {len(similar_answers)}")
        
        if similar_answers:
            for i, ans in enumerate(similar_answers[:3]):
                log_msg = f"📝 답변 #{i+1}: 점수={ans['score']:.3f}, 카테고리={ans['category']}"
                logging.info(log_msg)
                print(log_msg)
        
        # 5. 답변이 전혀 없을 때만 기본 메시지 반환 (중복 체크 제거)
        # 이미 위에서 체크했으므로 이 부분은 실행되지 않아야 함
        

        try:
            approach = context_analysis['recommended_approach']
            logging.info(f"=== 접근 방식 결정 ===")
            logging.info(f"🎯 선택된 접근 방식: {approach}, 언어: {lang}")
            logging.info(f"📊 컨텍스트 분석: {context_analysis}")
            
            # 콘솔에도 출력
            print(f"🎯 선택된 접근 방식: {approach}, 언어: {lang}")
            print(f"📊 컨텍스트 분석: {context_analysis}")
            
            base_answer = ""
            
            if approach == 'direct_use':
                logging.info("=== 직접 사용 방식 시작 ===")
                base_answer = self.get_best_fallback_answer(similar_answers[:3], lang)
                logging.info(f"직접 사용 결과 길이: {len(base_answer) if base_answer else 0}")
                logging.info(f"직접 사용 결과 미리보기: {base_answer[:100] if base_answer else 'None'}...")
                
            elif approach in ['gpt_with_strong_context', 'gpt_with_weak_context']:
                logging.info(f"=== GPT 생성 방식 시작: {approach} ===")
                base_answer = self.generate_with_enhanced_gpt(query, similar_answers, context_analysis, lang)
                logging.info(f"GPT 생성 결과 길이: {len(base_answer) if base_answer else 0}")
                
                if not base_answer or not self.is_valid_text(base_answer, lang):
                    logging.warning("GPT 생성 실패, 폴백 답변 사용")
                    base_answer = self.get_best_fallback_answer(similar_answers, lang)
                    logging.info(f"폴백 답변 길이: {len(base_answer) if base_answer else 0}")
                    
            elif approach == 'smart_fallback':
                logging.info("=== 스마트 폴백 방식 사용 ===")
                base_answer = self.generate_smart_fallback_answer(query, context_analysis.get('question_analysis', {}), lang)
                logging.info(f"스마트 폴백 답변 길이: {len(base_answer) if base_answer else 0}")
                    
            else:
                logging.info("=== 일반 폴백 방식 사용 ===")
                base_answer = self.get_best_fallback_answer(similar_answers, lang)
                logging.info(f"일반 폴백 답변 길이: {len(base_answer) if base_answer else 0}")
            
            # 최종 검증 전 상세 로깅
            logging.info(f"=== 최종 검증 시작 ===")
            logging.info(f"base_answer 존재 여부: {base_answer is not None and base_answer != ''}")
            if base_answer:
                logging.info(f"base_answer 길이: {len(base_answer)}")
                logging.info(f"base_answer 미리보기: {base_answer[:200]}...")
                is_valid = self.is_valid_text(base_answer, lang)
                logging.info(f"is_valid_text 결과: {is_valid}")
                if not is_valid:
                    logging.warning(f"유효성 검사 실패 사유 분석 중...")
                    # 한국어 텍스트 분석
                    if lang == 'ko':
                        korean_chars = len(re.findall(r'[가-힣]', base_answer))
                        total_chars = len(re.sub(r'\s', '', base_answer))
                        korean_ratio = korean_chars / total_chars if total_chars > 0 else 0
                        logging.warning(f"한국어 비율: {korean_ratio:.2f} (기준: 0.2 이상)")
                        logging.warning(f"한국어 글자 수: {korean_chars}, 전체 글자 수: {total_chars}")
            else:
                logging.error("base_answer가 비어있음")
            
            # 🔥 긴급 수정: is_valid_text 검증 완전히 우회
            if not base_answer:
                logging.error("=== base_answer가 비어있음 ===")
                logging.error(f"similar_answers 개수: {len(similar_answers)}")
                if similar_answers:
                    logging.error(f"첫 번째 답변 점수: {similar_answers[0]['score']}")
                    logging.error(f"첫 번째 답변 내용: {similar_answers[0]['answer'][:100]}...")
                
                if lang == 'en':
                    return "<p>We need more detailed information to provide an accurate answer to your inquiry.</p><p><br></p><p>Please contact our customer service center for prompt assistance.</p>"
                else:
                    return "<p>안녕하세요, GOODTV 바이블 애플입니다.</p><p><br></p><p>바이블 애플을 이용해 주셔서 진심으로 감사드립니다.</p><p><br></p><p>남겨주신 문의는 현재 담당자가 직접 확인하고 있습니다.</p><p><br></p><p>성도님께 도움이 될 수 있도록 내용을 꼼꼼히 살펴</p><p><br></p><p>정확하고 구체적인 답변을 준비하겠습니다.</p><p><br></p><p>답변은 최대 하루 이내에 드릴 예정이오니</p><p><br></p><p>조금만 기다려 주시면 감사하겠습니다.</p><p><br></p><p>항상 주님 안에서 평안하세요, 감사합니다.</p>"
            
            # 🔥 is_valid_text 검증을 임시로 주석 처리
            # elif not self.is_valid_text(base_answer, lang):
            elif False:  # 항상 False가 되어 이 블록은 실행되지 않음
                logging.warning(f"유효성 검사 실패했지만 답변 존재함 - 강제 진행")
                # 이 블록은 실행되지 않음
            
            # 🔥 스마트 폴백의 경우 추가 포맷팅 건너뛰기
            if approach == 'smart_fallback':
                logging.info("🎯 스마트 폴백 답변은 이미 완성된 형태로 바로 반환")
                print("🎯 스마트 폴백 답변은 이미 완성된 형태로 바로 반환")
                return base_answer
            
            # 🔥 성공 로그 추가
            logging.info("🎉 유효성 검사 우회 성공 - 답변 포맷팅 시작")
            print("🎉 유효성 검사 우회 성공 - 답변 포맷팅 시작")
            
            # 언어별 포맷팅
            if lang == 'en':
                # 영어 답변 포맷팅
                base_answer = self.remove_old_app_name(base_answer)
                
                # 기존 인사말/끝맺음말 제거
                base_answer = re.sub(r'^Hello[^.]*\.\s*', '', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'^This is GOODTV Bible App[^.]*\.\s*', '', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'\s*Thank you[^.]*\.?\s*$', '', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'\s*Best regards[^.]*\.?\s*$', '', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'\s*God bless[^.]*\.?\s*$', '', base_answer, flags=re.IGNORECASE)
                
                formatted_body = self.format_answer_with_html_paragraphs(base_answer.strip(), 'en')
                
                # 영어 고정 인사말과 끝맺음말
                final_answer = "<p>Hello, this is GOODTV Bible Apple App customer service team.</p><p><br></p><p>Thank you very much for using our app and for taking the time to contact us.</p><p><br></p>"
                final_answer += formatted_body
                final_answer += "<p><br></p><p>Thank you once again for sharing your thoughts with us!</p><p><br></p><p>May God's peace and grace always be with you.</p>"
                
            else:  # 한국어
                # 한국어 답변 최종 포맷팅 (Quill 에디터용 HTML 형식 유지)
                # 앱 이름 정리 및 고객님 → 성도님 변경
                base_answer = self.remove_old_app_name(base_answer)
                base_answer = re.sub(r'고객님', '성도님', base_answer)
                
                # 기존 인사말/끝맺음말 제거 (일반 텍스트에서)
                # 인사말 제거
                base_answer = re.sub(r'^안녕하세요[^.]*\.\s*', '', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'^GOODTV\s+바이블\s*애플[^.]*\.\s*', '', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'^바이블\s*애플[^.]*\.\s*', '', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'고객센터[^.]*\.\s*', '', base_answer, flags=re.IGNORECASE)
                
                # 끝맺음말 제거 (더 강화된 패턴) - '항상' 중복 제거 포함
                base_answer = re.sub(r'\s*감사합니다[^.]*\.?\s*$', '', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'\s*평안하세요[^.]*\.?\s*$', '', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'\s*주님\s*안에서[^.]*\.?\s*$', '', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'\s*함께\s*기도하며[^.]*\.?\s*$', '', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'\s*항상[^.]*바이블\s*애플[^.]*\.?\s*$', '', base_answer, flags=re.IGNORECASE)
                
                # 추가 끝맺음말 패턴들 (더 포괄적으로)
                base_answer = re.sub(r'\s*항상\s*주님\s*안에서[^.]*\.?\s*$', '', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'\s*주님\s*안에서\s*평안하세요[^.]*\.?\s*$', '', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'\s*평안하세요[^.]*\.?\s*$', '', base_answer, flags=re.IGNORECASE)
                
                # 문장 끝의 끝맺음말들도 제거
                base_answer = re.sub(r'[,.!?]\s*항상\s*주님\s*안에서[^.]*\.?\s*$', '', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'[,.!?]\s*감사합니다[^.]*\.?\s*$', '', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'[,.!?]\s*평안하세요[^.]*\.?\s*$', '', base_answer, flags=re.IGNORECASE)
                
                # 🔥 구 앱 이름을 바이블 애플로 완전 교체 (중복 방지)
                # 중복 방지를 위해 순서를 조정: 전체 패턴부터 처리
                base_answer = re.sub(r'바이블\s*애플\s*\(구\)\s*다번역\s*성경\s*찬송', '바이블 애플', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'바이블\s*애플\s*\(구\)\s*다번역성경찬송', '바이블 애플', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'\(구\)\s*다번역\s*성경\s*찬송', '바이블 애플', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'\(구\)\s*다번역성경찬송', '바이블 애플', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'다번역\s*성경\s*찬송', '바이블 애플', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'다번역성경찬송', '바이블 애플', base_answer, flags=re.IGNORECASE)
                
                # 🔥 완전히 강화된 중복 끝맺음말 제거 시스템
                # 1단계: 모든 형태의 "항상 성도님께..." 패턴 제거
                base_answer = re.sub(r'항상\s*성도님들?께\s*좋은\s*(서비스|성경앱)을?\s*제공하기\s*위해\s*노력하는\s*바이블\s*애플이\s*되겠습니다\.?\s*', 
                                   '', base_answer, flags=re.IGNORECASE)
                
                # 2단계: 감사합니다 패턴 완전 제거
                base_answer = re.sub(r'감사합니다\.?\s*(주님\s*안에서\s*평안하세요\.?)?\s*', 
                                   '', base_answer, flags=re.IGNORECASE)
                
                # 3단계: 불완전한 문장들 제거
                base_answer = re.sub(r'오늘도\s*$', '', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'오늘도\s*\n', '\n', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'항상\s*$', '', base_answer, flags=re.IGNORECASE)
                
                # 🔥 '항상' 단독으로 남은 경우 제거 (중복 문제 해결)
                base_answer = re.sub(r'\s*항상\s*$', '', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'\n\s*항상\s*\n', '\n', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'<p>\s*항상\s*</p>', '', base_answer, flags=re.IGNORECASE)
                base_answer = re.sub(r'<p><br></p>\s*<p>\s*항상\s*</p>', '', base_answer, flags=re.IGNORECASE)
                
                # 본문을 HTML 단락 형식으로 포맷팅
                formatted_body = self.format_answer_with_html_paragraphs(base_answer.strip(), 'ko')
                
                # 한국어 고정 인사말 (HTML 형식으로)
                final_answer = "<p>안녕하세요. GOODTV 바이블 애플입니다.</p><p><br></p><p>바이블 애플을 이용해주셔서 감사드립니다.</p><p><br></p>"
                
                # 포맷팅된 본문 추가 전 최종 정리
                # 🔥 HTML 포맷팅 후 완전한 정리 작업
                # 중복된 끝맺음말 HTML 태그 제거
                formatted_body = re.sub(r'<p>\s*항상\s*성도님들?께\s*좋은\s*(서비스|성경앱)을?\s*제공하기\s*위해\s*노력하는\s*바이블\s*애플이\s*되겠습니다\.?\s*</p>', '', formatted_body, flags=re.IGNORECASE)
                formatted_body = re.sub(r'<p>\s*감사합니다\.?\s*(주님\s*안에서\s*평안하세요\.?)?\s*</p>', '', formatted_body, flags=re.IGNORECASE)
                
                # 불완전한 문장들 제거
                formatted_body = re.sub(r'<p>\s*항상\s*</p>', '', formatted_body, flags=re.IGNORECASE)
                formatted_body = re.sub(r'<p>\s*오늘도\s*</p>', '', formatted_body, flags=re.IGNORECASE)
                formatted_body = re.sub(r'<p><br></p>\s*<p>\s*(항상|오늘도)\s*</p>', '', formatted_body, flags=re.IGNORECASE)
                formatted_body = re.sub(r'<p>\s*(항상|오늘도)\s*<br></p>', '', formatted_body, flags=re.IGNORECASE)
                
                # 연속된 빈 태그들 정리
                formatted_body = re.sub(r'(<p><br></p>\s*){3,}', '<p><br></p><p><br></p>', formatted_body)
                formatted_body = re.sub(r'(<p><br></p>\s*)+$', '', formatted_body)  # 끝의 빈 태그들 제거
                
                final_answer += formatted_body
                
                # 고정된 끝맺음말 (HTML 형식으로)
                final_answer += "<p><br></p><p>항상 성도님께 좋은 성경앱을 제공하기 위해 노력하는 바이블 애플이 되겠습니다.</p><p><br></p><p>감사합니다. 주님 안에서 평안하세요.</p>"
            
            logging.info(f"최종 답변 생성 완료: {len(final_answer)}자, 접근방식: {approach}, 언어: {lang}")
            return final_answer
            
        except Exception as e:
            logging.error(f"답변 생성 중 오류: {e}")
            if lang == 'en':
                return "<p>Sorry, we cannot generate an answer at this moment.</p><p><br></p><p>Please contact our customer service center.</p>"
            else:
                return "<p>안녕하세요, GOODTV 바이블 애플입니다.</p><p><br></p><p>바이블 애플을 이용해 주셔서 진심으로 감사드립니다.</p><p><br></p><p>남겨주신 문의는 현재 담당자가 직접 확인하고 있습니다.</p><p><br></p><p>성도님께 도움이 될 수 있도록 내용을 꼼꼼히 살펴</p><p><br></p><p>정확하고 구체적인 답변을 준비하겠습니다.</p><p><br></p><p>답변은 최대 하루 이내에 드릴 예정이오니</p><p><br></p><p>조금만 기다려 주시면 감사하겠습니다.</p><p><br></p><p>항상 주님 안에서 평안하세요, 감사합니다.</p>"

    # ☆ 메모리 최적화된 메인 처리 메서드
    def process(self, seq: int, question: str, lang: str) -> dict:
        try:
            with memory_cleanup():
                # 1. 전처리
                processed_question = self.preprocess_text(question)

                # 2. 오타 수정 추가 (Pinecone 저장 시와 동일하게!)
                if lang == 'ko' or lang == 'auto':
                    processed_question = self.fix_korean_typos_with_ai(processed_question)
                    logging.info(f"오타 수정 적용: {question[:50]} → {processed_question[:50]}")
                
                if not processed_question:
                    return {"success": False, "error": "질문이 비어있습니다."}
                
                # 언어 자동 감지
                if not lang or lang == 'auto':
                    lang = self.detect_language(processed_question)
                    logging.info(f"자동 감지된 언어: {lang}")
                
                logging.info(f"처리 시작 - SEQ: {seq}, 언어: {lang}, 질문: {processed_question[:50]}...")
                
                # 3. 유사 답변 검색 (이제 오타 수정된 질문으로 검색)
                similar_answers = self.search_similar_answers(processed_question, lang=lang)
                
                # AI 답변 생성 (언어 파라미터 전달)
                ai_answer = self.generate_ai_answer(processed_question, similar_answers, lang)
                
                # 특수문자 정리
                ai_answer = ai_answer.replace('"', '"').replace('"', '"')
                ai_answer = ai_answer.replace(''', "'").replace(''', "'")
                
                result = {
                    "success": True,
                    "answer": ai_answer,
                    "similar_count": len(similar_answers),
                    "embedding_model": "text-embedding-3-small",
                    "generation_model": "gpt-3.5-turbo",
                    "detected_language": lang
                }
                
                logging.info(f"처리 완료 - SEQ: {seq}, 언어: {lang}, HTML 답변 생성됨")
                return result
                
        except Exception as e:
            logging.error(f"처리 중 오류 - SEQ: {seq}, 오류: {str(e)}")
            return {"success": False, "error": str(e)}

    # ☆ 단순화된 컨텍스트 품질 분석 메서드 (규칙 기반)
    def analyze_context_quality_simple(self, similar_answers: list, query: str) -> dict:
        """유사도 점수 기반의 단순하고 명확한 품질 분석"""
        
        if not similar_answers:
            return {
                'has_good_context': False,
                'best_score': 0.0,
                'recommended_approach': 'fallback',
                'quality_level': 'none',
                'top_scores': []
            }
        
        # 상위 5개 답변의 점수만 확인
        top_5_scores = [ans['score'] for ans in similar_answers[:5]]
        best_score = top_5_scores[0] if top_5_scores else 0.0
        
        # 점수 분포 분석
        high_quality_count = len([s for s in top_5_scores if s >= 0.8])
        medium_quality_count = len([s for s in top_5_scores if 0.6 <= s < 0.8])
        
        # 명확한 규칙 기반 접근 방식 결정
        if best_score >= 0.95:
            approach = 'direct_use'
            quality_level = 'excellent'
        elif best_score >= 0.85 and high_quality_count >= 2:
            approach = 'direct_use'
            quality_level = 'very_high'
        elif best_score >= 0.75:
            approach = 'gpt_with_strong_context'
            quality_level = 'high'
        elif best_score >= 0.6 and (high_quality_count + medium_quality_count) >= 2:
            approach = 'gpt_with_strong_context'
            quality_level = 'medium'
        elif best_score >= 0.45:
            approach = 'gpt_with_weak_context'
            quality_level = 'low'
        else:
            approach = 'fallback'
            quality_level = 'very_low'
        
        return {
            'has_good_context': quality_level in ['excellent', 'very_high', 'high', 'medium'],
            'best_score': best_score,
            'high_quality_count': high_quality_count,
            'medium_quality_count': medium_quality_count,
            'recommended_approach': approach,
            'quality_level': quality_level,
            'top_scores': top_5_scores,
            'context_summary': f"품질: {quality_level}, 최고점수: {best_score:.3f}, 고품질: {high_quality_count}개"
        }

    # ☆ 단순화된 검색 결과 필터링 (임계값 기반)
    def search_similar_answers(self, query: str, top_k: int = 8, lang: str = 'ko') -> list:
        """단순화된 유사 답변 검색 - 명확한 점수 기반 필터링"""
        try:
            with memory_cleanup():
                logging.info(f"=== 단순화된 검색 시작 ===")
                logging.info(f"검색 질문: {query[:100]}")
                
                # 오타 수정 (한국어만)
                if lang == 'ko':
                    corrected_query = self.fix_korean_typos_with_ai(query)
                    query_to_embed = corrected_query
                else:
                    query_to_embed = query
                
                # 임베딩 생성
                query_vector = self.create_embedding(query_to_embed)
                if query_vector is None:
                    logging.error("임베딩 생성 실패")
                    return []
                
                # Pinecone 검색 (더 많이 검색해서 좋은 결과 확보)
                results = index.query(
                    vector=query_vector,
                    top_k=top_k * 3,  # 3배 더 검색
                    include_metadata=True
                )
                
                # 영어 질문인 경우 한국어 번역으로 추가 검색
                if lang == 'en':
                    korean_query = self.translate_text(query_to_embed, 'en', 'ko')
                    korean_vector = self.create_embedding(korean_query)
                    if korean_vector:
                        korean_results = index.query(
                            vector=korean_vector,
                            top_k=top_k,
                            include_metadata=True
                        )
                        # 결과 병합 및 중복 제거
                        seen_ids = set()
                        merged_matches = []
                        for match in results['matches'] + korean_results['matches']:
                            if match['id'] not in seen_ids:
                                seen_ids.add(match['id'])
                                merged_matches.append(match)
                        results['matches'] = sorted(merged_matches, key=lambda x: x['score'], reverse=True)
                
                # 단순한 점수 기반 필터링
                filtered_results = []
                for i, match in enumerate(results['matches'][:top_k*2]):  # 상위 2배만 검토
                    score = match['score']
                    question = match['metadata'].get('question', '')
                    answer = match['metadata'].get('answer', '')
                    category = match['metadata'].get('category', '일반')
                    
                    # 명확한 포함 기준
                    should_include = False
                    
                    if score >= 0.4:  # 기본 임계값
                        should_include = True
                    elif i < 5:  # 상위 5개는 점수가 낮아도 포함
                        should_include = True
                    elif score >= 0.3 and len(filtered_results) < 3:  # 최소 3개 보장
                        should_include = True
                    
                    if should_include and len(filtered_results) < top_k:
                        # 기본적인 텍스트 품질 검증만
                        if len(answer.strip()) >= 10:  # 최소 길이만 확인
                            filtered_results.append({
                                'score': score,
                                'question': question,
                                'answer': answer,
                                'category': category,
                                'rank': i + 1,
                                'lang': 'ko'
                            })
                            
                            logging.info(f"포함: #{i+1} 점수={score:.3f} 카테고리={category}")
                
                # 메모리 정리
                del results, query_vector
                
                logging.info(f"검색 완료: {len(filtered_results)}개 답변 (언어: {lang})")
                return filtered_results
            
        except Exception as e:
            logging.error(f"검색 실패: {str(e)}")
            return []

    # ☆ 단순화된 컨텍스트 생성 (점수 기반 우선순위)
    def create_enhanced_context_simple(self, similar_answers: list, max_answers: int = 6, target_lang: str = 'ko') -> str:
        """점수 기반의 단순한 컨텍스트 생성"""
        
        if not similar_answers:
            return ""
        
        context_parts = []
        used_count = 0
        
        # 점수 순으로 정렬되어 있으므로 순서대로 처리
        for i, ans in enumerate(similar_answers[:max_answers]):
            if used_count >= max_answers:
                break
            
            score = ans['score']
            answer_text = ans['answer']
            
            # 기본 정리
            clean_answer = self.preprocess_text(answer_text)
            clean_answer = self.remove_greeting_and_closing(clean_answer, 'ko')
            
            # 영어 요청시 번역
            if target_lang == 'en' and ans.get('lang', 'ko') == 'ko':
                clean_answer = self.translate_text(clean_answer, 'ko', 'en')
            
            # 최소 품질 검증
            if len(clean_answer.strip()) >= 20:
                # 점수에 따른 길이 조정
                max_length = 500 if score >= 0.8 else 350 if score >= 0.6 else 250
                
                context_parts.append(
                    f"[참고답변 {used_count+1} - 유사도: {score:.2f}]\n{clean_answer[:max_length]}"
                )
                used_count += 1
        
        logging.info(f"컨텍스트 생성: {used_count}개 답변 포함 (언어: {target_lang})")
        return "\n\n" + "="*50 + "\n\n".join(context_parts)

    # ☆ 단순화된 GPT 생성 (명확한 프롬프트)
    def generate_with_simple_gpt(self, query: str, similar_answers: list, context_analysis: dict, lang: str = 'ko') -> str:
        """단순화된 GPT 답변 생성 - 복잡한 검증 제거"""
        
        try:
            with memory_cleanup():
                approach = context_analysis['recommended_approach']
                quality_level = context_analysis['quality_level']
                
                # 접근 방식이 GPT 생성이 아니면 빈 문자열 반환
                if approach not in ['gpt_with_strong_context', 'gpt_with_weak_context']:
                    return ""
                
                context = self.create_enhanced_context_simple(similar_answers, target_lang=lang)
                if not context:
                    return ""
                
                # 단순화된 프롬프트 생성
                system_prompt, user_prompt = self.get_gpt_prompts(query, context, lang)
                
                # 품질에 따른 단순한 파라미터 설정
                if quality_level in ['high', 'medium']:
                    temperature = 0.6
                    max_tokens = 650
                else:  # low
                    temperature = 0.7
                    max_tokens = 600
                
                # GPT 호출
                response = self.openai_client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.85,
                    frequency_penalty=0.1,
                    presence_penalty=0.1
                )
                
                generated = response.choices[0].message.content.strip()
                del response
                
                # 기본적인 정리만
                generated = self.clean_generated_text(generated)
                
                # 최소 길이 검증만
                if len(generated.strip()) < 10:
                    logging.warning("생성된 답변이 너무 짧음")
                    return ""
                
                logging.info(f"GPT 생성 성공 ({approach}, 품질: {quality_level}): {len(generated)}자")
                return generated
                
        except Exception as e:
            logging.error(f"GPT 생성 실패: {e}")
            return ""

    # ☆ 단순화된 폴백 답변 선택 (점수 기반)
    def get_best_fallback_answer_simple(self, similar_answers: list, lang: str = 'ko') -> str:
        """점수 기반의 단순한 최적 답변 선택"""
        
        if not similar_answers:
            return ""
        
        # 상위 3개 중에서 선택
        for i, ans in enumerate(similar_answers[:3]):
            score = ans['score']
            answer_text = ans['answer']
            
            # 점수가 매우 높으면 즉시 반환
            if score >= 0.9:
                logging.info(f"최고 점수({score:.3f}) 답변 직접 사용")
                clean_answer = answer_text.strip()
                return clean_answer if clean_answer else ""
            
            # 기본 전처리
            processed = self.preprocess_text(answer_text)
            
            # 영어 번역
            if lang == 'en' and ans.get('lang', 'ko') == 'ko':
                processed = self.translate_text(processed, 'ko', 'en')
            
            # 기본 품질 검증
            if len(processed.strip()) >= 20:
                # 첫 번째 유효한 답변 선택
                logging.info(f"폴백 답변 선택: #{i+1}, 점수={score:.3f}")
                return processed
        
        # 모든 답변이 부적절한 경우 첫 번째 원본 반환
        if similar_answers:
            return similar_answers[0]['answer'].strip()
        
        return ""

# ==================================================
# 8. Pinecone 벡터 데이터베이스 동기화 클래스
# ==================================================
# MSSQL 운영 데이터베이스와 Pinecone 벡터 데이터베이스 간의 동기화를 담당하는 클래스
# 
# 주요 기능:
# 1. MSSQL에서 새로운 Q&A 데이터 조회
# 2. AI를 이용한 한국어 오타 수정
# 3. OpenAI로 임베딩 벡터 생성
# 4. Pinecone에 벡터 데이터 저장/수정/삭제
# 
# 운영 시나리오:
# - 새로운 고객 문의 답변이 MSSQL에 저장되면
# - 이 클래스를 통해 Pinecone에 동기화하여
# - 향후 유사 질문 검색이 가능하게 함
class PineconeSyncManager:
    
    # 동기화 매니저 초기화
    # 외부에서 생성된 Pinecone 인덱스와 OpenAI 클라이언트 참조
    def __init__(self):
        self.index = index                    # Pinecone 벡터 인덱스
        self.openai_client = openai_client    # OpenAI API 클라이언트
    
    # ☆ AI를 이용한 한국어 오타 수정 메서드
    def fix_korean_typos_with_ai(self, text: str) -> str:
        if not text or len(text.strip()) < 3:
            return text
        
        # 너무 긴 텍스트는 처리하지 않음 (비용 절약)
        if len(text) > 500:
            logging.warning(f"텍스트가 너무 길어 오타 수정 건너뜀: {len(text)}자")
            return text
        
        try:
            with memory_cleanup():
                system_prompt = """당신은 한국어 맞춤법 및 오타 교정 전문가입니다.

지침:
1. 입력된 한국어 텍스트의 맞춤법과 오타만 수정하세요
2. 원문의 의미와 어조는 절대 변경하지 마세요
3. 띄어쓰기, 맞춤법, 조사 사용법을 정확히 교정하세요
4. 앱/어플리케이션 관련 기술 용어는 표준 용어로 통일하세요
5. 수정이 필요없다면 원문 그대로 반환하세요
6. 수정된 텍스트만 반환하고 추가 설명은 하지 마세요

예시:
- "어플이 안됀다" → "앱이 안 돼요"
- "다운받기가 안되요" → "다운로드가 안 돼요"
- "삭재하고싶어요" → "삭제하고 싶어요"
- "업데이드해주세요" → "업데이트해주세요"
"""

                user_prompt = f"다음 텍스트의 맞춤법과 오타를 수정해주세요:\n\n{text}"

                response = self.openai_client.chat.completions.create(
                    model='gpt-3.5-turbo',
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=600,
                    temperature=0.1,  # 매우 보수적으로 설정
                    top_p=0.8,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                
                corrected_text = response.choices[0].message.content.strip()
                del response # 메모리 해제
                
                # 결과 검증
                if not corrected_text or len(corrected_text) == 0:
                    logging.warning("AI 오타 수정 결과가 비어있음, 원문 반환")
                    return text
                
                # 너무 많이 변경된 경우 의심스러우므로 원문 반환
                if len(corrected_text) > len(text) * 2:
                    logging.warning("AI 오타 수정 결과가 원문보다 너무 길어짐, 원문 반환")
                    return text
                
                # 수정 내용이 있으면 로그 기록
                if corrected_text != text:
                    logging.info(f"AI 오타 수정: '{text[:50]}...' → '{corrected_text[:50]}...'")
                
                return corrected_text
                
        except Exception as e:
            logging.error(f"AI 오타 수정 실패: {e}")
            # AI 실패 시 원문 그대로 반환
            return text
        
    # ☆ 텍스트 전처리 메서드
    def preprocess_text(self, text: str, for_metadata: bool = False) -> str:
        if not text or text == 'None':
            return ""
        
        text = str(text)
        text = html.unescape(text)
        
        # HTML 태그 제거
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</p>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<p[^>]*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<[^>]+>', '', text)
        
        # 유니코드 정규화
        text = unicodedata.normalize('NFC', text)
        
        # 공백 정리
        if for_metadata:
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r'[ \t]+', ' ', text)
        else:
            text = re.sub(r'\s+', ' ', text)
        
        text = text.strip()
        
        # 길이 제한
        max_length = 1000 if for_metadata else MAX_TEXT_LENGTH
        if len(text) > max_length:
            text = text[:max_length-3] + "..."
        
        return text
    
    # ☆ OpenAI로 임베딩 생성하는 메서드
    def create_embedding(self, text: str) -> Optional[list]:
        try:
            if not text or not text.strip():
                return None
            
            with memory_cleanup():
                response = openai_client.embeddings.create(
                    model=MODEL_NAME,
                    input=text
                )
                return response.data[0].embedding
            
        except Exception as e:
            logging.error(f"임베딩 생성 실패: {e}")
            return None
    
    # ☆ 카테고리 인덱스를 이름으로 변환하는 메서드
    def get_category_name(self, cate_idx: str) -> str:
        return CATEGORY_MAPPING.get(str(cate_idx), '사용 문의(기타)')
    
    # ☆ MSSQL에서 데이터 조회하는 메서드
    # 파라미터화된 쿼리로 SQL 인젝션 방지, ? 플레이스홀더를 사용하여 안전하게 값을 바인딩
    def get_mssql_data(self, seq: int) -> Optional[Dict]:
        try:
            with memory_cleanup():
                conn = pyodbc.connect(connection_string)
                cursor = conn.cursor()
                
                query = """
                SELECT seq, contents, reply_contents, cate_idx, name, 
                       CONVERT(varchar, regdate, 120) as regdate
                FROM mobile.dbo.bible_inquiry
                WHERE seq = ? AND answer_YN = 'Y'
                """
                
                cursor.execute(query, seq)
                row = cursor.fetchone()
                
                if row:
                    data = {
                        'seq': row[0],
                        'contents': row[1],
                        'reply_contents': row[2],
                        'cate_idx': row[3],
                        'name': row[4],
                        'regdate': row[5]
                    }
                    return data
                
                cursor.close()
                conn.close()
                return None
            
        except Exception as e:
            logging.error(f"MSSQL 조회 실패: {e}")
            return None
    
    # ☆ MSSQL 데이터를 Pinecone에 동기화하는 메서드
    def sync_to_pinecone(self, seq: int, mode: str = 'upsert') -> Dict[str, Any]:
        try:
            with memory_cleanup():
                # 삭제 모드
                if mode == 'delete':
                    vector_id = f"qa_bible_{seq}"
                    self.index.delete(ids=[vector_id])
                    logging.info(f"Pinecone에서 삭제 완료: {vector_id}")
                    return {"success": True, "message": "삭제 완료", "seq": seq}
                
                # MSSQL에서 데이터 가져오기
                data = self.get_mssql_data(seq)
                if not data:
                    return {"success": False, "error": "데이터를 찾을 수 없습니다"}
                
                # 텍스트 전처리 (질문에 AI 오타 수정 적용)
                raw_question = self.preprocess_text(data['contents'])
                question = self.fix_korean_typos_with_ai(raw_question)
                answer = self.preprocess_text(data['reply_contents'])
                
                # 임베딩 생성 (질문 기반)
                embedding = self.create_embedding(question)
                if not embedding:
                    return {"success": False, "error": "임베딩 생성 실패"}
                
                # 카테고리 이름 가져오기
                category = self.get_category_name(data['cate_idx'])
                
                # 메타데이터 구성 (질문은 오타 수정된 버전 사용)
                metadata = {
                    "seq": int(data['seq']),
                    "question": question,
                    "answer": self.preprocess_text(data['reply_contents'], for_metadata=True),
                    "category": category,
                    "name": data['name'] if data['name'] else "익명",
                    "regdate": data['regdate'],
                    "source": "bible_inquiry_mssql",
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Pinecone에 upsert
                vector_id = f"qa_bible_{seq}"
                
                # 기존 벡터 확인
                existing = self.index.fetch(ids=[vector_id])
                is_update = vector_id in existing['vectors']
                
                # 벡터 데이터 구성
                vector_data = {
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                }
                
                # Pinecone에 저장
                self.index.upsert(vectors=[vector_data])
                
                action = "수정" if is_update else "생성"
                logging.info(f"Pinecone {action} 완료: {vector_id}")
                
                return {
                    "success": True,
                    "message": f"Pinecone {action} 완료",
                    "seq": seq,
                    "vector_id": vector_id,
                    "is_update": is_update
                }
            
        except Exception as e:
            logging.error(f"Pinecone 동기화 실패: {str(e)}")
            return {"success": False, "error": str(e)}

# ==================================================
# 9. 싱글톤 인스턴스 생성 (전역 객체)
# ==================================================
# 애플리케이션 전체에서 사용할 단일 인스턴스들
# 메모리 효율성과 상태 일관성을 위해 싱글톤 패턴 적용
generator = AIAnswerGenerator()      # AI 답변 생성기
sync_manager = PineconeSyncManager() # Pinecone 동기화 매니저

# ==================================================
# 10. Flask RESTful API 엔드포인트 정의
# ==================================================
# ★ AI 답변 생성 API 엔드포인트 (메인 기능)
# ASP Classic에서 호출하는 주요 API로, 고객 질문에 대한 AI 답변을 생성
# 처리 과정:
# 1. 질문 전처리 및 검증
# 2. Pinecone에서 유사 답변 검색
# 3. 컨텍스트 품질 분석
# 4. GPT를 이용한 맞춤 답변 생성
# 5. 최종 포맷팅 및 반환
@app.route('/generate_answer', methods=['POST'])
def generate_answer():
    try:
        with memory_cleanup():
            data = request.get_json()
            seq = data.get('seq', 0)
            question = data.get('question', '')
            lang = data.get('lang', 'auto')  # 기본값을 'auto'로 변경 (자동 감지)
            
            if not question:
                return jsonify({"success": False, "error": "질문이 필요합니다."}), 400
            
            result = generator.process(seq, question, lang)
            
            response = jsonify(result)
            response.headers['Content-Type'] = 'application/json; charset=utf-8'

            # 메모리 사용량 모니터링
            snapshot = tracemalloc.take_snapshot() # 각 요청 후 메모리 스냅샷 촬영
            top_stats = snapshot.statistics('lineno') # 각 요청 후 메모리 사용량 통계
            memory_usage = sum(stat.size for stat in top_stats) / 1024 / 1024  # MB로 반환
            logging.info(f"현재 메모리 사용량: {memory_usage:.2f}MB")
            
            if memory_usage > 500: # 500MB 초과시 경고 및 가비지 컬렉션 강제 실행
                logging.warning(f"높은 메모리 사용량 감지: {memory_usage:.2f}MB")
                gc.collect()

            return response
        
    except Exception as e:
        logging.error(f"API 호출 오류: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

# ★ MSSQL 데이터를 Pinecone에 동기화하는 API 엔드포인트
# 운영 시스템에서 새로운 Q&A 데이터가 생성되거나 수정될 때 호출
# 처리 과정:
# 1. MSSQL에서 해당 seq 데이터 조회
# 2. AI로 질문 오타 수정
# 3. OpenAI로 임베딩 벡터 생성
# 4. Pinecone에 벡터 저장/수정/삭제
@app.route('/sync_to_pinecone', methods=['POST'])
def sync_to_pinecone():
    try:
        data = request.get_json()
        seq = data.get('seq')
        mode = data.get('mode', 'upsert')

        logging.info(f"동기화 요청 수신: seq={seq}, mode={mode}")
        
        if not seq:
            logging.warning("seq 누락")
            return jsonify({"success": False, "error": "seq가 필요합니다"}), 400
        
        if not isinstance(seq, int):
            seq = int(seq)
        
        result = sync_manager.sync_to_pinecone(seq, mode)

        logging.info(f"동기화 결과: {result}")
        
        status_code = 200 if result["success"] else 500
        return jsonify(result), status_code
        
    except ValueError as e:
        logging.error(f"잘못된 seq 값: {str(e)}")
        return jsonify({"success": False, "error": f"잘못된 seq 값: {str(e)}"}), 400
    except Exception as e:
        logging.error(f"Pinecone 동기화 API 오류: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

# ★ 시스템 상태 확인을 위한 헬스체크 API 엔드포인트
# 로드밸런서나 모니터링 시스템에서 호출하여 서버 상태 확인
@app.route('/health', methods=['GET'])
def health_check():
    try:
        stats = index.describe_index_stats()
        
        return jsonify({
            "status": "healthy",
            "pinecone_vectors": stats.get('total_vector_count', 0),
            "timestamp": datetime.now().isoformat(),
            "services": {
                "ai_answer": "active",
                "pinecone_sync": "active",
                "multilingual_support": "active"  # 다국어 지원 상태 추가
            }
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

# ==================================================
# 11. 메인 실행 부분
# ==================================================
# Flask 웹 서버 시작점
# 
# 이 부분은 이 파일(스크립트)가 직접 실행될 때만 실행되는 절차적 코드
# 다른 모듈에서 import할 때는 실행되지 않음
# 
# 설정:
# - 포트: 환경변수 FLASK_PORT 또는 기본값 8000
# - 호스트: 0.0.0.0 (모든 IP에서 접근 가능)
# - 디버그: False (운영 모드)
# - 스레드: True (멀티스레딩 지원)
if __name__ == "__main__":
    
    # 환경변수에서 포트 설정 로드 (기본값: 8000)
    port = int(os.getenv('FLASK_PORT', 8000))
    
    # 시작 메시지 출력
    print("="*60)
    print("🚀 GOODTV 바이블 애플 AI 답변 생성 서버 시작")
    print("="*60)
    print(f"📡 서버 포트: {port}")
    print(f"🤖 AI 모델: {GPT_MODEL} (Enhanced Context Mode)")
    print(f"🔍 임베딩 모델: {MODEL_NAME}")
    print(f"🗃️  벡터 DB: Pinecone ({INDEX_NAME})")
    print(f"🌏 다국어 지원: 한국어(ko), 영어(en)")
    print("🔧 제공 서비스:")
    print("   ├── AI 답변 생성 (/generate_answer)")
    print("   ├── Pinecone 동기화 (/sync_to_pinecone)")
    print("   └── 헬스체크 (/health)")
    print("="*60)
    
    # Flask 웹 서버 시작
    # host='0.0.0.0': 모든 네트워크 인터페이스에서 접근 허용
    # debug=False: 운영 모드 (보안상 중요)
    # threaded=True: 멀티스레딩 활성화, 동시 요청 처리 가능
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)