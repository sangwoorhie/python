#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=== AI 답변 생성 Flask API 서버 ===
파일명: free_4_ai_answer_generator.py
목적: ASP Classic에서 호출하는 AI 답변 생성 API + Pinecone 벡터DB 동기화
주요 기능:
1. OpenAI GPT-3.5-turbo를 이용한 자연어 답변 생성
2. Pinecone 벡터 데이터베이스에서 유사 답변 검색
3. MSSQL 데이터베이스와 Pinecone 동기화
4. 메모리 최적화 및 모니터링
"""

# ==================================================
# 1. 필수 라이브러리 임포트 구간
# ==================================================
# 기본 Python 모듈들
import os              # 환경변수 및 파일 시스템 작업
import sys             # 시스템 관련 기능
import json            # JSON 데이터 처리
import json as json_module  # JSON 모듈의 별칭 (코드 내 중복 방지)
import re              # 정규표현식 패턴 매칭
import html            # HTML 엔티티 처리
import unicodedata     # 유니코드 문자 정규화
import logging         # 로그 기록 시스템
import gc              # 가비지 컬렉션 (메모리 관리)

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

# 성능 모니터링 관련
from memory_profiler import profile        # 메모리 사용량 프로파일링
import tracemalloc                         # 메모리 추적
import threading                           # 멀티스레딩
from contextlib import contextmanager      # 컨텍스트 매니저 (with문 사용)

# ==================================================
# 2. 시스템 초기화 및 설정
# ==================================================
# 메모리 추적 시작 - 메모리 누수 및 사용량을 모니터링하기 위함 (이 시점부터 모든 메모리 할당이 기록되어 나중에 메모리 사용량 분석이 가능해짐)
tracemalloc.start() 

# Flask 웹 애플리케이션 인스턴스 생성
# __name__: 현재 모듈명을 전달하여 Flask가 리소스 위치를 찾을 수 있게 함
app = Flask(__name__)

# CORS 설정 - 모든 엔드포인트에서 cross-origin 요청청을 허용 (ASP Classic에서 호출하기 위함)
CORS(app)

# ==================================================
# 3. 로깅 시스템 설정
# ==================================================
# 운영환경에서 로그를 파일로 저장하여 디버깅 및 모니터링 가능
# 로깅 설정에서 중요한 것은 encoding='utf-8'. 한글 에러 메시지나 디버그 정보가 깨지지 않고 로그 파일에 기록되도록 함
logging.basicConfig(
    filename='/home/ec2-user/python/logs/ai_generator.log',  # 로그 파일 경로
    level=logging.INFO,                                      # INFO 레벨 이상 로그 기록
    format='%(asctime)s - %(levelname)s - %(message)s',     # 로그 포맷
    encoding='utf-8'                                         # 한글 지원을 위한 UTF-8 인코딩
)

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
TEMPERATURE = 0.3                # 창의성 수준 (낮을수록 일관된 답변)

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

# ==================================================
# 5. 유틸리티 함수 정의
# ==================================================
# 메모리 정리를 위한 컨텍스트 매니저
# with문 진입시 try 블록 실행
# yield에서 일시정지하고 with 블록 내부 코드 실행
# yield는 파이썬에서 제너레이터를 만들때 사용하는 키워드로써, 함수 내에서 yield를 사용하면 그 함수는 제너레이터 함수가 됩니다.
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
# 7. AI 답변 생성 메인 클래스 (객체지향 프로그래밍)
# ==================================================
    # AI 답변 생성을 담당하는 메인 클래스
    
    # 주요 기능:
    # 1. 텍스트 전처리 및 정제
    # 2. OpenAI를 이용한 임베딩 생성
    # 3. Pinecone에서 유사 답변 검색
    # 4. GPT를 이용한 맞춤형 답변 생성
    # 5. 한국어 텍스트 검증 및 포맷팅
    
    # 객체지향 설계로 관련 기능들을 하나의 클래스에 캡슐화

class AIAnswerGenerator:

    # 클래스 초기화 메서드
    # OpenAI 클라이언트를 인스턴스 변수로 설정. 이는 의존성 주입 패턴의 간소화된 형태
    def __init__(self):
        
        self.openai_client = openai_client
    
    # ☆ 입력 텍스트를 AI 처리에 적합하게 전처리하는 메서드
    # Args:
    #     text (str): 원본 텍스트
            
    # Returns:
    #     str: 정제된 텍스트
    def preprocess_text(self, text: str) -> str:
        
        # null 체크
        if not text:
            return ""
        
        # 문자열로 변환 및 HTML 엔티티 디코딩
        text = str(text)
        text = html.unescape(text)  # &amp; → &, &lt; → < 등
        
        # HTML 태그 제거 및및 텍스트 형태로 변환 (구조 유지)
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)      # <br> → 줄바꿈
        text = re.sub(r'</p>', '\n\n', text, flags=re.IGNORECASE)         # </p> → 단락 구분
        text = re.sub(r'<p[^>]*>', '\n', text, flags=re.IGNORECASE)       # <p> → 줄바꿈
        text = re.sub(r'<li[^>]*>', '\n• ', text, flags=re.IGNORECASE)    # <li> → 불릿포인트
        text = re.sub(r'</li>', '', text, flags=re.IGNORECASE)            # </li> 제거
        text = re.sub(r'<[^>]+>', '', text)                               # 나머지 HTML 태그 모두 제거
        
        # 공백 및 줄바꿈 정규화 - 일관된 형태로 변환
        text = re.sub(r'\n{3,}', '\n\n', text)    # 3개 이상 줄바꿈 → 2개로 제한
        text = re.sub(r'[ \t]+', ' ', text)       # 연속 공백/탭 → 단일 공백
        text = text.strip()                       # 앞뒤 공백 제거
        
        return text

    # ☆ JSON 문자열 이스케이프 처리
    # 특수문자가 포함된 텍스트를 JSON으로 안전하게 변환
    def escape_json_string(self, text: str) -> str:
        
        if not text:
            return ""
        escaped = json_module.dumps(text, ensure_ascii=False) # ensure_ascii=False: 한글 깨짐 방지
        return escaped[1:-1]  # 앞뒤 따옴표 제거


    # ☆ OpenAI API를 사용하여 텍스트를 벡터로 변환하는 메서드
    # 벡터 임베딩: 텍스트의 의미를 수치 배열로 표현하여 유사도 계산 가능
        
    # Args:
    #     text (str): 임베딩할 텍스트
            
    # Returns:
    #     Optional[list]: 1536차원 벡터 배열 또는 None(실패시)
    def create_embedding(self, text: str) -> Optional[list]:
        
        # 이중 검증을 수행. 빈 문자열뿐만 아니라 공백만 있는 문자열도 걸러냄. 이는 텍스트 비교 시 예상치 못한 오류를 방지하기 위함
        if not text or not text.strip():
            return None
            
        try:
            with memory_cleanup(): # 메모리 누수 방지 (블록 종료 시 가비지 컬렉션)
                # OpenAI Embedding API 호출
                response = self.openai_client.embeddings.create(
                    model='text-embedding-3-small',    # 작고 빠른 임베딩 모델
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
    #     similarity_threshold (float): 유사도 임계값 (기본값: 0.6)
            
    # Returns:
    #     list: 유사 답변 리스트 [{'score': float, 'question': str, 'answer': str, ...}, ...]
    def search_similar_answers(self, query: str, top_k: int = 5, similarity_threshold: float = 0.6) -> list:
        
        try:
            with memory_cleanup():
                # 1. 검색 질문을 벡터로 변환
                query_vector = self.create_embedding(query)
                if query_vector is None:
                    return []
                
                # 2. Pinecone에서 벡터 유사도 검색 수행
                # Pinecone 쿼리에서 include_metadata=True는 벡터뿐만 아니라 저장된 메타데이터(질문, 답변, 카테고리)도 함께 반환받음
                results = index.query(
                    vector=query_vector,           # 검색할 벡터
                    top_k=top_k,                   # 상위 5개 결과
                    include_metadata=True          # 메타데이터 포함 (질문, 답변, 카테고리 등)
                )
                
                # 3. 결과 필터링 및 구조화
                filtered_results = []
                for i, match in enumerate(results['matches']): # enumerate로 순위(rank) 생성
                    score = match['score']  # 유사도 점수 (0~1, 높을수록 유사)
                    question = match['metadata'].get('question', '')
                    answer = match['metadata'].get('answer', '')
                    category = match['metadata'].get('category', '일반')
                    
                    # 유사도 임계값(0.6) 이상만 포함하여 정확도 향상, 임계값 이하는 버림 (노이즈 제거)
                    if score >= similarity_threshold:
                        filtered_results.append({
                            'score': score,
                            'question': question,
                            'answer': answer,
                            'category': category,
                            'rank': i + 1
                        })
                        
                        # 디버깅을 위한 상세 로깅
                        logging.info(f"유사 답변 #{i+1}: 점수={score:.3f}, 카테고리={category}")
                        logging.info(f"참고 질문: {question[:50]}...")
                        logging.info(f"참고 답변: {answer[:100]}...")
                
                # 4. 메모리 정리
                del results # 원본 응답 객체 즉시 삭제 (메모리 해제)
                del query_vector # 검색 벡터 즉시 삭제 (메모리 해제)
                
                logging.info(f"총 {len(filtered_results)}개의 유사 답변 검색 완료")
                return filtered_results
                
        except Exception as e:
            logging.error(f"Pinecone 검색 실패: {str(e)}")
            return []

    # ☆ 검색된 유사 답변들의 품질을 분석하여 최적의 답변 생성 전략을 결정하는 메서드

    # Args:
    #     similar_answers (list): 검색된 유사 답변 리스트
    #     query (str): 원본 질문
    #     
    # Returns:
    #     dict: 분석 결과 및 권장 접근 방식
    def analyze_context_quality(self, similar_answers: list, query: str) -> dict:
        # 유사 답변이 없는 경우
        if not similar_answers:
            return {
                'has_good_context': False,
                'best_score': 0.0,
                'recommended_approach': 'fallback',
                'context_summary': '유사 답변이 없습니다.'
            }
        
        # 품질 지표 계산
        # 리스트 컴프리헨션 (리스트를 쉽게, 짧게 한 줄로 만들 수 있는 파이썬의 문법)을 사용한 효율적인 카운팅: 한 번의 순회로 조건에 맞는 항목 수를 계산
        # [ ( 변수를 활용한 값 ) for ( 사용할 변수 이름 ) in ( 순회할 수 있는 값 ) if ( 조건 ) ]
        best_score = similar_answers[0]['score']  # 가장 높은 유사도 점수
        high_quality_count = len([ans for ans in similar_answers if ans['score'] >= 0.7])    # 고품질(0.7+) 답변 개수
        medium_quality_count = len([ans for ans in similar_answers if 0.5 <= ans['score'] < 0.7])  # 중품질(0.5-0.7) 답변 개수
        
        # 상위 5개에서 카테고리 추출하여 분포 계산 (비슷한 카테고리가 많으면 도메인 특화된 답변 가능)
        # 딕셔너리 컴프리헨션: { 키: 값 for 키, 값 in 순회할 수 있는 값 if 조건 }
        categories = [ans['category'] for ans in similar_answers[:5]] # 상위 5개 답변의 카테고리 추출 (리스트 컴프리헨션)
        category_distribution = {cat: categories.count(cat) for cat in set(categories)} # 카테고리별 개수 계산 (딕셔너리 컴프리헨션), set()으로 중복 제거, 카운트 메서드 사용
        
        # 의사 결정 트리 : 최적의 답변 생성 전략을 결정하는 알고리즘
        # 최고 유사도 점수가 0.8 이상이면 기존 답변 직접 활용
        # 최고 유사도 점수가 0.7 이상이고 고품질 답변이 2개 이상이면 고품질 컨텍스트로 GPT 생성
        # 최고 유사도 점수가 0.6 이상이고 중품질 답변이 3개 이상이면 약한 컨텍스트로 GPT 생성
        # 그 외는 품질이 낮아 폴백 처리
        if best_score >= 0.8:
            approach = 'direct_use'                # 매우 유사 → 기존 답변 직접 활용
        elif best_score >= 0.7 or high_quality_count >= 2:
            approach = 'gpt_with_strong_context'   # 고품질 컨텍스트로 GPT 생성
        elif best_score >= 0.6 or medium_quality_count >= 3:
            approach = 'gpt_with_weak_context'     # 약한 컨텍스트로 GPT 생성
        else:
            approach = 'fallback'                  # 품질이 낮아 폴백 처리
        
        # 분석 결과 구조화
        analysis = {
            'has_good_context': best_score >= 0.6,
            'best_score': best_score,
            'high_quality_count': high_quality_count,
            'medium_quality_count': medium_quality_count,
            'category_distribution': category_distribution,
            'recommended_approach': approach,
            'context_summary': f"최고점수: {best_score:.3f}, 고품질: {high_quality_count}개, 중품질: {medium_quality_count}개"
        }
        
        logging.info(f"컨텍스트 분석 결과: {analysis}")
        return analysis

    # GPT 답변 생성을 위한 향상된 컨텍스트 생성 메서드
    # 
    # 컨텍스트 생성 전략:
    # 1. 품질별 답변 그룹핑 (고/중/낮은 품질)
    # 2. 고품질 답변 우선 선택 (최대 4개)
    # 3. 중품질 답변으로 보완 (최대 3개)
    # 4. 최소 개수 미달시 중간 품질 답변 추가
    # 5. 텍스트 정제 및 길이 제한
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
    def create_enhanced_context(self, similar_answers: list, max_answers: int = 7) -> str:
        if not similar_answers:
            return ""
        
        context_parts = []  # 컨텍스트 구성 요소들
        used_answers = 0    # 사용된 답변 개수
        
        # 유사도 점수에 따른 답변 그룹핑
        high_score = [ans for ans in similar_answers if ans['score'] >= 0.7]      # 고품질 (70% 이상 유사)
        medium_score = [ans for ans in similar_answers if 0.5 <= ans['score'] < 0.7]  # 중품질 (50-70%)
        low_score = [ans for ans in similar_answers if 0.4 <= ans['score'] < 0.5]     # 낮은 품질 (40-50%)
        
        # 1단계: 고품질 답변 우선 포함 (최대 4개)
        for ans in high_score[:4]:
            if used_answers >= max_answers:
                break
            # 제어 문자 및 HTML 태그 제거
            clean_answer = re.sub(r'[\b\r\f\v\x00-\x08\x0B\x0C\x0E-\x1F\x7F]|<[^>]+>', '', ans['answer'])
            # 유효한 한국어 텍스트이고 충분한 길이인지 검증
            if self.is_valid_korean_text(clean_answer) and len(clean_answer.strip()) > 20:
                context_parts.append(f"[참고답변 {used_answers+1} - 점수: {ans['score']:.2f}]\n{clean_answer[:400]}")
                used_answers += 1
        
        # 2단계: 중품질 답변으로 보완 (최대 3개)
        for ans in medium_score[:3]:
            if used_answers >= max_answers:
                break
            clean_answer = re.sub(r'[\b\r\f\v\x00-\x08\x0B\x0C\x0E-\x1F\x7F]|<[^>]+>', '', ans['answer'])
            if self.is_valid_korean_text(clean_answer) and len(clean_answer.strip()) > 20:
                context_parts.append(f"[참고답변 {used_answers+1} - 점수: {ans['score']:.2f}]\n{clean_answer[:300]}")
                used_answers += 1
        
        # 3단계: 답변이 부족한 경우 중간 품질 답변 추가 (50-60% 구간)
        if used_answers < 3:  # 최소 3개 이상 확보하기 위함
            medium_low_score = [ans for ans in similar_answers if 0.5 <= ans['score'] < 0.6]
            for ans in medium_low_score[:2]:
                if used_answers >= max_answers:
                    break
                clean_answer = re.sub(r'[\b\r\f\v\x00-\x08\x0B\x0C\x0E-\x1F\x7F]|<[^>]+>', '', ans['answer'])
                if self.is_valid_korean_text(clean_answer) and len(clean_answer.strip()) > 20:
                    context_parts.append(f"[참고답변 {used_answers+1} - 점수: {ans['score']:.2f}]\n{clean_answer[:250]}")
                    used_answers += 1
        
        logging.info(f"컨텍스트 생성: {used_answers}개의 답변 포함")
        
        # 최종 컨텍스트 조합 (구분선으로 답변들 분리)
        return "\n\n" + "="*50 + "\n\n".join(context_parts)

    # 이전 앱 이름을 제거하는 메서드 (구 다번역성경찬송 등)
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

    # 답변 텍스트를 HTML 단락 형식으로 포맷팅하는 메서드
    def format_answer_with_html_paragraphs(self, text: str) -> str:
        if not text:
            return ""
        
        text = self.remove_old_app_name(text)
        
        # 문장을 마침표, 느낌표, 물음표로 분리
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        paragraphs = []
        current_paragraph = []
        
        # 단락 분리 트리거 키워드들 (더 포괄적으로 확장)
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
            
            # 문장이 너무 길면 (50자 이상) 새 단락 고려
            if len(sentence) > 50 and current_paragraph:
                should_break = True
            
            if should_break and current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = [sentence]
            else:
                current_paragraph.append(sentence)
        
        # 마지막 단락 처리
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        # HTML 단락으로 변환
        html_paragraphs = []
        for i, paragraph in enumerate(paragraphs):
            html_paragraphs.append(f"<p>{paragraph}</p>")
            
            # 단락 사이에 빈 줄 추가 (마지막 단락 제외)
            if i < len(paragraphs) - 1:
                html_paragraphs.append("<p><br></p>")
        
        return ''.join(html_paragraphs)

    # 답변 텍스트를 정리하고 포맷팅하는 메서드 (Quill 에디터용)
    def clean_answer_text(self, text: str) -> str:
        if not text:
            return ""
        
        # 제어 문자만 제거하고 HTML 태그는 유지
        text = re.sub(r'[\b\r\f\v]', '', text)
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        # HTML 태그 제거하지 않음 (Quill 에디터용)
        # text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        
        # HTML 태그 내부의 공백만 정리 (태그 자체는 유지)
        text = re.sub(r'>\s+<', '><', text)  # 태그 사이 공백 제거
        text = re.sub(r'<p>\s+', '<p>', text)  # <p> 태그 내부 앞 공백 제거
        text = re.sub(r'\s+</p>', '</p>', text)  # </p> 태그 앞 공백 제거
        
        text = self.remove_old_app_name(text)
        text = self.format_answer_with_html_paragraphs(text)
        
        return text

    # 한국어 텍스트의 유효성을 검증하는 메서드
    def is_valid_korean_text(self, text: str) -> bool:
        if not text or len(text.strip()) < 3:
            return False
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'\s', '', text))
        
        if total_chars == 0:
            return False
            
        korean_ratio = korean_chars / total_chars
        
        if korean_ratio < 0.3:
            return False
        
        meaningless_patterns = [
            r'^[a-z\s\.,;:\(\)\[\]\-_&\/\'"]+$',
            r'^[A-Z\s\.,;:\(\)\[\]\-_&\/\'"]+$',
            r'^[\s\.,;:\(\)\[\]\-_&\/\'"]+$',
            r'^[0-9\s\.,;:\(\)\[\]\-_&\/\'"]+$',
            r'.*[а-я].*',
            r'.*[α-ω].*',
        ]
        
        for pattern in meaningless_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return False
        
        if re.search(r'(.)\1{5,}', text):
            return False
        
        random_pattern = r'[a-zA-Z]{8,}'
        if re.search(random_pattern, text) and korean_ratio < 0.5:
            return False
        
        return True

    # 생성된 텍스트를 정리하고 검증하는 메서드
    def clean_generated_text(self, text: str) -> str:
        if not text:
            return ""
        
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        text = re.sub(r'[\b\r\f\v]', '', text)
        
        text = re.sub(r'\b[a-z]{1,2}\b(?:\s+[a-z]{1,2}\b)*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[а-я]+', '', text)
        text = re.sub(r'[α-ω]+', '', text)
        
        text = re.sub(r'[^\w\s가-힣.,!?()"\'-]{3,}', '', text)
        text = re.sub(r'[.,;:!?]{3,}', '.', text)
        
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text

    # 통일된 GPT 프롬프트 생성 메서드 (모듈화)
    def get_gpt_prompts(self, query: str, context: str) -> tuple:
        """
        통일된 GPT 프롬프트를 생성하는 메서드
        
        Args:
            query (str): 고객 질문
            context (str): 참고 답변 컨텍스트
            
        Returns:
            tuple: (system_prompt, user_prompt)
        """
        system_prompt = """당신은 GOODTV 바이블 애플 고객센터 상담원입니다.

지침:
1. 제공된 참고 답변들의 스타일과 내용을 충실히 따라 작성하세요
2. 참고 답변에서 유사한 상황의 해결책을 찾아 적용하세요
3. 고객의 구체적 상황에 맞게 보완하되, 참고 답변의 톤과 스타일을 유지하세요

⚠️ 절대 금지사항:
- 존재하지 않는 기능이나 메뉴를 안내하지 마세요
- 구체적인 설정 방법이나 버튼 위치를 창작하지 마세요
- 참고 답변에 없는 기능은 "죄송하지만 현재 제공되지 않는 기능"으로 안내하세요
- 불확실한 경우 "내부적으로 검토하겠다"고 답변하세요

4. 기능 요청이나 개선 제안의 경우:
   - "좋은 의견 감사합니다"
   - "내부적으로 논의/검토하겠습니다"
   - "개선 사항으로 전달하겠습니다"
   위 표현들을 사용하세요

5. 고객은 반드시 '성도님'으로 호칭하세요
6. 앱 이름은 'GOODTV 바이블 애플' 또는 '바이블 애플'로 통일하세요
7. 인삿말 및 끝맺음말이 중복되서 표현되지 않도록 하세요
8. HTML 태그 사용 금지, 자연스러운 문장으로 작성하세요"""

        user_prompt = f"""고객 문의: {query}

참고 답변들:
{context}

위 참고 답변들의 해결 방식과 톤을 그대로 따라서 고객의 문제에 대한 구체적인 답변을 작성하세요."""

        return system_prompt, user_prompt

    # 향상된 GPT 생성 - 통일된 프롬프트 사용
    def generate_with_enhanced_gpt(self, query: str, similar_answers: list, context_analysis: dict) -> str:
        try:
            with memory_cleanup():
                approach = context_analysis['recommended_approach']
                context = self.create_enhanced_context(similar_answers)
                
                if not context:
                    logging.warning("유효한 컨텍스트가 없어 GPT 생성 중단")
                    return ""
                
                # 통일된 프롬프트 생성
                system_prompt, user_prompt = self.get_gpt_prompts(query, context)
                
                # 접근 방식별 temperature와 max_tokens 설정
                if approach == 'gpt_with_strong_context':
                    temperature = 0.2  # 매우 보수적
                    max_tokens = 600
                elif approach == 'gpt_with_weak_context':
                    temperature = 0.4  # 적당한 창의성
                    max_tokens = 650
                else:  # fallback이나 기타
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
                
                if not self.is_valid_korean_text(generated):
                    logging.warning(f"GPT가 무효한 텍스트 생성: {generated[:50]}...")
                    return ""
                
                logging.info(f"GPT 생성 성공 ({approach}): {len(generated)}자")
                return generated[:650]
                
        except Exception as e:
            logging.error(f"향상된 GPT 생성 실패: {e}")
            return ""

    # 최적의 폴백 답변 선택 메서드
    def get_best_fallback_answer(self, similar_answers: list) -> str:
        if not similar_answers:
            return ""
        
        # 점수와 텍스트 품질을 종합 평가
        best_answer = ""
        best_score = 0
        
        for ans in similar_answers[:5]:  # 상위 5개만 검토
            score = ans['score']
            answer_text = self.clean_generated_text(ans['answer'])
            
            if not self.is_valid_korean_text(answer_text):
                continue
            
            # 종합 점수 계산 (유사도 + 텍스트 길이 + 완성도)
            length_score = min(len(answer_text) / 200, 1.0)  # 200자 기준 정규화
            completeness_score = 1.0 if answer_text.endswith(('.', '!', '?')) else 0.8
            
            total_score = score * 0.7 + length_score * 0.2 + completeness_score * 0.1
            
            if total_score > best_score:
                best_score = total_score
                best_answer = answer_text
        
        return best_answer

    # 더 보수적인 GPT-3.5-turbo 생성 메서드 (기존 코드와의 호환성 유지)
    # 보수적이고 참고 답변에 충실한 GPT-3.5-turbo 텍스트 생성
    @profile
    def generate_with_gpt(self, query: str, similar_answers: list) -> str:
        try:
            with memory_cleanup():
                # 컨텍스트 준비 (더 많은 참고 답변 사용)
                context_answers = []
                for ans in similar_answers[:5]:  # 3개 → 5개로 늘려서 더 많은 참고
                    clean_ans = re.sub(r'[\b\r\f\v\x00-\x08\x0B\x0C\x0E-\x1F\x7F]|<[^>]+>', '', ans['answer'])
                    if self.is_valid_korean_text(clean_ans):
                        context_answers.append(clean_ans[:300])  # 200 → 300으로 늘림
                
                if not context_answers:
                    logging.warning("유효한 한국어 컨텍스트가 없어 GPT 생성 중단")
                    if similar_answers:
                        return self.clean_generated_text(similar_answers[0]['answer'])
                    return ""
                
                context = "\n\n---\n\n".join(context_answers)  # 구분자 명확히
                
                # ★ 더 제한적이고 보수적인 프롬프트
                system_prompt = """당신은 GOODTV 바이블 애플 고객센터 상담원입니다.

중요 규칙:
1. 제공된 참고 답변들과 거의 동일한 스타일과 내용으로 답변해주세요
2. 창의적인 답변보다는 참고 답변에 충실한 답변을 작성해주세요
3. 참고 답변의 톤, 문체, 표현을 최대한 따라하세요
4. 기술적 문제는 캡쳐나 영상을 요청하고 이메일(dev@goodtv.co.kr)로 문의하도록 안내하세요
5. 고객 호칭은 반드시 '성도님'으로만 사용하세요 (고객님 사용 금지)
6. HTML 태그나 마크다운 사용 금지, 일반 텍스트만 사용
7. 인사말과 끝맺음말은 제외하고 본문만 작성하세요"""

                user_prompt = f"""고객 질문: {query}

참고 답변들 (이와 유사하게 답변해주세요):
{context}

위 참고 답변들의 스타일과 톤을 그대로 따라서, 고객의 질문에 적절한 답변을 작성해주세요. 
창의적인 답변보다는 참고 답변과 유사한 답변을 작성하는 것이 중요합니다.
고객은 반드시 '성도님'으로 호칭해주세요."""

                # ★ 더 보수적인 API 설정
                response = self.openai_client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,  # 0.3으로 낮춤
                    top_p=0.7,  # 0.9 → 0.7로 낮춤
                    frequency_penalty=0.2,  # 0.1 → 0.2로 높임
                    presence_penalty=0.2   # 0.1 → 0.2로 높임
                )
                
                generated = response.choices[0].message.content.strip()
                
                # 메모리 해제
                del response
                
                # 생성된 텍스트 정리 및 검증
                generated = self.clean_generated_text(generated)
                
                # 한국어 텍스트 검증
                if not self.is_valid_korean_text(generated):
                    logging.warning("GPT가 무효한 텍스트를 생성했습니다. 폴백 사용.")
                    if similar_answers:
                        fallback = self.clean_generated_text(similar_answers[0]['answer'])
                        if self.is_valid_korean_text(fallback):
                            return fallback[:600]
                    return ""
                
                return generated[:600]
                
        except Exception as e:
            logging.error(f"GPT 모델 생성 실패: {e}")
            # 폴백: 첫 번째 유사 답변 반환
            if similar_answers:
                fallback = self.clean_generated_text(similar_answers[0]['answer'])
                if self.is_valid_korean_text(fallback):
                    return fallback[:600]
            return ""

    # 개선된 AI 답변 생성 메인 메서드
    def generate_ai_answer(self, query: str, similar_answers: list, lang: str) -> str:
        if not similar_answers:
            default_msg = "<p>문의해주신 내용에 대해 정확한 답변을 드리기 위해 더 자세한 정보가 필요합니다.</p><p><br></p><p>고객센터로 문의해주시면 신속하게 도움을 드리겠습니다.</p>"
            return default_msg
        
        # 1. 컨텍스트 분석
        context_analysis = self.analyze_context_quality(similar_answers, query)
        
        if not context_analysis['has_good_context']:
            logging.warning("유용한 컨텍스트가 없어 기본 메시지 반환")
            return "<p>문의해주신 내용에 대해 정확한 답변을 드리기 위해 더 자세한 정보가 필요합니다.</p><p><br></p><p>고객센터로 문의해주시면 신속하게 도움을 드리겠습니다.</p>"
        
        try:
            approach = context_analysis['recommended_approach']
            logging.info(f"선택된 접근 방식: {approach}")
            
            # 2. 접근 방식에 따른 답변 생성
            if approach == 'direct_use':
                # 직접 사용 - 최고 점수 답변 활용
                base_answer = self.get_best_fallback_answer(similar_answers[:3])
                logging.info("높은 유사도로 직접 사용")
                
            elif approach in ['gpt_with_strong_context', 'gpt_with_weak_context']:
                # GPT 생성
                base_answer = self.generate_with_enhanced_gpt(query, similar_answers, context_analysis)
                
                # GPT 실패 시 폴백
                if not base_answer or not self.is_valid_korean_text(base_answer):
                    logging.warning("GPT 생성 실패, 폴백 답변 사용")
                    base_answer = self.get_best_fallback_answer(similar_answers)
                    
            else:
                # 폴백
                base_answer = self.get_best_fallback_answer(similar_answers)
            
            # 3. 최종 검증 및 폴백
            if not base_answer or not self.is_valid_korean_text(base_answer):
                logging.error("모든 답변 생성 방법 실패")
                return "<p>문의해주신 내용에 대해 정확한 답변을 드리기 위해 더 자세한 정보가 필요합니다.</p><p><br></p><p>고객센터로 문의해주시면 신속하게 도움을 드리겠습니다.</p>"
            
            # 4. 최종 포맷팅 (Quill 에디터용 HTML 형식 유지)
            # 앱 이름 정리 및 고객님 → 성도님 변경
            base_answer = self.remove_old_app_name(base_answer)
            base_answer = re.sub(r'고객님', '성도님', base_answer)
            
            # 기존 인사말/끝맺음말 제거 (일반 텍스트에서)
            # 인사말 제거
            base_answer = re.sub(r'^안녕하세요[^.]*\.\s*', '', base_answer, flags=re.IGNORECASE)
            base_answer = re.sub(r'^GOODTV\s+바이블\s*애플[^.]*\.\s*', '', base_answer, flags=re.IGNORECASE)
            base_answer = re.sub(r'^바이블\s*애플[^.]*\.\s*', '', base_answer, flags=re.IGNORECASE)
            base_answer = re.sub(r'고객센터[^.]*\.\s*', '', base_answer, flags=re.IGNORECASE)
            
            # 끝맺음말 제거 (더 강화된 패턴)
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
            
            # 본문을 HTML 단락 형식으로 포맷팅
            formatted_body = self.format_answer_with_html_paragraphs(base_answer.strip())
            
            # 고정된 인사말 (HTML 형식으로)
            final_answer = "<p>안녕하세요. GOODTV 바이블 애플입니다.</p><p><br></p><p>바이블 애플을 이용해주셔서 감사드립니다.</p><p><br></p>"
            
            # 포맷팅된 본문 추가
            final_answer += formatted_body
            
            # 고정된 끝맺음말 (HTML 형식으로)
            final_answer += "<p><br></p><p>항상 성도님께 좋은 성경앱을 제공하기 위해 노력하는 바이블 애플이 되겠습니다.</p><p><br></p><p>감사합니다. 주님 안에서 평안하세요.</p>"
            
            logging.info(f"최종 답변 생성 완료: {len(final_answer)}자, 접근방식: {approach}")
            return final_answer
            
        except Exception as e:
            logging.error(f"답변 생성 중 오류: {e}")
            return "<p>죄송합니다. 현재 답변을 생성할 수 없습니다.</p><p><br></p><p>고객센터로 문의해주세요.</p>"

    # 메모리 최적화된 메인 처리 메서드
    def process(self, seq: int, question: str, lang: str) -> dict:
        try:
            with memory_cleanup():
                processed_question = self.preprocess_text(question)
                if not processed_question:
                    return {"success": False, "error": "질문이 비어있습니다."}
                
                logging.info(f"처리 시작 - SEQ: {seq}, 질문: {processed_question[:50]}...")
                
                # 유사 답변 검색
                similar_answers = self.search_similar_answers(processed_question)
                
                # AI 답변 생성
                ai_answer = self.generate_ai_answer(processed_question, similar_answers, lang)
                
                # 특수문자 정리
                ai_answer = ai_answer.replace('"', '"').replace('"', '"')
                ai_answer = ai_answer.replace(''', "'").replace(''', "'")
                
                result = {
                    "success": True,
                    "answer": ai_answer,
                    "similar_count": len(similar_answers),
                    "embedding_model": "text-embedding-3-small",
                    "generation_model": "gpt-3.5-turbo"
                }
                
                logging.info(f"처리 완료 - SEQ: {seq}, HTML 답변 생성됨")
                return result
                
        except Exception as e:
            logging.error(f"처리 중 오류 - SEQ: {seq}, 오류: {str(e)}")
            return {"success": False, "error": str(e)}

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
    
    # AI를 이용한 한국어 오타 수정 메서드
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
                del response
                
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
        
    # 텍스트 전처리 메서드
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
    
    # OpenAI로 임베딩 생성하는 메서드
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
    
    # 카테고리 인덱스를 이름으로 변환하는 메서드
    def get_category_name(self, cate_idx: str) -> str:
        return CATEGORY_MAPPING.get(str(cate_idx), '사용 문의(기타)')
    
    # MSSQL에서 데이터 조회하는 메서드
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
    
    # MSSQL 데이터를 Pinecone에 동기화하는 메서드
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
                question = self.fix_korean_typos_with_ai(raw_question)  # AI 오타 수정 적용
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
                    "question": question,  # 오타 수정된 질문 사용
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

# AI 답변 생성 API 엔드포인트 (메인 기능)
#
# ASP Classic에서 호출하는 주요 API로, 고객 질문에 대한 AI 답변을 생성
#
# 요청 형식 (JSON POST):
# {
#     "seq": 123,           # 문의 시퀀스 번호 (선택)
#     "question": "앱이 안되요",  # 고객 질문 (필수)
#     "lang": "kr"          # 언어 (선택, 기본값: kr)
# }
#
# 응답 형식:
# {
#     "success": true,
#     "answer": "<p>안녕하세요...</p>",  # HTML 형식 답변
#     "similar_count": 5,               # 검색된 유사 답변 개수
#     "embedding_model": "text-embedding-3-small",
#     "generation_model": "gpt-3.5-turbo"
# }
#
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
            lang = data.get('lang', 'kr')
            
            if not question:
                return jsonify({"success": False, "error": "질문이 필요합니다."}), 400
            
            result = generator.process(seq, question, lang)
            
            response = jsonify(result)
            response.headers['Content-Type'] = 'application/json; charset=utf-8'

            # ★ 메모리 사용량 모니터링
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            memory_usage = sum(stat.size for stat in top_stats) / 1024 / 1024  # MB
            logging.info(f"현재 메모리 사용량: {memory_usage:.2f}MB")
            
            if memory_usage > 500:  # 500MB 초과시 경고
                logging.warning(f"높은 메모리 사용량 감지: {memory_usage:.2f}MB")
                gc.collect()

            return response
        
    except Exception as e:
        logging.error(f"API 호출 오류: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

# MSSQL 데이터를 Pinecone에 동기화하는 API 엔드포인트
#
# 운영 시스템에서 새로운 Q&A 데이터가 생성되거나 수정될 때 호출
#
# 요청 형식 (JSON POST):
# {
#     "seq": 12345,        # MSSQL의 문의 시퀀스 번호 (필수)
#     "mode": "upsert"     # 동작 모드: "upsert"(생성/수정) 또는 "delete"(삭제)
# }
#
# 응답 형식:
# {
#     "success": true,
#     "message": "Pinecone 생성 완료",
#     "seq": 12345,
#     "vector_id": "qa_bible_12345",
#     "is_update": false
# }
#
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

# 시스템 상태 확인을 위한 헬스체크 API 엔드포인트
#
# 로드밸런서나 모니터링 시스템에서 호출하여 서비스 상태 확인
#
# 요청: GET /health
#
# 응답 형식 (정상):
# {
#     "status": "healthy",
#     "pinecone_vectors": 1500,
#     "timestamp": "2024-01-01T12:00:00",
#     "services": {
#         "ai_answer": "active",
#         "pinecone_sync": "active"
#     }
# }
#
# 응답 형식 (오류):
# {
#     "status": "unhealthy",
#     "error": "Pinecone connection failed"
# }
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
                "pinecone_sync": "active"
            }
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

# ==================================================
# 11. 메인 실행 부분 (절차적 프로그래밍)
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