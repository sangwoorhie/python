#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI Answer Generator Flask API for ASP Classic Integration
파일명: free_4_ai_answer_generator.py
설명: Flask API로 ASP Classic에서 호출 (Pinecone 동기화 기능 통합)
모델: gpt-3.5-turbo + OpenAI embeddings
"""

# 필수 라이브러리 임포트
import os
import sys
import json
import json as json_module
import re
import html
import unicodedata
import logging
import gc
from flask import Flask, request, jsonify
from pinecone import Pinecone
from dotenv import load_dotenv
from flask_cors import CORS
import openai
import pyodbc
from datetime import datetime
from typing import Optional, Dict, Any, List
from memory_profiler import profile
import tracemalloc
import threading
from contextlib import contextmanager

# Python 내장 모듈로 메모리 누수 추적
tracemalloc.start() 

# Flask 웹 애플리케이션 인스턴스 생성
app = Flask(__name__)
CORS(app)

# 로깅 시스템 설정 - 파일에 로그 저장
logging.basicConfig(
    filename='/home/ec2-user/python/logs/ai_generator.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# .env 파일에서 환경변수 로드
load_dotenv()

# ====== 설정 상수 ======
MODEL_NAME = 'text-embedding-3-small'
INDEX_NAME = "bible-app-support-1536-openai"
EMBEDDING_DIMENSION = 1536
MAX_TEXT_LENGTH = 8000

# ★ GPT 모델 설정 (더 보수적으로 변경)
GPT_MODEL = 'gpt-3.5-turbo'
MAX_TOKENS = 350  # 400 → 350으로 줄임
TEMPERATURE = 0.3  # 0.7 → 0.3으로 대폭 줄임 (창의성 억제)

# 카테고리 매핑 (cate_idx → 카테고리명)
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

@contextmanager
def memory_cleanup():
    """컨텍스트 매니저로 메모리 정리"""
    try:
        yield
    finally:
        gc.collect()

# AI 모델 및 벡터 데이터베이스 초기화
try:
    # Pinecone 벡터 데이터베이스 연결
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index(INDEX_NAME)
    
    # OpenAI 클라이언트 초기화
    openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # MSSQL 연결 문자열 (Pinecone 동기화용)
    mssql_config = {
        'server': os.getenv('MSSQL_SERVER'),
        'database': os.getenv('MSSQL_DATABASE'),
        'username': os.getenv('MSSQL_USERNAME'),
        'password': os.getenv('MSSQL_PASSWORD')
    }
    
    connection_string = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={mssql_config['server']},1433;"
            f"DATABASE={mssql_config['database']};"
            f"UID={mssql_config['username']};"
            f"PWD={mssql_config['password']};"
            f"TrustServerCertificate=yes;"
            f"Connection Timeout=30;"
    )

except Exception as e:
    logging.error(f"모듈 로드 실패: {str(e)}")
    app.logger.error(f"모듈 로드 실패: {str(e)}")
    raise

# ====== AI 답변 생성 클래스 (보수적 GPT-3.5-turbo 버전) ======
class AIAnswerGenerator:
    
    def __init__(self):
        self.openai_client = openai_client
    
    def preprocess_text(self, text: str) -> str:
        if not text:
            return ""
        
        text = str(text)
        text = html.unescape(text)
        
        # HTML 태그 제거
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</p>', '\n\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<p[^>]*>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<li[^>]*>', '\n• ', text, flags=re.IGNORECASE)
        text = re.sub(r'</li>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<[^>]+>', '', text)
        
        # 공백 정리
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)   
        text = text.strip()
        
        return text

    def escape_json_string(self, text: str) -> str:
        if not text:
            return ""
        escaped = json_module.dumps(text, ensure_ascii=False)
        return escaped[1:-1]

    def create_embedding(self, text: str) -> Optional[list]:
        """메모리 최적화된 임베딩 생성"""
        if not text or not text.strip():
            return None
            
        try:
            with memory_cleanup():
                response = self.openai_client.embeddings.create(
                    model='text-embedding-3-small',
                    input=text[:8000]
                )
                
                embedding = response.data[0].embedding.copy()
                del response
                return embedding
                
        except Exception as e:
            logging.error(f"임베딩 생성 실패: {e}")
            return None

    def search_similar_answers(self, query: str, top_k: int = 10, similarity_threshold: float = 0.6) -> list:
        """메모리 최적화된 유사 답변 검색"""
        try:
            with memory_cleanup():
                query_vector = self.create_embedding(query)
                if query_vector is None:
                    return []
                
                results = index.query(
                    vector=query_vector, 
                    top_k=top_k, 
                    include_metadata=True
                )
                
                filtered_results = [
                    {
                        'score': match['score'],
                        'question': match['metadata'].get('question', ''),
                        'answer': match['metadata'].get('answer', ''),
                        'category': match['metadata'].get('category', '일반')
                    }
                    for match in results['matches'] if match['score'] >= similarity_threshold
                ]
                
                del results
                del query_vector
                
                logging.info(f"유사 답변 {len(filtered_results)}개 검색 완료")
                return filtered_results
                
        except Exception as e:
            logging.error(f"Pinecone 검색 실패: {str(e)}")
            return []

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

    def format_answer_with_html_paragraphs(self, text: str) -> str:
        if not text:
            return ""
        
        text = self.remove_old_app_name(text)
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        paragraphs = []
        current_paragraph = []
        
        paragraph_triggers = [
            '해당', '이', '만약', '혹시', '성도님', '고객님',
            '번거로우시', '불편하시', '죄송하지만', '참고로',
            '항상', '늘', '앞으로도', '지속적으로',
            '스피커', '버튼', '메뉴', '화면', '설정',
        ]
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if i == 0 and any(greeting in sentence for greeting in ['안녕하세요', '안녕']):
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                paragraphs.append(sentence)
                continue
            
            should_break = False
            
            for trigger in paragraph_triggers:
                if sentence.startswith(trigger):
                    should_break = True
                    break
            
            if current_paragraph and len(current_paragraph) >= 2:
                should_break = True
            
            if any(closing in sentence for closing in ['감사합니다', '감사드립니다', '평안하세요']):
                should_break = True
            
            if should_break and current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = [sentence]
            else:
                current_paragraph.append(sentence)
        
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        html_paragraphs = []
        for i, paragraph in enumerate(paragraphs):
            html_paragraphs.append(f"<p>{paragraph}</p>")
            
            if i < len(paragraphs) - 1:
                if not any(keyword in paragraph for keyword in ['감사합니다', '감사드립니다', '평안하세요']):
                    html_paragraphs.append("<p><br></p>")
        
        return ''.join(html_paragraphs)

    def clean_answer_text(self, text: str) -> str:
        if not text:
            return ""
        
        text = re.sub(r'[\b\r\f\v]', '', text)
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\s+([,.!?])', r'\1', text)
        text = re.sub(r'([,.!?])\s+', r'\1 ', text)
        
        text = self.remove_old_app_name(text)
        text = self.format_answer_with_html_paragraphs(text)
        
        return text

    def is_valid_korean_text(self, text: str) -> bool:
        """한국어 텍스트의 유효성을 검증하는 함수"""
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

    def clean_generated_text(self, text: str) -> str:
        """생성된 텍스트를 정리하고 검증하는 함수"""
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

    # ★ 더 보수적인 GPT-3.5-turbo 생성 함수
    @profile
    def generate_with_gpt(self, query: str, similar_answers: list) -> str:
        """보수적이고 참고 답변에 충실한 GPT-3.5-turbo 텍스트 생성"""
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
                            return fallback[:350]
                    return ""
                
                return generated[:350]
                
        except Exception as e:
            logging.error(f"GPT 모델 생성 실패: {e}")
            # 폴백: 첫 번째 유사 답변 반환
            if similar_answers:
                fallback = self.clean_generated_text(similar_answers[0]['answer'])
                if self.is_valid_korean_text(fallback):
                    return fallback[:350]
            return ""

    def generate_ai_answer(self, query: str, similar_answers: list, lang: str) -> str:
        if not similar_answers:
            default_msg = "<p>문의해주신 내용에 대해 정확한 답변을 드리기 위해 더 자세한 정보가 필요합니다.</p><p><br></p><p>고객센터로 문의해주시면 신속하게 도움을 드리겠습니다.</p>"
            return default_msg
        
        try:
            # ★ 유사도 임계값을 높여서 더 보수적으로 GPT 사용
            if similar_answers[0]['score'] > 0.75:  # 0.85 → 0.75로 낮춤 (GPT 더 자주 사용 안함)
                base_answer = similar_answers[0]['answer']
                logging.info("높은 유사도로 인해 GPT 생성 생략")
                
                base_answer = self.clean_generated_text(base_answer)
                if not self.is_valid_korean_text(base_answer):
                    logging.warning("유사 답변이 무효한 텍스트입니다.")
                    for ans in similar_answers[1:3]:
                        candidate = self.clean_generated_text(ans['answer'])
                        if self.is_valid_korean_text(candidate):
                            base_answer = candidate
                            break
                    else:
                        return "<p>문의해주신 내용에 대해 정확한 답변을 드리기 위해 더 자세한 정보가 필요합니다.</p><p><br></p><p>고객센터로 문의해주시면 신속하게 도움을 드리겠습니다.</p>"
            else:
                # 유사도가 낮을 때만 GPT 사용
                base_answer = self.generate_with_gpt(query, similar_answers)
                
                # GPT 결과가 무효하면 폴백
                if not base_answer or not self.is_valid_korean_text(base_answer):
                    logging.warning("GPT 생성 결과가 무효합니다. 유사 답변 사용.")
                    for ans in similar_answers[:3]:
                        candidate = self.clean_generated_text(ans['answer'])
                        if self.is_valid_korean_text(candidate):
                            base_answer = candidate
                            break
                    else:
                        return "<p>문의해주신 내용에 대해 정확한 답변을 드리기 위해 더 자세한 정보가 필요합니다.</p><p><br></p><p>고객센터로 문의해주시면 신속하게 도움을 드리겠습니다.</p>"
            
            # HTML 태그 제거 및 앱 이름 정리
            base_answer = re.sub(r'<[^>]+>', '', base_answer)
            base_answer = self.remove_old_app_name(base_answer)
            
            # ★ '고객님' → '성도님'으로 통일
            base_answer = re.sub(r'고객님', '성도님', base_answer)
            
            # 최종 검증
            if not self.is_valid_korean_text(base_answer):
                logging.error("최종 답변이 무효한 텍스트입니다.")
                return "<p>죄송합니다. 현재 답변을 생성할 수 없습니다.</p><p><br></p><p>고객센터로 문의해주세요.</p>"
            
            # ★ 고정된 인사말
            final_answer = "안녕하세요. GOODTV 바이블 애플입니다. 바이블 애플을 애용해 주셔서 감사합니다. "
            
            # ★ 기존 인사말 완전 제거 (더 포괄적으로)
            base_answer = re.sub(r'^안녕하세요[^.]*바이블\s*애플[^.]*\.\s*', '', base_answer)
            base_answer = re.sub(r'^안녕하세요[^.]*GOODTV[^.]*\.\s*', '', base_answer)
            base_answer = re.sub(r'^안녕하세요[^.]*\.\s*', '', base_answer)
            base_answer = re.sub(r'^안녕[^.]*\.\s*', '', base_answer)
            base_answer = re.sub(r'^바이블\s*애플[^.]*\.\s*', '', base_answer)
            base_answer = re.sub(r'^GOODTV[^.]*\.\s*', '', base_answer)
            
            # ★ 기존 끝맺음말 완전 제거 (더 포괄적으로)
            base_answer = re.sub(r'\s*항상[^.]*바이블\s*애플[^.]*\.\s*', '', base_answer)
            base_answer = re.sub(r'\s*항상[^.]*성경앱[^.]*\.\s*', '', base_answer)
            base_answer = re.sub(r'\s*항상[^.]*평안하세요[^.]*\.\s*', '', base_answer)
            base_answer = re.sub(r'\s*감사합니다[^.]*평안하세요[^.]*\.\s*', '', base_answer)
            base_answer = re.sub(r'\s*감사합니다[^.]*\.\s*$', '', base_answer)
            base_answer = re.sub(r'\s*주님\s*안에서[^.]*\.\s*$', '', base_answer)
            base_answer = re.sub(r'\s*평안하세요[^.]*\.\s*$', '', base_answer)
            base_answer = re.sub(r'\s*함께[^.]*\.\s*$', '', base_answer)  # "함께." 제거
            
            # ★ 중복된 감사합니다 제거
            base_answer = re.sub(r'(\s*감사합니다[^.]*\.\s*){2,}', ' 감사합니다. ', base_answer)
            
            final_answer += base_answer.strip()
            
            # 마지막 문장이 완전하지 않으면 마침표 추가
            if final_answer and not final_answer.endswith(('.', '!', '?')):
                final_answer += "."
            
            # ★ 고정된 끝맺음말 (중복 방지)
            final_answer += " 항상 성도님께 좋은 성경앱을 제공하기 위해 노력하는 바이블 애플이 되겠습니다. 감사합니다. 주님 안에서 평안하세요."
            
            final_answer = self.clean_answer_text(final_answer)
            
            # 최종 검증
            if not self.is_valid_korean_text(final_answer):
                logging.error("최종 포맷된 답변이 무효합니다.")
                return "<p>죄송합니다. 현재 답변을 생성할 수 없습니다.</p><p><br></p><p>고객센터로 문의해주세요.</p>"
            
            return final_answer
            
        except Exception as e:
            logging.error(f"답변 생성 실패: {e}")
            return "<p>죄송합니다. 현재 답변을 생성할 수 없습니다.</p><p><br></p><p>고객센터로 문의해주세요.</p>"

    def process(self, seq: int, question: str, lang: str) -> dict:
        """메모리 최적화된 메인 처리 함수"""
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

# ====== Pinecone 동기화 클래스는 그대로 유지 ======
class PineconeSyncManager:
    """MSSQL 데이터를 Pinecone에 동기화하는 클래스"""
    
    def __init__(self):
        self.index = index
        self.openai_client = openai_client
        
    def preprocess_text(self, text: str, for_metadata: bool = False) -> str:
        """텍스트 전처리"""
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
    
    def create_embedding(self, text: str) -> Optional[list]:
        """OpenAI로 임베딩 생성"""
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
    
    def get_category_name(self, cate_idx: str) -> str:
        """카테고리 인덱스를 이름으로 변환"""
        return CATEGORY_MAPPING.get(str(cate_idx), '사용 문의(기타)')
    
    def get_mssql_data(self, seq: int) -> Optional[Dict]:
        """MSSQL에서 데이터 조회"""
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
    
    def sync_to_pinecone(self, seq: int, mode: str = 'upsert') -> Dict[str, Any]:
        """MSSQL 데이터를 Pinecone에 동기화"""
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
                
                # 텍스트 전처리
                question = self.preprocess_text(data['contents'])
                answer = self.preprocess_text(data['reply_contents'])
                
                # 임베딩 생성 (질문 기반)
                embedding = self.create_embedding(question)
                if not embedding:
                    return {"success": False, "error": "임베딩 생성 실패"}
                
                # 카테고리 이름 가져오기
                category = self.get_category_name(data['cate_idx'])
                
                # 메타데이터 구성
                metadata = {
                    "seq": int(data['seq']),
                    "question": self.preprocess_text(data['contents'], for_metadata=True),
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

# 싱글톤 인스턴스
generator = AIAnswerGenerator()
sync_manager = PineconeSyncManager()

# ====== Flask API 엔드포인트 ======

@app.route('/generate_answer', methods=['POST'])
def generate_answer():
    """AI 답변 생성 API (메모리 최적화)"""
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

@app.route('/sync_to_pinecone', methods=['POST'])
def sync_to_pinecone():
    """MSSQL 데이터를 Pinecone에 동기화하는 API"""
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

@app.route('/health', methods=['GET'])
def health_check():
    """헬스체크 엔드포인트"""
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

# 메인 실행 부분
if __name__ == "__main__":
    port = int(os.getenv('FLASK_PORT', 8000))
    
    print(f"Flask API starting on port {port}")
    print("Services: AI Answer Generation + Pinecone Sync")
    print(f"AI Model: {GPT_MODEL} (Conservative Mode)")
    
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)