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

    def search_similar_answers(self, query: str, top_k: int = 15, similarity_threshold: float = 0.4) -> list:
        """
        개선된 유사 답변 검색 - 더 낮은 임계값으로 더 많은 후보 확보
        """
        try:
            with memory_cleanup():
                query_vector = self.create_embedding(query)
                if query_vector is None:
                    return []
                
                results = index.query(
                    vector=query_vector, 
                    top_k=top_k,  # 10 → 15로 증가
                    include_metadata=True
                )
                
                filtered_results = []
                for i, match in enumerate(results['matches']):
                    score = match['score']
                    question = match['metadata'].get('question', '')
                    answer = match['metadata'].get('answer', '')
                    category = match['metadata'].get('category', '일반')
                    
                    # 유사도 0.4 이상이면 포함 (기존 0.6 → 0.4로 완화)
                    if score >= similarity_threshold:
                        filtered_results.append({
                            'score': score,
                            'question': question,
                            'answer': answer,
                            'category': category,
                            'rank': i + 1
                        })
                        
                        # 상세 로깅 추가
                        logging.info(f"유사 답변 #{i+1}: 점수={score:.3f}, 카테고리={category}")
                        logging.info(f"참고 질문: {question[:50]}...")
                        logging.info(f"참고 답변: {answer[:100]}...")
                
                del results
                del query_vector
                
                logging.info(f"총 {len(filtered_results)}개의 유사 답변 검색 완료")
                return filtered_results
                
        except Exception as e:
            logging.error(f"Pinecone 검색 실패: {str(e)}")
            return []

    def analyze_context_quality(self, similar_answers: list, query: str) -> dict:
        """
        컨텍스트 품질 분석 - 참고 답변의 활용 가능성 평가
        """
        if not similar_answers:
            return {
                'has_good_context': False,
                'best_score': 0.0,
                'recommended_approach': 'fallback',
                'context_summary': '유사 답변이 없습니다.'
            }
        
        best_score = similar_answers[0]['score']
        high_quality_count = len([ans for ans in similar_answers if ans['score'] >= 0.7])
        medium_quality_count = len([ans for ans in similar_answers if 0.5 <= ans['score'] < 0.7])
        
        # 카테고리 분포 분석
        categories = [ans['category'] for ans in similar_answers[:5]]
        category_distribution = {cat: categories.count(cat) for cat in set(categories)}
        
        # 접근 방식 결정
        if best_score >= 0.8:
            approach = 'direct_use'  # 직접 사용
        elif best_score >= 0.6 or high_quality_count >= 2:
            approach = 'gpt_with_strong_context'  # 강한 컨텍스트로 GPT 사용
        elif best_score >= 0.4 or medium_quality_count >= 3:
            approach = 'gpt_with_weak_context'  # 약한 컨텍스트로 GPT 사용
        else:
            approach = 'fallback'  # 폴백 사용
        
        analysis = {
            'has_good_context': best_score >= 0.4,
            'best_score': best_score,
            'high_quality_count': high_quality_count,
            'medium_quality_count': medium_quality_count,
            'category_distribution': category_distribution,
            'recommended_approach': approach,
            'context_summary': f"최고점수: {best_score:.3f}, 고품질: {high_quality_count}개, 중품질: {medium_quality_count}개"
        }
        
        logging.info(f"컨텍스트 분석 결과: {analysis}")
        return analysis

    def create_enhanced_context(self, similar_answers: list, max_answers: int = 7) -> str:
        """
        향상된 컨텍스트 생성 - 다양한 품질의 답변을 조합
        """
        if not similar_answers:
            return ""
        
        context_parts = []
        used_answers = 0
        
        # 점수별로 그룹핑
        high_score = [ans for ans in similar_answers if ans['score'] >= 0.7]
        medium_score = [ans for ans in similar_answers if 0.5 <= ans['score'] < 0.7]
        low_score = [ans for ans in similar_answers if 0.4 <= ans['score'] < 0.5]
        
        # 고품질 답변 우선 포함
        for ans in high_score[:4]:
            if used_answers >= max_answers:
                break
            clean_answer = re.sub(r'[\b\r\f\v\x00-\x08\x0B\x0C\x0E-\x1F\x7F]|<[^>]+>', '', ans['answer'])
            if self.is_valid_korean_text(clean_answer) and len(clean_answer.strip()) > 20:
                context_parts.append(f"[참고답변 {used_answers+1} - 점수: {ans['score']:.2f}]\n{clean_answer[:400]}")
                used_answers += 1
        
        # 중품질 답변 보완
        for ans in medium_score[:3]:
            if used_answers >= max_answers:
                break
            clean_answer = re.sub(r'[\b\r\f\v\x00-\x08\x0B\x0C\x0E-\x1F\x7F]|<[^>]+>', '', ans['answer'])
            if self.is_valid_korean_text(clean_answer) and len(clean_answer.strip()) > 20:
                context_parts.append(f"[참고답변 {used_answers+1} - 점수: {ans['score']:.2f}]\n{clean_answer[:300]}")
                used_answers += 1
        
        # 저품질 답변도 필요시 포함
        if used_answers < 3:  # 너무 적으면 저품질도 포함
            for ans in low_score[:2]:
                if used_answers >= max_answers:
                    break
                clean_answer = re.sub(r'[\b\r\f\v\x00-\x08\x0B\x0C\x0E-\x1F\x7F]|<[^>]+>', '', ans['answer'])
                if self.is_valid_korean_text(clean_answer) and len(clean_answer.strip()) > 20:
                    context_parts.append(f"[참고답변 {used_answers+1} - 점수: {ans['score']:.2f}]\n{clean_answer[:250]}")
                    used_answers += 1
        
        logging.info(f"컨텍스트 생성: {used_answers}개의 답변 포함")
        return "\n\n" + "="*50 + "\n\n".join(context_parts)

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
    def generate_with_gpt(self, query: str, similar_answers: list, context_analysis: dict) -> str:
        """
        향상된 GPT 생성 - 컨텍스트 품질에 따른 차별화된 프롬프트
        """
        try:
            with memory_cleanup():
                approach = context_analysis['recommended_approach']
                context = self.create_enhanced_context(similar_answers)
                
                if not context:
                    logging.warning("유효한 컨텍스트가 없어 GPT 생성 중단")
                    return ""
                
                # 접근 방식별 차별화된 프롬프트
                if approach == 'gpt_with_strong_context':
                    # 강한 컨텍스트 - 참고 답변 충실히 따라하기
                    system_prompt = """당신은 GOODTV 바이블 애플 고객센터 상담원입니다.

강력한 지침:
1. 제공된 참고 답변들의 스타일과 내용을 충실히 따라 작성하세요
2. 참고 답변에서 유사한 상황의 해결책을 찾아 적용하세요
3. 기술적 문제는 구체적인 해결 방법을 제시하되, 복잡한 경우 캡쳐/영상을 요청하세요
4. 고객은 반드시 '성도님'으로 호칭하세요
5. 앱 이름은 'GOODTV 바이블 애플' 또는 '바이블 애플'로 통일하세요
6. HTML 태그 사용 금지, 자연스러운 문장으로 작성하세요"""

                    user_prompt = f"""고객 문의: {query}

참고 답변들 (높은 유사도 - 이 답변들과 매우 유사하게 작성하세요):
{context}

위 참고 답변들의 해결 방식과 톤을 그대로 따라서 고객의 문제에 대한 구체적인 답변을 작성하세요."""

                    temperature = 0.2  # 매우 보수적
                    max_tokens = 400

                elif approach == 'gpt_with_weak_context':
                    # 약한 컨텍스트 - 참고하되 보완 필요
                    system_prompt = """당신은 GOODTV 바이블 애플 고객센터 상담원입니다.

지침:
1. 참고 답변들을 바탕으로 하되, 고객의 구체적 상황에 맞게 보완하세요
2. 기술적 문제 해결을 위한 구체적 단계를 제시하세요
3. 해결되지 않을 경우의 대안도 제시하세요
4. 이메일(dev@goodtv.co.kr) 문의나 고객센터 안내도 포함하세요
5. '성도님' 호칭 사용, 친근하고 도움이 되는 톤으로 작성하세요"""

                    user_prompt = f"""고객 문의: {query}

참고 답변들 (중간 유사도 - 참고하되 고객 상황에 맞게 보완하세요):
{context}

위 참고 답변들을 참고하여, 고객의 구체적인 문제 상황에 맞는 실용적인 해결책을 제시해주세요."""

                    temperature = 0.4  # 적당한 창의성
                    max_tokens = 450

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
                return generated[:450]
                
        except Exception as e:
            logging.error(f"향상된 GPT 생성 실패: {e}")
            return ""
    
    def get_best_fallback_answer(self, similar_answers: list) -> str:
        """
        최적의 폴백 답변 선택
        """
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

    def generate_ai_answer(self, query: str, similar_answers: list, lang: str) -> str:
        """
        개선된 AI 답변 생성 메인 함수
        """
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
                base_answer = self.generate_with_gpt(query, similar_answers, context_analysis)
                
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
            
            # 4. 최종 포맷팅
            # HTML 태그 제거 및 앱 이름 정리
            base_answer = re.sub(r'<[^>]+>', '', base_answer)
            base_answer = self.remove_old_app_name(base_answer)
            base_answer = re.sub(r'고객님', '성도님', base_answer)
            
            # 고정된 인사말
            final_answer = "안녕하세요. GOODTV 바이블 애플입니다. 바이블 애플을 애용해 주셔서 감사합니다. "
            
            # 기존 인사말/끝맺음말 제거
            base_answer = re.sub(r'^안녕하세요[^.]*\.\s*', '', base_answer)
            base_answer = re.sub(r'\s*감사합니다[^.]*\.\s*$', '', base_answer)
            base_answer = re.sub(r'\s*평안하세요[^.]*\.\s*$', '', base_answer)
            
            final_answer += base_answer.strip()
            
            # 마침표 추가
            if final_answer and not final_answer.endswith(('.', '!', '?')):
                final_answer += "."
            
            # 고정된 끝맺음말
            final_answer += " 항상 성도님께 좋은 성경앱을 제공하기 위해 노력하는 바이블 애플이 되겠습니다. 감사합니다. 주님 안에서 평안하세요."
            
            # HTML 포맷팅
            final_answer = self.clean_answer_text(final_answer)
            
            logging.info(f"최종 답변 생성 완료: {len(final_answer)}자, 접근방식: {approach}")
            return final_answer
            
        except Exception as e:
            logging.error(f"답변 생성 중 오류: {e}")
            return "<p>죄송합니다. 현재 답변을 생성할 수 없습니다.</p><p><br></p><p>고객센터로 문의해주세요.</p>"
        
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