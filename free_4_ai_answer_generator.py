#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI Answer Generator Flask API for ASP Classic Integration
파일명: free_4_ai_answer_generator.py
설명: Flask API로 ASP Classic에서 호출 (Pinecone 동기화 기능 통합)
모델: google/flan-t5-base + OpenAI embeddings
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
import torch
from flask import Flask, request, jsonify
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from dotenv import load_dotenv
from flask_cors import CORS
import openai
import pyodbc
from datetime import datetime
from typing import Optional, Dict, Any, List
import tracemalloc

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

# AI 모델 및 벡터 데이터베이스 초기화
try:
    # Pinecone 벡터 데이터베이스 연결
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index(INDEX_NAME)
    
    # OpenAI 클라이언트 초기화
    openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Google T5 텍스트 생성 모델 및 토크나이저 로드
    text_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')
    text_tokenizer = T5Tokenizer.from_pretrained(
        'google/flan-t5-base',
        legacy=True,
        clean_up_tokenization_spaces=False
    )
    
    # MSSQL 연결 문자열 (Pinecone 동기화용)
    mssql_config = {
        'server': os.getenv('MSSQL_SERVER'),
        'database': os.getenv('MSSQL_DATABASE'),
        'username': os.getenv('MSSQL_USERNAME'),
        'password': os.getenv('MSSQL_PASSWORD')
    }
    
    connection_string = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={mssql_config['server']},1433;"  # 포트 명시
            f"DATABASE={mssql_config['database']};"
            f"UID={mssql_config['username']};"
            f"PWD={mssql_config['password']};"
            f"TrustServerCertificate=yes;"
            f"Connection Timeout=30;"  # 타임아웃 추가
    )

except Exception as e:
    logging.error(f"모듈 로드 실패: {str(e)}")
    app.logger.error(f"모듈 로드 실패: {str(e)}")
    raise

# ====== 기존 AI 답변 생성 클래스 ======
class AIAnswerGenerator:
    
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

    # ★ 메모리 누수 해제
    def create_embedding(self, text: str) -> list:
        try:
            response = openai_client.embeddings.create(
                model='text-embedding-3-small',
                input=text
            )
            # return response.data[0].embedding
            embedding = response.data[0].embedding
            del response  # 불필요 객체 해제
            gc.collect()
            return embedding
        except Exception as e:
            logging.error(f"임베딩 생성 실패: {e}")
            return None

    def search_similar_answers(self, query: str, top_k: int = 10, similarity_threshold: float = 0.6) -> list:
        try:
            query_vector = self.create_embedding(query)
            if query_vector is None:
                return []
            
            results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
            
            filtered_results = [
                {
                    'score': match['score'],
                    'question': match['metadata'].get('question', ''),
                    'answer': match['metadata'].get('answer', ''),
                    'category': match['metadata'].get('category', '일반')
                }
                for match in results['matches'] if match['score'] >= similarity_threshold
            ]
            
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

    # ★ 메모리 누수 해제
    # torch.no_grad()는 불필요한 메모리 점유를 막고, gc.collect()는 사용 후 메모리를 즉시 해제. 스레드 제한은 CPU 과부하 방지
    def generate_with_t5(self, query: str, similar_answers: list) -> str:
        try:
            torch.set_num_threads(2)  # CPU 스레드 수 제한 (인스턴스 코어에 맞게 조정. 2개)

            context_answers = []
            for ans in similar_answers[:3]:
                clean_ans = ans['answer']
                clean_ans = re.sub(r'[\b\r\f\v]', '', clean_ans)
                clean_ans = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', clean_ans)
                clean_ans = re.sub(r'<[^>]+>', '', clean_ans)
                context_answers.append(clean_ans)
            
            context = " ".join(context_answers)
            
            prompt = f"질문: {query}\n참고답변: {context}\n답변:"
            
            # inputs = text_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():  # 그라디언트 계산 비활성화로 메모리/CPU 절약
                inputs = text_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

                outputs = text_model.generate(
                    **inputs, 
                    max_length=200,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=True,
                    temperature=0.7
                )
            
            generated = text_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            del inputs  # 텐서 해제
            del outputs # 텐서 해제
            gc.collect()  # 가비지 컬렉션 강제 실행

            if "답변:" in generated:
                generated = generated.split("답변:")[-1].strip()
            
            generated = re.sub(r'[\b\r\f\v]', '', generated)
            generated = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', generated)
            
            return generated
            
        except Exception as e:
            logging.error(f"T5 모델 생성 실패: {e}")
            if similar_answers:
                fallback_answer = similar_answers[0]['answer']
                fallback_answer = re.sub(r'[\b\r\f\v]', '', fallback_answer)
                fallback_answer = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', fallback_answer)
                return fallback_answer
            return ""

    def generate_ai_answer(self, query: str, similar_answers: list, lang: str) -> str:
        if not similar_answers:
            default_msg = "<p>문의해주신 내용에 대해 정확한 답변을 드리기 위해 더 자세한 정보가 필요합니다.</p><p><br></p><p>고객센터로 문의해주시면 신속하게 도움을 드리겠습니다.</p>"
            return default_msg
        
        try:
            base_answer = self.generate_with_t5(query, similar_answers)
            
            if not base_answer or len(base_answer.strip()) < 10:
                base_answer = similar_answers[0]['answer']
            
            base_answer = re.sub(r'<[^>]+>', '', base_answer)
            base_answer = self.remove_old_app_name(base_answer)
            
            has_greeting = any(greeting in base_answer.lower() for greeting in ['안녕하세요', '안녕'])
            has_closing = any(closing in base_answer.lower() for closing in ['감사합니다', '감사드립니다'])
            
            final_answer = ""
            
            if not has_greeting:
                final_answer += "안녕하세요, GOODTV 바이블 애플입니다. "
            
            final_answer += base_answer
            
            if not has_closing:
                final_answer += " 항상 주님 안에서 평안하세요. 감사합니다."
            
            final_answer = self.clean_answer_text(final_answer)
            
            return final_answer
            
        except Exception as e:
            logging.error(f"답변 생성 실패: {e}")
            return "<p>죄송합니다. 현재 답변을 생성할 수 없습니다.</p><p><br></p><p>고객센터로 문의해주세요.</p>"

    def process(self, seq: int, question: str, lang: str) -> dict:
        try:
            processed_question = self.preprocess_text(question)
            if not processed_question:
                return {"success": False, "error": "질문이 비어있습니다."}
            
            logging.info(f"처리 시작 - SEQ: {seq}, 질문: {processed_question[:50]}...")
            
            similar_answers = self.search_similar_answers(processed_question)
            ai_answer = self.generate_ai_answer(processed_question, similar_answers, lang)
            
            ai_answer = ai_answer.replace('"', '"').replace('"', '"')
            ai_answer = ai_answer.replace(''', "'").replace(''', "'")
            
            result = {
                "success": True,
                "answer": ai_answer,
                "similar_count": len(similar_answers),
                "embedding_model": "text-embedding-3-small",
                "generation_model": "google/flan-t5-base"
            }
            
            logging.info(f"처리 완료 - SEQ: {seq}, HTML 답변 생성됨")
            return result
            
        except Exception as e:
            logging.error(f"처리 중 오류 - SEQ: {seq}, 오류: {str(e)}")
            return {"success": False, "error": str(e)}

# ====== 새로운 Pinecone 동기화 클래스 ======
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
            
            response = self.openai_client.embeddings.create(
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
    """AI 답변 생성 API (기존)"""
    try:
        data = request.get_json()
        seq = data.get('seq', 0)
        question = data.get('question', '')
        lang = data.get('lang', 'kr')
        
        if not question:
            return jsonify({"success": False, "error": "질문이 필요합니다."}), 400
        
        result = generator.process(seq, question, lang)
        
        response = jsonify(result)
        response.headers['Content-Type'] = 'application/json; charset=utf-8'

        # 메모리 누수 추적용 tracemalloc 스냅샷 찍기 (엔드포인트 끝에서)
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        logging.info("Top 10 memory leaks: " + str(top_stats[:10]))

        return response
        
    except Exception as e:
        logging.error(f"API 호출 오류: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/sync_to_pinecone', methods=['POST'])
def sync_to_pinecone():
    """MSSQL 데이터를 Pinecone에 동기화하는 API (새로 추가)"""
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
    
    app.run(host='0.0.0.0', port=port, debug=False)