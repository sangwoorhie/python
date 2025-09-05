#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MSSQL to Pinecone 동기화 API
파일명: free_6_pinecone_sync.py
설명: MSSQL 문의 데이터를 Pinecone 벡터 DB에 저장/수정하는 Flask API
"""

import os
import sys
import json
import logging
import re
import html
import unicodedata
from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
import openai
from dotenv import load_dotenv
from typing import Optional, Dict, Any
import pyodbc
from datetime import datetime

# Flask 애플리케이션 설정
app = Flask(__name__)
CORS(app)

# 로깅 설정
logging.basicConfig(
    filename='/home/ec2-user/python/logs/pinecone_sync.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# 환경변수 로드
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

# 초기화
try:
    # Pinecone 초기화
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index(INDEX_NAME)
    
    # OpenAI 초기화
    openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # MSSQL 연결 문자열
    mssql_config = {
        'server': os.getenv('MSSQL_SERVER'),
        'database': os.getenv('MSSQL_DATABASE'),
        'username': os.getenv('MSSQL_USERNAME'),
        'password': os.getenv('MSSQL_PASSWORD')
    }
    
    connection_string = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={mssql_config['server']};"
        f"DATABASE={mssql_config['database']};"
        f"UID={mssql_config['username']};"
        f"PWD={mssql_config['password']};"
        f"TrustServerCertificate=yes;"
    )
    
    logging.info("모든 서비스 초기화 완료")
    
except Exception as e:
    logging.error(f"초기화 실패: {str(e)}")
    raise

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
        """
        MSSQL 데이터를 Pinecone에 동기화
        
        Args:
            seq: 문의 번호
            mode: 'upsert' (생성/수정) 또는 'delete' (삭제)
        
        Returns:
            처리 결과 딕셔너리
        """
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
            
            # Pinecone에 upsert (생성 또는 수정)
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
    
    def batch_sync(self, seq_list: list) -> Dict[str, Any]:
        """여러 건 일괄 동기화"""
        results = {
            "success": True,
            "total": len(seq_list),
            "succeeded": 0,
            "failed": 0,
            "details": []
        }
        
        for seq in seq_list:
            result = self.sync_to_pinecone(seq)
            if result["success"]:
                results["succeeded"] += 1
            else:
                results["failed"] += 1
            results["details"].append(result)
        
        if results["failed"] > 0:
            results["success"] = False
        
        return results

# 싱글톤 인스턴스
sync_manager = PineconeSyncManager()

# ====== Flask API 엔드포인트 ======

@app.route('/sync_to_pinecone', methods=['POST'])
def sync_to_pinecone():
    """
    MSSQL 데이터를 Pinecone에 동기화하는 API
    
    요청 형식:
    {
        "seq": 123,
        "mode": "upsert"  // "upsert" 또는 "delete"
    }
    """
    try:
        data = request.get_json()
        seq = data.get('seq')
        mode = data.get('mode', 'upsert')
        
        # 파라미터 검증
        if not seq:
            return jsonify({"success": False, "error": "seq가 필요합니다"}), 400
        
        if not isinstance(seq, int):
            seq = int(seq)
        
        # Pinecone 동기화 실행
        result = sync_manager.sync_to_pinecone(seq, mode)
        
        # 응답
        status_code = 200 if result["success"] else 500
        return jsonify(result), status_code
        
    except ValueError as e:
        return jsonify({"success": False, "error": f"잘못된 seq 값: {str(e)}"}), 400
    except Exception as e:
        logging.error(f"API 오류: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/batch_sync_to_pinecone', methods=['POST'])
def batch_sync_to_pinecone():
    """
    여러 건 일괄 동기화 API
    
    요청 형식:
    {
        "seq_list": [123, 124, 125]
    }
    """
    try:
        data = request.get_json()
        seq_list = data.get('seq_list', [])
        
        if not seq_list:
            return jsonify({"success": False, "error": "seq_list가 필요합니다"}), 400
        
        # 일괄 동기화 실행
        result = sync_manager.batch_sync(seq_list)
        
        return jsonify(result), 200
        
    except Exception as e:
        logging.error(f"배치 API 오류: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """헬스체크 엔드포인트"""
    try:
        # Pinecone 연결 확인
        stats = index.describe_index_stats()
        
        return jsonify({
            "status": "healthy",
            "pinecone_vectors": stats.get('total_vector_count', 0),
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.getenv('FLASK_PORT_SYNC', 8001))
    print(f"Pinecone 동기화 API 시작 - 포트: {port}")
    app.run(host='0.0.0.0', port=port, debug=False)