#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
동기화 서비스 모듈
"""

import logging
import pyodbc
from typing import Optional, Dict, Any
from datetime import datetime
from src.utils.memory_manager import memory_cleanup
from src.utils.text_preprocessor import TextPreprocessor
from src.models.embedding_generator import EmbeddingGenerator


class SyncService:
    """MSSQL과 Pinecone 간의 동기화를 담당하는 클래스"""
    
    def __init__(self, pinecone_index, openai_client, connection_string, category_mapping):
        self.index = pinecone_index
        self.connection_string = connection_string
        self.category_mapping = category_mapping
        self.text_processor = TextPreprocessor()
        self.embedding_generator = EmbeddingGenerator(openai_client)
        self.openai_client = openai_client
    
    def fix_korean_typos_with_ai(self, text: str) -> str:
        """AI를 이용한 한국어 오타 수정"""
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
    
    def get_category_name(self, cate_idx: str) -> str:
        """카테고리 인덱스를 이름으로 변환하는 메서드"""
        return self.category_mapping.get(str(cate_idx), '사용 문의(기타)')
    
    def get_mssql_data(self, seq: int) -> Optional[Dict]:
        """MSSQL에서 데이터 조회하는 메서드"""
        try:
            with memory_cleanup():
                conn = pyodbc.connect(self.connection_string)
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
        """MSSQL 데이터를 Pinecone에 동기화하는 메서드"""
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
                raw_question = self.text_processor.preprocess_text(data['contents'])
                question = self.fix_korean_typos_with_ai(raw_question)
                answer = self.text_processor.preprocess_text(data['reply_contents'])
                
                # 임베딩 생성 (질문 기반)
                embedding = self.embedding_generator.create_embedding(question)
                if not embedding:
                    return {"success": False, "error": "임베딩 생성 실패"}
                
                # 카테고리 이름 가져오기
                category = self.get_category_name(data['cate_idx'])
                
                # 메타데이터 구성 (질문은 오타 수정된 버전 사용)
                metadata = {
                    "seq": int(data['seq']),
                    "question": question,
                    "answer": self.text_processor.preprocess_text_for_metadata(data['reply_contents'], for_metadata=True),
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
