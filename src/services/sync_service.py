#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
동기화 서비스 모듈
- MSSQL 데이터베이스와 Pinecone 벡터 데이터베이스 간의 동기화
- AI 기반 텍스트 전처리 및 임베딩 생성
- 실시간 데이터 동기화 및 벡터 인덱스 관리
"""

import logging
import pyodbc
from typing import Optional, Dict, Any
from datetime import datetime
from src.utils.memory_manager import memory_cleanup
from src.utils.text_preprocessor import TextPreprocessor
from src.models.embedding_generator import EmbeddingGenerator

# ===== MSSQL과 Pinecone 간의 동기화를 담당하는 메인 클래스 =====
class SyncService:
    
    # SyncService 초기화 - 데이터베이스 연결 및 동기화 도구 설정
    # Args:
    #     pinecone_index: Pinecone 벡터 인덱스
    #     openai_client: OpenAI API 클라이언트
    #     connection_string: MSSQL 데이터베이스 연결 문자열
    #     category_mapping: 카테고리 매핑 딕셔너리
    def __init__(self, pinecone_index, openai_client, connection_string, category_mapping):
        self.index = pinecone_index                               # Pinecone 벡터 인덱스
        self.connection_string = connection_string                # MSSQL 연결 문자열
        self.category_mapping = category_mapping                  # 카테고리 ID → 이름 매핑
        self.text_processor = TextPreprocessor()                  # 텍스트 전처리 도구
        self.embedding_generator = EmbeddingGenerator(openai_client)  # 임베딩 생성기
        self.openai_client = openai_client                        # GPT 기반 텍스트 처리용
    
    # AI를 이용한 한국어 오타 수정 메서드
    # Args:
    #     text: 오타 수정할 한국어 텍스트
    # Returns:
    #     str: 오타가 수정된 텍스트 (수정 실패시 원본 반환)
    def fix_korean_typos_with_ai(self, text: str) -> str:
        # ===== 1단계: 기본 유효성 검사 =====
        if not text or len(text.strip()) < 3:
            return text
        
        # ===== 2단계: 텍스트 길이 제한 (비용 최적화) =====
        if len(text) > 500:
            logging.warning(f"텍스트가 너무 길어 오타 수정 건너뜀: {len(text)}자")
            return text
        
        try:
            # ===== 3단계: 메모리 최적화 컨텍스트 시작 =====
            with memory_cleanup():
                # ===== 4단계: GPT 시스템 프롬프트 구성 =====
                # 한국어 맞춤법 및 오타 교정 전문가 역할 부여
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

                # ===== 5단계: 사용자 프롬프트 구성 =====
                user_prompt = f"다음 텍스트의 맞춤법과 오타를 수정해주세요:\n\n{text}"

                # ===== 6단계: GPT API 호출 (오타 수정) =====
                response = self.openai_client.chat.completions.create(
                    model='gpt-5-mini',
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_completion_tokens=600,                                 # 충분한 텍스트 길이 허용
                    temperature=0.1,                                # 매우 보수적 설정 (일관성 중시)
                    top_p=0.8,                                      # 상위 80% 토큰만 사용
                    frequency_penalty=0.0,                          # 반복 페널티 없음
                    presence_penalty=0.0                            # 새로운 주제 페널티 없음
                )
                
                # ===== 7단계: 응답 결과 추출 및 메모리 정리 =====
                corrected_text = response.choices[0].message.content.strip()
                del response # 메모리 해제
                
                # ===== 8단계: 결과 품질 검증 =====
                # 8-1: 빈 결과 검증
                if not corrected_text or len(corrected_text) == 0:
                    logging.warning("AI 오타 수정 결과가 비어있음, 원문 반환")
                    return text
                
                # 8-2: 과도한 변경 검증 (길이가 2배 이상 늘어나면 의심)
                if len(corrected_text) > len(text) * 2:
                    logging.warning("AI 오타 수정 결과가 원문보다 너무 길어짐, 원문 반환")
                    return text
                
                # ===== 9단계: 수정 내용 로깅 =====
                if corrected_text != text:
                    logging.info(f"AI 오타 수정: '{text[:50]}...' → '{corrected_text[:50]}...'")
                
                # ===== 10단계: 수정된 텍스트 반환 =====
                return corrected_text
                
        except Exception as e:
            # ===== 예외 처리: AI 실패시 원문 반환 =====
            logging.error(f"AI 오타 수정 실패: {e}")
            return text
    
    # 카테고리 인덱스를 이름으로 변환하는 메서드
    # Args:
    #     cate_idx: 카테고리 인덱스 (문자열 또는 숫자)
    # Returns:
    #     str: 카테고리 이름 (매핑되지 않으면 기본값 반환)
    def get_category_name(self, cate_idx: str) -> str:
        # 카테고리 매핑 딕셔너리에서 이름 조회 (기본값: '사용 문의(기타)')
        return self.category_mapping.get(str(cate_idx), '사용 문의(기타)')
    
    # MSSQL에서 특정 seq의 문의 데이터를 조회하는 메서드
    # Args:
    #     seq: 조회할 문의의 시퀀스 번호
    # Returns:
    #     Optional[Dict]: 조회된 문의 데이터 (실패시 None)
    def get_mssql_data(self, seq: int) -> Optional[Dict]:
        try:
            # ===== 메모리 최적화 컨텍스트 시작 =====
            with memory_cleanup():
                # ===== 1단계: MSSQL 데이터베이스 연결 =====
                conn = pyodbc.connect(self.connection_string)
                cursor = conn.cursor()
                
                # ===== 2단계: SQL 쿼리 정의 =====
                # 답변이 완료된(answer_YN = 'Y') 문의만 조회
                query = """
                SELECT seq, contents, reply_contents, cate_idx, name, 
                       CONVERT(varchar, regdate, 120) as regdate
                FROM mobile.dbo.bible_inquiry
                WHERE seq = ? AND answer_YN = 'Y'
                """
                
                # ===== 3단계: 쿼리 실행 =====
                cursor.execute(query, seq)
                row = cursor.fetchone()
                
                # ===== 4단계: 조회 결과 처리 =====
                if row:
                    # 조회된 데이터를 딕셔너리로 구성
                    data = {
                        'seq': row[0],                              # 시퀀스 번호
                        'contents': row[1],                         # 질문 내용
                        'reply_contents': row[2],                   # 답변 내용
                        'cate_idx': row[3],                         # 카테고리 인덱스
                        'name': row[4],                             # 질문자 이름
                        'regdate': row[5]                           # 등록일자
                    }
                    
                    # ===== 5단계: 데이터베이스 연결 정리 =====
                    cursor.close()
                    conn.close()
                    
                    return data
                else:
                    # ===== 데이터 없음: 연결 정리 후 None 반환 =====
                    cursor.close()
                    conn.close()
                    return None
            
        except Exception as e:
            # ===== 예외 처리: MSSQL 조회 실패 =====
            logging.error(f"MSSQL 조회 실패: {e}")
            return None
    
    # MSSQL 데이터를 Pinecone에 동기화하는 메인 메서드
    # Args:
    #     seq: 동기화할 문의의 시퀀스 번호
    #     mode: 동기화 모드 ('upsert' 또는 'delete')
    # Returns:
    #     Dict[str, Any]: 동기화 결과 정보
    def sync_to_pinecone(self, seq: int, mode: str = 'upsert') -> Dict[str, Any]:
        try:
            # ===== 메모리 최적화 컨텍스트 시작 =====
            with memory_cleanup():
                # ===== 1단계: 삭제 모드 처리 =====
                if mode == 'delete':
                    vector_id = f"qa_bible_{seq}"
                    self.index.delete(ids=[vector_id])
                    logging.info(f"Pinecone에서 삭제 완료: {vector_id}")
                    return {"success": True, "message": "삭제 완료", "seq": seq}
                
                # ===== 2단계: MSSQL에서 원본 데이터 조회 =====
                data = self.get_mssql_data(seq)
                if not data:
                    return {"success": False, "error": "데이터를 찾을 수 없습니다"}
                
                # ===== 3단계: 텍스트 전처리 및 AI 오타 수정 =====
                # 3-1: 질문 텍스트 전처리 (기본 정제)
                raw_question = self.text_processor.preprocess_text(data['contents'])
                # 3-2: AI 기반 오타 수정 적용 (검색 정확도 향상)
                question = self.fix_korean_typos_with_ai(raw_question)
                # 3-3: 답변 텍스트 전처리
                answer = self.text_processor.preprocess_text(data['reply_contents'])
                
                # ===== 4단계: 임베딩 벡터 생성 =====
                # 오타 수정된 질문을 기반으로 임베딩 생성
                embedding = self.embedding_generator.create_embedding(question)
                if not embedding:
                    return {"success": False, "error": "임베딩 생성 실패"}
                
                # ===== 5단계: 카테고리 이름 변환 =====
                category = self.get_category_name(data['cate_idx'])
                
                # ===== 6단계: Pinecone 메타데이터 구성 =====
                metadata = {
                    "seq": int(data['seq']),                        # 문의 시퀀스 번호
                    "question": question,                           # 오타 수정된 질문
                    "answer": self.text_processor.preprocess_text_for_metadata(
                        data['reply_contents'], for_metadata=True   # 메타데이터용 답변 처리
                    ),
                    "category": category,                           # 카테고리 이름
                    "name": data['name'] if data['name'] else "익명", # 질문자 이름
                    "regdate": data['regdate'],                     # 등록일자
                    "source": "bible_inquiry_mssql",               # 데이터 출처
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 동기화 시간
                }
                
                # ===== 7단계: 벡터 ID 생성 및 기존 벡터 확인 =====
                vector_id = f"qa_bible_{seq}"
                
                # 기존 벡터 존재 여부 확인 (신규/수정 구분용)
                existing = self.index.fetch(ids=[vector_id])
                is_update = vector_id in existing['vectors']
                
                # ===== 8단계: 벡터 데이터 구성 =====
                vector_data = {
                    "id": vector_id,                                # 고유 벡터 ID
                    "values": embedding,                            # 임베딩 벡터 값
                    "metadata": metadata                            # 메타데이터
                }
                
                # ===== 9단계: Pinecone에 벡터 저장 (upsert) =====
                self.index.upsert(vectors=[vector_data])
                
                # ===== 10단계: 동기화 완료 처리 =====
                action = "수정" if is_update else "생성"
                logging.info(f"Pinecone {action} 완료: {vector_id}")
                
                # ===== 11단계: 성공 결과 반환 =====
                return {
                    "success": True,
                    "message": f"Pinecone {action} 완료",
                    "seq": seq,
                    "vector_id": vector_id,
                    "is_update": is_update
                }
            
        except Exception as e:
            # ===== 예외 처리: 동기화 실패 =====
            logging.error(f"Pinecone 동기화 실패: {str(e)}")
            return {"success": False, "error": str(e)}
