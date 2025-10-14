#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MSSQL 데이터베이스 업데이트 유틸리티
AI 답변 생성 후 DB에 직접 업데이트하는 기능 제공
"""

import logging
import pyodbc
import os
from typing import Optional

class MSSQLUpdater:
    """MSSQL 데이터베이스 업데이트 클래스"""
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        초기화 메서드
        
        Args:
            connection_string: MSSQL 연결 문자열 (없으면 환경변수에서 읽음)
        """
        if connection_string:
            self.connection_string = connection_string
        else:
            # 환경변수에서 MSSQL 연결 정보 읽기
            self.connection_string = self._build_connection_string()
        
        logging.info("MSSQLUpdater 초기화 완료")
    
    def _build_connection_string(self) -> str:
        """환경변수에서 MSSQL 연결 문자열 구성"""
        server = os.getenv('MSSQL_SERVER')
        database = os.getenv('MSSQL_DATABASE')
        username = os.getenv('MSSQL_USERNAME')
        password = os.getenv('MSSQL_PASSWORD')
        
        if not all([server, database, username, password]):
            raise ValueError("MSSQL 환경변수가 설정되지 않았습니다")
        
        return (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={server},1433;"
            f"DATABASE={database};"
            f"UID={username};"
            f"PWD={password};"
            f"TrustServerCertificate=yes;"
            f"Connection Timeout=30;"
        )
    
    def update_inquiry_answer(self, seq: int, answer: str, answer_yn: str = 'N') -> bool:
        """
        문의 답변을 DB에 업데이트
        
        Args:
            seq: 문의 시퀀스 번호
            answer: AI가 생성한 답변 (HTML 포맷)
            answer_yn: 답변 승인 여부 ('N' = AI 답변, 'Y' = 관리자 승인)
        
        Returns:
            bool: 업데이트 성공 여부
        """
        conn = None
        cursor = None
        
        try:
            # MSSQL 연결
            conn = pyodbc.connect(self.connection_string)
            cursor = conn.cursor()
            
            # SQL 쿼리 작성 (SQL Injection 방지를 위해 파라미터 사용)
            sql = """
                UPDATE mobile.dbo.bible_inquiry 
                SET reply_contents = ?, 
                    answer_YN = ?
                WHERE seq = ?
            """
            
            # 쿼리 실행
            cursor.execute(sql, (answer, answer_yn, seq))
            conn.commit()
            
            # 영향받은 행 수 확인
            if cursor.rowcount > 0:
                logging.info(f"✅ DB 업데이트 성공: SEQ={seq}, answer_YN={answer_yn}, 답변 길이={len(answer)}자")
                return True
            else:
                logging.warning(f"⚠️ DB 업데이트 실패: SEQ={seq} - 해당하는 레코드가 없습니다")
                return False
            
        except pyodbc.Error as e:
            # MSSQL 관련 오류
            logging.error(f"❌ MSSQL 오류 - SEQ={seq}: {str(e)}")
            if conn:
                conn.rollback()
            return False
            
        except Exception as e:
            # 기타 예외
            logging.error(f"❌ 예상치 못한 오류 - SEQ={seq}: {str(e)}")
            if conn:
                conn.rollback()
            return False
            
        finally:
            # 리소스 정리
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def get_inquiry_info(self, seq: int) -> Optional[dict]:
        """
        문의 정보 조회 (디버깅/검증용)
        
        Args:
            seq: 문의 시퀀스 번호
        
        Returns:
            dict: 문의 정보 (없으면 None)
        """
        conn = None
        cursor = None
        
        try:
            conn = pyodbc.connect(self.connection_string)
            cursor = conn.cursor()
            
            sql = """
                SELECT seq, contents, reply_contents, answer_YN, 
                       ISNULL(option_lang, 'kr') as option_lang
                FROM mobile.dbo.bible_inquiry 
                WHERE seq = ?
            """
            
            cursor.execute(sql, (seq,))
            row = cursor.fetchone()
            
            if row:
                return {
                    'seq': row.seq,
                    'contents': row.contents,
                    'reply_contents': row.reply_contents,
                    'answer_YN': row.answer_YN,
                    'option_lang': row.option_lang
                }
            else:
                return None
                
        except Exception as e:
            logging.error(f"문의 정보 조회 실패 - SEQ={seq}: {str(e)}")
            return None
            
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def test_connection(self) -> bool:
        """
        DB 연결 테스트
        
        Returns:
            bool: 연결 성공 여부
        """
        conn = None
        try:
            conn = pyodbc.connect(self.connection_string)
            logging.info("✅ MSSQL 연결 테스트 성공")
            return True
        except Exception as e:
            logging.error(f"❌ MSSQL 연결 테스트 실패: {str(e)}")
            return False
        finally:
            if conn:
                conn.close()