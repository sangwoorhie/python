"""
Bible AI 애플리케이션 MSSQL 연동 스크립트 (OpenAI 모델 버전)

이 스크립트는 MSSQL 데이터베이스와 연동하여 고객 문의에 대한 AI 답변을 생성하고 관리합니다.
OpenAI text-embedding-3-small과 GPT-4o-mini 모델을 사용하여 고품질 답변을 제공합니다.

주요 기능:
1. MSSQL 데이터베이스 연결 및 문의 조회
2. OpenAI 모델을 사용한 AI 답변 생성
3. 관리자 승인 워크플로우
4. 문의 처리 상태 관리

워크플로우:
1. 답변 없는 문의 조회
2. AI 답변 생성 (answer_YN='N')
3. 관리자 승인/반려
4. 승인 시 고객 확인 가능 (answer_YN='Y')

작성자: Bible AI Team
버전: 1.0
마지막 수정: 2024
"""

import os
import sys
import argparse
import json
import pyodbc
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
import subprocess
import re
import html
from typing import Optional, Dict, Any

# 환경 변수 로드
load_dotenv()

class BibleInquiryProcessorOpenAI:
    """바이블 애플 문의 처리 시스템 (OpenAI 모델 사용)"""
    
    def __init__(self):
        """
        MSSQL 연결 정보를 환경변수에서 로드하여 초기화합니다.
        
        필요한 환경변수:
        - MSSQL_SERVER: MSSQL 서버 주소
        - MSSQL_DATABASE: 데이터베이스 이름
        - MSSQL_USERNAME: 사용자명
        - MSSQL_PASSWORD: 비밀번호
        """
        # 환경변수에서 MSSQL 연결 정보 가져오기
        server = os.getenv('MSSQL_SERVER')
        database = os.getenv('MSSQL_DATABASE')
        username = os.getenv('MSSQL_USERNAME')
        password = os.getenv('MSSQL_PASSWORD')
        
        # 필수 환경변수 확인
        if not all([server, database, username, password]):
            missing_vars = []
            if not server: missing_vars.append('MSSQL_SERVER')
            if not database: missing_vars.append('MSSQL_DATABASE')
            if not username: missing_vars.append('MSSQL_USERNAME')
            if not password: missing_vars.append('MSSQL_PASSWORD')
            
            raise ValueError(f"다음 환경변수들이 설정되지 않았습니다: {', '.join(missing_vars)}")
        
        # MSSQL 연결 문자열 구성
        self.connection_string = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"UID={username};"
            f"PWD={password};"
            f"TrustServerCertificate=yes;"
        )
        self.conn = None
    
    def connect_database(self) -> bool:
        """
        MSSQL 데이터베이스에 연결합니다.
        
        Returns:
            bool: 연결 성공 시 True, 실패 시 False
        """
        try:
            self.conn = pyodbc.connect(self.connection_string)
            print("✅ MSSQL 데이터베이스 연결 성공")
            return True
        except Exception as e:
            print(f"❌ 데이터베이스 연결 실패: {e}")
            print("💡 연결 정보와 ODBC 드라이버를 확인하세요.")
            return False
    
    def disconnect_database(self) -> None:
        """데이터베이스 연결을 해제합니다."""
        if self.conn:
            self.conn.close()
            print("🔌 데이터베이스 연결 해제")
    
    def get_unanswered_inquiries(self, limit: int = 10) -> pd.DataFrame:
        """
        답변이 없는 문의들을 조회합니다.
        
        Args:
            limit (int): 조회할 최대 문의 수 (기본값: 10)
            
        Returns:
            pd.DataFrame: 답변 없는 문의들의 데이터프레임
        """
        if not self.conn:
            self.connect_database()
        
        query = f"""
        SELECT TOP {limit}
            [seq], [device_id], [member_id], [name], [contents], 
            [regdate], [platform], [app_version], [cate_idx]
        FROM [mobile].[dbo].[bible_inquiry]
        WHERE ([reply_contents] IS NULL OR [reply_contents] = '')
        ORDER BY [regdate] DESC
        """
        
        try:
            df = pd.read_sql(query, self.conn)
            print(f"📝 답변 대기 중인 문의: {len(df)}건")
            return df
        except Exception as e:
            print(f"❌ 문의 조회 실패: {e}")
            print("💡 데이터베이스 연결과 테이블 구조를 확인하세요.")
            return pd.DataFrame()
    
    def generate_ai_answer_for_inquiry(self, inquiry_seq: int, question: str) -> Optional[str]:
        """
        특정 문의에 대해 AI 답변을 생성합니다. (OpenAI 모델 사용)
        
        Args:
            inquiry_seq (int): 문의 번호
            question (str): 문의 내용
            
        Returns:
            Optional[str]: 생성된 AI 답변, 실패 시 None
        """
        try:
            print(f"🤖 OpenAI 모델로 답변 생성 중 (문의번호: {inquiry_seq})...")
            
            # OpenAI 버전 Python 스크립트 실행
            result = subprocess.run([
                'python', '4_ai_answer_generator.py',
                '--question', question,
                '--output', 'json'
            ], capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode == 0:
                response = json.loads(result.stdout)
                if response["success"]:
                    print(f"✅ OpenAI 모델 답변 생성 성공")
                    print(f"📊 사용 모델: {response.get('embedding_model', 'N/A')} + {response.get('generation_model', 'N/A')}")
                    print(f"🔍 참고 답변 수: {response.get('similar_count', 0)}개")
                    return response["answer"]
                else:
                    print(f"❌ AI 답변 생성 실패: {response['error']}")
                    return None
            else:
                print(f"❌ 스크립트 실행 실패: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"❌ AI 답변 생성 중 오류: {e}")
            print("💡 4_ai_answer_generator.py 파일과 OpenAI API 키를 확인하세요.")
            return None
    
    def save_ai_answer(self, inquiry_seq: int, ai_answer: str) -> bool:
        """
        생성된 AI 답변을 DB에 저장합니다. (answer_YN='N' - 관리자 승인 대기)
        
        Args:
            inquiry_seq (int): 문의 번호
            ai_answer (str): 생성된 AI 답변
            
        Returns:
            bool: 저장 성공 시 True, 실패 시 False
        """
        if not self.conn:
            self.connect_database()
        
        try:
            cursor = self.conn.cursor()
            
            # AI 답변을 reply_contents에 저장하되 answer_YN='N'으로 설정 (관리자 승인 필요)
            update_query = """
            UPDATE [mobile].[dbo].[bible_inquiry] 
            SET reply_contents = ?, answer_YN = 'N'
            WHERE seq = ?
            """
            
            cursor.execute(update_query, (ai_answer, inquiry_seq))
            self.conn.commit()
            print(f"✅ AI 답변이 저장됨 (문의 번호: {inquiry_seq}, 관리자 승인 대기)")
            print("💰 OpenAI 고품질 모델 사용으로 정확한 답변 제공!")
            return True
            
        except Exception as e:
            print(f"❌ AI 답변 저장 실패: {e}")
            print("💡 데이터베이스 연결과 권한을 확인하세요.")
            return False
    
    def get_pending_confirmations(self) -> pd.DataFrame:
        """
        관리자 승인 대기 중인 답변들을 조회합니다. (answer_YN='N')
        
        Returns:
            pd.DataFrame: 승인 대기 중인 답변들의 데이터프레임
        """
        if not self.conn:
            self.connect_database()
        
        query = """
        SELECT seq, name, contents, reply_contents, regdate
        FROM [mobile].[dbo].[bible_inquiry]
        WHERE reply_contents IS NOT NULL 
        AND reply_contents != ''
        AND answer_YN = 'N'
        ORDER BY regdate DESC
        """
        
        try:
            df = pd.read_sql(query, self.conn)
            print(f"👨‍💼 관리자 승인 대기: {len(df)}건")
            return df
        except Exception as e:
            print(f"❌ 승인 대기 목록 조회 실패: {e}")
            print("💡 데이터베이스 연결과 테이블 구조를 확인하세요.")
            return pd.DataFrame()
    
    def confirm_answer(self, inquiry_seq: int, admin_name: Optional[str] = None, final_answer: Optional[str] = None) -> bool:
        """
        관리자: 답변을 승인합니다. (answer_YN='Y'로 변경)
        
        Args:
            inquiry_seq (int): 문의 번호
            admin_name (Optional[str]): 승인자 이름 (미사용)
            final_answer (Optional[str]): 수정된 답변 (None이면 기존 답변 사용)
            
        Returns:
            bool: 승인 성공 시 True, 실패 시 False
        """
        if not self.conn:
            self.connect_database()
        
        try:
            cursor = self.conn.cursor()
            
            if final_answer:
                # 수정된 답변으로 업데이트
                update_query = """
                UPDATE [mobile].[dbo].[bible_inquiry] 
                SET reply_contents = ?, answer_YN = 'Y'
                WHERE seq = ?
                """
                cursor.execute(update_query, (final_answer, inquiry_seq))
                print(f"✅ 답변 수정 및 승인 완료 (문의 번호: {inquiry_seq})")
            else:
                # 기존 답변 승인
                update_query = """
                UPDATE [mobile].[dbo].[bible_inquiry] 
                SET answer_YN = 'Y'
                WHERE seq = ?
                """
                cursor.execute(update_query, (inquiry_seq,))
                print(f"✅ 답변 승인 완료 (문의 번호: {inquiry_seq})")
            
            self.conn.commit()
            print("🎉 고객이 답변을 확인할 수 있습니다!")
            return True
            
        except Exception as e:
            print(f"❌ 답변 승인 실패: {e}")
            print("💡 데이터베이스 연결과 권한을 확인하세요.")
            return False
    
    def reject_answer(self, inquiry_seq: int, reason: Optional[str] = None) -> bool:
        """
        관리자: 답변을 반려합니다. (reply_contents 초기화)
        
        Args:
            inquiry_seq (int): 문의 번호
            reason (Optional[str]): 반려 사유
            
        Returns:
            bool: 반려 성공 시 True, 실패 시 False
        """
        if not self.conn:
            self.connect_database()
        
        try:
            cursor = self.conn.cursor()
            
            # 답변 반려 - reply_contents 초기화
            update_query = """
            UPDATE [mobile].[dbo].[bible_inquiry] 
            SET reply_contents = NULL, answer_YN = NULL
            WHERE seq = ?
            """
            
            cursor.execute(update_query, (inquiry_seq,))
            self.conn.commit()
            print(f"✅ 답변 반려 완료 (문의 번호: {inquiry_seq})")
            if reason:
                print(f"📝 반려 사유: {reason}")
            return True
            
        except Exception as e:
            print(f"❌ 답변 반려 실패: {e}")
            print("💡 데이터베이스 연결과 권한을 확인하세요.")
            return False
    
    def process_single_inquiry(self, inquiry_seq: int) -> bool:
        """
        단일 문의를 처리합니다. (OpenAI 모델 사용)
        
        Args:
            inquiry_seq (int): 처리할 문의 번호
            
        Returns:
            bool: 처리 성공 시 True, 실패 시 False
        """
        if not self.conn:
            self.connect_database()
        
        try:
            cursor = self.conn.cursor()
            
            # 문의 내용 조회
            query = """
            SELECT contents, name 
            FROM [mobile].[dbo].[bible_inquiry] 
            WHERE seq = ?
            """
            cursor.execute(query, (inquiry_seq,))
            result = cursor.fetchone()
            
            if not result:
                print(f"❌ 문의 번호 {inquiry_seq}를 찾을 수 없습니다.")
                print("💡 문의 번호를 다시 확인해주세요.")
                return False
            
            question, name = result
            print(f"📝 처리 중: [{inquiry_seq}] {name} - {question[:50]}...")
            print("🤖 OpenAI 고품질 모델(text-embedding-3-small + GPT-4o-mini) 사용 중...")
            
            # AI 답변 생성 (OpenAI 모델 사용)
            ai_answer = self.generate_ai_answer_for_inquiry(inquiry_seq, question)
            
            if ai_answer:
                # DB에 저장 (answer_YN='N' - 관리자 승인 대기)
                success = self.save_ai_answer(inquiry_seq, ai_answer)
                if success:
                    print(f"🎉 문의 {inquiry_seq} 처리 완료 (관리자 승인 대기)")
                    print("💰 OpenAI 고품질 답변으로 고객 만족도 향상!")
                    return True
            
            print(f"❌ 문의 {inquiry_seq} 처리 실패")
            return False
            
        except Exception as e:
            print(f"❌ 문의 처리 중 오류: {e}")
            print("💡 데이터베이스 연결과 AI 모델 상태를 확인하세요.")
            return False

def show_system_info() -> None:
    """시스템 정보를 표시합니다."""
    print("\n🎯 바이블 애플 MSSQL 연동 AI 시스템 (OpenAI 버전)")
    print("=" * 60)
    print("🤖 사용 모델: OpenAI text-embedding-3-small + GPT-4o-mini")
    print("💰 고품질 AI 모델 사용 - 정확하고 자연스러운 답변 제공")
    print("💡 워크플로우: AI 답변 생성 → answer_YN='N' → 관리자 승인 → answer_YN='Y' → 고객 확인 가능")
    print("=" * 60)

def main() -> None:
    """메인 실행 함수 (OpenAI 버전)"""
    try:
        processor = BibleInquiryProcessorOpenAI()
        
        if not processor.connect_database():
            return
        
        show_system_info()
        
        # 명령행 인자로 특정 문의 처리
        if len(sys.argv) > 1:
            try:
                inquiry_seq = int(sys.argv[1])
                processor.process_single_inquiry(inquiry_seq)
                return
            except ValueError:
                print("❌ 문의 번호는 숫자여야 합니다.")
                print("💡 사용법: python 5_mssql_integration.py [문의번호]")
                return
        
        # 메인 메뉴 루프
        while True:
            print("\n📋 메뉴:")
            print("1. 답변 없는 문의 조회")
            print("2. 특정 문의 AI 답변 생성 (OpenAI 모델)")
            print("3. 승인 대기 목록 조회 (answer_YN='N')")
            print("4. 답변 승인 (answer_YN='Y')")
            print("5. 답변 반려")
            print("6. 종료")
            
            choice = input("\n선택하세요: ").strip()
            
            if choice == "1":
                unanswered = processor.get_unanswered_inquiries()
                if not unanswered.empty:
                    print("\n📋 답변 없는 문의들:")
                    for _, row in unanswered.iterrows():
                        print(f"[{row['seq']}] {row['name']} - {row['contents'][:50]}...")
                        
            elif choice == "2":
                seq = input("문의 번호 입력: ").strip()
                if seq.isdigit():
                    print("🤖 OpenAI 고품질 모델(text-embedding-3-small + GPT-4o-mini)로 처리합니다...")
                    processor.process_single_inquiry(int(seq))
                else:
                    print("❌ 올바른 문의 번호를 입력해주세요.")
                    
            elif choice == "3":
                pending = processor.get_pending_confirmations()
                if not pending.empty:
                    print("\n👨‍💼 승인 대기 중인 답변들 (answer_YN='N'):")
                    for _, row in pending.iterrows():
                        print(f"\n[{row['seq']}] {row['name']}")
                        print(f"질문: {row['contents'][:100]}...")
                        print(f"AI 답변: {row['reply_contents'][:150]}...")
                        print("-" * 50)
                        
            elif choice == "4":
                seq = input("승인할 문의 번호: ").strip()
                if seq.isdigit():
                    final_answer = input("수정할 답변 (엔터시 원본 사용): ").strip()
                    processor.confirm_answer(int(seq), final_answer=final_answer if final_answer else None)
                else:
                    print("❌ 올바른 문의 번호를 입력해주세요.")
                    
            elif choice == "5":
                seq = input("반려할 문의 번호: ").strip()
                if seq.isdigit():
                    reason = input("반려 사유 (선택사항): ").strip()
                    processor.reject_answer(int(seq), reason=reason if reason else None)
                else:
                    print("❌ 올바른 문의 번호를 입력해주세요.")
                    
            elif choice == "6":
                print("👋 프로그램을 종료합니다.")
                break
            else:
                print("❌ 올바른 번호를 선택해주세요.")
                
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 프로그램 실행 중 오류 발생: {e}")
        print("💡 설정과 환경변수를 확인하고 다시 시도하세요.")
    finally:
        if 'processor' in locals():
            processor.disconnect_database()

if __name__ == "__main__":
    main()