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

# 환경 변수 로드
load_dotenv()

class BibleInquiryProcessorFree:
    """바이블 애플 문의 처리 시스템 (무료 임베딩 모델 사용)"""
    
    def __init__(self):
        # 환경변수에서 MSSQL 연결 정보 가져오기
        server = os.getenv('MSSQL_SERVER')
        database = os.getenv('MSSQL_DATABASE')
        username = os.getenv('MSSQL_USERNAME')
        password = os.getenv('MSSQL_PASSWORD')
        
        if not all([server, database, username, password]):
            missing_vars = []
            if not server: missing_vars.append('MSSQL_SERVER')
            if not database: missing_vars.append('MSSQL_DATABASE')
            if not username: missing_vars.append('MSSQL_USERNAME')
            if not password: missing_vars.append('MSSQL_PASSWORD')
            
            raise ValueError(f"다음 환경변수들이 설정되지 않았습니다: {', '.join(missing_vars)}")
        
        self.connection_string = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"UID={username};"
            f"PWD={password};"
            f"TrustServerCertificate=yes;"
        )
        self.conn = None
    
    def connect_database(self):
        """MSSQL 데이터베이스 연결"""
        try:
            self.conn = pyodbc.connect(self.connection_string)
            print("✅ MSSQL 데이터베이스 연결 성공")
            return True
        except Exception as e:
            print(f"❌ 데이터베이스 연결 실패: {e}")
            return False
    
    def disconnect_database(self):
        """데이터베이스 연결 해제"""
        if self.conn:
            self.conn.close()
            print("🔌 데이터베이스 연결 해제")
    
    def get_unanswered_inquiries(self, limit=10):
        """답변이 없는 문의들 조회"""
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
            return pd.DataFrame()
    
    def generate_ai_answer_for_inquiry(self, inquiry_seq, question):
        """특정 문의에 대해 AI 답변 생성 (무료 모델 사용)"""
        try:
            # 무료 버전 Python 스크립트 실행
            result = subprocess.run([
                'python', 'free_4_ai_answer_generator.py',
                '--question', question,
                '--output', 'json'
            ], capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode == 0:
                response = json.loads(result.stdout)
                if response["success"]:
                    return response["answer"]
                else:
                    print(f"AI 답변 생성 실패: {response['error']}")
                    return None
            else:
                print(f"스크립트 실행 실패: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"AI 답변 생성 중 오류: {e}")
            return None
    
    def save_ai_answer(self, inquiry_seq, ai_answer):
        """생성된 AI 답변을 DB에 저장 (answer_YN='N' - 관리자 승인 대기)"""
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
            print("💰 완전 무료 모델 사용으로 모든 API 비용 없음!")
            return True
            
        except Exception as e:
            print(f"❌ AI 답변 저장 실패: {e}")
            return False
    
    def get_pending_confirmations(self):
        """관리자 승인 대기 중인 답변들 조회 (answer_YN='N')"""
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
            return pd.DataFrame()
    
    def confirm_answer(self, inquiry_seq, admin_name=None, final_answer=None):
        """관리자: 답변 승인 (answer_YN='Y'로 변경)"""
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
            else:
                # 기존 답변 승인
                update_query = """
                UPDATE [mobile].[dbo].[bible_inquiry] 
                SET answer_YN = 'Y'
                WHERE seq = ?
                """
                cursor.execute(update_query, (inquiry_seq,))
            
            self.conn.commit()
            print(f"✅ 답변 승인 완료 (문의 번호: {inquiry_seq}) - 고객이 답변을 볼 수 있습니다")
            return True
            
        except Exception as e:
            print(f"❌ 답변 승인 실패: {e}")
            return False
    
    def reject_answer(self, inquiry_seq, reason=None):
        """관리자: 답변 반려 (reply_contents 초기화)"""
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
                print(f"반려 사유: {reason}")
            return True
            
        except Exception as e:
            print(f"❌ 답변 반려 실패: {e}")
            return False
    
    def process_single_inquiry(self, inquiry_seq):
        """단일 문의 처리 (무료 모델 사용)"""
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
                return False
            
            question, name = result
            print(f"📝 처리 중: [{inquiry_seq}] {name} - {question[:50]}...")
            print("💰 완전 무료 모델(sentence-transformers + T5) 사용 중...")
            
            # AI 답변 생성 (무료 모델 사용)
            ai_answer = self.generate_ai_answer_for_inquiry(inquiry_seq, question)
            
            if ai_answer:
                # DB에 저장 (answer_YN='N' - 관리자 승인 대기)
                success = self.save_ai_answer(inquiry_seq, ai_answer)
                if success:
                    print(f"✅ 문의 {inquiry_seq} 처리 완료 (관리자 승인 대기)")
                    return True
            
            print(f"❌ 문의 {inquiry_seq} 처리 실패")
            return False
            
        except Exception as e:
            print(f"❌ 문의 처리 중 오류: {e}")
            return False

def main():
    """메인 실행 함수 (무료 버전)"""
    processor = BibleInquiryProcessorFree()
    
    if not processor.connect_database():
        return
    
    try:
        print("\n🎯 바이블 애플 MSSQL 연동 AI 시스템 (무료 버전)")
        print("=" * 60)
        print("💰 완전 무료 모델 사용 - 모든 API 비용 없음! (sentence-transformers + T5)")
        print("💡 로직: AI 답변 생성 → answer_YN='N' → 관리자 승인 → answer_YN='Y' → 고객 확인 가능")
        
        # 명령행 인자로 특정 문의 처리
        if len(sys.argv) > 1:
            try:
                inquiry_seq = int(sys.argv[1])
                processor.process_single_inquiry(inquiry_seq)
                return
            except ValueError:
                print("❌ 문의 번호는 숫자여야 합니다.")
                return
        
        while True:
            print("\n📋 메뉴:")
            print("1. 답변 없는 문의 조회")
            print("2. 특정 문의 AI 답변 생성 (무료 모델)")
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
                    print("💰 완전 무료 모델(sentence-transformers + T5)로 처리합니다...")
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
                break
            else:
                print("❌ 올바른 번호를 선택해주세요.")
                
    finally:
        processor.disconnect_database()

if __name__ == "__main__":
    main()