import os
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
import time
import re
from datetime import datetime
from openai import OpenAI
import html

# 환경 변수 로드
load_dotenv()

# OpenAI 클라이언트 초기화
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Pinecone 초기화 (3072차원용 인덱스)
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index("bible-app-support-3072")

print("✓ OpenAI text-embedding-3-small 모델 사용 준비 완료!")

def clean_html_text(text):
    """HTML 태그와 엔티티를 제거하고 깨끗한 텍스트로 변환"""
    if not text or pd.isna(text):
        return ""
    
    text = str(text)
    
    # HTML 엔티티 디코딩 (&nbsp; → 공백, &lt; → < 등)
    text = html.unescape(text)
    
    # <br>, <p> 태그를 줄바꿈으로 변환
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</p>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<p[^>]*>', '\n', text, flags=re.IGNORECASE)
    
    # <li> 태그는 앞에 "- " 추가
    text = re.sub(r'<li[^>]*>', '\n- ', text, flags=re.IGNORECASE)
    text = re.sub(r'</li>', '', text, flags=re.IGNORECASE)
    
    # <strong>, <b> 태그는 ** 로 변환 (강조 표시)
    text = re.sub(r'<(strong|b)[^>]*>', '**', text, flags=re.IGNORECASE)
    text = re.sub(r'</(strong|b)>', '**', text, flags=re.IGNORECASE)
    
    # 기타 모든 HTML 태그 제거
    text = re.sub(r'<[^>]+>', '', text)
    
    # 연속된 줄바꿈을 정리 (3개 이상 → 2개)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 연속된 공백을 하나로
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 줄의 앞뒤 공백 제거
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(line for line in lines if line)
    
    return text.strip()

def preprocess_text(text):
    """텍스트 전처리 (HTML 태그 제거 포함)"""
    if pd.isna(text):
        return ""
    
    # 문자열로 변환
    text = str(text)
    
    # HTML 태그 제거 및 정리
    text = clean_html_text(text)
    
    # 줄바꿈을 공백으로 변환 (임베딩을 위해)
    text = text.replace('\n', ' ')
    
    # 연속된 공백을 하나로
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 최대 길이 제한
    if len(text) > 8000:
        text = text[:8000]
    
    return text

def create_embedding(text, retry_count=3):
    """텍스트를 벡터로 변환 (OpenAI text-embedding-3-small 사용)"""
    for attempt in range(retry_count):
        try:
            # OpenAI text-embedding-3-small 모델 사용 (3,072차원)
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                encoding_format="float"
            )
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            print(f"  임베딩 생성 실패 (시도 {attempt + 1}/{retry_count}): {e}")
            if attempt < retry_count - 1:
                time.sleep(2 ** attempt)  # 지수적 백오프
            else:
                return None

def categorize_question(question):
    """질문 자동 카테고리 분류"""
    question_lower = question.lower()
    
    if any(word in question_lower for word in ['오디오', '음성', '소리', '들리', '녹음']):
        return '오디오'
    elif any(word in question_lower for word in ['검색', '찾기', '찾을']):
        return '검색'
    elif any(word in question_lower for word in ['로그인', '비밀번호', '아이디', '계정', '가입']):
        return '계정'
    elif any(word in question_lower for word in ['구독', '결제', '요금', '환불']):
        return '구독'
    elif any(word in question_lower for word in ['오류', '에러', '버그', '종료', '멈춤', '느려']):
        return '오류'
    elif any(word in question_lower for word in ['설정', '알림', '푸시']):
        return '설정'
    elif any(word in question_lower for word in ['통독', '읽기', '성경']):
        return '성경'
    else:
        return '일반'

def upload_bible_data(batch_size=20, max_items=None):
    """data_100.csv 파일을 Pinecone에 업로드 (HTML 태그 제거 포함)"""
    
    print("=" * 60)
    print("바이블 애플 샘플 데이터 업로드 시작 (data_100.csv)")
    print("HTML 태그 제거 및 텍스트 정리 포함")
    print("=" * 60)
    
    # 데이터 읽기 - data_100.csv로 변경
    print("\n📖 'data_100.csv' 파일 읽는 중...")
    try:
        # 여러 인코딩 시도
        encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv('data_100.csv', encoding=encoding)  # 파일명 변경
                print(f"✓ 인코딩 '{encoding}'으로 파일 읽기 성공")
                break
            except:
                continue
        
        if df is None:
            raise Exception("data_100.csv 파일을 읽을 수 없습니다. 파일이 존재하는지 확인해주세요.")
            
    except Exception as e:
        print(f"❌ 파일 읽기 오류: {e}")
        return
    
    print(f"✓ 총 {len(df)}개 데이터 발견")
    
    # 컬럼명 확인
    print(f"✓ 컬럼: {df.columns.tolist()}")
    
    # 데이터 전처리
    print("\n🔧 데이터 전처리 중 (HTML 태그 제거)...")
    
    # 전처리 전 샘플 출력
    if not df.empty:
        print("\n📝 전처리 전 샘플:")
        sample_reply = df['reply_contents'].iloc[0]
        print(f"원본: {sample_reply[:150]}...")
        
        # 전처리 적용
        df['contents'] = df['contents'].apply(preprocess_text)
        df['reply_contents'] = df['reply_contents'].apply(preprocess_text)
        
        print("\n📝 전처리 후 샘플:")
        cleaned_reply = df['reply_contents'].iloc[0]
        print(f"정리됨: {cleaned_reply[:150]}...")
    
    # 빈 값 제거
    df = df[(df['contents'] != '') & (df['reply_contents'] != '')]
    
    # 테스트를 위해 데이터 제한
    if max_items and len(df) > max_items:
        df = df.head(max_items)
        print(f"✓ 테스트를 위해 {max_items}개로 제한")
    
    print(f"✓ 유효한 데이터: {len(df)}개")
    
    # 업로드 시작
    print(f"\n📤 Pinecone 업로드 시작...")
    print(f"배치 크기: {batch_size}개")
    
    vectors_to_upsert = []
    success_count = 0
    failed_count = 0
    start_time = datetime.now()
    
    for idx, row in df.iterrows():
        # 진행 상황 표시
        if idx % 10 == 0:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            if idx > 0:
                avg_time_per_item = elapsed_time / idx
                remaining_items = len(df) - idx
                estimated_remaining = avg_time_per_item * remaining_items
                print(f"\n진행: {idx}/{len(df)} ({idx/len(df)*100:.1f}%) | "
                      f"성공: {success_count} | 실패: {failed_count} | "
                      f"예상 남은 시간: {estimated_remaining/60:.1f}분")
        
        # 질문 벡터화 (OpenAI 사용)
        embedding = create_embedding(row['contents'])
        
        if embedding is None:
            failed_count += 1
            continue
        
        # 카테고리 자동 분류
        category = categorize_question(row['contents'])
        
        # 메타데이터 구성
        metadata = {
            "seq": int(row['seq']),
            "question": row['contents'][:1000],  # 메타데이터 크기 제한
            "answer": row['reply_contents'][:1000],  # 메타데이터 크기 제한
            "category": category,
            "source": "data_100_sample_clean"
        }
        
        # 고유 ID 생성 (seq 사용)
        unique_id = f"qa_sample100_{row['seq']}"
        
        # 벡터 데이터 구성
        vectors_to_upsert.append({
            "id": unique_id,
            "values": embedding,
            "metadata": metadata
        })
        
        # 배치 크기에 도달하면 업로드
        if len(vectors_to_upsert) >= batch_size:
            try:
                index.upsert(vectors=vectors_to_upsert)
                success_count += len(vectors_to_upsert)
                print(f"  ✓ {len(vectors_to_upsert)}개 벡터 업로드 완료")
                vectors_to_upsert = []
                time.sleep(1)  # API 제한 방지
            except Exception as e:
                print(f"  ❌ 업로드 오류: {e}")
                failed_count += len(vectors_to_upsert)
                vectors_to_upsert = []
    
    # 남은 벡터 업로드
    if vectors_to_upsert:
        try:
            index.upsert(vectors=vectors_to_upsert)
            success_count += len(vectors_to_upsert)
            print(f"  ✓ 마지막 {len(vectors_to_upsert)}개 벡터 업로드 완료")
        except Exception as e:
            print(f"  ❌ 마지막 배치 업로드 오류: {e}")
            failed_count += len(vectors_to_upsert)
    
    # 최종 통계
    total_time = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 60)
    print("📊 업로드 완료 통계")
    print("=" * 60)
    print(f"✓ 성공: {success_count}개")
    print(f"✗ 실패: {failed_count}개")
    print(f"⏱ 총 소요 시간: {total_time/60:.1f}분")
    print(f"💾 평균 처리 속도: {success_count/(total_time/60):.1f}개/분")
    
    # Pinecone 인덱스 통계
    print("\n📈 Pinecone 인덱스 상태:")
    stats = index.describe_index_stats()
    print(f"총 벡터 수: {stats['total_vector_count']}")
    
    print("\n✅ data_100.csv 업로드가 완료되었습니다!")
    print("📝 HTML 태그가 모두 제거되고 깨끗한 텍스트로 변환되었습니다.")

if __name__ == "__main__":
    # 확인 메시지
    print("바이블 애플 샘플 데이터 업로드를 시작하시겠습니까?")
    print(f"파일: data_100.csv (총 {len(pd.read_csv('data_100.csv'))}개 데이터)")
    print(f"HTML 태그 제거 및 텍스트 정리 포함")
    print(f"모델: OpenAI text-embedding-3-small (3,072차원)")
    print(f"예상 시간: 약 5-10분")
    print(f"비용: OpenAI API 사용 (소량 요금 발생)")
    print("\n계속하려면 Enter, 취소하려면 Ctrl+C를 누르세요...")
    input()
    
    # 업로드 실행
    upload_bible_data(batch_size=20, max_items=None)
