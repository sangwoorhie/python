"""
Bible AI 애플리케이션 데이터 업로드 스크립트 (OpenAI 모델 버전)

이 스크립트는 Bible AI Q&A 데이터를 Pinecone 벡터 데이터베이스에 업로드합니다.
OpenAI text-embedding-3-small 모델을 사용하여 고품질 임베딩을 생성합니다.

주요 기능:
1. CSV 파일 데이터 읽기 및 전처리 (HTML 태그 제거)
2. OpenAI text-embedding-3-small 모델로 임베딩 생성
3. 질문 자동 카테고리 분류
4. Pinecone 벡터 데이터베이스 배치 업로드
5. 진행 상황 모니터링 및 통계 제공

데이터 구조:
- seq: 고유 식별자
- contents: 질문 내용 
- reply_contents: 답변 내용

작성자: Bible AI Team
버전: 1.0
마지막 수정: 2024
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
import time
import re
from datetime import datetime
from openai import OpenAI
import html
from typing import Optional, List, Dict, Any

# ====== 설정 상수 ======
# 사용할 임베딩 모델 이름 (OpenAI text-embedding-3-small)
MODEL_NAME = "text-embedding-3-small"
# Pinecone 인덱스 이름
INDEX_NAME = "bible-app-support-3072"
# 데이터 파일명
DATA_FILE = "data_100.csv"
# 임베딩 벡터 차원
EMBEDDING_DIMENSION = 3072
# 기본 배치 크기
DEFAULT_BATCH_SIZE = 20
# 텍스트 최대 길이
MAX_TEXT_LENGTH = 8000
# 메타데이터 텍스트 최대 길이
MAX_METADATA_LENGTH = 1000

def initialize_services() -> tuple[Pinecone, Any, OpenAI]:
    """
    필요한 서비스들을 초기화합니다.
    
    Returns:
        tuple: (Pinecone 클라이언트, 인덱스, OpenAI 클라이언트)
        
    Raises:
        SystemExit: 초기화 실패 시
    """
    print("🔐 환경변수 로드 중...")
    load_dotenv()
    
    # API 키 확인
    missing_keys = []
    if not os.getenv('PINECONE_API_KEY'):
        missing_keys.append('PINECONE_API_KEY')
    if not os.getenv('OPENAI_API_KEY'):
        missing_keys.append('OPENAI_API_KEY')
    
    if missing_keys:
        print(f"❌ 다음 API 키들이 .env 파일에 설정되지 않았습니다: {', '.join(missing_keys)}")
        print("💡 .env 파일에 다음과 같이 추가하세요:")
        for key in missing_keys:
            print(f"   {key}=your_api_key")
        sys.exit(1)
    
    print("✓ 환경변수 로드 완료!")
    
    # Pinecone 초기화
    print("🌲 Pinecone 클라이언트 초기화 중...")
    try:
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        index = pc.Index(INDEX_NAME)
        print("✓ Pinecone 연결 완료!")
    except Exception as e:
        print(f"❌ Pinecone 초기화 실패: {e}")
        print("💡 API 키와 인덱스 이름을 확인하세요.")
        sys.exit(1)
    
    # OpenAI 클라이언트 초기화
    print(f"🤖 OpenAI {MODEL_NAME} 모델 준비 중...")
    try:
        openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        print("✓ OpenAI text-embedding-3-small 모델 사용 준비 완료!")
    except Exception as e:
        print(f"❌ OpenAI 클라이언트 초기화 실패: {e}")
        print("💡 OpenAI API 키를 확인하세요.")
        sys.exit(1)
    
    return pc, index, openai_client

def clean_html_text(text: str) -> str:
    """
    HTML 태그와 엔티티를 제거하고 깨끗한 텍스트로 변환합니다.
    
    이 함수는 다음과 같은 HTML 요소들을 처리합니다:
    - HTML 엔티티 디코딩 (&nbsp;, &lt; 등)
    - 구조적 태그를 적절한 텍스트로 변환 (<br> → 줄바꿈, <li> → 리스트 항목)
    - 강조 태그를 마크다운 형식으로 변환 (<strong> → **)
    - 불필요한 공백과 줄바꿈 정리
    
    Args:
        text (str): 정리할 HTML 텍스트
        
    Returns:
        str: 정리된 순수 텍스트
    """
    # 빈 값 처리
    if not text or pd.isna(text):
        return ""
    
    # 문자열로 안전하게 변환
    text = str(text)
    
    # 1. HTML 엔티티 디코딩 (&nbsp; → 공백, &lt; → <, &amp; → & 등)
    text = html.unescape(text)
    
    # 2. 구조적 태그를 의미 있는 텍스트로 변환
    # <br>, <p> 태그를 줄바꿈으로 변환
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</p>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<p[^>]*>', '\n', text, flags=re.IGNORECASE)
    
    # <li> 태그는 리스트 항목으로 변환 (앞에 "- " 추가)
    text = re.sub(r'<li[^>]*>', '\n- ', text, flags=re.IGNORECASE)
    text = re.sub(r'</li>', '', text, flags=re.IGNORECASE)
    
    # 3. 강조 태그를 마크다운 형식으로 변환
    # <strong>, <b> 태그는 ** 로 변환 (강조 표시 유지)
    text = re.sub(r'<(strong|b)[^>]*>', '**', text, flags=re.IGNORECASE)
    text = re.sub(r'</(strong|b)>', '**', text, flags=re.IGNORECASE)
    
    # 4. 기타 모든 HTML 태그 제거
    text = re.sub(r'<[^>]+>', '', text)
    
    # 5. 공백과 줄바꿈 정리
    # 연속된 줄바꿈을 정리 (3개 이상 → 2개)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 연속된 공백을 하나로 통합
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 6. 각 줄의 앞뒤 공백 제거 및 빈 줄 제거
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(line for line in lines if line)
    
    return text.strip()

def preprocess_text(text: str) -> str:
    """
    임베딩 생성을 위한 텍스트 전처리를 수행합니다.
    
    이 함수는 다음과 같은 전처리 과정을 거칩니다:
    1. HTML 태그 제거 및 정리
    2. 줄바꿈을 공백으로 변환 (임베딩 모델을 위해)
    3. 연속된 공백 정리
    4. 텍스트 길이 제한 (모델 입력 제한 고려)
    
    Args:
        text (str): 전처리할 원본 텍스트
        
    Returns:
        str: 전처리된 텍스트
    """
    # 빈 값 처리
    if pd.isna(text) or not text:
        return ""
    
    # 문자열로 안전하게 변환
    text = str(text)
    
    # 1. HTML 태그 제거 및 정리
    text = clean_html_text(text)
    
    # 2. 줄바꿈을 공백으로 변환 (임베딩 모델에서 더 나은 성능을 위해)
    text = text.replace('\n', ' ')
    
    # 3. 연속된 공백을 하나로 통합
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 4. 최대 길이 제한 (토큰 제한 고려)
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]
        print(f"⚠️ 텍스트가 {MAX_TEXT_LENGTH}자로 잘렸습니다.")
    
    return text

def create_embedding(text: str, openai_client: OpenAI, retry_count: int = 3) -> Optional[List[float]]:
    """
    텍스트를 임베딩 벡터로 변환합니다.
    
    OpenAI text-embedding-3-small 모델을 사용하여 텍스트를 3072차원 벡터로 변환합니다.
    네트워크 오류나 일시적 장애에 대비하여 재시도 로직을 포함합니다.
    
    Args:
        text (str): 임베딩으로 변환할 텍스트
        openai_client (OpenAI): OpenAI 클라이언트 인스턴스
        retry_count (int): 최대 재시도 횟수 (기본값: 3)
        
    Returns:
        Optional[List[float]]: 성공 시 3072차원 임베딩 벡터, 실패 시 None
    """
    # 빈 텍스트 처리
    if not text or not text.strip():
        print("⚠️ 빈 텍스트로 인해 임베딩 생성을 건너뜁니다.")
        return None
    
    # 재시도 로직을 포함한 임베딩 생성
    for attempt in range(retry_count):
        try:
            # OpenAI text-embedding-3-small 모델 사용 (3,072차원)
            response = openai_client.embeddings.create(
                model=MODEL_NAME,
                input=text,
                encoding_format="float"
            )
            embedding = response.data[0].embedding
            
            # 차원 검증
            if len(embedding) != EMBEDDING_DIMENSION:
                print(f"⚠️ 예상치 못한 임베딩 차원: {len(embedding)} (예상: {EMBEDDING_DIMENSION})")
            
            return embedding
            
        except Exception as e:
            print(f"  임베딩 생성 실패 (시도 {attempt + 1}/{retry_count}): {e}")
            
            # 마지막 시도가 아니면 대기 후 재시도
            if attempt < retry_count - 1:
                wait_time = 2 ** attempt  # 지수적 백오프 (1초, 2초, 4초...)
                print(f"  {wait_time}초 후 재시도...")
                time.sleep(wait_time)
            else:
                print("  모든 재시도가 실패했습니다.")
                return None

def categorize_question(question: str) -> str:
    """
    질문 내용을 분석하여 자동으로 카테고리를 분류합니다.
    
    bible_inquiry_cate_list 테이블의 카테고리에 맞게 분류합니다:
    - 후원/해지: 후원, 결제, 구독, 해지 관련 질문
    - 성경 통독(읽기,듣기,녹음): 성경 읽기, 듣기, 녹음, 통독 관련 질문
    - 성경낭독 레이스: 성경낭독 레이스, 경쟁, 참여 관련 질문
    - 개선/제안: 개선사항, 제안, 기능 요청 관련 질문
    - 오류/장애: 버그, 에러, 오류, 장애 관련 질문
    - 사용 문의(기타): 일반적인 사용법, 기타 문의
    - 불만: 불만, 불편사항 관련 질문
    - 오탈자제보: 오탈자, 번역오류, 내용오류 제보
    
    Args:
        question (str): 분류할 질문 텍스트
        
    Returns:
        str: 분류된 카테고리명
    """
    # 빈 질문 처리
    if not question or not question.strip():
        return '사용 문의(기타)'
    
    # 대소문자 구분 없이 키워드 매칭을 위해 소문자로 변환
    question_lower = question.lower()
    
    # 카테고리별 키워드 정의 (우선순위대로 배치)
    category_keywords = {
        '후원/해지': [
            '후원', '기부', '결제', '구독', '해지', '취소', '환불', '요금', '유료', 
            '프리미엄', '정기결제', '자동결제', '결제수단', '카드', '계좌', '송금'
        ],
        '성경 통독(읽기,듣기,녹음)': [
            '통독', '읽기', '듣기', '녹음', '성경읽기', '말씀듣기', '음성', '오디오',
            '낭독', '독서', '성경공부', '묵상', '큐티', 'qt', '음성녹음', '재생',
            '독서계획', '성경전체', '구약', '신약', '성경듣기'
        ],
        '성경낭독 레이스': [
            '레이스', '경쟁', '대회', '참여', '순위', '랭킹', '경주', '도전',
            '성경낭독레이스', '낭독대회', '낭독경쟁', '성경암송'
        ],
        '개선/제안': [
            '개선', '제안', '건의', '요청', '바람', '기능추가', '새기능', '업데이트',
            '개발', '추가해주세요', '만들어주세요', '넣어주세요', '개선해주세요',
            '더좋게', '편리하게', '업그레이드'
        ],
        '오류/장애': [
            '오류', '에러', '버그', '문제', '고장', '장애', '안됨', '안되요', 
            '작동안함', '실행안됨', '멈춤', '종료', '느림', '느려', '끊김',
            '로딩', '접속불가', '연결안됨', '다운', '크래시', '튕김'
        ],
        '불만': [
            '불만', '불편', '짜증', '화남', '싫어', '마음에안듬', '별로',
            '실망', '불쾌', '기분나쁨', '서비스나쁨', '답답', '속상'
        ],
        '오탈자제보': [
            '오탈자', '오타', '오역', '번역오류', '번역틀림', '틀렸', '잘못',
            '내용오류', '성경오류', '구절틀림', '본문틀림', '수정', '정정',
            '잘못된내용', '오류제보', '내용잘못'
        ]
    }
    
    # 각 카테고리별로 키워드 매칭 검사 (우선순위대로)
    for category, keywords in category_keywords.items():
        if any(keyword in question_lower for keyword in keywords):
            return category
    
    # 매칭되는 키워드가 없으면 사용 문의(기타) 카테고리로 분류
    return '사용 문의(기타)'

def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    CSV 파일을 다양한 인코딩으로 시도하여 안전하게 로드합니다.
    
    Args:
        file_path (str): 로드할 CSV 파일 경로
        
    Returns:
        pd.DataFrame: 로드된 데이터프레임
        
    Raises:
        Exception: 모든 인코딩 시도가 실패한 경우
    """
    print(f"\n📖 '{file_path}' 파일 읽는 중...")
    
    # 시도할 인코딩 목록 (한국어 환경에서 일반적인 인코딩)
    encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'latin1']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"✓ 인코딩 '{encoding}'으로 파일 읽기 성공")
            print(f"✓ 총 {len(df)}개 행 발견")
            print(f"✓ 컬럼: {df.columns.tolist()}")
            return df
            
        except UnicodeDecodeError:
            print(f"  인코딩 '{encoding}' 실패, 다음 인코딩 시도...")
            continue
        except Exception as e:
            print(f"  인코딩 '{encoding}' 오류: {e}")
            continue
    
    # 모든 인코딩 시도가 실패한 경우
    raise Exception(f"'{file_path}' 파일을 읽을 수 없습니다. 파일이 존재하고 올바른 CSV 형식인지 확인해주세요.")

def upload_bible_data(batch_size: int = DEFAULT_BATCH_SIZE, max_items: Optional[int] = None) -> None:
    """
    CSV 파일의 Q&A 데이터를 Pinecone 벡터 데이터베이스에 업로드합니다.
    
    이 함수는 다음과 같은 과정을 거쳐 데이터를 처리합니다:
    1. CSV 파일 읽기 (다양한 인코딩 시도)
    2. HTML 태그 제거 및 텍스트 전처리
    3. OpenAI text-embedding-3-small 모델로 임베딩 생성
    4. 질문 자동 카테고리 분류
    5. Pinecone에 배치 업로드
    6. 진행 상황 모니터링 및 통계 제공
    
    Args:
        batch_size (int): 한 번에 업로드할 벡터 수 (기본값: 20)
        max_items (Optional[int]): 테스트용 최대 아이템 수 제한 (기본값: None, 모든 데이터)
    """
    # 서비스 초기화
    pc, index, openai_client = initialize_services()
    
    print("=" * 60)
    print("🚀 Bible AI Q&A 데이터 업로드 시작")
    print(f"📁 파일: {DATA_FILE}")
    print(f"🤖 모델: OpenAI {MODEL_NAME}")
    print(f"📏 차원: {EMBEDDING_DIMENSION}차원")
    print(f"💰 OpenAI API 사용 - 고품질 임베딩 제공")
    print("=" * 60)
    
    # CSV 파일 로드
    try:
        df = load_csv_data(DATA_FILE)
    except Exception as e:
        print(f"❌ 파일 읽기 오류: {e}")
        return
    
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
        embedding = create_embedding(row['contents'], openai_client)
        
        if embedding is None:
            failed_count += 1
            continue
        
        # 카테고리 자동 분류
        category = categorize_question(row['contents'])
        
        # 메타데이터 구성
        metadata = {
            "seq": int(row['seq']),
            "question": row['contents'][:MAX_METADATA_LENGTH],  # 메타데이터 크기 제한
            "answer": row['reply_contents'][:MAX_METADATA_LENGTH],  # 메타데이터 크기 제한
            "category": category,
            "source": "data_100_sample_openai"
        }
        
        # 고유 ID 생성 (seq 사용)
        unique_id = f"qa_openai_{row['seq']}"
        
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
    print("💰 OpenAI text-embedding-3-small 모델 사용으로 고품질 임베딩 제공!")

def main() -> None:
    """
    메인 실행 함수: 사용자 확인 후 데이터 업로드를 실행합니다.
    """
    print("=" * 60)
    print("🚀 Bible AI 샘플 데이터 업로드")
    print("=" * 60)
    print(f"📁 파일: {DATA_FILE}")
    print(f"🤖 모델: OpenAI {MODEL_NAME}")
    print(f"📏 차원: {EMBEDDING_DIMENSION}차원")
    print(f"💰 OpenAI API 사용 - 고품질 임베딩 제공")
    print(f"⏱ 예상 시간: 약 5-10분")
    print(f"💸 비용: OpenAI API 사용 (소량 요금 발생)")
    print("=" * 60)
    
    print("\n업로드를 시작하시겠습니까?")
    print("계속하려면 Enter, 취소하려면 Ctrl+C를 누르세요...")
    
    try:
        input()
        print("\n🚀 업로드를 시작합니다...")
        
        # 업로드 실행
        upload_bible_data(batch_size=DEFAULT_BATCH_SIZE, max_items=None)
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 업로드가 취소되었습니다.")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류가 발생했습니다: {e}")
        print("💡 로그를 확인하고 다시 시도하세요.")

# 스크립트가 직접 실행될 때만 main 함수 호출
if __name__ == "__main__":
    main()