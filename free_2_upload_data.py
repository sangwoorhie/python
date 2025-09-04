"""
Bible AI 애플리케이션 데이터 업로드 스크립트 (무료 모델 버전)

이 스크립트는 Bible AI Q&A 데이터를 Pinecone 벡터 데이터베이스에 업로드합니다.
sentence-transformers 무료 모델을 사용하여 API 비용 없이 임베딩을 생성합니다.

주요 기능:
1. CSV 파일 데이터 읽기 및 전처리 (HTML 태그 제거)
2. 무료 sentence-transformers 모델로 임베딩 생성
3. 질문 자동 카테고리 분류
4. Pinecone 벡터 데이터베이스 배치 업로드
5. 진행 상황 모니터링 및 통계 제공

데이터 구조:
- seq: 고유 식별자
- contents: 질문 내용 
- reply_contents: 답변 내용
"""

import os # 파일 경로 처리 파이썬 모듈
import sys # 시스템 관련 작업 파이썬 모듈
import pandas as pd # 데이터 처리 파이썬 모듈
from dotenv import load_dotenv # 환경변수 처리 파이썬 모듈
from pinecone import Pinecone # Pinecone 클라이언트 파이썬 모듈
import time # 진행 상황 모니터링
import re # 정규식 검사 파이썬 모듈
from datetime import datetime # 진행 상황 모니터링
from sentence_transformers import SentenceTransformer # 임베딩 모델 파이썬 모듈
import html # HTML 태그 처리 파이썬 모듈
from typing import Optional, List, Dict, Any # 타입 힌트 파이썬 모듈
import unicodedata # 유니코드 문자 처리
import logging # 로그 기록 파이썬 모듈
import openai # OpenAI API 클라이언트

# ====== 설정 상수 ======
MODEL_NAME = 'text-embedding-3-small'
INDEX_NAME = "bible-app-support-1536-openai"
DATA_FILE = "data_2025.csv"
EMBEDDING_DIMENSION = 1536
DEFAULT_BATCH_SIZE = 20
MAX_TEXT_LENGTH = 8000
MAX_METADATA_LENGTH = 1000

# 도메인 특화 중요 키워드 (가중치를 높일 단어들)
DOMAIN_KEYWORDS = set([
    '성경', '찬송가', '구절', '말씀', '기도', '예배', '찬양', '묵상', '큐티',
    '오류', '오타', '버그', '에러', '문제', '수정', '개선', '요청',
    '결제', '구독', '후원', '환불', '취소', '해지',
    '다운로드', '설치', '업데이트', '버전', '앱'
])

# 카테고리별 키워드 정의 (각 카테고리별 키워드들을 리스트로 정의)
CATEGORY_KEYWORDS = {
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

# ★ 함수 1. 필요한 서비스들을 초기화합니다.
# Returns:
# tuple: (Pinecone 클라이언트, 인덱스, OpenAI 클라이언트) 
# Raises:
# SystemExit: 초기화 실패 시
def initialize_services() -> tuple[Pinecone, Any, Any]:
    print(" 환경변수 로드 중...")
    load_dotenv()
    
    # API 키 확인
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    if not pinecone_api_key:
        print("❌ PINECONE_API_KEY가 .env 파일에 설정되지 않았습니다.")
        print("💡 .env 파일에 PINECONE_API_KEY=your_api_key를 추가하세요.")
        sys.exit(1)
    
    if not openai_api_key:
        print("❌ OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
        print("💡 .env 파일에 OPENAI_API_KEY=your_api_key를 추가하세요.")
        sys.exit(1)
    
    print("✓ 환경변수 로드 완료!")
    
    # Pinecone 초기화
    print(" Pinecone 클라이언트 초기화 중...")
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(INDEX_NAME)
        print("✓ Pinecone 연결 완료!")
    except Exception as e:
        print(f"❌ Pinecone 초기화 실패: {e}")
        print(" API 키와 인덱스 이름을 확인하세요.")
        sys.exit(1)
    
    # OpenAI 클라이언트 초기화 (안전한 방식)
    print(f" OpenAI {MODEL_NAME} 모델 준비 중...")
    try:
        # 방법 1: 환경변수 설정 후 기본 초기화
        os.environ['OPENAI_API_KEY'] = openai_api_key
        
        # OpenAI 클라이언트 초기화 (기본 설정만 사용)
        openai_client = openai.OpenAI()
        
        # 간단한 테스트 호출로 연결 확인
        test_response = openai_client.embeddings.create(
            model=MODEL_NAME,
            input="테스트"
        )
        
        print("✓ OpenAI 클라이언트 초기화 및 테스트 완료!")
        print(f"✓ 테스트 임베딩 차원: {len(test_response.data[0].embedding)}")
        
    except Exception as e:
        print(f"❌ OpenAI 클라이언트 초기화 실패: {e}")
        print("💡 OpenAI API 키를 확인하세요.")
        print(f"디버그: API 키 길이: {len(openai_api_key) if openai_api_key else 0}")
        
        # 대안 방법 시도
        try:
            print(" 대안 방법으로 OpenAI 클라이언트 초기화 시도...")
            openai_client = openai.OpenAI(api_key=openai_api_key)
            print("✓ 대안 방법으로 OpenAI 클라이언트 초기화 성공!")
        except Exception as e2:
            print(f"❌ 대안 방법도 실패: {e2}")
            print("💡 OpenAI 라이브러리 버전을 확인하고 다시 설치해보세요.")
            print(" pip install openai==1.3.0")
            sys.exit(1)
    
    return pc, index, openai_client

# ★ 함수 2. 통합 텍스트 전처리 함수
# Args:
#     text (str): 전처리할 원본 텍스트
#     for_metadata (bool): 메타데이터용 전처리 여부      
# Returns:
#     str: 전처리된 텍스트
def preprocess_text(text: str, for_metadata: bool = False) -> str:

    if not text or pd.isna(text):
        return ""
    
    # 1. 기본 전처리
    text = str(text)
    text = html.unescape(text)
    
    # 2. HTML 태그 제거
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</p>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<p[^>]*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<li[^>]*>', '\n- ', text, flags=re.IGNORECASE)
    text = re.sub(r'</li>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<(strong|b)[^>]*>', '**', text, flags=re.IGNORECASE)
    text = re.sub(r'</(strong|b)>', '**', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    
    # 3. 유니코드 정규화
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[\u00A0\u2000-\u200B\u202F\u205F\u3000\uFEFF]', ' ', text)
    
    # 4. 노이즈 제거
    text = re.sub(r'([!?.]){2,}', r'\1', text)
    text = re.sub(r'([ㄱ-ㅎㅏ-ㅣ])\1{3,}', r'\1\1', text)
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)
    text = re.sub(r'\d{2,4}-\d{3,4}-\d{4}', '[PHONE]', text)
    
    # 5. 공백 정리
    if for_metadata:
        # 메타데이터용: 줄바꿈 유지
        text = re.sub(r'\r\n|\r', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(line for line in lines if line)
    else:
        # 임베딩용: 줄바꿈을 공백으로
        text = re.sub(r'\r\n|\r|\n', ' ', text)
        text = text.replace('\t', ' ')
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])\s+', r'\1 ', text)
        text = re.sub(r'\(\s+', '(', text)
        text = re.sub(r'\s+\)', ')', text)
    
    text = text.strip()
    
    # 6. 길이 제한
    max_length = MAX_METADATA_LENGTH if for_metadata else MAX_TEXT_LENGTH
    if len(text) > max_length:
        if for_metadata:
            text = text[:max_length-3] + "..."
        else:
            front_len = int(max_length * 0.6)
            back_len = max_length - front_len
            text = text[:front_len] + " ... " + text[-back_len:]
            print(f"⚠️ 텍스트가 {max_length}자로 조정되었습니다.")
    
    return text

# ★ 함수 3. 도메인 특화 키워드 추출 함수
# Args:
#     text (str): 키워드 추출할 텍스트
# Returns:
#     List[str]: 추출된 키워드 리스트
def extract_keywords(text: str) -> List[str]:
    keywords = []
    
    # 성경 구절 패턴 추출
    bible_verses = re.findall(r'[가-힣]+[서복음기록상하전후편]+\s*\d+[장절:]+\s*\d*', text)
    keywords.extend(bible_verses) # extend: 리스트에 요소를 추가하는 내장 메서드 (반복 가능한 객체의 요소를 하나씩 꺼내서 리스트에 추가)
    
    # 찬송가 번호 추출
    hymn_numbers = re.findall(r'찬송가?\s*\d+장?', text) # findall: 정규식 패턴과 일치하는 모든 부분을 찾아서 리스트로 반환하는 내장 메서드
    keywords.extend(hymn_numbers) 
    
    # 도메인 키워드 추출
    for keyword in DOMAIN_KEYWORDS:
        if keyword in text:
            keywords.append(keyword) # append: 리스트 끝에 매개변수수를 추가하는 내장 메서드
    
    return keywords

# ★ 함수 4. 임베딩 생성 함수
# 텍스트를 OpenAI text-embedding-3-small 모델로 1536차원 벡터로 변환합니다.
# Args:
#     text (str): 임베딩으로 변환할 텍스트
#     openai_client (Any): OpenAI 클라이언트 인스턴스
#     retry_count (int): 최대 재시도 횟수       
# Returns:
#     Optional[List[float]]: 성공 시 1536차원 임베딩 벡터, 실패 시 None
def create_embedding(text: str, openai_client: Any, retry_count: int = 3) -> Optional[List[float]]:

    if not text or not text.strip():
        print("⚠️ 빈 텍스트로 인해 임베딩 생성을 건너뜁니다.")
        return None
    
    # 키워드 강조 처리
    keywords = extract_keywords(text)
    if keywords:
        keyword_str = ' '.join(keywords[:3])
        text = f"{keyword_str} {text}"
    
    # 재시도 로직을 포함한 임베딩 생성
    for attempt in range(retry_count):
        try:
            # OpenAI text-embedding-3-small 모델로 임베딩 생성
            response = openai_client.embeddings.create(
                model=MODEL_NAME,
                input=text
            )
            
            embedding_list = response.data[0].embedding
            
            # 차원 검증
            if len(embedding_list) != EMBEDDING_DIMENSION:
                print(f"⚠️ 예상치 못한 임베딩 차원: {len(embedding_list)} (예상: {EMBEDDING_DIMENSION})")
            
            return embedding_list
            
        except Exception as e:
            print(f"  임베딩 생성 실패 (시도 {attempt + 1}/{retry_count}): {e}")
            
            if attempt < retry_count - 1:
                wait_time = 2 ** attempt
                print(f"  {wait_time}초 후 재시도...")
                time.sleep(wait_time)
            else:
                print("  모든 재시도가 실패했습니다.")
                return None

# ★ 함수 5. 질문 내용을 분석하여 자동으로 카테고리를 분류합니다.
# Args:
#     question (str): 분류할 질문 텍스트
# Returns:
#     str: 분류된 카테고리명
def categorize_question(question: str) -> str:
    if not question or not question.strip():
        return '사용 문의(기타)'
    
    question_lower = question.lower()
    
    # 각 카테고리별로 키워드 매칭 검사
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in question_lower for keyword in keywords):
            return category
    
    return '사용 문의(기타)'

# ★ 함수 6. CSV 파일을 다양한 인코딩으로 시도하여 안전하게 로드합니다.
# Args:
#     file_path (str): 로드할 CSV 파일 경로
# Returns:
#     pd.DataFrame: 로드된 데이터프레임
# Raises:
#     Exception: 모든 인코딩 시도가 실패한 경우
def load_csv_data(file_path: str) -> pd.DataFrame:
    print(f"\n📖 '{file_path}' 파일 읽는 중...")
    
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
    
    raise Exception(f"'{file_path}' 파일을 읽을 수 없습니다. 파일이 존재하고 올바른 CSV 형식인지 확인해주세요.")

# ★ 함수 7. CSV 파일의 Q&A 데이터를 Pinecone 벡터 데이터베이스에 업로드합니다.
# Args:
#     batch_size (int): 한 번에 업로드할 벡터 수
#     max_items (Optional[int]): 테스트용 최대 아이템 수 제한
# Returns:
#     None: 업로드 완료 후 반환 값 없음
def upload_bible_data(batch_size: int = DEFAULT_BATCH_SIZE, max_items: Optional[int] = None) -> None:
    # 서비스 초기화
    pc, index, openai_client = initialize_services()
    
    print("=" * 60)
    print("🚀 Bible AI Q&A 데이터 업로드 시작")
    print(f"📁 파일: {DATA_FILE}")
    print(f"🤖 모델: {MODEL_NAME}")
    print(f"📏 차원: {EMBEDDING_DIMENSION}차원")
    print(f"💰 OpenAI 유료 모델 사용 - 더 정확한 의미 검색!")
    print("=" * 60)
    
    # 데이터 읽기
    try:
        df = load_csv_data(DATA_FILE)
    except Exception as e:
        print(f"❌ 파일 읽기 오류: {e}")
        return
    
    # 데이터 전처리
    print("\n🔧 데이터 전처리 중...")
    
    if not df.empty:
        print("\n📝 전처리 전 샘플:")
        sample_reply = df['reply_contents'].iloc[0]
        print(f"원본: {sample_reply[:150]}...")
        
        # 전처리 적용
        df['contents'] = df['contents'].apply(lambda x: preprocess_text(x, for_metadata=False))
        df['reply_contents'] = df['reply_contents'].apply(lambda x: preprocess_text(x, for_metadata=False))
        
        print("\n📝 전처리 후 샘플:")
        cleaned_reply = df['reply_contents'].iloc[0]
        print(f"정리됨: {cleaned_reply[:150]}...")
    
    # 빈 값 제거
    df = df[(df['contents'] != '') & (df['reply_contents'] != '')]
    
    # 테스트용 데이터 제한
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
        
        # 질문 벡터화 (OpenAI 클라이언트 사용)
        embedding = create_embedding(row['contents'], openai_client)
        
        if embedding is None:
            failed_count += 1
            continue
        
        # 카테고리 자동 분류
        category = categorize_question(row['contents'])
        
        # 메타데이터 구성 (메타데이터용 전처리 적용)
        metadata = {
            "seq": int(row['seq']),
            "question": preprocess_text(row['contents'], for_metadata=True),
            "answer": preprocess_text(row['reply_contents'], for_metadata=True),
            "category": category,
            "source": "data_2025_sample_free"
        }
        
        # 고유 ID 생성
        unique_id = f"qa_free_{row['seq']}"
        
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
    
    if total_time > 0:
        print(f"💾 평균 처리 속도: {success_count/(total_time/60):.1f}개/분")
    
    # Pinecone 인덱스 통계
    try:
        print("\n📈 Pinecone 인덱스 상태:")
        stats = index.describe_index_stats()
        print(f"총 벡터 수: {stats['total_vector_count']}")
    except Exception as e:
        print(f"📈 인덱스 상태 조회 실패: {e}")
    
    print(f"\n✅ {DATA_FILE} 업로드가 완료되었습니다!")
    print("💰 OpenAI 유료 모델 사용으로 API 비용 없음!")

def main() -> None:
    """
    메인 실행 함수: 사용자 확인 후 데이터 업로드를 실행합니다.
    """
    print("=" * 60)
    print("🚀 Bible AI 샘플 데이터 업로드")
    print("=" * 60)
    print(f"📁 파일: {DATA_FILE}")
    print(f"🤖 모델: {MODEL_NAME}")
    print(f"📏 차원: {EMBEDDING_DIMENSION}차원")
    print(f"💰 OpenAI 유료 모델 사용 - 더 정확한 의미 검색!")
    print(f"⏱ 예상 시간: 약 3-5분")
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

if __name__ == "__main__":
    main()
