import os
from dotenv import load_dotenv
from pinecone import Pinecone

# 환경 변수 로드
load_dotenv()

# Pinecone 초기화
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# 인덱스 이름 (OpenAI text-embedding-3-small용)
INDEX_NAME = "bible-app-support-3072"

print("Pinecone 설정 시작 (OpenAI text-embedding-3-small용 3072차원)...")

# 기존 인덱스 확인
existing_indexes = pc.list_indexes().names()
print(f"기존 인덱스: {existing_indexes}")

# 인덱스가 없으면 생성
if INDEX_NAME not in existing_indexes:
    print(f"'{INDEX_NAME}' 인덱스 생성 중...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=3072,  # OpenAI text-embedding-3-small 차원
        metric='cosine',
        spec={
            "serverless": {
                "cloud": "aws",
                "region": "us-east-1"
            }
        }
    )
    print("✓ 인덱스 생성 완료!")
else:
    print(f"'{INDEX_NAME}' 인덱스가 이미 존재합니다.")

# 인덱스 연결 테스트
index = pc.Index(INDEX_NAME)
stats = index.describe_index_stats()
print(f"인덱스 상태: {stats}")
