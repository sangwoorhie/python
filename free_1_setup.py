import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# 환경 변수 로드
load_dotenv()

# Pinecone 초기화
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# 인덱스 이름 (sentence-transformers/paraphrase-multilingual-mpnet-base-v2용)
INDEX_NAME = "bible-app-support-768-free"

print("Pinecone 설정 시작 (sentence-transformers 무료 모델용 768차원)...")

# 모델 로드 테스트
print("📦 sentence-transformers 모델 로드 중...")
try:
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    print("✓ sentence-transformers 모델 로드 완료!")
    
    # 테스트 임베딩
    test_embedding = model.encode("테스트 문장입니다.")
    print(f"✓ 임베딩 차원 확인: {len(test_embedding)}차원")
    
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    print("💡 다음 명령으로 설치하세요: pip install sentence-transformers")
    exit(1)

# 기존 인덱스 확인
existing_indexes = pc.list_indexes().names()
print(f"기존 인덱스: {existing_indexes}")

# 인덱스가 없으면 생성
if INDEX_NAME not in existing_indexes:
    print(f"'{INDEX_NAME}' 인덱스 생성 중...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,  # sentence-transformers 모델 차원
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

print("\n🎉 무료 sentence-transformers 모델 설정 완료!")
print("💰 OpenAI API 비용 없이 사용 가능합니다.")