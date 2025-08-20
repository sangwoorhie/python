import os
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# 환경 변수 로드
load_dotenv()

# OpenAI 클라이언트 초기화
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Pinecone 초기화 (3072차원 인덱스)
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index("bible-app-support-3072")

print("✓ OpenAI text-embedding-3-small 모델 사용 준비 완료!")

def create_embedding(text):
    """텍스트를 벡터로 변환 (OpenAI text-embedding-3-small)"""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"임베딩 생성 실패: {e}")
        return None

def search_question(query):
    """질문 검색 테스트 (OpenAI 모델 사용)"""
    
    print(f"\n검색어: '{query}'")
    print("=" * 50)
    
    # 질문을 벡터로 변환 (OpenAI 모델 사용)
    query_vector = create_embedding(query)
    
    if query_vector is None:
        print("❌ 검색 벡터 생성 실패")
        return
    
    # Pinecone에서 검색
    results = index.query(
        vector=query_vector,
        top_k=5,
        include_metadata=True
    )
    
    # 결과 출력
    if not results['matches']:
        print("❌ 검색 결과가 없습니다.")
        return
    
    for i, match in enumerate(results['matches'], 1):
        score = match['score']
        metadata = match['metadata']
        
        print(f"\n🔍 결과 {i}:")
        print(f"유사도: {score:.4f} ({score*100:.1f}%)")
        print(f"카테고리: {metadata.get('category', '미분류')}")
        print(f"질문: {metadata['question']}")
        print(f"답변: {metadata['answer'][:200]}{'...' if len(metadata['answer']) > 200 else ''}")
        print("-" * 30)

if __name__ == "__main__":
    print("🔍 바이블 애플 AI 검색 시스템 테스트")
    print("모델: OpenAI text-embedding-3-small (3,072차원)")
    print("데이터: 100개 샘플 FAQ")
    print()
    
    while True:
        query = input("\n검색할 질문을 입력하세요 (종료: 'quit'): ").strip()
        
        if query.lower() in ['quit', 'exit', '종료']:
            print("👋 테스트를 종료합니다.")
            break
            
        if not query:
            print("❌ 질문을 입력해주세요.")
            continue
        
        search_question(query)
