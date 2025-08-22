"""
Bible AI 애플리케이션 초기 설정 스크립트 (OpenAI 모델 버전)

이 스크립트는 Bible AI 애플리케이션에서 사용할 Pinecone 벡터 데이터베이스와
OpenAI text-embedding-3-small 모델을 초기화합니다.

주요 기능:
1. Pinecone 벡터 데이터베이스 연결
2. OpenAI API 키 확인 및 모델 테스트
3. 3072차원 벡터 인덱스 생성 또는 연결
4. 시스템 상태 확인

"""

import os
import sys
from typing import Optional
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# ====== 설정 상수 ======
# 사용할 임베딩 모델 이름 (OpenAI text-embedding-3-small)
MODEL_NAME = "text-embedding-3-small"
# Pinecone 인덱스 이름 (3072차원 OpenAI 모델용)
INDEX_NAME = "bible-app-support-3072"
# 임베딩 벡터의 차원 수
EMBEDDING_DIMENSION = 3072
# Pinecone 클라우드 설정
CLOUD_PROVIDER = "aws"
CLOUD_REGION = "us-east-1"

def load_environment_variables() -> None:
    """
    .env 파일에서 환경변수를 로드합니다.
    
    필요한 환경변수:
    - PINECONE_API_KEY: Pinecone API 키
    - OPENAI_API_KEY: OpenAI API 키
    """
    print("🔐 환경변수 로드 중...")
    load_dotenv()
    
    # API 키 존재 여부 확인
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

def initialize_pinecone() -> Pinecone:
    """
    Pinecone 클라이언트를 초기화합니다.
    
    Returns:
        Pinecone: 초기화된 Pinecone 클라이언트
    """
    print("🌲 Pinecone 클라이언트 초기화 중...")
    try:
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        print("✓ Pinecone 클라이언트 초기화 완료!")
        return pc
    except Exception as e:
        print(f"❌ Pinecone 초기화 실패: {e}")
        print("💡 API 키가 올바른지 확인하세요.")
        sys.exit(1)

def test_openai_api() -> OpenAI:
    """
    OpenAI API 연결을 테스트합니다.
    
    Returns:
        OpenAI: 초기화된 OpenAI 클라이언트
        
    Raises:
        SystemExit: API 테스트 실패 시
    """
    print(f"🤖 OpenAI {MODEL_NAME} 모델 테스트 중...")
    try:
        # OpenAI 클라이언트 초기화
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # 테스트용 임베딩 생성
        test_text = "테스트 문장입니다."
        response = client.embeddings.create(
            model=MODEL_NAME,
            input=test_text,
            encoding_format="float"
        )
        
        embedding = response.data[0].embedding
        actual_dimension = len(embedding)
        
        print(f"✓ OpenAI API 연결 성공!")
        print(f"✓ 임베딩 차원 확인: {actual_dimension}차원")
        
        # 예상 차원과 일치하는지 확인
        if actual_dimension != EMBEDDING_DIMENSION:
            print(f"⚠️ 경고: 예상 차원({EMBEDDING_DIMENSION})과 실제 차원({actual_dimension})이 다릅니다.")
        
        return client
        
    except Exception as e:
        print(f"❌ OpenAI API 테스트 실패: {e}")
        print("💡 API 키가 올바르고 충분한 크레딧이 있는지 확인하세요.")
        sys.exit(1)

def create_or_get_index(pc: Pinecone) -> None:
    """
    Pinecone 인덱스를 생성하거나 기존 인덱스에 연결합니다.
    
    Args:
        pc (Pinecone): 초기화된 Pinecone 클라이언트
    """
    print("📋 기존 인덱스 확인 중...")
    
    try:
        # 현재 계정의 모든 인덱스 목록 조회
        existing_indexes = pc.list_indexes().names()
        print(f"기존 인덱스: {existing_indexes}")
        
        # 대상 인덱스가 없으면 새로 생성
        if INDEX_NAME not in existing_indexes:
            print(f"🏗️ '{INDEX_NAME}' 인덱스 생성 중...")
            
            pc.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,  # 3072차원 벡터
                metric='cosine',  # 코사인 유사도 사용 (텍스트 임베딩에 최적)
                spec={
                    "serverless": {  # 서버리스 모드 (비용 효율적)
                        "cloud": CLOUD_PROVIDER,
                        "region": CLOUD_REGION
                    }
                }
            )
            print("✓ 인덱스 생성 완료!")
            
        else:
            print(f"✓ '{INDEX_NAME}' 인덱스가 이미 존재합니다.")
            
    except Exception as e:
        print(f"❌ 인덱스 생성/조회 실패: {e}")
        print("💡 Pinecone 대시보드에서 인덱스 상태를 확인하세요.")
        sys.exit(1)

def test_index_connection(pc: Pinecone) -> None:
    """
    생성된 인덱스에 연결하고 상태를 확인합니다.
    
    Args:
        pc (Pinecone): 초기화된 Pinecone 클라이언트
    """
    print("🔗 인덱스 연결 테스트 중...")
    
    try:
        # 인덱스 객체 생성
        index = pc.Index(INDEX_NAME)
        
        # 인덱스 통계 정보 조회
        stats = index.describe_index_stats()
        
        print("✓ 인덱스 연결 성공!")
        print(f"📊 인덱스 상태:")
        print(f"   - 총 벡터 수: {stats.get('total_vector_count', 0)}")
        print(f"   - 차원: {stats.get('dimension', 'N/A')}")
        print(f"   - 인덱스 용량: {stats.get('index_fullness', 0):.2%}")
        
    except Exception as e:
        print(f"❌ 인덱스 연결 실패: {e}")
        print("💡 잠시 후 다시 시도하세요. (인덱스 생성 직후에는 연결이 지연될 수 있습니다)")
        sys.exit(1)

def main() -> None:
    """
    메인 실행 함수: 전체 설정 프로세스를 순차적으로 실행합니다.
    """
    print("=" * 60)
    print("🚀 Bible AI 애플리케이션 초기 설정 시작")
    print("💰 OpenAI text-embedding-3-small 모델 버전 (3072차원)")
    print("=" * 60)
    
    try:
        # 1. 환경변수 로드
        load_environment_variables()
        
        # 2. Pinecone 클라이언트 초기화
        pc = initialize_pinecone()
        
        # 3. OpenAI API 테스트
        openai_client = test_openai_api()
        
        # 4. Pinecone 인덱스 생성 또는 연결
        create_or_get_index(pc)
        
        # 5. 인덱스 연결 테스트
        test_index_connection(pc)
        
        # 설정 완료 메시지
        print("\n" + "=" * 60)
        print("🎉 Bible AI 애플리케이션 설정 완료!")
        print("💰 OpenAI API 사용으로 고품질 임베딩 제공")
        print("📚 이제 성경 데이터를 업로드하고 검색 기능을 테스트할 수 있습니다.")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 설정이 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류가 발생했습니다: {e}")
        print("💡 로그를 확인하고 다시 시도하세요.")
        sys.exit(1)

# 스크립트가 직접 실행될 때만 main 함수 호출
if __name__ == "__main__":
    main()