"""
바이블 애플 AI 애플리케이션 초기 설정 스크립트 (무료 모델 버전)

이 스크립트는 바이블 애플 AI 애플리케이션에서 사용할 Pinecone 벡터 데이터베이스와
sentence-transformers 무료 모델을 초기화합니다.

주요 기능:
1. Pinecone 벡터 데이터베이스 연결
2. 무료 다국어 임베딩 모델 로드 및 테스트
3. 768차원 벡터 인덱스 생성 또는 연결
4. 시스템 상태 확인

"""

import os # 파일 경로 처리 파이썬 모듈
import sys # 시스템 관련 작업 파이썬 모듈
from typing import Optional # 타입 힌트 파이썬 모듈
from dotenv import load_dotenv # 환경변수 처리 파이썬 모듈
from pinecone import Pinecone # Pinecone 파이썬 모듈
from sentence_transformers import SentenceTransformer # 임베딩 모델 파이썬 모듈

# ====== 설정 상수 ======
# 사용할 임베딩 모델 이름 (다국어 지원, 768차원 출력)
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
# Pinecone 인덱스 이름 (768차원 무료 모델용)
INDEX_NAME = "bible-app-support-768-free"
# 임베딩 벡터의 차원 수
EMBEDDING_DIMENSION = 768
# Pinecone 클라우드 설정
CLOUD_PROVIDER = "aws"
CLOUD_REGION = "us-east-1" 

# 1. 환경변수 로드
def load_environment_variables() -> None:
    """
    .env 파일에서 환경변수를 로드합니다.
    
    필요한 환경변수:
    - PINECONE_API_KEY: Pinecone API 키
    """
    print("🔐 환경변수 로드 중...")
    load_dotenv()
    
    # API 키 존재 여부 확인
    if not os.getenv('PINECONE_API_KEY'):
        print("❌ PINECONE_API_KEY가 .env 파일에 설정되지 않았습니다.")
        print("💡 .env 파일에 PINECONE_API_KEY=your_api_key를 추가하세요.")
        sys.exit(1)
    
    print("✓ 환경변수 로드 완료!")

# 2. Pinecone 클라이언트 초기화
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

# 3. 벡터 임베딩 모델 로드 및 테스트
def load_and_test_model() -> SentenceTransformer:
    """
    sentence-transformers 모델을 로드하고 테스트합니다.
    
    Returns:
        SentenceTransformer: 로드된 임베딩 모델
        
    Raises:
        SystemExit: 모델 로드 실패 시
    """
    print(f"📦 {MODEL_NAME} 모델 로드 중...")
    try:
        # 다국어 지원 sentence-transformers 모델 로드
        model = SentenceTransformer(MODEL_NAME)
        print("✓ sentence-transformers 모델 로드 완료!")
        
        # 모델 테스트: 한국어 문장으로 임베딩 생성
        test_text = "테스트 문장입니다."
        test_embedding = model.encode(test_text)
        actual_dimension = len(test_embedding)
        
        print(f"✓ 임베딩 차원 확인: {actual_dimension}차원")
        
        # 예상 차원과 일치하는지 확인
        if actual_dimension != EMBEDDING_DIMENSION:
            print(f"⚠️ 경고: 예상 차원({EMBEDDING_DIMENSION})과 실제 차원({actual_dimension})이 다릅니다.")
        
        return model
        
    except ImportError:
        print("❌ sentence-transformers 패키지가 설치되지 않았습니다.")
        print("💡 다음 명령으로 설치하세요: pip install sentence-transformers")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        print("💡 인터넷 연결을 확인하고 다시 시도하세요.")
        sys.exit(1)

# 4. Pinecone 인덱스 생성 또는 연결
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
                dimension=EMBEDDING_DIMENSION,  # 768차원 벡터
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

# 5. 인덱스 연결 테스트
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
    print("🚀 바이블 애플 AI 애플리케이션 초기 설정 시작")
    print("📱 무료 sentence-transformers 모델 버전 (768차원)")
    print("=" * 60)
    
    try:
        # 1. 환경변수 로드
        load_environment_variables()
        
        # 2. Pinecone 클라이언트 초기화
        pc = initialize_pinecone()
        
        # 3. sentence-transformers 모델 로드 및 테스트
        model = load_and_test_model()
        
        # 4. Pinecone 인덱스 생성 또는 연결
        create_or_get_index(pc)
        
        # 5. 인덱스 연결 테스트
        test_index_connection(pc)
        
        # 설정 완료 메시지
        print("\n" + "=" * 60)
        print("🎉 바이블 애플 AI 애플리케이션 설정 완료!")
        print("💰 OpenAI API 비용 없이 무료로 사용 가능합니다.")
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