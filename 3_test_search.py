"""
Bible AI 애플리케이션 검색 테스트 스크립트 (OpenAI 모델 버전)

이 스크립트는 업로드된 Bible AI Q&A 데이터에서 유사한 질문을 검색하는 
기능을 테스트합니다. OpenAI text-embedding-3-small 모델을 사용하여 
고품질 의미 기반 검색을 수행합니다.

주요 기능:
1. 사용자 질문을 벡터로 변환
2. Pinecone에서 유사한 질문 검색
3. 유사도 점수와 함께 결과 표시
4. 대화형 검색 인터페이스 제공

검색 방식:
- 의미 기반 검색 (Semantic Search)
- 코사인 유사도 기반 매칭
- 상위 5개 결과 반환

작성자: Bible AI Team
버전: 1.0
마지막 수정: 2024
"""

import os
import sys
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# ====== 설정 상수 ======
# 사용할 임베딩 모델 이름 (OpenAI text-embedding-3-small)
MODEL_NAME = "text-embedding-3-small"
# Pinecone 인덱스 이름
INDEX_NAME = "bible-app-support-3072"
# 임베딩 벡터 차원
EMBEDDING_DIMENSION = 3072
# 기본 검색 결과 개수
DEFAULT_TOP_K = 5
# 답변 미리보기 최대 길이
ANSWER_PREVIEW_LENGTH = 200
# 유사도 임계값 (이 값 이하는 관련성이 낮은 것으로 판단)
SIMILARITY_THRESHOLD = 0.3

def initialize_services() -> tuple[Any, OpenAI]:
    """
    필요한 서비스들을 초기화합니다.
    
    Returns:
        tuple: (Pinecone 인덱스, OpenAI 클라이언트)
        
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
    print("🌲 Pinecone 연결 중...")
    try:
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        index = pc.Index(INDEX_NAME)
        
        # 인덱스 상태 확인
        stats = index.describe_index_stats()
        total_vectors = stats.get('total_vector_count', 0)
        
        print(f"✓ Pinecone 연결 완료! (총 {total_vectors}개 벡터)")
        
        if total_vectors == 0:
            print("⚠️ 경고: 인덱스에 데이터가 없습니다.")
            print("💡 먼저 2_upload_data.py를 실행하여 데이터를 업로드하세요.")
            
    except Exception as e:
        print(f"❌ Pinecone 연결 실패: {e}")
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
    
    return index, openai_client

def create_embedding(text: str, openai_client: OpenAI) -> Optional[List[float]]:
    """
    텍스트를 임베딩 벡터로 변환합니다.
    
    OpenAI text-embedding-3-small 모델을 사용하여 입력 텍스트를 3072차원 벡터로 변환합니다.
    이 벡터는 텍스트의 의미적 특성을 수치적으로 표현합니다.
    
    Args:
        text (str): 임베딩으로 변환할 텍스트
        openai_client (OpenAI): OpenAI 클라이언트 인스턴스
        
    Returns:
        Optional[List[float]]: 성공 시 3072차원 임베딩 벡터, 실패 시 None
    """
    # 빈 텍스트 검증
    if not text or not text.strip():
        print("⚠️ 빈 텍스트는 임베딩할 수 없습니다.")
        return None
    
    try:
        # OpenAI text-embedding-3-small 모델로 임베딩 생성
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
        print(f"❌ 임베딩 생성 실패: {e}")
        print("💡 텍스트 형식을 확인하고 다시 시도하세요.")
        return None

def search_question(query: str, index: Any, openai_client: OpenAI, top_k: int = DEFAULT_TOP_K) -> List[Dict]:
    """
    사용자 질문에 대해 유사한 Q&A를 검색합니다.
    
    이 함수는 다음과 같은 과정을 거쳐 검색을 수행합니다:
    1. 사용자 질문을 임베딩 벡터로 변환
    2. Pinecone에서 코사인 유사도 기반 검색 수행
    3. 유사도 점수와 함께 결과 반환
    4. 관련성이 낮은 결과 필터링
    
    Args:
        query (str): 검색할 질문
        index (Any): Pinecone 인덱스 객체
        openai_client (OpenAI): OpenAI 클라이언트
        top_k (int): 반환할 최대 결과 수 (기본값: 5)
        
    Returns:
        List[Dict]: 검색 결과 리스트 (빈 리스트면 결과 없음)
    """
    print(f"\n🔍 검색어: '{query}'")
    print("=" * 60)
    
    # 1. 질문을 벡터로 변환
    print("📊 검색 벡터 생성 중...")
    query_vector = create_embedding(query, openai_client)
    
    if query_vector is None:
        print("❌ 검색 벡터 생성 실패")
        return []
    
    try:
        # 2. Pinecone에서 유사도 검색 수행
        print("🔍 유사한 질문 검색 중...")
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
        
        # 3. 검색 결과 검증
        if not results or 'matches' not in results or not results['matches']:
            print("❌ 검색 결과가 없습니다.")
            print("💡 다른 키워드로 다시 검색해보세요.")
            return []
        
        # 4. 결과 필터링 및 정렬 (관련성이 높은 것만)
        filtered_results = []
        for match in results['matches']:
            if match['score'] >= SIMILARITY_THRESHOLD:
                filtered_results.append(match)
        
        if not filtered_results:
            print(f"❌ 유사도가 {SIMILARITY_THRESHOLD:.1%} 이상인 결과가 없습니다.")
            print("💡 더 구체적인 질문을 시도해보세요.")
            return []
        
        # 5. 결과 표시
        print(f"✓ {len(filtered_results)}개의 관련 결과를 찾았습니다!")
        display_search_results(filtered_results)
        
        return filtered_results
        
    except Exception as e:
        print(f"❌ 검색 중 오류 발생: {e}")
        print("💡 네트워크 연결을 확인하고 다시 시도하세요.")
        return []

def display_search_results(results: List[Dict]) -> None:
    """
    검색 결과를 사용자 친화적인 형식으로 표시합니다.
    
    Args:
        results (List[Dict]): Pinecone 검색 결과 리스트
    """
    for i, match in enumerate(results, 1):
        score = match['score']
        metadata = match.get('metadata', {})
        
        # 유사도 등급 판정
        if score >= 0.8:
            similarity_grade = "🟢 매우 높음"
        elif score >= 0.6:
            similarity_grade = "🟡 높음"
        elif score >= 0.4:
            similarity_grade = "🟠 보통"
        else:
            similarity_grade = "🔴 낮음"
        
        print(f"\n📋 결과 {i}:")
        print(f"   🎯 유사도: {score:.4f} ({score*100:.1f}%) - {similarity_grade}")
        print(f"   📂 카테고리: {metadata.get('category', '미분류')}")
        print(f"   ❓ 질문: {metadata.get('question', 'N/A')}")
        
        # 답변 미리보기 (긴 답변은 자르기)
        answer = metadata.get('answer', 'N/A')
        if len(answer) > ANSWER_PREVIEW_LENGTH:
            answer_preview = answer[:ANSWER_PREVIEW_LENGTH] + "..."
        else:
            answer_preview = answer
        
        print(f"   💬 답변: {answer_preview}")
        print("   " + "-" * 50)

def show_welcome_message() -> None:
    """환영 메시지와 시스템 정보를 표시합니다."""
    print("=" * 60)
    print("🔍 Bible AI 검색 시스템 테스트 (OpenAI 버전)")
    print("=" * 60)
    print(f"🤖 모델: OpenAI {MODEL_NAME}")
    print(f"📏 차원: {EMBEDDING_DIMENSION}차원")
    print("💰 OpenAI API 사용 - 고품질 검색 제공")
    print("📚 데이터: 100개 샘플 FAQ")
    print("=" * 60)
    print("\n💡 사용법:")
    print("- 성경, 앱 사용법, 기술적 문제 등에 대해 질문하세요")
    print("- 자연스러운 문장으로 질문해도 의미를 이해합니다")
    print("- 'quit', 'exit', '종료'를 입력하면 프로그램이 종료됩니다")
    print()

def get_search_examples() -> List[str]:
    """검색 예시 질문들을 반환합니다."""
    return [
        "성경 앱이 느려요",
        "로그인이 안돼요",
        "구독은 어떻게 하나요?",
        "음성 재생이 안됩니다",
        "통독 계획을 설정하고 싶어요"
    ]

def show_search_examples() -> None:
    """검색 예시를 표시합니다."""
    examples = get_search_examples()
    print("🔍 검색 예시:")
    for i, example in enumerate(examples, 1):
        print(f"   {i}. {example}")
    print()

def validate_user_input(query: str) -> bool:
    """
    사용자 입력을 검증합니다.
    
    Args:
        query (str): 사용자가 입력한 질문
        
    Returns:
        bool: 유효한 입력이면 True, 아니면 False
    """
    # 빈 입력 검증
    if not query:
        print("❌ 질문을 입력해주세요.")
        return False
    
    # 너무 짧은 입력 검증
    if len(query.strip()) < 2:
        print("❌ 너무 짧은 질문입니다. 더 구체적으로 질문해주세요.")
        return False
    
    # 너무 긴 입력 검증
    if len(query) > 500:
        print("❌ 질문이 너무 깁니다. 500자 이내로 질문해주세요.")
        return False
    
    return True

def main() -> None:
    """
    메인 실행 함수: 대화형 검색 인터페이스를 제공합니다.
    """
    try:
        # 1. 환영 메시지 표시
        show_welcome_message()
        
        # 2. 서비스 초기화
        index, openai_client = initialize_services()
        
        # 3. 검색 예시 표시
        show_search_examples()
        
        # 4. 대화형 검색 루프
        search_count = 0
        while True:
            try:
                # 사용자 입력 받기
                query = input("검색할 질문을 입력하세요 (종료: 'quit'): ").strip()
                
                # 종료 명령 확인
                if query.lower() in ['quit', 'exit', '종료', 'q']:
                    print(f"\n👋 테스트를 종료합니다. (총 {search_count}회 검색)")
                    break
                
                # 도움말 명령 확인
                if query.lower() in ['help', '도움말', '?']:
                    show_search_examples()
                    continue
                
                # 입력 검증
                if not validate_user_input(query):
                    continue
                
                # 검색 실행
                results = search_question(query, index, openai_client)
                search_count += 1
                
                # 검색 결과가 없을 때 예시 제안
                if not results:
                    print("\n💡 다음과 같은 질문을 시도해보세요:")
                    examples = get_search_examples()[:3]  # 처음 3개만
                    for example in examples:
                        print(f"   - {example}")
                
            except KeyboardInterrupt:
                print(f"\n👋 테스트를 종료합니다. (총 {search_count}회 검색)")
                break
            except Exception as e:
                print(f"\n❌ 검색 중 오류 발생: {e}")
                print("💡 다시 시도해주세요.")
                continue
                
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 프로그램 실행 중 오류 발생: {e}")
        print("💡 설정을 확인하고 다시 시도하세요.")

# 스크립트가 직접 실행될 때만 main 함수 호출
if __name__ == "__main__":
    main()