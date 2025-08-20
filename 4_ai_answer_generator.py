import os
import sys
import argparse
import json
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
import re
import html

# 환경 변수 로드
load_dotenv()

# OpenAI 클라이언트 초기화
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Pinecone 초기화 (3072차원 인덱스)
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index("bible-app-support-3072")

def preprocess_text(text):
    """텍스트 전처리"""
    if not text:
        return ""
    
    # 문자열로 변환
    text = str(text)
    
    # HTML 태그 제거
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    
    # 연속된 공백을 하나로
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def create_embedding(text):
    """텍스트를 벡터로 변환"""
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

def search_similar_answers(query, top_k=10, similarity_threshold=0.6):
    """유사 답변 검색"""
    query_vector = create_embedding(query)
    
    if query_vector is None:
        return []
    
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    
    filtered_results = []
    for match in results['matches']:
        if match['score'] >= similarity_threshold:
            filtered_results.append(match)
    
    return filtered_results

def generate_ai_answer(query, similar_answers):
    """GPT-4o-mini를 사용하여 AI 답변 생성"""
    
    if not similar_answers:
        return "문의해주신 내용에 대해 정확한 답변을 드리기 위해 더 자세한 정보가 필요합니다. 고객센터(dev@goodtv.co.kr)로 문의해주시면 신속하게 도움을 드리겠습니다."
    
    # 참고 답변들을 컨텍스트로 구성
    context = ""
    for i, match in enumerate(similar_answers[:5], 1):  # 상위 5개만 사용
        metadata = match['metadata']
        context += f"\n참고답변 {i} (유사도: {match['score']:.3f}):\n"
        context += f"질문: {metadata['question']}\n"
        context += f"답변: {metadata['answer']}\n"
        context += f"카테고리: {metadata.get('category', '일반')}\n"
        context += "-" * 40 + "\n"
    
    # 시스템 프롬프트 (프롬프트 엔지니어링)
    system_prompt = """당신은 바이블 애플의 전문 고객 서비스 AI 상담원입니다.

역할과 가이드라인:
1. 친절하고 전문적인 톤으로 답변하세요
2. 제공된 참고답변들을 기반으로 정확한 정보를 전달하세요
3. 답변은 명확하고 단계별로 설명하세요
4. 고객이 쉽게 따라할 수 있도록 구체적으로 작성하세요
5. 바이블 애플 앱의 기능과 특징을 잘 이해하고 있다고 가정하세요
6. 답변 길이는 200-400자 내외로 작성하세요

답변 형식:
- "안녕하세요, 바이블 애플입니다." 인사말로 시작
- 문제 해결 방법을 단계별로 제시
- 추가 도움이 필요한 경우 안내
- "감사합니다." 마무리

참고답변들을 종합하여 현재 고객의 문의에 가장 적절한 답변을 작성해주세요.
참고답변과 완전히 다른 내용은 답변하지 마세요."""

    user_prompt = f"""고객 문의: {query}

참고할 수 있는 기존 답변들:
{context}

위 참고답변들을 종합하여 현재 고객 문의에 대한 최적의 답변을 작성해주세요."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=800,
            temperature=0.7
        )
        
        generated_answer = response.choices[0].message.content
        return generated_answer
        
    except Exception as e:
        print(f"AI 답변 생성 실패: {e}")
        # 폴백: 가장 유사한 답변 사용
        if similar_answers:
            best_match = similar_answers[0]
            return f"안녕하세요, 바이블 애플입니다.\n\n{best_match['metadata']['answer']}\n\n감사합니다."
        return "죄송합니다. 현재 답변을 생성할 수 없습니다. 고객센터로 문의해주세요."

def main():
    """메인 함수 - ASP에서 호출"""
    parser = argparse.ArgumentParser(description='바이블 애플 AI 답변 생성기')
    parser.add_argument('--question', required=True, help='고객 질문')
    parser.add_argument('--output', default='console', choices=['console', 'json'], help='출력 형식')
    
    args = parser.parse_args()
    
    # 질문 전처리
    processed_question = preprocess_text(args.question)
    
    if not processed_question:
        result = {
            "success": False,
            "error": "질문이 비어있습니다."
        }
    else:
        try:
            # 유사 답변 검색
            similar_answers = search_similar_answers(processed_question)
            
            # AI 답변 생성
            ai_answer = generate_ai_answer(processed_question, similar_answers)
            
            result = {
                "success": True,
                "question": processed_question,
                "answer": ai_answer,
                "similar_count": len(similar_answers),
                "similar_answers": [
                    {
                        "score": match['score'],
                        "question": match['metadata']['question'][:100],
                        "answer": match['metadata']['answer'][:100],
                        "category": match['metadata'].get('category', '일반')
                    }
                    for match in similar_answers[:3]  # 상위 3개만
                ]
            }
        except Exception as e:
            result = {
                "success": False,
                "error": str(e)
            }
    
    # 결과 출력
    if args.output == 'json':
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        if result["success"]:
            print(f"질문: {result['question']}")
            print(f"답변: {result['answer']}")
            print(f"참고된 유사 답변 수: {result['similar_count']}")
        else:
            print(f"오류: {result['error']}")

if __name__ == "__main__":
    main()
