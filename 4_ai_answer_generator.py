#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI Answer Generator Flask API for ASP Classic Integration (OpenAI 버전)
파일명: 4_ai_answer_generator.py
설명: Flask API로 ASP Classic에서 호출
모델: OpenAI text-embedding-3-small + GPT-4o-mini
"""

import os
import json
import re
import html
import logging
from flask import Flask, request, jsonify
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv
from flask_cors import CORS 

# Flask 앱 초기화
app = Flask(__name__)
CORS(app) # CORS 설정

# 로깅 설정
logging.basicConfig(
    filename='/home/ec2-user/python/logs/ai_generator_openai.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# 환경 변수 로드
load_dotenv()

# Pinecone 및 OpenAI 초기화
try:
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index("bible-app-support-3072")
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
except Exception as e:
    logging.error(f"모듈 로드 실패: {str(e)}")
    app.logger.error(f"모듈 로드 실패: {str(e)}")
    raise

class AIAnswerGenerator:
    def preprocess_text(self, text: str) -> str:
        if not text:
            return ""
        text = str(text)
        text = html.unescape(text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def create_embedding(self, text: str) -> list:
        try:
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"임베딩 생성 실패: {e}")
            return None

    def search_similar_answers(self, query: str, top_k: int = 10, similarity_threshold: float = 0.6) -> list:
        try:
            query_vector = self.create_embedding(query)
            if query_vector is None:
                return []
            results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
            filtered_results = [
                {
                    'score': match['score'],
                    'question': match['metadata'].get('question', ''),
                    'answer': match['metadata'].get('answer', ''),
                    'category': match['metadata'].get('category', '일반')
                }
                for match in results['matches'] if match['score'] >= similarity_threshold
            ]
            logging.info(f"유사 답변 {len(filtered_results)}개 검색 완료")
            return filtered_results
        except Exception as e:
            logging.error(f"Pinecone 검색 실패: {str(e)}")
            return []

    def generate_ai_answer(self, query: str, similar_answers: list, lang: str) -> str:
        if not similar_answers:
            return "문의해주신 내용에 대해 정확한 답변을 드리기 위해 더 자세한 정보가 필요합니다. 고객센터(dev@goodtv.co.kr)로 문의해주시면 신속하게 도움을 드리겠습니다."
        
        try:
            # 참고 답변들을 컨텍스트로 구성
            context = ""
            for i, ans in enumerate(similar_answers[:5], 1):  # 상위 5개만 사용
                context += f"\n참고답변 {i} (유사도: {ans['score']:.3f}):\n"
                context += f"질문: {ans['question']}\n"
                context += f"답변: {ans['answer']}\n"
                context += f"카테고리: {ans.get('category', '일반')}\n"
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
            logging.error(f"GPT-4o-mini 답변 생성 실패: {e}")
            # 폴백: 가장 유사한 답변 사용
            if similar_answers:
                best_match = similar_answers[0]
                return f"안녕하세요, 바이블 애플입니다.\n\n{best_match['answer']}\n\n감사합니다."
            return "죄송합니다. 현재 답변을 생성할 수 없습니다. 고객센터(dev@goodtv.co.kr)로 문의해주세요."

    def process(self, seq: int, question: str, lang: str) -> dict:
        try:
            processed_question = self.preprocess_text(question)
            if not processed_question:
                return {"success": False, "error": "질문이 비어있습니다."}
            logging.info(f"처리 시작 - SEQ: {seq}, 질문: {processed_question[:50]}...")
            similar_answers = self.search_similar_answers(processed_question)
            ai_answer = self.generate_ai_answer(processed_question, similar_answers, lang)
            result = {
                "success": True,
                "answer": ai_answer,
                "similar_count": len(similar_answers),
                "embedding_model": "text-embedding-3-small",
                "generation_model": "gpt-4o-mini",
                "similar_answers": [
                    {
                        "score": ans['score'],
                        "question": ans['question'][:100],
                        "answer": ans['answer'][:100],
                        "category": ans.get('category', '일반')
                    }
                    for ans in similar_answers[:3]  # 상위 3개만
                ]
            }
            logging.info(f"처리 완료 - SEQ: {seq}")
            return result
        except Exception as e:
            logging.error(f"처리 중 오류 - SEQ: {seq}, 오류: {str(e)}")
            return {"success": False, "error": str(e)}

generator = AIAnswerGenerator()

@app.route('/generate_answer', methods=['POST'])
def generate_answer():
    try:
        data = request.get_json()
        seq = data.get('seq', 0)
        question = data.get('question', '')
        lang = data.get('lang', 'kr')
        if not question:
            return jsonify({"success": False, "error": "질문이 필요합니다."}), 400
        result = generator.process(seq, question, lang)
        return jsonify(result)
    except Exception as e:
        logging.error(f"API 호출 오류: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv('FLASK_PORT', 8000))

    print(f"Flask API starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)