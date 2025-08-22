#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI Answer Generator Flask API for ASP Classic Integration
파일명: free_4_ai_answer_generator.py
설명: Flask API로 ASP Classic에서 호출
모델: google/flan-t5-base
"""

import os
import json
import re
import html
import logging
from flask import Flask, request, jsonify
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from dotenv import load_dotenv
from flask_cors import CORS 

# Flask 앱 초기화
app = Flask(__name__)
CORS(app) # CORS 설정

# 로깅 설정
logging.basicConfig(
    filename='/home/ec2-user/python/logs/ai_generator.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# 환경 변수 로드
load_dotenv()

# Pinecone 및 모델 초기화
try:
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index("bible-app-support-768-free")
    embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    text_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')
    text_tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
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
            embedding = embedding_model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
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
            return "문의해주신 내용에 대해 정확한 답변을 드리기 위해 더 자세한 정보가 필요합니다. 고객센터로 문의해주시면 신속하게 도움을 드리겠습니다."
        try:
            context = ""
            for i, ans in enumerate(similar_answers[:3], 1):
                context += f"참고 {i}: {ans['answer'][:200]} "
            prompt = f"Question in {lang}: {query}\nContext: {context}\nProvide a helpful answer in {lang}:"
            inputs = text_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            outputs = text_model.generate(
                inputs.input_ids,
                max_length=200,
                num_beams=4,
                temperature=0.7,
                do_sample=True,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.0
            )
            generated_text = text_tokenizer.decode(outputs[0], skip_special_tokens=True)
            if generated_text.strip():
                if len(generated_text) < 20:
                    return f"안녕하세요.\n\n{similar_answers[0]['answer']}\n\n추가 문의사항이 있으시면 언제든 연락해주세요. 감사합니다."
                return f"안녕하세요.\n\n{generated_text}\n\n추가 문의사항이 있으시면 언제든 연락해주세요. 감사합니다."
            return f"안녕하세요.\n\n{similar_answers[0]['answer']}\n\n감사합니다."
        except Exception as e:
            logging.error(f"T5 모델 답변 생성 실패: {e}")
            if similar_answers:
                return f"안녕하세요.\n\n{similar_answers[0]['answer']}\n\n감사합니다."
            return "죄송합니다. 현재 답변을 생성할 수 없습니다. 고객센터로 문의해주세요."

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
                "embedding_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                "generation_model": "google/flan-t5-base"
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