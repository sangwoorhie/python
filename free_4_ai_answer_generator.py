#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI Answer Generator for ASP Classic Integration
파일명: free_4_ai_answer_generator.py
설명: ASP Classic에서 호출되어 AI 답변을 생성하는 스크립트
모델: google/flan-t5-base (무료 모델)
"""

import os
import sys
import argparse
import json
import re
import html
import logging
from datetime import datetime
from typing import Dict, List, Tuple

# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()

# 로깅 설정
logging.basicConfig(
    filename='D:/AI_Scripts/logs/ai_generator.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

try:
    # Pinecone 초기화 (768차원 인덱스)
    from pinecone import Pinecone
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index("bible-app-support-768-free")
    
    # sentence-transformers 모델 로드 (임베딩용)
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    
    # 무료 텍스트 생성 모델 로드 (T5-base)
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    text_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')
    text_tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
    
    # MSSQL 연결 (선택사항)
    import pyodbc
    DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
    
except Exception as e:
    logging.error(f"모듈 로드 실패: {str(e)}")
    print(json.dumps({"success": False, "error": f"모듈 로드 실패: {str(e)}"}, ensure_ascii=False))
    sys.exit(1)


class AIAnswerGenerator:
    def __init__(self):
        """초기화"""
        self.db_connection = None
        if DB_CONNECTION_STRING:
            try:
                self.db_connection = pyodbc.connect(DB_CONNECTION_STRING)
                logging.info("DB 연결 성공")
            except Exception as e:
                logging.error(f"DB 연결 실패: {str(e)}")
    
    def preprocess_text(self, text: str) -> str:
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
    
    def create_embedding(self, text: str) -> List[float]:
        """텍스트를 벡터로 변환 (sentence-transformers 사용)"""
        try:
            embedding = embedding_model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logging.error(f"임베딩 생성 실패: {e}")
            return None
    
    def search_similar_answers(self, query: str, top_k: int = 10, similarity_threshold: float = 0.6) -> List[Dict]:
        """유사 답변 검색 (무료 모델 사용)"""
        try:
            query_vector = self.create_embedding(query)
            
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
                    filtered_results.append({
                        'score': match['score'],
                        'question': match['metadata'].get('question', ''),
                        'answer': match['metadata'].get('answer', ''),
                        'category': match['metadata'].get('category', '일반')
                    })
            
            logging.info(f"유사 답변 {len(filtered_results)}개 검색 완료")
            return filtered_results
            
        except Exception as e:
            logging.error(f"Pinecone 검색 실패: {str(e)}")
            return []
    
    def generate_ai_answer(self, query: str, similar_answers: List[Dict], lang: str) -> str:
        """무료 T5 모델을 사용하여 AI 답변 생성"""
        
        if not similar_answers:
            return "문의해주신 내용에 대해 정확한 답변을 드리기 드리기 위해 더 자세한 정보가 필요합니다. 고객센터로 문의해주시면 신속하게 도움을 드리겠습니다."
        
        try:
            # 가장 유사한 답변들을 컨텍스트로 사용
            context = ""
            for i, ans in enumerate(similar_answers[:3], 1):  # 상위 3개 사용
                context += f"참고 {i}: {ans['answer'][:200]} "
            
            # T5 모델을 위한 프롬프트 구성 (lang 반영)
            prompt = f"Question in {lang}: {query}\nContext: {context}\nProvide a helpful answer in {lang}:"
            
            # T5 모델로 답변 생성
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
            
            # 답변 포맷팅
            if generated_text.strip():
                # 생성된 텍스트가 너무 짧으면 가장 유사한 답변 사용
                if len(generated_text) < 20:
                    formatted_answer = f"안녕하세요.\n\n{similar_answers[0]['answer']}\n\n추가 문의사항이 있으시면 언제든 연락해주세요. 감사합니다."
                else:
                    formatted_answer = f"안녕하세요.\n\n{generated_text}\n\n추가 문의사항이 있으시면 언제든 연락해주세요. 감사합니다."
            else:
                # T5 모델이 답변을 생성하지 못한 경우 기존 답변 사용
                formatted_answer = f"안녕하세요.\n\n{similar_answers[0]['answer']}\n\n감사합니다."
            
            logging.info("AI 답변 생성 완료")
            return formatted_answer
            
        except Exception as e:
            logging.error(f"T5 모델 답변 생성 실패: {e}")
            # 폴백: 가장 유사한 답변 사용
            if similar_answers:
                return f"안녕하세요.\n\n{similar_answers[0]['answer']}\n\n감사합니다."
            return "죄송합니다. 현재 답변을 생성할 수 없습니다. 고객센터로 문의해주세요."
    
    def save_to_database(self, seq: int, ai_answer: str, similar_count: int) -> bool:
        """생성된 답변을 데이터베이스에 저장 (선택사항)"""
        if not self.db_connection:
            return True  # DB 연결이 없으면 성공으로 처리
        
        try:
            cursor = self.db_connection.cursor()
            
            # 답변 저장
            update_sql = """
                UPDATE mobile.dbo.bible_inquiry 
                SET reply_contents = ?, 
                    answer_YN = 'N',
                    ai_generated = 'Y',
                    ai_generated_date = GETDATE(),
                    model = 'google/flan-t5-base',
                    ai_similarity_score = ?
                WHERE seq = ?
            """
            
            cursor.execute(update_sql, (ai_answer, similar_count, seq))
            self.db_connection.commit()
            
            logging.info(f"DB 저장 성공 - SEQ: {seq}")
            return True
            
        except Exception as e:
            logging.error(f"DB 저장 실패: {str(e)}")
            if self.db_connection:
                self.db_connection.rollback()
            return False
    
    def process(self, seq: int, question: str, lang: str, save_to_db: bool = False) -> Dict:
        """메인 처리 함수"""
        try:
            # 1. 질문 전처리
            processed_question = self.preprocess_text(question)
            
            if not processed_question:
                return {
                    "success": False,
                    "error": "질문이 비어있습니다."
                }
            
            logging.info(f"처리 시작 - SEQ: {seq}, 질문: {processed_question[:50]}...")
            
            # 2. 유사 질문 검색
            similar_answers = self.search_similar_answers(processed_question)
            
            # 3. AI 답변 생성
            ai_answer = self.generate_ai_answer(processed_question, similar_answers, lang)
            
            # 4. 데이터베이스에 저장 (옵션)
            if save_to_db and seq > 0:
                self.save_to_database(seq, ai_answer, len(similar_answers))
            
            result = {
                "success": True,
                "question": processed_question,
                "answer": ai_answer,
                "similar_count": len(similar_answers),
                "embedding_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                "generation_model": "google/flan-t5-base",
                "similar_answers": [
                    {
                        "score": ans['score'],
                        "question": ans['question'][:100],
                        "answer": ans['answer'][:100],
                        "category": ans['category']
                    }
                    for ans in similar_answers[:3]  # 상위 3개만
                ]
            }
            
            logging.info(f"처리 완료 - SEQ: {seq}")
            return result
            
        except Exception as e:
            logging.error(f"처리 중 오류 - SEQ: {seq}, 오류: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            if self.db_connection:
                self.db_connection.close()


def main():
    """메인 함수 - ASP Classic에서 호출"""
    parser = argparse.ArgumentParser(description='AI Answer Generator (T5 Model)')
    parser.add_argument('--seq', type=int, default=0, help='Inquiry sequence number')
    parser.add_argument('--question', type=str, required=True, help='Customer question')
    parser.add_argument('--input_file', type=str, help='Input file path')
    parser.add_argument('--output', default='json', choices=['console', 'json'], help='Output format')
    parser.add_argument('--save_db', action='store_true', help='Save to database')
    parser.add_argument('--lang', type=str, default='kr', help='Language code')
    
    args = parser.parse_args()
    
    # 질문 읽기
    if args.input_file and os.path.exists(args.input_file):
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                question = f.read()
        except:
            # UTF-8 실패 시 cp949로 시도
            with open(args.input_file, 'r', encoding='cp949') as f:
                question = f.read()
    else:
        question = args.question
    
    # AI 답변 생성
    generator = AIAnswerGenerator()
    result = generator.process(args.seq, question, args.lang, args.save_db)
    
    # 결과 출력
    if args.output == 'json':
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        if result["success"]:
            print(f"질문: {result['question']}")
            print(f"답변: {result['answer']}")
            print(f"참고된 유사 답변 수: {result['similar_count']}")
            print(f"모델: {result['generation_model']}")
        else:
            print(f"오류: {result['error']}")
    
    sys.exit(0 if result['success'] else 1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"실행 오류: {str(e)}")
        print(json.dumps({"success": False, "error": str(e)}, ensure_ascii=False))
        sys.exit(1)