#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
API 엔드포인트 모듈
"""

import gc
import logging
import tracemalloc
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS


def create_endpoints(app: Flask, generator, sync_manager, index):
    """Flask 앱에 API 엔드포인트를 등록"""
    
    # CORS 설정
    CORS(app)
    
    @app.route('/generate_answer', methods=['POST'])
    def generate_answer():
        """AI 답변 생성 API 엔드포인트"""
        try:
            from src.utils.memory_manager import memory_cleanup
            
            with memory_cleanup():
                data = request.get_json()
                seq = data.get('seq', 0)
                question = data.get('question', '')
                lang = data.get('lang', 'auto')  # 기본값을 'auto'로 변경 (자동 감지)
                
                if not question:
                    return jsonify({"success": False, "error": "질문이 필요합니다."}), 400
                
                result = generator.process(seq, question, lang)
                
                response = jsonify(result)
                response.headers['Content-Type'] = 'application/json; charset=utf-8'

                # 메모리 사용량 모니터링
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')
                memory_usage = sum(stat.size for stat in top_stats) / 1024 / 1024  # MB로 반환
                logging.info(f"현재 메모리 사용량: {memory_usage:.2f}MB")
                
                if memory_usage > 500: # 500MB 초과시 경고 및 가비지 컬렉션 강제 실행
                    logging.warning(f"높은 메모리 사용량 감지: {memory_usage:.2f}MB")
                    gc.collect()

                return response
            
        except Exception as e:
            logging.error(f"API 호출 오류: {str(e)}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/sync_to_pinecone', methods=['POST'])
    def sync_to_pinecone():
        """MSSQL 데이터를 Pinecone에 동기화하는 API 엔드포인트"""
        try:
            data = request.get_json()
            seq = data.get('seq')
            mode = data.get('mode', 'upsert')

            logging.info(f"동기화 요청 수신: seq={seq}, mode={mode}")
            
            if not seq:
                logging.warning("seq 누락")
                return jsonify({"success": False, "error": "seq가 필요합니다"}), 400
            
            if not isinstance(seq, int):
                seq = int(seq)
            
            result = sync_manager.sync_to_pinecone(seq, mode)

            logging.info(f"동기화 결과: {result}")
            
            status_code = 200 if result["success"] else 500
            return jsonify(result), status_code
            
        except ValueError as e:
            logging.error(f"잘못된 seq 값: {str(e)}")
            return jsonify({"success": False, "error": f"잘못된 seq 값: {str(e)}"}), 400
        except Exception as e:
            logging.error(f"Pinecone 동기화 API 오류: {str(e)}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/health', methods=['GET'])
    def health_check():
        """시스템 상태 확인을 위한 헬스체크 API 엔드포인트"""
        try:
            stats = index.describe_index_stats()
            
            return jsonify({
                "status": "healthy",
                "pinecone_vectors": stats.get('total_vector_count', 0),
                "timestamp": datetime.now().isoformat(),
                "services": {
                    "ai_answer": "active",
                    "pinecone_sync": "active",
                    "multilingual_support": "active"
                }
            }), 200
        except Exception as e:
            return jsonify({
                "status": "unhealthy",
                "error": str(e)
            }), 500
    
    return app
