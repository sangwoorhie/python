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

# API 엔드포인트 생성
def create_endpoints(app: Flask, generator, sync_manager, index):
    """Flask 앱에 API 엔드포인트를 등록"""
    
    # CORS 설정
    CORS(app)
    
    # 1. AI 답변 생성 API 엔드포인트
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

    # 2. Pinecone 동기화 API 엔드포인트
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

    # 3. 시스템 상태 확인 API 엔드포인트
    @app.route('/health', methods=['GET'])
    def health_check():
        """시스템 상태 확인을 위한 헬스체크 API 엔드포인트 (최적화 정보 포함)"""
        try:
            stats = index.describe_index_stats()
            
            # 최적화 정보 포함 (OptimizedAIAnswerGenerator인 경우)
            optimization_info = {}
            if hasattr(generator, 'get_optimization_summary'):
                try:
                    optimization_stats = generator.get_optimization_summary()
                    detailed_stats = generator.get_detailed_performance_stats()
                    optimization_info = {
                        "optimization": {
                            "cache_hit_rate": f"{optimization_stats['cache_hit_rate']:.1f}%",
                            "api_calls_saved": optimization_stats['api_calls_saved'],
                            "avg_processing_time": f"{optimization_stats['avg_processing_time']:.2f}s",
                            "batch_processed": optimization_stats['batch_processed']
                        },
                        "detailed_performance": detailed_stats
                    }
                except Exception as opt_error:
                    logging.warning(f"최적화 정보 조회 실패: {opt_error}")
            
            response_data = {
                "status": "healthy",
                "pinecone_vectors": stats.get('total_vector_count', 0),
                "timestamp": datetime.now().isoformat(),
                "services": {
                    "ai_answer": "active",
                    "pinecone_sync": "active",
                    "multilingual_support": "active",
                    "optimization_system": "active" if optimization_info else "not_available"
                }
            }
            
            # 최적화 정보가 있으면 추가
            response_data.update(optimization_info)
            
            return jsonify(response_data), 200
        except Exception as e:
            return jsonify({
                "status": "unhealthy",
                "error": str(e)
            }), 500

    # 4. 최적화 통계 조회 API 엔드포인트
    @app.route('/optimization/stats', methods=['GET'])
    def get_optimization_stats():
        """최적화 통계 조회 API"""
        try:
            stats = generator.get_detailed_performance_stats()
            return jsonify({
                "success": True,
                "stats": stats,
                "timestamp": datetime.now().isoformat()
            }), 200
        except Exception as e:
            logging.error(f"최적화 통계 조회 실패: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500

    # 5. 캐시 지우기 API 엔드포인트
    @app.route('/optimization/cache/clear', methods=['POST'])
    def clear_cache():
        """캐시 지우기 API"""
        try:
            data = request.get_json() or {}
            cache_type = data.get('type', 'all')  # 'all', 'embedding', 'intent', 'translation', etc.
            
            if cache_type == 'all':
                # 모든 캐시 지우기
                generator.cache_manager.clear_cache_by_prefix('embedding')
                generator.cache_manager.clear_cache_by_prefix('intent')
                generator.cache_manager.clear_cache_by_prefix('translation')
                generator.cache_manager.clear_cache_by_prefix('typo')
                generator.cache_manager.clear_cache_by_prefix('search')
                generator.search_service.clear_caches()
                generator.api_manager.clear_recent_requests()
                cleared_count = "all"
            else:
                # 특정 캐시만 지우기
                cleared_count = generator.cache_manager.clear_cache_by_prefix(cache_type)
            
            return jsonify({
                "success": True,
                "message": f"캐시 지우기 완료: {cache_type}",
                "cleared_count": cleared_count,
                "timestamp": datetime.now().isoformat()
            }), 200
            
        except Exception as e:
            logging.error(f"캐시 지우기 실패: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500

    # 6. 최적화 설정 업데이트 API 엔드포인트
    @app.route('/optimization/config', methods=['POST'])
    def update_optimization_config():
        """최적화 설정 업데이트 API"""
        try:
            data = request.get_json()
            
            # API 관리자 설정 업데이트
            api_settings = data.get('api_manager', {})
            if api_settings:
                generator.api_manager.optimize_settings(**api_settings)
            
            # 검색 서비스 설정 업데이트
            search_settings = data.get('search_service', {})
            if search_settings:
                generator.search_service.update_search_config(**search_settings)
            
            return jsonify({
                "success": True,
                "message": "최적화 설정 업데이트 완료",
                "updated_settings": {
                    "api_manager": api_settings,
                    "search_service": search_settings
                },
                "timestamp": datetime.now().isoformat()
            }), 200
            
        except Exception as e:
            logging.error(f"최적화 설정 업데이트 실패: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    return app
