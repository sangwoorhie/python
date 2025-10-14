#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
API 엔드포인트 모듈
"""

import gc
import logging
import tracemalloc
import threading
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from src.utils.memory_manager import memory_cleanup
from src.utils.mssql_updater import MSSQLUpdater
import json
import os
import pytz

# API 엔드포인트 생성
def create_endpoints(app: Flask, generator, sync_manager, index):
    """Flask 앱에 API 엔드포인트를 등록"""
    
    # CORS 설정 - 웹 브라우저의 교차 출처 요청 허용
    CORS(app)

    # MSSQL 업데이터 초기화 (전역으로 한 번만 생성)
    mssql_updater = MSSQLUpdater()

    # 연결 테스트
    if not mssql_updater.test_connection():
        logging.warning("⚠️ MSSQL 연결 테스트 실패 - DB 업데이트 기능 비활성화")
    else:
        logging.info("✅ MSSQL 연결 테스트 성공 - DB 업데이트 기능 활성화")

    # ============================== 백그라운드 작업 함수 ==============================
    def _process_answer_in_background(seq: int, question: str, lang: str):
        """
        백그라운드에서 AI 답변 생성 및 DB 업데이트
        - 기존의 모든 로깅 및 메모리 관리 기능 포함
        
        Args:
            seq: 문의 시퀀스 번호
            question: 사용자 질문
            lang: 언어 코드
        """
        try:
            logging.info(f"🔄 백그라운드 처리 시작 - SEQ: {seq}")
            
            # 1단계: AI 답변 생성
            result = generator.process(seq, question, lang)
            
            if not result.get('success'):
                logging.error(f"❌ AI 답변 생성 실패 - SEQ: {seq}, 오류: {result.get('error')}")
                return
            
            ai_answer = result.get('answer', '')
            
            if not ai_answer:
                logging.error(f"❌ 생성된 답변이 비어있음 - SEQ: {seq}")
                return
            
            # 2단계: DB 업데이트 (answer_YN = 'N'으로 저장)
            update_success = mssql_updater.update_inquiry_answer(
                seq=seq,
                answer=ai_answer,
                answer_yn='N'  # AI 답변 (관리자 승인 전)
            )
            
            if update_success:
                logging.info(f"✅ 백그라운드 처리 완료 - SEQ: {seq}, 답변 길이: {len(ai_answer)}자")
            else:
                logging.error(f"❌ DB 업데이트 실패 - SEQ: {seq}")
            
        except Exception as e:
            logging.error(f"❌ 백그라운드 처리 중 오류 - SEQ: {seq}, 오류: {str(e)}")
    
    # ===== 1. AI 답변 생성 API 엔드포인트 =====
    # ☆ 1. 사용자 질문 입력 (/generate_answer 엔드포인트)
    # POST요청 수신, JSON 데이터에서 seq, question, lang 파싱
    @app.route('/generate_answer', methods=['POST'])
    def generate_answer():
        """
        AI 답변 생성 API 엔드포인트 - 비동기 처리 버전
        
        변경사항:
        1. 요청을 받으면 즉시 202 Accepted 응답 반환
        2. 백그라운드 스레드에서 AI 답변 생성 및 DB 업데이트 수행
        3. 프론트엔드는 응답을 기다리지 않고 즉시 다음 작업 진행 가능
        """
        try:
            # 메모리 자동 정리 컨텍스트 시작
            with memory_cleanup():
                # 1단계: 요청 데이터 파싱 및 검증
                data = request.get_json()
                seq = data.get('seq', 0)                    # 시퀀스 ID (기본값: 0)
                question = data.get('question', '')         # 사용자 질문
                lang = data.get('lang', 'auto')             # 언어 설정 (자동 감지) 프론트엔드에서 받을 수 있음, 기본값 'auto'
                
                # seq별 로그 핸들러 추가
                kst = pytz.timezone('Asia/Seoul')
                current_date = datetime.now(kst).strftime('%Y%m%d')
                log_dir = '/home/ec2-user/python/logs'
                seq_log_file = f'{log_dir}/log_{seq}_{current_date}.log'

                seq_handler = logging.FileHandler(seq_log_file, encoding='utf-8')
                seq_handler.setLevel(logging.INFO)
                seq_formatter = logging.Formatter(
                    '%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                seq_handler.setFormatter(seq_formatter)

                root_logger = logging.getLogger()
                root_logger.addHandler(seq_handler)
                
                # 🔍 추가 로그
                logging.info(f"================================= API 요청 수신 (POST /generate_answer) ====================================")
                logging.info(f"SEQ: {seq}")
                logging.info(f"질문: {question}")
                logging.info(f"언어: {lang}")
                logging.info(f"사용된 generator 타입: {type(generator).__name__}")

                # 2단계: 필수 데이터 검증
                if not seq or not question:
                    return jsonify({
                        "success": False, 
                        "error": "seq와 question이 필요합니다."
                    }), 400

                # 3단계: AI 답변 생성 처리 (핵심 로직)
                # - Pinecone에서 유사 구절 검색
                # - 백그라운드 스레드에서 실행
                # daemon=True로 설정하여 메인 프로그램 종료시 자동으로 종료되도록 함
                background_thread = threading.Thread(
                    target=_process_answer_in_background,
                    args=(seq, question, lang),
                    daemon=True,
                    name=f"AIAnswerThread-{seq}"  # 스레드 이름 설정 (디버깅용)
                )
                background_thread.start()

                # 4단계: 즉시 응답 반환
                logging.info(f"✅ 비동기 작업 시작 - SEQ: {seq}, 질문: '{question[:50]}...'")
                logging.info(f"백그라운드 스레드 시작: {background_thread.name}")
                logging.info(f"==================================== API 응답 반환 (202 Accepted) ====================================")
                
                response = jsonify({
                    "success": True,
                    "message": "답변 생성 작업이 시작되었습니다. 백그라운드에서 처리 중입니다.",
                    "seq": seq,
                    "status": "processing",
                    "thread_name": background_thread.name
                })
                response.headers['Content-Type'] = 'application/json; charset=utf-8'
                
                return response, 202  # HTTP 202 Accepted (비동기 처리 시작)
            
        except Exception as e:
            # 예외 발생시 로깅 및 에러 응답 반환
            logging.error(f"API 호출 오류: {str(e)}")
            return jsonify({"success": False, "error": str(e)}), 500

    # ===== 2. Pinecone 동기화 API 엔드포인트 =====
    @app.route('/sync_to_pinecone', methods=['POST'])
    def sync_to_pinecone():
        """MSSQL 데이터를 Pinecone에 동기화하는 API 엔드포인트"""
        try:
            # 1단계: 요청 데이터 파싱
            data = request.get_json()
            seq = data.get('seq')                           # 동기화할 데이터의 시퀀스 ID
            mode = data.get('mode', 'upsert')               # 동기화 모드 (기본값: upsert)

            logging.info(f"동기화 요청 수신: seq={seq}, mode={mode}")
            
            # 2단계: 필수 파라미터 검증
            if not seq:
                logging.warning("seq 누락")
                return jsonify({"success": False, "error": "seq가 필요합니다"}), 400
            
            # 3단계: 데이터 타입 변환 (문자열 -> 정수)
            if not isinstance(seq, int):
                seq = int(seq)
            
            # 4단계: Pinecone 동기화 실행
            # - MSSQL에서 해당 seq 데이터 조회
            # - 임베딩 생성 후 Pinecone에 업로드
            result = sync_manager.sync_to_pinecone(seq, mode)

            logging.info(f"동기화 결과: {result}")
            
            # 5단계: 결과에 따른 HTTP 상태 코드 설정
            status_code = 200 if result["success"] else 500
            return jsonify(result), status_code
            
        except ValueError as e:
            # 데이터 타입 변환 오류 처리
            logging.error(f"잘못된 seq 값: {str(e)}")
            return jsonify({"success": False, "error": f"잘못된 seq 값: {str(e)}"}), 400
        except Exception as e:
            # 기타 예외 처리
            logging.error(f"Pinecone 동기화 API 오류: {str(e)}")
            return jsonify({"success": False, "error": str(e)}), 500

    # ===== 3. 시스템 상태 확인 API 엔드포인트 =====
    @app.route('/health', methods=['GET'])
    def health_check():
        """시스템 상태 확인을 위한 헬스체크 API 엔드포인트 (최적화 정보 포함)"""
        try:
            # 1단계: Pinecone 인덱스 통계 조회
            stats = index.describe_index_stats()
            
            # 2단계: 최적화 정보 수집 (OptimizedAIAnswerGenerator인 경우)
            optimization_info = {}
            if hasattr(generator, 'get_optimization_summary'):
                try:
                    # 최적화 통계 수집
                    optimization_stats = generator.get_optimization_summary()
                    detailed_stats = generator.get_detailed_performance_stats()
                    
                    # 최적화 정보 구성
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
            
            # 3단계: 응답 데이터 구성
            response_data = {
                "status": "healthy",
                "pinecone_vectors": stats.get('total_vector_count', 0),  # Pinecone 벡터 수
                "timestamp": datetime.now().isoformat(),                # 현재 시간
                "services": {
                    "ai_answer": "active",                              # AI 답변 서비스 상태
                    "pinecone_sync": "active",                          # Pinecone 동기화 서비스 상태
                    "multilingual_support": "active",                   # 다국어 지원 상태
                    "optimization_system": "active" if optimization_info else "not_available"
                }
            }
            
            # 4단계: 최적화 정보가 있으면 응답에 추가
            response_data.update(optimization_info)
            
            return jsonify(response_data), 200
            
        except Exception as e:
            # 예외 발생시 비정상 상태 응답
            return jsonify({
                "status": "unhealthy",
                "error": str(e)
            }), 500

    # ===== 4. 최적화 통계 조회 API 엔드포인트 =====
    @app.route('/optimization/stats', methods=['GET'])
    def get_optimization_stats():
        """최적화 통계 조회 API - 상세한 성능 데이터 제공"""
        try:
            # 상세 성능 통계 조회
            stats = generator.get_detailed_performance_stats()
            
            return jsonify({
                "success": True,
                "stats": stats,                               # 캐시 히트율, API 호출 수 등
                "timestamp": datetime.now().isoformat()
            }), 200
            
        except Exception as e:
            logging.error(f"최적화 통계 조회 실패: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500

    # ===== 5. 캐시 지우기 API 엔드포인트 =====
    @app.route('/optimization/cache/clear', methods=['POST'])
    def clear_cache():
        """캐시 지우기 API - 성능 최적화를 위한 캐시 관리"""
        try:
            # 1단계: 요청 데이터 파싱
            data = request.get_json() or {}
            cache_type = data.get('type', 'all')  # 캐시 타입 지정
            
            # 2단계: 캐시 타입별 처리
            if cache_type == 'all':
                # 모든 캐시 지우기
                generator.cache_manager.clear_cache_by_prefix('embedding')    # 임베딩 캐시
                generator.cache_manager.clear_cache_by_prefix('intent')       # 의도 분석 캐시
                generator.cache_manager.clear_cache_by_prefix('translation')  # 번역 캐시
                generator.cache_manager.clear_cache_by_prefix('typo')         # 오타 수정 캐시
                generator.cache_manager.clear_cache_by_prefix('search')       # 검색 결과 캐시
                generator.search_service.clear_caches()                       # 검색 서비스 캐시
                generator.api_manager.clear_recent_requests()                 # API 요청 기록
                cleared_count = "all"
            else:
                # 특정 캐시만 지우기
                cleared_count = generator.cache_manager.clear_cache_by_prefix(cache_type)
            
            # 3단계: 성공 응답 반환
            return jsonify({
                "success": True,
                "message": f"캐시 지우기 완료: {cache_type}",
                "cleared_count": cleared_count,
                "timestamp": datetime.now().isoformat()
            }), 200
            
        except Exception as e:
            # 예외 처리
            logging.error(f"캐시 지우기 실패: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500

    # ===== 6. 최적화 설정 업데이트 API 엔드포인트 =====
    @app.route('/optimization/config', methods=['POST'])
    def update_optimization_config():
        """최적화 설정 업데이트 API - 실시간 성능 튜닝"""
        try:
            # 1단계: 요청 데이터 파싱
            data = request.get_json()
            
            # 2단계: API 관리자 설정 업데이트
            api_settings = data.get('api_manager', {})
            if api_settings:
                # API 호출 최적화 설정 (요청 간격, 재시도 등)
                generator.api_manager.optimize_settings(**api_settings)
            
            # 3단계: 검색 서비스 설정 업데이트  
            search_settings = data.get('search_service', {})
            if search_settings:
                # 검색 성능 설정 (유사도 임계값, 검색 범위 등)
                generator.search_service.update_search_config(**search_settings)
            
            # 4단계: 성공 응답 반환
            return jsonify({
                "success": True,
                "message": "최적화 설정 업데이트 완료",
                "updated_settings": {
                    "api_manager": api_settings,      # 업데이트된 API 설정
                    "search_service": search_settings # 업데이트된 검색 설정
                },
                "timestamp": datetime.now().isoformat()
            }), 200
            
        except Exception as e:
            # 예외 처리
            logging.error(f"최적화 설정 업데이트 실패: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    # ===== 7. 오안내 피드백 수집 API 엔드포인트 =====
    @app.route('/feedback/mismatch', methods=['POST'])
    def report_mismatch():
        """오안내 피드백 수집"""
        try:
            data = request.get_json()
            
            # 오안내 데이터 저장 (학습용)
            mismatch_data = {
                'question': data.get('question'),
                'given_answer': data.get('given_answer'),
                'expected_topic': data.get('expected_topic'),
                'timestamp': datetime.now().isoformat()
            }
            
            # 로그 디렉토리 확인
            log_dir = '/home/ec2-user/python/logs'
            os.makedirs(log_dir, exist_ok=True)
            
            # 패턴 분석 및 저장
            log_file = os.path.join(log_dir, 'mismatches.jsonl')
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(mismatch_data, ensure_ascii=False) + '\n')
            
            logging.info(f"오안내 피드백 저장: {mismatch_data['question'][:50]}...")
            
            return jsonify({"success": True, "message": "피드백이 저장되었습니다"}), 200
            
        except Exception as e:
            logging.error(f"피드백 저장 실패: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    # 디버그용 엔드포인트 추가
    @app.route('/debug/last_search', methods=['GET'])
    def get_last_search_debug():
        """마지막 검색의 디버그 정보"""
        try:
            # generator의 마지막 검색 정보 가져오기
            if hasattr(generator, '_last_search_info'):
                return jsonify(generator._last_search_info), 200
            else:
                return jsonify({"message": "디버그 정보가 없습니다"}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    # ===== 엔드포인트 등록 완료 =====
    return app
