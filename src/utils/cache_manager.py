#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Redis 기반 지능형 캐싱 시스템
- AI API 호출 최적화 및 성능 향상을 위한 캐싱 매니저
- 임베딩, 의도분석, 오타수정, 번역, 검색결과 캐싱 지원
- Redis 기반 분산 캐싱 및 메모리 폴백 시스템
- SHA256 해시 기반 캐시 키 생성 및 데이터 직렬화
"""

import json
import hashlib
import logging
import redis
import pickle
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

# ===== Redis 기반 지능형 캐싱 시스템 =====
class CacheManager:
    
    # CacheManager 초기화 - Redis 연결 및 폴백 시스템 설정
    # Args:
    #     redis_host: Redis 서버 호스트 (기본값: localhost)
    #     redis_port: Redis 서버 포트 (기본값: 6379)
    #     redis_db: Redis 데이터베이스 번호 (기본값: 0)
    #     redis_password: Redis 인증 패스워드 (선택적)
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0, redis_password=None):
        try:
            # ===== 1단계: Redis 클라이언트 초기화 =====
            self.redis_client = redis.Redis(
                host=redis_host,                     # Redis 서버 주소
                port=redis_port,                     # Redis 서버 포트
                db=redis_db,                         # 사용할 데이터베이스 번호
                password=redis_password,             # 인증 패스워드
                decode_responses=False,              # 바이너리 데이터 지원 (pickle 사용)
                socket_timeout=5,                    # 소켓 타임아웃 (5초)
                socket_connect_timeout=5,            # 연결 타임아웃 (5초)
                retry_on_timeout=True                # 타임아웃시 재시도
            )
            
            # ===== 2단계: Redis 연결 테스트 =====
            try:
                pong = self.redis_client.ping()
                logging.info(f"Redis 연결 성공: PONG={pong}, 비밀번호 사용됨={bool(redis_password)}")
            except redis.exceptions.AuthenticationError as auth_err:
                logging.error(f"Redis 인증 실패: {auth_err} - 비밀번호 확인 필요")
                raise
            except redis.exceptions.ConnectionError as conn_err:
                logging.error(f"Redis 연결 실패: {conn_err} - 호스트/포트 확인 필요")
                raise
            
            # ===== 3단계: 초기화 완료 로깅 =====
            logging.info("Redis 캐시 연결 성공")
            
        except Exception as e:
            # ===== 4단계: Redis 연결 실패시 메모리 캐시 폴백 =====
            logging.warning(f"Redis 연결 실패, 메모리 캐시로 폴백: {e}")
            self.redis_client = None                 # Redis 클라이언트 비활성화
            self._memory_cache = {}                  # 인메모리 캐시 딕셔너리 초기화
    
    # 캐시 키 생성 메서드 (SHA256 해시 기반)
    # Args:
    #     prefix: 캐시 키 접두사 (작업 유형 구분용)
    #     data: 해시화할 원본 데이터 (텍스트)
    # Returns:
    #     str: 생성된 캐시 키 (prefix:해시값)
    def _generate_cache_key(self, prefix: str, data: str) -> str:
        # ===== 1단계: UTF-8 인코딩 및 SHA256 해시 생성 =====
        # 텍스트를 바이트로 변환 후 SHA256 해시 객체 생성
        hash_obj = hashlib.sha256(data.encode('utf-8'))
        
        # ===== 2단계: 캐시 키 조합 =====
        # 접두사와 해시값 앞 16자리로 캐시 키 생성 (충돌 방지 + 키 길이 최적화)
        return f"{prefix}:{hash_obj.hexdigest()[:16]}"
    
    # 데이터 직렬화 메서드 (Python 객체 → 바이너리)
    # Args:
    #     data: 직렬화할 Python 객체 (리스트, 딕셔너리 등)
    # Returns:
    #     bytes: pickle로 직렬화된 바이너리 데이터
    def _serialize_data(self, data: Any) -> bytes:
        # pickle을 사용하여 Python 객체를 바이너리로 변환
        return pickle.dumps(data)
    
    # 데이터 역직렬화 메서드 (바이너리 → Python 객체)
    # Args:
    #     data: 역직렬화할 바이너리 데이터
    # Returns:
    #     Any: 복원된 Python 객체
    def _deserialize_data(self, data: bytes) -> Any:
        # pickle을 사용하여 바이너리를 Python 객체로 복원
        return pickle.loads(data)

    # =================================
    # 임베딩 캐싱 시스템
    # =================================
    
    # 텍스트 임베딩 캐시 조회 메서드
    # Args:
    #     text: 임베딩을 조회할 텍스트
    # Returns:
    #     Optional[List[float]]: 캐시된 임베딩 벡터 (없으면 None)
    def get_embedding_cache(self, text: str) -> Optional[List[float]]:
        try:
            # ===== 1단계: 캐시 키 생성 =====
            cache_key = self._generate_cache_key("embedding", text)
            logging.info(f"임베딩 캐시 조회 시작: 키={cache_key}, 텍스트={text[:50]}...")  # 조회 시작 로그 추가
            
            # ===== 2단계: Redis 캐시 조회 =====
            if self.redis_client:
                try:
                    cached_data = self.redis_client.get(cache_key)
                    if cached_data:
                        # 2-1: 바이너리 데이터를 임베딩 벡터로 역직렬화
                        embedding = self._deserialize_data(cached_data)
                        logging.info(f"임베딩 캐시 히트 (Redis): 키={cache_key}, 길이={len(embedding)}")
                        return embedding
                    else:
                        logging.info(f"임베딩 캐시 미스 (Redis): 키={cache_key}")  # 미스 로그 추가
                except Exception as redis_err:
                    logging.error(f"Redis 조회 실패: {redis_err} - 메모리 폴백으로 전환")
            else:
                logging.info("Redis 클라이언트 없음 - 메모리 캐시 폴백 사용")  # 폴백 이유 로그 추가
            
            # ===== 3단계: 메모리 캐시 폴백 조회 =====
            if cache_key in self._memory_cache:
                embedding = self._memory_cache[cache_key]
                logging.info(f"임베딩 캐시 히트 (Memory): 키={cache_key}, 길이={len(embedding)}")
                return embedding
            else:
                logging.info(f"임베딩 캐시 미스 (Memory): 키={cache_key}")  # 미스 로그 추가
            
            # ===== 4단계: 캐시 미스 =====
            logging.info("임베딩 캐시 전체 미스 - 새로 생성 필요")
            return None
            
        except Exception as e:
            # ===== 예외 처리: 캐시 조회 실패 =====
            logging.error(f"임베딩 캐시 조회 실패: {e}")
            return None
    
    # 텍스트 임베딩 캐시 저장 메서드
    # Args:
    #     text: 원본 텍스트
    #     embedding: 저장할 임베딩 벡터 (1536차원 리스트)
    #     expire_hours: 만료 시간 (시간 단위, 기본 7일)
    # Returns:
    #     bool: 저장 성공 여부
    def set_embedding_cache(self, text: str, embedding: List[float], expire_hours: int = 168) -> bool:
        try:
            # ===== 1단계: 캐시 키 생성 및 데이터 직렬화 =====
            cache_key = self._generate_cache_key("embedding", text)
            serialized_data = self._serialize_data(embedding)
            
            # ===== 2단계: Redis 캐시 저장 =====
            if self.redis_client:
                # 2-1: 만료 시간 계산 (시간 → 초)
                expire_seconds = expire_hours * 3600
                # 2-2: 만료시간과 함께 데이터 저장
                result = self.redis_client.setex(cache_key, expire_seconds, serialized_data)
                logging.info(f"임베딩 캐시 저장: {text[:50]}... ({len(embedding)}차원)")
                return result
            else:
                # ===== 3단계: 메모리 캐시 폴백 저장 =====
                # 메모리 캐시는 만료시간 없음 (프로세스 생존 기간동안 유지)
                self._memory_cache[cache_key] = embedding
                return True
                
        except Exception as e:
            # ===== 예외 처리: 캐시 저장 실패 =====
            logging.error(f"임베딩 캐시 저장 실패: {e}")
            return False

    # =================================
    # 의도 분석 캐싱 시스템
    # =================================
    
    # 질문 의도 분석 캐시 조회 메서드
    # Args:
    #     query: 의도 분석을 조회할 사용자 질문
    # Returns:
    #     Optional[Dict]: 캐시된 의도 분석 결과 (없으면 None)
    def get_intent_analysis_cache(self, query: str) -> Optional[Dict]:
        try:
            # ===== 1단계: 캐시 키 생성 =====
            cache_key = self._generate_cache_key("intent", query)
            logging.info(f"🔍 의도 분석 캐시 조회 시작: 질문='{query[:50]}...', 키='{cache_key}'")
            
            # ===== 2단계: Redis 캐시 조회 =====
            if self.redis_client:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    # 2-1: 바이너리 데이터를 의도 분석 딕셔너리로 역직렬화
                    intent_data = self._deserialize_data(cached_data)
                    logging.info(f"✅ 의도 분석 캐시 히트 (Redis): 질문='{query[:50]}...', 의도='{intent_data.get('core_intent', 'N/A')}', 액션='{intent_data.get('primary_action', 'N/A')}'")
                    return intent_data
                else:
                    logging.info(f"❌ 의도 분석 캐시 미스 (Redis): 질문='{query[:50]}...'")
            else:
                # ===== 3단계: 메모리 캐시 폴백 조회 =====
                if cache_key in self._memory_cache:
                    intent_data = self._memory_cache[cache_key]
                    logging.info(f"✅ 의도 분석 메모리 캐시 히트: 질문='{query[:50]}...', 의도='{intent_data.get('core_intent', 'N/A')}', 액션='{intent_data.get('primary_action', 'N/A')}'")
                    return intent_data
                else:
                    logging.info(f"❌ 의도 분석 메모리 캐시 미스: 질문='{query[:50]}...'")
            
            # ===== 4단계: 캐시 미스 =====
            return None
            
        except Exception as e:
            # ===== 예외 처리: 캐시 조회 실패 =====
            logging.error(f"의도 분석 캐시 조회 실패: {e}")
            return None
    
    # 질문 의도 분석 캐시 저장 메서드
    # Args:
    #     query: 원본 사용자 질문
    #     intent_data: 저장할 의도 분석 결과 딕셔너리 (core_intent, intent_category 등)
    #     expire_hours: 만료 시간 (시간 단위, 기본 3일)
    # Returns:
    #     bool: 저장 성공 여부
    def set_intent_analysis_cache(self, query: str, intent_data: Dict, expire_hours: int = 72) -> bool:
        try:
            # ===== 1단계: 캐시 키 생성 및 데이터 직렬화 =====
            cache_key = self._generate_cache_key("intent", query)
            serialized_data = self._serialize_data(intent_data)
            
            # ===== 2단계: Redis 캐시 저장 =====
            if self.redis_client:
                # 2-1: 만료 시간 계산 (시간 → 초)
                expire_seconds = expire_hours * 3600
                # 2-2: 만료시간과 함께 의도 분석 데이터 저장
                result = self.redis_client.setex(cache_key, expire_seconds, serialized_data)
                logging.info(f"의도 분석 캐시 저장: {query[:50]}...")
                return result
            else:
                # ===== 3단계: 메모리 캐시 폴백 저장 =====
                self._memory_cache[cache_key] = intent_data
                return True
                
        except Exception as e:
            # ===== 예외 처리: 캐시 저장 실패 =====
            logging.error(f"의도 분석 캐시 저장 실패: {e}")
            return False

    # =================================
    # 오타 수정 캐싱 시스템
    # =================================
    
    # 텍스트 오타 수정 캐시 조회 메서드
    # Args:
    #     text: 오타 수정 결과를 조회할 원본 텍스트
    # Returns:
    #     Optional[str]: 캐시된 수정된 텍스트 (없으면 None)
    def get_typo_correction_cache(self, text: str) -> Optional[str]:
        try:
            # ===== 1단계: 캐시 키 생성 =====
            cache_key = self._generate_cache_key("typo", text)
            
            # ===== 2단계: Redis 캐시 조회 =====
            if self.redis_client:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    # 2-1: 바이트 데이터를 UTF-8 문자열로 디코딩
                    corrected_text = cached_data.decode('utf-8')
                    logging.info(f"오타 수정 캐시 히트: {text[:50]}...")
                    return corrected_text
            else:
                # ===== 3단계: 메모리 캐시 폴백 조회 =====
                if cache_key in self._memory_cache:
                    logging.info(f"메모리 오타 수정 캐시 히트: {text[:50]}...")
                    return self._memory_cache[cache_key]
            
            # ===== 4단계: 캐시 미스 =====
            return None
            
        except Exception as e:
            # ===== 예외 처리: 캐시 조회 실패 =====
            logging.error(f"오타 수정 캐시 조회 실패: {e}")
            return None
    
    # 텍스트 오타 수정 캐시 저장 메서드
    # Args:
    #     original_text: 원본 텍스트 (오타 포함)
    #     corrected_text: 수정된 텍스트 (오타 수정 완료)
    #     expire_hours: 만료 시간 (시간 단위, 기본 7일)
    # Returns:
    #     bool: 저장 성공 여부
    def set_typo_correction_cache(self, original_text: str, corrected_text: str, expire_hours: int = 168) -> bool:
        try:
            # ===== 1단계: 캐시 키 생성 =====
            cache_key = self._generate_cache_key("typo", original_text)
            
            # ===== 2단계: Redis 캐시 저장 =====
            if self.redis_client:
                # 2-1: 만료 시간 계산 (시간 → 초)
                expire_seconds = expire_hours * 3600
                # 2-2: 수정된 텍스트를 UTF-8로 인코딩하여 저장
                result = self.redis_client.setex(cache_key, expire_seconds, corrected_text.encode('utf-8'))
                logging.info(f"오타 수정 캐시 저장: {original_text[:30]}... → {corrected_text[:30]}...")
                return result
            else:
                # ===== 3단계: 메모리 캐시 폴백 저장 =====
                self._memory_cache[cache_key] = corrected_text
                return True
                
        except Exception as e:
            # ===== 예외 처리: 캐시 저장 실패 =====
            logging.error(f"오타 수정 캐시 저장 실패: {e}")
            return False

    # =================================
    # 번역 캐싱 시스템
    # =================================
    
    # 다국어 번역 캐시 조회 메서드
    # Args:
    #     text: 번역할 원본 텍스트
    #     source_lang: 원본 언어 코드 (ko, en 등)
    #     target_lang: 목표 언어 코드 (ko, en 등)
    # Returns:
    #     Optional[str]: 캐시된 번역 결과 (없으면 None)
    def get_translation_cache(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        try:
            # ===== 1단계: 언어쌍 포함 캐시 데이터 구성 =====
            # 동일 텍스트라도 언어쌍이 다르면 다른 캐시 키 사용
            cache_data = f"{source_lang}→{target_lang}:{text}"
            cache_key = self._generate_cache_key("translation", cache_data)
            
            # ===== 2단계: Redis 캐시 조회 =====
            if self.redis_client:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    # 2-1: 바이트 데이터를 UTF-8 문자열로 디코딩
                    translated_text = cached_data.decode('utf-8')
                    logging.info(f"번역 캐시 히트: {source_lang}→{target_lang}, {text[:30]}...")
                    return translated_text
            else:
                # ===== 3단계: 메모리 캐시 폴백 조회 =====
                if cache_key in self._memory_cache:
                    logging.info(f"메모리 번역 캐시 히트: {source_lang}→{target_lang}, {text[:30]}...")
                    return self._memory_cache[cache_key]
            
            # ===== 4단계: 캐시 미스 =====
            return None
            
        except Exception as e:
            # ===== 예외 처리: 캐시 조회 실패 =====
            logging.error(f"번역 캐시 조회 실패: {e}")
            return None
    
    # 다국어 번역 캐시 저장 메서드
    # Args:
    #     original_text: 원본 텍스트
    #     translated_text: 번역된 텍스트
    #     source_lang: 원본 언어 코드
    #     target_lang: 목표 언어 코드
    #     expire_hours: 만료 시간 (시간 단위, 기본 7일)
    # Returns:
    #     bool: 저장 성공 여부
    def set_translation_cache(self, original_text: str, translated_text: str, 
                            source_lang: str, target_lang: str, expire_hours: int = 168) -> bool:
        try:
            # ===== 1단계: 언어쌍 포함 캐시 데이터 구성 =====
            cache_data = f"{source_lang}→{target_lang}:{original_text}"
            cache_key = self._generate_cache_key("translation", cache_data)
            
            # ===== 2단계: Redis 캐시 저장 =====
            if self.redis_client:
                # 2-1: 만료 시간 계산 (시간 → 초)
                expire_seconds = expire_hours * 3600
                # 2-2: 번역 결과를 UTF-8로 인코딩하여 저장
                result = self.redis_client.setex(cache_key, expire_seconds, translated_text.encode('utf-8'))
                logging.info(f"번역 캐시 저장: {source_lang}→{target_lang}, {original_text[:30]}...")
                return result
            else:
                # ===== 3단계: 메모리 캐시 폴백 저장 =====
                self._memory_cache[cache_key] = translated_text
                return True
                
        except Exception as e:
            # ===== 예외 처리: 캐시 저장 실패 =====
            logging.error(f"번역 캐시 저장 실패: {e}")
            return False

    # =================================
    # 검색 결과 캐싱 시스템
    # =================================
    
    # 벡터 검색 결과 캐시 조회 메서드
    # Args:
    #     query: 검색 질문
    #     search_params: 검색 파라미터 (유사도 임계값, 검색 카운트 등)
    # Returns:
    #     Optional[List[Dict]]: 캐시된 검색 결과 목록 (없으면 None)
    def get_search_results_cache(self, query: str, search_params: Dict) -> Optional[List[Dict]]:
        try:
            # ===== 1단계: 검색 파라미터 포함 캐시 키 생성 =====
            # 동일 질문이라도 검색 파라미터가 다르면 다른 결과를 생성
            cache_data = f"{query}:{json.dumps(search_params, sort_keys=True)}"
            cache_key = self._generate_cache_key("search", cache_data)
            
            # ===== 2단계: Redis 캐시 조회 =====
            if self.redis_client:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    # 2-1: 바이너리 데이터를 검색 결과 리스트로 역직렬화
                    search_results = self._deserialize_data(cached_data)
                    logging.info(f"검색 결과 캐시 히트: {query[:50]}...")
                    return search_results
            else:
                # ===== 3단계: 메모리 캐시 폴백 조회 =====
                if cache_key in self._memory_cache:
                    logging.info(f"메모리 검색 결과 캐시 히트: {query[:50]}...")
                    return self._memory_cache[cache_key]
            
            # ===== 4단계: 캐시 미스 =====
            return None
            
        except Exception as e:
            # ===== 예외 처리: 캐시 조회 실패 =====
            logging.error(f"검색 결과 캐시 조회 실패: {e}")
            return None
    
    # 벡터 검색 결과 캐시 저장 메서드
    # Args:
    #     query: 검색 질문
    #     search_params: 검색 파라미터
    #     search_results: 저장할 검색 결과 목록 (Pinecone 결과)
    #     expire_hours: 만료 시간 (시간 단위, 기본 1일)
    # Returns:
    #     bool: 저장 성공 여부
    def set_search_results_cache(self, query: str, search_params: Dict, 
                               search_results: List[Dict], expire_hours: int = 24) -> bool:
        try:
            # ===== 1단계: 검색 파라미터 포함 캐시 데이터 구성 =====
            cache_data = f"{query}:{json.dumps(search_params, sort_keys=True)}"
            cache_key = self._generate_cache_key("search", cache_data)
            serialized_data = self._serialize_data(search_results)
            
            # ===== 2단계: Redis 캐시 저장 =====
            if self.redis_client:
                # 2-1: 만료 시간 계산 (시간 → 초)
                expire_seconds = expire_hours * 3600
                # 2-2: 검색 결과를 직렬화하여 저장
                result = self.redis_client.setex(cache_key, expire_seconds, serialized_data)
                logging.info(f"검색 결과 캐시 저장: {query[:50]}... ({len(search_results)}개 결과)")
                return result
            else:
                # ===== 3단계: 메모리 캐시 폴백 저장 =====
                self._memory_cache[cache_key] = search_results
                return True
                
        except Exception as e:
            # ===== 예외 처리: 캐시 저장 실패 =====
            logging.error(f"검색 결과 캐시 저장 실패: {e}")
            return False

    # =================================
    # 캐시 관리 및 모니터링
    # =================================
    
    # 캐시 시스템 통계 정보 조회 메서드
    # Returns:
    #     Dict[str, Any]: 캐시 성능 통계 정보
    def get_cache_stats(self) -> Dict[str, Any]:
        try:
            # ===== Redis 캐시 통계 =====
            if self.redis_client:
                # Redis INFO 명령으로 상세 정보 수집
                info = self.redis_client.info()
                return {
                    'cache_type': 'Redis',                                     # 캐시 유형
                    'connected_clients': info.get('connected_clients', 0),     # 연결된 클라이언트 수
                    'used_memory': info.get('used_memory_human', '0B'),        # 사용 중인 메모리 (사람이 읽기 쉬운 형태)
                    'total_commands_processed': info.get('total_commands_processed', 0),  # 총 처리 명령 수
                    'cache_hit_ratio': self._calculate_hit_ratio()             # 캐시 히트 비율
                }
            else:
                # ===== 메모리 캐시 통계 =====
                return {
                    'cache_type': 'Memory',                                    # 캐시 유형
                    'cached_items': len(self._memory_cache),                   # 캐시된 항목 수
                    'cache_hit_ratio': 'N/A'                                   # 히트 비율 측정 불가
                }
        except Exception as e:
            # ===== 예외 처리: 통계 수집 실패 =====
            logging.error(f"캐시 통계 조회 실패: {e}")
            return {'error': str(e)}
    
    # 캐시 히트 비율 계산 메서드 (Redis 전용)
    # Returns:
    #     float: 히트 비율 (백분율, 0-100)
    def _calculate_hit_ratio(self) -> float:
        try:
            if self.redis_client:
                # ===== Redis keyspace 통계로 히트 비율 계산 =====
                info = self.redis_client.info()             # Redis 서버 정보 조회
                hits = info.get('keyspace_hits', 0)        # 캐시 히트 횟수
                misses = info.get('keyspace_misses', 0)     # 캐시 미스 횟수
                total = hits + misses                       # 총 요청 수
                
                # 히트 비율 계산 (소수점 2자리까지)
                return round((hits / total * 100), 2) if total > 0 else 0.0
            return 0.0
        except Exception as e:
            # ===== 예외 처리: 계산 실패 =====
            logging.error(f"히트 비율 계산 실패: {e}")
            return 0.0
    
    # 특정 접두사의 캐시 데이터 일괄 삭제 메서드
    # Args:
    #     prefix: 삭제할 캐시 키 접두사 (embedding, intent, typo 등)
    # Returns:
    #     int: 삭제된 항목 수
    def clear_cache_by_prefix(self, prefix: str) -> int:
        try:
            # ===== Redis 캐시 삭제 =====
            if self.redis_client:
                # 1단계: 패턴 매칭으로 대상 키 찾기
                pattern = f"{prefix}:*"
                keys = self.redis_client.keys(pattern)
                
                if keys:
                    # 2단계: 일괄 삭제 실행
                    deleted_count = self.redis_client.delete(*keys)
                    logging.info(f"캐시 삭제 완료: {prefix} 접두사, {deleted_count}개 항목")
                    return deleted_count
                return 0
            else:
                # ===== 메모리 캐시 삭제 =====
                # 1단계: 접두사 매칭 키 찾기
                keys_to_delete = [key for key in self._memory_cache.keys() if key.startswith(f"{prefix}:")]
                
                # 2단계: 각 키를 개별 삭제
                for key in keys_to_delete:
                    del self._memory_cache[key]
                    
                logging.info(f"메모리 캐시 삭제 완료: {prefix} 접두사, {len(keys_to_delete)}개 항목")
                return len(keys_to_delete)
                
        except Exception as e:
            # ===== 예외 처리: 캐시 삭제 실패 =====
            logging.error(f"캐시 삭제 실패: {e}")
            return 0
    
    # 캐시 시스템 사용 가능 여부 확인 메서드
    # Returns:
    #     bool: 캐시 시스템 정상 작동 여부
    def is_cache_available(self) -> bool:
        try:
            # ===== Redis 연결 상태 확인 =====
            if self.redis_client:
                self.redis_client.ping()  # Redis 서버 응답 테스트
                return True
            
            # ===== 메모리 캐시는 항상 사용 가능 =====
            return True
            
        except Exception as e:
            # ===== 예외 처리: 연결 확인 실패 =====
            logging.error(f"캐시 연결 확인 실패: {e}")
            return False
    
    # 캐시 워밍업 메서드 (자주 사용되는 데이터 미리 로드)
    # Args:
    #     warm_up_data: 워밍업할 데이터 목록 (임베딩, 의도분석 등)
    # Returns:
    #     int: 성공적으로 로드된 항목 수
    def warm_up_cache(self, warm_up_data: List[Dict]) -> int:
        # 워밍업 성공 카운터 초기화
        warmed_count = 0
        
        try:
            # ===== 각 워밍업 데이터 처리 =====
            for item in warm_up_data:
                cache_type = item.get('type')
                
                # ===== 임베딩 데이터 워밍업 =====
                if cache_type == 'embedding' and 'text' in item and 'embedding' in item:
                    if self.set_embedding_cache(item['text'], item['embedding']):
                        warmed_count += 1
                        
                # ===== 의도 분석 데이터 워밍업 =====
                elif cache_type == 'intent' and 'query' in item and 'intent_data' in item:
                    if self.set_intent_analysis_cache(item['query'], item['intent_data']):
                        warmed_count += 1
                        
            # ===== 워밍업 완료 로깅 =====
            logging.info(f"캐시 워밍업 완료: {warmed_count}개 항목 로드")
            return warmed_count
            
        except Exception as e:
            # ===== 예외 처리: 워밍업 실패 =====
            logging.error(f"캐시 워밍업 실패: {e}")
            return warmed_count
