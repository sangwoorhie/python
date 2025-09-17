#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Redis 기반 지능형 캐싱 시스템
API 호출 최적화 및 성능 향상을 위한 캐싱 매니저
"""

import json
import hashlib
import logging
import redis
import pickle
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta


class CacheManager:
    """Redis 기반 지능형 캐싱 시스템"""
    
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0, redis_password=None):
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                decode_responses=False,  # 바이너리 데이터 지원
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            # 연결 테스트
            self.redis_client.ping()
            logging.info("Redis 캐시 연결 성공")
        except Exception as e:
            logging.warning(f"Redis 연결 실패, 메모리 캐시로 폴백: {e}")
            self.redis_client = None
            self._memory_cache = {}
    
    def _generate_cache_key(self, prefix: str, data: str) -> str:
        """캐시 키 생성 (해시 기반)"""
        # 텍스트를 SHA256 해시로 변환하여 키 생성
        hash_obj = hashlib.sha256(data.encode('utf-8'))
        return f"{prefix}:{hash_obj.hexdigest()[:16]}"
    
    def _serialize_data(self, data: Any) -> bytes:
        """데이터 직렬화"""
        return pickle.dumps(data)
    
    def _deserialize_data(self, data: bytes) -> Any:
        """데이터 역직렬화"""
        return pickle.loads(data)

    # =================================
    # 임베딩 캐싱 시스템
    # =================================
    
    def get_embedding_cache(self, text: str) -> Optional[List[float]]:
        """임베딩 캐시 조회"""
        try:
            cache_key = self._generate_cache_key("embedding", text)
            
            if self.redis_client:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    embedding = self._deserialize_data(cached_data)
                    logging.info(f"임베딩 캐시 히트: {text[:50]}...")
                    return embedding
            else:
                # 메모리 캐시 폴백
                if cache_key in self._memory_cache:
                    logging.info(f"메모리 임베딩 캐시 히트: {text[:50]}...")
                    return self._memory_cache[cache_key]
            
            return None
            
        except Exception as e:
            logging.error(f"임베딩 캐시 조회 실패: {e}")
            return None
    
    def set_embedding_cache(self, text: str, embedding: List[float], expire_hours: int = 168) -> bool:
        """임베딩 캐시 저장 (기본 7일)"""
        try:
            cache_key = self._generate_cache_key("embedding", text)
            serialized_data = self._serialize_data(embedding)
            
            if self.redis_client:
                expire_seconds = expire_hours * 3600
                result = self.redis_client.setex(cache_key, expire_seconds, serialized_data)
                logging.info(f"임베딩 캐시 저장: {text[:50]}... ({len(embedding)}차원)")
                return result
            else:
                # 메모리 캐시 폴백 (만료시간 없음)
                self._memory_cache[cache_key] = embedding
                return True
                
        except Exception as e:
            logging.error(f"임베딩 캐시 저장 실패: {e}")
            return False

    # =================================
    # 의도 분석 캐싱 시스템
    # =================================
    
    def get_intent_analysis_cache(self, query: str) -> Optional[Dict]:
        """의도 분석 캐시 조회"""
        try:
            cache_key = self._generate_cache_key("intent", query)
            
            if self.redis_client:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    intent_data = self._deserialize_data(cached_data)
                    logging.info(f"의도 분석 캐시 히트: {query[:50]}...")
                    return intent_data
            else:
                if cache_key in self._memory_cache:
                    logging.info(f"메모리 의도 분석 캐시 히트: {query[:50]}...")
                    return self._memory_cache[cache_key]
            
            return None
            
        except Exception as e:
            logging.error(f"의도 분석 캐시 조회 실패: {e}")
            return None
    
    def set_intent_analysis_cache(self, query: str, intent_data: Dict, expire_hours: int = 72) -> bool:
        """의도 분석 캐시 저장 (기본 3일)"""
        try:
            cache_key = self._generate_cache_key("intent", query)
            serialized_data = self._serialize_data(intent_data)
            
            if self.redis_client:
                expire_seconds = expire_hours * 3600
                result = self.redis_client.setex(cache_key, expire_seconds, serialized_data)
                logging.info(f"의도 분석 캐시 저장: {query[:50]}...")
                return result
            else:
                self._memory_cache[cache_key] = intent_data
                return True
                
        except Exception as e:
            logging.error(f"의도 분석 캐시 저장 실패: {e}")
            return False

    # =================================
    # 오타 수정 캐싱 시스템
    # =================================
    
    def get_typo_correction_cache(self, text: str) -> Optional[str]:
        """오타 수정 캐시 조회"""
        try:
            cache_key = self._generate_cache_key("typo", text)
            
            if self.redis_client:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    corrected_text = cached_data.decode('utf-8')
                    logging.info(f"오타 수정 캐시 히트: {text[:50]}...")
                    return corrected_text
            else:
                if cache_key in self._memory_cache:
                    logging.info(f"메모리 오타 수정 캐시 히트: {text[:50]}...")
                    return self._memory_cache[cache_key]
            
            return None
            
        except Exception as e:
            logging.error(f"오타 수정 캐시 조회 실패: {e}")
            return None
    
    def set_typo_correction_cache(self, original_text: str, corrected_text: str, expire_hours: int = 168) -> bool:
        """오타 수정 캐시 저장 (기본 7일)"""
        try:
            cache_key = self._generate_cache_key("typo", original_text)
            
            if self.redis_client:
                expire_seconds = expire_hours * 3600
                result = self.redis_client.setex(cache_key, expire_seconds, corrected_text.encode('utf-8'))
                logging.info(f"오타 수정 캐시 저장: {original_text[:30]}... → {corrected_text[:30]}...")
                return result
            else:
                self._memory_cache[cache_key] = corrected_text
                return True
                
        except Exception as e:
            logging.error(f"오타 수정 캐시 저장 실패: {e}")
            return False

    # =================================
    # 번역 캐싱 시스템
    # =================================
    
    def get_translation_cache(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """번역 캐시 조회"""
        try:
            cache_data = f"{source_lang}→{target_lang}:{text}"
            cache_key = self._generate_cache_key("translation", cache_data)
            
            if self.redis_client:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    translated_text = cached_data.decode('utf-8')
                    logging.info(f"번역 캐시 히트: {source_lang}→{target_lang}, {text[:30]}...")
                    return translated_text
            else:
                if cache_key in self._memory_cache:
                    logging.info(f"메모리 번역 캐시 히트: {source_lang}→{target_lang}, {text[:30]}...")
                    return self._memory_cache[cache_key]
            
            return None
            
        except Exception as e:
            logging.error(f"번역 캐시 조회 실패: {e}")
            return None
    
    def set_translation_cache(self, original_text: str, translated_text: str, 
                            source_lang: str, target_lang: str, expire_hours: int = 168) -> bool:
        """번역 캐시 저장 (기본 7일)"""
        try:
            cache_data = f"{source_lang}→{target_lang}:{original_text}"
            cache_key = self._generate_cache_key("translation", cache_data)
            
            if self.redis_client:
                expire_seconds = expire_hours * 3600
                result = self.redis_client.setex(cache_key, expire_seconds, translated_text.encode('utf-8'))
                logging.info(f"번역 캐시 저장: {source_lang}→{target_lang}, {original_text[:30]}...")
                return result
            else:
                self._memory_cache[cache_key] = translated_text
                return True
                
        except Exception as e:
            logging.error(f"번역 캐시 저장 실패: {e}")
            return False

    # =================================
    # 검색 결과 캐싱 시스템
    # =================================
    
    def get_search_results_cache(self, query: str, search_params: Dict) -> Optional[List[Dict]]:
        """검색 결과 캐시 조회"""
        try:
            # 검색 파라미터를 포함한 캐시 키 생성
            cache_data = f"{query}:{json.dumps(search_params, sort_keys=True)}"
            cache_key = self._generate_cache_key("search", cache_data)
            
            if self.redis_client:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    search_results = self._deserialize_data(cached_data)
                    logging.info(f"검색 결과 캐시 히트: {query[:50]}...")
                    return search_results
            else:
                if cache_key in self._memory_cache:
                    logging.info(f"메모리 검색 결과 캐시 히트: {query[:50]}...")
                    return self._memory_cache[cache_key]
            
            return None
            
        except Exception as e:
            logging.error(f"검색 결과 캐시 조회 실패: {e}")
            return None
    
    def set_search_results_cache(self, query: str, search_params: Dict, 
                               search_results: List[Dict], expire_hours: int = 24) -> bool:
        """검색 결과 캐시 저장 (기본 1일)"""
        try:
            cache_data = f"{query}:{json.dumps(search_params, sort_keys=True)}"
            cache_key = self._generate_cache_key("search", cache_data)
            serialized_data = self._serialize_data(search_results)
            
            if self.redis_client:
                expire_seconds = expire_hours * 3600
                result = self.redis_client.setex(cache_key, expire_seconds, serialized_data)
                logging.info(f"검색 결과 캐시 저장: {query[:50]}... ({len(search_results)}개 결과)")
                return result
            else:
                self._memory_cache[cache_key] = search_results
                return True
                
        except Exception as e:
            logging.error(f"검색 결과 캐시 저장 실패: {e}")
            return False

    # =================================
    # 캐시 관리 및 모니터링
    # =================================
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 정보 조회"""
        try:
            if self.redis_client:
                info = self.redis_client.info()
                return {
                    'cache_type': 'Redis',
                    'connected_clients': info.get('connected_clients', 0),
                    'used_memory': info.get('used_memory_human', '0B'),
                    'total_commands_processed': info.get('total_commands_processed', 0),
                    'cache_hit_ratio': self._calculate_hit_ratio()
                }
            else:
                return {
                    'cache_type': 'Memory',
                    'cached_items': len(self._memory_cache),
                    'cache_hit_ratio': 'N/A'
                }
        except Exception as e:
            logging.error(f"캐시 통계 조회 실패: {e}")
            return {'error': str(e)}
    
    def _calculate_hit_ratio(self) -> float:
        """캐시 히트 비율 계산 (Redis 전용)"""
        try:
            if self.redis_client:
                info = self.redis_client.info()
                hits = info.get('keyspace_hits', 0)
                misses = info.get('keyspace_misses', 0)
                total = hits + misses
                return round((hits / total * 100), 2) if total > 0 else 0.0
            return 0.0
        except Exception as e:
            logging.error(f"히트 비율 계산 실패: {e}")
            return 0.0
    
    def clear_cache_by_prefix(self, prefix: str) -> int:
        """특정 접두사의 캐시 데이터 삭제"""
        try:
            if self.redis_client:
                pattern = f"{prefix}:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    deleted_count = self.redis_client.delete(*keys)
                    logging.info(f"캐시 삭제 완료: {prefix} 접두사, {deleted_count}개 항목")
                    return deleted_count
                return 0
            else:
                # 메모리 캐시에서 삭제
                keys_to_delete = [key for key in self._memory_cache.keys() if key.startswith(f"{prefix}:")]
                for key in keys_to_delete:
                    del self._memory_cache[key]
                logging.info(f"메모리 캐시 삭제 완료: {prefix} 접두사, {len(keys_to_delete)}개 항목")
                return len(keys_to_delete)
                
        except Exception as e:
            logging.error(f"캐시 삭제 실패: {e}")
            return 0
    
    def is_cache_available(self) -> bool:
        """캐시 시스템 사용 가능 여부 확인"""
        try:
            if self.redis_client:
                self.redis_client.ping()
                return True
            return True  # 메모리 캐시는 항상 사용 가능
        except Exception as e:
            logging.error(f"캐시 연결 확인 실패: {e}")
            return False
    
    def warm_up_cache(self, warm_up_data: List[Dict]) -> int:
        """캐시 워밍업 (자주 사용되는 데이터 미리 로드)"""
        warmed_count = 0
        try:
            for item in warm_up_data:
                cache_type = item.get('type')
                if cache_type == 'embedding' and 'text' in item and 'embedding' in item:
                    if self.set_embedding_cache(item['text'], item['embedding']):
                        warmed_count += 1
                elif cache_type == 'intent' and 'query' in item and 'intent_data' in item:
                    if self.set_intent_analysis_cache(item['query'], item['intent_data']):
                        warmed_count += 1
                        
            logging.info(f"캐시 워밍업 완료: {warmed_count}개 항목 로드")
            return warmed_count
            
        except Exception as e:
            logging.error(f"캐시 워밍업 실패: {e}")
            return warmed_count
