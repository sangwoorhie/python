#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
최적화된 검색 서비스 모듈
- 캐싱, 배치 처리, 지능형 API 관리를 통합한 고성능 검색 시스템
- 다층 검색 및 의미론적 유사성 계산
- 실시간 성능 최적화 및 조기 종료 메커니즘
"""

import logging
import time
from typing import List, Dict, Set, Optional
from src.utils.memory_manager import memory_cleanup
from src.utils.text_preprocessor import TextPreprocessor
from src.utils.intelligent_api_manager import (
    IntelligentAPIManager, APICallRequest, APICallStrategy
)
from src.models.question_analyzer import QuestionAnalyzer
import numpy as np

# ===== 최적화된 Pinecone 벡터 검색을 담당하는 메인 클래스 =====
class OptimizedSearchService:
    
    # OptimizedSearchService 초기화
    # Args:
    #     pinecone_index: Pinecone 벡터 인덱스
    #     api_manager: 지능형 API 관리자
    def __init__(self, pinecone_index, api_manager: IntelligentAPIManager):
        self.index = pinecone_index                           # Pinecone 벡터 검색 인덱스
        self.api_manager = api_manager                        # 캐싱/배치 API 관리자
        self.text_processor = TextPreprocessor()              # 텍스트 전처리 도구
        
        # QuestionAnalyzer 초기화 (의도 관련성 계산용)
        self._question_analyzer = QuestionAnalyzer(api_manager.openai_client)
        
        # ===== 검색 최적화 설정 =====
        self.search_config = {
            'max_layers': 5,                                  # 최대 검색 레이어 수
            'adaptive_layer_count': True,                     # 동적 레이어 수 조정
            'early_termination': True,                        # 조기 종료 활성화
            'similarity_threshold': 0.75,                     # 유사도 임계값
            'min_results_threshold': 3,                       # 최소 결과 수 임계값
            'enable_result_caching': True,                    # 검색 결과 캐싱 활성화
            'cache_ttl_hours': 24                            # 캐시 유효시간 (시간)
        }
        
        # ===== 성능 최적화를 위한 캐시 시스템 =====
        self.embedding_cache = {}                            # 임베딩 재사용 캐시
        self.search_history = {}                             # 검색 기록 캐시
        
        # logging.info("최적화된 검색 서비스 초기화 완료")
