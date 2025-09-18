#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
임베딩 생성 모델 모듈
- OpenAI API를 사용한 텍스트 임베딩 생성
- 메모리 최적화 및 오류 처리 포함
- Pinecone 벡터 데이터베이스 연동용
"""

import logging
from typing import Optional
from src.utils.memory_manager import memory_cleanup

# ===== 텍스트 임베딩 생성을 담당하는 메인 클래스 =====
class EmbeddingGenerator:
    
    # EmbeddingGenerator 초기화
    # Args:
    #     openai_client: OpenAI API 클라이언트 인스턴스
    def __init__(self, openai_client):
        self.openai_client = openai_client                    # OpenAI API 클라이언트
        self.model_name = 'text-embedding-3-small'            # 사용할 임베딩 모델 (cost-effective)
        self.max_text_length = 8000                           # 최대 텍스트 길이 제한
    
    # OpenAI API를 사용하여 텍스트를 벡터로 변환하는 메서드
    # Args:
    #     text: 임베딩을 생성할 텍스트
    # Returns:
    #     Optional[list]: 생성된 임베딩 벡터 (실패시 None)
    def create_embedding(self, text: str) -> Optional[list]:
        # ===== 1단계: 입력 텍스트 유효성 검증 =====
        # 빈 문자열뿐만 아니라 공백만 있는 문자열도 걸러냄
        if not text or not text.strip():
            return None
            
        try:
            # ===== 2단계: 메모리 최적화 컨텍스트 시작 =====
            with memory_cleanup():
                # ===== 3단계: OpenAI Embedding API 호출 =====
                # - text-embedding-3-small 모델 사용 (성능과 비용의 균형)
                # - 텍스트 길이 제한으로 API 오류 방지
                response = self.openai_client.embeddings.create(
                    model=self.model_name,
                    input=text[:self.max_text_length]  # 텍스트 길이 제한 (8000자)
                )
                
                # ===== 4단계: 임베딩 벡터 추출 및 메모리 최적화 =====
                # 메모리 효율성을 위해 벡터만 복사 후 응답 객체 삭제
                embedding = response.data[0].embedding.copy()  # 벡터 데이터만 추출
                del response  # 원본 응답 객체 즉시 삭제 (메모리 절약)
                
                # ===== 5단계: 임베딩 벡터 반환 =====
                return embedding
                
        except Exception as e:
            # ===== 예외 처리: 임베딩 생성 실패 =====
            logging.error(f"임베딩 생성 실패: {e}")
            return None
