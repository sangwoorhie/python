#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
임베딩 생성 모델 모듈
"""

import logging
from typing import Optional
from src.utils.memory_manager import memory_cleanup


class EmbeddingGenerator:
    """텍스트 임베딩 생성을 담당하는 클래스"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.model_name = 'text-embedding-3-small'
        self.max_text_length = 8000
    
    def create_embedding(self, text: str) -> Optional[list]:
        """OpenAI API를 사용하여 텍스트를 벡터로 변환하는 메서드"""
        # 빈 문자열뿐만 아니라 공백만 있는 문자열도 걸러냄
        if not text or not text.strip():
            return None
            
        try:
            with memory_cleanup():
                # OpenAI Embedding API 호출
                response = self.openai_client.embeddings.create(
                    model=self.model_name,
                    input=text[:self.max_text_length]  # 텍스트 길이 제한
                )
                
                # 메모리 효율성을 위해 벡터만 복사 후 응답 객체 삭제
                embedding = response.data[0].embedding.copy()
                del response  # 원본 응답 객체 즉시 삭제
                return embedding
                
        except Exception as e:
            logging.error(f"임베딩 생성 실패: {e}")
            return None
