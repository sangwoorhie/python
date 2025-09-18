#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
메모리 관리 유틸리티 모듈
- AI API 호출 및 대용량 데이터 처리시 메모리 최적화
- 컨텍스트 매니저를 통한 자동 메모리 정리
- 가비지 컬렉션 강제 실행으로 메모리 누수 방지
"""

import gc
from contextlib import contextmanager


# ===== 메모리 정리를 위한 컨텍스트 매니저 =====
@contextmanager
def memory_cleanup():
    # 메모리 정리를 위한 컨텍스트 매니저
    # - with 블록 실행 전후로 메모리 정리 수행
    # - AI API 호출, 대용량 데이터 처리시 메모리 누수 방지
    # - 자동 가비지 컬렉션으로 메모리 효율성 향상
    try:
        # ===== 1단계: with 블록 내부 코드 실행 =====
        # 사용자가 with memory_cleanup(): 블록에서 실행하는 코드
        yield
    finally:
        # ===== 2단계: 메모리 정리 (항상 실행) =====
        # with 블록이 정상 종료되거나 예외 발생시에도 반드시 실행
        gc.collect()  # 가비지 컬렉션 강제 실행으로 메모리 정리
