#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
메모리 관리 유틸리티 모듈
"""

import gc
from contextlib import contextmanager


@contextmanager
def memory_cleanup():
    """메모리 정리를 위한 컨텍스트 매니저"""
    try:
        yield  # with 블록 내부 코드 실행
    finally:
        gc.collect()  # 가비지 컬렉션 강제 실행으로 메모리 정리
