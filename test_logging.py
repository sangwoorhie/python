#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
로깅 시스템 테스트 스크립트
"""

import logging
import os
import sys

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_logging():
    """로깅 시스템 테스트"""
    print("=== 로깅 시스템 테스트 시작 ===")
    
    # 로깅 설정 (free_4_ai_answer_generator.py와 동일한 설정)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    
    # 로그 포맷 정의
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    try:
        # 로그 디렉토리 생성
        os.makedirs('/home/ec2-user/python/logs', exist_ok=True)
        
        # 파일 핸들러 생성
        file_handler = logging.FileHandler('/home/ec2-user/python/logs/test_logging.log', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # 콘솔 핸들러 생성
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # 핸들러 추가
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # 테스트 로그 출력
        logging.info("=== 로깅 시스템 테스트 ===")
        logging.info("이 메시지가 보이면 로깅이 정상 작동합니다.")
        logging.info("=== 테스트 완료 ===")
        
        print("✅ 로깅 테스트 완료")
        print("📁 로그 파일: /home/ec2-user/python/logs/test_logging.log")
        
        return True
        
    except Exception as e:
        print(f"❌ 로깅 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    test_logging()
