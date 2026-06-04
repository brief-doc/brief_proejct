#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DB_CONFIG 확인 스크립트
PostgreSQL 연결 설정의 인코딩 문제를 진단합니다.
"""

from .config import DB_CONFIG
import sys

print("=" * 60)
print("DB_CONFIG 진단")
print("=" * 60)

for key, value in DB_CONFIG.items():
    print(f"\n[{key}]")
    print(f"  값: {value}")
    print(f"  타입: {type(value).__name__}")
    
    if isinstance(value, str):
        try:
            # UTF-8 인코딩 테스트
            encoded = value.encode('utf-8')
            print(f"  UTF-8 바이트: {encoded}")
            print(f"  길이: {len(value)} 글자, {len(encoded)} 바이트")
            
            # ASCII 확인
            try:
                value.encode('ascii')
                print(f"  ASCII 호환: ✓ (특수문자 없음)")
            except UnicodeEncodeError as e:
                print(f"  ASCII 호환: ✗ (위치 {e.start}: {value[e.start]})")
        except Exception as e:
            print(f"  인코딩 에러: {e}")

print("\n" + "=" * 60)
print("권장사항:")
print("=" * 60)
print("1. password는 ASCII 문자만 사용하세요 (특수문자, 한글 제외)")
print("2. 다른 필드도 ASCII 문자만 사용을 권장합니다")
print("3. 현재 config.py의 password를 변경하려면 PostgreSQL에서도 변경해야 합니다")
