#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
임베딩 모델 로드 없이 qa_data를 ChromaDB에 저장하는 간단한 스크립트
기존 ChromaDB의 임베딩을 재사용하거나, 임베딩 없이 메타데이터만 저장
"""

import json
import os
import sys
from pathlib import Path
import chromadb
from .config import CHROMA_DB_PATH

def extract_content_from_json(data):
    """JSON 파일에서 텍스트 추출"""
    content_parts = []
    
    # 기본 정보
    info = data.get('info', {})
    case_nm = info.get('caseNm', '')
    case_title = info.get('caseTitle', '')
    
    if case_nm:
        content_parts.append(f"사건명: {case_nm}")
    if case_title:
        content_parts.append(f"판결: {case_title}")
    
    # 질의응답
    jdgmn_info = data.get('jdgmnInfo', [])
    for item in jdgmn_info:
        question = item.get('question', '')
        answer = item.get('answer', '')
        if question:
            content_parts.append(f"Q: {question}")
        if answer:
            content_parts.append(f"A: {answer}")
    
    # 요약
    summary = data.get('Summary', [])
    for item in summary:
        summ_pass = item.get('summ_pass', '')
        if summ_pass:
            content_parts.append(f"요약: {summ_pass}")
    
    # 키워드
    keywords = data.get('keyword_tagg', [])
    keyword_list = [kw.get('keyword', '') for kw in keywords if kw.get('keyword')]
    if keyword_list:
        content_parts.append(f"키워드: {', '.join(keyword_list)}")
    
    return '\n'.join(content_parts)

def load_qa_data_simple():
    """임베딩 없이 ChromaDB에 데이터 로드"""
    
    print("=" * 60)
    print("ChromaDB 직접 접근 (임베딩 없이)")
    print("=" * 60)
    
    # ChromaDB 클라이언트 생성 (임베딩 함수 없음)
    print("\n📂 ChromaDB 접근 중...")
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_or_create_collection(name="qa_collection")
        print("✓ ChromaDB 접근 성공")
    except Exception as e:
        print(f"❌ ChromaDB 접근 실패: {e}")
        return False
    
    # 이미 저장된 문서 ID 조회
    print("\n🔍 기존 문서 조회 중...")
    try:
        existing_data = collection.get()
        existing_ids = set(existing_data.get('ids', []))
        print(f"✓ 이미 저장된 문서: {len(existing_ids)}개")
    except Exception as e:
        print(f"⚠️ 기존 문서 조회 실패: {e}")
        existing_ids = set()
    
    # qa_data 디렉토리
    qa_data_dir = Path(__file__).parent.parent / "qa_data"
    print(f"\n📂 qa_data 경로: {qa_data_dir}")
    
    if not qa_data_dir.exists():
        print(f"❌ qa_data 디렉토리를 찾을 수 없습니다")
        return False
    
    # JSON 파일 목록
    json_files = sorted(list(qa_data_dir.glob("*.json")))
    total_files = len(json_files)
    print(f"📊 총 파일: {total_files}개")
    
    # 배치 로드
    batch_size = 100
    new_documents = 0
    skipped_documents = 0
    
    print(f"\n🚀 로드 시작 ({batch_size}개씩 배치)...\n")
    
    texts = []
    ids = []
    metadatas = []
    
    for idx, json_file in enumerate(json_files):
        try:
            # 파일 읽기
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # doc_id 생성 (caseNo 또는 파일명 사용)
            case_no = data.get('info', {}).get('caseNo')
            if not case_no:
                case_no = json_file.stem
            
            # 이미 존재하는지 확인
            if case_no in existing_ids:
                skipped_documents += 1
            else:
                # 콘텐츠 추출
                content = extract_content_from_json(data)
                
                # 배치에 추가
                texts.append(content)
                ids.append(case_no)
                metadatas.append({
                    "source": json_file.name,
                    "case_no": case_no,
                    "user_id": 1,  # qa_data = user_id 1
                    "doc_type": "qa_data"
                })
                
                new_documents += 1
            
            # 진행 상황 보고 (50개마다)
            if (idx + 1) % 50 == 0:
                total_processed = new_documents + skipped_documents
                print(f"  처리: {idx + 1}/{total_files} | 신규: {new_documents} | 중복: {skipped_documents}")
            
            # 배치 추가
            if len(texts) >= batch_size:
                print(f"  💾 {batch_size}개 저장 중...")
                collection.upsert(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas
                )
                texts, ids, metadatas = [], [], []
                print(f"  ✓ 저장 완료")
        
        except Exception as e:
            print(f"❌ 파일 처리 실패 ({json_file.name}): {e}")
            continue
    
    # 남은 배치 저장
    if texts:
        print(f"\n💾 마지막 {len(texts)}개 저장 중...")
        collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        print(f"✓ 마지막 배치 저장 완료")
    
    # 최종 보고
    print(f"\n{'=' * 60}")
    print(f"📊 로드 완료")
    print(f"{'=' * 60}")
    print(f"신규 문서: {new_documents}개")
    print(f"중복 스킵: {skipped_documents}개")
    print(f"총 처리: {new_documents + skipped_documents}개")
    
    # 현재 총 문서 수
    try:
        final_data = collection.get()
        total_in_db = len(final_data.get('ids', []))
        print(f"\n💾 현재 ChromaDB 총 문서: {total_in_db}개")
        print(f"진행률: {total_in_db / 77716 * 100:.1f}% (목표: 77,716개)")
    except Exception as e:
        print(f"⚠️ 최종 문서 수 조회 실패: {e}")
    
    return True

if __name__ == "__main__":
    try:
        success = load_qa_data_simple()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ 사용자 중단")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 예기치 않은 오류: {e}")
        sys.exit(1)
