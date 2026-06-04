#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qa_data JSON 파일들을 ChromaDB에 로드하는 스크립트
민사법 판결문 질의응답 데이터를 벡터DB에 저장
"""

import json
import os
import sys
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from .config import EMBEDDING_CONFIG, CHROMA_DB_PATH

def extract_content_from_json(data):
    """JSON 파일에서 학습할 텍스트 추출"""
    
    content_parts = []
    
    # 1. 기본 정보 추출
    info = data.get('info', {})
    case_nm = info.get('caseNm', '')
    case_title = info.get('caseTitle', '')
    
    if case_nm:
        content_parts.append(f"사건명: {case_nm}")
    if case_title:
        content_parts.append(f"판결: {case_title}")
    
    # 2. 질의응답 정보 추출
    jdgmn_info = data.get('jdgmnInfo', [])
    for item in jdgmn_info:
        question = item.get('question', '')
        answer = item.get('answer', '')
        if question:
            content_parts.append(f"Q: {question}")
        if answer:
            content_parts.append(f"A: {answer}")
    
    # 3. 요약 정보 추출
    summary = data.get('Summary', [])
    for item in summary:
        summ_pass = item.get('summ_pass', '')
        if summ_pass:
            content_parts.append(f"요약: {summ_pass}")
    
    # 4. 키워드 추출
    keywords = data.get('keyword_tagg', [])
    keyword_list = [kw.get('keyword', '') for kw in keywords if kw.get('keyword')]
    if keyword_list:
        content_parts.append(f"키워드: {', '.join(keyword_list)}")
    
    return '\n'.join(content_parts)

def load_qa_data_to_chroma(limit=None):
    """qa_data 디렉토리의 JSON 파일들을 ChromaDB에 로드
    
    Args:
        limit: 로드할 파일 개수 제한 (None이면 전체 로드)
    """
    
    # 임베딩 모델 초기화
    print("🤖 임베딩 모델 초기화 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_CONFIG["model_name"],
        model_kwargs={'device': EMBEDDING_CONFIG["device"]},
        encode_kwargs={'normalize_embeddings': EMBEDDING_CONFIG["normalize_embeddings"]}
    )
    
    # ChromaDB 초기화
    print("📊 ChromaDB 초기화 중...")
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )
    
    # 이미 저장된 문서의 doc_id 조회
    print("🔍 이미 저장된 문서 조회 중...")
    existing_docs = vectorstore.get(include=['metadatas'])
    existing_doc_ids = set()
    if existing_docs and 'metadatas' in existing_docs:
        for metadata in existing_docs['metadatas']:
            if metadata.get('doc_id'):
                existing_doc_ids.add(metadata['doc_id'])
    
    print(f"   ✓ 이미 저장된 문서: {len(existing_doc_ids)}개")
    
    # qa_data 디렉토리 경로
    qa_data_dir = Path(__file__).parent.parent / "qa_data"
    
    print(f"\n📂 qa_data 디렉토리: {qa_data_dir}")
    
    # JSON 파일 목록
    json_files = sorted(list(qa_data_dir.glob("*.json")))
    total_files = len(json_files)
    print(f"📊 총 파일 개수: {total_files}")
    
    # 제한 적용
    if limit:
        json_files = json_files[:limit]
    
    print(f"📥 로드할 파일: {len(json_files)}개\n")
    
    # JSON 파일들 로드
    documents = []
    failed_files = []
    skipped_files = []
    
    for idx, json_file in enumerate(json_files, 1):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 문서 정보 추출
                info = data.get('info', {})
                case_no = info.get('caseNo') or json_file.stem
                
                # 이미 저장된 문서는 건너뛰기
                if case_no in existing_doc_ids:
                    skipped_files.append(json_file.name)
                    if idx % 50 == 0:
                        print(f"  ⊘ {idx}/{len(json_files)} 처리 중 (건너뜀: {len(skipped_files)})")
                    continue
                
                case_nm = info.get('caseNm', '미분류')
                announce_date = info.get('judmnAdjuDe', '')
                
                # 메인 콘텐츠 추출
                content = extract_content_from_json(data)
                
                if content.strip():
                    documents.append({
                        'content': content,
                        'doc_id': case_no,
                        'case_name': case_nm,
                        'announce_date': announce_date,
                        'file_name': json_file.name,
                        'page_num': 1,
                        'user_id': 1,
                        'cat_id': 0
                    })
                    
                    if idx % 20 == 0:
                        print(f"  ✓ {idx}/{len(json_files)} 처리 완료 (새로 추가: {len(documents)})")
                        
        except Exception as e:
            failed_files.append((json_file.name, str(e)))
            print(f"  ✗ {json_file.name} 로드 실패: {e}")
    
    print(f"\n✅ 새로 추가할 문서: {len(documents)}개")
    print(f"⊘  건너뛴 문서: {len(skipped_files)}개")
    if failed_files:
        print(f"⚠️  실패한 파일: {len(failed_files)}개")
    
    # ChromaDB에 추가
    if documents:
        print(f"\n🔄 ChromaDB에 {len(documents)}개 새로운 문서 저장 중...")
        
        # metadatas 준비
        metadatas = []
        texts = []
        
        for doc in documents:
            texts.append(doc['content'])
            metadatas.append({
                'doc_id': doc['doc_id'],
                'case_name': doc['case_name'],
                'announce_date': doc['announce_date'],
                'file_name': doc['file_name'],
                'page_num': str(doc['page_num']),
                'user_id': str(doc['user_id']),
                'cat_id': str(doc['cat_id'])
            })
        
        # 배치 단위로 저장 (메모리 효율)
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            vectorstore.add_texts(texts=batch_texts, metadatas=batch_metadatas)
            print(f"  ✓ {min(i+batch_size, len(texts))}/{len(texts)} 저장 완료")
        
        print(f"\n✅ ChromaDB 저장 완료!")
        print(f"   - 새로 저장된 문서: {len(texts)}개")
        print(f"   - 기존 문서: {len(existing_doc_ids)}개")
        print(f"   - 총 문서: {len(existing_doc_ids) + len(texts)}개")
        print(f"   - 저장 위치: {CHROMA_DB_PATH}")
        
        return True
    else:
        print(f"\nℹ️  저장할 새로운 문서가 없습니다")
        print(f"   - 기존 문서: {len(existing_doc_ids)}개")
        print(f"   - 모든 파일이 이미 처리되었습니다")
        return False

def load_validation_data_to_chroma(embeddings=None, vectorstore=None):
    """Validation 폴더의 라벨링 JSON 파일들을 ChromaDB에 로드
    
    Args:
        embeddings: 재사용할 임베딩 모델 (None이면 새로 생성)
        vectorstore: 재사용할 Chroma 인스턴스 (None이면 새로 생성)
    """
    
    # 임베딩 모델 초기화 (재사용하지 않으면)
    if embeddings is None:
        print("\n🤖 임베딩 모델 초기화 중...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_CONFIG["model_name"],
            model_kwargs={'device': EMBEDDING_CONFIG["device"]},
            encode_kwargs={'normalize_embeddings': EMBEDDING_CONFIG["normalize_embeddings"]}
        )
    
    # ChromaDB 초기화 (재사용하지 않으면)
    if vectorstore is None:
        print("📊 ChromaDB 초기화 중...")
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )
    
    # 이미 저장된 문서의 doc_id 조회
    print("🔍 이미 저장된 문서 조회 중...")
    existing_docs = vectorstore.get(include=['metadatas'])
    existing_doc_ids = set()
    if existing_docs and 'metadatas' in existing_docs:
        for metadata in existing_docs['metadatas']:
            if metadata.get('doc_id'):
                existing_doc_ids.add(metadata['doc_id'])
    
    print(f"   ✓ 이미 저장된 문서: {len(existing_doc_ids)}개")
    
    # Validation 라벨링 데이터 디렉토리 경로
    validation_dir = Path(__file__).parent.parent.parent / "Desktop" / "ai-hub 민사 질의응답 all" / "01.민사법_LLM_사전학습_및_Instruction_Tuning_데이터" / "3.개방데이터" / "1.데이터" / "Validation" / "02.라벨링데이터"
    
    print(f"\n📂 라벨링 데이터 디렉토리: {validation_dir}")
    
    # JSON 파일 목록
    json_files = sorted(list(validation_dir.glob("*.json")))
    total_files = len(json_files)
    print(f"📊 총 파일 개수: {total_files}")
    print(f"📥 로드할 파일: {len(json_files)}개\n")
    
    # JSON 파일들 로드
    documents = []
    failed_files = []
    skipped_files = []
    
    for idx, json_file in enumerate(json_files, 1):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 문서 정보 추출 (라벨링 데이터 형식)
                info = data.get('info', {})
                case_no = info.get('caseNo') or json_file.stem
                
                # 이미 저장된 문서는 건너뛰기
                if case_no in existing_doc_ids:
                    skipped_files.append(json_file.name)
                    if idx % 100 == 0:
                        print(f"  ⊘ {idx}/{len(json_files)} 처리 중 (건너뜀: {len(skipped_files)})")
                    continue
                
                case_nm = info.get('caseNm', '미분류')
                announce_date = info.get('judmnAdjuDe', '')
                
                # 메인 콘텐츠 추출
                content = extract_content_from_json(data)
                
                if content.strip():
                    documents.append({
                        'content': content,
                        'doc_id': case_no,
                        'case_name': case_nm,
                        'announce_date': announce_date,
                        'file_name': json_file.name,
                        'page_num': 1,
                        'user_id': 2,  # Validation 데이터는 user_id 2로 구분
                        'cat_id': 0
                    })
                    
                    if idx % 100 == 0:
                        print(f"  ✓ {idx}/{len(json_files)} 처리 완료 (새로 추가: {len(documents)})")
                        
        except Exception as e:
            failed_files.append((json_file.name, str(e)))
            if idx % 500 == 0:
                print(f"  ✗ {idx}번째 파일에서 일부 로드 실패")
    
    print(f"\n✅ 새로 추가할 문서: {len(documents)}개")
    print(f"⊘  건너뛴 문서: {len(skipped_files)}개")
    if failed_files:
        print(f"⚠️  실패한 파일: {len(failed_files)}개")
    
    # ChromaDB에 추가
    if documents:
        print(f"\n🔄 ChromaDB에 {len(documents)}개 새로운 문서 저장 중...")
        
        # metadatas 준비
        metadatas = []
        texts = []
        
        for doc in documents:
            texts.append(doc['content'])
            metadatas.append({
                'doc_id': doc['doc_id'],
                'case_name': doc['case_name'],
                'announce_date': doc['announce_date'],
                'file_name': doc['file_name'],
                'page_num': str(doc['page_num']),
                'user_id': str(doc['user_id']),
                'cat_id': str(doc['cat_id'])
            })
        
        # 배치 단위로 저장 (메모리 효율)
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            vectorstore.add_texts(texts=batch_texts, metadatas=batch_metadatas)
            print(f"  ✓ {min(i+batch_size, len(texts))}/{len(texts)} 저장 완료")
        
        print(f"\n✅ ChromaDB 저장 완료!")
        print(f"   - 새로 저장된 문서: {len(texts)}개")
        print(f"   - 기존 문서: {len(existing_doc_ids)}개")
        print(f"   - 총 문서: {len(existing_doc_ids) + len(texts)}개")
        print(f"   - 저장 위치: {CHROMA_DB_PATH}")
        
        return len(existing_doc_ids) + len(texts)
    else:
        print(f"\nℹ️  저장할 새로운 문서가 없습니다")
        print(f"   - 기존 문서: {len(existing_doc_ids)}개")
        return len(existing_doc_ids)

def load_all_data_automatically():
    """전체 데이터를 두 단계로 자동으로 로드
    
    단계 1: qa_data 폴더의 기본 데이터 로드 (~4553개)
    단계 2: Validation 폴더의 라벨링 데이터 로드 (나머지 ~77716개)
    """
    
    print("=" * 60)
    print("전체 민사법 데이터 자동 로드 프로세스")
    print("=" * 60)
    
    # 데이터 디렉토리 경로 확인
    qa_data_dir = Path(__file__).parent.parent / "qa_data"
    validation_dir = Path(__file__).parent.parent.parent / "Desktop" / "ai-hub 민사 질의응답 all" / "01.민사법_LLM_사전학습_및_Instruction_Tuning_데이터" / "3.개방데이터" / "1.데이터" / "Validation" / "02.라벨링데이터"
    
    if not qa_data_dir.exists():
        print(f"\n❌ qa_data 디렉토리를 찾을 수 없습니다: {qa_data_dir}")
        return
    
    print(f"\n✓ qa_data 디렉토리 확인됨")
    
    if not validation_dir.exists():
        print(f"\n⚠️  Validation 디렉토리를 찾을 수 없습니다 (선택사항): {validation_dir}")
        validation_dir = None
    else:
        print(f"✓ Validation 디렉토리 확인됨")
    
    # ChromaDB 초기화 (기존 데이터 보존)
    print("\n📊 ChromaDB 초기화 중...")
    try:
        chroma_db_path = CHROMA_DB_PATH
        db_exists = os.path.exists(chroma_db_path)
        
        if db_exists:
            print(f"   ✓ 기존 ChromaDB 발견: {chroma_db_path}")
            # 기존 DB를 열되, 임베딩 함수 없이
            vectorstore = Chroma(
                persist_directory=chroma_db_path,
                embedding_function=None
            )
            print("   ✓ 기존 임베딩 함수 재사용")
        else:
            print(f"   새로운 ChromaDB 생성: {chroma_db_path}")
            print("   🤖 임베딩 모델 초기화 중... (첫 실행시 2-5분 소요)")
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_CONFIG["model_name"],
                model_kwargs={'device': EMBEDDING_CONFIG["device"]},
                encode_kwargs={'normalize_embeddings': EMBEDDING_CONFIG["normalize_embeddings"]}
            )
            vectorstore = Chroma(
                persist_directory=chroma_db_path,
                embedding_function=embeddings
            )
            print("   ✓ 새 ChromaDB 생성 완료")
    except Exception as e:
        print(f"   ❌ ChromaDB 초기화 실패: {e}")
        return
    
    # ===== 단계 1: qa_data 로드 =====
    print("\n" + "=" * 60)
    print("📌 단계 1: qa_data 폴더 데이터 로드")
    print("=" * 60)
    
    qa_data_dir = Path(__file__).parent.parent / "qa_data"
    json_files_qa = sorted(list(qa_data_dir.glob("*.json")))
    
    print(f"📂 qa_data 디렉토리: {qa_data_dir}")
    print(f"📊 qa_data 파일 개수: {len(json_files_qa)}")
    
    # 이미 저장된 문서의 doc_id 조회
    existing_docs = vectorstore.get(include=['metadatas'])
    existing_doc_ids = set()
    if existing_docs and 'metadatas' in existing_docs:
        for metadata in existing_docs['metadatas']:
            if metadata.get('doc_id'):
                existing_doc_ids.add(metadata['doc_id'])
    
    print(f"✓ 이미 저장된 문서: {len(existing_doc_ids)}개\n")
    
    # qa_data 로드
    documents_qa = []
    for idx, json_file in enumerate(json_files_qa, 1):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                info = data.get('info', {})
                # caseNo 없으면 파일명 사용 (unique 보장)
                case_no = info.get('caseNo') or json_file.stem
                
                if case_no not in existing_doc_ids:
                    case_nm = info.get('caseNm', '미분류')
                    announce_date = info.get('judmnAdjuDe', '')
                    content = extract_content_from_json(data)
                    
                    if content.strip():
                        documents_qa.append({
                            'content': content,
                            'doc_id': case_no,
                            'case_name': case_nm,
                            'announce_date': announce_date,
                            'file_name': json_file.name,
                            'page_num': 1,
                            'user_id': 1,
                            'cat_id': 0
                        })
                        existing_doc_ids.add(case_no)
                
                if idx % 100 == 0:
                    print(f"  ✓ qa_data: {idx}/{len(json_files_qa)} 처리 완료 (새로 추가: {len(documents_qa)})")
        except Exception as e:
            pass
    
    # qa_data 저장
    if documents_qa:
        print(f"\n🔄 qa_data: {len(documents_qa)}개 문서 저장 중...")
        batch_size = 50
        for i in range(0, len(documents_qa), batch_size):
            batch = documents_qa[i:i+batch_size]
            texts = [doc['content'] for doc in batch]
            metadatas = [{
                'doc_id': doc['doc_id'],
                'case_name': doc['case_name'],
                'announce_date': doc['announce_date'],
                'file_name': doc['file_name'],
                'page_num': str(doc['page_num']),
                'user_id': str(doc['user_id']),
                'cat_id': str(doc['cat_id'])
            } for doc in batch]
            vectorstore.add_texts(texts=texts, metadatas=metadatas)
            if (i + batch_size) % 500 == 0:
                print(f"  ✓ qa_data: {min(i+batch_size, len(documents_qa))}/{len(documents_qa)} 저장 완료")
        print(f"✅ qa_data 저장 완료: {len(documents_qa)}개\n")
    
    current_total = len(existing_doc_ids)
    print(f"📊 현재까지의 총 문서 개수: {current_total}개")
    
    # ===== 단계 2: Validation 데이터 로드 (선택사항) =====
    if validation_dir:
        print("\n" + "=" * 60)
        print("📌 단계 2: Validation 라벨링 데이터 로드")
        print("=" * 60)
    
    validation_dir = Path(__file__).parent.parent.parent / "Desktop" / "ai-hub 민사 질의응답 all" / "01.민사법_LLM_사전학습_및_Instruction_Tuning_데이터" / "3.개방데이터" / "1.데이터" / "Validation" / "02.라벨링데이터"
    
    json_files_val = sorted(list(validation_dir.glob("*.json")))
    print(f"📂 라벨링 데이터 디렉토리: {validation_dir}")
    print(f"📊 라벨링 데이터 파일 개수: {len(json_files_val)}")
    print(f"✓ 이미 저장된 문서: {len(existing_doc_ids)}개\n")
    
    # Validation 데이터 로드
    documents_val = []
    error_count = 0
    for idx, json_file in enumerate(json_files_val, 1):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                info = data.get('info', {})
                # caseNo 없으면 파일명 사용 (unique 보장)
                case_no = info.get('caseNo') or json_file.stem
                
                if case_no not in existing_doc_ids:
                    case_nm = info.get('caseNm', '미분류')
                    announce_date = info.get('judmnAdjuDe', '')
                    content = extract_content_from_json(data)
                    
                    if content.strip():
                        documents_val.append({
                            'content': content,
                            'doc_id': case_no,
                            'case_name': case_nm,
                            'announce_date': announce_date,
                            'file_name': json_file.name,
                            'page_num': 1,
                            'user_id': 2,
                            'cat_id': 0
                        })
                        existing_doc_ids.add(case_no)
                
                if idx % 500 == 0:
                    print(f"  ✓ Validation: {idx}/{len(json_files_val)} 처리 완료 (새로 추가: {len(documents_val)})")
        except Exception as e:
            error_count += 1
            if error_count <= 5:  # 처음 5개 에러만 상세 출력
                print(f"  ⚠️  {json_file.name} 처리 실패: {str(e)[:100]}")
    
    # Validation 데이터 저장
    if documents_val:
        print(f"\n🔄 Validation: {len(documents_val)}개 문서 저장 중...")
        batch_size = 50
        for i in range(0, len(documents_val), batch_size):
            batch = documents_val[i:i+batch_size]
            texts = [doc['content'] for doc in batch]
            metadatas = [{
                'doc_id': doc['doc_id'],
                'case_name': doc['case_name'],
                'announce_date': doc['announce_date'],
                'file_name': doc['file_name'],
                'page_num': str(doc['page_num']),
                'user_id': str(doc['user_id']),
                'cat_id': str(doc['cat_id'])
            } for doc in batch]
            vectorstore.add_texts(texts=texts, metadatas=metadatas)
            if (i + batch_size) % 500 == 0 or (i + batch_size) >= len(documents_val):
                print(f"  ✓ Validation: {min(i+batch_size, len(documents_val))}/{len(documents_val)} 저장 완료")
        print(f"✅ Validation 저장 완료: {len(documents_val)}개\n")
    
    final_total = len(existing_doc_ids)
    
    # ===== 최종 결과 =====
    print("=" * 60)
    print("✨ 모든 데이터 로드 완료!")
    print("=" * 60)
    print(f"📊 qa_data: {len(documents_qa)}개 추가")
    print(f"📊 Validation: {len(documents_val)}개 추가")
    print(f"📊 총 저장된 문서: {final_total}개")
    print(f"📌 목표: 77716개 (진행률: {final_total/77716*100:.1f}%)")
    print(f"💾 저장 위치: {CHROMA_DB_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    print("=" * 60)
    print("민사법 판결문 질의응답 데이터 로드")
    print("=" * 60)
    
    # 명령줄 인자 확인
    # 사용법:
    # python -m LLM.load_qa_data          -> 전체 자동 로드 (qa_data + Validation)
    # python -m LLM.load_qa_data qa      -> qa_data만 로드
    # python -m LLM.load_qa_data validation -> Validation만 로드
    # python -m LLM.load_qa_data 100     -> qa_data에서 처음 100개만 로드
    
    mode = "auto"
    limit = None
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg in ["auto", "all"]:
            mode = "auto"
            print(f"\n📌 전체 데이터 자동 로드 모드")
        elif arg == "qa":
            mode = "qa"
            print(f"\n📌 qa_data만 로드합니다")
        elif arg == "validation":
            mode = "validation"
            print(f"\n📌 Validation 데이터만 로드합니다")
        else:
            try:
                limit = int(arg)
                mode = "qa"
                print(f"\n📌 qa_data에서 {limit}개 파일만 로드합니다")
            except ValueError:
                mode = "auto"
                print(f"\n📌 전체 데이터 자동 로드 모드 (잘못된 인자 무시)")
    else:
        mode = "auto"
        print(f"\n📌 전체 데이터 자동 로드 모드 (qa_data + Validation)")
        print(f"   사용법: python -m LLM.load_qa_data [auto|qa|validation|숫자]")
    
    # 모드에 따라 실행
    if mode == "auto":
        # 전체 자동 로드
        load_all_data_automatically()
    elif mode == "qa":
        # qa_data만 로드
        success = load_qa_data_to_chroma(limit=limit)
        if success:
            print("\n✨ qa_data 로드 완료!")
        else:
            print("\n⚠️ 새로운 데이터가 없습니다")
    elif mode == "validation":
        # Validation만 로드
        total = load_validation_data_to_chroma()
        print(f"\n✨ Validation 데이터 로드 완료! (총 {total}개)")
    
    print("\n프로세스 완료")
