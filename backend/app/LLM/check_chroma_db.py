"""
ChromaDB에 저장된 벡터 데이터 조회 스크립트
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# config.py에서 설정 가져오기
from .config import EMBEDDING_CONFIG, CHROMA_DB_PATH

# 임베딩 모델 (config.py의 EMBEDDING_CONFIG 사용, 저장할 때와 동일)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_CONFIG["model_name"],
    model_kwargs={'device': EMBEDDING_CONFIG["device"]},
    encode_kwargs={'normalize_embeddings': EMBEDDING_CONFIG["normalize_embeddings"]}
)

# ChromaDB 로드
print("=" * 80)
print("ChromaDB 저장 데이터 조회")
print("=" * 80)
print(f"📁 저장 위치: {CHROMA_DB_PATH}\n")

try:
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )
    
    # 저장된 문서 개수
    print(f"✅ ChromaDB 연결 성공!\n")
    
    # 전체 데이터 조회
    try:
        all_data = vectorstore.get()
        total_chunks = len(all_data['ids'])
        print(f"📊 저장된 청크 총 개수: {total_chunks}개\n")
        
        # 메타데이터 확인
        if all_data['metadatas']:
            print("📝 메타데이터 정보:")
            print("-" * 80)
            
            # 문서별로 그룹화
            docs_info = {}
            for i, metadata in enumerate(all_data['metadatas']):
                if metadata:
                    doc_id = metadata.get('doc_id', 'Unknown')
                    if doc_id not in docs_info:
                        docs_info[doc_id] = {
                            'chunks': 0,
                            'user_id': metadata.get('user_id', 'Unknown')
                        }
                    docs_info[doc_id]['chunks'] += 1
            
            for doc_id, info in docs_info.items():
                print(f"  📄 문서 ID: {doc_id}")
                print(f"     - 사용자 ID: {info['user_id']}")
                print(f"     - 저장된 청크 수: {info['chunks']}개")
            print()
        
        # 샘플 데이터 출력
        if total_chunks > 0:
            print("📄 샘플 데이터 (첫 3개 청크):")
            print("-" * 80)
            for i in range(min(3, total_chunks)):
                print(f"\n[청크 {i+1}]")
                print(f"ID: {all_data['ids'][i]}")
                print(f"문서: {all_data.get('metadatas', [{}])[i]}")
                if all_data.get('documents'):
                    doc_content = all_data['documents'][i][:200]  # 처음 200자만
                    print(f"내용 (미리보기): {doc_content}...")
                if all_data.get('distances'):
                    print(f"거리: {all_data['distances'][i]}")
        
        print("\n" + "=" * 80)
        print("✅ 모든 벡터 데이터가 성공적으로 저장되었습니다!")
        print("=" * 80)
        
    except Exception as e:
        print(f"⚠️ 데이터 조회 중 오류: {e}")
    
except Exception as e:
    print(f"❌ ChromaDB 연결 실패: {e}")
    print("💡 팁: 먼저 test_api.py를 실행해서 PDF를 업로드하세요.")
