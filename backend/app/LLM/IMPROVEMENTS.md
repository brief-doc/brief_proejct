# LLM 파트 코드 개선 가이드

## 📋 개선 내용 요약

### 1️⃣ **메타데이터 확장**

**기존 (old):**
```python
chunk.metadata = {
    "doc_id": document_id,
    "user_id": user_id
}
```

**개선된 것 (new):**
```python
chunk.metadata = {
    "doc_id": document_id,
    "user_id": user_id,
    "file_name": "document.pdf",      # ← 파일명 추가
    "cat_id": 2,                       # ← 카테고리 ID 추가
    "content_sum": "요약 텍스트",      # ← 요약 내용 추가
    "created_at": "2026-05-29T12:30"  # ← 생성 시간 추가
}      
```

**효과:** PostgreSQL의 doc, category 테이블과 동기화 가능

---

### 2️⃣ **PostgreSQL 연동**

**새로운 함수들:**

```python
# DB 연결
def get_db_connection():
    """PostgreSQL 연결 (실패 시 None 반환)"""

# 문서 메타데이터 저장
def save_doc_to_db(doc_id, file_name, file_type, content_sum, cat_id, user_id):
    """INSERT INTO doc 테이블"""

# 작업 기록 저장
def create_job_record(doc_id, user_id, job_type):
    """INSERT INTO job 테이블"""
```

**사용 예:**
```python
# 벡터 DB + PostgreSQL 동시 저장
save_to_vector_db(
    file_path,
    doc_id=1,
    user_id=100,
    file_name="report.pdf",
    cat_id=2,
    content_sum="2026년 상반기 매출 분석..."
)
```

---

### 3️⃣ **필터링 기능**

**사용자별 문서 검색:**
```python
# 특정 사용자(user_id=100)의 문서만 검색
results = search_documents(
    query="매출 분석",
    user_id=100,         # ← 필터링
    cat_id=None,         # None이면 전체 카테고리
    top_k=5
)
```

**카테고리별 문서 검색:**
```python
# 카테고리 2의 문서만 검색
results = search_documents(
    query="계약서",
    user_id=None,
    cat_id=2,            # ← 필터링
    top_k=5
)
```

---

## 🔧 API 엔드포인트 변경

### 기존
```
POST /upload-and-summarize/
- doc_id, user_id

POST /batch-upload/
- start_doc_id, user_id
```

### 개선된 것
```
POST /upload-and-summarize/
- doc_id, user_id, cat_id ← 추가

POST /batch-upload/
- start_doc_id, user_id, cat_id ← 추가

POST /search/ ← 새로운 엔드포인트
- query, user_id (optional), cat_id (optional), top_k

POST /rag-query/ ← 새로운 엔드포인트
- query, user_id (optional), cat_id (optional), top_k
```

---

## 📦 새 파일 설명

### 1. **main.py (개선됨)**
- PostgreSQL 연동 함수 추가
- 메타데이터 확장
- `/search/` 엔드포인트 추가 (필터링 검색)
- `/rag-query/` 엔드포인트 추가 (RAG 기반 질의응답)

### 2. **vectordb_manager_v2.py** (새로운 모듈)
```python
from LLM.vectordb_manager_v2 import ImprovedVectorDBManager

manager = ImprovedVectorDBManager(chroma_path="./chroma_pdf_db")

# 벡터 저장
manager.create_and_save_vector_store(
    chunks=chunks,
    doc_id=1,
    user_id=100,
    file_name="document.pdf",
    cat_id=2,
    content_sum="요약..."
)

# 필터링 검색
results = manager.search_with_filter(
    query="검색어",
    user_id=100,
    cat_id=2,
    top_k=5
)
```

### 3. **rag_pipeline_v2.py** (새로운 모듈)
```python
from LLM.rag_pipeline_v2 import ImprovedRAGPipeline

pipeline = ImprovedRAGPipeline(chroma_path="./chroma_pdf_db")

# RAG 질의응답
result = pipeline.query_with_context(
    query="매출이 얼마나 증가했나?",
    user_id=100,
    cat_id=2
)
# 결과: {'status': 'success', 'answer': '...', 'references': [...]}

# 사용자 통계
stats = pipeline.get_user_statistics(user_id=100)
```

---

## 🔌 연동 방법

### 옵션 1: 기존 코드 유지 + 새 모듈 병행
```python
# 기존 API는 그대로 사용
# + 새로운 모듈을 별도로 import

from LLM.vectordb_manager_v2 import ImprovedVectorDBManager
from LLM.rag_pipeline_v2 import ImprovedRAGPipeline
```

### 옵션 2: 완전 마이그레이션 (추천)
```python
# main.py에서 새 모듈 사용
manager = ImprovedVectorDBManager()
pipeline = ImprovedRAGPipeline()
```

---

## 📊 DB 테이블 매핑

| Chroma 메타데이터 | PostgreSQL 테이블 | 설명 |
|---|---|---|
| doc_id | doc.doc_id | 문서 ID |
| user_id | doc.user_id | 사용자 ID |
| file_name | doc.file_name | 파일명 |
| cat_id | doc.cat_id | 카테고리 ID |
| content_sum | doc.content_full | 문서 내용/요약 |
| - | doc.file_type | 파일 타입 (pdf) |
| - | job.* | 작업 기록 |
| - | history.* | 쿼리 이력 |

---

## ⚠️ 주의사항

1. **PostgreSQL 설정**
   ```python
   DB_CONFIG = {
       "host": "localhost",
       "port": 5432,
       "database": "pdf_db",
       "user": "postgres",
       "password": "your_password"  # ← 실제 비밀번호로 변경
   }
   ```

2. **테이블 존재 확인**
   - `doc` 테이블 필수
   - `category` 테이블 필수
   - `job` 테이블 필수 (배치 작업 추적용)
   - `history` 테이블 (선택사항, 쿼리 이력용)

3. **임베딩 모델**
   - 기존: `jhgan/ko-sroberta-multitask` (유지)
   - GPU 필수 (CUDA 활성화됨)

---

## 🚀 사용 예제

### 1. 단일 PDF 업로드 (메타데이터 포함)
```bash
curl -X POST "http://localhost:8000/upload-and-summarize/" \
  -F "file=@document.pdf" \
  -F "doc_id=1" \
  -F "user_id=100" \
  -F "cat_id=2"
```

### 2. 배치 업로드 (여러 파일)
```bash
curl -X POST "http://localhost:8000/batch-upload/" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.pdf" \
  -F "start_doc_id=10" \
  -F "user_id=100" \
  -F "cat_id=3"
```

### 3. 필터링 검색
```bash
curl -X POST "http://localhost:8000/search/" \
  -d "query=매출 분석" \
  -d "user_id=100" \
  -d "cat_id=2" \
  -d "top_k=5"
```

### 4. RAG 질의응답
```bash
curl -X POST "http://localhost:8000/rag-query/" \
  -d "query=2026년 목표 매출은?" \
  -d "user_id=100" \
  -d "cat_id=2"
```

---

## ✅ 체크리스트

- [ ] `main.py` 개선 사항 검토
- [ ] `vectordb_manager_v2.py` 모듈 확인
- [ ] `rag_pipeline_v2.py` 모듈 확인
- [ ] PostgreSQL DB_CONFIG 업데이트
- [ ] 필요 테이블 생성 확인
- [ ] 테스트 실행 (단일 업로드 → 배치 업로드 → 검색)
