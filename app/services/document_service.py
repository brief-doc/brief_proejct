from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.models import Document, Job


# 문서 목록 조회(최신 작업 상태와 함께)
def get_docs_with_latest_job(
    db: Session,
    user_id: int | None = None,
    category: str | None = None,
    skip: int = 0,
    limit: int = 50,
):
    # 1. 기본 쿼리 생성 (조인 및 DISTINCT 설정)
    query = (
        db.query(Document, Job)
        .join(Job, Job.jdoc_id == Document.doc_id)  # 조인 조건 (이전 질문의 jdoc_id 기준)
        .distinct(Document.doc_id)
        .filter(Document.is_hidden.is_(False))
    )

    # 2. 동적 필터링 처리 (값이 들어온 경우에만 필터 추가)
    if user_id is not None:
        query = query.filter(Document.user_id == user_id)

    if category is not None:
        query = query.filter(Document.category == category)

    # 3. 정렬 및 페이징 적용 후 실행
    # (DISTINCT ON을 쓸 때 첫 번째 정렬 기준은 반드시 distinct에 넣은 컬럼이어야 합니다)
    stmt = query.order_by(Document.doc_id, desc(Job.created_at)).offset(skip).limit(limit)

    results = stmt.all()
    return results
