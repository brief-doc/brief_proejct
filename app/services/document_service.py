from datetime import datetime, timezone

from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.db.models import Document, Job
from app.schemas.document import DocResponse, DocUpdate


# 문서 목록 조회(최신 작업 상태와 함께)
def get_docs_with_latest_job(
    db: Session,
    user_id: int,
    category: str | None = None,
    keyword: str | None = None, 
    sort_by: str = "created_at",
    skip: int = 0,
    limit: int = 50,
) -> tuple[int, list[DocResponse]]: 
    
    # 1. 기본 베이스 쿼리 빌드
    query = (
        db.query(Document, Job)
        .join(Job, Job.doc_id == Document.doc_id)
        .distinct(Document.doc_id)
        .filter(
            Document.user_id == user_id,    
            Document.is_deleted.is_(False)
        )
    )

    # 2. 카테고리
    if category is not None:
        query = query.filter(Document.category == category)
        
    # 3. 검색어(keyword)
    if keyword is not None and keyword.strip() != "":
        query = query.filter(Document.file_name.contains(keyword))

    if sort_by == "title":
        # 제목순 (오름차순)
        order_stmt = [Document.file_name.asc(), Document.doc_id.desc()]
    elif sort_by == "oldest":
        # 오래된순
        order_stmt = [Document.doc_id.asc(), Job.job_start.asc()]
    else:
        # 최신순 (기본값: Job 시작 시간 혹은 Document 생성일 역순)
        order_stmt = [Document.doc_id.desc(), desc(Job.job_start)]

    # 4. 필터링된 전체 개수를 구함.
    total_count = query.count()

    # 5. 정렬 및 페이징 분량=
    stmt = query.order_by(*order_stmt).offset(skip).limit(limit)
    results = stmt.all()  # list[(Document, Job)]

    # 6. DTO 변환 연산
    docs_list = [
        DocResponse(
            doc_id=doc.doc_id,
            file_name=doc.file_name,
            category=doc.category,
            created_at=doc.created_at,
            user_id=doc.user_id,
            job_status=job.job_status,
        )
        for doc, job in results
    ]
    
    # 7. 총 개수와 리스트를 함께 리턴
    return total_count, docs_list

# 문서 상세 조회
def get_docs_detail(
    db: Session,
    doc_id: int,
    user_id: int,
):
    doc_detail = (
        db.query(Document)
        .filter(
            Document.user_id == user_id,    
            Document.is_deleted.is_(False),
            Document.doc_id == doc_id,
            #완료 상태만 조회 가능
        ).first()   
    )
    
    return doc_detail


# 문서 삭제
def soft_delete_doc(
    db: Session,
    doc_id: int,
    user_id: int,
        
):
    doc = (
        db.query(Document)
        .filter(
            Document.doc_id == doc_id,
            Document.user_id == user_id,
            Document.is_deleted.is_(False)
        )
        .first()

    )
    if not doc:
        return False
    
    doc.is_deleted = True

    db.commit()
    return True


def update_doc(db: Session, doc_id: int, user_id: int, payload: DocUpdate):
    doc = get_docs_detail(db, doc_id, user_id)
    if not doc:
        return None

    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        setattr(doc, key, value)

    doc.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(doc)
    return doc