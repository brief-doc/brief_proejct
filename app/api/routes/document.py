from fastapi import APIRouter, Depends, status, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.document import DocResponse, DocDetail, DocUpdate
from app.services import document_service as doc_service
from app.core.security import get_current_user

router = APIRouter(prefix="/documents", tags=["docs"])

# 1. 전체 조회(Soft Delete된 것은 제외)
@router.get("/", response_model=list[DocResponse])
def list_documents(
    category: str | None = None,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
):
    return doc_service.get_docs_with_latest_job(
        db,
        user_id=current_user.user_id,
        category=category,
        skip=skip,
        limit=limit,
    )

# 2. 단건 상세 조회
@router.get("/{doc_id}", response_model=DocDetail)
def document(
    doc_id: int,
    db: Session = Depends(get_db),
    current_user =  Depends(get_current_user),
    
):
    return doc_service.get_docs_detail(
        db,
        doc_id = doc_id,
        user_id = current_user.user_id,
    )

# 3. Soft Delete 실행
@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def soft_delete_document(
    doc_id: int, 
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
):
    # 핵심 로직은 모두 서비스로 위임합니다.
    success = doc_service.soft_delete_doc(db, doc_id=doc_id, user_id=current_user.user_id)
    
    # 결과에 따른 분기(예외 처리)만 라우터가 담당
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="문서를 찾을 수 없거나 이미 삭제된 문서입니다."
        )
    return

# 4. 수정
@router.patch("/{doc_id}")
def update_document(
    doc_id: int,
    payload: DocUpdate,                       
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),   
):

    doc = doc_service.update_doc(db, doc_id, current_user.user_id, payload)
    if doc is None:
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다")
    return doc