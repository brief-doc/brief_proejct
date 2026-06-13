from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.database import get_db
from app.schemas.draft import (
    DraftCreate,
    DraftUpdate,
    DraftDetail,
    PaginatedDraftResponse,
)
from app.services import draft_service

router = APIRouter(prefix="/drafts", tags=["drafts"])


@router.post("/", response_model=DraftDetail, status_code=status.HTTP_201_CREATED)
def create_draft(
    payload: DraftCreate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    return draft_service.create_draft(db, author_id=current_user.user_id, payload=payload)


@router.get("/", response_model=PaginatedDraftResponse)
def list_drafts(
    status: str | None = Query(None),
    keyword: str | None = Query(None),
    sort_by: str = Query("created_at"),
    skip: int = 0,
    limit: int = 3,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    total_count, drafts = draft_service.get_drafts(
        db,
        author_id=current_user.user_id,
        status=status,
        keyword=keyword,
        sort_by=sort_by,
        skip=skip,
        limit=limit,
    )
    page = (skip // limit) + 1 if limit > 0 else 1
    return {"items": drafts, "total_count": total_count, "page": page, "limit": limit}


@router.get("/{draft_id}", response_model=DraftDetail)
def get_draft(
    draft_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    # 존재 여부 먼저 확인 → 404 / 타인 소유 → 403 구별
    existing = draft_service.get_draft_by_id(db, draft_id=draft_id)
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="기안을 찾을 수 없습니다.")
    if existing.author_id != current_user.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="접근 권한이 없습니다.")
    return existing


@router.patch("/{draft_id}", response_model=DraftDetail)
def update_draft(
    draft_id: int,
    payload: DraftUpdate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    try:
        draft = draft_service.update_draft(
            db, draft_id=draft_id, author_id=current_user.user_id, payload=payload
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    if not draft:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="기안을 찾을 수 없습니다.")
    return draft


@router.delete("/{draft_id}", status_code=status.HTTP_204_NO_CONTENT)
def cancel_draft(
    draft_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    try:
        draft = draft_service.cancel_draft(db, draft_id=draft_id, author_id=current_user.user_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    if not draft:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="기안을 찾을 수 없습니다.")
