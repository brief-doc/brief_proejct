from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.database import get_db
from app.schemas.draft import (
    ApprovalDetail,
    DecisionRequest,
    DraftCreate,
    DraftDetail,
    DraftUpdate,
    PaginatedApprovalResponse,
    PaginatedDraftResponse,
)
from app.services import draft_service, notification_service

router = APIRouter(prefix="/drafts", tags=["drafts"])


# ── 기안 작성 / 목록 ────────────────────────────────────────────────────────────


@router.post("/", response_model=DraftDetail, status_code=status.HTTP_201_CREATED)
def create_draft(
    payload: DraftCreate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    draft = draft_service.create_draft(db, author_id=current_user.user_id, payload=payload)
    if payload.action == "submit" and draft.approver_id:
        try:
            notification_service.create_notification(
                db=db,
                user_id=draft.approver_id,
                message=f"'{draft.title}' 기안이 상신되었습니다.",
                domain_type="APPROVAL",
                resource_id=draft.draft_id,
            )
        except Exception:
            pass
    return draft


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


# ── 결재자 전용 엔드포인트 (/{draft_id} 보다 먼저 등록해야 라우팅 충돌 방지) ──────


@router.get("/approvals/", response_model=PaginatedApprovalResponse)
def list_approvals(
    skip: int = Query(0, ge=0),
    limit: int = Query(3, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    total, items = draft_service.get_approval_list(db, approver_id=current_user.user_id, skip=skip, limit=limit)
    page = (skip // limit) + 1 if limit > 0 else 1
    return {"items": items, "total_count": total, "page": page, "limit": limit}


@router.get("/approvals/{draft_id}", response_model=ApprovalDetail)
def get_approval_detail(
    draft_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    detail = draft_service.get_approval_detail(db, draft_id=draft_id, approver_id=current_user.user_id)
    if not detail:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="기안을 찾을 수 없거나 접근 권한이 없습니다.",
        )
    return detail


@router.post("/{draft_id}/decision", response_model=DraftDetail)
def process_decision(
    draft_id: int,
    payload: DecisionRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    draft = draft_service.process_decision(
        db,
        draft_id=draft_id,
        approver_id=current_user.user_id,
        action=payload.action,
        reject_reason=payload.reject_reason,
    )
    if not draft:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="대기 중인 기안을 찾을 수 없거나 결재 권한이 없습니다.",
        )
    # 기안자에게 결재 결과 알림
    try:
        label = "승인" if payload.action == "approved" else "반려"
        notification_service.create_notification(
            db=db,
            user_id=draft.author_id,
            message=f"'{draft.title}' 기안이 {label}되었습니다.",
            domain_type="APPROVAL",
            resource_id=draft.draft_id,
        )
    except Exception:
        pass
    return draft


# ── 기안 단건 / 수정 / 취소 ────────────────────────────────────────────────────


@router.get("/{draft_id}", response_model=DraftDetail)
def get_draft(
    draft_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
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
        draft = draft_service.update_draft(db, draft_id=draft_id, author_id=current_user.user_id, payload=payload)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    if not draft:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="기안을 찾을 수 없습니다.")
    if payload.action == "submit" and draft.approver_id:
        try:
            notification_service.create_notification(
                db=db,
                user_id=draft.approver_id,
                message=f"'{draft.title}' 기안이 상신되었습니다.",
                domain_type="APPROVAL",
                resource_id=draft.draft_id,
            )
        except Exception:
            pass
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
