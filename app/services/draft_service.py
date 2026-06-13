from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session

from app.db.models import Draft
from app.schemas.draft import DraftCreate, DraftUpdate

KST = timezone(timedelta(hours=9))


def _now() -> datetime:
    return datetime.now(KST)


def create_draft(db: Session, author_id: int, payload: DraftCreate) -> Draft:
    status = "pending" if payload.action == "submit" else "draft"
    draft = Draft(
        author_id=author_id,
        title=payload.title,
        content=payload.content,
        source_doc_id=payload.source_doc_id,
        status=status,
        created_at=_now(),
        updated_at=_now(),
    )
    db.add(draft)
    db.commit()
    db.refresh(draft)
    return draft


def get_drafts(
    db: Session,
    author_id: int,
    status: str | None = None,
    keyword: str | None = None,
    sort_by: str = "created_at",
    skip: int = 0,
    limit: int = 3,
) -> tuple[int, list[Draft]]:
    query = db.query(Draft).filter(
        Draft.author_id == author_id,
        Draft.status != "canceled",  # 취소된 기안은 기본 제외
    )

    if status is not None:
        query = query.filter(Draft.status == status)

    if keyword:
        query = query.filter(Draft.title.ilike(f"%{keyword}%"))

    total_count = query.count()

    if sort_by == "title":
        query = query.order_by(Draft.title.asc())
    elif sort_by == "asc":
        query = query.order_by(Draft.created_at.asc())
    else:
        query = query.order_by(Draft.created_at.desc())

    drafts = query.offset(skip).limit(limit).all()
    return total_count, drafts


def get_draft_by_id(db: Session, draft_id: int) -> Draft | None:
    """author 필터 없이 draft_id만으로 조회 (403/404 구별용)"""
    return db.query(Draft).filter(Draft.draft_id == draft_id).first()


def get_draft_detail(db: Session, draft_id: int, author_id: int) -> Draft | None:
    return (
        db.query(Draft)
        .filter(Draft.draft_id == draft_id, Draft.author_id == author_id)
        .first()
    )


def update_draft(
    db: Session, draft_id: int, author_id: int, payload: DraftUpdate
) -> Draft | None:
    draft = get_draft_detail(db, draft_id, author_id)
    if not draft:
        return None

    # 임시저장(draft) 또는 반려(rejected) 상태만 수정 가능
    if draft.status not in ("draft", "rejected"):
        raise ValueError(f"'{draft.status}' 상태의 기안은 수정할 수 없습니다.")

    data = payload.model_dump(exclude_unset=True)
    action = data.pop("action", None)

    for key, value in data.items():
        setattr(draft, key, value)

    # 반려 후 재상신이면 pending으로 전환하고 반려 정보 초기화
    if action == "submit":
        draft.status = "pending"
        draft.reject_reason = None
        draft.decided_at = None
    elif action == "save":
        draft.status = "draft"

    draft.updated_at = _now()
    db.commit()
    db.refresh(draft)
    return draft


def cancel_draft(db: Session, draft_id: int, author_id: int) -> Draft | None:
    draft = get_draft_detail(db, draft_id, author_id)
    if not draft:
        return None

    # 대기(pending) 상태만 취소 가능
    if draft.status != "pending":
        raise ValueError(f"'{draft.status}' 상태의 기안은 취소할 수 없습니다.")

    draft.status = "canceled"
    draft.updated_at = _now()
    db.commit()
    db.refresh(draft)
    return draft
