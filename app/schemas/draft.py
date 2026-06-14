from __future__ import annotations
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, model_validator


class DraftCreate(BaseModel):
    title: str
    content: str
    source_doc_id: Optional[int] = None
    action: Literal["save", "submit"]  # save → draft, submit → pending


class DraftUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    source_doc_id: Optional[int] = None
    action: Optional[Literal["save", "submit"]] = None  # 반려 후 재상신 시 submit


# 목록 조회용 (대시보드) — 최소 필드만
class DraftListItem(BaseModel):
    draft_id: int
    title: str
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


# 단건 상세 조회용 — 전체 필드, reject_reason은 rejected 상태일 때만 반환
class DraftDetail(BaseModel):
    draft_id: int
    author_id: int
    source_doc_id: Optional[int] = None
    title: str
    content: str
    status: str
    approver_id: Optional[int] = None
    reject_reason: Optional[str] = None
    decided_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def clear_reject_reason_unless_rejected(self) -> "DraftDetail":
        if self.status != "rejected":
            self.reject_reason = None
        return self

    model_config = {"from_attributes": True}


class PaginatedDraftResponse(BaseModel):
    items: list[DraftListItem]
    total_count: int
    page: int
    limit: int
