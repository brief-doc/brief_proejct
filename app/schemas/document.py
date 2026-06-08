from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class DocResponse(BaseModel):
    doc_id: int
    file_name: Optional[str] = None
    category: Optional[str] = None
    created_at: Optional[datetime] = None

    user_id: int

    job_status: Optional[str] = None

    class Config:
        from_attributes = True


# 상세용 — 원문 포함
class DocDetail(DocResponse):
    content_sum: Optional[str] = None
    content_full: Optional[str] = None
    file_type: Optional[str] = None
    updated_at: Optional[datetime] = None
