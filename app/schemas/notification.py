from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class NotificationOut(BaseModel):
    noti_id: int
    user_id: int
    message: str
    link: Optional[str] = None
    is_read: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class PaginatedNotificationResponse(BaseModel):
    items: list[NotificationOut]
    total_count: int
    page: int
    limit: int
