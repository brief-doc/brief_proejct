from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.database import get_db
from app.schemas.notification import NotificationOut, PaginatedNotificationResponse
from app.services import notification_service

router = APIRouter(prefix="/notifications", tags=["notifications"])


# ── SSE 구독 ──────────────────────────────────────────────────────────────────


@router.get("/subscribe")
async def subscribe(current_user=Depends(get_current_user)):
    """
    SSE 연결을 수립합니다. 클라이언트는 이 엔드포인트에 연결을 유지하며
    새 알림이 발생하면 JSON 이벤트를 수신합니다.
    """
    user_id: int = current_user.user_id
    queue = notification_service.subscribe(user_id)

    async def event_stream():
        # 연결 확인 초기 이벤트
        yield f"data: {json.dumps({'type': 'connected', 'user_id': user_id})}\n\n"
        try:
            while True:
                try:
                    # 30초마다 heartbeat (프록시/방화벽 연결 유지)
                    payload = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(payload)}\n\n"
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            notification_service.unsubscribe(user_id)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # nginx 버퍼링 비활성화
            "Connection": "keep-alive",
        },
    )


# ── 알림 목록 조회 ─────────────────────────────────────────────────────────────


@router.get("/", response_model=PaginatedNotificationResponse)
def list_notifications(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    total, items = notification_service.get_notifications(db, user_id=current_user.user_id, skip=skip, limit=limit)
    page = (skip // limit) + 1 if limit > 0 else 1
    return {"items": items, "total_count": total, "page": page, "limit": limit}


# ── 읽음 처리 ─────────────────────────────────────────────────────────────────


@router.patch("/{noti_id}/read", response_model=NotificationOut)
def read_notification(
    noti_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    noti = notification_service.mark_as_read(db, noti_id=noti_id, user_id=current_user.user_id)
    if not noti:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="알림을 찾을 수 없습니다.",
        )
    return noti
