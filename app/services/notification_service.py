from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from sqlalchemy.orm import Session

from app.db.models import Notification

KST = timezone(timedelta(hours=9))

# ── 전역 이벤트 루프 참조 (startup 시 설정) ─────────────────────────────────
_loop: Optional[asyncio.AbstractEventLoop] = None

# ── 사용자별 SSE 큐 ──────────────────────────────────────────────────────────
_user_queues: Dict[int, asyncio.Queue] = {}


def set_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _loop
    _loop = loop


# ── SSE 구독 관리 ─────────────────────────────────────────────────────────────

def subscribe(user_id: int) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue()
    _user_queues[user_id] = q
    return q


def unsubscribe(user_id: int) -> None:
    _user_queues.pop(user_id, None)


# ── 알림 생성 + 실시간 Push ──────────────────────────────────────────────────

def create_notification(
    db: Session,
    user_id: int,
    message: str,
    domain_type: str,
    resource_id: int,
) -> Notification:
    """DB에 알림을 저장하고 해당 사용자의 SSE 큐로 즉시 push."""
    link = f"{domain_type}:{resource_id}"
    now = datetime.now(KST)

    noti = Notification(
        user_id=user_id,
        message=message,
        link=link,
        is_read=False,
        created_at=now,
    )
    db.add(noti)
    db.commit()
    db.refresh(noti)

    # 동기 코드(스레드풀) → 비동기 큐로 안전하게 push
    _push_sync(user_id, _serialize(noti))
    return noti


def _serialize(noti: Notification) -> dict:
    return {
        "noti_id": noti.noti_id,
        "message": noti.message,
        "link": noti.link,
        "is_read": noti.is_read,
        "created_at": noti.created_at.isoformat() if noti.created_at else None,
    }


def _push_sync(user_id: int, payload: dict) -> None:
    if _loop and _loop.is_running() and user_id in _user_queues:
        _loop.call_soon_threadsafe(_user_queues[user_id].put_nowait, payload)


def push_event(user_id: int, payload: dict) -> None:
    """알림 이외의 실시간 이벤트(진행률 등)를 SSE 큐로 전송."""
    _push_sync(user_id, payload)


# ── DB 조회 / 읽음 처리 ────────────────────────────────────────────────────────

def get_notifications(
    db: Session,
    user_id: int,
    skip: int = 0,
    limit: int = 20,
) -> tuple[int, list[Notification]]:
    query = (
        db.query(Notification)
        .filter(Notification.user_id == user_id)
        .order_by(Notification.created_at.desc())
    )
    total = query.count()
    items = query.offset(skip).limit(limit).all()
    return total, items


def mark_as_read(db: Session, noti_id: int, user_id: int) -> Optional[Notification]:
    noti = (
        db.query(Notification)
        .filter(Notification.noti_id == noti_id, Notification.user_id == user_id)
        .first()
    )
    if not noti:
        return None
    noti.is_read = True
    db.commit()
    db.refresh(noti)
    return noti
