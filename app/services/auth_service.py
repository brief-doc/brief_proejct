from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.core.security import hash_password, verify_password
from app.db.models import User, UserSession
from app.schemas.user import UserCreate


def get_user(db: Session, id: int):
    return db.query(User).filter(User.user_id == id).first()


def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.user_email == email).first()


def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(User).offset(skip).limit(limit).all()


def get_user_session_by_token(db: Session, session_token: str):
    return db.query(UserSession).filter(UserSession.session_token == session_token).first()


def get_session_by_id(db: Session, session_id: int):
    return db.query(UserSession).filter(UserSession.session_id == session_id).first()


def get_user_by_session_token(db: Session, session_token: str):
    session = get_user_session_by_token(db, session_token)
    if session is None:
        return None
    if session.expires_at and session.expires_at < datetime.now(timezone.utc):
        session.is_active = False
        db.commit()
        return None
    if not session.is_active:
        return None
    return get_user(db, session.user_id)


def get_user_sessions(db: Session, user_id: int):
    return db.query(UserSession).filter(UserSession.user_id == user_id).all()


def create_user_session(
    db: Session,
    user_id: int,
    session_token: str,
    expires_at: datetime,
    ip_address: str | None = None,
    user_agent: str | None = None,
):
    user_session = UserSession(
        user_id=user_id,
        session_token=session_token,
        created_at=datetime.now(timezone.utc),
        expires_at=expires_at,
        is_active=True,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    db.add(user_session)
    db.commit()
    db.refresh(user_session)
    return user_session


def deactivate_session(db: Session, session: UserSession):
    session.is_active = False
    db.commit()
    return session


def create_user(db: Session, user: UserCreate):
    db_user = User(
        user_email=user.email,
        user_password=hash_password(user.password),
        user_rank=1,
        user_name=user.name,
        user_create=datetime.now(),
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def reset_user_password(db: Session, user_id: int, password: str = "000000"):
    user = get_user(db, user_id)
    if not user:
        return None

    user.user_password = hash_password(password)
    user.user_login = None
    user.user_update = datetime.now(timezone.utc)

    db.commit()
    db.refresh(user)
    return user


def delete_user(db: Session, id: int):
    db_user = get_user(db, id)
    if not db_user:
        return None
    db.delete(db_user)
    db.commit()
    return db_user


def change_password(db: Session, user_id: int, current_password: str, new_password: str, user_login: datetime | None = None):
    """
    사용자 비밀번호 변경 및 user_login 시간 업데이트
    
    Args:
        db: Database session
        user_id: 사용자 ID
        current_password: 현재 비밀번호 (평문)
        new_password: 새로운 비밀번호 (평문)
        user_login: 로그인 시간 (기본값: 현재 시간)
    
    Returns:
        업데이트된 User 객체 또는 None (현재 비밀번호 불일치 시)
    """
    user = get_user(db, user_id)
    if not user:
        return None
    
    # 현재 비밀번호 검증
    if not verify_password(current_password, user.user_password):
        return None
    
    user.user_password = hash_password(new_password)
    user.user_login = user_login or datetime.now(timezone.utc)
    user.user_update = datetime.now(timezone.utc)
    
    db.commit()
    db.refresh(user)
    return user
