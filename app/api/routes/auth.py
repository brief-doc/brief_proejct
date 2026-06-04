from fastapi import APIRouter, Depends, HTTPException, status, Cookie, Request, Response
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime, timedelta, timezone
import secrets
from app.core.config import settings 
import os

from app.db.database import get_db
from app.schemas.user import Token, UserCreate, RefreshTokenRequest
from app.services.auth_service import (
    get_user_by_email,
    create_user,
    create_user_session,
    get_user_by_session_token,
    get_user_session_by_token,
    get_session_by_id,
    get_user_sessions,
    deactivate_session,
)
from app.core.security import (
    verify_password,
    create_access_token,
    create_refresh_token,
    verify_token,
    get_current_active_user
)

router = APIRouter(prefix="/auth", tags=["auth"])



class UserResponse(BaseModel):
    id: int # ID는 Integer입니다.
    email: str
    name: str # username 대신 name 사용
    class Config:
        from_attributes = True


class SessionResponse(BaseModel):
    session_id: int
    created_at: datetime
    expires_at: Optional[datetime]
    is_active: bool
    ip_address: Optional[str]
    user_agent: Optional[str]

    class Config:
        from_attributes = True

    
def validate_access_and_session(request: Request, db: Session):
    access_token = request.cookies.get("access_token")
    session_token = request.cookies.get("session_token")
    if not access_token or not session_token:
        return None

    try:
        payload = jwt.decode(access_token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        if payload.get("type") != "access" or payload.get("sub") is None:
            return None
    except JWTError:
        return None

    user = get_user_by_session_token(db, session_token)
    if not user or str(user.user_id) != str(payload.get("sub")):
        return None

    return user


@router.get("/me")
def get_me(request: Request, response: Response, db: Session = Depends(get_db)):
    print("get_me 호출됨")
    user = validate_access_and_session(request, db)
    if user:
        return {"authenticated": True, "id": user.user_id, "email": user.user_email, "name": user.user_name}

    session_token = request.cookies.get("session_token")
    refresh_token_cookie = request.cookies.get("refresh_token")
    if not session_token or not refresh_token_cookie:
        return {"authenticated": False}

    user = get_user_by_session_token(db, session_token)
    if not user:
        response.delete_cookie(key="access_token", httponly=True, samesite="lax")
        response.delete_cookie(key="refresh_token", httponly=True, samesite="lax")
        response.delete_cookie(key="session_token", httponly=True, samesite="lax")
        return {"authenticated": False}

    try:
        payload = verify_token(refresh_token_cookie, token_type="refresh")
        if str(payload.get("sub")) != str(user.user_id):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="토큰 정보가 일치하지 않습니다")

        new_access_token = create_access_token(data={"sub": str(user.user_id)})
        response.set_cookie(
            key="access_token",
            value=new_access_token,
            httponly=True,
            secure=False,
            samesite="lax",
            max_age=60 * 30,
        )
        return {"authenticated": True, "id": user.user_id, "email": user.user_email, "name": user.user_name}
    except HTTPException:
        response.delete_cookie(key="access_token", httponly=True, samesite="lax")
        response.delete_cookie(key="refresh_token", httponly=True, samesite="lax")
        response.delete_cookie(key="session_token", httponly=True, samesite="lax")
        return {"authenticated": False}


@router.post("/register", response_model=UserResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    if get_user_by_email(db, user.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="이미 존재하는 이메일입니다"
        )
    new_user = create_user(db, user)
    return {
        "id": new_user.user_id,
        "email": new_user.user_email,
        "name": new_user.user_name,
    }

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

@router.post("/login")
def login(
    login_data: LoginRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db)
):
    user = get_user_by_email(db, login_data.email)
    if not user or not verify_password(login_data.password, user.user_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="이메일 또는 비밀번호가 올바르지 않습니다"
        )

    session_token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    create_user_session(
        db,
        user.user_id,
        session_token,
        expires_at,
        request.client.host if request.client else None,
        request.headers.get("user-agent"),
    )

    access_token = create_access_token(data={"sub": str(user.user_id)})
    refresh_token = create_refresh_token(data={"sub": str(user.user_id)})

    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=False,       # 개발환경은 HTTP이므로 False
        samesite="lax",
        max_age=60 * 30,
    )
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=False,       # 개발환경은 HTTP이므로 False
        samesite="lax",
        max_age=60 * 60 * 24 * settings.REFRESH_TOKEN_EXPIRE_DAYS,
    )
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=60 * 60 * 24 * settings.REFRESH_TOKEN_EXPIRE_DAYS,
    )

    return {"message": "로그인 성공"}

@router.post("/refresh", response_model=Token)
def refresh_token(
    request: Request,
    response: Response,
    refresh_data: RefreshTokenRequest,
    db: Session = Depends(get_db),
):
    session_token = request.cookies.get("session_token")
    if not session_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="세션이 없습니다")

    user = get_user_by_session_token(db, session_token)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="세션이 만료되었거나 비활성화되었습니다")

    payload = verify_token(refresh_data.refresh_token, token_type="refresh")
    if str(payload.get("sub")) != str(user.user_id):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="토큰 정보가 일치하지 않습니다")

    access_token = create_access_token(data={"sub": str(user.user_id)})
    refresh_token = create_refresh_token(data={"sub": str(user.user_id)})

    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=60 * settings.ACCESS_TOKEN_EXPIRE_MINUTES,
    )
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=60 * 60 * 24 * settings.REFRESH_TOKEN_EXPIRE_DAYS,
    )

    return Token(access_token=access_token, refresh_token=refresh_token)

# @router.get("/me", response_model=UserResponse)
# def get_me(current_user=Depends(get_current_active_user)):
#     return current_user


@router.get("/sessions", response_model=List[SessionResponse])
def get_sessions(request: Request, db: Session = Depends(get_db)):
    user = validate_access_and_session(request, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="인증이 필요합니다")
    return get_user_sessions(db, user.user_id)


@router.delete("/sessions/{session_id}")
def revoke_session(session_id: int, request: Request, response: Response, db: Session = Depends(get_db)):
    user = validate_access_and_session(request, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="인증이 필요합니다")

    session = get_session_by_id(db, session_id)
    if not session or session.user_id != user.user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="세션을 찾을 수 없습니다")

    deactivate_session(db, session)
    if request.cookies.get("session_token") == session.session_token:
        response.delete_cookie(key="access_token", httponly=True, samesite="lax")
        response.delete_cookie(key="refresh_token", httponly=True, samesite="lax")
        response.delete_cookie(key="session_token", httponly=True, samesite="lax")

    return {"message": "세션이 종료되었습니다"}


@router.post("/logout")
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    session_token = request.cookies.get("session_token")
    if session_token:
        session = get_user_session_by_token(db, session_token)
        if session:
            deactivate_session(db, session)

    response.delete_cookie(key="access_token", httponly=True, samesite="lax")
    response.delete_cookie(key="refresh_token", httponly=True, samesite="lax")
    response.delete_cookie(key="session_token", httponly=True, samesite="lax")
    return {"message": "로그아웃 성공"}


