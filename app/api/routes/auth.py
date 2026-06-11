import secrets
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import (
    create_access_token,
    create_refresh_token,
    verify_password,
    verify_token,
)
from app.db.database import get_db
from app.schemas.user import RefreshTokenRequest, Token, UserCreate
from app.services.auth_service import (
    change_password,
    create_user,
    create_user_session,
    deactivate_session,
    get_session_by_id,
    get_user,
    get_user_by_email,
    get_user_by_session_token,
    get_user_session_by_token,
    get_user_sessions,
    get_users,
    reset_user_password,
)

router = APIRouter(prefix="/auth", tags=["auth"])


def _role_names(user) -> List[str]:
    """user_role 연결을 통해 이 유저가 가진 역할 이름 목록을 반환."""
    return [ur.role.role_name for ur in user.user_roles]


class UserResponse(BaseModel):
    id: int  # ID는 Integer입니다.
    email: str
    name: str  # username 대신 name 사용
    roles: List[str] = []  # user_rank(단일 등급) → roles(다중 역할)

    class Config:
        from_attributes = True


class UserListResponse(BaseModel):
    id: int
    email: str
    name: str
    roles: List[str] = []
    user_login: Optional[datetime] = None
    user_create: Optional[datetime] = None

    class Config:
        from_attributes = True


class SessionResponse(BaseModel):
    session_id: int
    user_id: int
    created_at: datetime
    expires_at: Optional[datetime]
    is_active: bool
    ip_address: Optional[str]
    user_agent: Optional[str]

    class Config:
        from_attributes = True

class ActivityUserDetail(BaseModel):
    name: str
    email: str
    roles: List[str]
    joinDate: Optional[str] = "2025.03.15"

class RagQueryItem(BaseModel):
    id: int
    query: str
    timestamp: str
    tokensUsed: Optional[int] = None

class DraftItem(BaseModel):
    id: int
    title: str
    type: str
    date: str
    status: str
    statusLabel: str

class UserActivityResponse(BaseModel):
    user: ActivityUserDetail
    #ragQueries: List[RagQueryItem]
    #drafts: List[DraftItem]


def validate_access_and_session(request: Request, db: Session):
    access_token = request.cookies.get("access_token")
    session_token = request.cookies.get("session_token")
    if not access_token or not session_token:
        return None

    try:
        payload = verify_token(access_token, token_type="access")
    except HTTPException:
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
        return {
            "authenticated": True,
            "id": user.user_id,
            "email": user.user_email,
            "name": user.user_name,
            "roles": _role_names(user),
        }

    session_token = request.cookies.get("session_token")
    refresh_token_cookie = request.cookies.get("refresh_token")
    if not session_token or not refresh_token_cookie:
        return {"authenticated": False}

    user = get_user_by_session_token(db, session_token)
    if not user:
        response.delete_cookie(key="access_token", httponly=True, secure=True, samesite="none")
        response.delete_cookie(key="refresh_token", httponly=True, secure=True, samesite="none")
        response.delete_cookie(key="session_token", httponly=True, secure=True, samesite="none")
        return {"authenticated": False}

    try:
        payload = verify_token(refresh_token_cookie, token_type="refresh")
        if str(payload.get("sub")) != str(user.user_id):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="토큰 정보가 일치하지 않습니다",
            )

        new_access_token = create_access_token(data={"sub": str(user.user_id)})
        response.set_cookie(
            key="access_token",
            value=new_access_token,
            httponly=True,
            secure=True,
            samesite="none",
            max_age=60 * 30,
        )
        return {
            "authenticated": True,
            "id": user.user_id,
            "email": user.user_email,
            "name": user.user_name,
            "roles": _role_names(user),
        }
    except HTTPException:
        response.delete_cookie(key="access_token", httponly=True, secure=True, samesite="none")
        response.delete_cookie(key="refresh_token", httponly=True, secure=True, samesite="none")
        response.delete_cookie(key="session_token", httponly=True, secure=True, samesite="none")
        return {"authenticated": False}


@router.post("/register", response_model=UserResponse)
def register(user: UserCreate, request: Request, db: Session = Depends(get_db)):
    current = validate_access_and_session(request,db)
    if not current or "관리자" not in _role_names(current):
        raise HTTPException(status_code = 403, detail ="계정 생성 권한이 없습니다.")
    
    if get_user_by_email(db, user.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="이미 존재하는 이메일입니다"
        )
    new_user = create_user(db, user)
    return {
        "id": new_user.user_id,
        "email": new_user.user_email,
        "name": new_user.user_name,
        "roles": _role_names(new_user),  # 갓 가입한 유저는 보통 []
    }


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class ChangePasswordRequest(BaseModel):
    email: str
    userId: int
    current_password: str
    new_password: str
    user_login: Optional[datetime] = None


@router.post("/login")
def login(
    login_data: LoginRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    user = get_user_by_email(db, login_data.email)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="이메일 없다")
    if not verify_password(login_data.password, user.user_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="이메일 또는 비밀번호가 올바르지 않습니다",
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
        secure=True,
        samesite="none",
        max_age=60 * 30,
    )
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=True,
        samesite="none",
        max_age=60 * 60 * 24 * settings.REFRESH_TOKEN_EXPIRE_DAYS,
    )
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        secure=True,
        samesite="none",
        max_age=60 * 60 * 24 * settings.REFRESH_TOKEN_EXPIRE_DAYS,
    )

    return {
        "message": "로그인 성공",
        "id": user.user_id,
        "email": user.user_email,
        "name": user.user_name,
        "roles": _role_names(user),
        "user_login": user.user_login,
    }


@router.get("/users", response_model=List[UserListResponse])
def list_users(request: Request, db: Session = Depends(get_db)):
    current_user = validate_access_and_session(request, db)
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="인증이 필요합니다")

    users = get_users(db)
    return [
        {
            "id": user.user_id,
            "email": user.user_email,
            "name": user.user_name,
            "roles": _role_names(user),
            "user_login": user.user_login,
            "user_create": user.created_at,  # 모델 컬럼명이 created_at으로 변경됨
        }
        for user in users
    ]

@router.get("/users/activity", response_model=UserActivityResponse)
def get_user_activity(
    request: Request, 
    user_id: Optional[int] = None, # 💡 Query Parameter나 Path Parameter로 선택적 수신
    db: Session = Depends(get_db)
):
    # 1. 현재 로그인한 사용자 검증 (이전 세션 검증 함수 활용)
    current_user = validate_access_and_session(request, db)
    if not current_user:
        raise HTTPException(status_code=401, detail="인증되지 않은 사용자입니다.")
        
    current_roles = [ur.role.role_name for ur in current_user.user_roles]
    
    # 2. 로직 분기 처리
    if user_id is not None:
        # 타인의 ID를 조회하려는 경우 -> 오직 '관리자'만 허용
        if "관리자" not in current_roles:
            raise HTTPException(status_code=403, detail="타인의 활동 내역을 조회할 권한이 없습니다.")
        target_user_id = user_id
    else:
        # user_id가 누락된 경우 -> 일반 유저가 '내 정보'를 요청한 것으로 간주
        target_user_id = current_user.user_id

    # 3. target_user_id를 기반으로 DB에서 질의 이력 및 결재 문서 조회 및 가공
    #queries = db.query(RAGQuery).filter(RAGQuery.user_id == target_user_id).all()
    #drafts = db.query(Document).filter(Document.user_id == target_user_id).all()
    user_info = get_user(db, target_user_id)
    user_roles = _role_names(user_info)
    
    return {
        "user": {"name": user_info.user_name, "email": user_info.user_email, "roles": user_roles},
        #"ragQueries": queries,
        #"drafts": drafts
    }

@router.post("/users/{user_id}/reset-password")
def reset_password(user_id: int, request: Request, db: Session = Depends(get_db)):
    current_user = validate_access_and_session(request, db)
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="인증이 필요합니다")

    updated_user = reset_user_password(db, user_id)
    if not updated_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="사용자를 찾을 수 없습니다")

    return {"message": "암호가 초기화되었습니다"}


@router.post("/users/{user_id}/force-logout")
def force_logout_user(user_id: int, request: Request, response: Response, db: Session = Depends(get_db)):
    current_user = validate_access_and_session(request, db)
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="인증이 필요합니다")

    sessions = get_user_sessions(db, user_id)
    for session in sessions:
        deactivate_session(db, session)

    return {"message": "사용자가 로그아웃되었습니다", "logged_out_sessions": len(sessions)}


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
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="세션이 만료되었거나 비활성화되었습니다",
        )

    payload = verify_token(refresh_data.refresh_token, token_type="refresh")
    if str(payload.get("sub")) != str(user.user_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="토큰 정보가 일치하지 않습니다",
        )

    access_token = create_access_token(data={"sub": str(user.user_id)})
    refresh_token = create_refresh_token(data={"sub": str(user.user_id)})

    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=True,
        samesite="none",
        max_age=60 * settings.ACCESS_TOKEN_EXPIRE_MINUTES,
    )
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=True,
        samesite="none",
        max_age=60 * 60 * 24 * settings.REFRESH_TOKEN_EXPIRE_DAYS,
    )

    return Token(access_token=access_token, refresh_token=refresh_token)


@router.get("/sessions", response_model=List[SessionResponse])
def get_sessions(request: Request, db: Session = Depends(get_db)):
    user = validate_access_and_session(request, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="인증이 필요합니다")
    return get_user_sessions(db, user.user_id)


@router.delete("/sessions/{session_id}")
def revoke_session(
    session_id: int, request: Request, response: Response, db: Session = Depends(get_db)
):
    user = validate_access_and_session(request, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="인증이 필요합니다")

    session = get_session_by_id(db, session_id)
    if not session or session.user_id != user.user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="세션을 찾을 수 없습니다")

    deactivate_session(db, session)
    if request.cookies.get("session_token") == session.session_token:
        response.delete_cookie(key="access_token", httponly=True, secure=True, samesite="none")
        response.delete_cookie(key="refresh_token", httponly=True, secure=True, samesite="none")
        response.delete_cookie(key="session_token", httponly=True, secure=True, samesite="none")

    return {"message": "세션이 종료되었습니다"}


@router.post("/change-password")
def change_password_endpoint(
    change_req: ChangePasswordRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    사용자 비밀번호 변경 및 user_login 시간 업데이트

    - 현재 비밀번호 검증 후 새로운 비밀번호로 변경
    - user_login이 null인 경우, 현재 시간으로 설정
    - user_login이 이미 있는 경우, 새로운 로그인 시간으로 업데이트
    """
    user = get_user_by_email(db, change_req.email)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="사용자를 찾을 수 없습니다")

    if user.user_id != change_req.userId:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="사용자 ID가 일치하지 않습니다")

    login_time = change_req.user_login if change_req.user_login else datetime.now(timezone.utc)
    updated_user = change_password(
        db,
        user.user_id,
        change_req.current_password,
        change_req.new_password,
        login_time,
    )

    if not updated_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="현재 비밀번호가 올바르지 않습니다")

    return {
        "message": "비밀번호가 성공적으로 변경되었습니다",
        "id": updated_user.user_id,
        "email": updated_user.user_email,
        "user_login": updated_user.user_login,
    }


@router.post("/logout")
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    session_token = request.cookies.get("session_token")
    if session_token:
        session = get_user_session_by_token(db, session_token)
        if session:
            deactivate_session(db, session)

    response.delete_cookie(key="access_token", httponly=True, secure=True, samesite="none")
    response.delete_cookie(key="refresh_token", httponly=True, secure=True, samesite="none")
    response.delete_cookie(key="session_token", httponly=True, secure=True, samesite="none")
    return {"message": "로그아웃 성공"}