from datetime import datetime, timedelta, timezone

from sqlalchemy import TIMESTAMP, Boolean, Column, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.db.database import Base

# 한국 시간 (KST) 설정
KST = timezone(timedelta(hours=9))


def get_now():
    # 한국 시간 (KST) 기준 현재 시간 반환
    return datetime.now(KST)


class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    user_email = Column(String, nullable=False, unique=True)
    user_password = Column(String, nullable=False)
    user_rank = Column(Integer)
    user_name = Column(String)
    user_create = Column(TIMESTAMP(timezone=True))
    user_update = Column(TIMESTAMP(timezone=True))
    user_login = Column(TIMESTAMP(timezone=True))

    # 양방향 관계 정의 (상대방 클래스의 변수명과 완벽 매칭)
    sessions = relationship("UserSession", back_populates="user")
    histories = relationship("History", back_populates="user")
    documents = relationship("Document", back_populates="user")
    jobs = relationship("Job", back_populates="user")


class UserSession(Base):
    __tablename__ = "user_sessions"

    session_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    session_token = Column(String, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True))
    expires_at = Column(TIMESTAMP(timezone=True))
    is_active = Column(Boolean, default=True)
    ip_address = Column(String)
    user_agent = Column(Text)

    user = relationship("User", back_populates="sessions")


class History(Base):
    __tablename__ = "history"

    history_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    change_table = Column(String)
    change_text = Column(Text)
    change_time = Column(TIMESTAMP(timezone=True))

    user = relationship("User", back_populates="histories")


class Category(Base):
    __tablename__ = "category"

    cat_id = Column(Integer, primary_key=True, autoincrement=True)
    main = Column(String)
    sub = Column(String)
    extension = Column(String)

    documents = relationship("Document", back_populates="category")


class Document(Base):
    __tablename__ = "documents"

    doc_id = Column(Integer, primary_key=True, autoincrement=True)
    file_name = Column(String)
    file_type = Column(String)
    file_size = Column(String)
    cat_id = Column(Integer, ForeignKey("category.cat_id"))
    content_full = Column(Text)
    content_sum = Column(Text)
    time_saved = Column(TIMESTAMP(timezone=True))
    time_updated = Column(TIMESTAMP(timezone=True))
    is_hidden = Column(Boolean, default=False)
    user_id = Column(Integer, ForeignKey("users.user_id"))

    user = relationship("User", back_populates="documents")
    category = relationship("Category", back_populates="documents")
    jobs = relationship("Job", back_populates="document")


class Job(Base):
    __tablename__ = "jobs"

    job_id = Column(Integer, primary_key=True, autoincrement=True)
    # 🛠️ ERD 구조 반영: timestamp with time zone으로 수정
    job_start = Column(TIMESTAMP(timezone=True))
    job_finish = Column(TIMESTAMP(timezone=True))

    doc_id = Column(Integer, ForeignKey("documents.doc_id"))
    user_id = Column(Integer, ForeignKey("users.user_id"))
    job_type = Column(String)
    job_status = Column(String)

    user = relationship("User", back_populates="jobs")
    document = relationship("Document", back_populates="jobs")


"""
파일원본

from sqlalchemy import Column, Integer, String, Boolean, DateTime, TIMESTAMP, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone, timedelta
from app.db.database import Base

# 한국 시간 (KST) 설정
KST = timezone(timedelta(hours=9))

def get_now():
    # 한국 시간 (KST) 기준 현재 시간 반환
    return datetime.now(KST)

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=get_now, nullable=False)
    updated_at = Column(DateTime, default=get_now, onupdate=get_now, nullable=False)

    documents = relationship('Document', back_populates='owner', lazy=True)

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    category = Column(String(50), default="기타")
    title = Column(String(500), nullable=False)
    content = Column(LONGTEXT, nullable= False) 
    summary = Column(LONGTEXT, nullable=True)
    is_deleted = Column(Boolean, default=False)
    created_at = Column(DateTime, default=get_now)
    updated_at = Column(DateTime, default=get_now, onupdate=get_now)

    owner = relationship("User", back_populates="documents")
"""
