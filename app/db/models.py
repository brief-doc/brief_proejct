from datetime import datetime, timedelta, timezone

from sqlalchemy import (
    CheckConstraint,
    Boolean,
    Column,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TIMESTAMP,
    text,
)
from sqlalchemy.orm import relationship

from app.db.database import Base

# 한국 시간 (KST) 설정
KST = timezone(timedelta(hours=9))


def get_now():
    # 한국 시간 (KST) 기준 현재 시간 반환
    return datetime.now(KST)


# ============================================================
#  [A] 기존 테이블
# ============================================================


class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    user_email = Column(String, nullable=False, unique=True)
    user_password = Column(String, nullable=False)
    user_name = Column(String)
    created_at = Column(TIMESTAMP(timezone=True))
    updated_at = Column(TIMESTAMP(timezone=True))
    user_login = Column(TIMESTAMP(timezone=True))  # 최종 로그인 시간

    # 관계
    sessions = relationship("UserSession", back_populates="user")
    histories = relationship("History", back_populates="user")
    documents = relationship("Document", back_populates="user")
    jobs = relationship("Job", back_populates="user")
    user_roles = relationship("UserRole", back_populates="user")
    rag_queries = relationship("RagQuery", back_populates="user")
    notifications = relationship("Notification", back_populates="user")
    drafts_authored = relationship(
        "Draft", foreign_keys="Draft.author_id", back_populates="author"
    )
    drafts_approved = relationship(
        "Draft", foreign_keys="Draft.approver_id", back_populates="approver"
    )


class UserSession(Base):
    __tablename__ = "user_sessions"

    session_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    session_token = Column(String, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True))
    expires_at = Column(TIMESTAMP(timezone=True), nullable=False)
    is_active = Column(Boolean)
    ip_address = Column(String)
    user_agent = Column(Text) 

    user = relationship("User", back_populates="sessions")


class Document(Base):
    __tablename__ = "doc"  # 스키마상 테이블명은 'doc'

    doc_id = Column(Integer, primary_key=True, autoincrement=True)
    file_name = Column(String)
    file_type = Column(String)
    category = Column(String, index=True)  # 카테고리 (별도 테이블 없이 문자열로 직접 보관)
    content_full = Column(Text)  # 원문 (PDF 추출 전체 텍스트)
    content_sum = Column(Text)  # 요약본 (LLM 생성)
    created_at = Column(TIMESTAMP(timezone=True))
    updated_at = Column(TIMESTAMP(timezone=True))
    is_deleted = Column(Boolean, server_default=text("false"))  # 삭제여부
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False, index=True)

    user = relationship("User", back_populates="documents")
    jobs = relationship("Job", back_populates="document")


class Job(Base):
    __tablename__ = "job"  # 스키마상 테이블명은 'job'

    job_id = Column(Integer, primary_key=True, autoincrement=True)
    # 스키마: timestamp without time zone
    job_start = Column(TIMESTAMP(timezone=False))
    job_finish = Column(TIMESTAMP(timezone=False))
    doc_id = Column(Integer, ForeignKey("doc.doc_id"), index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    job_type = Column(String)  # summarize / embed / batch
    job_status = Column(String)  # pending / running / success / failed

    user = relationship("User", back_populates="jobs")
    document = relationship("Document", back_populates="jobs")


class History(Base):
    __tablename__ = "history"

    history_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    change_table = Column(String)
    change_text = Column(Text)
    change_time = Column(TIMESTAMP(timezone=True))

    user = relationship("User", back_populates="histories")


# ============================================================
#  [B] 추가 테이블
# ============================================================


class Role(Base):
    __tablename__ = "role"

    role_id = Column(Integer, primary_key=True, autoincrement=True)
    role_name = Column(String, nullable=False, unique=True)  # 실무 담당자 / 결재권자 / 관리자
    description = Column(String)

    user_roles = relationship("UserRole", back_populates="role")


class UserRole(Base):
    __tablename__ = "user_role"

    user_id = Column(
        Integer,
        ForeignKey("users.user_id", ondelete="CASCADE"),
        primary_key=True,
    )
    role_id = Column(
        Integer,
        ForeignKey("role.role_id", ondelete="RESTRICT"),
        primary_key=True,
    )
    assigned_at = Column(TIMESTAMP(timezone=True), server_default=text("now()"))

    user = relationship("User", back_populates="user_roles")
    role = relationship("Role", back_populates="user_roles")


class RagQuery(Base):
    __tablename__ = "rag_query"

    query_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(
        Integer,
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    query_text = Column(Text, nullable=False)
    answer_text = Column(Text)  # LLM 생성 답변
    source_count = Column(Integer, server_default=text("0"))  # 참고 문서 개수
    created_at = Column(TIMESTAMP(timezone=True), server_default=text("now()"))

    user = relationship("User", back_populates="rag_queries")
    refs = relationship("RagQueryRef", back_populates="query")


class RagQueryRef(Base):
    __tablename__ = "rag_query_ref"

    ref_id = Column(Integer, primary_key=True, autoincrement=True)
    query_id = Column(
        Integer,
        ForeignKey("rag_query.query_id", ondelete="CASCADE"),
        nullable=False,
    )
    doc_id = Column(Integer, ForeignKey("doc.doc_id", ondelete="SET NULL"))
    snippet = Column(Text)

    query = relationship("RagQuery", back_populates="refs")
    document = relationship("Document")


class Draft(Base):
    __tablename__ = "draft"
    __table_args__ = (
        CheckConstraint(
            "status IN ('draft', 'pending', 'approved', 'rejected', 'canceled')",
            name="chk_draft_status",
        ),
    )

    draft_id = Column(Integer, primary_key=True, autoincrement=True)
    author_id = Column(
        Integer, ForeignKey("users.user_id"), nullable=False, index=True
    )  # 작성자(실무 담당자)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)  # 상신 내용
    source_doc_id = Column(
        Integer, ForeignKey("doc.doc_id", ondelete="SET NULL")
    )  # 첨부 근거 문서(요약). 없을 수 있음
    status = Column(String, nullable=False, server_default=text("'draft'"), index=True)

    # ▼ 결재 정보 (병합) — 대기(pending) 중엔 NULL
    approver_id = Column(Integer, ForeignKey("users.user_id"))  # 결재한 사람
    reject_reason = Column(Text)  # 반려 시에만 채움
    decided_at = Column(TIMESTAMP(timezone=True))  # 결재(승인/반려) 시각
    # ▲

    created_at = Column(TIMESTAMP(timezone=True), server_default=text("now()"))
    updated_at = Column(TIMESTAMP(timezone=True), server_default=text("now()"))

    author = relationship(
        "User", foreign_keys=[author_id], back_populates="drafts_authored"
    )
    approver = relationship(
        "User", foreign_keys=[approver_id], back_populates="drafts_approved"
    )
    source_doc = relationship("Document", foreign_keys=[source_doc_id])


class Notification(Base):
    __tablename__ = "notification"
    __table_args__ = (
        Index("idx_noti_user_read", "user_id", "is_read"),
    )

    noti_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(
        Integer,
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
    )  # 수신자
    message = Column(Text, nullable=False)
    link = Column(String)
    is_read = Column(Boolean, server_default=text("false"))
    created_at = Column(TIMESTAMP(timezone=True), server_default=text("now()"))

    user = relationship("User", back_populates="notifications")