from sqlalchemy import (
    TIMESTAMP,
    Boolean,
    CheckConstraint,
    Column,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.orm import relationship

from app.db.database import Base


# ============================================================
#  사용자 / 세션
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

    sessions = relationship("UserSession", back_populates="user")
    documents = relationship("Document", back_populates="user")
    jobs = relationship("Job", back_populates="user")
    histories = relationship("History", back_populates="user")
    rag_queries = relationship("RagQuery", back_populates="user")
    notifications = relationship("Notification", back_populates="user")
    user_roles = relationship("UserRole", back_populates="user")
    drafts_authored = relationship("Draft", foreign_keys="Draft.author_id", back_populates="author")
    drafts_to_approve = relationship(
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

    user = relationship("User", back_populates="sessions")


# ============================================================
#  문서 / 작업
# ============================================================
class Document(Base):
    __tablename__ = "doc"

    doc_id = Column(Integer, primary_key=True, autoincrement=True)
    file_name = Column(String)
    file_type = Column(String)
    category = Column(String)  # 플랫 카테고리 (varchar, 별도 테이블 없음)
    content_full = Column(Text)  # 원문 (PDF 추출 전체 텍스트)
    content_sum = Column(Text)  # 요약본 (LLM 생성)
    created_at = Column(TIMESTAMP(timezone=True))
    updated_at = Column(TIMESTAMP(timezone=True))
    is_hidden = Column(Boolean, server_default=text("false"))  # 삭제여부
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)

    user = relationship("User", back_populates="documents")
    jobs = relationship("Job", back_populates="document")


class Job(Base):
    __tablename__ = "job"

    job_id = Column(Integer, primary_key=True, autoincrement=True)
    job_start = Column(TIMESTAMP(timezone=False))  # schema.sql: without time zone
    job_finish = Column(TIMESTAMP(timezone=False))
    doc_id = Column(Integer, ForeignKey("doc.doc_id"))
    user_id = Column(Integer, ForeignKey("users.user_id"))
    job_type = Column(String)  # summarize / embed / batch
    job_status = Column(String)  # pending / running / success / failed

    user = relationship("User", back_populates="jobs")
    document = relationship("Document", back_populates="jobs")


# ============================================================
#  변경 감사 로그
# ============================================================
class History(Base):
    __tablename__ = "history"

    history_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    change_table = Column(String)
    change_text = Column(Text)
    change_time = Column(TIMESTAMP(timezone=True))

    user = relationship("User", back_populates="histories")


# ============================================================
#  권한 (멀티롤)
# ============================================================
class Role(Base):
    __tablename__ = "role"

    role_id = Column(Integer, primary_key=True, autoincrement=True)
    role_name = Column(String, nullable=False, unique=True)  # 실무 담당자 / 결재권자 / 관리자
    description = Column(String)

    user_roles = relationship("UserRole", back_populates="role")


class UserRole(Base):
    __tablename__ = "user_role"

    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), primary_key=True)
    role_id = Column(Integer, ForeignKey("role.role_id", ondelete="RESTRICT"), primary_key=True)
    assigned_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="user_roles")
    role = relationship("Role", back_populates="user_roles")


# ============================================================
#  RAG 질의 로그
# ============================================================
class RagQuery(Base):
    __tablename__ = "rag_query"

    query_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    query_text = Column(Text, nullable=False)
    answer_text = Column(Text)  # LLM 생성 답변
    source_count = Column(Integer, server_default=text("0"))  # 참고 문서 개수
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="rag_queries")
    refs = relationship("RagQueryRef", back_populates="query")


class RagQueryRef(Base):
    __tablename__ = "rag_query_ref"

    ref_id = Column(Integer, primary_key=True, autoincrement=True)
    query_id = Column(Integer, ForeignKey("rag_query.query_id", ondelete="CASCADE"), nullable=False)
    doc_id = Column(Integer, ForeignKey("doc.doc_id", ondelete="SET NULL"))
    snippet = Column(Text)

    query = relationship("RagQuery", back_populates="refs")
    document = relationship("Document")


# ============================================================
#  기안 / 결재 (단일 결재 → 한 테이블 병합)
# ============================================================
class Draft(Base):
    __tablename__ = "draft"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'approved', 'rejected')",
            name="chk_draft_status",
        ),
    )

    draft_id = Column(Integer, primary_key=True, autoincrement=True)
    author_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)  # 작성자
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    source_doc_id = Column(Integer, ForeignKey("doc.doc_id", ondelete="SET NULL"))
    status = Column(String, nullable=False, server_default=text("'pending'"))
    approver_id = Column(Integer, ForeignKey("users.user_id"))  # 결재자
    reject_reason = Column(Text)  # 반려 시에만
    decided_at = Column(TIMESTAMP(timezone=True))  # 결재 시각
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    author = relationship("User", foreign_keys=[author_id], back_populates="drafts_authored")
    approver = relationship("User", foreign_keys=[approver_id], back_populates="drafts_to_approve")
    source_doc = relationship("Document", foreign_keys=[source_doc_id])


# ============================================================
#  알림
# ============================================================
class Notification(Base):
    __tablename__ = "notification"

    noti_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(
        Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False
    )  # 수신자
    message = Column(Text, nullable=False)
    link = Column(String)
    is_read = Column(Boolean, server_default=text("false"))
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="notifications")
