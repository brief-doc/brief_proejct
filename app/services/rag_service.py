from sqlalchemy.orm import Session

from app.db.models import RagQuery, RagQueryRef


def log_query(
    db: Session,
    user_id: int,
    question: str,
    answer: str,
    references: list[dict],
) -> RagQuery:
    """RAG 질의 결과를 rag_query / rag_query_ref 테이블에 저장합니다."""
    query = RagQuery(
        user_id=user_id,
        query_text=question,
        answer_text=answer,
        source_count=len(references),
    )
    db.add(query)
    db.flush()  # query_id 획득

    for ref in references:
        db.add(
            RagQueryRef(
                query_id=query.query_id,
                doc_id=ref.get("doc_id"),
                snippet=(ref.get("snippet") or "")[:500],
            )
        )
    db.commit()
    db.refresh(query)
    return query


def get_query_history(
    db: Session,
    user_id: int,
    skip: int = 0,
    limit: int = 20,
) -> list[RagQuery]:
    """사용자의 최근 RAG 질의 이력을 반환합니다."""
    return db.query(RagQuery).filter(RagQuery.user_id == user_id).order_by(RagQuery.created_at.desc()).offset(skip).limit(limit).all()
