from sqlalchemy.orm import Session

from app.db.models import Document, User


def get_doc_id(db: Session, user_id: int):
    return db.query(Document).filter(User.user_id == user_id).all()
