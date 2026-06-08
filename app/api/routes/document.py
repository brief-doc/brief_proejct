from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.database import get_db
from app.db.models import User
from app.schemas.document import DocResponse
from app.services import doc_service

router = APIRouter(prefix="/docs", tags=["docs"])


@router.get("/", response_model=list[DocResponse])
def list_documents(
    category: str | None = None,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return doc_service.get_docs(
        db,
        user_id=current_user.user_id,
        category=category,
        skip=skip,
        limit=limit,
    )
