from __future__ import annotations

import asyncio
import os
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.database import get_db
from app.services import document_pipeline_service as svc

router = APIRouter(prefix="/documents/pipeline", tags=["document-pipeline"])

UPLOAD_DIR = "pdf_files"


# ── 응답 스키마 ──────────────────────────────────────────────────────────────


class JobOut(BaseModel):
    job_id: int
    job_status: str
    pipeline_stage: str | None
    is_cancelled: bool | None
    file_path: str | None
    error_stage: str | None
    error_message: str | None
    doc_id: int | None

    class Config:
        from_attributes = True


# ── 엔드포인트 ────────────────────────────────────────────────────────────────


@router.post("/upload", response_model=JobOut, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    문서를 업로드하고 파이프라인 Job을 시작합니다.

    - 파일을 임시 저장 후 Job 레코드를 생성하고 즉시 202를 반환합니다.
    - 파이프라인(OCR → 임베딩 → 요약)은 서버 백그라운드에서 계속 실행됩니다.
    - 진행률은 SSE /notifications/subscribe 에서 `type: pipeline_progress` 이벤트로 수신합니다.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일명이 없습니다.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="빈 파일은 업로드할 수 없습니다.")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    safe_name = f"pipe_{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_name)

    with open(file_path, "wb") as f:
        f.write(content)

    job = svc.create_pipeline_job(
        db=db,
        user_id=current_user.user_id,
        file_path=file_path,
    )

    # 백그라운드 실행: 화면 이탈 후에도 서버에서 계속 진행
    asyncio.create_task(svc.run_pipeline(job.job_id, current_user.user_id))

    return job


@router.get("/jobs", response_model=list[JobOut])
def list_jobs(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """현재 사용자의 파이프라인 Job 목록을 반환합니다."""
    return svc.list_jobs(db, user_id=current_user.user_id, skip=skip, limit=limit)


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(
    job_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """특정 Job의 상태·진행률을 반환합니다."""
    job = svc.get_job(db, job_id=job_id, user_id=current_user.user_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job을 찾을 수 없습니다.")
    return job


@router.post("/jobs/{job_id}/cancel", response_model=JobOut)
def cancel_job(
    job_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    진행 중인 파이프라인 Job에 취소 플래그를 설정합니다.

    파이프라인은 각 단계 시작 전에 플래그를 확인하여 중단합니다.
    이미 완료/실패/취소된 Job에는 효과가 없습니다.
    """
    job = svc.cancel_job(db, job_id=job_id, user_id=current_user.user_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job을 찾을 수 없습니다.")
    return job
