"""문서 처리 파이프라인 서비스

파이프라인 흐름:
    업로드 → OCR(원문 추출) → Document DB 저장 → 임베딩(벡터DB) → 요약+분류 → 완료

특징:
- asyncio.create_task() 로 백그라운드 실행 → 화면 이탈 후에도 서버에서 계속 진행
- 각 단계마다 SSE push_event 로 실시간 진행률 전송
- is_cancelled 플래그로 단계 사이 취소 가능
- OCR/요약 결과 빈 값 시 트랜잭션 롤백 + 실패 알림 전송
"""

from __future__ import annotations

import asyncio
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session, sessionmaker

from app.db.database import engine
from app.db.models import Document, Job
from app.services import notification_service

KST = timezone(timedelta(hours=9))
_executor = ThreadPoolExecutor(max_workers=2)

UPLOAD_DIR = "pdf_files"

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "법령·조례": ["법령", "조례", "시행령", "시행규칙", "법률"],
    "가이드라인·지침": ["가이드라인", "지침", "매뉴얼", "안내서", "절차"],
    "공모·사업": ["공모", "사업", "신청", "지원", "공고", "선정"],
    "감사": ["감사", "점검", "처분", "감사원", "조사"],
    "내부 규정": ["내부", "내규", "방침", "사규"],
}


# ── 헬퍼 ────────────────────────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now(KST)


def _filename_from_path(file_path: str) -> str:
    """pipe_{uuid}_{원본파일명} → 원본파일명 추출"""
    base = os.path.basename(file_path)
    parts = base.split("_", 2)
    return parts[2] if len(parts) >= 3 else base


def _auto_classify(text: str) -> str:
    """원문 앞부분 키워드 매칭으로 카테고리 자동 분류"""
    preview = text[:3000]
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in preview for kw in keywords):
            return category
    return "기타"


def _push_stage(user_id: int, job_id: int, stage: str) -> None:
    """SSE를 통해 클라이언트에 현재 단계 전송"""
    notification_service.push_event(user_id, {
        "type": "pipeline_progress",
        "job_id": job_id,
        "stage": stage,
    })


def _update_job(db: Session, job: Job, stage: str, **kwargs) -> None:
    """DB 상태 업데이트 + SSE 단계 푸시"""
    job.pipeline_stage = stage
    job.job_status = "running"
    for k, v in kwargs.items():
        setattr(job, k, v)
    db.commit()
    _push_stage(job.user_id, job.job_id, stage)


def _fail_job(db: Session, job: Job, stage: str, message: str) -> None:
    """실패 처리: 현재 트랜잭션 롤백 → failed 상태 저장 → SSE 푸시"""
    try:
        db.rollback()
    except Exception:
        pass
    job.job_status = "failed"
    job.pipeline_stage = "failed"
    job.error_stage = stage
    job.error_message = message
    job.job_finish = _now()
    db.commit()
    _push_stage(job.user_id, job.job_id, "failed")


def _cancel_cleanup(db: Session, job: Job) -> None:
    """취소 처리: 상태 저장 → SSE 푸시 → 임시 파일 삭제"""
    job.job_status = "cancelled"
    job.pipeline_stage = "cancelled"
    job.job_finish = _now()
    db.commit()
    _push_stage(job.user_id, job.job_id, "cancelled")
    _cleanup_file(job.file_path)


def _cleanup_file(file_path: str | None) -> None:
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass


def _notify_failure(db: Session, user_id: int, job_id: int, filename: str, stage: str) -> None:
    try:
        notification_service.create_notification(
            db=db,
            user_id=user_id,
            message=f"'{filename}' 문서 처리 실패 — {stage} 단계에서 오류가 발생했습니다.",
            domain_type="SUMMARY",
            resource_id=job_id,
        )
    except Exception:
        pass


# ── 공개 API ─────────────────────────────────────────────────────────────────


def create_pipeline_job(db: Session, user_id: int, file_path: str) -> Job:
    """임시 파일 저장 완료 후 Job 레코드 생성"""
    job = Job(
        user_id=user_id,
        job_type="document_pipeline",
        job_status="pending",
        pipeline_stage="uploaded",
        is_cancelled=False,
        file_path=file_path,
        job_start=_now(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def cancel_job(db: Session, job_id: int, user_id: int) -> Job | None:
    """취소 플래그 설정 — 파이프라인이 다음 단계 시작 전에 확인하여 중단"""
    job = db.query(Job).filter(
        Job.job_id == job_id,
        Job.user_id == user_id,
        Job.job_type == "document_pipeline",
    ).first()
    if not job:
        return None
    if job.job_status in ("completed", "failed", "cancelled"):
        return job
    job.is_cancelled = True
    db.commit()
    return job


def get_job(db: Session, job_id: int, user_id: int) -> Job | None:
    return db.query(Job).filter(
        Job.job_id == job_id,
        Job.user_id == user_id,
        Job.job_type == "document_pipeline",
    ).first()


def list_jobs(db: Session, user_id: int, skip: int = 0, limit: int = 20) -> list[Job]:
    return (
        db.query(Job)
        .filter(Job.user_id == user_id, Job.job_type == "document_pipeline")
        .order_by(Job.job_start.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


# ── 파이프라인 실행 ───────────────────────────────────────────────────────────


async def run_pipeline(job_id: int, user_id: int) -> None:
    """
    asyncio.create_task()로 실행되는 백그라운드 파이프라인.
    화면을 이탈해도 서버에서 계속 실행됩니다.
    """
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    file_path: str | None = None

    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job:
            return

        file_path = job.file_path
        filename = _filename_from_path(file_path)
        loop = asyncio.get_event_loop()

        # ── 1. OCR: 원문 추출 (0% → 40%) ─────────────────────
        if job.is_cancelled:
            _cancel_cleanup(db, job)
            return

        _update_job(db, job, "ocr")

        try:
            from app.ocr.extractor import process_document
            raw_text: str = await loop.run_in_executor(
                _executor, lambda: process_document(file_path)
            )
        except Exception as e:
            _fail_job(db, job, "ocr", f"OCR 처리 중 오류: {e}")
            _notify_failure(db, user_id, job_id, filename, "OCR")
            return

        if not raw_text or len(raw_text.strip()) < 20:
            _fail_job(
                db, job, "ocr",
                "원문 텍스트를 추출할 수 없습니다. 스캔 이미지이거나 암호화된 파일일 수 있습니다.",
            )
            _notify_failure(db, user_id, job_id, filename, "원문 추출")
            return

        _update_job(db, job, "ocr")

        # ── 2. Document 레코드 생성 및 원문 저장 ──────────────
        if job.is_cancelled:
            _cancel_cleanup(db, job)
            return

        try:
            doc = Document(
                file_name=filename,
                file_type=os.path.splitext(filename)[1].lstrip(".").lower(),
                content_full=raw_text,
                user_id=user_id,
                created_at=_now(),
                updated_at=_now(),
            )
            db.add(doc)
            db.flush()        # doc_id 확보
            job.doc_id = doc.doc_id
            db.commit()
        except Exception as e:
            _fail_job(db, job, "ocr_save", f"원문 DB 저장 실패: {e}")
            _notify_failure(db, user_id, job_id, filename, "DB 저장")
            return

        # ── 3. 임베딩: 청킹 → 벡터 DB 저장 (40% → 70%) ───────
        if job.is_cancelled:
            _cancel_cleanup(db, job)
            return

        _update_job(db, job, "embedding")

        try:
            from app.llm.ingest import ingest_markdown
            await loop.run_in_executor(
                _executor,
                lambda: ingest_markdown(
                    raw_text,
                    doc.doc_id,
                    user_id,
                    doc_name=filename,
                    chunking_method="sections",
                    enable_summary=False,
                ),
            )
        except Exception as e:
            # 임베딩 실패는 요약·저장에 영향 없으므로 경고만 남기고 계속 진행
            print(f"[pipeline job={job_id}] 임베딩 실패 (계속 진행): {e}")

        _update_job(db, job, "embedding")

        # ── 4. 카테고리 자동 분류 + LLM 요약 (70% → 95%) ──────
        if job.is_cancelled:
            _cancel_cleanup(db, job)
            return

        _update_job(db, job, "summarizing")
        category = _auto_classify(raw_text)

        try:
            from app.llm.summarizer import summarize_document
            summary_result: dict = await loop.run_in_executor(
                _executor,
                lambda: summarize_document(raw_text, category),
            )
        except Exception as e:
            _fail_job(db, job, "summarizing", f"요약 처리 중 오류: {e}")
            _notify_failure(db, user_id, job_id, filename, "요약")
            return

        if summary_result.get("status") == "error" or not summary_result.get("summary", "").strip():
            _fail_job(
                db, job, "summarizing",
                summary_result.get("message", "요약문이 생성되지 않았습니다."),
            )
            _notify_failure(db, user_id, job_id, filename, "요약")
            return

        _update_job(db, job, "summarizing")

        # ── 5. 최종 저장 (95% → 100%) ──────────────────────────
        try:
            doc.content_sum = summary_result["summary"]
            doc.category = summary_result.get("category", category)
            doc.updated_at = _now()
            job.job_status = "completed"
            job.pipeline_stage = "completed"
            job.job_finish = _now()
            db.commit()
        except Exception as e:
            _fail_job(db, job, "save", f"최종 저장 실패: {e}")
            _notify_failure(db, user_id, job_id, filename, "최종 저장")
            return

        # 벡터 캐시 무효화
        try:
            from app.llm.pipeline import invalidate_cache
            invalidate_cache(user_id)
        except Exception:
            pass

        # 완료: SSE 단계 푸시 + 알림
        _push_stage(user_id, job_id, "completed")
        try:
            notification_service.create_notification(
                db=db,
                user_id=user_id,
                message=f"'{filename}' 문서 요약이 완료되었습니다.",
                domain_type="SUMMARY",
                resource_id=doc.doc_id,
            )
        except Exception:
            pass

    except Exception as e:
        print(f"[pipeline job={job_id}] 예상치 못한 오류: {e}")

    finally:
        db.close()
        _cleanup_file(file_path)
