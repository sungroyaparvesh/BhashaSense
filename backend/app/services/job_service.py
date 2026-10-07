import uuid
from typing import Dict, Optional
from datetime import datetime
from app.schemas.pipeline import JobStatusEnum, PipelineJobStatus, PipelineResponseSchema
from app.core.exceptions import ResourceNotFoundError

class JobService:
    """In-memory job management service for asynchronous audio analysis tracking."""
    _jobs: Dict[str, PipelineJobStatus] = {}

    @classmethod
    def create_job(cls, file_id: str) -> PipelineJobStatus:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        status = PipelineJobStatus(
            job_id=job_id,
            file_id=file_id,
            status=JobStatusEnum.PENDING,
            progress_percentage=0,
            current_step="Queued for analysis",
            updated_at=datetime.utcnow()
        )
        cls._jobs[job_id] = status
        return status

    @classmethod
    def get_job(cls, job_id: str) -> PipelineJobStatus:
        if job_id not in cls._jobs:
            raise ResourceNotFoundError(f"Job with ID '{job_id}' does not exist.")
        return cls._jobs[job_id]

    @classmethod
    def update_job(
        cls,
        job_id: str,
        status: JobStatusEnum,
        progress: int,
        step: str,
        result: Optional[PipelineResponseSchema] = None,
        error_message: Optional[str] = None
    ) -> PipelineJobStatus:
        job = cls.get_job(job_id)
        job.status = status
        job.progress_percentage = progress
        job.current_step = step
        job.updated_at = datetime.utcnow()
        if result:
            job.result = result
        if error_message:
            job.error_message = error_message
        return job

job_service = JobService()
