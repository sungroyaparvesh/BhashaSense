from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from app.schemas.pipeline import PipelineRequest, PipelineResponseSchema, PipelineJobStatus, JobStatusEnum
from app.services.pipeline_orchestrator import PipelineOrchestrator
from app.services.job_service import job_service
from app.core.exceptions import AudioProcessingError
from app.core.logging import logger

router = APIRouter()

@router.post("/run", response_model=PipelineResponseSchema, tags=["Pipeline"])
async def run_pipeline(payload: PipelineRequest) -> PipelineResponseSchema:
    """
    Executes synchronous end-to-end processing:
    Audio Ingestion -> IndicConformer ASR -> IndicBERT Sentiment -> spaCy NER -> Synthesis.
    Recommended for audio recordings under 60 seconds.
    """
    try:
        return await PipelineOrchestrator.run_pipeline(payload)
    except AudioProcessingError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Synchronous pipeline failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Pipeline error: {str(e)}")

@router.post("/run-async", response_model=PipelineJobStatus, status_code=status.HTTP_202_ACCEPTED, tags=["Pipeline"])
async def run_pipeline_async(payload: PipelineRequest, background_tasks: BackgroundTasks) -> PipelineJobStatus:
    """
    Spawns background task for longer audio files.
    Returns a Job ID immediately for status polling or SSE streaming.
    """
    job = job_service.create_job(payload.file_id)

    async def execute_task():
        try:
            await PipelineOrchestrator.run_pipeline(payload, job_id=job.job_id)
        except Exception as e:
            logger.error(f"Async pipeline job {job.job_id} failed: {e}")
            job_service.update_job(
                job.job_id,
                JobStatusEnum.FAILED,
                progress=0,
                step="Execution failed",
                error_message=str(e)
            )

    background_tasks.add_task(execute_task)
    return job
