from fastapi import APIRouter, HTTPException, status, Response
from app.services.job_service import job_service
from app.services.report_service import ReportService
from app.schemas.pipeline import PipelineJobStatus
from app.core.exceptions import ResourceNotFoundError

router = APIRouter()

@router.get("/{job_id}", response_model=PipelineJobStatus, tags=["Jobs"])
def get_job_status(job_id: str) -> PipelineJobStatus:
    """Retrieves current execution status, progress percentage, and results for a job."""
    try:
        return job_service.get_job(job_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

@router.get("/{job_id}/export/{format}", tags=["Jobs"])
def export_job_report(job_id: str, format: str):
    """
    Exports completed pipeline job output to JSON or CSV.
    Supported formats: 'json', 'segments-csv', 'entities-csv'.
    """
    try:
        job = job_service.get_job(job_id)
        if not job.result:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Job has no result yet.")

        if format == "json":
            content = ReportService.export_to_json(job.result)
            return Response(content=content, media_type="application/json", headers={"Content-Disposition": f"attachment; filename={job_id}_report.json"})
        elif format == "segments-csv":
            content = ReportService.export_segments_to_csv(job.result)
            return Response(content=content, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={job_id}_segments.csv"})
        elif format == "entities-csv":
            content = ReportService.export_entities_to_csv(job.result)
            return Response(content=content, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={job_id}_entities.csv"})
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid format. Choose 'json', 'segments-csv', or 'entities-csv'.")
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
