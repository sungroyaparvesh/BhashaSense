from fastapi import APIRouter
from app.core.config import settings
from app.ml.device import CURRENT_DEVICE
from app.schemas.common import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse, tags=["Health"])
def get_health() -> HealthResponse:
    """Service liveness and readiness check with hardware accelerator telemetry."""
    return HealthResponse(
        status="healthy",
        environment=settings.ENVIRONMENT,
        device=str(CURRENT_DEVICE),
        mock_mode=settings.MOCK_ML_MODELS,
        version=settings.VERSION
    )
