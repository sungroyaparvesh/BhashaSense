from fastapi import APIRouter
from app.api.v1.endpoints import health, audio, asr, sentiment, entities, pipeline, jobs

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(audio.router, prefix="/audio")
api_router.include_router(asr.router, prefix="/asr")
api_router.include_router(sentiment.router, prefix="/sentiment")
api_router.include_router(entities.router, prefix="/entities")
api_router.include_router(pipeline.router, prefix="/pipeline")
api_router.include_router(jobs.router, prefix="/jobs")
