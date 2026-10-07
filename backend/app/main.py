from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import BhashaSenseException
from app.api.middlewares import ProcessTimeMiddleware
from app.api.v1.api_router import api_router
from app.ml.registry import ml_registry

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan lifecycle: handles startup and graceful shutdown."""
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION} [{settings.ENVIRONMENT}]")
    settings.setup_directories()

    # Pre-warm ML registry in background / lazy mode
    try:
        ml_registry.initialize_all()
    except Exception as e:
        logger.warning(f"ML Registry warmup encountered warning: {e}. Lazy loading will be used.")

    yield

    logger.info(f"Shutting down {settings.PROJECT_NAME}...")

def create_application() -> FastAPI:
    """Factory function for FastAPI application."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=(
            "Enterprise End-to-End Code-Mixed Audio Sentiment & Entity Extraction Pipeline. "
            "Integrates torchaudio, IndicConformer ASR, IndicBERT Sentiment, and spaCy NER."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS if isinstance(settings.CORS_ORIGINS, list) else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom Middleware
    app.add_middleware(ProcessTimeMiddleware)

    # Mount API v1 router
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    # Global Domain Exception Handler
    @app.exception_handler(BhashaSenseException)
    async def bhashasense_exception_handler(request: Request, exc: BhashaSenseException):
        logger.warning(f"Domain Exception on {request.url.path}: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "message": exc.message,
                "details": exc.details,
            }
        )

    # Root index endpoint
    @app.get("/", tags=["Root"])
    def root():
        return {
            "name": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "docs": "/docs",
            "health": f"{settings.API_V1_PREFIX}/health"
        }

    return app

app = create_application()
