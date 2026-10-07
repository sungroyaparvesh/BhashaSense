import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.logging import logger

class ProcessTimeMiddleware(BaseHTTPMiddleware):
    """Measures total request processing latency and injects X-Process-Time header."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
        
        # Don't log spammy health checks
        if not request.url.path.endswith("/health"):
            logger.info(f"{request.method} {request.url.path} completed in {process_time:.2f}ms with status {response.status_code}")
            
        return response
