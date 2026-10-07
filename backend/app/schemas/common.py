from typing import Generic, Optional, TypeVar, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone

DataT = TypeVar("DataT")

class BaseResponse(BaseModel):
    success: bool = True
    message: str = "Operation completed successfully"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class APIResponse(BaseResponse, Generic[DataT]):
    data: Optional[DataT] = None

class HealthResponse(BaseModel):
    status: str
    environment: str
    device: str
    mock_mode: bool
    version: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
