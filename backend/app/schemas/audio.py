from typing import List, Optional
from pydantic import BaseModel, Field

class AudioMetadataSchema(BaseModel):
    file_id: str
    original_filename: str
    stored_path: str
    duration_seconds: float
    sample_rate: int
    channels: int
    format: str
    file_size_bytes: int

class AudioUploadResponse(BaseModel):
    file_id: str
    filename: str
    metadata: AudioMetadataSchema
    waveform_peaks: Optional[List[float]] = Field(
        default=None,
        description="Downsampled waveform peaks (100-200 values) for immediate UI rendering"
    )

class AudioWaveformResponse(BaseModel):
    file_id: str
    peaks: List[float]
    sample_count: int
