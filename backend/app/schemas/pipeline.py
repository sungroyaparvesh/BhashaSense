from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from app.schemas.audio import AudioMetadataSchema
from app.schemas.asr import TranscriptSegmentSchema
from app.schemas.sentiment import SegmentSentimentSchema, SentimentSummarySchema
from app.schemas.entity import ExtractedEntitySchema, EntityGroupSummary

class JobStatusEnum(str, Enum):
    PENDING = "pending"
    PROCESSING_AUDIO = "processing_audio"
    TRANSCRIBING = "transcribing"
    ANALYZING_SENTIMENT = "analyzing_sentiment"
    EXTRACTING_ENTITIES = "extracting_entities"
    COMPLETED = "completed"
    FAILED = "failed"

class PipelineOptions(BaseModel):
    detect_aspects: bool = True
    extract_entities: bool = True
    asr_model: Optional[str] = "indicconformer"
    sentiment_model: Optional[str] = "indicbert"

class PipelineRequest(BaseModel):
    file_id: str
    options: PipelineOptions = Field(default_factory=PipelineOptions)

class PipelineMetrics(BaseModel):
    audio_duration_seconds: float
    total_processing_time_ms: float
    asr_time_ms: float
    sentiment_time_ms: float
    ner_time_ms: float
    device_used: str

class PipelineResponseSchema(BaseModel):
    job_id: str
    status: JobStatusEnum
    audio_metadata: AudioMetadataSchema
    full_transcript: str
    detected_language: str
    segments: List[TranscriptSegmentSchema]
    sentiment_summary: SentimentSummarySchema
    segment_sentiments: List[SegmentSentimentSchema]
    entities: List[ExtractedEntitySchema]
    entity_summary: List[EntityGroupSummary]
    metrics: PipelineMetrics
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class PipelineJobStatus(BaseModel):
    job_id: str
    file_id: str
    status: JobStatusEnum
    progress_percentage: int
    current_step: str
    error_message: Optional[str] = None
    result: Optional[PipelineResponseSchema] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
