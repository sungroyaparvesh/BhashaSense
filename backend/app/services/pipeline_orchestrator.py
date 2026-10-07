import time
import uuid
from pathlib import Path
from typing import Optional
from app.core.logging import logger
from app.core.exceptions import AudioProcessingError
from app.storage.file_manager import FileManager
from app.ml.preprocessors.audio_preprocessor import AudioPreprocessor
from app.ml.registry import ml_registry
from app.ml.device import CURRENT_DEVICE
from app.services.job_service import job_service
from app.schemas.pipeline import (
    PipelineRequest,
    PipelineResponseSchema,
    PipelineMetrics,
    JobStatusEnum,
)
from app.schemas.audio import AudioMetadataSchema

class PipelineOrchestrator:
    """
    Coordinates the complete multi-modal inference pipeline:
    Audio Ingestion -> IndicConformer ASR -> IndicBERT Sentiment -> spaCy NER -> Synthesis.
    """

    @classmethod
    async def run_pipeline(
        cls,
        request: PipelineRequest,
        job_id: Optional[str] = None
    ) -> PipelineResponseSchema:
        start_total = time.time()
        file_id = request.file_id

        if job_id:
            job_service.update_job(job_id, JobStatusEnum.PROCESSING_AUDIO, 15, "Resampling audio to 16kHz mono")

        # 1. Fetch and Preprocess Audio
        file_path = FileManager.get_file_by_id(file_id)
        waveform, sample_rate, duration = AudioPreprocessor.load_and_resample(file_path)

        metadata = AudioMetadataSchema(
            file_id=file_id,
            original_filename=file_path.name,
            stored_path=str(file_path),
            duration_seconds=round(duration, 2),
            sample_rate=sample_rate,
            channels=1,
            format=file_path.suffix.lstrip(".").lower(),
            file_size_bytes=file_path.stat().st_size
        )

        # 2. Speech-to-Text (ASR)
        if job_id:
            job_service.update_job(job_id, JobStatusEnum.TRANSCRIBING, 40, "Transcribing code-mixed speech with IndicConformer")

        asr_start = time.time()
        asr_response = ml_registry.asr_engine.transcribe(waveform, sample_rate, duration)
        asr_time = (time.time() - asr_start) * 1000

        # Convert segment models to dicts for downstream consumers
        raw_segments = [s.model_dump() for s in asr_response.segments]

        # 3. Sentiment Analysis (IndicBERT)
        if job_id:
            job_service.update_job(job_id, JobStatusEnum.ANALYZING_SENTIMENT, 65, "Classifying sentiment & aspects with IndicBERT")

        sentiment_start = time.time()
        segment_sentiments, sentiment_summary = ml_registry.sentiment_engine.analyze_segments(raw_segments)
        sentiment_time = (time.time() - sentiment_start) * 1000

        # 4. Entity Extraction (spaCy)
        if job_id:
            job_service.update_job(job_id, JobStatusEnum.EXTRACTING_ENTITIES, 85, "Extracting domain entities with spaCy")

        ner_start = time.time()
        entities, entity_summaries = ml_registry.ner_engine.extract_from_segments(raw_segments)
        ner_time = (time.time() - ner_start) * 1000

        # 5. Compile Performance Metrics
        total_time = (time.time() - start_total) * 1000
        metrics = PipelineMetrics(
            audio_duration_seconds=round(duration, 2),
            total_processing_time_ms=round(total_time, 2),
            asr_time_ms=round(asr_time, 2),
            sentiment_time_ms=round(sentiment_time, 2),
            ner_time_ms=round(ner_time, 2),
            device_used=str(CURRENT_DEVICE)
        )

        effective_job_id = job_id or f"job_{uuid.uuid4().hex[:12]}"
        response = PipelineResponseSchema(
            job_id=effective_job_id,
            status=JobStatusEnum.COMPLETED,
            audio_metadata=metadata,
            full_transcript=asr_response.full_transcript,
            detected_language=asr_response.detected_language,
            segments=asr_response.segments,
            sentiment_summary=sentiment_summary,
            segment_sentiments=segment_sentiments,
            entities=entities,
            entity_summary=entity_summaries,
            metrics=metrics
        )

        if job_id:
            job_service.update_job(job_id, JobStatusEnum.COMPLETED, 100, "Analysis complete", result=response)

        logger.info(f"Pipeline completed for {file_id} in {round(total_time, 2)}ms")
        return response
