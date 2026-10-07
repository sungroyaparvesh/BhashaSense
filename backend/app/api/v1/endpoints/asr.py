from fastapi import APIRouter, HTTPException, status
from app.storage.file_manager import FileManager
from app.ml.preprocessors.audio_preprocessor import AudioPreprocessor
from app.ml.registry import ml_registry
from app.schemas.asr import ASRResponseSchema
from app.core.exceptions import AudioProcessingError

router = APIRouter()

@router.post("/transcribe/{file_id}", response_model=ASRResponseSchema, tags=["ASR"])
def transcribe_audio_file(file_id: str) -> ASRResponseSchema:
    """Performs direct speech-to-text inference with IndicConformer."""
    try:
        file_path = FileManager.get_file_by_id(file_id)
        waveform, sample_rate, duration = AudioPreprocessor.load_and_resample(file_path)
        return ml_registry.asr_engine.transcribe(waveform, sample_rate, duration)
    except AudioProcessingError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"ASR failed: {str(e)}")
