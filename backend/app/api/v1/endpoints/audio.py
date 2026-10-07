from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import FileResponse
from app.services.audio_service import AudioService
from app.storage.file_manager import FileManager
from app.schemas.audio import AudioUploadResponse, AudioWaveformResponse
from app.core.exceptions import AudioProcessingError, AudioFileTooLargeError, UnsupportedAudioFormatError

router = APIRouter()

@router.post("/upload", response_model=AudioUploadResponse, status_code=status.HTTP_201_CREATED, tags=["Audio"])
async def upload_audio(file: UploadFile = File(...)) -> AudioUploadResponse:
    """
    Ingests and validates raw audio (WAV, MP3, M4A, FLAC).
    Resamples to 16kHz mono and computes waveform peaks for immediate UI visualization.
    """
    try:
        return await AudioService.process_upload(file)
    except (UnsupportedAudioFormatError, AudioFileTooLargeError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except AudioProcessingError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected error: {str(e)}")

@router.get("/waveform/{file_id}", response_model=AudioWaveformResponse, tags=["Audio"])
def get_audio_waveform(file_id: str, peaks: int = 150) -> AudioWaveformResponse:
    """Returns normalized amplitude peaks for waveform rendering."""
    try:
        return AudioService.get_waveform(file_id, num_peaks=peaks)
    except AudioProcessingError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

@router.get("/stream/{file_id}", tags=["Audio"])
def stream_audio(file_id: str):
    """Streams audio file directly for frontend audio playback."""
    try:
        file_path = FileManager.get_file_by_id(file_id)
        media_type = "audio/wav" if file_path.suffix.lower() == ".wav" else "audio/mpeg"
        return FileResponse(path=str(file_path), media_type=media_type, filename=file_path.name)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"File not found: {str(e)}")
