from pathlib import Path
from fastapi import UploadFile
from app.core.logging import logger
from app.storage.file_manager import FileManager
from app.ml.preprocessors.audio_preprocessor import AudioPreprocessor
from app.schemas.audio import AudioMetadataSchema, AudioUploadResponse, AudioWaveformResponse

class AudioService:
    """Service handling audio file upload, inspection, and waveform peak extraction."""

    @classmethod
    async def process_upload(cls, file: UploadFile) -> AudioUploadResponse:
        """Saves, preprocesses, and extracts metadata + peaks for an uploaded audio file."""
        file_id, stored_path, original_filename = await FileManager.save_uploaded_audio(file)

        # Preprocess and inspect with torchaudio/soundfile
        waveform, sample_rate, duration = AudioPreprocessor.load_and_resample(stored_path)
        peaks = AudioPreprocessor.generate_waveform_peaks(waveform, num_peaks=120)

        file_size = stored_path.stat().st_size
        file_format = stored_path.suffix.lstrip(".").lower()

        metadata = AudioMetadataSchema(
            file_id=file_id,
            original_filename=original_filename,
            stored_path=str(stored_path),
            duration_seconds=round(duration, 2),
            sample_rate=sample_rate,
            channels=1,
            format=file_format,
            file_size_bytes=file_size
        )

        logger.info(f"Audio processed successfully: {file_id} (Duration: {metadata.duration_seconds}s)")

        return AudioUploadResponse(
            file_id=file_id,
            filename=original_filename,
            metadata=metadata,
            waveform_peaks=peaks
        )

    @classmethod
    def get_waveform(cls, file_id: str, num_peaks: int = 150) -> AudioWaveformResponse:
        """Retrieves or recalculates waveform peaks for a specific audio file."""
        file_path = FileManager.get_file_by_id(file_id)
        waveform, _, _ = AudioPreprocessor.load_and_resample(file_path)
        peaks = AudioPreprocessor.generate_waveform_peaks(waveform, num_peaks=num_peaks)

        return AudioWaveformResponse(
            file_id=file_id,
            peaks=peaks,
            sample_count=len(peaks)
        )
