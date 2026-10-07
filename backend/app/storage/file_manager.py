import os
import uuid
from pathlib import Path
from typing import Tuple
import aiofiles
from fastapi import UploadFile
from app.core.config import settings
from app.core.security import sanitize_filename
from app.core.exceptions import AudioProcessingError, AudioFileTooLargeError, UnsupportedAudioFormatError
from app.core.logging import logger

class FileManager:
    """Manages secure audio file intake, validation, disk persistence, and cleanup."""

    @staticmethod
    def get_upload_path() -> Path:
        path = Path(settings.UPLOAD_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    async def save_uploaded_audio(cls, file: UploadFile) -> Tuple[str, Path, str]:
        """
        Validates, sanitizes, and streams audio data to local disk.
        Returns:
            file_id (str): Unique UUID for this audio session
            stored_path (Path): Absolute file path
            original_filename (str): Sanitized original filename
        """
        if not file.filename:
            raise AudioProcessingError("Uploaded file has no filename.")

        sanitized_name = sanitize_filename(file.filename)
        file_ext = Path(sanitized_name).suffix.lstrip(".").lower()

        if file_ext not in settings.ALLOWED_AUDIO_FORMATS:
            raise UnsupportedAudioFormatError(
                f"Audio format '{file_ext}' is not supported. Allowed: {settings.ALLOWED_AUDIO_FORMATS}"
            )

        file_id = f"aud_{uuid.uuid4().hex[:12]}"
        target_filename = f"{file_id}_{sanitized_name}"
        target_path = cls.get_upload_path() / target_filename

        max_bytes = settings.MAX_AUDIO_SIZE_MB * 1024 * 1024
        bytes_read = 0

        try:
            async with aiofiles.open(target_path, "wb") as out_file:
                while chunk := await file.read(1024 * 64):  # 64 KB chunks
                    bytes_read += len(chunk)
                    if bytes_read > max_bytes:
                        # Clean up partial file
                        if target_path.exists():
                            target_path.unlink()
                        raise AudioFileTooLargeError(
                            f"File exceeds maximum allowed size of {settings.MAX_AUDIO_SIZE_MB} MB."
                        )
                    await out_file.write(chunk)

            logger.info(f"Successfully saved uploaded audio: {target_path} ({bytes_read} bytes)")
            return file_id, target_path, sanitized_name

        except Exception as e:
            if target_path.exists():
                try:
                    target_path.unlink()
                except Exception:
                    pass
            if isinstance(e, (AudioFileTooLargeError, UnsupportedAudioFormatError)):
                raise
            logger.error(f"Failed to persist uploaded audio: {e}")
            raise AudioProcessingError(f"Could not save uploaded file: {str(e)}")

    @classmethod
    def get_file_by_id(cls, file_id: str) -> Path:
        """Finds stored audio file matching file_id."""
        upload_dir = cls.get_upload_path()
        for f in upload_dir.glob(f"{file_id}_*"):
            if f.is_file():
                return f
        raise AudioProcessingError(f"Audio file with ID '{file_id}' not found.")
