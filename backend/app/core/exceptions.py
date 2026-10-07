from typing import Any, Dict, Optional
from fastapi import HTTPException, status

class BhashaSenseException(Exception):
    """Base domain exception for BhashaSense."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

class AudioProcessingError(BhashaSenseException):
    """Raised when audio conversion, decoding, or resampling fails."""
    pass

class ModelInferenceError(BhashaSenseException):
    """Raised when machine learning model forward pass fails."""
    pass

class UnsupportedAudioFormatError(BhashaSenseException):
    """Raised when uploaded audio format is not in whitelist."""
    pass

class AudioFileTooLargeError(BhashaSenseException):
    """Raised when file exceeds allowed size."""
    pass

class ResourceNotFoundError(BhashaSenseException):
    """Raised when requested job, audio or artifact does not exist."""
    pass
