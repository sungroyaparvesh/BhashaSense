from typing import List, Optional
from pydantic import BaseModel, Field

class WordAlignmentSchema(BaseModel):
    word: str
    start_time: float
    end_time: float
    confidence: float = 1.0

class TranscriptSegmentSchema(BaseModel):
    id: int
    text: str
    start_time: float
    end_time: float
    confidence: float
    language_code: str = Field(
        default="hi-en",
        description="Detected dialect or code-mixed tag (e.g. hi-en, ta-en, bn-en)"
    )
    words: Optional[List[WordAlignmentSchema]] = None

class ASRResponseSchema(BaseModel):
    full_transcript: str
    detected_language: str
    confidence: float
    segments: List[TranscriptSegmentSchema]
    duration: float
