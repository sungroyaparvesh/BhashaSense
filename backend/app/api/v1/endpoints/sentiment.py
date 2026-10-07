from typing import List
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, status
from app.ml.registry import ml_registry
from app.schemas.sentiment import SegmentSentimentSchema, SentimentSummarySchema

router = APIRouter()

class SentimentAnalyzeRequest(BaseModel):
    text: str

class SentimentAnalyzeResponse(BaseModel):
    segment_sentiments: List[SegmentSentimentSchema]
    summary: SentimentSummarySchema

@router.post("/analyze", response_model=SentimentAnalyzeResponse, tags=["Sentiment"])
def analyze_text_sentiment(payload: SentimentAnalyzeRequest) -> SentimentAnalyzeResponse:
    """Performs direct sentiment and aspect analysis on text using IndicBERT."""
    try:
        segments = [{"id": 1, "text": payload.text, "start_time": 0.0}]
        segment_sentiments, summary = ml_registry.sentiment_engine.analyze_segments(segments)
        return SentimentAnalyzeResponse(
            segment_sentiments=segment_sentiments,
            summary=summary
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Sentiment analysis failed: {str(e)}")
