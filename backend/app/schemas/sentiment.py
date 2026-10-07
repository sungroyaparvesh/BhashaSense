from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"

class SentimentScoreDistribution(BaseModel):
    positive: float = Field(ge=0.0, le=1.0)
    neutral: float = Field(ge=0.0, le=1.0)
    negative: float = Field(ge=0.0, le=1.0)

class AspectSentimentSchema(BaseModel):
    aspect: str = Field(description="Aspect or topic, e.g., 'customer_service', 'delivery', 'pricing'")
    sentiment: SentimentLabel
    score: float
    mention_span: Optional[str] = None

class SegmentSentimentSchema(BaseModel):
    segment_id: int
    text: str
    sentiment: SentimentLabel
    confidence: float
    distribution: SentimentScoreDistribution
    aspects: List[AspectSentimentSchema] = []

class SentimentTimelinePointSchema(BaseModel):
    time: float
    score: float
    sentiment: str

class SentimentSummarySchema(BaseModel):
    overall_sentiment: SentimentLabel
    confidence: float
    distribution: SentimentScoreDistribution
    sentiment_score_timeline: List[SentimentTimelinePointSchema]
    aspect_breakdown: List[AspectSentimentSchema]
