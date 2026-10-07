from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class EntityCategory(str, Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    PRODUCT = "PRODUCT"
    MONEY = "MONEY"
    DATE_TIME = "DATE_TIME"
    ISSUE_TYPE = "ISSUE_TYPE"
    OTHER = "OTHER"

class ExtractedEntitySchema(BaseModel):
    id: str
    text: str
    category: EntityCategory
    start_char: int
    end_char: int
    confidence: float
    segment_id: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    context_sentence: Optional[str] = None

class EntityGroupSummary(BaseModel):
    category: EntityCategory
    count: int
    items: List[str]

class EntityExtractionResponseSchema(BaseModel):
    entities: List[ExtractedEntitySchema]
    total_count: int
    categories_summary: List[EntityGroupSummary]
