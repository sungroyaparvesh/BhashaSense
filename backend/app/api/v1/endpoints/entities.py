from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, status
from app.ml.registry import ml_registry
from app.schemas.entity import EntityExtractionResponseSchema

router = APIRouter()

class EntityExtractRequest(BaseModel):
    text: str

@router.post("/extract", response_model=EntityExtractionResponseSchema, tags=["Entities"])
def extract_text_entities(payload: EntityExtractRequest) -> EntityExtractionResponseSchema:
    """Performs direct named entity extraction on code-mixed text using spaCy."""
    try:
        segments = [{"id": 1, "text": payload.text, "start_time": 0.0}]
        entities, summaries = ml_registry.ner_engine.extract_from_segments(segments)
        return EntityExtractionResponseSchema(
            entities=entities,
            total_count=len(entities),
            categories_summary=summaries
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Entity extraction failed: {str(e)}")
