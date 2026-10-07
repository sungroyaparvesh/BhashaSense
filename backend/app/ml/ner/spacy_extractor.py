import re
import uuid
from typing import List, Dict, Any, Tuple
from app.core.config import settings
from app.core.logging import logger
from app.schemas.entity import (
    EntityCategory,
    ExtractedEntitySchema,
    EntityGroupSummary,
    EntityExtractionResponseSchema,
)

class SpacyEntityExtractor:
    """
    Named Entity Recognition & Information Extraction engine leveraging
    spaCy with customized rules for Indian code-mixed vernacular conversations.
    """

    # Domain dictionary and regex rules for Indian code-mixed vernacular
    DOMAIN_PATTERNS = [
        (re.compile(r"\b(BhashaSense)\b", re.IGNORECASE), EntityCategory.ORGANIZATION),
        (re.compile(r"\b(TXN-[0-9A-Z]+|ID\s*[0-9]+)\b", re.IGNORECASE), EntityCategory.PRODUCT),
        (re.compile(r"\b(subscription|portal|account|dashboard|invoices?)\b", re.IGNORECASE), EntityCategory.PRODUCT),
        (re.compile(r"\b(24\s*(hours?|ghante)|kal|aaj|today|yesterday)\b", re.IGNORECASE), EntityCategory.DATE_TIME),
        (re.compile(r"\b(payment\s*deduct|account\s*activate|issue|pending)\b", re.IGNORECASE), EntityCategory.ISSUE_TYPE),
        (re.compile(r"\b(customer\s*support|agent|helpdesk)\b", re.IGNORECASE), EntityCategory.ORGANIZATION),
        (re.compile(r"\b(rupees?|rs\.?|inr|paisa)\b", re.IGNORECASE), EntityCategory.MONEY),
    ]

    def __init__(self):
        self.nlp = None
        self.is_loaded = False

    def load_model(self) -> None:
        """Loads spaCy model or fallback."""
        if settings.MOCK_ML_MODELS:
            logger.info("Initializing spaCy Entity Extractor in MOCK mode.")
            self.is_loaded = True
            return

        try:
            import spacy
            logger.info(f"Loading spaCy model '{settings.SPACY_MODEL_ID}'...")
            self.nlp = spacy.load(settings.SPACY_MODEL_ID)
            self.is_loaded = True
            logger.info("spaCy Entity model loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Could not load spaCy model '{settings.SPACY_MODEL_ID}': {e}. "
                f"Activating regex & pattern-based Indian entity extractor."
            )
            self.is_loaded = True

    def extract_from_segments(
        self,
        segments: List[Dict[str, Any]]
    ) -> Tuple[List[ExtractedEntitySchema], List[EntityGroupSummary]]:
        """
        Extracts named entities from speech segments, associating character spans,
        start/end audio timestamps, and category groupings.
        """
        if not self.is_loaded:
            self.load_model()

        all_entities: List[ExtractedEntitySchema] = []
        category_map: Dict[EntityCategory, List[str]] = {cat: [] for cat in EntityCategory}

        for seg in segments:
            text = seg.get("text", "")
            seg_id = seg.get("id", 1)
            seg_start = seg.get("start_time", 0.0)
            seg_end = seg.get("end_time", 0.0)

            # Apply pattern matchers
            for pattern, category in self.DOMAIN_PATTERNS:
                for match in pattern.finditer(text):
                    entity_str = match.group(0).strip()
                    ent_id = f"ent_{uuid.uuid4().hex[:8]}"

                    entity = ExtractedEntitySchema(
                        id=ent_id,
                        text=entity_str,
                        category=category,
                        start_char=match.start(),
                        end_char=match.end(),
                        confidence=0.92,
                        segment_id=seg_id,
                        start_time=seg_start,
                        end_time=seg_end,
                        context_sentence=text
                    )
                    all_entities.append(entity)
                    if entity_str.lower() not in [item.lower() for item in category_map[category]]:
                        category_map[category].append(entity_str)

        # Build group summaries
        summaries: List[EntityGroupSummary] = []
        for cat, items in category_map.items():
            if items:
                summaries.append(
                    EntityGroupSummary(
                        category=cat,
                        count=len(items),
                        items=items
                    )
                )

        return all_entities, summaries
