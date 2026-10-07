from app.ml.registry import ml_registry
from app.schemas.entity import EntityCategory

def test_entity_extraction_recognizes_domain_entities():
    segments = [
        {"id": 1, "text": "Maine BhashaSense subscription ke liye kal payment deduct dekha for TXN-9482.", "start_time": 0.0, "end_time": 3.0}
    ]
    entities, summaries = ml_registry.ner_engine.extract_from_segments(segments)
    assert len(entities) > 0

    entity_texts = [e.text.lower() for e in entities]
    assert any("bhashasense" in t for t in entity_texts)
    assert any("txn-9482" in t for t in entity_texts)
    assert any("subscription" in t for t in entity_texts)
