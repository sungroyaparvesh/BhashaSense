from app.ml.registry import ml_registry
from app.schemas.sentiment import SentimentLabel

def test_sentiment_analysis_negative_signals():
    segments = [
        {"id": 1, "text": "Mera payment deduct ho gaya par balance update nahi hua, bahut problem hai.", "start_time": 0.0}
    ]
    seg_results, summary = ml_registry.sentiment_engine.analyze_segments(segments)
    assert len(seg_results) == 1
    assert seg_results[0].sentiment == SentimentLabel.NEGATIVE
    assert summary.overall_sentiment == SentimentLabel.NEGATIVE
    assert any(a.aspect == "billing" for a in seg_results[0].aspects)

def test_sentiment_analysis_positive_signals():
    segments = [
        {"id": 1, "text": "App interface bahut smooth aur achha laga, thank you so much!", "start_time": 0.0}
    ]
    seg_results, summary = ml_registry.sentiment_engine.analyze_segments(segments)
    assert len(seg_results) == 1
    assert seg_results[0].sentiment == SentimentLabel.POSITIVE
    assert summary.overall_sentiment == SentimentLabel.POSITIVE
