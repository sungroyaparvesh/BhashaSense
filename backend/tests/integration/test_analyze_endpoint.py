import io
import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def standalone_client():
    return TestClient(app)

def test_analyze_endpoint_with_wav_audio(standalone_client: TestClient, sample_wav_bytes: bytes):
    files = {"file": ("customer_speech.wav", io.BytesIO(sample_wav_bytes), "audio/wav")}
    response = standalone_client.post("/analyze", files=files)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["audio"]["filename"] == "customer_speech.wav"
    assert "detected_language" in data["audio"]
    assert "transcription" in data
    assert "transcript" in data["transcription"]
    assert "sentiment_analysis" in data
    assert "overall_sentiment" in data["sentiment_analysis"]
    assert "distribution" in data["sentiment_analysis"]
    assert "entity_extraction" in data
    assert "entities" in data["entity_extraction"]
    assert "pipeline_metadata" in data
    assert data["pipeline_metadata"]["execution_mode"] == "real_inference"

def test_analyze_endpoint_empty_file_fails(standalone_client: TestClient):
    files = {"file": ("empty.wav", io.BytesIO(b""), "audio/wav")}
    response = standalone_client.post("/analyze", files=files)
    assert response.status_code == 400
