import io
from fastapi.testclient import TestClient

def test_audio_upload_and_pipeline_execution(client: TestClient, sample_wav_bytes: bytes):
    # 1. Upload audio file
    files = {"file": ("test_sample.wav", io.BytesIO(sample_wav_bytes), "audio/wav")}
    upload_res = client.post("/api/v1/audio/upload", files=files)
    assert upload_res.status_code == 201
    upload_data = upload_res.json()
    assert "file_id" in upload_data
    assert upload_data["metadata"]["duration_seconds"] > 0
    assert len(upload_data["waveform_peaks"]) > 0

    file_id = upload_data["file_id"]

    # 2. Run Synchronous Pipeline
    pipeline_res = client.post(
        "/api/v1/pipeline/run",
        json={"file_id": file_id}
    )
    assert pipeline_res.status_code == 200
    pipeline_data = pipeline_res.json()

    assert pipeline_data["status"] == "completed"
    assert "full_transcript" in pipeline_data
    assert len(pipeline_data["segments"]) > 0
    assert "sentiment_summary" in pipeline_data
    assert len(pipeline_data["entities"]) > 0
    assert "metrics" in pipeline_data
    assert pipeline_data["metrics"]["total_processing_time_ms"] > 0
