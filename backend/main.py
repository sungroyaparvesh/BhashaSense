"""
BhashaSense API — Standalone FastAPI server (mock/dev mode)
=============================================================
Implements every endpoint the React frontend calls:
  • POST /api/v1/audio/upload    – ingest audio file, return metadata + waveform
  • POST /api/v1/pipeline/run    – run mock ASR → Sentiment → NER pipeline
  • GET  /api/v1/jobs/{job_id}/export/{fmt} – download results as JSON / CSV
  • GET  /health                 – liveness probe
  • GET  /                       – app info

All ML results are deterministic mocks so the full UI can be exercised
without GPU / model weights.  Set MOCK_ML_MODELS=false (future) to swap
in real inference engines.
"""

import asyncio
import io
import csv
import json
import math
import os
import random
import time
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from models.ml_pipeline import (
    transcribe_audio,
    analyze_sentiment,
    extract_entities,
)

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="BhashaSense API",
    description="End-to-end code-mixed audio sentiment and entity extraction pipeline",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS — allow the Vite dev server
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory job/file store (dev only — resets on server restart)
# ---------------------------------------------------------------------------

_uploaded_files: dict[str, dict] = {}   # file_id -> metadata
_pipeline_jobs: dict[str, dict] = {}    # job_id  -> PipelineResponse

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".aac", ".m4a", ".ogg", ".webm", ".flac"}
MAX_UPLOAD_MB = 50

# ---------------------------------------------------------------------------
# Helper – probe a WAV file for duration / sample-rate / channels
# ---------------------------------------------------------------------------

def _probe_wav(data: bytes) -> dict:
    try:
        with wave.open(io.BytesIO(data)) as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            frames = wf.getnframes()
            duration = frames / sr if sr else 0.0
            return {"sample_rate": sr, "channels": ch, "duration_seconds": round(duration, 3)}
    except Exception:
        return {"sample_rate": 16000, "channels": 1, "duration_seconds": 5.0}


def _mock_waveform(n: int = 150) -> list[float]:
    """Generate a plausible-looking waveform for UI rendering."""
    peaks = []
    for i in range(n):
        t = i / n
        val = (
            0.4 * math.sin(2 * math.pi * 3 * t)
            + 0.25 * math.sin(2 * math.pi * 7 * t)
            + 0.15 * random.gauss(0, 1)
        )
        peaks.append(round(max(-1.0, min(1.0, val)), 4))
    return peaks


# ---------------------------------------------------------------------------
# Helper – build the full mock PipelineResponse payload
# ---------------------------------------------------------------------------

MOCK_TRANSCRIPT_SEGMENTS = [
    {
        "id": 0,
        "text": "Mera account blocked ho gaya hai, please help karo.",
        "start_time": 0.0,
        "end_time": 4.2,
        "confidence": 0.91,
        "language_code": "hi-en",
        "words": [
            {"word": "Mera",      "start_time": 0.0,  "end_time": 0.4,  "confidence": 0.95},
            {"word": "account",   "start_time": 0.4,  "end_time": 0.9,  "confidence": 0.92},
            {"word": "blocked",   "start_time": 0.9,  "end_time": 1.4,  "confidence": 0.89},
            {"word": "ho",        "start_time": 1.4,  "end_time": 1.6,  "confidence": 0.97},
            {"word": "gaya",      "start_time": 1.6,  "end_time": 1.9,  "confidence": 0.93},
            {"word": "hai",       "start_time": 1.9,  "end_time": 2.1,  "confidence": 0.96},
            {"word": "please",    "start_time": 2.1,  "end_time": 2.5,  "confidence": 0.98},
            {"word": "help",      "start_time": 2.5,  "end_time": 2.8,  "confidence": 0.97},
            {"word": "karo",      "start_time": 2.8,  "end_time": 3.1,  "confidence": 0.91},
        ],
    },
    {
        "id": 1,
        "text": "Teen din se koi response nahi aaya from customer care.",
        "start_time": 4.5,
        "end_time": 8.9,
        "confidence": 0.88,
        "language_code": "hi-en",
        "words": [
            {"word": "Teen",      "start_time": 4.5,  "end_time": 4.8,  "confidence": 0.90},
            {"word": "din",       "start_time": 4.8,  "end_time": 5.1,  "confidence": 0.88},
            {"word": "se",        "start_time": 5.1,  "end_time": 5.2,  "confidence": 0.95},
            {"word": "koi",       "start_time": 5.2,  "end_time": 5.5,  "confidence": 0.87},
            {"word": "response",  "start_time": 5.5,  "end_time": 6.0,  "confidence": 0.93},
            {"word": "nahi",      "start_time": 6.0,  "end_time": 6.3,  "confidence": 0.92},
            {"word": "aaya",      "start_time": 6.3,  "end_time": 6.6,  "confidence": 0.89},
            {"word": "from",      "start_time": 6.6,  "end_time": 6.8,  "confidence": 0.97},
            {"word": "customer",  "start_time": 6.8,  "end_time": 7.3,  "confidence": 0.94},
            {"word": "care",      "start_time": 7.3,  "end_time": 7.6,  "confidence": 0.95},
        ],
    },
    {
        "id": 2,
        "text": "Main bahut frustrated hoon, yeh service bilkul theek nahi hai.",
        "start_time": 9.1,
        "end_time": 13.4,
        "confidence": 0.86,
        "language_code": "hi-en",
        "words": [
            {"word": "Main",         "start_time": 9.1,  "end_time": 9.4,  "confidence": 0.94},
            {"word": "bahut",        "start_time": 9.4,  "end_time": 9.7,  "confidence": 0.90},
            {"word": "frustrated",   "start_time": 9.7,  "end_time": 10.3, "confidence": 0.88},
            {"word": "hoon",         "start_time": 10.3, "end_time": 10.6, "confidence": 0.91},
            {"word": "yeh",          "start_time": 10.6, "end_time": 10.8, "confidence": 0.96},
            {"word": "service",      "start_time": 10.8, "end_time": 11.3, "confidence": 0.89},
            {"word": "bilkul",       "start_time": 11.3, "end_time": 11.7, "confidence": 0.87},
            {"word": "theek",        "start_time": 11.7, "end_time": 12.0, "confidence": 0.85},
            {"word": "nahi",         "start_time": 12.0, "end_time": 12.3, "confidence": 0.92},
            {"word": "hai",          "start_time": 12.3, "end_time": 12.5, "confidence": 0.95},
        ],
    },
    {
        "id": 3,
        "text": "Refund bhi nahi mila abhi tak, total ₹2,499 ka issue hai.",
        "start_time": 13.7,
        "end_time": 17.8,
        "confidence": 0.84,
        "language_code": "hi-en",
        "words": [
            {"word": "Refund",   "start_time": 13.7, "end_time": 14.1, "confidence": 0.86},
            {"word": "bhi",      "start_time": 14.1, "end_time": 14.3, "confidence": 0.93},
            {"word": "nahi",     "start_time": 14.3, "end_time": 14.6, "confidence": 0.91},
            {"word": "mila",     "start_time": 14.6, "end_time": 14.9, "confidence": 0.88},
            {"word": "abhi",     "start_time": 14.9, "end_time": 15.2, "confidence": 0.90},
            {"word": "tak",      "start_time": 15.2, "end_time": 15.4, "confidence": 0.94},
            {"word": "total",    "start_time": 15.4, "end_time": 15.8, "confidence": 0.87},
            {"word": "₹2,499",   "start_time": 15.8, "end_time": 16.4, "confidence": 0.83},
            {"word": "ka",       "start_time": 16.4, "end_time": 16.6, "confidence": 0.96},
            {"word": "issue",    "start_time": 16.6, "end_time": 16.9, "confidence": 0.85},
            {"word": "hai",      "start_time": 16.9, "end_time": 17.1, "confidence": 0.93},
        ],
    },
]

MOCK_ENTITIES = [
    {
        "id": "ent-001",
        "text": "customer care",
        "category": "ORGANIZATION",
        "start_char": 55,
        "end_char": 68,
        "confidence": 0.92,
        "segment_id": 1,
        "start_time": 6.8,
        "end_time": 7.6,
        "context_sentence": "Teen din se koi response nahi aaya from customer care.",
    },
    {
        "id": "ent-002",
        "text": "₹2,499",
        "category": "MONEY",
        "start_char": 144,
        "end_char": 150,
        "confidence": 0.97,
        "segment_id": 3,
        "start_time": 15.8,
        "end_time": 16.4,
        "context_sentence": "Refund bhi nahi mila abhi tak, total ₹2,499 ka issue hai.",
    },
    {
        "id": "ent-003",
        "text": "Teen din",
        "category": "DATE_TIME",
        "start_char": 52,
        "end_char": 60,
        "confidence": 0.85,
        "segment_id": 1,
        "start_time": 4.5,
        "end_time": 5.1,
        "context_sentence": "Teen din se koi response nahi aaya from customer care.",
    },
    {
        "id": "ent-004",
        "text": "account blocked",
        "category": "ISSUE_TYPE",
        "start_char": 5,
        "end_char": 20,
        "confidence": 0.88,
        "segment_id": 0,
        "start_time": 0.4,
        "end_time": 1.4,
        "context_sentence": "Mera account blocked ho gaya hai, please help karo.",
    },
    {
        "id": "ent-005",
        "text": "Refund",
        "category": "ISSUE_TYPE",
        "start_char": 133,
        "end_char": 139,
        "confidence": 0.91,
        "segment_id": 3,
        "start_time": 13.7,
        "end_time": 14.1,
        "context_sentence": "Refund bhi nahi mila abhi tak, total ₹2,499 ka issue hai.",
    },
]

MOCK_ENTITY_SUMMARY = [
    {"category": "ORGANIZATION", "count": 1, "items": ["customer care"]},
    {"category": "MONEY",        "count": 1, "items": ["₹2,499"]},
    {"category": "DATE_TIME",    "count": 1, "items": ["Teen din"]},
    {"category": "ISSUE_TYPE",   "count": 2, "items": ["account blocked", "Refund"]},
]

MOCK_SEGMENT_SENTIMENTS = [
    {
        "segment_id": 0,
        "text": "Mera account blocked ho gaya hai, please help karo.",
        "sentiment": "negative",
        "confidence": 0.87,
        "distribution": {"positive": 0.05, "neutral": 0.08, "negative": 0.87},
        "aspects": [
            {"aspect": "account_access", "sentiment": "negative", "score": 0.91, "mention_span": "account blocked"},
        ],
    },
    {
        "segment_id": 1,
        "text": "Teen din se koi response nahi aaya from customer care.",
        "sentiment": "negative",
        "confidence": 0.82,
        "distribution": {"positive": 0.04, "neutral": 0.14, "negative": 0.82},
        "aspects": [
            {"aspect": "response_time", "sentiment": "negative", "score": 0.85, "mention_span": "Teen din"},
            {"aspect": "support_quality", "sentiment": "negative", "score": 0.79, "mention_span": "customer care"},
        ],
    },
    {
        "segment_id": 2,
        "text": "Main bahut frustrated hoon, yeh service bilkul theek nahi hai.",
        "sentiment": "negative",
        "confidence": 0.93,
        "distribution": {"positive": 0.02, "neutral": 0.05, "negative": 0.93},
        "aspects": [
            {"aspect": "service_quality", "sentiment": "negative", "score": 0.94, "mention_span": "service bilkul theek nahi"},
        ],
    },
    {
        "segment_id": 3,
        "text": "Refund bhi nahi mila abhi tak, total ₹2,499 ka issue hai.",
        "sentiment": "negative",
        "confidence": 0.89,
        "distribution": {"positive": 0.03, "neutral": 0.08, "negative": 0.89},
        "aspects": [
            {"aspect": "refund", "sentiment": "negative", "score": 0.92, "mention_span": "Refund bhi nahi mila"},
            {"aspect": "billing", "sentiment": "negative", "score": 0.87, "mention_span": "₹2,499"},
        ],
    },
]

MOCK_SENTIMENT_SUMMARY = {
    "overall_sentiment": "negative",
    "confidence": 0.88,
    "distribution": {"positive": 0.035, "neutral": 0.087, "negative": 0.878},
    "sentiment_score_timeline": [
        {"time": 0.0,  "score": -0.52, "sentiment": "negative"},
        {"time": 2.1,  "score": -0.61, "sentiment": "negative"},
        {"time": 4.5,  "score": -0.69, "sentiment": "negative"},
        {"time": 6.8,  "score": -0.74, "sentiment": "negative"},
        {"time": 9.1,  "score": -0.88, "sentiment": "negative"},
        {"time": 11.5, "score": -0.91, "sentiment": "negative"},
        {"time": 13.7, "score": -0.85, "sentiment": "negative"},
        {"time": 16.0, "score": -0.79, "sentiment": "negative"},
    ],
    "aspect_breakdown": [
        {"aspect": "account_access",  "sentiment": "negative", "score": 0.91},
        {"aspect": "response_time",   "sentiment": "negative", "score": 0.85},
        {"aspect": "support_quality", "sentiment": "negative", "score": 0.79},
        {"aspect": "service_quality", "sentiment": "negative", "score": 0.94},
        {"aspect": "refund",          "sentiment": "negative", "score": 0.92},
        {"aspect": "billing",         "sentiment": "negative", "score": 0.87},
    ],
}


def _build_pipeline_response(file_id: str, audio_meta: dict) -> dict:
    job_id = f"job-{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()

    # Simulate processing latency
    asr_ms = random.randint(180, 340)
    sentiment_ms = random.randint(90, 170)
    ner_ms = random.randint(40, 90)
    total_ms = asr_ms + sentiment_ms + ner_ms + random.randint(20, 60)

    full_transcript = " ".join(seg["text"] for seg in MOCK_TRANSCRIPT_SEGMENTS)

    return {
        "job_id": job_id,
        "status": "completed",
        "audio_metadata": {
            "file_id": file_id,
            "original_filename": audio_meta.get("filename", "audio.wav"),
            "stored_path": f"uploads/{file_id}",
            "duration_seconds": audio_meta.get("duration_seconds", 17.8),
            "sample_rate": audio_meta.get("sample_rate", 16000),
            "channels": audio_meta.get("channels", 1),
            "format": audio_meta.get("format", "wav"),
            "file_size_bytes": audio_meta.get("file_size_bytes", 0),
        },
        "full_transcript": full_transcript,
        "detected_language": "hi-en (Hinglish)",
        "segments": MOCK_TRANSCRIPT_SEGMENTS,
        "sentiment_summary": MOCK_SENTIMENT_SUMMARY,
        "segment_sentiments": MOCK_SEGMENT_SENTIMENTS,
        "entities": MOCK_ENTITIES,
        "entity_summary": MOCK_ENTITY_SUMMARY,
        "metrics": {
            "audio_duration_seconds": audio_meta.get("duration_seconds", 17.8),
            "total_processing_time_ms": total_ms,
            "asr_time_ms": asr_ms,
            "sentiment_time_ms": sentiment_ms,
            "ner_time_ms": ner_ms,
            "device_used": "CPU (mock)",
        },
        "created_at": now,
    }


# ---------------------------------------------------------------------------
# Routes — liveness
# ---------------------------------------------------------------------------

@app.get("/")
def read_root():
    return {
        "app": "BhashaSense",
        "status": "online",
        "version": "0.1.0",
        "message": "Welcome to BhashaSense Audio Intelligence Platform",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "frontend_cors_allowed": "http://localhost:5173",
        "mock_mode": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Route — Unified Audio Intelligence Analysis Pipeline
# ---------------------------------------------------------------------------

@app.post("/analyze")
async def analyze_audio_endpoint(file: UploadFile = File(...)):
    """
    Accepts an audio file upload (.wav, .mp3, .aac, .m4a, .ogg, .webm), saves it
    preserving the original file extension so FFmpeg accurately detects the container,
    transcribes speech via IndicConformer ASR, and concurrently executes IndicSentiment
    and BERT NER via asyncio.gather(). Cleans up the temporary audio file after inference.
    """
    filename = file.filename or "audio.wav"
    ext = Path(filename).suffix.lower() or ".wav"
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Preserve original extension (e.g. temp_audio_<uuid>.aac or .m4a) for FFmpeg container detection
    unique_id = uuid.uuid4().hex[:8]
    temp_file_path = f"temp_audio_{unique_id}{ext}"

    try:
        # Save uploaded bytes to temporary audio file with original container extension
        with open(temp_file_path, "wb") as f:
            f.write(audio_bytes)

        # Step 1: Sequential audio transcription (IndicConformer ASR via FFmpeg)
        asr_result = await transcribe_audio(temp_file_path, filename=filename)
        transcript_text = asr_result["transcript"]

        # Step 2: Concurrent sentiment analysis & entity extraction via asyncio.gather()
        sentiment_result, entity_result = await asyncio.gather(
            analyze_sentiment(transcript=transcript_text),
            extract_entities(transcript=transcript_text),
        )

        # Step 3: Combine into unified JSON payload matching React frontend schema
        return {
            "status": "success",
            "audio": {
                "filename": filename,
                "size_bytes": len(audio_bytes),
                "detected_language": asr_result["detected_language"],
                "language_confidence": asr_result["language_confidence"],
            },
            "transcription": asr_result,
            "sentiment_analysis": sentiment_result,
            "entity_extraction": entity_result,
            "pipeline_metadata": {
                "orchestration": "asyncio.gather",
                "models_used": [
                    "AI4Bharat IndicConformer 600M (ASR)",
                    "airzipm/sentiment-analysis-muril-v2 (IndicSentiment)",
                    "dslim/bert-base-NER (NER)",
                ],
                "execution_mode": "real_inference",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }
    finally:
        # Cleanly delete the temporary file after inference finishes
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Routes — audio
# ---------------------------------------------------------------------------

@app.post("/api/v1/audio/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Accept an audio file upload; return metadata + mock waveform peaks."""

    ext = Path(file.filename or "audio.wav").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB} MB limit.")

    file_id = f"f-{uuid.uuid4().hex[:16]}"

    # Probe WAV for real duration; fall back to mock values for other formats
    probe = _probe_wav(data) if ext == ".wav" else {"sample_rate": 44100, "channels": 1, "duration_seconds": 10.0}

    meta = {
        "file_id": file_id,
        "filename": file.filename,
        "sample_rate": probe["sample_rate"],
        "channels": probe["channels"],
        "duration_seconds": probe["duration_seconds"],
        "format": ext.lstrip("."),
        "file_size_bytes": len(data),
    }
    _uploaded_files[file_id] = meta

    waveform_peaks = _mock_waveform(150)

    return {
        "file_id": file_id,
        "filename": file.filename,
        "metadata": {
            "file_id": file_id,
            "original_filename": file.filename,
            "stored_path": f"uploads/{file_id}",
            "duration_seconds": probe["duration_seconds"],
            "sample_rate": probe["sample_rate"],
            "channels": probe["channels"],
            "format": ext.lstrip("."),
            "file_size_bytes": len(data),
        },
        "waveform_peaks": waveform_peaks,
    }


@app.get("/api/v1/audio/waveform/{file_id}")
def get_waveform(file_id: str, peaks: int = 150):
    if file_id not in _uploaded_files:
        raise HTTPException(status_code=404, detail="File not found.")
    return {
        "file_id": file_id,
        "peaks": _mock_waveform(peaks),
        "sample_count": peaks,
    }


# ---------------------------------------------------------------------------
# Routes — pipeline
# ---------------------------------------------------------------------------

@app.post("/api/v1/pipeline/run")
async def run_pipeline(body: dict[str, Any]):
    """
    Accepts { "file_id": "...", "options": {...} }
    Returns the full PipelineResponse JSON.
    """
    file_id = body.get("file_id")
    if not file_id:
        raise HTTPException(status_code=422, detail="'file_id' is required.")

    if file_id not in _uploaded_files:
        raise HTTPException(status_code=404, detail=f"file_id '{file_id}' not found. Upload the file first.")

    audio_meta = _uploaded_files[file_id]

    # Simulate processing time
    time.sleep(0.6)

    result = _build_pipeline_response(file_id, audio_meta)
    _pipeline_jobs[result["job_id"]] = result
    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# Routes — job export
# ---------------------------------------------------------------------------

@app.get("/api/v1/jobs/{job_id}/export/{fmt}")
def export_job(job_id: str, fmt: str):
    if job_id not in _pipeline_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    result = _pipeline_jobs[job_id]

    if fmt == "json":
        content = json.dumps(result, ensure_ascii=False, indent=2)
        return StreamingResponse(
            io.BytesIO(content.encode()),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="bhashasense_{job_id}.json"'},
        )

    elif fmt == "segments-csv":
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["segment_id", "text", "start_time", "end_time", "sentiment", "confidence"])
        for seg, sent in zip(result["segments"], result["segment_sentiments"]):
            writer.writerow([
                seg["id"], seg["text"], seg["start_time"], seg["end_time"],
                sent["sentiment"], sent["confidence"],
            ])
        return StreamingResponse(
            io.BytesIO(buf.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="segments_{job_id}.csv"'},
        )

    elif fmt == "entities-csv":
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["id", "text", "category", "confidence", "start_time", "end_time", "context"])
        for ent in result["entities"]:
            writer.writerow([
                ent["id"], ent["text"], ent["category"], ent["confidence"],
                ent.get("start_time", ""), ent.get("end_time", ""), ent.get("context_sentence", ""),
            ])
        return StreamingResponse(
            io.BytesIO(buf.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="entities_{job_id}.csv"'},
        )

    else:
        raise HTTPException(status_code=400, detail=f"Unknown export format '{fmt}'. Use: json, segments-csv, entities-csv")


# ---------------------------------------------------------------------------
# Dev entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
