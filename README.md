# BhashaSense

**Enterprise End-to-End Code-Mixed Audio Sentiment & Entity Extraction Platform**

BhashaSense is an enterprise-grade AI system designed to transcribe, analyze sentiment, and extract domain entities from code-mixed Indian vernacular speech (e.g., Hinglish, Tamlish, Benglish) in near real-time.

---

## Architecture Overview

- **Frontend**: React 18, Vite, TypeScript, Tailwind CSS, WaveSurfer.js, Zustand.
- **Backend**: Python 3.10+, FastAPI, Pydantic v2, PyTorch, `torchaudio`.
- **Speech-to-Text (ASR)**: AI4Bharat IndicConformer / Multilingual Whisper fallback with timestamp alignment.
- **Sentiment Engine**: Fine-tuned IndicBERT transformer for code-mixed speech polarity and aspect sentiment.
- **Entity Extraction (NER)**: Hybrid spaCy + IndicBERT token classification for Indian domain entities.

---

## Quick Start

### 1. Backend Setup
```bash
cd backend
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

Visit the frontend at `http://localhost:5173` and backend API docs at `http://localhost:8000/docs`.

---

## Environment Configuration
Copy `.env.example` in both `/backend` and `/frontend` and configure hardware acceleration (`DEVICE=cuda|mps|cpu`) and model paths.
Set `MOCK_ML_MODELS=true` to develop and test without heavy GPU/model downloads.
