import io
import wave
import struct
import math
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings

@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)

@pytest.fixture(scope="session")
def sample_wav_bytes() -> bytes:
    """Generates a 1-second 16kHz mono sine wave in memory as valid WAV bytes."""
    sample_rate = 16000
    duration_s = 1.0
    frequency = 440.0
    num_samples = int(sample_rate * duration_s)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)       # Mono
        wav_file.setsampwidth(2)       # 16-bit
        wav_file.setframerate(sample_rate)

        for i in range(num_samples):
            value = int(32767.0 * 0.5 * math.sin(2.0 * math.pi * frequency * i / sample_rate))
            data = struct.pack("<h", value)
            wav_file.writeframesraw(data)

    buffer.seek(0)
    return buffer.read()
