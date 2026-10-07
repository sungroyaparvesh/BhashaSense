from pathlib import Path
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import json

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Base App Settings
    PROJECT_NAME: str = "BhashaSense"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_V1_PREFIX: str = "/api/v1"

    # CORS
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000"
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                return [i.strip() for i in v.split(",") if i.strip()]
        return v

    # Storage & Upload Paths
    UPLOAD_DIR: Path = Path("./storage/uploads")
    MODELS_CACHE_DIR: Path = Path("./models_cache")
    MAX_AUDIO_SIZE_MB: int = 50
    ALLOWED_AUDIO_FORMATS: List[str] = ["wav", "mp3", "m4a", "flac", "ogg"]

    # ML Inference Settings
    # auto, cuda, mps, cpu
    DEVICE: str = "auto"
    MOCK_ML_MODELS: bool = True  # Default to true for easy setup and testing

    # Model Identifiers
    ASR_MODEL_ID: str = "ai4bharat/indicconformer_all"
    SENTIMENT_MODEL_ID: str = "ai4bharat/indic-bert"
    SPACY_MODEL_ID: str = "xx_ent_wiki_sm"

    def setup_directories(self) -> None:
        """Ensure necessary directories exist on disk."""
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.setup_directories()
