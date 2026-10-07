from typing import Optional
from app.core.logging import logger
from app.ml.device import get_device
from app.ml.asr.indic_conformer import IndicConformerASR
from app.ml.sentiment.indic_bert_sentiment import IndicBERTSentimentAnalyzer
from app.ml.ner.spacy_extractor import SpacyEntityExtractor

class MLRegistry:
    """
    Central Singleton Model Registry for BhashaSense.
    Manages lifetime, GPU memory allocation, and lazy initialization
    of torchaudio, IndicConformer, IndicBERT, and spaCy models.
    """
    _instance: Optional["MLRegistry"] = None

    def __init__(self):
        self.device = get_device()
        self.asr_engine = IndicConformerASR(device=self.device)
        self.sentiment_engine = IndicBERTSentimentAnalyzer(device=self.device)
        self.ner_engine = SpacyEntityExtractor()
        self._is_initialized = False

    @classmethod
    def get_instance(cls) -> "MLRegistry":
        if cls._instance is None:
            cls._instance = MLRegistry()
        return cls._instance

    def initialize_all(self) -> None:
        """Initializes and pre-warms all ML engines."""
        if self._is_initialized:
            return

        logger.info("Initializing BhashaSense ML Pipeline Models...")
        self.asr_engine.load_model()
        self.sentiment_engine.load_model()
        self.ner_engine.load_model()
        self._is_initialized = True
        logger.info("All BhashaSense ML models are loaded and ready.")

ml_registry = MLRegistry.get_instance()
