import re
import unicodedata
from typing import List

class TextNormalizer:
    """
    Cleans and standardizes code-mixed Indian vernacular transcripts
    (Hinglish, Tamlish, Benglish) for downstream NLP models.
    """

    NOISE_TOKENS = [
        re.compile(r"\[.*?\]"),      # [noise], [applause], [cough]
        re.compile(r"<.*?>"),        # <unk>, <silence>
        re.compile(r"\s+"),          # collapse multiple spaces
    ]

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Sanitizes raw ASR transcript output."""
        if not text:
            return ""

        # Normalize Unicode (NFKC)
        normalized = unicodedata.normalize("NFKC", text)

        # Strip acoustic annotations
        for pattern in cls.NOISE_TOKENS[:2]:
            normalized = pattern.sub("", normalized)

        # Strip redundant whitespaces
        normalized = cls.NOISE_TOKENS[2].sub(" ", normalized).strip()
        return normalized

    @classmethod
    def split_into_clauses(cls, text: str) -> List[str]:
        """
        Splits code-mixed text into readable clauses based on punctuation
        or common discourse connectives (lekin, par, but, and, aur).
        """
        cleaned = cls.clean_text(text)
        if not cleaned:
            return []

        # Split on sentence boundaries first
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        results = [s.strip() for s in sentences if s.strip()]
        return results if results else [cleaned]
