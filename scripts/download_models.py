"""
BhashaSense Model Download Script
Pre-caches Hugging Face weights and spaCy models into local models_cache/ directory.
"""
import os
import sys
from pathlib import Path

def download_models():
    print("=== BhashaSense Model Pre-Cache Utility ===")
    cache_dir = Path(__file__).resolve().parent.parent / "backend" / "models_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)

    # 1. Download spaCy multilingual model
    print("\n[1/3] Downloading spaCy model (xx_ent_wiki_sm)...")
    try:
        import spacy
        import spacy.cli
        spacy.cli.download("xx_ent_wiki_sm")
        print("✓ spaCy model ready.")
    except Exception as e:
        print(f"Warning: spaCy download failed: {e}")

    # 2. Pre-cache IndicBERT
    print("\n[2/3] Pre-caching IndicBERT (ai4bharat/indic-bert)...")
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        model_id = "ai4bharat/indic-bert"
        AutoTokenizer.from_pretrained(model_id, cache_dir=str(cache_dir))
        print("✓ IndicBERT tokenizer cached.")
    except Exception as e:
        print(f"Warning: IndicBERT cache step failed: {e}")

    # 3. Pre-cache ASR Checkpoint
    print("\n[3/3] Pre-caching ASR Checkpoint...")
    try:
        from transformers import AutoProcessor
        asr_id = "openai/whisper-tiny" # Fast base fallback
        AutoProcessor.from_pretrained(asr_id, cache_dir=str(cache_dir))
        print("✓ ASR processor cached.")
    except Exception as e:
        print(f"Warning: ASR pre-cache failed: {e}")

    print("\nModel pre-caching finished.")

if __name__ == "__main__":
    download_models()
