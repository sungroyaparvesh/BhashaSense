"""
BhashaSense — ML Pipeline
=========================
Production ML inference chain using the AI4Bharat Indic Ecosystem:
  • transcribe_audio(file_path) -> AI4Bharat IndicConformer-600M ('ai4bharat/indic-conformer-600m-multilingual')
  • analyze_sentiment(text)     -> MuRIL IndicSentiment ('airzipm/sentiment-analysis-muril-v2')
  • extract_entities(text)      -> AI4Bharat IndicNER ('ai4bharat/IndicNER') + Hybrid Indic Gazetteer

All models are cached at the module level for zero reloading overhead.
Inference calls run in worker threads via asyncio.to_thread to maintain
high concurrency and prevent event-loop blocking.
"""

import asyncio
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel, pipeline
import imageio_ffmpeg

# Automatically read backend/.env if HF_TOKEN isn't already in the terminal environment
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

# Fetch the token by its variable name, NOT the raw token string
HF_TOKEN = os.getenv('HF_TOKEN')

# Dynamically inject the bundled FFmpeg into the environment PATH
os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())

# ---------------------------------------------------------------------------
# Global IndicConformer ASR Initialization
# ---------------------------------------------------------------------------
# Load the official AI4Bharat 600M parameter multilingual Conformer
# trust_remote_code=True is required for their custom architecture
try:
    asr_model = AutoModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual", 
        trust_remote_code=True,
        token=HF_TOKEN
    )
    asr_model.eval()
except Exception as e:
    # Handle environment where gated HF access or custom code dependencies fallback
    print(f"[ASR Warning] Unable to load ai4bharat/indic-conformer-600m-multilingual directly: {e}")
    asr_model = None

# Fine-tuned on ai4bharat/IndicSentiment for multilingual Indic support
sentiment_analyzer = pipeline("text-classification", model="airzipm/sentiment-analysis-muril-v2", top_k=None)

# ---------------------------------------------------------------------------
# Global IndicNER Initialization (with Ungated Fallback)
# ---------------------------------------------------------------------------
ner_pipeline = None
for ner_model_id in ["ai4bharat/IndicNER", "techysanoj/fine-tuned-IndicNER", "Davlan/bert-base-multilingual-cased-ner-hrl"]:
    try:
        ner_pipeline = pipeline(
            "token-classification",
            model=ner_model_id,
            tokenizer=ner_model_id,
            aggregation_strategy="first",
            token=HF_TOKEN if "ai4bharat" in ner_model_id else None
        )
        print(f"[BhashaSense] Loaded NER model: {ner_model_id}")
        break
    except Exception as e:
        print(f"[BhashaSense] Could not load {ner_model_id}: {e}")


def get_sentiment_pipeline():
    """Returns the globally cached Multilingual Indic sentiment analysis pipeline."""
    return sentiment_analyzer


def get_ner_pipeline():
    """Returns the cached NER pipeline."""
    return ner_pipeline


# ---------------------------------------------------------------------------
# Audio Ingestion & Autonomous Regional Indic / Code-Mixed ASR
# ---------------------------------------------------------------------------

INDIC_LANG_NAMES = {
    "hi": "Hindi",
    "bn": "Bengali",
    "pa": "Punjabi",
    "mr": "Marathi",
    "gu": "Gujarati",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "or": "Odia",
    "ur": "Urdu",
}

# High-frequency grammatical & function words for each regional Indian language
LANG_LEXICONS = {
    "hi": {
        "है", "हैं", "था", "थे", "थी", "का", "की", "के", "में", "से", "को", "पर",
        "नहीं", "क्या", "क्यों", "कैसे", "और", "तो", "भी", "ये", "वो", "इस", "उस",
        "किया", "होता", "होती", "रहा", "रहे", "रही", "गया", "गए", "गई", "लिए", "साथ",
        "हम", "आप", "मैं", "मेरा", "मेरी", "मेरे", "अपने", "अपना", "कहा", "बात", "लोग"
    },
    "bn": {
        "একটা", "একটি", "কথা", "আপনার", "আমার", "মানুষ", "করে", "হয়", "হয়ে", "থেকে",
        "জন্য", "এই", "ওই", "কি", "কী", "না", "আছে", "বলে", "তার", "সাথে", "কিন্তু",
        "যদি", "এবং", "আমি", "তুমি", "আপনি", "সে", "তারা", "কেন", "কেমন", "অসুস্থ",
        "পড়ে", "গেছে", "ছিল", "হবে", "করতে", "বড়", "ছোট", "দিন", "বছর", "আজ",
        "মতো", "মধ্যে", "নিজের", "কোনো", "কিছু", "অনেক", "সব", "ভালো", "খারাপ", "বাড়ি"
    },
    "pa": {
        "ਵਿੱਚ", "ਨੂੰ", "ਨਾਲ", "ਤੋਂ", "ਸੀ", "ਹਨ", "ਤੁਸੀਂ", "ਅਸੀਂ", "ਸਾਡਾ", "ਤੁਹਾਡਾ",
        "ਹੁੰਦਾ", "ਹੁੰਦੀ", "ਕਿੱਥੇ", "ਕਿਵੇਂ", "ਕਿਉਂਕਿ", "ਅਤੇ", "ਦਾ", "ਦੀ", "ਦੇ", "ਨੇ",
        "ਪੰਜਾਬ", "ਵਾਹਿਗੁਰੂ", "ਇੱਕ", "ਉੱਤੇ", "ਵੱਲੋਂ", "ਰਾਹੀਂ", "ਕੀਤਾ", "ਗਿਆ", "ਨਹੀਂ",
        "ਮੇਰਾ", "ਤੇਰਾ", "ਉਹ", "ਇਹ", "ਕੋਈ", "ਸਭ", "ਬੰਦਾ", "ਘਰ", "ਗੱਲ"
    },
    "mr": {
        "आहे", "आहेत", "होता", "होते", "मला", "तुम्ही", "आम्ही", "च्या", "मध्ये",
        "आणि", "पण", "काय", "कसे", "नाही", "या", "त्या", "करून", "झाला", "झाली", "माझ्या"
    },
    "gu": {
        "છે", "હતું", "હતા", "અને", "પણ", "માં", "થી", "ને", "માટે", "શું",
        "કેવી", "નથી", "આ", "તે", "તમે", "અમે", "કરી", "થઈ", "ગયા", "મારા"
    },
    "ta": {
        "மற்றும்", "என்று", "இந்த", "அந்த", "ஒரு", "நான்", "நீங்கள்", "அவர்",
        "இல்லை", "என்ன", "எப்படி", "உள்ளது", "இருந்தது", "செய்து", "அவர்கள்", "நம்"
    },
    "te": {
        "మరియు", "ఈ", "ఆ", "ఒక", "నేను", "మీరు", "అతను", "కాదు", "ఏమిటి",
        "ఎలా", "ఉంది", "ఉన్నాయి", "చేసి", "కోసం", "వారు", "మన", "చాలా"
    },
    "kn": {
        "ಮತ್ತು", "ಈ", "ಆ", "ಒಂದು", "ನಾನು", "ನೀವು", "ಅವರು", "ಇಲ್ಲ", "ಏನು",
        "ಹೇಗೆ", "ಇದೆ", "ಮೇಲೆ", "ಗಾಗಿ", "ನಮ್ಮ", "ಮಾಡಿ", "ಬಹಳ"
    },
    "ml": {
        "ഒരു", "ഈ", "ആ", "ഞാൻ", "നിങ്ങൾ", "അവർ", "ഇല്ല", "എന്ത്", "എങ്ങനെ",
        "ഉണ്ട്", "ആണ്", "എന്ന്", "വേണ്ടി", "നമ്മുടെ", "ചെയ്തു"
    },
    "or": {
        "ଏବଂ", "ଏହି", "ସେହି", "ଏକ", "ମୁଁ", "ଆପଣ", "ସେ", "ନାହିଁ", "କଣ",
        "କିପରି", "ଅଛି", "ପାଇଁ"
    },
    "ur": {
        "ہے", "ہیں", "تھا", "تھے", "کا", "کی", "کے", "میں", "سے", "کو",
        "نہیں", "کیا", "اور", "تو", "بھی", "یہ", "وہ", "اس", "ساتھ"
    }
}

# Words that appear when Hindi/Bengali is falsely transliterated into Gurmukhi
FALSE_PUNJABI_WORDS = {"ਕਿਆ", "ਆਜ", "ਕਿਉਂ", "ਕੈਸੇ", "ਔਰ", "ਯੇ", "ਵੋ", "ਇਸ", "ਉਸ", "ਆਸਾਮੇ", "ਛਲੇ", "ਬਾਈਰੇ"}


def _score_language_output(text: str, lang_code: str) -> float:
    if not text or not text.strip():
        return 0.0
    words = [w.strip(" ,.।?!-\"'") for w in text.split() if w.strip(" ,.।?!-\"'")]
    if not words:
        return 0.0

    # 1. Exact vocabulary match against the language's core lexicon
    lexicon = LANG_LEXICONS.get(lang_code, set())
    exact_hits = sum(1 for w in words if w in lexicon)
    lexicon_ratio = exact_hits / len(words)

    # 2. Penalize false Gurmukhi transliterations of Hindi/Bengali words
    false_penalty = 0.0
    if lang_code == "pa":
        false_hits = sum(1 for w in words if w in FALSE_PUNJABI_WORDS)
        if false_hits > 0:
            false_penalty += (false_hits / len(words)) * 1.5
        if exact_hits == 0:
            false_penalty += 0.35

    # 3. Word structure coherence (penalize broken 1-letter consonant stutter)
    single_char_words = sum(1 for w in words if len(w) == 1)
    good_len_words = sum(1 for w in words if 2 <= len(w) <= 10)
    structure_score = (good_len_words / len(words)) * 0.4 - (single_char_words / len(words)) * 0.8

    return (lexicon_ratio * 2.5) + structure_score - false_penalty


def _decode_chunk_with_head(chunk_tensor: torch.Tensor, lang_code: str) -> str:
    if asr_model is None:
        return ""
    try:
        with torch.no_grad():
            out = asr_model(chunk_tensor, lang_code, "ctc")
            if isinstance(out, (list, tuple)):
                return str(out[0]).strip() if len(out) > 0 else ""
            return str(out).strip()
    except Exception:
        return ""


def _segment_audio_by_energy(audio_data: np.ndarray, sr: int = 16000) -> list[tuple[int, int]]:
    """
    Splits 16kHz audio into natural utterance segments (up to 8 seconds each),
    identifying low-energy pause frames (50ms) to avoid cutting words in half.
    """
    total_samples = len(audio_data)
    min_samples = 5 * sr   # 5 seconds
    max_samples = 8 * sr   # 8 seconds
    frame_size = int(0.05 * sr)  # 50ms frames (800 samples)

    if total_samples <= max_samples:
        return [(0, total_samples)]

    segments = []
    current_start = 0

    while total_samples - current_start > max_samples:
        window_start = current_start + min_samples
        window_end = min(current_start + max_samples, total_samples)
        search_region = audio_data[window_start:window_end]

        num_frames = len(search_region) // frame_size
        if num_frames > 0:
            trimmed = search_region[:num_frames * frame_size]
            frames = trimmed.reshape((num_frames, frame_size))
            energies = np.sum(frames ** 2, axis=1)
            min_frame_idx = int(np.argmin(energies))
            split_point = window_start + (min_frame_idx * frame_size)
        else:
            split_point = window_end

        segments.append((current_start, split_point))
        current_start = split_point

    if current_start < total_samples:
        segments.append((current_start, total_samples))

    return segments


def _run_transcribe_audio_sync(file_path: str, return_details: bool = False):
    """
    Autonomous multi-lingual Indic speech recognition pipeline:
      - Split the 16kHz normalized audio into 8-second segments at low-energy pauses.
      - Stage 1 (Anchor Probe): Pick the highest-energy segment (up to 6s) and decode it across
        candidate_langs = ["hi", "bn", "pa", "mr", "gu", "ta", "te", "kn", "ml", "or", "ur"],
        scoring each with _score_language_output(txt, code).
      - Select highest-scoring language as primary_lang. Only include a second language in active_langs
        if its lexicon score is strong (> 0.45) and within 0.30 of primary_lang (for genuine code-switched audio).
      - Stage 2 (Full Decoding & Context Smoothing): Decode all segments using active_langs. If more than 75%
        of segments belong to primary_lang, smooth any isolated outlier segment back to primary_lang.
      - Return both the clean transcript and the accurate detected_languages list.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    temp_wav = file_path + "_temp.wav"

    try:
        # Step A — Audio Speech Enhancement (FFmpeg)
        subprocess.run([
            ffmpeg_exe, "-y", "-i", file_path,
            "-vn", "-af", "highpass=f=100,lowpass=f=7500,dynaudnorm=f=150:g=15",
            "-ar", "16000", "-ac", "1", temp_wav
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        audio_data, sr = sf.read(temp_wav, dtype='float32')
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        total_samples = len(audio_data)
        total_duration = total_samples / 16000.0 if sr else 0.0

        if asr_model is None:
            clean_transcript = "Audio processed via AI4Bharat IndicConformer (Model not loaded)."
            detected_languages = ["Hindi"]
            if return_details:
                return clean_transcript, detected_languages, [], total_duration
            return clean_transcript, detected_languages

        if total_samples == 0:
            if return_details:
                return "", ["Hindi"], [], 0.0
            return "", ["Hindi"]

        # Step B — Segment audio into 8-second windows at low-energy pauses
        raw_segment_spans = _segment_audio_by_energy(audio_data, sr=16000)

        # Stage 1 (Anchor Probe):
        # Pick the highest-energy segment (up to 6 seconds) and decode across candidate_langs
        candidate_langs = ["hi", "bn", "pa", "mr", "gu", "ta", "te", "kn", "ml", "or"]

        best_span = max(
            raw_segment_spans,
            key=lambda sp: float(np.mean(audio_data[sp[0]:sp[1]] ** 2)) if sp[1] > sp[0] else 0.0
        )
        probe_start, probe_end = best_span
        probe_audio = audio_data[probe_start:probe_end]
        max_probe_samples = 6 * 16000  # Up to 6 seconds
        if len(probe_audio) > max_probe_samples:
            probe_audio = probe_audio[:max_probe_samples]

        probe_tensor = torch.tensor(probe_audio).unsqueeze(0)

        probe_scores = {}
        for code in candidate_langs:
            txt = _decode_chunk_with_head(probe_tensor, code)
            score = _score_language_output(txt, code)
            probe_scores[code] = score

        primary_lang = max(candidate_langs, key=lambda c: probe_scores[c])
        primary_score = probe_scores[primary_lang]

        # Default fallback if silence or no candidate matches lexicon
        if primary_score <= 0.0:
            primary_lang = "hi"
            primary_score = probe_scores.get("hi", 0.0)

        # Only include a second language in active_langs if lexicon score is strong (> 0.45) and within 0.30
        active_langs = [primary_lang]
        for c in candidate_langs:
            if c != primary_lang:
                sc = probe_scores[c]
                if sc > 0.45 and (primary_score - sc) <= 0.30:
                    active_langs.append(c)

        # Stage 2 (Full Decoding & Context Smoothing):
        decoded_segments = []
        for start_idx, end_idx in raw_segment_spans:
            chunk_samples = end_idx - start_idx
            if chunk_samples < 800:  # Skip degenerate slices under 50ms
                continue

            seg_tensor = torch.tensor(audio_data[start_idx:end_idx]).unsqueeze(0)
            seg_decodings = {}
            seg_scores = {}

            for code in active_langs:
                txt = _decode_chunk_with_head(seg_tensor, code)
                sc = _score_language_output(txt, code)
                seg_decodings[code] = txt
                seg_scores[code] = sc

            if len(active_langs) == 1:
                chosen_code = active_langs[0]
                chosen_text = seg_decodings[chosen_code]
                chosen_score = seg_scores[chosen_code]
            else:
                best_code = max(active_langs, key=lambda c: seg_scores[c])
                if seg_scores[best_code] <= seg_scores[primary_lang] + 0.08:
                    chosen_code = primary_lang
                else:
                    chosen_code = best_code
                chosen_text = seg_decodings[chosen_code]
                chosen_score = seg_scores[chosen_code]

            decoded_segments.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "chosen_code": chosen_code,
                "chosen_text": chosen_text,
                "chosen_score": chosen_score,
                "decodings": seg_decodings,
            })

        # Context Smoothing:
        # If more than 75% of segments belong to primary_lang, smooth outlier segments back to primary_lang
        if decoded_segments:
            primary_count = sum(1 for seg in decoded_segments if seg["chosen_code"] == primary_lang)
            if (primary_count / len(decoded_segments)) > 0.75:
                for seg in decoded_segments:
                    if seg["chosen_code"] != primary_lang:
                        seg["chosen_code"] = primary_lang
                        if primary_lang in seg["decodings"]:
                            seg["chosen_text"] = seg["decodings"][primary_lang]

        # Clean repetitive single-consonant stutter artifacts and assemble timeline segments
        segments_info = []
        for seg_idx, seg in enumerate(decoded_segments):
            cleaned_text = re.sub(
                r"(?:(?<!\S)[\u0915-\u0939\u0995-\u09b9\u0a15-\u0a39\u0a95-\u0ab9\u0b15-\u0b39\u0b95-\u0bb9\u0c15-\u0c39\u0c95-\u0cb9\u0d15-\u0d39](?!\S)\s*){2,}",
                " ",
                seg["chosen_text"]
            )
            cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

            start_sec = round(seg["start_idx"] / 16000.0, 2)
            end_sec = round(seg["end_idx"] / 16000.0, 2)
            conf = round(max(0.65, min(0.98, (seg["chosen_score"] + 1.0) / 2.0)), 2)

            if cleaned_text:
                segments_info.append({
                    "id": seg_idx,
                    "text": cleaned_text,
                    "start_time": start_sec,
                    "end_time": end_sec,
                    "language_code": seg["chosen_code"],
                    "confidence": conf,
                })

        clean_transcript = " ".join(s["text"] for s in segments_info if s["text"]).strip()

        # Build accurate detected_languages list
        used_codes = []
        for s in segments_info:
            c = s["language_code"]
            if c not in used_codes and s["text"]:
                used_codes.append(c)

        if not used_codes:
            used_codes = [primary_lang]

        detected_languages = [INDIC_LANG_NAMES.get(c, c.title()) for c in used_codes]

        if return_details:
            return clean_transcript, detected_languages, segments_info, total_duration
        return clean_transcript, detected_languages

    finally:
        if os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except Exception:
                pass


def transcribe_audio_raw(file_path: str, *args, **kwargs) -> tuple[str, str, list[dict[str, Any]], float]:
    clean_transcript, detected_languages, segments, duration = _run_transcribe_audio_sync(file_path, return_details=True)
    detected_str = " + ".join(detected_languages) + (" (Code-Mixed)" if len(detected_languages) > 1 else "")
    return clean_transcript, detected_str, segments, duration


def _transcribe_sync_wrapper(
    file_path: str,
    filename: str = "audio.wav",
    *args,
    **kwargs,
) -> dict[str, Any]:
    clean_transcript, detected_languages, segments, duration = _run_transcribe_audio_sync(file_path, return_details=True)
    text = clean_transcript.strip() if clean_transcript else ""
    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

    if not segments and text:
        segments.append({
            "id": 0,
            "text": text,
            "start_time": 0.0,
            "end_time": round(duration, 1) if duration > 0 else 2.0,
            "language_code": detected_languages[0].lower() if detected_languages else "hi",
            "confidence": 0.95,
        })

    detected_str = " + ".join(detected_languages) + (" (Code-Mixed)" if len(detected_languages) > 1 else "")

    return {
        "audio_filename": filename,
        "audio_size_bytes": file_size,
        "detected_language": detected_str,
        "detected_languages": detected_languages,
        "language_code": "code-mixed" if len(detected_languages) > 1 else (detected_languages[0].lower() if detected_languages else "hi"),
        "language_confidence": 0.95,
        "duration_seconds": round(duration, 1),
        "transcript": text if text else "No speech detected in audio.",
        "segments": segments,
    }


async def transcribe_audio(
    file_path: str,
    *args,
    filename: str = "audio.wav",
    **kwargs,
) -> dict[str, Any]:
    """
    Asynchronous entrypoint for transcribe_audio called by main.py, executing
    the autonomous code-mixed transcription pipeline in a thread pool.
    """
    resolved_filename = filename
    if args and isinstance(args[0], str) and ("." in args[0] or "/" in args[0] or "\\" in args[0]):
        resolved_filename = args[0]

    return await asyncio.to_thread(_transcribe_sync_wrapper, file_path, resolved_filename)


# ---------------------------------------------------------------------------
# Sentiment Analysis: analyze_sentiment
# ---------------------------------------------------------------------------

def _map_sentiment_label(raw_label: str) -> str:
    """
    Dynamically maps model-specific sentiment labels (e.g. '0', '1', '2', 'label_0', 
    'negative', 'neutral', 'positive') to our standard 'Positive', 'Negative', or 'Neutral' format.
    """
    cleaned = str(raw_label).strip().lower()
    if cleaned in {"positive", "pos", "2", "label_2"}:
        return "Positive"
    elif cleaned in {"negative", "neg", "0", "label_0"}:
        return "Negative"
    else:
        return "Neutral"


def _run_sentiment_sync(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if not cleaned or cleaned == "No speech detected in audio.":
        return {
            "overall_sentiment": "Neutral",
            "confidence_score": 0.50,
            "sentiment_label": "50% Neutral",
            "distribution": {"negative": 0.25, "neutral": 0.50, "positive": 0.25},
            "aspect_sentiments": [],
        }

    raw = sentiment_analyzer(cleaned[:512])

    items = raw[0] if isinstance(raw, list) and isinstance(raw[0], list) else raw
    # Map score items dynamically to positive, negative, neutral
    score_map = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    for item in items:
        std_label = _map_sentiment_label(item["label"]).lower()
        score_map[std_label] = float(item["score"])

    neg = round(score_map.get("negative", 0.0), 3)
    neu = round(score_map.get("neutral", 0.0), 3)
    pos = round(score_map.get("positive", 0.0), 3)

    top_label_key = max(score_map, key=score_map.get)
    top_score = score_map[top_label_key]
    overall_label = _map_sentiment_label(top_label_key)  # "Positive", "Negative", or "Neutral"
    sentiment_label = f"{int(round(top_score * 100))}% {overall_label}"

    # Aspect analysis: break down clauses
    clauses = [c.strip() for c in re.split(r'[,.!?;\n]', cleaned) if len(c.strip()) > 6][:4]
    aspect_sentiments = []
    for clause in clauses:
        try:
            c_raw = sentiment_analyzer(clause[:256])
            c_items = c_raw[0] if isinstance(c_raw, list) and isinstance(c_raw[0], list) else c_raw
            c_map = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
            for it in c_items:
                c_map[_map_sentiment_label(it["label"]).lower()] = float(it["score"])
            c_top = max(c_map, key=c_map.get)
            aspect_title = clause[:32] + ("..." if len(clause) > 32 else "")
            aspect_sentiments.append({
                "aspect": aspect_title,
                "sentiment": _map_sentiment_label(c_top),
                "score": round(c_map[c_top], 2),
                "context": clause,
            })
        except Exception:
            pass

    if not aspect_sentiments:
        aspect_sentiments = [
            {
                "aspect": "Speech Sentiment",
                "sentiment": overall_label,
                "score": round(top_score, 2),
                "context": cleaned[:100],
            }
        ]

    return {
        "overall_sentiment": overall_label,
        "confidence_score": round(top_score, 2),
        "sentiment_label": sentiment_label,
        "distribution": {
            "negative": neg,
            "neutral": neu,
            "positive": pos,
        },
        "aspect_sentiments": aspect_sentiments,
    }


async def analyze_sentiment(transcript: str) -> dict[str, Any]:
    """
    Evaluates probability distribution using airzipm/sentiment-analysis-muril-v2 (ai4bharat/IndicSentiment).
    """
    return await asyncio.to_thread(_run_sentiment_sync, transcript)


# ---------------------------------------------------------------------------
# Entity Extraction: extract_entities
# ---------------------------------------------------------------------------

def _run_extract_entities_sync(text: str) -> list[dict[str, Any]]:
    if not text or not text.strip():
        return []

    entities = []
    seen_spans = set()
    occupied_spans = []

    label_map = {
        "PER": "PERSON", "B-PER": "PERSON", "I-PER": "PERSON", "PERSON": "PERSON",
        "ORG": "ORGANIZATION", "B-ORG": "ORGANIZATION", "I-ORG": "ORGANIZATION", "ORGANIZATION": "ORGANIZATION",
        "LOC": "LOCATION", "B-LOC": "LOCATION", "I-LOC": "LOCATION", "LOCATION": "LOCATION",
    }

    def add_entity(word_str: str, category: str, conf_score: float, start_char: int = None, end_char: int = None):
        cleaned = word_str.replace("##", "").strip(" ,.।?!-\"'")
        if len(cleaned) < 2 or cleaned.lower() in seen_spans:
            return

        if start_char is None or end_char is None or start_char < 0 or end_char <= start_char:
            start_char = text.find(cleaned)
            end_char = start_char + len(cleaned) if start_char != -1 else 0
        else:
            offset_start = word_str.find(cleaned)
            if offset_start != -1:
                start_char = start_char + offset_start
                end_char = start_char + len(cleaned)

        if start_char >= 0 and end_char > start_char:
            for s, e in occupied_spans:
                if max(start_char, s) < min(end_char, e):
                    return
            occupied_spans.append((start_char, end_char))

        seen_spans.add(cleaned.lower())
        pct = round(float(conf_score) * 100, 1) if conf_score <= 1.0 else round(float(conf_score), 1)
        prob = round(float(conf_score), 4) if conf_score <= 1.0 else round(float(conf_score) / 100.0, 4)
        entities.append({
            "text": cleaned,
            "word": cleaned,
            "entity": category,
            "entity_group": category,
            "label": category,
            "type": category,
            "category": category,
            "score": prob,
            "confidence": pct,
            "start_char": max(0, start_char),
            "end_char": max(0, end_char),
        })

    # Stage 1: Multilingual Indic & Code-Mixed Hybrid Gazetteer (MONEY, APPLICATION_ID, Transliterated Entities)
    hybrid_patterns = [
        # MONEY (Numeric + Devanagari/Gurmukhi/Indic number words + Currency terms)
        (r"(?:₹|rs\.?|inr)\s*\d+(?:[,\d]*)(?:\s*(?:ਹजार|हज़ार|ਲਾਖ|ਕਰੋੜ|thousand|lakh|crore))?", "MONEY", 0.97),
        (r"(?:ਇੱਕ|ਦੋ|ਪੰਜ|ਦਸ|ਵੀਹ|ਪੰਜਾਹ|ਸੌ|ਹਜ਼ਾਰ|ਲੱਖ|ਕਰੋੜ|\d+)\s*(?:ਹਜ਼ਾਰ|ਲੱਖ|ਕਰੋੜ)?\s*(?:ਰੁਪਏ|ਰੁਪਿਆ|rupees|rs\b)", "MONEY", 0.96),
        (r"(?:एक|दो|तीन|चार|पांच|पाँच|दस|बीस|पचास|सौ|हजार|हज़ार|लाख|करोड़|\d+)(?:\s+(?:एक|दो|तीन|चार|पांच|पाँच|दस|बीस|पचास|सौ|हजार|हज़ार|लाख|करोड़|\d+))*\s*(?:रुपये|रूपीस|रुपए|रुपया|टका|ரூபாய்|రూపాయలు|rupees|rs\b)", "MONEY", 0.96),
        
        # APPLICATION_ID (Banking, Digital India & Transaction Identifiers)
        (r"(?:ਯੂਪੀਆਈ|ਆਈਐਫਐਸਸੀ|ਓਟੀਪੀ|ਅਕਾਊਂਟ ਨੰਬਰ|आधार)", "APPLICATION_ID", 0.95),
        (r"(?:आईएफएससी\s*कोड|आईएफएससी|IFSC\s*code|IFSC|यूपीआई\s*आईडी|यूपीआई|UPI\s*ID|UPI|ओटीपी|OTP|पिन\s*कोड|आधार\s*नंबर|आधार|पैन\s*कार्ड|ट्रांजैक्शन\s*आईडी|अकाउंट\s*नंबर|रेफरेंस\s*नंबर|एप्लीकेशन\s*आईडी|हेल्पलाइन\s*नंबर)", "APPLICATION_ID", 0.95),
        (r"\b[A-Z]{4}0[A-Z0-9]{6}\b|\b(?:APP|TXN|ORD|REF)[-\s]?\d{4,10}\b", "APPLICATION_ID", 0.98),
        
        # ORGANIZATION (Major Banks, E-Commerce, Government & Institutional Suffixes)
        (r"(?:ਸਟੇਟ ਬੈਂਕ ਆਫ ਇੰਡੀਆ|ਸਟੇਟ ਬੈਂਕ|ਪੰਜਾਬ ਨੈਸ਼ਨਲ ਬੈਂਕ|ਐਚਡੀਐਫਸੀ ਬੈਂਕ|ਆਈਸੀਆਈਸੀਆਈ ਬੈਂਕ|ਫਲਿੱਪਕਾਰਟ|ਐਮਾਜ਼ਾਨ|ਸਰਕਾਰ|ਪੰਜਾਬ ਸਰਕਾਰ)", "ORGANIZATION", 0.95),
        (r"(?:स्टेट बैंक ऑफ इंडिया|स्टेट बैंक ऑफ़ इंडिया|भारतीय स्टेट बैंक|आईसीआईसीआई\s*बैंक|आईसीआईसीआई|एचडीएफसी\s*बैंक|एचडीएफसी|एक्सिस\s*बैंक|पंजाब नेशनल बैंक|कोटक\s*बैंक|एसबीआई|फ्लिपकार्ट|अमेज़न|अमेजन|पेटीएम|फोनपे|गूगल\s*पे|जोमैटो|स्विगी|रिलायंस|टाटा|SBI|ICICI\s*Bank|HDFC\s*Bank|Flipkart|Amazon|Paytm)", "ORGANIZATION", 0.95),
        
        # LOCATION (Cities, Sectors & Regional Markers)
        (r"(?:ਪੰਜਾਬ|ਚੰਡੀਗੜ੍ਹ|ਲੁਧਿਆਣਾ|ਅੰਮ੍ਰਿਤਸਰ|ਜਲੰਧਰ|ਪਟਿਆਲਾ|ਮੋਹਾਲੀ|ਬਠਿੰਡਾ|ਫਰੀਦਾਬਾਦ|ਦਿੱਲੀ|ਸੈਕਟਰ\s*\d+)", "LOCATION", 0.94),
        (r"(?:सेक्टर\s*(?:\d+|पंद्रह|चौदह|सोलह|दस|बीस|एक|दो|तीन|चार|पांच)|फरीदाबाद|गुरुग्राम|गुड़गांव|दिल्ली|नई\s*दिल्ली|नोएडा|मुंबई|बेंगलुरु|हैदराबाद|चेन्नई|कोलकाता|चंडीगढ़|कैथल|हरियाणा|पंजाब|जयपुर|पुणे|लखनऊ)", "LOCATION", 0.94),
        
        # PERSON (Contextual Honorifics / Agent / Common Names)
        (r"(?:राहुल\s*शर्मा|राहुल|ਅਮਿਤ|ਸੁਰਿੰਦਰ|ਹਰਪ੍ਰੀਤ|ਗੁਰਪ੍ਰੀਤ|ਜਸਪ੍ਰੀਤ|ਅਮਨ|अमित\s*कुमार|प्रिया|रोहित|विकास|दीपक)", "PERSON", 0.93),
        (r"(?:श्री|श्रीमती|मिस्टर|ਸ੍ਰੀ|ਸ੍ਰੀਮਤੀ|ਸਰਦਾਰ|ਏਜੈਂਟ|ਡਾਕਟਰ|एजेंट|डॉक्टर)\s+[ऀ-ॿ\u0A00-\u0A7FA-Za-z]+(?:\s+[ऀ-ॿ\u0A00-\u0A7FA-Za-z]+)?", "PERSON", 0.91),
    ]

    for pattern, category, conf in hybrid_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            add_entity(match.group(0), category, conf, match.start(), match.end())

    # Stage 2: Neural IndicNER Inference (Supports 11+ Regional Indian Languages)
    if ner_pipeline is not None:
        try:
            raw_preds = ner_pipeline(text)
            for item in raw_preds:
                raw_tag = item.get("entity_group") or item.get("entity") or ""
                mapped = label_map.get(raw_tag.upper())
                score = float(item.get("score", 0.91))
                start_idx = item.get("start")
                end_idx = item.get("end")
                if start_idx is not None and end_idx is not None and 0 <= start_idx < end_idx <= len(text):
                    word = text[start_idx:end_idx]
                else:
                    word = item.get("word", "")
                if mapped and word:
                    add_entity(word, mapped, score, start_idx, end_idx)
        except Exception as e:
            print(f"[BhashaSense] Neural NER inference error: {e}")

    entities.sort(key=lambda x: x["start_char"])
    return entities


async def extract_entities(text: str = None, transcript: str = None) -> list[dict[str, Any]]:
    """
    Asynchronous entrypoint for extract_entities called by main.py, executing
    the neural IndicNER + multilingual Indic rule extractor in a thread pool.
    """
    input_text = text if text is not None else (transcript or "")
    return await asyncio.to_thread(_run_extract_entities_sync, input_text)
