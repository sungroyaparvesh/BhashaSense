import math
import wave
from pathlib import Path
from typing import List, Tuple, Any
import numpy as np

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    sf = None
    HAS_SOUNDFILE = False

try:
    import torch
    import torchaudio
    import torchaudio.transforms as T
    HAS_TORCHAUDIO = True
except ImportError:
    torch = None
    torchaudio = None
    T = None
    HAS_TORCHAUDIO = False

from app.core.exceptions import AudioProcessingError
from app.core.logging import logger

TARGET_SAMPLE_RATE = 16000

class AudioPreprocessor:
    """
    Standardizes input audio for Speech-to-Text and downstream pipeline:
    - Resampling to 16 kHz
    - Mono channel conversion
    - Dynamic peak generation for UI waveforms
    """

    @staticmethod
    def load_and_resample(file_path: Path) -> Tuple[Any, int, float]:
        """
        Loads an audio file and resamples it to 16,000 Hz mono.
        Returns:
            waveform (Tensor or ndarray): Shape [1, num_samples]
            sample_rate (int): 16000
            duration (float): in seconds
        """
        try:
            if HAS_TORCHAUDIO:
                try:
                    waveform, sr = torchaudio.load(str(file_path))
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    if sr != TARGET_SAMPLE_RATE:
                        resampler = T.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
                        waveform = resampler(waveform)
                        sr = TARGET_SAMPLE_RATE
                    duration = waveform.shape[1] / float(sr)
                    return waveform, sr, duration
                except Exception:
                    pass

            # Fallback to soundfile or standard wave
            if HAS_SOUNDFILE:
                data, sr = sf.read(str(file_path), dtype="float32")
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)
                duration = len(data) / float(sr)
                # Resample using simple interpolation if needed
                if sr != TARGET_SAMPLE_RATE and len(data) > 0:
                    num_target = int(len(data) * TARGET_SAMPLE_RATE / sr)
                    indices = np.linspace(0, len(data) - 1, num_target)
                    data = np.interp(indices, np.arange(len(data)), data)
                    sr = TARGET_SAMPLE_RATE
                waveform = torch.from_numpy(data).unsqueeze(0) if torch is not None else np.expand_dims(data, 0)
                return waveform, sr, duration

            # Fallback to standard library wave
            with wave.open(str(file_path), "rb") as wf:
                sr = wf.getframerate()
                num_frames = wf.getnframes()
                raw_data = wf.readframes(num_frames)
                data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                duration = len(data) / float(sr)
                waveform = torch.from_numpy(data).unsqueeze(0) if torch is not None else np.expand_dims(data, 0)
                return waveform, sr, duration

        except Exception as e:
            logger.error(f"Failed to load and preprocess audio at {file_path}: {e}")
            raise AudioProcessingError(f"Audio preprocessing failed: {str(e)}")

    @staticmethod
    def generate_waveform_peaks(waveform: Any, num_peaks: int = 150) -> List[float]:
        """
        Generates normalized downsampled amplitude peaks between 0.0 and 1.0
        for high-performance UI waveform rendering in the frontend.
        """
        try:
            if hasattr(waveform, "squeeze"):
                waveform = waveform.squeeze()
            if hasattr(waveform, "cpu"):
                samples = waveform.cpu().numpy()
            else:
                samples = np.asarray(waveform).flatten()

            total_samples = len(samples)
            if total_samples == 0:
                return [0.0] * num_peaks

            chunk_size = max(1, math.floor(total_samples / num_peaks))
            peaks = []

            for i in range(num_peaks):
                start = i * chunk_size
                end = min(start + chunk_size, total_samples)
                if start >= total_samples:
                    peaks.append(0.0)
                else:
                    chunk = samples[start:end]
                    peak = float(np.max(np.abs(chunk)))
                    peaks.append(round(peak, 4))

            # Normalize peaks to max 1.0 if not flat
            max_val = max(peaks) if peaks else 0.0
            if max_val > 0:
                peaks = [round(p / max_val, 4) for p in peaks]

            return peaks
        except Exception as e:
            logger.warning(f"Could not generate waveform peaks: {e}")
            return [0.1] * num_peaks

    @staticmethod
    def save_wav(waveform: Any, sample_rate: int, output_path: Path) -> None:
        """Saves tensor or numpy waveform to standard 16-bit PCM WAV."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if HAS_TORCHAUDIO and hasattr(waveform, "cpu"):
                torchaudio.save(str(output_path), waveform, sample_rate)
            elif HAS_SOUNDFILE:
                arr = waveform.squeeze().cpu().numpy() if hasattr(waveform, "cpu") else np.asarray(waveform).squeeze()
                sf.write(str(output_path), arr, sample_rate, subtype="PCM_16")
            else:
                arr = (np.asarray(waveform).flatten() * 32767.0).astype(np.int16)
                with wave.open(str(output_path), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(arr.tobytes())
        except Exception as e:
            logger.error(f"Failed to save processed wav to {output_path}: {e}")
            raise AudioProcessingError(f"Saving processed audio failed: {str(e)}")
