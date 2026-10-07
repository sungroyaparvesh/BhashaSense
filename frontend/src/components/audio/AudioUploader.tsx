import React, { useRef, useState } from 'react';
import { UploadCloud, Music, AlertCircle, Sparkles, CheckCircle2 } from 'lucide-react';
import { audioService } from '../../services/audioService';
import { useAppStore } from '../../store/useAppStore';

export const AudioUploader: React.FC = () => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const {
    activeAudio,
    isUploading,
    uploadError,
    setActiveAudio,
    setIsUploading,
    setUploadError,
  } = useAppStore();

  const handleFile = async (file: File) => {
    try {
      setIsUploading(true);
      setUploadError(null);

      const response = await audioService.uploadAudio(file);
      const streamUrl = audioService.getStreamUrl(response.file_id);
      setActiveAudio(response, streamUrl);
    } catch (err: any) {
      setUploadError(err.message || 'Failed to upload audio file');
    } finally {
      setIsUploading(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const loadSampleAudio = async () => {
    // Generate a synthetic sample WAV file on the fly for immediate demonstration
    const sampleRate = 16000;
    const duration = 8.0; // 8 seconds
    const numSamples = sampleRate * duration;
    const wavBuffer = new ArrayBuffer(44 + numSamples * 2);
    const view = new DataView(wavBuffer);

    // Write WAV header
    const writeString = (offset: number, str: string) => {
      for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
      }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + numSamples * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // Mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, numSamples * 2, true);

    // Audio samples (gentle frequency sweeps simulating speech rhythm)
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      const freq = 200 + 100 * Math.sin(2 * Math.PI * 2 * t);
      const sample = Math.sin(2 * Math.PI * freq * t) * 0.3;
      view.setInt16(44 + i * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    }

    const sampleBlob = new Blob([wavBuffer], { type: 'audio/wav' });
    const sampleFile = new File([sampleBlob], 'sample_hinglish_support_call.wav', { type: 'audio/wav' });
    await handleFile(sampleFile);
  };

  return (
    <div className="bg-slate-800/40 border border-slate-700/60 rounded-2xl p-6 backdrop-blur-sm shadow-xl">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <Music className="h-5 w-5 text-indigo-400" />
            Audio Intake & Preprocessing
          </h2>
          <p className="text-sm text-slate-400">
            Upload code-mixed vernacular recordings (WAV, MP3, M4A, FLAC)
          </p>
        </div>

        <button
          onClick={loadSampleAudio}
          disabled={isUploading}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-indigo-500/10 text-indigo-300 border border-indigo-500/30 hover:bg-indigo-500/20 transition-all duration-200"
        >
          <Sparkles className="h-3.5 w-3.5 text-indigo-400" />
          Load Demo Clip
        </button>
      </div>

      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200 ${
          isDragging
            ? 'border-indigo-400 bg-indigo-500/10 scale-[1.01]'
            : 'border-slate-700 hover:border-slate-500 hover:bg-slate-800/60'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg"
          className="hidden"
          onChange={(e) => {
            if (e.target.files && e.target.files[0]) {
              handleFile(e.target.files[0]);
            }
          }}
        />

        <div className="flex flex-col items-center justify-center gap-3">
          <div className="p-4 rounded-full bg-slate-800 border border-slate-700 text-indigo-400 shadow-inner">
            <UploadCloud className="h-8 w-8" />
          </div>

          <div>
            <p className="text-sm font-medium text-slate-200">
              {isUploading ? 'Uploading and preprocessing audio...' : 'Click to upload or drag & drop'}
            </p>
            <p className="text-xs text-slate-500 mt-1">
              Supports WAV, MP3, FLAC, M4A up to 50 MB (auto-resampled to 16 kHz mono)
            </p>
          </div>
        </div>
      </div>

      {uploadError && (
        <div className="mt-4 p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-300 text-sm flex items-center gap-2">
          <AlertCircle className="h-4 w-4 shrink-0" />
          <span>{uploadError}</span>
        </div>
      )}

      {activeAudio && !uploadError && (
        <div className="mt-4 p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-300 text-xs flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0" />
            <span className="font-medium text-slate-200">{activeAudio.filename}</span>
            <span className="text-slate-400">({activeAudio.metadata.duration_seconds}s • 16 kHz mono)</span>
          </div>
          <span className="px-2 py-0.5 rounded bg-emerald-500/20 font-semibold text-emerald-300">
            Ready
          </span>
        </div>
      )}
    </div>
  );
};
