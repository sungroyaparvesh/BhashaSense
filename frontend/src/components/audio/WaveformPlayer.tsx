import React, { useRef, useEffect } from 'react';
import { Play, Pause, RotateCcw, Volume2 } from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';

export const WaveformPlayer: React.FC = () => {
  const {
    activeAudio,
    audioUrl,
    isPlaying,
    currentTime,
    duration,
    setIsPlaying,
    setCurrentTime,
    setDuration,
  } = useAppStore();

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (activeAudio?.metadata.duration_seconds) {
      setDuration(activeAudio.metadata.duration_seconds);
    }
  }, [activeAudio, setDuration]);

  // Audio element sync
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handleTimeUpdate = () => {
      setCurrentTime(audio.currentTime);
    };

    const handleEnded = () => {
      setIsPlaying(false);
      setCurrentTime(0);
    };

    audio.addEventListener('timeupdate', handleTimeUpdate);
    audio.addEventListener('ended', handleEnded);

    return () => {
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      audio.removeEventListener('ended', handleEnded);
    };
  }, [setCurrentTime, setIsPlaying]);

  // Play / Pause toggle
  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
      setIsPlaying(false);
    } else {
      audio.play().then(() => setIsPlaying(true)).catch(console.error);
    }
  };

  const handleRestart = () => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = 0;
    setCurrentTime(0);
    audio.play();
    setIsPlaying(true);
  };

  // Draw interactive waveform canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !activeAudio) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, width, height);

    const peaks = activeAudio.waveform_peaks || [];
    const barCount = peaks.length;
    if (barCount === 0) return;

    const barWidth = Math.max(2, (width / barCount) - 2);
    const progressPercent = duration > 0 ? currentTime / duration : 0;

    for (let i = 0; i < barCount; i++) {
      const x = i * (width / barCount);
      const peakVal = Math.max(0.1, peaks[i]);
      const barHeight = peakVal * (height * 0.85);
      const y = (height - barHeight) / 2;

      // Color bars based on played status
      const barProgress = i / barCount;
      if (barProgress <= progressPercent) {
        ctx.fillStyle = '#6366f1'; // Indigo-500 for played
      } else {
        ctx.fillStyle = '#334155'; // Slate-700 for unplayed
      }

      ctx.beginPath();
      ctx.roundRect(x, y, barWidth, barHeight, 2);
      ctx.fill();
    }
  }, [activeAudio, currentTime, duration]);

  // Handle clicking on waveform to seek
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const audio = audioRef.current;
    if (!canvas || !audio || duration <= 0) return;

    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, clickX / rect.width));
    const seekTime = ratio * duration;

    audio.currentTime = seekTime;
    setCurrentTime(seekTime);
  };

  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  if (!activeAudio) return null;

  return (
    <div className="bg-slate-800/40 border border-slate-700/60 rounded-2xl p-6 backdrop-blur-sm shadow-xl mt-6">
      <audio ref={audioRef} src={audioUrl || ''} preload="metadata" />

      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <button
            onClick={togglePlay}
            className="h-12 w-12 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white flex items-center justify-center shadow-lg shadow-indigo-600/30 transition-all duration-200 active:scale-95"
          >
            {isPlaying ? <Pause className="h-6 w-6" /> : <Play className="h-6 w-6 ml-0.5" />}
          </button>

          <button
            onClick={handleRestart}
            className="h-10 w-10 rounded-lg bg-slate-700/60 hover:bg-slate-700 text-slate-300 flex items-center justify-center transition-colors"
            title="Restart playback"
          >
            <RotateCcw className="h-4 w-4" />
          </button>

          <div>
            <span className="text-sm font-semibold text-slate-200 block">
              {activeAudio.filename}
            </span>
            <div className="text-xs text-slate-400 flex items-center gap-2 mt-0.5">
              <span>{formatTime(currentTime)}</span>
              <span>/</span>
              <span>{formatTime(duration)}</span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs text-slate-400">
          <Volume2 className="h-4 w-4 text-slate-400" />
          <span>Interactive Waveform</span>
        </div>
      </div>

      <div className="relative bg-slate-900/60 rounded-xl p-3 border border-slate-800">
        <canvas
          ref={canvasRef}
          onClick={handleCanvasClick}
          className="w-full h-24 cursor-pointer"
        />
        <p className="text-[10px] text-slate-500 text-center mt-1">
          Click anywhere along the waveform to scrub audio playback
        </p>
      </div>
    </div>
  );
};
