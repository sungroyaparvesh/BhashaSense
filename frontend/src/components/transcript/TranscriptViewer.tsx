import React from 'react';
import { MessageSquareText, Clock, Play } from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';

export const TranscriptViewer: React.FC = () => {
  const { pipelineResult, currentTime, setCurrentTime, setIsPlaying, sentimentFilter } = useAppStore();

  if (!pipelineResult) return null;

  const sentimentMap = new Map(
    pipelineResult.segment_sentiments.map((s) => [s.segment_id, s])
  );

  const filteredSegments = pipelineResult.segments.filter((seg) => {
    if (sentimentFilter === 'all') return true;
    const sent = sentimentMap.get(seg.id);
    return sent?.sentiment === sentimentFilter;
  });

  const seekToSegment = (startTime: number) => {
    const audioEl = document.querySelector('audio');
    if (audioEl) {
      audioEl.currentTime = startTime;
      setCurrentTime(startTime);
      audioEl.play().catch(console.error);
      setIsPlaying(true);
    }
  };

  const formatSeconds = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <div className="bg-slate-800/40 border border-slate-700/60 rounded-2xl p-6 backdrop-blur-sm shadow-xl mt-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-base font-semibold text-white flex items-center gap-2">
            <MessageSquareText className="h-5 w-5 text-indigo-400" />
            Aligned Code-Mixed Transcript
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Synchronized speech segments extracted via IndicConformer
          </p>
        </div>

        <span className="text-xs px-2.5 py-1 rounded-md bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
          Detected: {pipelineResult.detected_language}
        </span>
      </div>

      <div className="space-y-3 max-h-[380px] overflow-y-auto pr-1">
        {filteredSegments.map((segment) => {
          const sent = sentimentMap.get(segment.id);
          const isActive = currentTime >= segment.start_time && currentTime <= segment.end_time;

          const sentimentColors = {
            positive: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
            neutral: 'bg-slate-700/50 text-slate-300 border-slate-600',
            negative: 'bg-rose-500/10 text-rose-400 border-rose-500/20',
          };

          return (
            <div
              key={segment.id}
              onClick={() => seekToSegment(segment.start_time)}
              className={`p-4 rounded-xl border transition-all duration-200 cursor-pointer ${
                isActive
                  ? 'bg-indigo-950/40 border-indigo-500 shadow-md shadow-indigo-500/10 scale-[1.01]'
                  : 'bg-slate-900/60 border-slate-800/80 hover:border-slate-700 hover:bg-slate-900'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="flex items-center gap-1 text-xs text-slate-400 font-mono">
                    <Clock className="h-3.5 w-3.5" />
                    {formatSeconds(segment.start_time)} - {formatSeconds(segment.end_time)}
                  </span>
                  <span className="text-[10px] uppercase font-semibold px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                    {segment.language_code}
                  </span>
                </div>

                {sent && (
                  <span
                    className={`text-xs capitalize font-medium px-2 py-0.5 rounded-full border ${
                      sentimentColors[sent.sentiment]
                    }`}
                  >
                    {sent.sentiment} ({Math.round(sent.confidence * 100)}%)
                  </span>
                )}
              </div>

              <p className="text-sm text-slate-200 leading-relaxed font-normal">
                {segment.text}
              </p>

              <div className="mt-2 flex items-center justify-between text-[11px] text-slate-500">
                <span className="flex items-center gap-1 text-indigo-400 opacity-80 hover:opacity-100">
                  <Play className="h-3 w-3 fill-indigo-400" />
                  Jump to audio
                </span>
                {sent?.aspects && sent.aspects.length > 0 && (
                  <div className="flex items-center gap-1">
                    <span className="text-slate-500">Aspects:</span>
                    {sent.aspects.map((asp, idx) => (
                      <span
                        key={idx}
                        className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 font-mono text-[10px]"
                      >
                        {asp.aspect}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
