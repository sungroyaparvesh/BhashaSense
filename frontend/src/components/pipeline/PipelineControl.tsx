import React from 'react';
import { Play, Loader2, Zap, Clock, Cpu, CheckCircle } from 'lucide-react';
import { pipelineService } from '../../services/pipelineService';
import { useAppStore } from '../../store/useAppStore';

export const PipelineControl: React.FC = () => {
  const {
    activeAudio,
    isAnalyzing,
    analysisError,
    pipelineResult,
    setIsAnalyzing,
    setAnalysisError,
    setPipelineResult,
  } = useAppStore();

  const handleRunPipeline = async () => {
    if (!activeAudio) return;

    try {
      setIsAnalyzing(true);
      setAnalysisError(null);

      const result = await pipelineService.runPipeline(activeAudio.file_id);
      setPipelineResult(result);
    } catch (err: any) {
      setAnalysisError(err.message || 'Pipeline analysis failed.');
      setIsAnalyzing(false);
    }
  };

  if (!activeAudio) return null;

  return (
    <div className="bg-slate-800/40 border border-slate-700/60 rounded-2xl p-6 backdrop-blur-sm shadow-xl mt-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h3 className="text-base font-semibold text-white flex items-center gap-2">
            <Zap className="h-5 w-5 text-indigo-400" />
            End-to-End Extraction Pipeline
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Execute torchaudio resampling, IndicConformer ASR, IndicBERT Sentiment, and spaCy NER in a unified pass.
          </p>
        </div>

        <button
          onClick={handleRunPipeline}
          disabled={isAnalyzing}
          className="flex items-center justify-center gap-2 px-6 py-3 rounded-xl font-semibold text-sm bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 text-white shadow-lg shadow-indigo-500/25 transition-all duration-200 disabled:opacity-50 active:scale-95 shrink-0"
        >
          {isAnalyzing ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>Analyzing Speech...</span>
            </>
          ) : (
            <>
              <Play className="h-4 w-4 fill-white" />
              <span>Run BhashaSense Pipeline</span>
            </>
          )}
        </button>
      </div>

      {isAnalyzing && (
        <div className="mt-6 p-4 rounded-xl bg-slate-900/60 border border-slate-800">
          <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
            <span className="font-medium text-indigo-400 animate-pulse">
              Pipeline actively executing forward pass...
            </span>
            <span>Step-by-step telemetry</span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
            <div className="p-2 rounded bg-slate-800/80 border border-slate-700 text-slate-300">
              1. 16kHz Resampling
            </div>
            <div className="p-2 rounded bg-indigo-500/20 border border-indigo-500/30 text-indigo-300">
              2. IndicConformer ASR
            </div>
            <div className="p-2 rounded bg-purple-500/20 border border-purple-500/30 text-purple-300">
              3. IndicBERT Sentiment
            </div>
            <div className="p-2 rounded bg-sky-500/20 border border-sky-500/30 text-sky-300">
              4. spaCy Hybrid NER
            </div>
          </div>
        </div>
      )}

      {analysisError && (
        <div className="mt-4 p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs">
          {analysisError}
        </div>
      )}

      {pipelineResult && !isAnalyzing && (
        <div className="mt-6 p-4 rounded-xl bg-slate-900/80 border border-slate-800 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-2 text-xs text-emerald-400 font-medium">
            <CheckCircle className="h-4 w-4" />
            <span>Inference Completed Successfully</span>
          </div>

          <div className="flex flex-wrap items-center gap-4 text-xs text-slate-400">
            <span className="flex items-center gap-1">
              <Clock className="h-3.5 w-3.5 text-slate-500" />
              Latency: <strong className="text-slate-200">{pipelineResult.metrics.total_processing_time_ms}ms</strong>
            </span>
            <span className="flex items-center gap-1">
              ASR: <strong className="text-slate-200">{pipelineResult.metrics.asr_time_ms}ms</strong>
            </span>
            <span className="flex items-center gap-1">
              Sentiment: <strong className="text-slate-200">{pipelineResult.metrics.sentiment_time_ms}ms</strong>
            </span>
            <span className="flex items-center gap-1">
              NER: <strong className="text-slate-200">{pipelineResult.metrics.ner_time_ms}ms</strong>
            </span>
            <span className="flex items-center gap-1">
              <Cpu className="h-3.5 w-3.5 text-indigo-400" />
              Device: <strong className="text-indigo-300">{pipelineResult.metrics.device_used}</strong>
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
