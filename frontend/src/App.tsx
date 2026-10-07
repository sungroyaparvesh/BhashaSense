import React from 'react';
import { Navbar } from './components/layout/Navbar';
import { AudioUploader } from './components/audio/AudioUploader';
import { WaveformPlayer } from './components/audio/WaveformPlayer';
import { PipelineControl } from './components/pipeline/PipelineControl';
import { TranscriptViewer } from './components/transcript/TranscriptViewer';
import { SentimentDashboard } from './components/sentiment/SentimentDashboard';
import { EntityExplorer } from './components/entities/EntityExplorer';
import { ExportModal } from './components/common/ExportModal';
import { useAppStore } from './store/useAppStore';
import { Sparkles, Activity } from 'lucide-react';

export const App: React.FC = () => {
  const { pipelineResult } = useAppStore();

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 text-slate-100">
      <Navbar />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Hero Section */}
        <div className="mb-8">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-xs font-semibold mb-3">
            <Sparkles className="h-3.5 w-3.5" />
            <span>AI4Bharat IndicConformer & IndicBERT Powered</span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white">
            Code-Mixed Audio Sentiment & Entity Intelligence
          </h1>
          <p className="text-slate-400 text-sm sm:text-base mt-2 max-w-3xl">
            Ingest multilingual and Romanized vernacular speech (Hinglish, Tamlish, Benglish), align verbatim timestamps, quantify segment polarities, and extract business entities seamlessly.
          </p>
        </div>

        {/* Audio Intake & Waveform Player */}
        <div className="grid grid-cols-1 gap-6">
          <AudioUploader />
          <WaveformPlayer />
          <PipelineControl />
        </div>

        {/* Inference Analytics & Extraction Dashboard */}
        {pipelineResult && (
          <div className="space-y-6 mt-2">
            <SentimentDashboard />

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <TranscriptViewer />
              <EntityExplorer />
            </div>

            <ExportModal />
          </div>
        )}

        {!pipelineResult && (
          <div className="mt-12 p-8 rounded-2xl border border-dashed border-slate-800 text-center">
            <div className="h-12 w-12 rounded-xl bg-slate-800/80 border border-slate-700 flex items-center justify-center mx-auto mb-3 text-slate-400">
              <Activity className="h-6 w-6" />
            </div>
            <h3 className="text-sm font-semibold text-slate-200">
              Ready for Audio Ingestion
            </h3>
            <p className="text-xs text-slate-500 mt-1 max-w-md mx-auto">
              Upload an audio file or click "Load Demo Clip" to see speech-to-text alignment, sentiment classification, and entity extraction in action.
            </p>
          </div>
        )}
      </main>

      <footer className="border-t border-slate-800/80 py-6 text-center text-xs text-slate-500">
        <p>BhashaSense Enterprise Platform • IndicConformer • IndicBERT • spaCy</p>
      </footer>
    </div>
  );
};

export default App;
