import { create } from 'zustand';
import { AudioUploadResponse } from '../types/audio.types';
import { PipelineResponse } from '../types/pipeline.types';
import { ExtractedEntity } from '../types/entity.types';
import { SentimentType } from '../types/sentiment.types';

interface AppState {
  // Audio state
  activeAudio: AudioUploadResponse | null;
  audioUrl: string | null;
  isUploading: boolean;
  uploadError: string | null;

  // Playback state
  isPlaying: boolean;
  currentTime: number;
  duration: number;

  // Pipeline Analysis state
  isAnalyzing: boolean;
  analysisError: string | null;
  pipelineResult: PipelineResponse | null;

  // Filters & Interactivity
  selectedEntity: ExtractedEntity | null;
  sentimentFilter: SentimentType | 'all';
  activeSegmentId: number | null;

  // Actions
  setActiveAudio: (audio: AudioUploadResponse | null, url: string | null) => void;
  setIsUploading: (val: boolean) => void;
  setUploadError: (err: string | null) => void;
  setIsPlaying: (val: boolean) => void;
  setCurrentTime: (time: number) => void;
  setDuration: (dur: number) => void;
  setIsAnalyzing: (val: boolean) => void;
  setAnalysisError: (err: string | null) => void;
  setPipelineResult: (res: PipelineResponse | null) => void;
  setSelectedEntity: (ent: ExtractedEntity | null) => void;
  setSentimentFilter: (filter: SentimentType | 'all') => void;
  setActiveSegmentId: (id: number | null) => void;
  resetAll: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  activeAudio: null,
  audioUrl: null,
  isUploading: false,
  uploadError: null,

  isPlaying: false,
  currentTime: 0,
  duration: 0,

  isAnalyzing: false,
  analysisError: null,
  pipelineResult: null,

  selectedEntity: null,
  sentimentFilter: 'all',
  activeSegmentId: null,

  setActiveAudio: (activeAudio, audioUrl) =>
    set({
      activeAudio,
      audioUrl,
      pipelineResult: null,
      currentTime: 0,
      isPlaying: false,
      uploadError: null,
      analysisError: null,
    }),

  setIsUploading: (isUploading) => set({ isUploading }),
  setUploadError: (uploadError) => set({ uploadError }),
  setIsPlaying: (isPlaying) => set({ isPlaying }),
  setCurrentTime: (currentTime) => set({ currentTime }),
  setDuration: (duration) => set({ duration }),
  setIsAnalyzing: (isAnalyzing) => set({ isAnalyzing }),
  setAnalysisError: (analysisError) => set({ analysisError }),
  setPipelineResult: (pipelineResult) => set({ pipelineResult, isAnalyzing: false }),
  setSelectedEntity: (selectedEntity) => set({ selectedEntity }),
  setSentimentFilter: (sentimentFilter) => set({ sentimentFilter }),
  setActiveSegmentId: (activeSegmentId) => set({ activeSegmentId }),

  resetAll: () =>
    set({
      activeAudio: null,
      audioUrl: null,
      isPlaying: false,
      currentTime: 0,
      duration: 0,
      pipelineResult: null,
      selectedEntity: null,
      sentimentFilter: 'all',
      activeSegmentId: null,
      uploadError: null,
      analysisError: null,
    }),
}));
