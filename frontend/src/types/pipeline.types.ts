import { AudioMetadata } from './audio.types';
import { SentimentSummary, SegmentSentiment } from './sentiment.types';
import { ExtractedEntity, EntityGroupSummary } from './entity.types';

export interface WordAlignment {
  word: string;
  start_time: number;
  end_time: number;
  confidence: number;
}

export interface TranscriptSegment {
  id: number;
  text: string;
  start_time: number;
  end_time: number;
  confidence: number;
  language_code: string;
  words?: WordAlignment[];
}

export interface PipelineMetrics {
  audio_duration_seconds: number;
  total_processing_time_ms: number;
  asr_time_ms: number;
  sentiment_time_ms: number;
  ner_time_ms: number;
  device_used: string;
}

export interface PipelineResponse {
  job_id: string;
  status: 'completed' | 'failed' | 'pending';
  audio_metadata: AudioMetadata;
  full_transcript: string;
  detected_language: string;
  segments: TranscriptSegment[];
  sentiment_summary: SentimentSummary;
  segment_sentiments: SegmentSentiment[];
  entities: ExtractedEntity[];
  entity_summary: EntityGroupSummary[];
  metrics: PipelineMetrics;
  created_at: string;
}
