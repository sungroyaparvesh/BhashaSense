export type SentimentType = 'positive' | 'neutral' | 'negative';

export interface SentimentDistribution {
  positive: number;
  neutral: number;
  negative: number;
}

export interface AspectSentiment {
  aspect: string;
  sentiment: SentimentType;
  score: number;
  mention_span?: string;
}

export interface SegmentSentiment {
  segment_id: number;
  text: string;
  sentiment: SentimentType;
  confidence: number;
  distribution: SentimentDistribution;
  aspects: AspectSentiment[];
}

export interface SentimentTimelinePoint {
  time: number;
  score: number;
  sentiment: SentimentType;
}

export interface SentimentSummary {
  overall_sentiment: SentimentType;
  confidence: number;
  distribution: SentimentDistribution;
  sentiment_score_timeline: SentimentTimelinePoint[];
  aspect_breakdown: AspectSentiment[];
}
