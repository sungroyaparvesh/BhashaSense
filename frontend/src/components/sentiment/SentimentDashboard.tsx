import React from 'react';
import { BarChart3, TrendingUp, Filter, ThumbsUp, ThumbsDown, Minus } from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';
import { SentimentType } from '../../types/sentiment.types';

export const SentimentDashboard: React.FC = () => {
  const { pipelineResult, sentimentFilter, setSentimentFilter } = useAppStore();

  if (!pipelineResult) return null;

  const { sentiment_summary } = pipelineResult;
  const { overall_sentiment, distribution, sentiment_score_timeline, aspect_breakdown } = sentiment_summary;

  const sentimentIcons = {
    positive: <ThumbsUp className="h-5 w-5 text-emerald-400" />,
    neutral: <Minus className="h-5 w-5 text-slate-400" />,
    negative: <ThumbsDown className="h-5 w-5 text-rose-400" />,
  };

  const sentimentBorders = {
    positive: 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300',
    neutral: 'border-slate-600 bg-slate-800/40 text-slate-300',
    negative: 'border-rose-500/30 bg-rose-500/10 text-rose-300',
  };

  return (
    <div className="bg-slate-800/40 border border-slate-700/60 rounded-2xl p-6 backdrop-blur-sm shadow-xl mt-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
        <div>
          <h3 className="text-base font-semibold text-white flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-purple-400" />
            Sentiment & Aspect Breakdown (IndicBERT)
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Fine-grained sentiment classification tailored for code-mixed speech
          </p>
        </div>

        <div className="flex items-center gap-1.5 p-1 rounded-xl bg-slate-900 border border-slate-800 self-start sm:self-auto">
          <span className="text-[11px] font-medium text-slate-400 px-2 flex items-center gap-1">
            <Filter className="h-3 w-3" /> Filter:
          </span>
          {(['all', 'positive', 'neutral', 'negative'] as (SentimentType | 'all')[]).map((tab) => (
            <button
              key={tab}
              onClick={() => setSentimentFilter(tab)}
              className={`px-2.5 py-1 rounded-lg text-xs font-medium capitalize transition-all ${
                sentimentFilter === tab
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {/* Overall Polarity Card */}
        <div className={`p-4 rounded-xl border ${sentimentBorders[overall_sentiment]} flex items-center gap-4`}>
          <div className="p-3 rounded-xl bg-slate-900/50 shadow-inner">
            {sentimentIcons[overall_sentiment]}
          </div>
          <div>
            <span className="text-xs text-slate-400 uppercase tracking-wider font-semibold block">
              Overall Sentiment
            </span>
            <span className="text-lg font-bold capitalize text-white">
              {overall_sentiment}
            </span>
            <span className="text-xs text-slate-400 block mt-0.5">
              Confidence: {Math.round(sentiment_summary.confidence * 100)}%
            </span>
          </div>
        </div>

        {/* Polarity Distribution */}
        <div className="md:col-span-2 p-4 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-col justify-center">
          <span className="text-xs text-slate-400 font-semibold uppercase tracking-wider mb-2 block">
            Polarity Distribution
          </span>

          <div className="space-y-2">
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-emerald-400 font-medium">Positive</span>
                <span className="text-slate-400">{Math.round(distribution.positive * 100)}%</span>
              </div>
              <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
                <div
                  className="h-full bg-emerald-500 rounded-full transition-all duration-500"
                  style={{ width: `${distribution.positive * 100}%` }}
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-400 font-medium">Neutral</span>
                <span className="text-slate-400">{Math.round(distribution.neutral * 100)}%</span>
              </div>
              <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
                <div
                  className="h-full bg-slate-400 rounded-full transition-all duration-500"
                  style={{ width: `${distribution.neutral * 100}%` }}
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-rose-400 font-medium">Negative</span>
                <span className="text-slate-400">{Math.round(distribution.negative * 100)}%</span>
              </div>
              <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
                <div
                  className="h-full bg-rose-500 rounded-full transition-all duration-500"
                  style={{ width: `${distribution.negative * 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Sentiment Trajectory Timeline */}
      {sentiment_score_timeline && sentiment_score_timeline.length > 0 && (
        <div className="mt-4 pt-4 border-t border-slate-700/60">
          <div className="flex items-center gap-2 mb-3">
            <TrendingUp className="h-4 w-4 text-indigo-400" />
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Sentiment Shift Trajectory
            </span>
          </div>
          <div className="flex items-center gap-1.5 h-10 bg-slate-900/60 p-2 rounded-xl border border-slate-800">
            {sentiment_score_timeline.map((point, idx) => (
              <div
                key={idx}
                className="flex-1 h-full rounded flex items-center justify-center text-[10px] text-white font-mono opacity-80 hover:opacity-100 transition-opacity cursor-default"
                style={{ backgroundColor: point.sentiment === 'positive' ? '#10b981' : point.sentiment === 'negative' ? '#ef4444' : '#64748b' }}
                title={`Time ${point.time}s: ${point.sentiment} (Score: ${point.score})`}
              >
                {point.time}s
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Aspect Sentiment Pills */}
      {aspect_breakdown && aspect_breakdown.length > 0 && (
        <div className="mt-4 pt-4 border-t border-slate-700/60">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block mb-3">
            Extracted Aspects & Topics
          </span>
          <div className="flex flex-wrap gap-2">
            {aspect_breakdown.map((asp, idx) => (
              <div
                key={idx}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-700/80 text-xs"
              >
                <span className="font-medium text-slate-200 capitalize">
                  {asp.aspect.replace('_', ' ')}
                </span>
                <span
                  className={`text-[10px] font-semibold px-1.5 py-0.5 rounded capitalize ${
                    asp.sentiment === 'positive'
                      ? 'bg-emerald-500/20 text-emerald-400'
                      : asp.sentiment === 'negative'
                      ? 'bg-rose-500/20 text-rose-400'
                      : 'bg-slate-700 text-slate-300'
                  }`}
                >
                  {asp.sentiment}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
