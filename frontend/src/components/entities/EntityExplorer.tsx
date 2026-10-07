import React, { useState } from 'react';
import { Tag, Search, ExternalLink } from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';
import { EntityCategory } from '../../types/entity.types';

export const EntityExplorer: React.FC = () => {
  const { pipelineResult, setCurrentTime, setIsPlaying } = useAppStore();
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<EntityCategory | 'ALL'>('ALL');

  if (!pipelineResult) return null;

  const { entities, entity_summary } = pipelineResult;

  const categoryBadgeColors: Record<EntityCategory, string> = {
    ORGANIZATION: 'bg-indigo-500/10 text-indigo-400 border-indigo-500/30',
    PRODUCT: 'bg-sky-500/10 text-sky-400 border-sky-500/30',
    ISSUE_TYPE: 'bg-rose-500/10 text-rose-400 border-rose-500/30',
    DATE_TIME: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
    MONEY: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
    PERSON: 'bg-purple-500/10 text-purple-400 border-purple-500/30',
    LOCATION: 'bg-orange-500/10 text-orange-400 border-orange-500/30',
    OTHER: 'bg-slate-700 text-slate-300 border-slate-600',
  };

  const filteredEntities = entities.filter((ent) => {
    const matchesSearch =
      ent.text.toLowerCase().includes(searchQuery.toLowerCase()) ||
      ent.context_sentence?.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = selectedCategory === 'ALL' || ent.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  const jumpToTimestamp = (startTime?: number) => {
    if (startTime === undefined) return;
    const audioEl = document.querySelector('audio');
    if (audioEl) {
      audioEl.currentTime = startTime;
      setCurrentTime(startTime);
      audioEl.play().catch(console.error);
      setIsPlaying(true);
    }
  };

  return (
    <div className="bg-slate-800/40 border border-slate-700/60 rounded-2xl p-6 backdrop-blur-sm shadow-xl mt-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
        <div>
          <h3 className="text-base font-semibold text-white flex items-center gap-2">
            <Tag className="h-5 w-5 text-sky-400" />
            Extracted Named Entities & Taxonomy (spaCy + IndicBERT)
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Domain entities identified across code-mixed conversation spans
          </p>
        </div>

        <div className="relative">
          <Search className="h-4 w-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search entities or context..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full sm:w-64 pl-9 pr-3 py-1.5 rounded-xl bg-slate-900 border border-slate-700 text-xs text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-indigo-500"
          />
        </div>
      </div>

      {/* Category Summary Pills */}
      <div className="flex flex-wrap gap-2 mb-6">
        <button
          onClick={() => setSelectedCategory('ALL')}
          className={`px-3 py-1 rounded-lg text-xs font-medium border transition-colors ${
            selectedCategory === 'ALL'
              ? 'bg-indigo-600 text-white border-indigo-500'
              : 'bg-slate-900/60 text-slate-400 border-slate-800 hover:border-slate-700'
          }`}
        >
          All Entities ({entities.length})
        </button>

        {entity_summary.map((group) => (
          <button
            key={group.category}
            onClick={() => setSelectedCategory(group.category)}
            className={`px-3 py-1 rounded-lg text-xs font-medium border transition-colors ${
              selectedCategory === group.category
                ? 'bg-indigo-600 text-white border-indigo-500'
                : 'bg-slate-900/60 text-slate-400 border-slate-800 hover:border-slate-700'
            }`}
          >
            {group.category} ({group.count})
          </button>
        ))}
      </div>

      {/* Entities Table */}
      <div className="overflow-x-auto rounded-xl border border-slate-800">
        <table className="w-full text-left text-xs">
          <thead className="bg-slate-900/80 text-slate-400 uppercase font-semibold text-[10px] tracking-wider border-b border-slate-800">
            <tr>
              <th className="px-4 py-3">Entity Token</th>
              <th className="px-4 py-3">Category</th>
              <th className="px-4 py-3">Context Sentence</th>
              <th className="px-4 py-3">Confidence</th>
              <th className="px-4 py-3 text-right">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 bg-slate-900/30">
            {filteredEntities.length === 0 ? (
              <tr>
                <td colSpan={5} className="px-4 py-6 text-center text-slate-500">
                  No entities found matching criteria.
                </td>
              </tr>
            ) : (
              filteredEntities.map((ent) => (
                <tr key={ent.id} className="hover:bg-slate-800/40 transition-colors">
                  <td className="px-4 py-3 font-semibold text-white">
                    {ent.text}
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className={`inline-block px-2 py-0.5 rounded-md border text-[10px] font-semibold ${
                        categoryBadgeColors[ent.category] || categoryBadgeColors.OTHER
                      }`}
                    >
                      {ent.category}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-slate-300 max-w-xs truncate" title={ent.context_sentence}>
                    {ent.context_sentence || '-'}
                  </td>
                  <td className="px-4 py-3 text-slate-400 font-mono">
                    {Math.round(ent.confidence * 100)}%
                  </td>
                  <td className="px-4 py-3 text-right">
                    {ent.start_time !== undefined && (
                      <button
                        onClick={() => jumpToTimestamp(ent.start_time)}
                        className="inline-flex items-center gap-1 text-[11px] text-indigo-400 hover:text-indigo-300 font-medium"
                      >
                        Play <ExternalLink className="h-3 w-3" />
                      </button>
                    )}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
