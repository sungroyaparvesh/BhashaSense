import React, { useState } from 'react';
import { Download, FileJson, FileSpreadsheet, Loader2 } from 'lucide-react';
import { pipelineService } from '../../services/pipelineService';
import { useAppStore } from '../../store/useAppStore';

export const ExportModal: React.FC = () => {
  const { pipelineResult } = useAppStore();
  const [downloadingFormat, setDownloadingFormat] = useState<string | null>(null);

  if (!pipelineResult) return null;

  const handleExport = async (format: 'json' | 'segments-csv' | 'entities-csv') => {
    try {
      setDownloadingFormat(format);
      const blob = await pipelineService.exportReport(pipelineResult.job_id, format);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `bhashasense_${pipelineResult.job_id}_${format}.${format.includes('csv') ? 'csv' : 'json'}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      console.error('Export failed:', err);
    } finally {
      setDownloadingFormat(null);
    }
  };

  return (
    <div className="bg-slate-800/40 border border-slate-700/60 rounded-2xl p-6 backdrop-blur-sm shadow-xl mt-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h3 className="text-base font-semibold text-white flex items-center gap-2">
            <Download className="h-5 w-5 text-indigo-400" />
            Enterprise Export & Telemetry
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Export unified code-mixed transcripts, sentiment distributions, and extracted entity data
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={() => handleExport('json')}
            disabled={downloadingFormat !== null}
            className="flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-medium bg-slate-900 border border-slate-700 hover:border-slate-600 text-slate-200 transition-colors"
          >
            {downloadingFormat === 'json' ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <FileJson className="h-3.5 w-3.5 text-indigo-400" />
            )}
            Full JSON Report
          </button>

          <button
            onClick={() => handleExport('segments-csv')}
            disabled={downloadingFormat !== null}
            className="flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-medium bg-slate-900 border border-slate-700 hover:border-slate-600 text-slate-200 transition-colors"
          >
            {downloadingFormat === 'segments-csv' ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <FileSpreadsheet className="h-3.5 w-3.5 text-emerald-400" />
            )}
            Transcript & Sentiment CSV
          </button>

          <button
            onClick={() => handleExport('entities-csv')}
            disabled={downloadingFormat !== null}
            className="flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-medium bg-slate-900 border border-slate-700 hover:border-slate-600 text-slate-200 transition-colors"
          >
            {downloadingFormat === 'entities-csv' ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <FileSpreadsheet className="h-3.5 w-3.5 text-sky-400" />
            )}
            Entities CSV
          </button>
        </div>
      </div>
    </div>
  );
};
