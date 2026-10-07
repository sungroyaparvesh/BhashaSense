import { apiClient } from './api';
import { PipelineResponse } from '../types/pipeline.types';

export interface PipelineRunOptions {
  detect_aspects?: boolean;
  extract_entities?: boolean;
}

export const pipelineService = {
  /**
   * Executes synchronous end-to-end pipeline processing.
   */
  async runPipeline(fileId: string, options?: PipelineRunOptions): Promise<PipelineResponse> {
    const response = await apiClient.post<PipelineResponse>('/pipeline/run', {
      file_id: fileId,
      options: options || {
        detect_aspects: true,
        extract_entities: true,
      },
    });
    return response.data;
  },

  /**
   * Exports pipeline results to json or csv.
   */
  async exportReport(jobId: string, format: 'json' | 'segments-csv' | 'entities-csv'): Promise<Blob> {
    const response = await apiClient.get(`/jobs/${jobId}/export/${format}`, {
      responseType: 'blob',
    });
    return response.data;
  },
};
