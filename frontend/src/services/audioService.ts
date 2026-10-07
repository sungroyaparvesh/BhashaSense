import { apiClient } from './api';
import { AudioUploadResponse, AudioWaveformResponse } from '../types/audio.types';

export const audioService = {
  /**
   * Uploads an audio file with multipart/form-data.
   */
  async uploadAudio(file: File): Promise<AudioUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await apiClient.post<AudioUploadResponse>('/audio/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  /**
   * Fetches waveform peak data for an uploaded audio file.
   */
  async getWaveform(fileId: string, peaks: number = 150): Promise<AudioWaveformResponse> {
    const response = await apiClient.get<AudioWaveformResponse>(`/audio/waveform/${fileId}?peaks=${peaks}`);
    return response.data;
  },

  /**
   * Generates stream URL for audio playback.
   */
  getStreamUrl(fileId: string): string {
    const baseURL = apiClient.defaults.baseURL || '/api/v1';
    return `${baseURL}/audio/stream/${fileId}`;
  },
};
