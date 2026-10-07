export interface AudioMetadata {
  file_id: string;
  original_filename: string;
  stored_path: string;
  duration_seconds: number;
  sample_rate: number;
  channels: number;
  format: string;
  file_size_bytes: number;
}

export interface AudioUploadResponse {
  file_id: string;
  filename: string;
  metadata: AudioMetadata;
  waveform_peaks: number[];
}

export interface AudioWaveformResponse {
  file_id: string;
  peaks: number[];
  sample_count: number;
}
