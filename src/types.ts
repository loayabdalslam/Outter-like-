export interface TranscriptionMessage {
  text: string;
  timestamp: number;
  mode: 'system' | 'user' | 'both';
} 