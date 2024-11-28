export interface TranscriptionMessage {
  text: string;
  timestamp: Date;
  speaker?: string;
}

export interface AudioCaptureState {
  isRecording: boolean;
  error: Error | null;
  mode: 'microphone' | 'screen';
}