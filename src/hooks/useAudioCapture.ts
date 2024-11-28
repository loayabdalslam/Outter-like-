import { useState, useEffect, useCallback } from 'react';
import { AudioCaptureService } from '../services/AudioCaptureService';
import { ScreenAudioCaptureService } from '../services/ScreenAudioCaptureService';

export type CaptureMode = 'microphone' | 'screen';

export function useAudioCapture() {
  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [mode, setMode] = useState<CaptureMode>('microphone');
  const [micService] = useState(() => new AudioCaptureService());
  const [screenService] = useState(() => new ScreenAudioCaptureService());

  const startRecording = useCallback(async () => {
    try {
      const service = mode === 'microphone' ? micService : screenService;
      await service.initialize();
      setIsRecording(true);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to start recording'));
      setIsRecording(false);
    }
  }, [mode, micService, screenService]);

  const stopRecording = useCallback(() => {
    const service = mode === 'microphone' ? micService : screenService;
    service.cleanup();
    setIsRecording(false);
  }, [mode, micService, screenService]);

  useEffect(() => {
    return () => {
      if (isRecording) {
        stopRecording();
      }
    };
  }, [isRecording, stopRecording]);

  const toggleMode = useCallback(() => {
    if (isRecording) {
      stopRecording();
    }
    setMode(prevMode => prevMode === 'microphone' ? 'screen' : 'microphone');
    setError(null);
  }, [isRecording, stopRecording]);

  return {
    isRecording,
    error,
    mode,
    startRecording,
    stopRecording,
    toggleMode,
  };
}