import React from 'react';
import { Mic, MicOff, Loader2 } from 'lucide-react';

interface RecordingControlProps {
  isRecording: boolean;
  onToggle: () => void;
  recordingTime: number;
}

export function RecordingControl({ isRecording, onToggle, recordingTime }: RecordingControlProps) {
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="fixed bottom-6 left-1/2 transform -translate-x-1/2 z-50">
      <div className="bg-white rounded-full shadow-lg px-6 py-4 flex items-center space-x-4">
        <button
          onClick={onToggle}
          className={`
            rounded-full w-14 h-14 flex items-center justify-center transition-all
            ${isRecording 
              ? 'bg-red-500 hover:bg-red-600 animate-pulse' 
              : 'bg-blue-500 hover:bg-blue-600'
            }
          `}
        >
          {isRecording ? (
            <MicOff className="w-6 h-6 text-white" />
          ) : (
            <Mic className="w-6 h-6 text-white" />
          )}
        </button>
        {isRecording && (
          <div className="flex items-center space-x-3">
            <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></div>
            <div className="text-gray-700 font-mono text-lg">
              {formatTime(recordingTime)}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}