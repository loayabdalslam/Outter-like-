import React from 'react';
import { TranscriptionHeader } from '../components/Transcription/TranscriptionHeader';
import { TranscriptionList } from '../components/Transcription/TranscriptionList';
import { RecordingControl } from '../components/RecordingControl/RecordingControl';
import { TranscriptionMessage } from '../types';

interface TranscriptionViewProps {
  isRecording: boolean;
  messages: TranscriptionMessage[];
  onToggleRecording: () => void;
}

export function TranscriptionView({ 
  isRecording, 
  messages = [], 
  onToggleRecording 
}: TranscriptionViewProps) {
  return (
    <div className="max-w-4xl mx-auto">
      <TranscriptionHeader isRecording={isRecording} />
      
      <div className="bg-white rounded-lg shadow-sm p-6 min-h-[calc(100vh-200px)]">
        <TranscriptionList messages={messages || []} />
        
        {messages?.length === 0 && (
          <div className="flex flex-col items-center justify-center h-[400px] text-gray-500">
            <p>Start recording to begin transcription</p>
          </div>
        )}
      </div>

      <RecordingControl 
        isRecording={isRecording} 
        onToggle={onToggleRecording}
        recordingTime={0}
      />
    </div>
  );
}