import React from 'react';
import { format } from 'date-fns';

interface TranscriptionHeaderProps {
  isRecording: boolean;
}

export function TranscriptionHeader({ isRecording }: TranscriptionHeaderProps) {
  const currentTime = new Date();
  
  return (
    <div className="mb-6">
      <h1 className="text-2xl font-bold mb-4">Note</h1>
      <div className="flex items-center gap-4 text-sm text-gray-500">
        <span>{format(currentTime, 'EEE, MMM d, yyyy • h:mm a')}</span>
        <span>{isRecording ? 'Recording...' : '0:00'}</span>
      </div>
    </div>
  );
}