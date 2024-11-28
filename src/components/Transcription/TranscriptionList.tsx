import React from 'react';
import { format } from 'date-fns';
import { TranscriptionMessage } from '../../types';

interface TranscriptionListProps {
  messages: TranscriptionMessage[];
}

export function TranscriptionList({ messages }: TranscriptionListProps) {
  return (
    <div className="space-y-6">
      {messages.map((message, index) => (
        <div key={index} className="flex gap-4">
          <div className="w-12 h-12 rounded-full bg-gray-200 flex-shrink-0" />
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-gray-500">{format(message.timestamp, 'h:mm')}</span>
            </div>
            <p className="text-gray-800">{message.text}</p>
          </div>
        </div>
      ))}
    </div>
  );
}