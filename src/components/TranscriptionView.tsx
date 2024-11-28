import { useState, useEffect, useRef } from 'react';
import { TranscriptionMessage } from '../types';
import { format } from 'date-fns';
import { io, Socket } from 'socket.io-client';

interface TranscriptionViewProps {
    isRecording: boolean;
    onToggleRecording: () => void;
}

export function TranscriptionView({ isRecording, onToggleRecording }: TranscriptionViewProps) {
    const [messages, setMessages] = useState<TranscriptionMessage[]>([]);
    const socketRef = useRef<Socket | null>(null);

    useEffect(() => {
        // Create socket connection
        socketRef.current = io('http://localhost:5000', {
            transports: ['websocket'],
            reconnection: true,
            reconnectionAttempts: 5,
            reconnectionDelay: 1000
        });

        // Socket event handlers
        socketRef.current.on('connect', () => {
            console.log('Socket connected');
        });

        socketRef.current.on('disconnect', () => {
            console.log('Socket disconnected');
        });

        socketRef.current.on('transcription', (data: { text: string, timestamp: number, mode: string }) => {
            console.log('Received transcription:', data);
            const newMessage: TranscriptionMessage = {
                text: data.text,
                timestamp: data.timestamp || Date.now(),
                mode: data.mode || 'system'
            };
            
            setMessages(prevMessages => [...prevMessages, newMessage]);
        });

        // Cleanup on unmount
        return () => {
            if (socketRef.current) {
                socketRef.current.disconnect();
            }
        };
    }, []);

    return (
        <div className="max-w-2xl mx-auto p-4">
            <div className="mb-4">
                <button
                    onClick={onToggleRecording}
                    className={`px-4 py-2 rounded ${
                        isRecording ? 'bg-red-500' : 'bg-blue-500'
                    } text-white`}
                >
                    {isRecording ? 'Stop Recording' : 'Start Recording'}
                </button>
            </div>

            <div className="space-y-4">
                {messages.map((message, index) => (
                    <div 
                        key={`${message.timestamp}-${index}`}
                        className="bg-white rounded-lg shadow p-4"
                    >
                        <div className="text-sm text-gray-500 mb-1">
                            {format(message.timestamp, 'HH:mm:ss')}
                        </div>
                        <div className="text-gray-800 whitespace-pre-wrap" dir="auto">
                            {message.text}
                        </div>
                    </div>
                ))}

                {messages.length === 0 && (
                    <div className="text-center text-gray-500">
                        No transcriptions yet. Start recording to see them appear here.
                    </div>
                )}
            </div>
        </div>
    );
}