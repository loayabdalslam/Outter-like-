import React from 'react';
import { Sidebar } from './components/Sidebar';
import { TranscriptionView } from './components/TranscriptionView';
import { AIChatSidebar } from './components/Sidebar/AIChatSidebar';
import { useAudioCapture } from './hooks/useAudioCapture';
import { TranscriptionHeader } from './TranscriptionHeader';

function App() {
  const { 
    isRecording, 
    messages, 
    startRecording, 
    stopRecording 
  } = useAudioCapture();

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Left Sidebar */}
      <Sidebar />

      {/* Main Content */}
      <main className="flex-1 overflow-auto px-8 py-6 ml-[240px] mr-[320px]">
        <TranscriptionView 
          isRecording={isRecording}
          messages={messages}
          onToggleRecording={isRecording ? stopRecording : startRecording}
        />
      </main>

      {/* Right Sidebar */}
      <AIChatSidebar />
    </div>
  );
}

export default App;