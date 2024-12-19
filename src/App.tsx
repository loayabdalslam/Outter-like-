import { useState } from 'react'
import { MonitorStop, StopCircle } from 'lucide-react'
import ScreenRecorder from './components/ScreenRecorder'
import TranscriptionDisplay from './components/TranscriptionDisplay'

function App() {
  const [isRecording, setIsRecording] = useState(false)

  const startRecording = () => {
    setIsRecording(true)
  }

  const stopRecording = () => {
    setIsRecording(false)
  }

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">Screen & Voice Recorder</h1>
        
        <div className="flex gap-4 mb-8">
          {!isRecording ? (
            <button
              onClick={startRecording}
              className="flex items-center gap-2 px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
            >
              <MonitorStop size={20} />
              Record Screen with Mic
            </button>
          ) : (
            <button
              onClick={stopRecording}
              className="flex items-center gap-2 px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
            >
              <StopCircle size={20} />
              Stop Recording
            </button>
          )}
        </div>

        <ScreenRecorder isRecording={isRecording} onStop={stopRecording} />
        <TranscriptionDisplay />
      </div>
    </div>
  )
}

export default App 