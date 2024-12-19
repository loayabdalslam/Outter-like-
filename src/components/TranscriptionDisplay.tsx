import { useEffect, useState } from 'react'
import { socket } from '../socket'
import { FileText } from 'lucide-react'

interface Transcription {
  text: string
  timestamp: number
  mode: 'screen' | 'mic'
}

const TranscriptionDisplay = () => {
  const [transcriptions, setTranscriptions] = useState<Transcription[]>([])
  const [summary, setSummary] = useState<string>('')
  const [isGeneratingSummary, setIsGeneratingSummary] = useState(false)

  useEffect(() => {
    socket.on('transcription', (data: Transcription) => {
      setTranscriptions(prev => [...prev, data])
    })

    socket.on('summary_result', (data: { text?: string, error?: string }) => {
      setIsGeneratingSummary(false)
      if (data.text) {
        setSummary(data.text)
      } else if (data.error) {
        alert(`Error generating summary: ${data.error}`)
      }
    })

    return () => {
      socket.off('transcription')
      socket.off('summary_result')
    }
  }, [])

  const handleGenerateSummary = () => {
    setIsGeneratingSummary(true)
    socket.emit('generate_summary')
  }

  return (
    <div className="mt-8">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold">Transcriptions</h2>
        <button
          onClick={handleGenerateSummary}
          disabled={isGeneratingSummary || transcriptions.length === 0}
          className="flex items-center gap-2 px-4 py-2 bg-purple-500 text-white rounded hover:bg-purple-600 disabled:opacity-50"
        >
          <FileText size={20} />
          {isGeneratingSummary ? 'Generating...' : 'Generate Summary'}
        </button>
      </div>

      {summary && (
        <div className="mb-6 p-4 bg-purple-50 rounded-lg border border-purple-200">
          <h3 className="text-lg font-semibold text-purple-800 mb-2">Summary</h3>
          <p className="text-purple-900">{summary}</p>
        </div>
      )}

      <div className="bg-white rounded shadow p-4 max-h-96 overflow-y-auto">
        {transcriptions.length === 0 ? (
          <p className="text-gray-500">No transcriptions yet...</p>
        ) : (
          transcriptions.map((t, i) => (
            <div key={i} className="mb-4 last:mb-0">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-sm text-gray-500">
                  {new Date(t.timestamp).toLocaleTimeString()}
                </span>
                <span className="text-xs px-2 py-1 rounded bg-green-100 text-green-800">
                  Recording
                </span>
              </div>
              <p className="text-gray-700">{t.text}</p>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

export default TranscriptionDisplay 