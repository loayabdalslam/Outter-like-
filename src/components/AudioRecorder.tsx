import { useEffect, useRef } from 'react'
import { socket } from '../socket'

interface AudioRecorderProps {
  isRecording: boolean
  onStop: () => void
}

const AudioRecorder = ({ isRecording, onStop }: AudioRecorderProps) => {
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const processorNodeRef = useRef<ScriptProcessorNode | null>(null)
  const audioChunkRef = useRef<Float32Array[]>([])
  const lastSendTimeRef = useRef<number>(Date.now())

  useEffect(() => {
    if (isRecording) {
      startRecording()
    } else {
      stopRecording()
    }

    return () => {
      stopRecording()
    }
  }, [isRecording])

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      
      audioContextRef.current = new AudioContext()
      sourceNodeRef.current = audioContextRef.current.createMediaStreamSource(stream)
      processorNodeRef.current = audioContextRef.current.createScriptProcessor(4096, 1, 1)

      // Reset the chunk buffer and timer
      audioChunkRef.current = []
      lastSendTimeRef.current = Date.now()

      processorNodeRef.current.onaudioprocess = (e) => {
        const float32Array = e.inputBuffer.getChannelData(0)
        audioChunkRef.current.push(new Float32Array(float32Array))

        const currentTime = Date.now()
        if (currentTime - lastSendTimeRef.current >= 5000) { // 5 seconds
          // Concatenate all chunks into a single Float32Array
          const totalLength = audioChunkRef.current.reduce((acc, chunk) => acc + chunk.length, 0)
          const mergedArray = new Float32Array(totalLength)
          let offset = 0
          
          audioChunkRef.current.forEach(chunk => {
            mergedArray.set(chunk, offset)
            offset += chunk.length
          })

          // Send the merged chunk
          socket.emit('audio_data', {
            buffer: mergedArray,
            timestamp: currentTime,
            mode: 'mic'
          })

          // Reset for next chunk
          audioChunkRef.current = []
          lastSendTimeRef.current = currentTime
        }
      }

      sourceNodeRef.current.connect(processorNodeRef.current)
      processorNodeRef.current.connect(audioContextRef.current.destination)

      mediaRecorderRef.current = new MediaRecorder(stream)
      mediaRecorderRef.current.start()

    } catch (error) {
      console.error('Error starting recording:', error)
      onStop()
    }
  }

  const stopRecording = () => {
    // Send any remaining audio data before stopping
    if (audioChunkRef.current.length > 0) {
      const totalLength = audioChunkRef.current.reduce((acc, chunk) => acc + chunk.length, 0)
      const mergedArray = new Float32Array(totalLength)
      let offset = 0
      
      audioChunkRef.current.forEach(chunk => {
        mergedArray.set(chunk, offset)
        offset += chunk.length
      })

      socket.emit('audio_data', {
        buffer: mergedArray,
        timestamp: Date.now(),
        mode: 'mic'
      })

      audioChunkRef.current = []
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop()
    }

    if (processorNodeRef.current) {
      processorNodeRef.current.disconnect()
      processorNodeRef.current = null
    }

    if (sourceNodeRef.current) {
      sourceNodeRef.current.disconnect()
      sourceNodeRef.current = null
    }

    if (audioContextRef.current) {
      audioContextRef.current.close()
      audioContextRef.current = null
    }
  }

  return (
    <div className="p-4 bg-white rounded shadow">
      <div className="flex items-center gap-2">
        <div className={`w-3 h-3 rounded-full ${isRecording ? 'bg-red-500 animate-pulse' : 'bg-gray-300'}`} />
        <span>{isRecording ? 'Recording microphone...' : 'Microphone recording stopped'}</span>
      </div>
    </div>
  )
}

export default AudioRecorder 