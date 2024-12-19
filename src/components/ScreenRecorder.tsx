import { useEffect, useRef } from 'react'
import { socket } from '../socket'

interface ScreenRecorderProps {
  isRecording: boolean
  onStop: () => void
}

const ScreenRecorder = ({ isRecording, onStop }: ScreenRecorderProps) => {
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const screenSourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const micSourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const mergerNodeRef = useRef<ChannelMergerNode | null>(null)
  const processorNodeRef = useRef<ScriptProcessorNode | null>(null)
  const audioChunkRef = useRef<Float32Array[]>([])
  const lastSendTimeRef = useRef<number>(Date.now())
  const streamsRef = useRef<MediaStream[]>([])

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
      // Get screen with audio
      const screenStream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100,
        }
      })

      // Get microphone
      const micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100,
        }
      })

      // Store streams for cleanup
      streamsRef.current = [screenStream, micStream]

      // Create audio context and nodes
      audioContextRef.current = new AudioContext()
      
      // Create source nodes for both streams
      screenSourceNodeRef.current = audioContextRef.current.createMediaStreamSource(screenStream)
      micSourceNodeRef.current = audioContextRef.current.createMediaStreamSource(micStream)
      
      // Create merger node to combine audio streams
      mergerNodeRef.current = audioContextRef.current.createChannelMerger(2)
      
      // Create processor node for the merged audio
      processorNodeRef.current = audioContextRef.current.createScriptProcessor(4096, 2, 1)

      // Reset the chunk buffer and timer
      audioChunkRef.current = []
      lastSendTimeRef.current = Date.now()

      processorNodeRef.current.onaudioprocess = (e) => {
        // Get the mixed audio data
        const mixedData = e.inputBuffer.getChannelData(0)
        audioChunkRef.current.push(new Float32Array(mixedData))

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
            mode: 'recording'
          })

          // Reset for next chunk
          audioChunkRef.current = []
          lastSendTimeRef.current = currentTime
        }
      }

      // Connect nodes:
      // Screen audio -> merger input 0
      // Mic audio -> merger input 1
      // Merger -> processor -> destination
      screenSourceNodeRef.current.connect(mergerNodeRef.current, 0, 0)
      micSourceNodeRef.current.connect(mergerNodeRef.current, 0, 1)
      mergerNodeRef.current.connect(processorNodeRef.current)
      processorNodeRef.current.connect(audioContextRef.current.destination)

      // Start recording
      mediaRecorderRef.current = new MediaRecorder(screenStream)
      mediaRecorderRef.current.start()

      // Handle when user stops sharing screen
      screenStream.getVideoTracks()[0].onended = () => {
        onStop()
      }

    } catch (error) {
      console.error('Error starting screen recording:', error)
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
        mode: 'recording'
      })

      audioChunkRef.current = []
    }

    // Stop all tracks in all streams
    streamsRef.current.forEach(stream => {
      stream.getTracks().forEach(track => track.stop())
    })
    streamsRef.current = []

    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop()
    }

    if (processorNodeRef.current) {
      processorNodeRef.current.disconnect()
      processorNodeRef.current = null
    }

    if (mergerNodeRef.current) {
      mergerNodeRef.current.disconnect()
      mergerNodeRef.current = null
    }

    if (screenSourceNodeRef.current) {
      screenSourceNodeRef.current.disconnect()
      screenSourceNodeRef.current = null
    }

    if (micSourceNodeRef.current) {
      micSourceNodeRef.current.disconnect()
      micSourceNodeRef.current = null
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
        <span>{isRecording ? 'Recording screen with microphone...' : 'Screen recording stopped'}</span>
      </div>
    </div>
  )
}

export default ScreenRecorder 