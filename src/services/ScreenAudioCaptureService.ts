import { io, Socket } from 'socket.io-client';

export class ScreenAudioCaptureService {
  private audioContext: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private processor: ScriptProcessorNode | null = null;
  private socket: Socket | null = null;
  private isProcessing: boolean = false;
  private buffer: Float32Array[] = [];
  private lastSendTime: number = 0;
  private readonly SEND_INTERVAL = 5000; // 5 seconds

  async initialize(): Promise<void> {
    try {
      // Initialize socket connection
      this.socket = io('http://localhost:5000', {
        transports: ['websocket'],
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 1000
      });

      // Initialize audio context
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: 44100,
        latencyHint: 'interactive'
      });

      // Request system audio capture
      const stream = await navigator.mediaDevices.getDisplayMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: 1,
          mediaSource: "browser"
        } as MediaTrackConstraints,
        video: false
      });

      this.mediaStream = stream;
      const source = this.audioContext.createMediaStreamSource(stream);
      this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);
      
      this.processor.onaudioprocess = this.handleAudioProcess.bind(this);
      source.connect(this.processor);
      this.processor.connect(this.audioContext.destination);

      this.setupSocketHandlers();
      this.isProcessing = true;

      console.log('Screen audio capture initialized successfully');
    } catch (error) {
      console.error('Screen audio capture initialization error:', error);
      this.cleanup();
      throw error;
    }
  }

  private handleAudioProcess(event: AudioProcessingEvent) {
    if (!this.isProcessing || !this.socket?.connected) return;

    const inputData = event.inputBuffer.getChannelData(0);
    const currentTime = Date.now();

    this.buffer.push(new Float32Array(inputData));

    if (currentTime - this.lastSendTime >= this.SEND_INTERVAL) {
      this.sendAudioData();
      this.lastSendTime = currentTime;
    }
  }

  private sendAudioData() {
    if (this.buffer.length === 0 || !this.socket?.connected) return;

    try {
      const totalLength = this.buffer.reduce((sum, chunk) => sum + chunk.length, 0);
      const combinedBuffer = new Float32Array(totalLength);
      let offset = 0;

      for (const chunk of this.buffer) {
        combinedBuffer.set(chunk, offset);
        offset += chunk.length;
      }

      this.socket.emit('audio_data', {
        buffer: combinedBuffer.buffer,
        timestamp: Date.now(),
        mode: 'system'
      });

      this.buffer = [];
    } catch (error) {
      console.error('Error sending audio data:', error);
    }
  }

  private setupSocketHandlers() {
    if (!this.socket) return;

    this.socket.on('connect', () => {
      console.log('Socket connected');
    });

    this.socket.on('disconnect', (reason) => {
      console.log('Socket disconnected:', reason);
      if (reason === 'io server disconnect') {
        this.socket?.connect();
      }
    });
  }

  cleanup() {
    this.isProcessing = false;

    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }

    if (this.processor) {
      this.processor.disconnect();
      this.processor = null;
    }

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => {
        track.stop();
      });
      this.mediaStream = null;
    }

    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
  }
}