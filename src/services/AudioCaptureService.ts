import { io, Socket } from 'socket.io-client';

export class AudioCaptureService {
  private audioContext: AudioContext | null = null;
  private micStream: MediaStream | null = null;
  private systemStream: MediaStream | null = null;
  private processor: ScriptProcessorNode | null = null;
  private socket: Socket | null = null;
  private audioBuffer: Float32Array[] = [];
  private lastSendTime: number = 0;
  private readonly SEND_INTERVAL = 5000; // 5 seconds in milliseconds
  private mode: 'both' | 'system' | 'microphone' = 'both';
  private transcriptionCallback: ((text: string) => void) | null = null;
  private isProcessing: boolean = false;
  private isInitialized: boolean = false;

  constructor(mode: 'both' | 'system' | 'microphone' = 'both') {
    this.mode = mode;
  }

  async initialize(): Promise<void> {
    if (this.isInitialized) {
      console.log('Already initialized, skipping...');
      return;
    }

    try {
      console.log('Initializing AudioCaptureService...');
      
      // Connect to WebSocket server
      this.socket = io('http://localhost:5000', {
        transports: ['websocket'],
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 1000
      });

      // Initialize audio context first
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: 44100,
        latencyHint: 'interactive'
      });

      console.log('Audio context created');

      // Initialize audio streams based on mode
      if (this.mode === 'both') {
        console.log('Initializing both microphone and system audio...');
        await this.initializeMicAndSystem();
      } else if (this.mode === 'system') {
        console.log('Initializing system audio only...');
        await this.initializeSystem();
      } else {
        console.log('Initializing microphone only...');
        await this.initializeMicrophone();
      }

      // Set up socket handlers
      this.setupSocketHandlers();
      
      // Start processing
      this.isProcessing = true;
      this.isInitialized = true;
      console.log('Audio capture initialized successfully');
      
    } catch (error) {
      console.error('Initialization error:', error);
      this.handleError(error);
    }
  }

  private async initializeMicAndSystem() {
    // Get microphone stream
    const micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1
      },
      video: false
    });

    // Get system audio stream
    const sysStream = await navigator.mediaDevices.getDisplayMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        channelCount: 1,
        mediaSource: "browser"
      } as MediaTrackConstraints,
      video: {
        width: 1,
        height: 1,
        frameRate: 1
      }
    });

    // Create audio processing chain
    const micSource = this.audioContext!.createMediaStreamSource(micStream);
    const sysSource = this.audioContext!.createMediaStreamSource(sysStream);
    
    // Create gain nodes for mixing
    const micGain = this.audioContext!.createGain();
    const sysGain = this.audioContext!.createGain();
    
    micGain.gain.value = 0.5; // Adjust microphone volume
    sysGain.gain.value = 0.5; // Adjust system audio volume

    // Create processor node
    this.processor = this.audioContext!.createScriptProcessor(2048, 1, 1);
    
    // Connect the audio graph
    micSource.connect(micGain);
    sysSource.connect(sysGain);
    micGain.connect(this.processor);
    sysGain.connect(this.processor);
    this.processor.connect(this.audioContext!.destination);

    // Store streams for cleanup
    this.micStream = micStream;
    this.systemStream = sysStream;

    // Set up audio processing
    this.processor.onaudioprocess = this.handleAudioProcess.bind(this);
  }

  private async initializeSystem() {
    const stream = await navigator.mediaDevices.getDisplayMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        channelCount: 1,
        mediaSource: "browser"
      } as MediaTrackConstraints,
      video: {
        width: 1,
        height: 1,
        frameRate: 1
      }
    });

    const source = this.audioContext!.createMediaStreamSource(stream);
    this.processor = this.audioContext!.createScriptProcessor(2048, 1, 1);
    source.connect(this.processor);
    this.processor.connect(this.audioContext!.destination);
    
    this.systemStream = stream;
    this.processor.onaudioprocess = this.handleAudioProcess.bind(this);
  }

  private async initializeMicrophone() {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1
      },
      video: false
    });

    const source = this.audioContext!.createMediaStreamSource(stream);
    this.processor = this.audioContext!.createScriptProcessor(2048, 1, 1);
    source.connect(this.processor);
    this.processor.connect(this.audioContext!.destination);
    
    this.micStream = stream;
    this.processor.onaudioprocess = this.handleAudioProcess.bind(this);
  }

  private handleAudioProcess(event: AudioProcessingEvent) {
    if (!this.isProcessing || !this.isInitialized) {
      console.log('Skipping audio processing - not in processing state');
      return;
    }

    const inputData = event.inputBuffer.getChannelData(0);
    const currentTime = Date.now();

    // Add current chunk to buffer
    this.audioBuffer.push(new Float32Array(inputData));

    // Send accumulated data every 5 seconds
    if (currentTime - this.lastSendTime >= this.SEND_INTERVAL) {
      console.log(`Buffer size before sending: ${this.audioBuffer.length} chunks`);
      this.sendAudioData();
      this.lastSendTime = currentTime;
    }
  }

  private sendAudioData() {
    if (this.audioBuffer.length === 0) {
      console.log('No audio data to send');
      return;
    }
    
    if (!this.socket?.connected) {
      console.log('Socket not connected, attempting to reconnect...');
      this.socket?.connect();
      return;
    }

    try {
      // Combine all chunks in the buffer
      const totalLength = this.audioBuffer.reduce((sum, chunk) => sum + chunk.length, 0);
      const combinedBuffer = new Float32Array(totalLength);
      
      let offset = 0;
      for (const chunk of this.audioBuffer) {
        combinedBuffer.set(chunk, offset);
        offset += chunk.length;
      }

      console.log(`Sending audio data - Size: ${combinedBuffer.length}, Time: ${new Date().toISOString()}`);
      
      // Send the audio data through socket
      this.socket?.emit('audio_data', {
        buffer: combinedBuffer.buffer,
        timestamp: Date.now(),
        mode: this.mode,
        duration: this.SEND_INTERVAL
      });

      // Clear the buffer
      this.audioBuffer = [];
    } catch (error) {
      console.error('Error sending audio data:', error);
    }
  }

  private setupSocketHandlers() {
    if (!this.socket) return;

    this.socket.on('connect', () => {
      console.log('Socket connected successfully');
    });

    this.socket.on('disconnect', (reason) => {
      console.log('Socket disconnected:', reason);
      if (reason === 'io server disconnect') {
        // Reconnect if server disconnected
        this.socket?.connect();
      }
    });

    this.socket.on('transcription', (data: { text: string }) => {
      console.log('Received transcription:', data.text);
      if (this.transcriptionCallback) {
        this.transcriptionCallback(data.text);
      }
    });

    this.socket.on('connect_error', (error: Error) => {
      console.error('Socket connection error:', error);
    });

    this.socket.on('error', (error: Error) => {
      console.error('Socket error:', error);
    });
  }

  cleanup() {
    if (!this.isInitialized) {
      console.log('Not initialized, skipping cleanup...');
      return;
    }

    console.log('Starting cleanup process...');
    this.isProcessing = false;
    this.isInitialized = false;
    
    if (this.socket) {
      console.log('Disconnecting socket...');
      this.socket.disconnect();
      this.socket = null;
    }

    if (this.processor) {
      console.log('Disconnecting audio processor...');
      this.processor.disconnect();
      this.processor = null;
    }

    if (this.micStream) {
      console.log('Stopping microphone tracks...');
      this.micStream.getTracks().forEach(track => {
        track.stop();
        console.log('Microphone track stopped');
      });
      this.micStream = null;
    }

    if (this.systemStream) {
      console.log('Stopping system audio tracks...');
      this.systemStream.getTracks().forEach(track => {
        track.stop();
        console.log('System audio track stopped');
      });
      this.systemStream = null;
    }

    if (this.audioContext) {
      console.log('Closing audio context...');
      this.audioContext.close();
      this.audioContext = null;
    }

    this.audioBuffer = [];
    this.lastSendTime = 0;
    
    console.log('Cleanup completed');
  }

  private handleError(error: unknown) {
    console.error('Audio capture error:', error);
    if (this.isInitialized) {
      this.cleanup();
    }
    throw error;
  }

  onTranscription(callback: (text: string) => void) {
    this.transcriptionCallback = callback;
  }
}
