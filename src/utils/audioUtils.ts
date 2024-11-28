// Audio processing utilities
export const CHUNK_SIZE = 4096;
export const SEND_INTERVAL = 10;

export function formatAudioData(audioData: Float32Array): Blob {
  const wavBuffer = new ArrayBuffer(44 + audioData.length * 2);
  const view = new DataView(wavBuffer);
  const sampleRate = 44100;

  // Write WAV header
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + audioData.length * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, audioData.length * 2, true);

  // Write audio data
  const length = audioData.length;
  let index = 44;
  for (let i = 0; i < length; i++) {
    view.setInt16(index, audioData[i] * 0x7fff, true);
    index += 2;
  }

  return new Blob([wavBuffer], { type: 'audio/wav' });
}

function writeString(view: DataView, offset: number, string: string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}