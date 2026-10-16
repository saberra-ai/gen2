export const SAMPLE_RATE = 16_000;
const LIMIT = SAMPLE_RATE * 60;
const MINIMUM = 320;

// Every sample belongs to exactly one part. Prefer quiet points near the end
// of each minute; continuous speech falls back to the model's hard limit.
export function splitAudio(samples) {
  const parts = [];
  let start = 0;
  while (samples.length - start > LIMIT) {
    const maximum = start + LIMIT;
    let end = maximum;
    let lowestEnergy = 0.0001;
    for (let window = maximum - SAMPLE_RATE * 8; window + 1600 <= maximum; window += 800) {
      let energy = 0;
      for (let i = window; i < window + 1600; i++) energy += samples[i] ** 2;
      energy /= 1600;
      if (energy < lowestEnergy) { lowestEnergy = energy; end = window + 800; }
    }
    end = Math.min(end, samples.length - MINIMUM);
    parts.push(samples.subarray(start, end));
    start = end;
  }
  parts.push(samples.subarray(start));
  return parts;
}

export function encodeWav(samples) {
  const wav = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(wav);
  const ascii = (offset, text) => [...text].forEach((char, index) => view.setUint8(offset + index, char.charCodeAt(0)));
  ascii(0, 'RIFF'); view.setUint32(4, wav.byteLength - 8, true); ascii(8, 'WAVE'); ascii(12, 'fmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, SAMPLE_RATE, true); view.setUint32(28, SAMPLE_RATE * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  ascii(36, 'data'); view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    if (!Number.isFinite(samples[i])) throw new Error('This recording contains invalid audio samples.');
    const sample = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, Math.round(sample * (sample < 0 ? 32768 : 32767)), true);
  }
  return new Blob([wav], { type: 'audio/wav' });
}
