// Keep only one eight-second PCM buffer on the audio thread. The AudioContext
// runs at 16 kHz, so every posted block is ready for Bark without re-encoding.
class PnolaCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(16000 * 8);
    this.offset = 0;
    this.active = true;
    this.port.onmessage = event => {
      if (event.data.type === 'flush') {
        this.active = false;
        this.flush();
        this.port.postMessage({ type: 'flushed' });
      }
    };
  }

  flush() {
    if (!this.offset) return;
    const samples = this.buffer.slice(0, this.offset);
    this.port.postMessage({ type: 'chunk', samples }, [samples.buffer]);
    this.offset = 0;
  }

  process(inputs) {
    if (!this.active) return false;
    const channels = inputs[0];
    if (!channels?.length) return true;
    for (let frame = 0; frame < channels[0].length; frame++) {
      let sample = 0;
      for (const channel of channels) sample += channel[frame] / channels.length;
      this.buffer[this.offset++] = sample;
      if (this.offset === this.buffer.length) this.flush();
    }
    return true;
  }
}
registerProcessor('pnola-capture', PnolaCapture);
