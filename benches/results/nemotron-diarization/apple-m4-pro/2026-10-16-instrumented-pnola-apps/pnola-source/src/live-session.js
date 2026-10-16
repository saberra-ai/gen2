import { encodeWav } from './audio.js';

export class LiveSession {
  constructor({ store, id, transcribe, onUpdate = () => {}, onError = () => {}, onWaiting = () => {}, retryMs = 1500 }) {
    Object.assign(this, { store, id, transcribe, onUpdate, onError, onWaiting, retryMs });
    this.writes = Promise.resolve(); this.writerCount = 0; this.accepting = true;
  }

  enqueue(samples) {
    if (!this.accepting) return Promise.reject(new Error('This live session is closed.'));
    // Pad only the last sub-20ms block to the model minimum. Track its real
    // duration separately so the session clock does not count padding.
    const duration = samples.length / 16000;
    const audio = encodeWav(samples.length < 320 ? new Float32Array(320).map((_, i) => samples[i] || 0) : samples);
    this.writerCount++;
    const write = this.writes.then(async () => {
      const session = await this.store.append(this.id, audio, duration);
      this.onUpdate(session); this.kick();
    });
    this.writes = write.catch(error => { this.accepting = false; this.failure = error; this.onError(error); }).finally(() => this.writerCount--);
    // Stop capture if local storage cannot keep up, rather than accumulating
    // an unbounded chain of PCM buffers in browser memory.
    if (this.writerCount >= 4) this.onError(new Error('Local storage is falling behind. Capture is stopping; saved sections will finish transcribing.'));
    return write;
  }

  kick() {
    if (this.running || this.failure) return this.running;
    this.running = this.drain().catch(error => { this.failure = error; this.onError(error); }).finally(() => this.running = undefined);
    return this.running;
  }

  async drain() {
    for (;;) {
      const segment = await this.store.next(this.id);
      if (!segment) return;
      let result;
      try { result = await this.transcribe(segment.audio); }
      catch (error) {
        if (error.retryable && this.accepting) {
          this.onWaiting(error);
          await new Promise(resolve => setTimeout(resolve, this.retryMs)); continue;
        }
        throw error;
      }
      this.onUpdate(await this.store.commit(segment, result));
    }
  }

  async finish() {
    this.accepting = false;
    await this.writes;
    // A write can finish just as an empty drain exits. Recheck the durable
    // outbox after the running request has completed so the final block stays.
    if (this.running) await this.running;
    if (!this.failure) await this.kick();
    if (this.failure) throw this.failure;
    return this.store.get(this.id);
  }
}
