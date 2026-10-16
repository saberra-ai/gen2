const requestResult = request => new Promise((resolve, reject) => {
  request.onsuccess = () => resolve(request.result);
  request.onerror = () => reject(request.error);
});
const finished = transaction => new Promise((resolve, reject) => {
  transaction.oncomplete = () => resolve();
  transaction.onabort = () => reject(transaction.error || new Error('Local storage transaction was interrupted.'));
  transaction.onerror = () => { /* onabort supplies the final error. */ };
});

export class LiveStore {
  async database() {
    if (!this.connection) this.connection = new Promise((resolve, reject) => {
      const request = indexedDB.open('pnola-live', 1);
      request.onupgradeneeded = () => {
        const database = request.result;
        database.createObjectStore('sessions', { keyPath: 'id' });
        const segments = database.createObjectStore('segments', { keyPath: ['sessionId', 'sequence'] });
        segments.createIndex('state', ['sessionId', 'state', 'sequence']);
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
      request.onblocked = () => reject(new Error('Close other Pnola tabs to open local recording storage.'));
    });
    return this.connection;
  }

  async create(name) {
    const session = { id: crypto.randomUUID(), name, date: new Date().toISOString(), live: true,
      nextSequence: 0, completed: 0, duration: 0, words: 0, execution: 0, recent: [] };
    const db = await this.database(), transaction = db.transaction('sessions', 'readwrite'), done = finished(transaction);
    transaction.objectStore('sessions').add(session); await done; return session;
  }

  async list() {
    const db = await this.database();
    return requestResult(db.transaction('sessions').objectStore('sessions').getAll());
  }

  async get(id) {
    const db = await this.database();
    return requestResult(db.transaction('sessions').objectStore('sessions').get(id));
  }

  async append(id, audio, duration) {
    const db = await this.database(), transaction = db.transaction(['sessions', 'segments'], 'readwrite'), done = finished(transaction);
    const sessions = transaction.objectStore('sessions'), session = await requestResult(sessions.get(id));
    transaction.objectStore('segments').add({ sessionId: id, sequence: session.nextSequence++, state: 'pending', audio, duration });
    sessions.put(session); await done; return session;
  }

  async next(id) {
    const db = await this.database();
    const range = IDBKeyRange.bound([id, 'pending', 0], [id, 'pending', Number.MAX_SAFE_INTEGER]);
    return requestResult(db.transaction('segments').objectStore('segments').index('state').get(range));
  }

  async commit(segment, result) {
    const db = await this.database(), transaction = db.transaction(['sessions', 'segments'], 'readwrite'), done = finished(transaction);
    const sessions = transaction.objectStore('sessions'), segments = transaction.objectStore('segments');
    const stored = await requestResult(segments.get([segment.sessionId, segment.sequence]));
    const session = await requestResult(sessions.get(segment.sessionId));
    if (stored?.state === 'pending') {
      const text = result.text.trim();
      // Replacing the segment atomically removes processed audio and saves text.
      segments.put({ sessionId: segment.sessionId, sequence: segment.sequence, state: 'done', text, duration: segment.duration });
      session.completed++; session.duration += segment.duration;
      session.words += text ? text.split(/\s+/u).length : 0;
      session.execution += result.execution_micros || 0;
      session.recent = [...session.recent, text].slice(-30);
      sessions.put(session);
    }
    await done; return session;
  }

  async export(id) {
    const db = await this.database(), transaction = db.transaction('segments');
    const request = transaction.objectStore('segments').openCursor(IDBKeyRange.bound([id, 0], [id, Number.MAX_SAFE_INTEGER]));
    const parts = [];
    await new Promise((resolve, reject) => {
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const cursor = request.result;
        if (!cursor) return resolve();
        if (cursor.value.state === 'done') { if (parts.length) parts.push('\n\n'); parts.push(cursor.value.text); }
        cursor.continue();
      };
    });
    return new Blob(parts, { type: 'text/plain;charset=utf-8' });
  }

  async remove(id) {
    const db = await this.database(), transaction = db.transaction(['sessions', 'segments'], 'readwrite'), done = finished(transaction);
    transaction.objectStore('sessions').delete(id);
    transaction.objectStore('segments').delete(IDBKeyRange.bound([id, 0], [id, Number.MAX_SAFE_INTEGER]));
    await done;
  }
}
