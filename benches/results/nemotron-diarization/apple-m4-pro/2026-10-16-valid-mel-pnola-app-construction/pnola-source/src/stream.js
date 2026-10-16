export async function* readMessages(stream) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let pending = '';
  try {
    while (true) {
      const { done, value } = await reader.read();
      pending += done ? decoder.decode() : decoder.decode(value, { stream: true });
      let newline;
      while ((newline = pending.indexOf('\n')) >= 0) {
        const line = pending.slice(0, newline); pending = pending.slice(newline + 1);
        if (line.trim()) yield JSON.parse(line);
      }
      if (done) break;
    }
    if (pending.trim()) yield JSON.parse(pending);
  } finally { reader.releaseLock(); }
}
