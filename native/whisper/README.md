# Private multilingual CPU speech shim

Build with CMake against clean official `ggml-org/whisper.cpp` commit
`d1be6fde11ac6e0407606b4e42fe72d34add8037` (v1.9.5). No binary or model download
runs during inference. `WHISPER_SOURCE_DIR` must identify that clean checkout.
The source/model license is MIT; include upstream LICENSE when shipping its
binary/weights. The consumer pins model repository revision and full SHA256.

```sh
cmake -S native/whisper -B /tmp/gen2-whisper -G Ninja \
  -DWHISPER_SOURCE_DIR=/path/to/pinned/whisper.cpp -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/gen2-whisper --target gen2_whisper
```

The stable C header hides upstream parameter-struct layouts. Only seven
`gen2_whisper_*` symbols export; statically linked GGML/Whisper symbols stay
private to prevent collision with a separate Gen2 llama.cpp backend. Inspect the
actual final binary exports and test coexistence; a source setting alone is not
release proof. CPU only; no GPU, network, context prompts or translation.

Use the opt-in `backend-asr-whisper` Rust feature and
`Runtime::load_whisper(model, whisper::LoadOptions::cpu(native_library))` on an
owned host worker. The exact base-q5_1 model is read/verified after Runtime peak
admission, then loaded from those verified bytes. Its384MiB reservation includes
that temporary input buffer plus native weights/workspace; it stays held until
native context teardown. Bark's separate4GiB floor is unchanged. The actual
mobile/app memory and co-residency budgets still require qualification.

`Transcriber` stays on its creating thread; `&mut self` serializes calls. A host
owns its job queue, cancellation/deadline, durable journal/progress and worker
admission through shutdown. Inputs are finite normalized mono16kHz,1800–480000
frames (112.5ms–30s). The host must partition longer recordings with complete
frame coverage; unsupported short tails must stay with preceding real audio.

Select English, Spanish or Auto explicitly. Native abort callbacks check the
host token/deadline; cancellation is checked again after return because upstream
can return success when an encoder callback aborts. Results carry exact input
frame count, bounded UTF-8, raw model timestamps and a separate clipped playable
source intersection/flag. Model timestamps are approximate; wholly out-of-source
or zero-length segments fail validation. No words are dropped to pass bounds.

An exactly digital-zero buffer returns typed `DigitalSilence`, empty text and no
language claim, without inference. This is not acoustic VAD or a claim that noisy
silence is safe: the model returned `[BLANK_AUDIO]` on a raw zero fixture before
that explicit signal classification. Bilingual/long/noise/silence/thermal/battery
and physical mobile app quality remain unqualified.
