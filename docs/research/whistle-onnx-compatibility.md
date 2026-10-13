# Whistle ONNX compatibility

Reviewed 2026-10-13. Hugging Face revision: `14fc92cb09ad5c271fe9a4ec4be70274790050d2`. No weights downloaded and no inference run.

## Conclusion

Whistle is an ASR model, not speech synthesis and not the Bark model targeted by the current Gen2 ONNX work. It can share an ONNX runtime dependency, audio capture/resampling, workers and admission infrastructure, but needs its own model bundle, frontend, decoding implementation, and result conversion. The current Bark contract cannot load Whistle by substituting filenames. [Whistle model card](https://huggingface.co/mrfakename/whistle-ONNX/blob/14fc92cb09ad5c271fe9a4ec4be70274790050d2/README.md)

Gen2 source inspected: `/Users/victor/workspace/gen2-bark-onnx`, branch `codex/bark-onnx`, commit `8277263`. [Bark provisioning](/Users/victor/workspace/gen2-bark-onnx/src/speech/bundle.rs:9) pins its repository/revision/manifest digest, while [ONNX execution](/Users/victor/workspace/gen2-bark-onnx/src/speech/onnx.rs:77) hardcodes Bark frontend/TDT signatures. Supporting Whistle requires a distinct model adapter and provisioning contract, followed by real-audio parity tests against its reference pipeline.

## Runtime contract

| Component | Whistle | Ongoing Gen2 Bark contract |
| --- | --- | --- |
| Frontend | 16 kHz mono PCM → normalized 80-bin log-mel `[T,80]`; FFT 512, window 400, hop 160 | 128-bin features `[1,128,T]` |
| Encoder output | Cross-attention keys `[8,8,S,48]` and values `[8,8,S,64]` | Acoustic features `[1,1024,T]` |
| Decoder state | `prev [8,2,608]`; growing K/V caches `[8,2,P,48]`, `[8,2,P,64]` | Recurrent state `[2,1,640]` |
| Other decoder inputs | Token and position; host-computed engram features `[2,4,512]`; encoder K/V | Bark decoder/joint inputs |
| Logits | 8,199: 8,192 text/special IDs plus seven language IDs | 8,193 vocabulary logits plus five TDT duration logits |
| Decode | Autoregressive cross-attention, BOS → language → greedy text until EOS | TDT frame advancement/duration decoding |
| Word timing | DTW over decoder cross-attention, 80 ms frames | TDT alignment |

Whistle's graph export uses ONNX opset 17 and dynamic audio/token-cache lengths. The host must implement the mel frontend, SentencePiece-style BPE with byte fallback, engram hash/table lookup, state handling, language selection, greedy decoding, and optional keyword biasing/DTW. The supplied JavaScript does this; the ONNX graphs alone are insufficient. [Export script](https://huggingface.co/mrfakename/whistle-ONNX/blob/14fc92cb09ad5c271fe9a4ec4be70274790050d2/scripts/export_onnx.py), [host pipeline](https://huggingface.co/mrfakename/whistle-ONNX/blob/14fc92cb09ad5c271fe9a4ec4be70274790050d2/js/whistle.js), [backend](https://huggingface.co/mrfakename/whistle-ONNX/blob/14fc92cb09ad5c271fe9a4ec4be70274790050d2/js/backend.js)

## Packaging and supported paths

The published Node example uses `onnxruntime-node`, two sessions, FP32 encoder/decoder graphs with external `.data` weights, and `whistle.pack` for tokenizer, mel filterbank, and engram data. It accepts 16 kHz mono 16-bit WAV and returns transcript, language, word start/end/probability, token IDs, and timings. The browser demo uses `onnxruntime-web`, FP16 WebGPU encoding and WASM decoding. This is evidence that ONNX Runtime can run the export, not evidence that a particular Rust ORT build/provider has been validated. [Example](https://huggingface.co/mrfakename/whistle-ONNX/blob/14fc92cb09ad5c271fe9a4ec4be70274790050d2/js/example.mjs), [model card](https://huggingface.co/mrfakename/whistle-ONNX/blob/14fc92cb09ad5c271fe9a4ec4be70274790050d2/README.md)

The 17 MB packed download reconstructs dequantized ONNX weight data; FP32 graphs total about 145 MB before runtime state and overhead. The small pack is not a 17 MB inference-memory claim. Apache-2.0; seven languages: English, German, French, Spanish, Italian, Dutch, Polish. [Model card](https://huggingface.co/mrfakename/whistle-ONNX/blob/14fc92cb09ad5c271fe9a4ec4be70274790050d2/README.md)

## Limits of the evidence

The exporter reports matching greedy transcripts on four reference clips, with approximate word probabilities and timestamps. Keyword biasing approximates the original engine. Its supplied `transcribe` function processes a whole PCM buffer, defaults to a maximum of 320 generated tokens, and supplies no live-stream callback or long-audio windowing. Those features would need additional orchestration. The original Cactus engine advertises long-audio windowing, streaming and embeddings, but these should not be assumed to exist in the ONNX wrapper. [ONNX model card](https://huggingface.co/mrfakename/whistle-ONNX/blob/14fc92cb09ad5c271fe9a4ec4be70274790050d2/README.md), [host pipeline](https://huggingface.co/mrfakename/whistle-ONNX/blob/14fc92cb09ad5c271fe9a4ec4be70274790050d2/js/whistle.js), [original model](https://huggingface.co/Cactus-Compute/whistle)
