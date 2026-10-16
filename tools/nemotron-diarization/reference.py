"""Independent CPU/FP32 oracle using the official original safetensors and Transformers."""
import argparse
import json
from pathlib import Path
import platform
import time
import importlib.metadata

import numpy as np
import soundfile as sf
import torch
import transformers
from transformers import AutoModelForAudioFrameClassification, AutoProcessor

from artifacts import digest, matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(Path(__file__).with_name("runtime.lock.json").read_text())
    for entry in lock["artifacts"]:
        if entry["role"] == "reference" and not matches(args.assets / entry["name"], entry):
            parser.error(f"missing/corrupt oracle asset: {entry['name']}")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    info = sf.info(args.wav)
    if info.samplerate != 16000 or info.channels != 1 or not 320 <= info.frames <= 16000 * 600:
        parser.error("oracle accepts finite mono 16 kHz audio, from 20 ms to 600 seconds")
    if args.output.with_suffix('.npy').exists() or args.output.with_suffix('.json').exists():
        parser.error("refusing to overwrite existing reference evidence")
    audio, rate = sf.read(args.wav, dtype="float32", frames=info.frames + 1)
    if rate != 16000 or audio.ndim != 1 or audio.size != info.frames or not np.isfinite(audio).all():
        parser.error("audio changed or contains invalid samples")
    processor = AutoProcessor.from_pretrained(args.assets, local_files_only=True)
    model = AutoModelForAudioFrameClassification.from_pretrained(
        args.assets, local_files_only=True, dtype=torch.float32).to("cpu").eval()
    processor.set_streaming_mode("low_latency")
    cache = None
    parts = []

    def infer(chunk, first, last):
        nonlocal cache
        inputs = processor(chunk, sampling_rate=rate, is_streaming=True,
                           is_first_audio_chunk=first, is_last_audio_chunk=last,
                           return_tensors="pt")
        with torch.inference_mode():
            result = model(**inputs, speaker_cache=cache)
        cache = result.speaker_cache
        parts.append(result.logits)

    started = time.perf_counter()
    first_size = processor.num_samples_first_audio_chunk
    if audio.size <= first_size:
        infer(audio, True, True)
    else:
        infer(audio[:first_size], True, False)
        cursor = processor.num_mel_frames_per_step
        while True:
            start = processor.audio_chunk_start(cursor)
            if start >= audio.size:
                raise ValueError("oracle did not flush its final audio chunk")
            end = start + processor.num_samples_per_audio_chunk
            last = end >= audio.size
            infer(audio[start:min(end, audio.size)], False, last)
            if last:
                break
            cursor += processor.num_mel_frames_per_step
    probabilities = torch.cat(parts, dim=1).sigmoid().squeeze(0).cpu().numpy().astype(np.float32)
    # The pinned processor's centered first chunk and uncentered padded final
    # chunk both mask to floor(N/160) real frames, including partial 10 ms tails.
    if probabilities.shape != (len(audio) // 160, 8) or not np.isfinite(probabilities).all() or not ((0 <= probabilities) & (probabilities <= 1)).all():
        raise ValueError("invalid FP32 reference output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_suffix(".npy"), probabilities)
    metadata = {"model_revision": lock["model"]["revision"], "wav_sha256": digest(args.wav),
                "script_sha256": digest(Path(__file__)),
                "lock_sha256": digest(Path(__file__).with_name("runtime.lock.json")),
                "environment_lock_sha256": digest(Path(__file__).with_name("reference") / "uv.lock"),
                "transformers_direct_url": json.loads(importlib.metadata.distribution("transformers").read_text("direct_url.json")),
                "installed_packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
                "geometry": {"chunk": processor.streaming_modes[processor.streaming_mode][0],
                             "right": processor.streaming_modes[processor.streaming_mode][1], "left": 0,
                             "fifo": model.config.streaming_config.fifo_length,
                             "cache": model.config.streaming_config.speaker_cache_length,
                             "update": model.config.streaming_config.speaker_cache_update_period},
                "mode": "low_latency", "device": "cpu", "dtype": "float32", "threads": 4,
                "torch": torch.__version__, "transformers": transformers.__version__,
                "python": platform.python_version(), "platform": platform.platform(),
                "audio_samples": len(audio), "shape": list(probabilities.shape),
                "valid_frame_rule": "floor(audio_samples/160); pinned processor attention mask; native extra terminal frame retained separately",
                "seconds": time.perf_counter() - started, "cadence_seconds": .01}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
