"""Independent CPU/FP32 oracle using the official original safetensors and Transformers."""
import argparse
import json
from pathlib import Path
import platform
import time

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
    audio, rate = sf.read(args.wav, dtype="float32")
    if rate != 16000 or audio.ndim != 1 or not 0 < audio.size <= 16000 * 120 or not np.isfinite(audio).all():
        parser.error("oracle accepts finite mono 16 kHz audio, up to 120 seconds")
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
    if probabilities.ndim != 2 or probabilities.shape[1] != 8 or not np.isfinite(probabilities).all():
        raise ValueError("invalid FP32 reference output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_suffix(".npy"), probabilities)
    metadata = {"model_revision": lock["model"]["revision"], "wav_sha256": digest(args.wav),
                "mode": "low_latency", "device": "cpu", "dtype": "float32", "threads": 4,
                "torch": torch.__version__, "transformers": transformers.__version__,
                "python": platform.python_version(), "platform": platform.platform(),
                "audio_samples": len(audio), "shape": list(probabilities.shape),
                "seconds": time.perf_counter() - started, "cadence_seconds": .01}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
