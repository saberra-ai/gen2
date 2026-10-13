"""Export helper-normalized PCM for an already frozen supplementary WAV corpus."""
import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import soxr

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path)
args = parser.parse_args()
for record in json.loads((args.root / "audio-reference.json").read_text())["records"]:
    path = args.root / record["wav"]
    wave, rate = sf.read(path, dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)
    if rate != 16000:
        wave = soxr.resample(wave, rate, 16000, quality="HQ")
    wave.astype("<f4").tofile(path.with_suffix(".normalized.f32"))
