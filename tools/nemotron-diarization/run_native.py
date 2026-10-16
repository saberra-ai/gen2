"""Run the locked native C ABI without Gen2 for N0 smoke and parity evidence."""
import argparse
import ctypes
import json
from pathlib import Path
import wave

from artifacts import digest, matches


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shim", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--geometry", choices=["official", "upstream-example"], default="official")
    p.add_argument("--gpu", type=int, default=-1, help="GPU index; -1 requests CPU without fallback")
    a = p.parse_args()
    if a.gpu < -1:
        p.error("GPU index must be at least -1")
    lock = json.loads(Path(__file__).with_name("runtime.lock.json").read_text())
    for entry in lock["artifacts"]:
        if entry["role"] != "reference" and not matches(a.assets / entry["name"], entry):
            p.error(f"missing/corrupt native asset: {entry['name']}")
    a.output.mkdir(parents=True, exist_ok=True)
    ami = a.assets / "ami_en2002d_2132.wav"
    with wave.open(str(ami)) as src:
        pcm = src.readframes(src.getnframes())
    for name, samples in [("silence", bytes(16000 * 2 * 2)), ("tail", pcm[:17872 * 2])]:
        with wave.open(str(a.output / f"{name}.wav"), "wb") as out:
            out.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            out.writeframes(samples)
    library = ctypes.CDLL(str(a.shim.resolve()))
    library.n0_open.argtypes = [ctypes.c_char_p, ctypes.c_int]
    library.n0_open.restype = ctypes.c_void_p
    library.n0_close.argtypes = [ctypes.c_void_p]
    library.n0_close.restype = None
    library.n0_run.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    library.n0_run.restype = ctypes.c_int
    library.n0_last_error.argtypes = []
    library.n0_last_error.restype = ctypes.c_char_p
    open_model = library.n0_open
    arguments = []
    if a.gpu >= 0:
        open_model = library.n0_open_device
        open_model.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        open_model.restype = ctypes.c_void_p
        arguments = [a.gpu]
    model = open_model(str(a.assets / "Nemotron-3-Diarization.q8_0.gguf").encode(),
                            int(a.geometry == "official"), *arguments)
    if not model:
        raise RuntimeError(library.n0_last_error().decode())
    records = []
    try:
        for name, wav in [("ami", ami), ("silence", a.output / "silence.wav"),
                          ("tail", a.output / "tail.wav"), ("ami-repeat", ami)]:
            output = a.output / f"{name}.json"
            if library.n0_run(model, str(wav).encode(), str(output).encode()):
                raise RuntimeError(library.n0_last_error().decode())
            result = json.loads(output.read_text())
            if result["frames"] <= 0:
                raise ValueError(f"no frames after finish: {name}")
            if name == "silence" and result["segments"]:
                raise ValueError("silence produced speaker segments")
            records.append({"name": name, "wav_sha256": digest(wav), **result})
    finally:
        library.n0_close(model)
    if records[0]["segments"] != records[-1]["segments"]:
        raise ValueError("A–B–A native state contamination")
    receipt = {"records": records, "geometry": records[0]["geometry"], "push_samples": 2560,
               "runner_sha256": digest(Path(__file__)),
               "lock_sha256": digest(Path(__file__).with_name("runtime.lock.json")),
               "shim_sha256": digest(a.shim), "requested_gpu": a.gpu, "model_revision": lock["model"]["revision"]}
    (a.output / "native.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(a.output / "native.json")


if __name__ == "__main__":
    main()
