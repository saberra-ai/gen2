"""Freeze real Bark CPU transcripts and frontend parity for native Rust tests.

Install the versions in reference-requirements.txt into a dedicated environment.
Model weights and the test parquet are explicit local inputs; no implicit download.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import io
import json
import time
from pathlib import Path

import jiwer
import numpy as np
import onnx
import onnxruntime as ort
import pyarrow.parquet as pq
import soundfile as sf
import soxr
from onnx_asr.loader import Manager

MODEL_REVISION = "0d5f814105042fe801bb019dbf34b4e6f7d3eae6"
DATASET_REVISION = "b7da294fe3a70dd96df6640893f2a5dfc2c87638"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--export-assets", type=Path)
    parser.add_argument("--skip-candidate-frontend", action="store_true",
                        help="Release CI needs only the qualified NumPy oracle, not the rejected ONNX frontend audit")
    args = parser.parse_args()
    manifest = json.loads((args.bundle / "manifest.json").read_bytes())
    assert digest((args.bundle / "manifest.json").read_bytes()) == "8cd94f24f677b74f9e92306039491f23e3e22863205e59a5bdda322026f98f8c"
    assert digest(args.parquet.read_bytes()) == "e78837be393206fc0ae6a0cefb4ec8c942c3e26401640eb8af67cf0dac21b4b0", "corpus pin mismatch"
    for name, entry in manifest["files"].items():
        data = (args.bundle / name).read_bytes()
        assert len(data) == entry["bytes"] and digest(data) == entry["sha256"], name
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1

    def load(numpy):
        return Manager(
            providers=["CPUExecutionProvider"], sess_options=options,
            preprocessor_config={"providers": ["CPUExecutionProvider"],
                                 "max_concurrent_workers": 1,
                                 "use_numpy_preprocessors": numpy,
                                 "use_conv_preprocessors": False},
        ).create_asr("nemo-conformer-tdt", args.bundle, offline=True)

    reference = load(True)
    graph_reference = None if args.skip_candidate_frontend else load(False)
    # Observe the helper's own decisions during recognize(), without replacing
    # its encoder, decoder or text conversion. Avoid a second corpus decode.
    captured_decisions = []
    original_decoding = reference.asr._decoding
    def observe_decoding(*inputs):
        for result in original_decoding(*inputs):
            captured_decisions.append((list(result[0]), list(result[1])))
            yield result
    reference.asr._decoding = observe_decoding
    graph_inventory = {}
    for name in ("encoder-model.onnx", "decoder_joint-model.onnx"):
        graph = onnx.load(args.bundle / name, load_external_data=False)
        graph_inventory[name] = {
            "opsets": {o.domain: o.version for o in graph.opset_import},
            "operators": sorted({(n.domain, n.op_type) for n in graph.graph.node}),
            "inputs": [{"name": t.name, "type": str(t.type)} for t in graph.graph.input],
            "outputs": [{"name": t.name, "type": str(t.type)} for t in graph.graph.output],
        }
    args.output.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(args.parquet).to_pylist()[:args.limit]
    records, reference_times, graph_times = [], [], []
    frontend_max_error = 0.0
    for index, row in enumerate(rows):
        original, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=True)
        wave = original.mean(axis=1)
        if rate != 16000:
            wave = soxr.resample(wave, rate, 16000, quality="HQ")
        wave = np.asarray(wave, dtype=np.float32)
        assert np.isfinite(wave).all() and 320 <= len(wave) <= 960000
        inputs = (wave[None], np.array([len(wave)], dtype=np.int64))
        f1, l1 = reference.asr._preprocessor(*inputs)
        err = None
        if graph_reference is not None:
            f2, l2 = graph_reference.asr._preprocessor(*inputs)
            assert np.array_equal(l1, l2) and f1.shape == f2.shape
            err = float(np.max(np.abs(f1-f2)))
            frontend_max_error = max(frontend_max_error, err)
        captured_decisions.clear()
        started = time.perf_counter()
        text = reference.recognize(wave, sample_rate=16000).strip()
        reference_times.append(time.perf_counter() - started)
        assert len(captured_decisions) == 1, "expected one helper decoding result per recording"
        tokens, timestamps = captured_decisions[0]
        graph_text = None
        if graph_reference is not None:
            started = time.perf_counter()
            graph_text = graph_reference.recognize(wave, sample_rate=16000).strip()
            graph_times.append(time.perf_counter() - started)
        pcm = wave.astype("<f4").tobytes()
        (args.output / f"{index:03}.f32").write_bytes(pcm)
        sf.write(args.output / f"{index:03}.wav", wave, 16000, subtype="FLOAT")
        if index < 3:
            (args.output / f"{index:03}.features.f32").write_bytes(f1.astype("<f4").tobytes())
        records.append({"id": index, "pcm": f"{index:03}.f32", "sha256": digest(pcm),
                        "samples": len(wave), "text": text, "graph_text": graph_text,
                        "cleaned_reference": row["transcript_c"], "frontend_max_error": err,
                        "token_ids": tokens, "timestamps": timestamps})
        print(f"{index+1}/{len(rows)} candidate_error={err}", flush=True)
    normalize = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
                              jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.ReduceToListOfListOfWords()])
    def score(key):
        return jiwer.process_words([r["cleaned_reference"] for r in records],
                                   [r[key] for r in records],
                                   reference_transform=normalize, hypothesis_transform=normalize).wer
    receipt = {"model_revision": MODEL_REVISION, "dataset_revision": DATASET_REVISION,
               "versions": {p: metadata.version(p) for p in
                            ("numpy", "onnx-asr", "onnxruntime", "soundfile", "soxr", "onnx", "jiwer", "pyarrow")},
               "manifest_sha256": digest((args.bundle / "manifest.json").read_bytes()),
               "provider": "CPUExecutionProvider", "intra_threads": 4, "inter_threads": 1,
               "graphs": graph_inventory, "candidate_frontend_evaluated": graph_reference is not None,
               "frontend_max_error": frontend_max_error if graph_reference is not None else None,
               "graph_transcript_matches": sum(r["text"] == r["graph_text"] for r in records) if graph_reference is not None else None,
               "wer": score("text"), "graph_wer": score("graph_text") if graph_reference is not None else None,
               "reference_seconds": reference_times, "graph_seconds": graph_times,
               "records": records}
    (args.output / "reference.json").write_text(json.dumps(receipt, indent=2)+"\n")
    if args.export_assets:
        import onnx_asr
        root = Path(onnx_asr.__file__).parent
        args.export_assets.mkdir(parents=True, exist_ok=True)
        (args.export_assets / "nemo128.onnx").write_bytes((root / "preprocessors/data/nemo128.onnx").read_bytes())
        # Native DSP route: the CPU reference coefficients, in a fixed binary layout.
        with np.load(root / "preprocessors/data/fbanks.npz") as filters:
            matrix = filters["nemo128"]
            assert matrix.shape == (257, 128) and matrix.dtype == np.float32
            (args.export_assets / "nemo128.f32").write_bytes(matrix.astype("<f4").tobytes())
    print(json.dumps({k: receipt[k] for k in ("versions", "frontend_max_error", "graph_transcript_matches", "wer", "graph_wer")}, indent=2))


if __name__ == "__main__":
    main()
