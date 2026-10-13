"""Export independent NumPy features, encoder tensors and greedy token decisions."""
import argparse
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from onnx_asr.loader import Manager

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bundle", type=Path, required=True)
parser.add_argument("--reference", type=Path, required=True)
args = parser.parse_args()
options = ort.SessionOptions()
options.intra_op_num_threads = 4
options.inter_op_num_threads = 1
asr = Manager(providers=["CPUExecutionProvider"], sess_options=options,
              preprocessor_config={"use_numpy_preprocessors": True}).create_asr(
                  "nemo-conformer-tdt", args.bundle, offline=True).asr
receipt = json.loads((args.reference / "reference.json").read_text())
for record in receipt["records"]:
    wave = np.fromfile(args.reference / record["pcm"], dtype="<f4")
    features, lengths = asr._preprocessor(wave[None], np.array([len(wave)], np.int64))
    encoded, encoded_lengths = asr._encode(features, lengths)
    tokens, timestamps, _ = next(asr._decoding(encoded, encoded_lengths))
    record["token_ids"] = list(tokens)
    record["timestamps"] = list(timestamps)
    if record["id"] < 3 or record["id"] == 34:
        prefix = args.reference / f'{record["id"]:03}'
        features.astype("<f4").tofile(str(prefix) + ".features.f32")
        encoded.transpose(0, 2, 1).astype("<f4").tofile(str(prefix) + ".encoder.f32")
        states = asr._create_state()
        logits, duration, states = asr._decode([], states, encoded[0, 0])
        logits.astype("<f4").tofile(str(prefix) + ".logits.f32")
        states[0].astype("<f4").tofile(str(prefix) + ".state1.f32")
        states[1].astype("<f4").tofile(str(prefix) + ".state2.f32")
        record["first_duration"] = int(duration)
        record["first_token"] = int(logits.argmax())
    print(record["id"], flush=True)
(args.reference / "reference.json").write_text(json.dumps(receipt, indent=2) + "\n")
