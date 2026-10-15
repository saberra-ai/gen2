"""Check raw-text/token/timestamp parity and publish quality/timing evidence."""
import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
import jiwer
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--native", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
reference = json.loads(args.reference.read_text())
native = json.loads(args.native.read_text())
assert len(native) == len(reference["records"])
normalize = jiwer.Compose([jiwer.ToLowerCase(),jiwer.RemovePunctuation(),jiwer.RemoveMultipleSpaces(),jiwer.Strip(),jiwer.ReduceToListOfListOfWords()])
score = jiwer.process_words([r["cleaned_reference"] for r in reference["records"]], [r["fulltext"] for r in native],reference_transform=normalize,hypothesis_transform=normalize)
text_matches = sum(a["fulltext"] == b["text"] for a,b in zip(native,reference["records"]))
token_matches = sum(a["token_ids"] == b["token_ids"] for a,b in zip(native,reference["records"]))
timestamp_matches = sum(
    len(a["timestamps"]) == len(a["token_ids"])
    and [t["startms"] for t in a["timestamps"]] == [frame * 80 for frame in b["timestamps"]]
    and [t["endms"] for t in a["timestamps"]] == [frame * 80 for frame in b["timestamps"][1:]] + ([b["samples"] // 16] if b["timestamps"] else [])
    and "".join(t["text"] for t in a["timestamps"]) == a["fulltext"]
    for a, b in zip(native, reference["records"])
)
times = np.array([r["execution_micros"]/1e6 for r in native])
receipt = {"git_sha":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
           "model_revision":reference["model_revision"],"manifest_sha256":reference["manifest_sha256"],
           "dataset_revision":reference["dataset_revision"],"versions":reference["versions"],
           "machine":platform.platform(),"provider":"CPUExecutionProvider","intra_threads":4,"inter_threads":1,
           "normalizer":"lowercase-remove-punctuation-collapse-space-v1", "clips":len(native),
           "raw_text_matches":text_matches,"token_matches":token_matches,"token_timestamp_matches":timestamp_matches,
           "wer":score.wer,"substitutions":score.substitutions,"deletions":score.deletions,"insertions":score.insertions,
           "reference_wer":reference["wer"],"median_seconds":float(np.median(times)),"p95_seconds":float(np.quantile(times,.95)),
           "median_native_to_python_ratio":float(np.median(times/np.array(reference["reference_seconds"]))),
           "reference_sha256":hashlib.sha256(args.reference.read_bytes()).hexdigest(),
           "native_sha256":hashlib.sha256(args.native.read_bytes()).hexdigest()}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))
assert text_matches == len(native) and token_matches == len(native) and timestamp_matches == len(native), "parity gate failed"
assert score.wer <= reference["wer"] + 1e-12, "accuracy regression"
