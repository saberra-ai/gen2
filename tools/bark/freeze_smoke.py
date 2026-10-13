"""Freeze identified human labels/audio hashes, without redistributing audio."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("reference", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
reference = json.loads(args.reference.read_text())
fields = ["id", "sha256", "samples", "cleaned_reference", "text", "token_ids"]
fixture = {"model_revision": reference["model_revision"],
           "dataset_revision": reference["dataset_revision"],
           "dataset": "amaai-lab/DisfluencySpeech test, transcript_c, Apache-2.0",
           "provider": "CPUExecutionProvider", "sample_rate": 16000, "channels": 1,
           "oracle": reference["versions"],
           "selection": "First 25 rows, frozen independently of native results; supplement with critical-content.json and silence/noise fixtures.",
           "records": [{k: r[k] for k in fields} for r in reference["records"][:25]]}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(fixture, indent=2) + "\n")
