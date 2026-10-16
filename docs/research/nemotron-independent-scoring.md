# Independent scoring of saved Nemotron segments

Read-only research on2026-10-16. No scoring, installs or inference performed. This supports a separately frozen diagnostic beside the original failed qualification; it does not change its labels, settings or result.

## Exact official implementation

At NeMo Speech commit `218d0b443441b7d8aa1c832f47df310f186cf0a4`, evaluation calls `nemo.collections.asr.metrics.der.score_labels`. That function uses **NeMo's own Python port of NIST md-eval22**, not `pyannote.metrics`. Lhotse is used for annotation carrier objects in the high-level wrapper; its lower-level scoring engine accepts parsed RTTM dictionaries directly. [Evaluation wrapper](https://github.com/NVIDIA-NeMo/Speech/blob/218d0b443441b7d8aa1c832f47df310f186cf0a4/nemo/collections/asr/metrics/der.py), [actual engine](https://github.com/NVIDIA-NeMo/Speech/blob/218d0b443441b7d8aa1c832f47df310f186cf0a4/nemo/collections/asr/metrics/md_eval.py).

Engine source to freeze:

- [Raw md_eval.py](https://raw.githubusercontent.com/NVIDIA-NeMo/Speech/218d0b443441b7d8aa1c832f47df310f186cf0a4/nemo/collections/asr/metrics/md_eval.py).
- **54,914bytes**, SHA256 `c34af6eeec07e1df795c6820f36cb05459c28ac8d5d84fb0efea433035546de3` (verified against the downloaded source bytes).
- Imports: stdlib `re`, `warnings`, `typing`; `numpy`; `scipy.optimize.linear_sum_assignment`; and `nemo.utils.logging`.
- The algorithm/source file can remain unmodified. A dedicated fresh Python process can supply **only** a stdlib logging substitute for the final import, avoiding NeMo package initialization, Torch and Lhotse. This is the official low-level engine with a local import adapter, **not execution of the complete official NeMo package**.

API:

```python
ref, hyp = {}, {}
engine.get_rttm_data(ref, reference_rttm_path)  # mutates nested dictionaries
engine.get_rttm_data(hyp, hypothesis_rttm_path)
uem = engine.get_uem_data(uem_path)
all_scores, cumulative = engine.evaluate(
    ref, hyp, uem_data=uem, collar=0.0, opt_1=False, verbose=False
)
```

`opt_1=False` includes overlap. `collar` is a boundary half-width; zero removes width-convention ambiguity. `uem_data` must explicitly describe the full120s target; otherwise this lower-level engine derives extent from references. This is a continuous-interval engine, independent of the local10ms-midpoint scorer. Its Hungarian speaker mapping uses SciPy. [Engine implementation and evaluate contract](https://github.com/NVIDIA-NeMo/Speech/blob/218d0b443441b7d8aa1c832f47df310f186cf0a4/nemo/collections/asr/metrics/md_eval.py), [official evaluation reporting/UEM convention](https://huggingface.co/nvidia/Nemotron-3-Diarization/raw/f667ed73aee57d40cc39428eb768b4fd87a0a29e/diarization_evaluation.md).

## Minimal runnable adapter, source-reviewed and not executed here

Input files should already contain the frozen saved segments in RTTM, with no model loading or new threshold decision. Use one simple shared recording ID, for example `diagnostic_ami_close`, and channel `1` in both RTTMs. Each RTTM speaker line is:

```text
SPEAKER diagnostic_ami_close 1 START DURATION <NA> <NA> SPEAKER_ID <NA> <NA>
```

Create a separate immutable UEM file containing exactly:

```text
diagnostic_ami_close 1 0.000 120.000
```

Use a recording ID without dots/path suffixes: the source UEM parser strips filename extensions, whereas its RTTM parser preserves the file ID. Preserve segment precision when serializing and record that serialization rule; do not introduce a new10ms rounding step. For a crop, reference and hypothesis timestamps must both be relative to the same crop origin. Keep source-reference variants in separately named files/protocols.

The following adapter imports the hash-verified source unchanged. Run it in a dedicated process, never inside an existing application with NeMo already imported:

```python
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
from pathlib import Path
import sys
import types

p = argparse.ArgumentParser()
p.add_argument("engine")
p.add_argument("reference")
p.add_argument("hypothesis")
p.add_argument("uem")
a = p.parse_args()
source = Path(a.engine).read_bytes()
expected = "c34af6eeec07e1df795c6820f36cb05459c28ac8d5d84fb0efea433035546de3"
assert len(source) == 54914 and hashlib.sha256(source).hexdigest() == expected
assert "nemo" not in sys.modules and "nemo.utils" not in sys.modules
nemo_stub = types.ModuleType("nemo")
nemo_stub.__path__ = []
utils_stub = types.ModuleType("nemo.utils")
utils_stub.logging = logging
nemo_stub.utils = utils_stub
sys.modules["nemo"] = nemo_stub
sys.modules["nemo.utils"] = utils_stub
spec = importlib.util.spec_from_file_location("pinned_nemo_md_eval", a.engine)
engine = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = engine
spec.loader.exec_module(engine)
ref, hyp = {}, {}
engine.get_rttm_data(ref, a.reference)
engine.get_rttm_data(hyp, a.hypothesis)
uem = engine.get_uem_data(a.uem)
assert set(ref) == {"diagnostic_ami_close"}
assert set(hyp) <= set(ref)  # Empty hypothesis is valid; extra recordings are not.
assert set(uem) == set(ref)
region = uem["diagnostic_ami_close"]["1"]
assert len(region) == 1 and region[0]["TBEG"] == 0 and region[0]["TEND"] == 120
scores, cum = engine.evaluate(ref, hyp, uem_data=uem, collar=0.0,
                              opt_1=False, verbose=False)
denominator = cum["SCORED_SPEAKER"]
if denominator <= engine.EPSILON:
    raise ValueError("No reference speaker time inside UEM")
miss, fa, confusion = (cum[k] for k in
    ["MISSED_SPEAKER", "FALARM_SPEAKER", "SPEAKER_ERROR"])
print(json.dumps({
    "engine_sha256": expected,
    "numpy": importlib.metadata.version("numpy"),
    "scipy": importlib.metadata.version("scipy"),
    "collar": 0.0, "ignore_overlap": False,
    "uem_seconds": [0, 120], "official_package_imported": False,
    "logging_adapter": "stdlib logging only; scoring source unchanged",
    "scored_speaker_seconds": denominator,
    "missed_speaker_seconds": miss, "false_alarm_speaker_seconds": fa,
    "confusion_speaker_seconds": confusion,
    "der": (miss + fa + confusion) / denominator,
    "miss": miss / denominator, "fa": fa / denominator,
    "cer": confusion / denominator,
    "input_sha256": {f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
                     for f in [a.reference, a.hypothesis, a.uem]},
    "cumulative": cum
}, indent=2))
```

The calculation of DER/MISS/FA/CER above follows `der.py::_extract_errors`: `SCORED_SPEAKER` is the denominator; `SPEAKER_ERROR` is confusion, not character error. Input/output IDs, explicit UEM, logger shim and preservation of source bytes must be documented. This adapter bypasses high-level result formatting, Lhotse conversion and speaker-count summary—not the core DER algorithm. Source inspection establishes the route; actual execution and scorer parity remain to be validated. [Official metric extraction](https://github.com/NVIDIA-NeMo/Speech/blob/218d0b443441b7d8aa1c832f47df310f186cf0a4/nemo/collections/asr/metrics/der.py).

## Dependency pins: distinguish official resolution from the existing oracle environment

The pinned [NeMo uv.lock](https://github.com/NVIDIA-NeMo/Speech/blob/218d0b443441b7d8aa1c832f47df310f186cf0a4/uv.lock) is5,104,389bytes, SHA256 `dcdf1d54559ef26ee796d4e2afc943db95659aa7b8be8cbf0af431d07d2eff5e`. It contains multiple platform/extra resolutions; inspecting its markers for Python3.12/macOS/arm64 selects **NumPy2.4.4 and SciPy1.17.1**, while lower versions2.2.6/1.15.3 serve other markers. Lhotse2.0.0a6 is present for the full high-level path. No pyannote-metrics/core package is required by the exact low-level engine.

The existing independently locked FP32-oracle environment reports **NumPy2.2.6/SciPy1.18.1**. Reusing those explicitly pinned installed versions is a reasonable separately frozen diagnostic choice with no new installation, but it is **not a claim to reproduce the complete official NeMo dependency lock**. Record installed versions and the actual lock hash; if dependency-sensitive mapping differences emerge, compare against the official resolved versions in a separate isolated environment later. Do not mutate the original inference environment or its receipts.

## Optional second implementation: pinned NIST-origin Perl scorer

The NeMo source attributes its port to NIST md-eval22 as mirrored in `nryant/dscore`. This provides a second implementation with no NumPy/SciPy dependency:

- dscore commit `e02f949ac6592279300a2c33d03daf9e0c12fd27`.
- [Raw md-eval-22.pl](https://raw.githubusercontent.com/nryant/dscore/e02f949ac6592279300a2c33d03daf9e0c12fd27/scorelib/md-eval-22.pl),109,574bytes, SHA256 `872aa955cbc3d57e6d4fd6fe2e699791739c1cb716cc89105bb4717415b9400d`.
- This is the NIST-origin source maintained/mirrored by dscore, not a claim that this GitHub commit is an official current NIST distribution. Retain upstream notices if vendoring it; the NeMo port itself carries Apache2.0 and the inherited dscore BSD notice.

Candidate command, **not run here**:

```sh
perl md-eval-22.pl -r reference.rttm -s hypothesis.rttm -u full120.uem -c 0
```

Omit `-1`, which restricts scoring to single-speaker regions. The `-o` flag concerns metadata evaluation, not the switch needed to include overlap in speaker DER. The source uses Perl's core `Getopt::Std`; no Python package or model is needed. Pin the Perl executable/version and script bytes. Preserve raw component seconds rather than comparing only its rounded overall percentage. [Pinned option definitions and component reporting](https://github.com/nryant/dscore/blob/e02f949ac6592279300a2c33d03daf9e0c12fd27/scorelib/md-eval-22.pl).

A pyannote.metrics invocation would be another independent metric implementation, but it is **not the actual scorer at the pinned NeMo commit**. Prioritize the unmodified official port with explicit UEM, then the Perl implementation if a cross-check is needed. No new scoring route changes the original frozen reference-policy failure or authorizes threshold search.
