"""Archive local run artifacts without copying model weights or private inputs.
Only the repository's fixed test corpus and measured shape reports are used.
"""
import hashlib,json,platform
from pathlib import Path

root=Path("target/laya");out=Path("tools/laya/evidence/windows-x64");out.mkdir(parents=True,exist_ok=True)
for checkpoint in ["english","multilingual","typed-decisions"]:
    raw=(root/checkpoint/"manifest.json").read_bytes();digest=hashlib.sha256(raw).hexdigest()
    (out/f"{checkpoint}-manifest.json").write_bytes(raw)
    for mode in ["eager","onnx"]:
        path=root/f"{checkpoint}-{mode}-reference.json"
        reference=json.loads(path.read_text(encoding="utf-8"))
        # Earlier runs predated metadata stamping; bind them to their unchanged
        # local manifest when archiving. Inference numbers are left untouched.
        reference.update(checkpoint=checkpoint,manifest_sha256=digest)
        suffix="" if checkpoint=="english" else "-"+checkpoint
        preprocessing=json.loads(Path(f"tests/fixtures/laya/preprocessing{suffix}.json").read_text(encoding="utf-8"))
        corpus={case["name"]:case for case in preprocessing["cases"]}
        for case in reference["cases"]:
            expected=corpus[case["name"]]
            if case["request"]!=expected["request"] or case["options"]!=expected["options"]:raise ValueError("corpus changed since reference run")
            case["inputs"]=expected["expected"]
        (out/path.name).write_text(json.dumps(reference,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    path=root/f"{checkpoint}-windows-cpu.json"
    report=json.loads(path.read_text());report["processor"]=platform.processor()
    (out/path.name).write_text(json.dumps(report,indent=2)+"\n")
print(f"Archived manifests, reference outputs and host measurements in {out}")
