"""Collect existing static source evidence only; never execute a native program."""
import json,hashlib,shutil
from pathlib import Path
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization'
source=cache/'valid-mel-platform-source-preflight-v2';target=repo/'benches/results/nemotron-diarization/cross-platform/2026-10-16-valid-mel-source-preflight'
assert not target.exists();target.mkdir(parents=True)
def desc(path):return {'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
for path in source.iterdir():
 if path.is_file() and path.name!='platform-pristine-llama.tar':shutil.copy2(path,target/path.name)
for path in [cache/'preflight-platform-valid-mel-source-v2.py',Path(__file__)]:shutil.copy2(path,target/path.name)
a=json.loads((target/'platform-llama-materialized-audit.json').read_text());commands=json.loads((target/'commands.json').read_text())
assert len(a['patch_series'])==19 and len(a['files'])==3425 and len(a['actual_compiler_sources'])==28 and len(a['all_compiler_sources'])==98
assert len(commands)==20 and all(c['returncode']==0 for c in commands)
record={'schema_version':1,'static_source_audit_passed':True,'release_qualified':False,'scope':'Model-free algorithm preflight against saved tagged macOS CPU compiler tree; not a fresh Linux/Windows build, version query, helper or neural qualification.','source_tree':'/Users/victor/.cache/gen2-nemotron3-n0/NeMo-Speech-valid-mel-tagged-v1','patches':19,'materialized_files':3425,'actual_materialized_compiler_files':28,'all_compiler_files':98,'commands':20,'new_native_or_neural_calls':0,'auditor':desc(tools/'platform_valid_mel_sources.py'),'archive_payload_omitted':a['archive'],'limits':['Audit occurs after the saved SDK construction; not a pre-compilation measurement.','Platform candidate builds and actual runtime qualification remain pending.'],'files':{p.name:desc(p)for p in sorted(target.iterdir())if p.is_file()}}
(target/'qualification.json').write_text(json.dumps(record,indent=2)+'\n');print(target)
