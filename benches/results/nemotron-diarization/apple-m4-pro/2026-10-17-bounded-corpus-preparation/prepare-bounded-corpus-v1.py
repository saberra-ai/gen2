"""Offline file preparation only; no compiler, SDK, helper or model execution."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import types


def descriptor(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


parser = argparse.ArgumentParser()
parser.add_argument('--protocol', type=Path, required=True)
parser.add_argument('--sha256', required=True)
args = parser.parse_args()
raw = args.protocol.read_bytes()
assert hashlib.sha256(raw).hexdigest() == args.sha256
pre = json.loads(raw)
for name, expected in pre['files'].items():
    assert descriptor(Path(name)) == expected, name
repo, cache = Path(pre['repo']), Path(pre['cache'])
tools = repo / 'tools/nemotron-diarization'
read = lambda path: json.loads(path.read_bytes())
old = read(tools / 'valid-mel-corpus.protocol.json')
app = read(tools / 'macos-bounded-pnola-app.protocol.json')
old_work, new_work = Path(old['work_path']), Path(app['work_path'])
stage = cache / 'bounded-corpus-inputs-v1'
assert not stage.exists()
# Execute only the pinned pure stdlib inventory function, not a project module.
inventory_source = tools / 'sdk_archive.py'
assert descriptor(inventory_source) == app['tools']['sdk_archive.py']
node, = [node for node in ast.parse(inventory_source.read_bytes()).body
         if isinstance(node, ast.FunctionDef) and node.name == 'inventory']
scope = {'Path': Path, 'hashlib': hashlib, 'stat': stat}
exec(compile(ast.Module(body=[node], type_ignores=[]), 'admitted-original-filesystem-inventory', 'exec'), scope)
inventory = scope['inventory']


def subset(records, prefix):
    result = {'.': records[prefix]}
    result.update({name[len(prefix) + 1:]: value for name, value in records.items() if name.startswith(prefix + '/')})
    return result


source_inputs = {'corpus': (old_work / 'corpus', subset(old['inventory'], 'corpus')),
                 'oracle': (old_work / 'oracle', subset(old['inventory'], 'oracle')),
                 'cpu/nemo': (new_work / 'nemo', subset(app['work_inventory'], 'nemo')),
                 'cpu/bin': (new_work / 'bin', subset(app['work_inventory'], 'bin'))}
for name, (path, wanted) in source_inputs.items():
    assert inventory(path) == wanted, name
assert descriptor(new_work / 'assets/model.gguf') == {key: app['work_inventory']['assets/model.gguf'][key] for key in ('bytes', 'sha256')}
for name, expected in {**old['preparation_records'], **app['preparation_records']}.items():
    assert descriptor(Path(name)) == expected, name
helper = tools / 'remote_q8_apack_replay.py'
helper_raw = helper.read_bytes()
assert len(helper_raw) == 22591 and hashlib.sha256(helper_raw).hexdigest() == '6b7f4c6d3a9a672ad0a89c790fa0d51608adad4c3667395c2da1329c68e48b08'
module = types.ModuleType('admitted_copy_supervisor')
module.__file__ = str(helper)
exec(compile(helper_raw, str(helper), 'exec'), module.__dict__)
stage.mkdir()
work = stage / 'work'
work.mkdir()
(work / 'cpu').mkdir()
(work / 'assets').mkdir()
commands = []
for name, (path, wanted) in source_inputs.items():
    row = module.bounded_process(['/bin/cp', '-cRpP', str(path), str(work / name)],
        stage / (name.replace('/', '-') + '.stdout.log'), stage / (name.replace('/', '-') + '.stderr.log'),
        cwd=repo, env={'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'}, timeout=120, limit=1024**2)
    commands.append(row)
    assert inventory(work / name) == wanted
model = work / 'assets/Nemotron-3-Diarization.q8_0.gguf'
row = module.bounded_process(['/bin/cp', '-cp', str(new_work / 'assets/model.gguf'), str(model)],
    stage / 'model.stdout.log', stage / 'model.stderr.log', cwd=repo,
    env={'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'}, timeout=120, limit=1024**2)
commands.append(row)
assert descriptor(model) == descriptor(new_work / 'assets/model.gguf')
for name, (path, wanted) in source_inputs.items():
    assert inventory(path) == inventory(work / name) == wanted
assert descriptor(new_work / 'assets/model.gguf') == {key: app['work_inventory']['assets/model.gguf'][key] for key in ('bytes', 'sha256')}
manifest = inventory(work)
for name, expected in pre['files'].items():
    assert descriptor(Path(name)) == expected, name
preparation = {'schema_version': 1, 'prepared': True, 'models_loaded': False,
    'compiler_helper_or_neural_calls': 0, 'commands': commands, 'inventory': manifest,
    'source_inputs': {name: {'path': str(path), 'inventory': wanted} for name, (path, wanted) in source_inputs.items()},
    'model': descriptor(model), 'source_bootstrap': descriptor(args.protocol),
    'copy_semantics': 'cp -c requests copy-on-write; exact regular/alias/mode inventories establish the copied bytes, not a claim of physical storage savings or isolation from privileged writers.'}
(stage / 'preparation.json').write_text(json.dumps(preparation, indent=2) + '\n')
records = {**old['preparation_records'], **app['preparation_records']}
long = cache / 'bounded-long-local-v1'
public = long / 'public-delivery'
for path in [stage / 'preparation.json', args.protocol, Path(__file__),
             long / 'qualification.json', long / 'bounded-long-helpers-files.json',
             public / 'build-inputs.json', public / 'qualification.json']:
    records[str(path)] = descriptor(path)
build = read(public / 'build-inputs.json')
names = set(old['tools']) | set(app['tools']) | {
    'qualify_bounded_corpus.py', 'bounded_corpus_metrics.py', 'test_bounded_corpus_metrics.py',
    'corpus.protocol.json', 'valid-mel-corpus.protocol.json', 'macos-bounded-pnola-app.protocol.json'}
checked = set()
while names - checked:
    for name in sorted(names - checked):
        checked.add(name)
        if not name.endswith('.py'):
            continue
        for node in ast.walk(ast.parse((tools / name).read_bytes())):
            modules = [node.module] if isinstance(node, ast.ImportFrom) and node.module else [value.name for value in node.names] if isinstance(node, ast.Import) else []
            for name in modules:
                candidate = name.split('.')[0] + '.py'
                if (tools / candidate).is_file():
                    names.add(candidate)
denied = [repo, Path('/opt/homebrew')]
denied += [path for path in cache.iterdir() if path.is_dir() and path != stage]
denied += [path for path in stage.iterdir() if path != work]
p = {'schema_version': 1, 'scope': 'Fresh bounded CPU native/public pairs on all original eight frozen slices; exact cached official FP32 arrays, original UEM/references/scorer/gates. No Metal, new unseen-data or human-word/desktop/release claim.',
    'work_path': str(work), 'output_path': str(cache / 'bounded-corpus-runtime-v1'), 'expected_cpu': 'Apple M4 Pro',
    'tools': {name: descriptor(tools / name) for name in sorted(names)},
    'repository_files': build['source_pins'], 'inventory': manifest, 'preparation_records': records,
    'helper_reuse': {'public_build_record': str(public / 'build-inputs.json'),
        'public_source_commit': read(public / 'qualification.json')['source_commit'],
        'long_qualification': str(long / 'qualification.json'),
        'helper_archive_manifest': str(long / 'bounded-long-helpers-files.json')},
    'legacy_receipt_path': old['legacy_receipt_path'], 'legacy_scoring': old['legacy_scoring'],
    'quality_gates': old['quality_gates'], 'scoring_environment': old['scoring_environment'],
    'expected_fresh_child_processes': 16, 'expected_scored_pairs': 8,
    'storage': old['storage'], 'command_timeout_seconds': old['command_timeout_seconds'],
    'denied_paths': sorted(set(map(str, denied))),
    'limits': ['Original quality failures and 20% per-slice/two-point Q8 gates remain unchanged.',
        'Qualified helper/payload reuse only; no compiler, download, conversion, new HF/original-NeMo or policy selection.',
        'Original corpus5GiB start/1GiB running/256MiB physical reserve/600s process-group deadlines preserved; separate bounded-public8GiB build/download and VM120/80GiB floors unchanged.',
        'All native/public rows must be byte-exact at the exact bounded CPU runtime identity. No Metal or unseen-data/word-gold/GUI/fresh-machine/signing/release pass.']}
destination = tools / 'bounded-corpus.protocol.json'
assert not destination.exists()
destination.write_text(json.dumps(p, indent=2) + '\n')
print(json.dumps({'stage': str(stage), 'files': len(manifest), 'tools': len(p['tools']),
    'repository_files': len(p['repository_files']), 'protocol': descriptor(destination),
    'compiler_helper_or_neural_calls': 0}))
