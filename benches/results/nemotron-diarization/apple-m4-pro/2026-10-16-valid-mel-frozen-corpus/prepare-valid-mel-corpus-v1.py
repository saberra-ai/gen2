"""Stage exact existing helpers, packages, frozen corpus and cached oracles without inference."""
import json
from pathlib import Path
import subprocess
import sys

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0'); cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
sys.path.insert(0, str(repo / 'tools/nemotron-diarization'))
from build_valid_mel_candidate import descriptor
from qualify_corpus import validate_corpus
from qualify_valid_mel_corpus import admit_reference
from sdk_archive import inventory

out = cache / 'valid-mel-corpus-inputs-v1'; assert not out.exists(); out.mkdir()
work = out / 'work'; work.mkdir()
sources = {}; commands = []
def clone(source, target, directory=False):
    target.parent.mkdir(parents=True, exist_ok=True)
    argv = ['/bin/cp', '-cR' if directory else '-c', str(source), str(target)]
    result = subprocess.run(argv, capture_output=True)
    commands.append({'argv': argv, 'returncode': result.returncode})
    (out / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n'); result.check_returncode()
    assert (inventory(source) == inventory(target)) if directory else (descriptor(source) == descriptor(target))
    sources[str(target.relative_to(work))] = {'source': str(source), 'inventory': inventory(source)} if directory else {'source': str(source), **descriptor(source)}
clone(cache / 'quality-corpus-v1', work / 'corpus', True)
manifest = validate_corpus(work / 'corpus')
builds = {}
for backend, kind in [('cpu', 'macos'), ('metal', 'metal')]:
    origin = cache / ('valid-mel-' + kind + '-runtime-inputs-v1'); build = json.loads((origin / 'build.json').read_text()); builds[backend] = str(origin / 'build.json')
    assert build['models_loaded'] is False and build['neural_calls'] == 0
    for name, expected in {**build['rust_source_files'], **build['helper_source_files']}.items():
        assert descriptor(Path(name)) == expected, name
    clone(origin / 'work/nemo', work / backend / 'nemo', True)
    for name in ['native-stream', 'public-stream']:
        clone(origin / 'work/bin' / name, work / backend / 'bin' / name)
        assert descriptor(work / backend / 'bin' / name) == {key: build['inputs']['bin/' + name][key] for key in ['bytes', 'sha256']}
clone(cache / 'valid-mel-macos-runtime-inputs-v1/work/assets', work / 'assets', True)
legacy_protocol = json.loads((repo / 'tools/nemotron-diarization/corpus.protocol.json').read_text())
legacy_receipt = json.loads((repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-frozen-corpus/receipt.json').read_text())
for item in manifest['slices']:
    for suffix in ['.npy', '.json']:
        name = item['id'] + '-fp32' + suffix
        clone(cache / 'quality qualification 125f907' / name, work / 'oracle' / name)
    admit_reference(item, work, legacy_protocol, legacy_receipt)
record = {'scope': __doc__, 'models_loaded': False, 'native_or_model_or_version_calls': 0,
          'helper_build_records': builds, 'sources': sources, 'inventory': inventory(work),
          'preparation_source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
          'launcher': descriptor(Path(__file__))}
(out / 'preparation.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({'entries': len(record['inventory']), 'fixed_slices': len(manifest['slices']), 'neural_calls': 0}))
