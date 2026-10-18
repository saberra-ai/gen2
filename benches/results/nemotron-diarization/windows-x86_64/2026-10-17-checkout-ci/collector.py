"""Publish the exact failed/fixed CI checkout records; no model/helper execution."""
from pathlib import Path
import hashlib
import json
import re
import stat
import subprocess

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
bootstrap = cache / 'windows-checkout-ci-v1.pre-execution.json'
bootstrap_raw = bootstrap.read_bytes()
p = json.loads(bootstrap_raw)
assert p['fixed_source'] == 'f421e960599efe0f7ee37df5596cf450079fdb80'
out = repo / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-checkout-ci'
assert not out.exists() and not out.is_symlink()


def descriptor(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


raw = {}
for name, entry in p['inputs'].items():
    path = Path(entry['path'])
    assert stat.S_ISREG(path.lstat().st_mode)
    raw[name] = path.read_bytes()
    assert descriptor(raw[name]) == entry['identity'], name
assert raw['collector.py'] == Path(__file__).read_bytes()
failed, fixed = [json.loads(raw[name]) for name in ['failed-run.json', 'fixed-run.json']]
assert failed['url'] == 'https://github.com/saberra-ai/gen2/actions/runs/39088182262'
assert failed['headSha'] == 'ba48a1d0101aa454e74f6eacd9838e9ae724d5a4'
assert failed['conclusion'] == 'failure'
bad = [job for job in failed['jobs'] if job['conclusion'] == 'failure']
assert len(bad) == 4 and all('windows-latest' in job['name'] for job in bad)
assert all(next(step for step in job['steps'] if step['name'] == 'Run actions/checkout@v4')['conclusion'] == 'failure' for job in bad)
assert all(job['conclusion'] == 'success' for job in failed['jobs'] if job not in bad)
errors = [line for line in raw['failed-windows.log'].decode().splitlines()
          if '##[error]error: unable to create file ' in line]
assert len(errors) == 8 and all(line.endswith(': Filename too long') for line in errors)
assert fixed['url'] == 'https://github.com/saberra-ai/gen2/actions/runs/39089200762'
assert fixed['headSha'] == p['fixed_source'] and fixed['status'] == 'completed'
assert fixed['conclusion'] == 'success' and len(fixed['jobs']) == 13
assert all(job['status'] == 'completed' and job['conclusion'] == 'success' for job in fixed['jobs'])
assert {job['name'] for job in fixed['jobs']} == {job['name'] for job in failed['jobs']}
windows = [job for job in fixed['jobs'] if 'windows-latest' in job['name']]
assert len(windows) == 4
counts = {}
for job in windows:
    for name in ['Run actions/checkout@v4', 'Verify Windows long-path checkout']:
        step = next(step for step in job['steps'] if step['name'] == name)
        assert step['status'] == 'completed' and step['conclusion'] == 'success'
    text = raw['fixed-windows-' + str(job['databaseId']) + '.log'].decode()
    matches = re.findall(r'\t[^\n]*Z Verified (\d+) tracked paths at least 260 characters long\.$', text, re.M)
    assert matches == ['22'], (job['databaseId'], matches)
    counts[str(job['databaseId'])] = 22
for name, relative in [('diarization-workflow.yml', '.github/workflows/diarization.yml'),
                       ('ci-workflow.yml', '.github/workflows/ci.yml')]:
    assert raw[name] == subprocess.check_output(['git', 'show', p['fixed_source'] + ':' + relative], cwd=repo)
    assert raw[name] == (repo / relative).read_bytes()
    assert b'GIT_CONFIG_KEY_0: core.longpaths' in raw[name]
    assert b"GIT_CONFIG_VALUE_0: 'true'" in raw[name]


def unchanged():
    assert bootstrap.read_bytes() == bootstrap_raw
    for name, entry in p['inputs'].items():
        assert Path(entry['path']).read_bytes() == raw[name], name


unchanged()
out.mkdir()
for name, value in raw.items():
    assert Path(name).name == name
    (out / name).write_bytes(value)
(out / 'pre-execution.json').write_bytes(bootstrap_raw)
record = {'schema_version': 1, 'fixed_source_commit': p['fixed_source'],
    'original_run': failed['url'], 'fixed_run': fixed['url'],
    'original_windows_checkout_failures': 4, 'original_long_path_errors': 8,
    'full_feature_matrix_jobs_passed': 13, 'windows_checkout_and_regression_jobs_passed': 4,
    'verified_long_tracked_files_each_windows_job': counts,
    'new_model_or_helper_calls': 0, 'release_qualified': False,
    'limits': ['Repairs and verifies Windows repository checkout and existing feature compatibility tests.',
        'Normal CI shares the environment repair but its separate complete workflow was not run here.',
        'Does not establish Windows real-time performance or close original accuracy, human, physical app, fresh-machine or public release gates.']}
(out / 'qualification.json').write_text(json.dumps(record, indent=2) + '\n')
files = {path.name: descriptor(path.read_bytes()) for path in out.iterdir()}
assert set(files) == set(raw) | {'pre-execution.json', 'qualification.json'}
unchanged()
assert all(stat.S_ISREG(path.lstat().st_mode) for path in out.iterdir())
assert all(descriptor((out / name).read_bytes()) == value for name, value in files.items())
(out / 'collection.json').write_text(json.dumps({'schema_version': 1, 'files': files}, indent=2) + '\n')
assert {path.name for path in out.iterdir()} == set(files) | {'collection.json'}
print(out, len(files) + 1, 'members; four Windows checkout regressions passed; no model/helper calls')
