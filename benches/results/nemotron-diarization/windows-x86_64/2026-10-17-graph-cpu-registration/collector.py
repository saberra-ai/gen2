"""Retain actual branch workflow registration; no native diagnostic dispatch or execution."""
import hashlib
import json
from pathlib import Path
import re
import stat
import subprocess

REPO = Path('/Users/victor/workspace/gen2-nemotron3-n0')
CACHE = Path('/Users/victor/.cache/gen2-nemotron3-n0')
SOURCE = '0b2b7a7e9e81c83772d8059cc675cdb00a75e452'
CONTEXT = '71a27e26a6285adf393f4b7d5862b21a5302efc4'
BASE = 'ce65591dffd22e387416f0950a88f607b1761d44'
WORKFLOW = '.github/workflows/diarization-windows-graph-cpu-profile.yml'
OUTPUT = REPO / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-graph-cpu-registration'


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def command(argv):
    return subprocess.check_output(argv, cwd=REPO, timeout=60)


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('fresh registration receipt required')
    own = Path(__file__).read_bytes()
    names = [WORKFLOW, 'tools/nemotron-diarization/windows-graph-cpu-profile.protocol.json',
        'docs/research/nemotron-windows-graph-cpu-profile.md',
        'tools/nemotron-diarization/graph_cpu_source_admission.py',
        'tools/nemotron-diarization/graph-cpu-source-admission.protocol.json',
        'docs/research/nemotron-graph-cpu-source-admission.md']
    bound = {name: (REPO / name).read_bytes() for name in names}
    for name, raw in bound.items():
        if raw != command(['git', 'show', SOURCE + ':' + name]):
            raise ValueError('current registration input differs from pushed source: ' + name)
    native_raw = bound[names[1]]
    if (native_raw != command(['git', 'show', CONTEXT + ':' + names[1]])
            or identity(native_raw) != {'bytes': 674377, 'sha256': '581a25fee8e4cf5191bc9b5d22884cc9429b4d6e8ceada46e954a531f10484fa'}):
        raise ValueError('exact committed registration context required')
    previous = json.loads(command(['git', 'show', BASE + ':' + names[1]]))
    native = json.loads(native_raw)
    changed = [name for name, value in native['repository_files'].items() if previous['repository_files'][name] != value]
    if (changed != [WORKFLOW, names[2]] or {**previous, 'repository_files': native['repository_files']} != native):
        raise ValueError('native protocol changes exceed workflow/document registration')
    old_job = command(['git', 'show', BASE + ':' + WORKFLOW]).split(b'  graph-cpu:\n', 1)[1]
    new_job = bound[WORKFLOW].split(b'  graph-cpu:\n', 1)[1]
    if new_job.replace(b"    if: github.event_name == 'workflow_dispatch'\n", b'', 1) != old_job:
        raise ValueError('native job changed beyond explicit manual guard')
    raw_run = command(['gh', 'run', 'view', '39094365683', '--json',
                       'databaseId,headSha,event,status,conclusion,jobs,url,workflowName'])
    raw_workflow = command(['gh', 'api', 'repos/saberra-ai/gen2/actions/workflows/diarization-windows-graph-cpu-profile.yml'])
    run, workflow = json.loads(raw_run), json.loads(raw_workflow)
    jobs = {job['name']: job for job in run['jobs']}
    if (run['databaseId'] != 39094365683 or run['headSha'] != SOURCE or run['event'] != 'push'
            or run['status'] != 'completed' or run['conclusion'] != 'success'
            or run['workflowName'] != 'Diarization Windows graph-only CPU diagnostic'
            or len(run['jobs']) != 2 or set(jobs) != {'register', 'graph-cpu'}
            or any(job['status'] != 'completed' for job in run['jobs'])
            or jobs['register']['databaseId'] != 117577611570 or jobs['register']['conclusion'] != 'success'
            or jobs['graph-cpu']['databaseId'] != 117577612370 or jobs['graph-cpu']['conclusion'] != 'skipped'
            or jobs['graph-cpu']['steps'] != []
            or workflow['id'] != 388152309 or workflow['path'] != WORKFLOW or workflow['state'] != 'active'
            or workflow['name'] != run['workflowName']):
        raise ValueError('exact successful registration and skipped native job required')
    raw_log = command(['gh', 'run', 'view', '39094365683', '--job', '117577611570', '--log'])
    message = 'Graph CPU workflow registered. Native diagnostic requires workflow_dispatch.'
    if sum(line.endswith(message) for line in raw_log.decode().splitlines()) != 1:
        raise ValueError('one actual registration message required')
    local_path = CACHE / 'graph-cpu-registration-model-free-v1.log'
    local_raw = local_path.read_bytes()
    if not re.search(rb'Ran 661 tests in 50\.100s\s+OK \(skipped=8\)\s*$', local_raw):
        raise ValueError('complete registration-context local suite required')
    if (any((REPO / name).read_bytes() != raw for name, raw in bound.items())
            or Path(__file__).read_bytes() != own or local_path.read_bytes() != local_raw):
        raise ValueError('first-bound registration input changed')
    result = {'source_commit': SOURCE, 'native_context_commit': CONTEXT, 'registration_run': 39094365683,
        'registration_job': 117577611570, 'workflow_id': 388152309, 'workflow_state': 'active',
        'native_job': 117577612370, 'native_job_skipped': True, 'native_steps_executed': 0,
        'new_native_calls': 0, 'native_protocol_changed_references': changed,
        'native_job_equal_except_dispatch_guard': True, 'all_other_native_protocol_fields_equal': True,
        'source_bindings': {name: identity(raw) for name, raw in bound.items()},
        'limits': 'Actual push registration only. No manual dispatch, compiler/SDK/graph/model execution, timing, production performance or release qualification. Source/feature CI and artifact return remain separate.'}
    payloads = {'run.json': raw_run, 'workflow.json': raw_workflow, 'register-job.log': raw_log,
                'local-tests.log': local_raw, 'collector.py': own,
                'registration.json': (json.dumps(result, indent=2) + '\n').encode()}
    OUTPUT.mkdir()
    for name, raw in payloads.items():
        with (OUTPUT / name).open('xb') as stream:
            stream.write(raw)
    manifest = {name: identity(raw) for name, raw in payloads.items()}
    (OUTPUT / 'collection.json').write_text(json.dumps({'files': manifest}, indent=2) + '\n')
    if not stat.S_ISDIR(OUTPUT.lstat().st_mode):
        raise ValueError('regular registration publication root required')
    if {path.name for path in OUTPUT.iterdir()} != set(payloads) | {'collection.json'}:
        raise ValueError('closed flat registration receipt required')
    for path in OUTPUT.iterdir():
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError('regular registration payload required')
    if any(identity((OUTPUT / name).read_bytes()) != wanted for name, wanted in manifest.items()):
        raise ValueError('published registration payload changed')
    print(json.dumps({'receipt': str(OUTPUT), 'payloads': len(manifest), 'registration_completed': True,
                      'native_job_skipped': True, 'new_native_calls': 0}))


if __name__ == '__main__':
    main()
