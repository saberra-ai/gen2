"""Collect terminal feature CI and its narrow graph-clock API evidence, without model execution."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

REPO = Path('/Users/victor/workspace/gen2-nemotron3-n0')
CACHE = Path('/Users/victor/.cache/gen2-nemotron3-n0')
SOURCE = '8db9c0e5100de91a8e18b431e72359d586ed16bd'
RUN = '39091355966'
API_JOB = 117568618848
TOOLS_JOB = 117568618722
OUTPUT = REPO / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-graph-cpu-api-ci'
STEP = 'Windows graph CPU API control'
SOURCE_FILES = ['.github/workflows/diarization.yml', *[
    'tools/nemotron-diarization/' + name for name in (
        'ggml_graph_cpu_profile.py', 'ggml_graph_cpu_profile_source.py',
        'test_ggml_graph_cpu_profile.py', 'test_ggml_graph_cpu_windows.py')]]


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def command(arguments):
    return subprocess.check_output(arguments, cwd=REPO, timeout=60)


def sources():
    result = {}
    for name in SOURCE_FILES:
        raw = (REPO / name).read_bytes()
        if raw != command(['git', 'show', SOURCE + ':' + name]):
            raise ValueError('current source differs from compiled CI commit: ' + name)
        result[name] = identity(raw)
    return result


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('fresh collection destination required')
    own = Path(__file__).read_bytes()
    bound_sources = sources()
    local_path = CACHE / 'graph-cpu-prototype-model-free-v2.log'
    local = local_path.read_bytes()
    if not re.search(rb'Ran 648 tests in 47\.815s\s+OK \(skipped=8\)\s*$', local):
        raise ValueError('complete recorded local suite required')
    metadata = command(['gh', 'run', 'view', RUN, '--json',
                        'databaseId,headSha,status,conclusion,jobs,url,workflowName'])
    run = json.loads(metadata)
    features = ['', 'backend-diar-nemo', 'asr-dynamic,backend-diar-nemo',
                'backend-llamacpp,asr-dynamic,backend-diar-nemo']
    names = {'evidence-tools'} | {
        'features (' + system + (', ' + feature if feature else '') + ')'
        for system in ['ubuntu-latest', 'windows-latest', 'macos-latest'] for feature in features}
    if (run['databaseId'] != int(RUN) or run['headSha'] != SOURCE
            or run['workflowName'] != 'Diarization feature checks'
            or run['status'] != 'completed' or run['conclusion'] != 'success'
            or len(run['jobs']) != 13 or {job['name'] for job in run['jobs']} != names
            or any(job['status'] != 'completed' or job['conclusion'] != 'success' for job in run['jobs'])):
        raise ValueError('all thirteen terminal matching CI jobs must succeed')
    api = [job for job in run['jobs'] if job['databaseId'] == API_JOB]
    if len(api) != 1 or api[0]['name'] != 'features (windows-latest)':
        raise ValueError('exact featureless Windows job required')
    tools = [job for job in run['jobs'] if job['databaseId'] == TOOLS_JOB]
    if len(tools) != 1 or tools[0]['name'] != 'evidence-tools':
        raise ValueError('exact evidence-tools job required')
    steps = [step for job in run['jobs'] for step in job['steps'] if step['name'] == STEP]
    if (len(steps) != 12 or sum(step['conclusion'] == 'success' for step in steps) != 1
            or sum(step['conclusion'] == 'skipped' for step in steps) != 11
            or any(step['status'] != 'completed' for step in steps)
            or not any(step['name'] == STEP and step['conclusion'] == 'success' for step in api[0]['steps'])):
        raise ValueError('exactly one executed API step required')
    # Sequential requests avoid the known shared gh log-cache race.
    api_log = command(['gh', 'run', 'view', RUN, '--job', str(API_JOB), '--log'])
    tools_log = command(['gh', 'run', 'view', RUN, '--job', str(TOOLS_JOB), '--log'])
    if max(len(api_log), len(tools_log)) > 64 * 1024**2:
        raise ValueError('bounded complete CI logs required')
    # Parse only the fixed API wire row with stdlib; execute no project imports.
    fields = ('tid', 'kernel_100ns', 'user_100ns', 'errors', 'error_bits', 'first_error')
    graph = re.compile(r'\[DEBUG-graph-cpu-v1\] graph pool=([0-9a-fA-Fx]+) generation=(\d+) thread=(\d+)'
                       r' tid=(\d+) kernel_100ns=(\d+) user_100ns=(\d+) errors=(\d+) error_bits=(\d+) first_error=(\d+)')
    rows = [line.split('Actual Windows graph-clock API control: ', 1)[1]
            for line in api_log.decode().splitlines() if 'Actual Windows graph-clock API control: ' in line]
    if len(rows) != 1 or not (match := graph.fullmatch(rows[0])):
        raise ValueError('one complete raw actual Windows API row required')
    values = dict(zip(fields, map(int, match.groups()[3:]), strict=True))
    if ((int(match[1], 16), int(match[2]), int(match[3])) != (1, 1, 0)
            or not 0 < values['tid'] <= 0xffffffff
            or any(values[key] for key in ['errors', 'error_bits', 'first_error'])
            or any(not 0 <= values[key] <= 2**64 - 1 for key in ['kernel_100ns', 'user_100ns'])
            or 'Ran 1 test in ' not in api_log.decode()
            or 'Ran 648 tests in ' not in tools_log.decode()):
        raise ValueError('actual API success and complete suite logs required')
    if sources() != bound_sources or local_path.read_bytes() != local or Path(__file__).read_bytes() != own:
        raise ValueError('collection inputs changed')
    payloads = {'run.json': metadata, 'windows-api-job.log': api_log,
                'evidence-tools-job.log': tools_log, 'local-model-free.log': local,
                'collector.py': own}
    payloads['collection.json'] = (json.dumps({
        'source_commit': SOURCE, 'run_id': int(RUN), 'url': run['url'],
        'source_files': bound_sources, 'successful_jobs': 13,
        'api_job_id': API_JOB, 'actual_api_row': rows[0], 'actual_api_values': values,
        'inputs': {name: identity(raw) for name, raw in payloads.items()},
        'limits': 'Actual Windows API adapter and model-free feature CI only. No GGML graph, native/model parity, observer timing, performance, package or release qualification.',
        'acquisition_note': 'An earlier gh log request while the run was active was refused; its empty placeholder is excluded. These terminal log requests are sequential.'
    }, indent=2) + '\n').encode()
    OUTPUT.mkdir()
    for name, raw in payloads.items():
        with (OUTPUT / name).open('xb') as stream:
            stream.write(raw)
    manifest = {name: identity(raw) for name, raw in payloads.items()}
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for name, expected in manifest.items():
        if identity((OUTPUT / name).read_bytes()) != expected:
            raise ValueError('published payload changed')
    print(json.dumps({'destination': str(OUTPUT), 'payload_members': len(manifest),
                      'successful_jobs': 13, 'actual_api_values': values}))


if __name__ == '__main__':
    main()
