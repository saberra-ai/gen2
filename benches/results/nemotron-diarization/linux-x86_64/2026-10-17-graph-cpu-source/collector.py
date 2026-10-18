"""Replay source-only CI proof and publish bounded inert evidence; no source acquisition or native execution."""
import hashlib
import io
import json
from pathlib import Path
import stat
import sys
import zipfile

REPO = Path('/Users/victor/workspace/gen2-nemotron3-n0')
CACHE = Path('/Users/victor/.cache/gen2-nemotron3-n0')
DELIVERY = CACHE / 'graph-cpu-source-delivery-v1'
OUTPUT = REPO / 'benches/results/nemotron-diarization/linux-x86_64/2026-10-17-graph-cpu-source'
SOURCE = '16344d808d3d9d00e382b47a6aa711420a2ffe35'
ARCHIVE = {'bytes': 289996, 'sha256': '1bc7608bb9032263aa9465df7ab7ed35bca26dc725b5c9fe5420eedda7d2e78e'}
observed = {}


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def read(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('regular bound proof input required')
    raw = path.read_bytes()
    if path in observed and observed[path] != raw:
        raise ValueError('proof input changed')
    observed[path] = raw
    return raw


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('fresh receipt required')
    own = read(Path(__file__))
    archive_path = CACHE / 'graph-cpu-source-12123622270.zip'
    archive_raw = read(archive_path)
    if identity(archive_raw) != ARCHIVE:
        raise ValueError('exact source-only archive required')
    extraction = json.loads(read(DELIVERY / 'extraction.json'))
    if extraction['archive_sha256'] != ARCHIVE['sha256'] or len(extraction['files']) != 70:
        raise ValueError('complete source-only extraction required')
    payloads = {}
    with zipfile.ZipFile(io.BytesIO(archive_raw)) as archive:
        names = {item.filename for item in archive.infolist() if not item.is_dir()}
        if names != set(extraction['files']):
            raise ValueError('archive/extraction member set differs')
        for name, wanted in extraction['files'].items():
            raw = read(DELIVERY / name)
            if identity(raw) != wanted or archive.read(name) != raw:
                raise ValueError('source proof bytes differ from original ZIP')
            payloads['proof/' + name] = raw
    root_raw = payloads['proof/qualification.json']
    stage_raw = payloads['proof/stage/qualification.json']
    outer, stage = json.loads(root_raw), json.loads(stage_raw)
    outer_process = json.loads(payloads['proof/stage.stdout.log.process.json'])
    if (outer['completed'] is not True or outer['compiled_sdk_or_model'] is not False
            or stage['completed'] is not True or stage['compiled_sdk_or_model'] is not False
            or outer['source_commit'] != SOURCE or stage['source_commit'] != SOURCE
            or outer['stage'] != outer_process or outer['stage_qualification'] != identity(stage_raw)
            or any(outer_process[key] is not True for key in ['completed', 'leader_reaped', 'group_gone'])
            or outer_process['returncode'] != 0 or len(stage['commands']) != 19):
        raise ValueError('complete outer-owned source stage required')
    if (outer['minimum_free_bytes'] != 8 * 1024**3 or stage['minimum_free_bytes'] != 8 * 1024**3
            or stage['initial_free_bytes'] < 8 * 1024**3):
        raise ValueError('original source-only initial storage admission required')
    tools = REPO / 'tools/nemotron-diarization'
    native_raw = read(tools / 'windows-graph-cpu-profile.protocol.json')
    native = json.loads(native_raw)
    source_protocol_raw = read(tools / 'graph-cpu-source-admission.protocol.json')
    source_protocol = json.loads(source_protocol_raw)
    if identity(native_raw) != stage['native_protocol'] or payloads['proof/stage/protocol.json'] != source_protocol_raw:
        raise ValueError('native and source-only protocol bindings differ')
    fixed = {'tools/nemotron-diarization/' + name: wanted for name, wanted in native['tools'].items()}
    fixed.update(native['repository_files'])
    for name, wanted in source_protocol['files'].items():
        if name in fixed and fixed[name] != wanted:
            raise ValueError('source-only binding overrides native input')
        fixed[name] = wanted
    fixed['tools/nemotron-diarization/windows-graph-cpu-profile.protocol.json'] = identity(native_raw)
    fixed['tools/nemotron-diarization/graph-cpu-source-admission.protocol.json'] = identity(source_protocol_raw)
    if stage['inputs'] != fixed or len(fixed) != 3049:
        raise ValueError('complete source-only input closure required')
    for name, wanted in fixed.items():
        if identity(read(REPO / name)) != wanted:
            raise ValueError('bound source-only input differs: ' + name)
    # Only first-bound project modules execute, never artifact-contained code.
    sys.path.insert(0, str(tools))
    from pooled_runtime import BOUNDED_PROFILE, load_profile
    from windows_graph_cpu_profile_metrics import canonical_audit
    from windows_graph_cpu_profile_sources import CPU_SOURCE
    from ggml_graph_cpu_profile_source import PATH
    _, lock = load_profile(BOUNDED_PROFILE)
    expected = canonical_audit()['files'].copy()
    expected[PATH] = CPU_SOURCE
    if (stage['materialized_files'] != expected or len(expected) != 3425
            or stage['changed_ggml_files'] != [PATH] or stage['profiled_cpu_source'] != CPU_SOURCE
            or stage['source_hashes'] != native['variants']['4']['source_sha256']):
        raise ValueError('complete materialized source audit differs')
    ci_repo = Path('/home/runner/work/gen2/gen2')
    ci_root = Path('/home/runner/work/_temp/graph-cpu-source-admission/stage')
    pristine, source = ci_root / 'pristine', ci_root / 'source'
    expected_commands = []
    for name, path, url, revision in [
        ('native', pristine / 'native', 'https://github.com/NVIDIA/NeMo-Speech.cpp.git', lock['runtime']['revision']),
        ('llama', pristine / 'native/llama.cpp', 'https://github.com/ggml-org/llama.cpp.git', lock['runtime']['llama_revision'])]:
        for suffix, argv in [('init', ['git', 'init', path]),
                             ('origin', ['git', '-C', path, 'remote', 'add', 'origin', url]),
                             ('fetch', ['git', '-C', path, 'fetch', '--depth=1', 'origin', revision]),
                             ('checkout', ['git', '-C', path, 'checkout', '--detach', revision])]:
            expected_commands.append((name + '-' + suffix, argv, ci_repo, 'acquisition'))
    for name, origin, target, revision in [
        ('native', pristine / 'native', source, lock['runtime']['revision']),
        ('llama', pristine / 'native/llama.cpp', source / 'llama.cpp', lock['runtime']['llama_revision'])]:
        expected_commands += [('clone-' + name, ['git', 'clone', '--no-hardlinks', '--no-checkout', origin, target], ci_repo, 'composition'),
                              ('checkout-' + name, ['git', '-C', target, 'checkout', '--detach', revision], ci_repo, 'composition')]
    for index, item in enumerate(lock['runtime']['compatibility_patches']):
        patch = ci_repo / 'tools/nemotron-diarization' / item['path']
        expected_commands += [('check-patch-' + str(index), ['git', 'apply', '--check', patch], source, 'composition'),
                              ('apply-patch-' + str(index), ['git', 'apply', patch], source, 'composition')]
    expected_commands.append(('materialize-twenty-patches', ['cmake', '-DSOURCE_DIR=' + str(source / 'llama.cpp'),
        '-DPATCH_DIR=' + str(source / 'patches'), '-DDEST_DIR=' + str(source / 'build/gen2-cpu-diar/_deps/llama.cpp'),
        '-P', source / 'cmake/llama_cpp.cmake'], ci_repo, 'composition'))
    for row, (name, argv, cwd, phase) in zip(stage['commands'], expected_commands, strict=True):
        sidecar = json.loads(payloads['proof/stage/evidence/' + name + '.stdout.log.process.json'])
        if (row['name'] != name or row['command'] != list(map(str, argv)) or row['cwd'] != str(cwd) or row['phase'] != phase
                or row['completed'] is not True or row['leader_reaped'] is not True or row['returncode'] != 0
                or row['stage_group'] != outer_process['pid'] or row['group_owner'] != 'outer bounded stage supervisor'
                or row['free_bytes_before'] < 8 * 1024**3 or any(row.get(key) != value for key, value in sidecar.items())
                or len(payloads['proof/stage/evidence/' + name + '.stdout.log']) != row['stdout_bytes']
                or len(payloads['proof/stage/evidence/' + name + '.stderr.log']) != row['stderr_bytes']):
            raise ValueError('source command, storage or ownership evidence differs: ' + name)
    extras = {
        'run.json': 'graph-cpu-source-run-39093415395.completed.json',
        'artifacts.json': 'graph-cpu-source-artifacts-39093415395.json',
        'job.log': 'graph-cpu-source-117574755510.completed.log',
        'native-feature-ci.json': 'graph-cpu-native-feature-ce65591d.completed.json',
        'download-pre-execution.json': 'graph-cpu-source-download-v1.pre-execution.json',
        'download-process.json': 'graph-cpu-source-12123622270.zip.process.json',
        'download.stderr.log': 'graph-cpu-source-download-v1.stderr.log',
        'local-tests.log': 'graph-cpu-source-model-free-v1.log'}
    for name, original in extras.items():
        payloads[name] = read(CACHE / original)
    run, artifacts = json.loads(payloads['run.json']), json.loads(payloads['artifacts.json'])
    job, = run['jobs']
    artifact, = artifacts['artifacts']
    feature = json.loads(payloads['native-feature-ci.json'])
    download, pre = json.loads(payloads['download-process.json']), json.loads(payloads['download-pre-execution.json'])
    feature_names = {'evidence-tools'} | {
        'features (' + system + (', ' + feature if feature else '') + ')'
        for system in ['ubuntu-latest', 'windows-latest', 'macos-latest']
        for feature in ['', 'backend-diar-nemo', 'asr-dynamic,backend-diar-nemo', 'backend-llamacpp,asr-dynamic,backend-diar-nemo']}
    if (run['databaseId'] != 39093415395 or run['headSha'] != SOURCE or run['status'] != 'completed' or run['conclusion'] != 'success'
            or run['workflowName'] != 'Diarization graph CPU source-only admission' or job['name'] != 'source'
            or job['databaseId'] != 117574755510 or job['conclusion'] != 'success' or job['status'] != 'completed'
            or artifacts['total_count'] != 1 or artifact['id'] != 12123622270 or artifact['name'] != 'nemotron-graph-cpu-source-admission' or artifact['expired'] is not False
            or artifact['size_in_bytes'] != ARCHIVE['bytes'] or artifact['digest'] != 'sha256:' + ARCHIVE['sha256']
            or artifact['workflow_run']['id'] != run['databaseId'] or artifact['workflow_run']['head_sha'] != SOURCE
            or feature['databaseId'] != 39092822458 or feature['workflowName'] != 'Diarization feature checks'
            or feature['headSha'] != source_protocol['native_source'] or feature['status'] != 'completed' or feature['conclusion'] != 'success'
            or {j['name'] for j in feature['jobs']} != feature_names
            or len(feature['jobs']) != 13 or any(j['status'] != 'completed' or j['conclusion'] != 'success' for j in feature['jobs'])
            or pre['free_bytes_before'] < 4 * 1024**3 or pre['minimum_start_bytes'] != 4 * 1024**3
            or pre['run'] != 39093415395 or pre['job'] != 117574755510 or pre['source'] != SOURCE
            or pre['metadata'] != identity(payloads['artifacts.json']) or pre['archive_cap'] != 8 * 1024**2 or pre['extract_cap'] != 16 * 1024**2
            or pre['artifact'] != {'id': 12123622270, **ARCHIVE} or download['completed'] is not True or download['group_gone'] is not True
            or download['command'] != pre['argv'] or download['cwd'] != str(REPO) or download['leader_reaped'] is not True
            or download['returncode'] != 0 or download['stdout_bytes'] != ARCHIVE['bytes']):
        raise ValueError('terminal CI/archive/download metadata differs')
    payloads.update({'artifact.zip': archive_raw, 'extraction.json': observed[DELIVERY / 'extraction.json'], 'collector.py': own})
    result = {'source_commit': SOURCE, 'native_preparation_source': source_protocol['native_source'],
        'source_run': 39093415395, 'source_job': 117574755510, 'archive': ARCHIVE,
        'source_stage_completed': True, 'new_native_calls': 0, 'compiler_sdk_model_execution': False,
        'commands_replayed': 19, 'input_bindings': len(fixed), 'materialized_files': 3425,
        'profiled_cpu_source': CPU_SOURCE, 'minimum_ci_free_bytes': min(row['free_bytes_before'] for row in stage['commands']),
        'original_source_floor_bytes': 8 * 1024**3, 'outer_group_gone': True,
        'limits': 'Saved source/protocol/command replay only. No local source acquisition or native execution. Windows graph/model parity, observer timing, production performance and release remain unqualified.'}
    payloads['replay.json'] = (json.dumps(result, indent=2) + '\n').encode()
    if any(path.read_bytes() != raw for path, raw in observed.items()):
        raise ValueError('first-bound collection input changed')
    OUTPUT.mkdir(parents=True)
    for name, raw in payloads.items():
        path = OUTPUT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
    manifest = {name: identity(raw) for name, raw in payloads.items()}
    (OUTPUT / 'collection.json').write_text(json.dumps({'files': manifest}, indent=2) + '\n')
    if any(identity((OUTPUT / name).read_bytes()) != wanted for name, wanted in manifest.items()):
        raise ValueError('published source proof changed')
    wanted_files = set(manifest) | {'collection.json'}
    wanted_directories = {parent.as_posix() for name in wanted_files
                          for parent in Path(name).parents if parent != Path('.')}
    actual_files, actual_directories, folded = set(), set(), set()
    if not stat.S_ISDIR(OUTPUT.lstat().st_mode):
        raise ValueError('regular publication root required')
    for path in OUTPUT.rglob('*'):
        name = path.relative_to(OUTPUT).as_posix()
        if name.casefold() in folded:
            raise ValueError('case-colliding publication member')
        folded.add(name.casefold())
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            actual_directories.add(name)
        elif stat.S_ISREG(mode):
            actual_files.add(name)
        else:
            raise ValueError('symlink or special publication member')
    if actual_files != wanted_files or actual_directories != wanted_directories:
        raise ValueError('publication is not the closed declared namespace')
    print(json.dumps({'receipt': str(OUTPUT), 'members': len(manifest), **result}))


if __name__ == '__main__':
    main()
