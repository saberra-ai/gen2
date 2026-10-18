"""Independently replay one admitted graph CPU return; no subprocess or native execution."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import stat
import sys

ORIGIN = 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-thread-cpu-local-replay/replay/replay-thread-cpu-returned-evidence-v3.py'
ORIGIN_ID = {'bytes': 21739, 'sha256': '3d727f402d410280aa18d1fa098ee49fef7e9bf70ba378dc40f28836b843562a'}
RESULT = 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-graph-cpu-profile'
NATIVE_PROTOCOL = {'bytes': 674377, 'sha256': '581a25fee8e4cf5191bc9b5d22884cc9429b4d6e8ceada46e954a531f10484fa'}


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def capture(path, expected=None):
    if not stat.S_ISREG(path.lstat().st_mode) or any(parent.is_symlink() for parent in path.parents):
        raise ValueError('regular nonsymlink replay input required: ' + str(path))
    raw = path.read_bytes()
    if expected is not None and identity(raw) != expected:
        raise ValueError('first-bound replay input changed: ' + str(path))
    return raw


def read(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique)


REVIEWED = '''    route_path = repo / 'tools/nemotron-diarization/graph-cpu-bundle-receive.protocol.json'
    route = read(route_path)
    require(descriptor(route_path) == pre['route_protocol'] and route['files'] == pre['route_files'],
            'first-bound local return route changed')
    from graph_cpu_receiver_sources import bound_origins, derive
    generated, _ = derive(bound_origins(repo), pre['native_source'], pre['native_run'], pre['native_job'])
    names = {'collector.py': 'collect-windows-graph-cpu-profile-v1.py',
        'admit-windows-graph-cpu-profile-delivery-v1.py': 'admit-windows-graph-cpu-profile-delivery-v1.py',
        'replay-windows-graph-cpu-profile-supervision-v1.py': 'replay-windows-graph-cpu-profile-supervision-v1.py'}
    reviewed = {name: reviewed_body(generated[original], delivered / name, pre['literal_relocation'])
        for name, original in names.items()}
    from graph_cpu_transport import configured_module
    native_admission = configured_module(repo, route, b'', False)
    native_admission.admit_api(read(delivered / 'producer-run.json'), read(delivered / 'job.json'),
        read(delivered / 'artifacts.json'))
    feature = native_admission.admit_graph_feature(
        read(local / 'delivery/commands/api-native-feature.stdout.log'),
        read(local / 'delivery/commands/api-native-feature-jobs.stdout.log'))
    bundle_metadata = read(local / 'delivery/bundle.json')
    require(bundle_metadata['native_feature_run_id'] == pre['native_feature_run'] and
        bundle_metadata['feature_admission'] == feature, 'independent feature proof differs')
    print('bootstrap: complete graph-only return and current/retained source closure verified', flush=True)
'''

SOURCE_PROOF = '''    local_commands = read(local / 'commands.json')
    sidecars = pre['local_sidecars']
    require(len(local_commands) == len(sidecars) == pre['result_collection']['members'] + 21,
            'complete local command proof required')
    canonical = lambda row: json.dumps(row, sort_keys=True, separators=(',', ':'))
    require(Counter(map(canonical, local_commands)) ==
        Counter(canonical(read(Path(name))) for name in sidecars), 'local ledger/sidecar mismatch')
    require(all(row.get('completed') is True and row.get('leader_reaped') is True and
        row.get('group_gone') is True and type(row.get('returncode')) is int and row['returncode'] == 0
        for row in local_commands), 'local command cleanup failure')
    prep = read(prep_root / 'qualification.json')
    prep_commands = prep['commands']
    require(prep['completed'] is True and prep['compiled_sdk_or_model'] is False and
        prep['source_commit'] == pre['remote_source'] and len(prep_commands) == 19 and
        len(prep['materialized_files']) == 3425 and prep['materialized_files'] == audit['files'],
        'complete source-only composition required')
    source_protocol_path = tools / 'graph-cpu-source-admission.protocol.json'
    source_protocol = read(source_protocol_path)
    inputs = {'tools/nemotron-diarization/' + name: value for name, value in p['tools'].items()}
    for additions in (p['repository_files'], source_protocol['files'], {
        'tools/nemotron-diarization/windows-graph-cpu-profile.protocol.json': pre['native_protocol'],
        'tools/nemotron-diarization/graph-cpu-source-admission.protocol.json': descriptor(source_protocol_path)}):
        for name, wanted in additions.items():
            require(name not in inputs or inputs[name] == wanted, 'conflicting source-only binding')
            inputs[name] = wanted
    require(prep['inputs'] == inputs and len(inputs) == 3049, 'complete source-only input binding required')
    prep_sidecars = list((prep_root / 'evidence').glob('*.process.json'))
    require(len(prep_sidecars) == 19 and Counter(canonical({key: value for key, value in row.items()
        if key not in ('name', 'phase', 'free_bytes_before')}) for row in prep_commands) ==
        Counter(canonical(read(path)) for path in prep_sidecars), 'source-only command sidecar mismatch')
    require([row['phase'] for row in prep_commands] == ['acquisition'] * 8 + ['composition'] * 11,
        'original acquisition/composition order required')
    remote_commands = read(local / 'delivery/commands.json')
    source_stage, = [row for row in remote_commands if row['name'] == 'source-only-preparation']
    require(source_stage['completed'] and source_stage['leader_reaped'] and source_stage['group_gone'],
        'source outer owned group must be gone')
    from graph_cpu_local_replay import expected_source_commands
    from pooled_runtime import BOUNDED_PROFILE, load_profile
    _, source_lock = load_profile(BOUNDED_PROFILE)
    source_stage_path = Path(source_stage['command'][-2])
    remote_repo = Path(pre['literal_relocation']['/Users/victor/workspace/gen2-nemotron3-n0'])
    require(source_stage['command'][1:] == ['-B', str(remote_repo / 'tools/nemotron-diarization/graph_cpu_source_admission.py'),
        '--output', str(source_stage_path), '--worker'], 'exact outer-owned source worker command required')
    require([(row['name'], row['command'], row['cwd']) for row in prep_commands] ==
        expected_source_commands(remote_repo, source_stage_path, source_lock), 'exact nineteen source argv/cwd records required')
    require(all(row['completed'] is True and row['leader_reaped'] is True and row['returncode'] == 0
        and row['stage_group'] == source_stage['pid'] and row['group_owner'] == 'outer bounded stage supervisor'
        and type(row['free_bytes_before']) is int and row['free_bytes_before'] >= 8 * 1024**3
        for row in prep_commands), 'source-only owned command/floor failure')
    for row in remote_commands:
        sidecar = read(local / 'delivery/commands' / (row['name'] + '.stdout.log.process.json'))
        require({key: value for key, value in row.items() if key not in
            ('name', 'owned_stage_inherited_children', 'free_bytes_before')} == sidecar,
            'remote ledger/sidecar mismatch')
        require(sidecar['completed'] is True and sidecar['leader_reaped'] is True and
            sidecar['group_gone'] is True and sidecar['returncode'] == 0 and
            row['free_bytes_before'] >= 8 * 1024**3, 'remote owned command/floor failure')
    bundle_process = read(local / 'delivery/receiver-result.bundle.process.json')
    require(bundle_process['completed'] is True and bundle_process['leader_reaped'] is True and
        bundle_process['group_gone'] is True and bundle_process['returncode'] == 0,
        'result bundle owned process failure')
'''

OBSERVATIONS = '''    zero_fields, worker_observations = Counter(), []
    for index in (1, 2):
        saved = read(evidence / ('case-' + str(index) + '.matrix-profile.json'))
        totals = {}
        for graph in saved['graphs']:
            for worker in graph['workers']:
                slot = totals.setdefault(str(worker['thread']), {'graph_wall_us': 0,
                    'kernel_100ns': 0, 'user_100ns': 0})
                slot['graph_wall_us'] += worker['elapsed_us']
                for name in ('kernel_100ns', 'user_100ns'):
                    slot[name] += worker['graph_cpu'][name]
                    zero_fields['graph.' + name] += worker['graph_cpu'][name] == 0
        worker_observations.append({'index': index, 'per_worker': totals})
'''


def derive(raw):
    if identity(raw) != ORIGIN_ID:
        raise ValueError('exact reviewed local replay origin required')
    text, edits = raw.decode(), []

    def change(old, new, count=1):
        nonlocal text
        if text.count(old) != count:
            raise ValueError('exact local replay seam required: ' + old[:100])
        text = text.replace(old, new)
        edits.append((old, new, count))

    change("def read(path):\n    return json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=no_duplicate_keys)",
        "READ_EXPECTED, READ_CAPTURED = {}, {}\n\ndef read(path):\n    raw = path.read_bytes()\n    name = str(path.resolve())\n    observed = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}\n    require(name not in READ_EXPECTED or observed == READ_EXPECTED[name], 'parsed JSON differs from admitted bytes')\n    first = READ_CAPTURED.setdefault(name, observed)\n    require(first == observed, 'previously parsed JSON changed')\n    return json.loads(raw.decode('utf-8-sig'), object_pairs_hook=no_duplicate_keys)")
    change("    pre = read(args.protocol)",
        "    READ_EXPECTED[str(args.protocol.resolve())] = first_protocol\n    pre = read(args.protocol)\n    for name, expected in pre['files'].items():\n        require(name not in READ_EXPECTED or READ_EXPECTED[name] == expected, 'conflicting first JSON binding')\n        READ_EXPECTED[name] = expected")
    change("    # Recheck the original first-bound identities, never a newly captured set.",
        "    for name, expected in READ_CAPTURED.items():\n        require(descriptor(Path(name)) == expected, 'parsed JSON changed after validation')\n    # Recheck the original first-bound identities, never a newly captured set.")
    change("    collection = read(delivered / 'collection-qualification.json')",
        "    collection = read(delivered / 'collection-qualification.json')\n    require(collection['source_commit'] == pre['native_source'] and collection['run_id'] == pre['native_run']\n        and collection['job_id'] == pre['native_job'] and collection['artifact_id'] == route['artifact_id']\n        and collection['archive'] == route['archive'] and collection['protocol'] == pre['native_protocol'],\n        'independent native collection provenance differs')")
    change('ast.parse(original.read_bytes())', 'ast.parse(original)', 2)
    for old, new in [
        ("'c7bc18994fb8247fd3723ce3c3645039cd14dcaf'", "pre['remote_source']"),
        ("'580e79ac6943384a3ec545c0c9f2a540e48e9e6d'", "pre['remote_result']"),
        ('39084790919', "pre['remote_run']"), ('117548983776', "pre['remote_job']"),
        ("'15c905d8815190a54f25452e7e51378c24f85914'", "pre['native_source']"),
        ("len(p['tools']) == 135", "len(p['tools']) == 149"),
        ("len(p['repository_files']) == 2880", "len(p['repository_files']) == 2889"),
        ('from ggml_thread_cpu_profile import MAX_LOG_BYTES, summarize',
         'from ggml_graph_cpu_profile import MAX_LOG_BYTES, summarize'),
    ]:
        change(old, new)
    first = text.index("    route = read(repo / 'tools/nemotron-diarization/remote-thread-cpu-replay.protocol.json')")
    last = text.index('    # All executable project code', first)
    change(text[first:last], REVIEWED + '\n')
    first = text.index("    local_commands = read(local / 'commands.json')")
    last = text.index('    zero_fields, worker_observations = Counter(), []', first)
    change(text[first:last], SOURCE_PROOF + '\n')
    first = text.index('    zero_fields, worker_observations = Counter(), []')
    last = text.index("    result = {'schema_version': 1", first)
    change(text[first:last], OBSERVATIONS + '\n')
    for old, new in [
        ("'current_frozen_tools_verified': 135", "'current_frozen_tools_verified': 149"),
        ("'current_frozen_repository_references_verified': 2880", "'current_frozen_repository_references_verified': 2889"),
        ("'retained_frozen_tools_verified': 135", "'retained_frozen_tools_verified': 149"),
        ("'local_command_sidecars_verified': 470", "'local_command_sidecars_verified': len(local_commands)"),
        ("'source_only_command_sidecars_verified': 11", "'source_only_command_sidecars_verified': 19"),
        ('Remote native ZIP-download owned-process sidecar stayed in the remote cache and was not returned. Native59WindowsJob, returned source11 and local470 records are separately retained and checked.',
         'Native ZIP-download owned-process sidecar remains remote. All59 native Windows Job records,19 source commands and every returned remote/local owned command are independently checked.'),
        ('Raw thread cycles and CPU100ns units stay separate from wall microseconds. No frequency conversion, exact wait, pure arithmetic or causal scheduling inference. Preserve genuine zeros; timing interpretation requires both original observer bounds.',
         'Graph-only kernel/user CPU100ns units remain separate from wall microseconds. No per-region CPU attribution, cycle conversion, exact descheduled time, pure arithmetic or causal scheduling inference. Preserve genuine zeros; interpretation requires both original observer bounds.'),
    ]:
        change(old, new)
    for old, new, count in [('windows-thread-cpu-profile', 'windows-graph-cpu-profile', 2),
            ('windows_thread_cpu_profile_', 'windows_graph_cpu_profile_', 2),
            ('thread_cpu_totals', 'graph_cpu_totals', 4)]:
        change(old, new, count)
    ast.parse(text)
    return text.encode(), edits


def local_sidecars(local, members):
    names = ['source-head', 'source-refs', 'source-index', 'committed-graph_cpu_transport.py',
        'committed-remote_q8_apack_replay.py', 'committed-graph-cpu-bundle-receive.protocol.json',
        'api-run', 'api-job', 'api-artifacts', 'gh-help', 'source-objects',
        'init-quarantine', 'verify-bundle', 'unbundle', 'parents', 'base-roots', 'changes', 'trees',
        'final-refs', 'final-head']
    return [local / 'delivery.zip.process.json', *[local / 'commands' / (name + '.stdout.process.json') for name in names]] + [
        local / 'blob-stage' / (str(index) + '.process.json') for index in range(members)]


def expected_source_commands(repo, stage, lock):
    """Reconstruct the fixed worker recipe independently from its returned ledger."""
    rows = []
    pristine = stage / 'pristine'
    for name, path, url, revision in [
            ('native', pristine / 'native', 'https://github.com/NVIDIA/NeMo-Speech.cpp.git', lock['runtime']['revision']),
            ('llama', pristine / 'native/llama.cpp', 'https://github.com/ggml-org/llama.cpp.git', lock['runtime']['llama_revision'])]:
        for suffix, command in [('init', ['git', 'init', path]),
                ('origin', ['git', '-C', path, 'remote', 'add', 'origin', url]),
                ('fetch', ['git', '-C', path, 'fetch', '--depth=1', 'origin', revision]),
                ('checkout', ['git', '-C', path, 'checkout', '--detach', revision])]:
            rows.append((name + '-' + suffix, list(map(str, command)), str(repo)))
    source = stage / 'source'
    for name, origin, target, revision in [
            ('native', pristine / 'native', source, lock['runtime']['revision']),
            ('llama', pristine / 'native/llama.cpp', source / 'llama.cpp', lock['runtime']['llama_revision'])]:
        rows.append(('clone-' + name, list(map(str, ['git', 'clone', '--no-hardlinks', '--no-checkout', origin, target])), str(repo)))
        rows.append(('checkout-' + name, ['git', '-C', str(target), 'checkout', '--detach', revision], str(repo)))
    for index, item in enumerate(lock['runtime']['compatibility_patches']):
        patch = repo / 'tools/nemotron-diarization' / item['path']
        rows.append(('check-patch-' + str(index), ['git', 'apply', '--check', str(patch)], str(source)))
        rows.append(('apply-patch-' + str(index), ['git', 'apply', str(patch)], str(source)))
    rows.append(('materialize-twenty-patches', ['cmake', '-DSOURCE_DIR=' + str(source / 'llama.cpp'),
        '-DPATCH_DIR=' + str(source / 'patches'), '-DDEST_DIR=' + str(source / 'build/gen2-cpu-diar/_deps/llama.cpp'),
        '-P', str(source / 'cmake/llama_cpp.cmake')], str(repo)))
    return rows


def bootstrap(repo):
    tools = repo / 'tools/nemotron-diarization'
    recipe_path = tools / 'graph-cpu-local-replay.protocol.json'
    raw = capture(recipe_path)
    recipe = read(raw)
    if recipe['schema_version'] != 1 or recipe['native_protocol'] != NATIVE_PROTOCOL or sys.flags.optimize:
        raise ValueError('assertion-enabled fixed local replay recipe required')
    fixed = {recipe_path: identity(raw)}
    for name, expected in recipe['files'].items():
        capture(repo / name, expected)
        fixed[repo / name] = expected
    for name in ('tools/nemotron-diarization/graph_cpu_local_replay.py',
                 'tools/nemotron-diarization/graph_cpu_transport.py', ORIGIN):
        if repo / name not in fixed:
            raise ValueError('required replay source binding missing: ' + name)
    native_path = tools / 'windows-graph-cpu-profile.protocol.json'
    native_raw = capture(native_path, NATIVE_PROTOCOL)
    fixed[native_path] = NATIVE_PROTOCOL
    native = read(native_raw)
    for group, root in [('tools', tools), ('repository_files', repo)]:
        for name, expected in native[group].items():
            if root / name in fixed and fixed[root / name] != expected:
                raise ValueError('conflicting native replay binding')
            capture(root / name, expected)
            fixed[root / name] = expected
    from graph_cpu_transport import bind_route
    route, route_raw, route_frozen = bind_route(repo, True)

    def frozen():
        route_frozen()
        for path, expected in fixed.items():
            capture(path, expected)
    frozen()
    return route, route_raw, fixed, frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='fresh external replay directory')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if (args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir()
            or any(parent.is_symlink() for parent in args.output.parents)
            or args.output.resolve().is_relative_to(repo)):
        parser.error('fresh external replay output required')
    if shutil.disk_usage(args.output.parent).free < 4 * 1024**3:
        parser.error('original4GiB bounded local floor unmet')
    route, route_raw, fixed, frozen = bootstrap(repo)
    local = args.local.resolve(strict=True)
    if args.local.is_symlink() or local.is_relative_to(repo) or args.output.resolve().is_relative_to(local):
        parser.error('disjoint external local admission required')
    origin = capture(repo / ORIGIN, ORIGIN_ID)
    generated, edits = derive(origin)
    # Pure stdlib origin helpers verify all returned payloads before imports.
    tree = ast.parse(origin)
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in
        ('descriptor', 'no_duplicate_keys', 'read', 'require', 'verify_files', 'verify_collection')]
    scope = {'Path': Path, 'stat': stat, 'json': json, 'hashlib': hashlib}
    exec(compile(ast.Module(helpers, type_ignores=[]), 'bound-local-collection-helpers', 'exec'), scope)
    admission_path = local / 'admission.json'
    admission_raw = capture(admission_path)
    admission = read(admission_raw)
    delivered = local / 'materialized' / RESULT
    retained = scope['verify_collection'](delivered, route['collection'])
    source_root = local / 'delivery/source-preparation'
    source_collection = admission['source_preparation_proof']
    source_files = scope['verify_collection'](source_root, source_collection)
    scope['verify_files'](local / 'delivery', admission['delivery_members'])
    expected_zip = {key: route['delivery'][key] for key in ('bytes', 'sha256')}
    capture(local / 'delivery.zip', expected_zip)
    if (admission['archive'] != expected_zip or admission['remote_result_commit'] != route['result_commit']
            or admission['remote_source_commit'] != route['remote_source']
            or admission['remote_run_id'] != route['remote_run_id'] or admission['remote_job_id'] != route['remote_job_id']):
        raise ValueError('actual local admission differs from frozen return')
    commands_raw = capture(local / 'commands.json')
    sidecars = local_sidecars(local, route['collection']['members'])
    remote_pre = read(capture(local / 'delivery/pre-execution.json', admission['delivery_members']['pre-execution.json']))
    remote_commands = read(capture(local / 'delivery/commands.json', admission['delivery_members']['commands.json']))
    base, = [row for row in remote_commands if row['name'] == 'base']
    filenames = remote_pre['relocated_files']
    cache_paths = {Path(name).parent for name in filenames}
    if len(cache_paths) != 1 or len(filenames) != 3:
        raise ValueError('one original receiver cache and three programs required')
    cache_path, = cache_paths
    relocation = {'/Users/victor/workspace/gen2-nemotron3-n0': base['cwd'],
        '/Users/victor/.cache/gen2-nemotron3-n0': str(cache_path)}
    first = {str(path): expected for path, expected in fixed.items()}
    first.update({str(admission_path): identity(admission_raw), str(local / 'commands.json'): identity(commands_raw),
        str(local / 'delivery.zip'): expected_zip,
        str(delivered / 'collection.json'): route['collection']['manifest'],
        str(source_root / 'collection.json'): source_collection['manifest']})
    for root, files in [(delivered, retained), (source_root, source_files),
                        (local / 'delivery', admission['delivery_members'])]:
        for name, expected in files.items():
            path = root / name
            if str(path) in first and first[str(path)] != expected:
                raise ValueError('conflicting local replay binding')
            first[str(path)] = expected
    for path in sidecars:
        first[str(path)] = identity(capture(path))
    frozen()
    args.output.mkdir()
    replayer = args.output / 'replayer.py'
    with replayer.open('xb') as stream:
        stream.write(generated)
    first[str(replayer.resolve())] = identity(generated)
    pre = {'schema_version': 1, 'repo': str(repo), 'local': str(local), 'result_path': RESULT,
        'native_protocol': NATIVE_PROTOCOL, 'result_collection': route['collection'],
        'source_preparation_collection': source_collection, 'files': first,
        'local_sidecars': [str(path) for path in sidecars], 'literal_relocation': relocation,
        'route_protocol': identity(route_raw), 'route_files': route['files'],
        'native_source': route['native_source'], 'native_run': route['native_run'], 'native_job': route['native_job'],
        'native_feature_run': route['native_feature_run'], 'remote_source': route['remote_source'],
        'remote_result': route['result_commit'], 'remote_run': route['remote_run_id'], 'remote_job': route['remote_job_id'],
        'declared_edits': edits, 'derived_replay': identity(generated), 'new_compiler_sdk_model_calls': 0}
    pre_raw = (json.dumps(pre, indent=2) + '\n').encode()
    pre_path = args.output / 'pre-execution.json'
    with pre_path.open('xb') as stream:
        stream.write(pre_raw)
    frozen()
    capture(replayer, identity(generated))
    module = {'__name__': 'gen2.graph_cpu_local_replay', '__file__': str(replayer.resolve())}
    exec(compile(generated, str(replayer), 'exec'), module)
    previous = sys.argv
    try:
        sys.argv = [str(replayer), '--protocol', str(pre_path), '--protocol-sha256', identity(pre_raw)['sha256'],
            '--output', str(args.output / 'qualification.json')]
        module['main']()
    finally:
        sys.argv = previous
    frozen()


if __name__ == '__main__':
    main()
