"""Bind the graph-only replay route to the existing owned transport algorithms."""
import ast
import hashlib
import json
from pathlib import Path
import re
import stat
import sys
import types

NATIVE_PROTOCOL = {'bytes': 674377, 'sha256': '581a25fee8e4cf5191bc9b5d22884cc9429b4d6e8ceada46e954a531f10484fa'}
RECIPE = {'bytes': 5255, 'sha256': 'a304cd15a12cc6d60d42493c2871c14e668554f69b0d98c4733d084dfba717d4'}
ORIGINS = {
    'remote_thread_cpu_replay.py': {'bytes': 20063, 'sha256': '3b58c3c8c320c6bacf45b2e0b7a9649da9b382ffed6b5af890a6795408b1a14c'},
    'receive_thread_cpu_bundle.py': {'bytes': 17675, 'sha256': '4f5f30613e5245fedb5ba8936bfd2b6c0ec3c8689329b04cf56d2e42e1054ecf'},
}
RESULT = 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-graph-cpu-profile'
EXECUTED = ('admit-windows-graph-cpu-profile-delivery-v1.py',
    'replay-windows-graph-cpu-profile-supervision-v1.py', 'collect-windows-graph-cpu-profile-v1.py')

REMOTE_PREPARATION = '''    from graph_cpu_receiver_sources import bound_origins, derive, relocate as graph_relocate
    originals = bound_origins(repo)
    generated, declared_edits = derive(originals, SOURCE, RUN, JOB)
    relocated = {}
    for name, raw in generated.items():
        changed = graph_relocate(raw, repo, cache)
        target = cache / name
        with target.open('xb') as stream:
            stream.write(changed)
        relocated[str(target)] = byte_descriptor(changed)
    fixed.update(relocated)
    def prepared_frozen():
        frozen()
    dump(output / 'pre-execution.json', {'source': base, 'native_source': SOURCE,
        'native_run': RUN, 'native_job': JOB, 'native_archive': ARCHIVE,
        'relocated_files': relocated, 'declared_edits': declared_edits,
        'artifact_consumed': False, 'new_native_calls': 0})
    python = [sys.executable, '-B']
    source_parent = cache / 'windows-graph-cpu-profile-source-admission-v1'
    source_parent.mkdir()
    source_root = source_parent / 'stage'
    run('source-only-preparation', python + [tools / 'graph_cpu_source_admission.py',
        '--output', source_root, '--worker'], owned_stage=True)
    prepared_frozen()
'''


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def strict_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate protocol key: ' + key)
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique)


def admit_identities(protocol, local):
    def commit(value):
        return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{40}', value) is not None

    def positive(value):
        return type(value) is int and value > 0

    def descriptor(value, cap):
        return (isinstance(value, dict) and set(value) == {'bytes', 'sha256'}
            and positive(value['bytes']) and value['bytes'] <= cap
            and isinstance(value['sha256'], str) and re.fullmatch(r'[0-9a-f]{64}', value['sha256']) is not None)

    if (not commit(protocol.get('native_source'))
            or not all(positive(protocol.get(key)) for key in ('native_run', 'native_job', 'native_feature_run', 'artifact_id'))
            or not descriptor(protocol.get('archive'), 128 * 1024**2)):
        raise ValueError('freeze actual native source/run/job/feature/archive identities before consumption')
    if local:
        delivery = protocol.get('delivery')
        collection = protocol.get('collection')
        collection_valid = (isinstance(collection, dict)
            and set(collection) == {'members', 'bytes_without_manifest', 'manifest'}
            and positive(collection['members']) and collection['members'] <= 10000
            and type(collection['bytes_without_manifest']) is int
            and 0 <= collection['bytes_without_manifest'] <= 1024**3
            and descriptor(collection['manifest'], 128 * 1024**2))
        if (not commit(protocol.get('remote_source')) or not commit(protocol.get('result_commit'))
                or not all(positive(protocol.get(key)) for key in ('remote_run_id', 'remote_job_id'))
                or not descriptor(protocol.get('bundle'), 128 * 1024**2)
                or not isinstance(delivery, dict) or set(delivery) != {'id', 'bytes', 'sha256'}
                or not positive(delivery['id'])
                or not descriptor({key: delivery[key] for key in ('bytes', 'sha256')}, 128 * 1024**2)
                or not collection_valid):
            raise ValueError('freeze actual bounded return and result identities before local admission')


def bind_route(repo, local):
    """Bind the complete route/native closure before executing any project helper."""
    if sys.flags.optimize:
        raise ValueError('assertion-enabled saved-data replay required')
    tools = repo / 'tools/nemotron-diarization'
    filename = 'graph-cpu-bundle-receive.protocol.json' if local else 'remote-graph-cpu-replay.protocol.json'
    path = tools / filename
    raw = path.read_bytes()
    protocol = strict_json(raw)
    admit_identities(protocol, local)
    if (protocol['schema_version'] != 1 or protocol['native_protocol'] != NATIVE_PROTOCOL
            or protocol['receiver_recipe'] != RECIPE
            or protocol['original_start_floor_bytes'] != 8 * 1024**3
            or protocol['native_archive_cap_bytes'] != 256 * 1024**2
            or protocol['bundle_cap_bytes'] != 128 * 1024**2
            or (local and (protocol['start_floor'] != 4 * 1024**3
                or protocol['zip_cap'] != 128 * 1024**2 or protocol['extract_cap'] != 192 * 1024**2
                or protocol['bundle_cap'] != 128 * 1024**2))):
        raise ValueError('exact graph-only route and original bounds required')
    native_path = tools / 'windows-graph-cpu-profile.protocol.json'
    native_raw = native_path.read_bytes()
    if identity(native_raw) != NATIVE_PROTOCOL:
        raise ValueError('exact graph-only native protocol required')
    native = strict_json(native_raw)
    recipe_path = tools / 'graph-cpu-receiver-sources.protocol.json'
    recipe_raw = recipe_path.read_bytes()
    if identity(recipe_raw) != RECIPE:
        raise ValueError('exact saved-data receiver recipe required')
    recipe = strict_json(recipe_raw)
    fixed = {native_path: NATIVE_PROTOCOL, path: identity(raw), recipe_path: RECIPE}
    for additions in ({tools / name: value for name, value in native['tools'].items()},
            {repo / name: value for name, value in native['repository_files'].items()},
            {repo / name: value for name, value in recipe['files'].items()},
            {repo / name: value for name, value in protocol['files'].items()}):
        for target, expected in additions.items():
            if target in fixed and fixed[target] != expected:
                raise ValueError('route cannot replace an existing native binding')
            fixed[target] = expected
    for name in ('graph_cpu_transport.py', 'remote_q8_apack_replay.py', *ORIGINS):
        if tools / name not in fixed:
            raise ValueError('required first-bound transport source missing: ' + name)

    def frozen():
        for target, expected in fixed.items():
            if not stat.S_ISREG(target.lstat().st_mode) or identity(target.read_bytes()) != expected:
                raise ValueError('first-bound transport input changed: ' + str(target))
    frozen()
    return protocol, raw, frozen


def adapt_source(raw, local):
    origin = 'receive_thread_cpu_bundle.py' if local else 'remote_thread_cpu_replay.py'
    if identity(raw) != ORIGINS[origin]:
        raise ValueError('exact original owned transport source required')
    text = raw.decode()
    changes = []

    def change(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError('exact transport seam required: ' + old[:100])
        text = text.replace(old, new)
        changes.append((old, new))

    if local:
        change("protocol_path=Path(__file__).parent/'thread-cpu-bundle-receive.protocol.json';protocol_raw=protocol_path.read_bytes()",
            "protocol_path=Path(__file__).parent/'thread-cpu-bundle-receive.protocol.json';protocol_raw=protocol_path.read_bytes()\n    if protocol_raw != WRAPPER_PROTOCOL_RAW:raise ValueError('first-bound wrapper protocol changed')")
        change("    admit_bundle_metadata(metadata,bundle)",
            "    admit_bundle_metadata(metadata,bundle)\n    feature = admit_graph_feature(\n        strict_json((extracted/'commands/api-native-feature.stdout.log').read_bytes()),\n        strict_json((extracted/'commands/api-native-feature-jobs.stdout.log').read_bytes()))\n    if metadata['native_feature_run_id'] != NATIVE_FEATURE or metadata['feature_admission'] != feature:raise ValueError('returned feature proof differs')")
    else:
        first = text.index('    parent=repo/PARENT\n')
        last = text.index("    source_q_raw=(source_root/'qualification.json').read_bytes()", first)
        change(text[first:last], REMOTE_PREPARATION)
        change("or len(source_q['materialized_files'])!=3425 or len(source_q['commands'])!=11",
            "or len(source_q['materialized_files'])!=3425 or len(source_q['commands'])!=19")
        change("protocol_raw=protocol_path.read_bytes();protocol=json.loads(protocol_raw)",
            "protocol_raw=protocol_path.read_bytes();protocol=json.loads(protocol_raw)\n    if protocol_raw != WRAPPER_PROTOCOL_RAW:raise ValueError('first-bound wrapper protocol changed')")
        change("        frozen()\n        row=bounded_process(",
            "        frozen()\n        free = shutil.disk_usage(cache).free\n        if free < 8*1024**3:raise ValueError('original8GiB command start floor unmet')\n        row=bounded_process(")
        change("commands.append({'name':name,'owned_stage_inherited_children':owned_stage,**row})",
            "commands.append({'name':name,'owned_stage_inherited_children':owned_stage,'free_bytes_before':free,**row})")
        change("    archive_path=data/'artifact.zip'",
            "    if shutil.disk_usage(cache).free < 8*1024**3:raise ValueError('original8GiB archive start floor unmet')\n    archive_path=data/'artifact.zip'")
        change("    admit_api(api['producer-run.json'],api['job.json'],api['artifacts.json'])",
            "    admit_api(api['producer-run.json'],api['job.json'],api['artifacts.json'])\n    feature_raw = run('api-native-feature', gh+[f'repos/saberra-ai/gen2/actions/runs/{FEATURE}'])\n    feature_jobs_raw = run('api-native-feature-jobs', gh+[f'repos/saberra-ai/gen2/actions/runs/{FEATURE}/jobs?per_page=100'])\n    feature_proof = admit_graph_feature(strict_json(feature_raw),strict_json(feature_jobs_raw))\n    fixed[str(logs/'api-native-feature.stdout.log')] = byte_descriptor(feature_raw)\n    fixed[str(logs/'api-native-feature-jobs.stdout.log')] = byte_descriptor(feature_jobs_raw)")
        change("    bundle=output/'receiver-result.bundle'",
            "    if shutil.disk_usage(cache).free < 8*1024**3:raise ValueError('original8GiB bundle start floor unmet')\n    bundle=output/'receiver-result.bundle'")
        change("'native_source_commit':SOURCE,'native_run_id':RUN,'native_job_id':JOB,'native_archive':ARCHIVE,",
            "'native_source_commit':SOURCE,'native_run_id':RUN,'native_job_id':JOB,'native_archive':ARCHIVE,\n        'native_feature_run_id':FEATURE,'feature_admission':feature_proof,")
    # Only string literals receive the diagnostic identity/name substitution.
    tree = ast.parse(text)

    class Names(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(node.value.replace('thread-cpu', 'graph-cpu')), node)
            return node

    tree = Names().visit(tree)
    changed = (ast.unparse(ast.fix_missing_locations(tree)) + '\n').encode()
    if ast.dump(ast.parse(changed), include_attributes=False) != ast.dump(tree, include_attributes=False):
        raise ValueError('transport AST differs beyond declared source/name changes')
    return changed, changes


def configured_module(repo, protocol, raw_protocol, local):
    """Compile only hash-bound transport code; configuration changes data globals."""
    admit_identities(protocol, local)
    tools = repo / 'tools/nemotron-diarization'
    origin = 'receive_thread_cpu_bundle.py' if local else 'remote_thread_cpu_replay.py'
    changed, _ = adapt_source((tools / origin).read_bytes(), local)
    module = types.ModuleType('gen2.graph_cpu_local' if local else 'gen2.graph_cpu_remote')
    module.__file__ = str(tools / 'graph_cpu_transport.py')
    exec(compile(changed, module.__file__, 'exec'), module.__dict__)
    configuration = {'WRAPPER_PROTOCOL_RAW': raw_protocol, 'ARCHIVE': protocol['archive'],
        'NATIVE_PROTOCOL': NATIVE_PROTOCOL}
    source, feature_id = protocol['native_source'], protocol['native_feature_run']
    configuration['admit_graph_feature'] = lambda run, jobs: admit_feature(run, jobs, source, feature_id)
    if local:
        configuration.update(NATIVE_SOURCE=protocol['native_source'], NATIVE_RUN=protocol['native_run'],
            NATIVE_JOB=protocol['native_job'], RESULTS=(RESULT,), VERIFIER=protocol['remote_source'],
            RUN=protocol['remote_run_id'], JOB=protocol['remote_job_id'], DELIVERY=protocol['delivery'],
            RESULT_COMMIT=protocol['result_commit'], BUNDLE=protocol['bundle'], COLLECTION=protocol['collection'],
            NATIVE_FEATURE=feature_id)
    else:
        configuration.update(SOURCE=protocol['native_source'], RUN=protocol['native_run'],
            JOB=protocol['native_job'], FEATURE=protocol['native_feature_run'], ARTIFACT=protocol['artifact_id'],
            RESULT=RESULT, EXECUTED=EXECUTED, strict_json=strict_json)
    module.__dict__.update(configuration)
    return module


def main(local=False):
    repo = Path(__file__).resolve().parents[2]
    protocol, raw, frozen = bind_route(repo, local)
    module = configured_module(repo, protocol, raw, local)
    frozen()
    module.main()
    frozen()


def admit_feature(run, jobs, source, feature_id):
    expected = {'evidence-tools'}
    for system in ('ubuntu-latest', 'windows-latest', 'macos-latest'):
        for features in ('', 'backend-diar-nemo', 'asr-dynamic,backend-diar-nemo',
                         'backend-llamacpp,asr-dynamic,backend-diar-nemo'):
            expected.add('features (' + system + (', ' + features if features else '') + ')')
    rows = jobs['jobs']
    if (run['id'] != feature_id or run['head_sha'] != source
            or run['repository']['full_name'] != 'saberra-ai/gen2'
            or run['path'] != '.github/workflows/diarization.yml'
            or run['status'] != 'completed' or run['conclusion'] != 'success'
            or jobs['total_count'] != 13 or len(rows) != 13 or {row['name'] for row in rows} != expected
            or any(row['run_id'] != feature_id or row['head_sha'] != source
                or row['status'] != 'completed' or row['conclusion'] != 'success' for row in rows)):
        raise ValueError('all thirteen exact native-source feature jobs must pass')
    return {'source_commit': source, 'run_id': feature_id, 'jobs': 13, 'all_passed': True}


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ('remote', 'local'):
        raise SystemExit('usage: graph_cpu_transport.py {remote|local} [receiver arguments]')
    mode = sys.argv.pop(1)
    main(local=mode == 'local')
