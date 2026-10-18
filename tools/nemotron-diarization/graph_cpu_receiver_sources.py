"""Derive saved-data receivers for graph-only CPU evidence; never consume an artifact."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import stat

ORIGIN = 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-thread-cpu-admission-preparation'
ORIGINAL_SOURCE = '15c905d8815190a54f25452e7e51378c24f85914'
ORIGINAL_REPO = '/Users/victor/workspace/gen2-nemotron3-n0'
ORIGINAL_CACHE = '/Users/victor/.cache/gen2-nemotron3-n0'
NATIVE_PROTOCOL = {'bytes': 674377, 'sha256': '581a25fee8e4cf5191bc9b5d22884cc9429b4d6e8ceada46e954a531f10484fa'}
ORIGIN_FILES = {
    'collection.json': {'bytes': 3837, 'sha256': 'b293c9b4dec3a1ed503dc0ae48c7859a11077dbe6b07e5540f0b6868f62ebfd3'},
    'prepared/admit-windows-thread-cpu-profile-delivery-v1.py':
        {'bytes': 6720, 'sha256': '2e66fa2eeb099214044456d4dbf367306b9f6f91934e65220fb517f699b0ede3'},
    'prepared/replay-windows-thread-cpu-profile-supervision-v1.py':
        {'bytes': 6156, 'sha256': '7e4f2aaf3c00b547314374c6018f290579ad1ea2b860a6708412b11379d9df66'},
    'prepared/collect-windows-thread-cpu-profile-v1.py':
        {'bytes': 30461, 'sha256': '60f38bc5cfd875985de70c56b4e521c666532fc63787ec5ba9412120b4e26906'},
}

# This code is embedded in the collector, so no unbound project import is added.
PUBLICATION_CHECK = '''
expected_publication = {**copied, **bound, 'collector.py': identity,
    **{'sources/' + name: value for name, value in p['tools'].items()}}
summary_raw = (json.dumps(summary, indent=2) + '\\n').encode()
expected_publication['collection-qualification.json'] = {
    'bytes': len(summary_raw), 'sha256': hashlib.sha256(summary_raw).hexdigest()}
with (dest / 'collection-qualification.json').open('xb') as stream:
    stream.write(summary_raw)
def closed_publication(expected):
    import stat
    if not stat.S_ISDIR(dest.lstat().st_mode):
        raise ValueError('regular publication root required')
    directories = {parent.as_posix() for name in expected for parent in Path(name).parents if parent != Path('.')}
    observed = set()
    total = 0
    for path in dest.rglob('*'):
        name = path.relative_to(dest).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            if name not in directories:
                raise ValueError('unexpected publication directory')
        elif stat.S_ISREG(mode) and name in expected:
            if descriptor(path) != expected[name]:
                raise ValueError('published payload differs')
            observed.add(name)
            total += expected[name]['bytes']
        else:
            raise ValueError('unexpected or nonregular publication payload')
    if observed != set(expected) or len(observed) > 10000 or total > 1024**3:
        raise ValueError('closed bounded publication required')
closed_publication(expected_publication)
manifest_raw = (json.dumps({'schema_version': 1, 'files': expected_publication}, indent=2) + '\\n').encode()
with (dest / 'collection.json').open('xb') as stream:
    stream.write(manifest_raw)
closed_publication({**expected_publication, 'collection.json': {
    'bytes': len(manifest_raw), 'sha256': hashlib.sha256(manifest_raw).hexdigest()}})
'''


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def bound_origins(repo):
    files = {}
    for name, expected in ORIGIN_FILES.items():
        path = repo / ORIGIN / name
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError('regular retained receiver origin required')
        raw = path.read_bytes()
        if identity(raw) != expected:
            raise ValueError('retained receiver origin differs: ' + name)
        files[name] = raw
    manifest = json.loads(files['collection.json'])['files']
    if any(manifest[name] != expected for name, expected in ORIGIN_FILES.items() if name != 'collection.json'):
        raise ValueError('origin manifest does not bind all receiver programs')
    native = (repo / 'tools/nemotron-diarization/windows-graph-cpu-profile.protocol.json').read_bytes()
    if identity(native) != NATIVE_PROTOCOL:
        raise ValueError('exact graph-only native protocol required')
    return files


def relocate(raw, repo, cache):
    tree = ast.parse(raw)

    class Paths(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(node.value.replace(ORIGINAL_REPO, str(repo))
                    .replace(ORIGINAL_CACHE, str(cache))), node)
            return node

    tree = Paths().visit(tree)
    changed = (ast.unparse(ast.fix_missing_locations(tree)) + '\n').encode()
    if ast.dump(ast.parse(changed), include_attributes=False) != ast.dump(tree, include_attributes=False):
        raise ValueError('receiver AST changed beyond literal path relocation')
    return changed


def derive(originals, source, run_id, job_id):
    """Return source bytes and reversible declared edits, without running receivers."""
    if (not isinstance(source, str) or re.fullmatch(r'[0-9a-f]{40}', source) is None
            or any(type(value) is not int or value <= 0 for value in (run_id, job_id))):
        raise ValueError('full source commit and positive integer run/job identities required')
    results, journal = {}, {}
    for name, original in originals.items():
        if not name.startswith('prepared/'):
            continue
        if identity(original) != ORIGIN_FILES[name]:
            raise ValueError('exact receiver origin required before derivation')
        raw, edits = original.decode(), []

        def change(old, new, count=1):
            nonlocal raw
            if raw.count(old) != count or old == new:
                raise ValueError('exact receiver seam required: ' + old[:100])
            raw = raw.replace(old, new)
            edits.append({'old': old, 'new': new, 'count': count})

        change("def read(path):\n    return json.loads(path.read_text(encoding='utf-8-sig'))",
            "parsed_inputs = {}\n\ndef read(path):\n    raw = path.read_bytes()\n    observed = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}\n    first = parsed_inputs.setdefault(str(path), observed)\n    if observed != first:\n        raise ValueError('previously parsed JSON changed')\n    return json.loads(raw.decode('utf-8-sig'))")
        change(ORIGINAL_SOURCE, source)
        change('39083120058', str(run_id), 2 if name.startswith('prepared/replay-') else 1)
        change('117544022272', str(job_id))
        change('windows-thread-cpu-profile', 'windows-graph-cpu-profile', raw.count('windows-thread-cpu-profile'))
        if not name.startswith('prepared/replay-'):
            change("len(p['tools']) == 135", "len(p['tools']) == 149")
            change("len(p['repository_files']) == 2880", "len(p['repository_files']) == 2889")
            change("'tools_verified': 135", "'tools_verified': 149")
            change("'repository_files_verified': 2880", "'repository_files_verified': 2889")
        if name.startswith('prepared/admit-'):
            change("assert all(descriptor(root / name) == value for name, value in metadata.items())",
                "assert all(descriptor(root / name) == value for name, value in metadata.items())\nassert all(descriptor(Path(name)) == expected for name, expected in parsed_inputs.items())")
            change("'bytes': 670257, 'sha256': 'e9809ffe025bbe127c27daa5f8dfa07916cab7fe24a40d074651103b8c3cfc66'",
                "'bytes': 674377, 'sha256': '581a25fee8e4cf5191bc9b5d22884cc9429b4d6e8ceada46e954a531f10484fa'")
            change("             'Matrix source,coverage and saved report controls',\n             'Batched matrix source, compiler and evidence controls',\n             'Thread clock API adapter and complete counter traces',",
                "             'Source, parser and mock CPU API controls',\n             'Actual Windows CPU API control',\n             'Exact diagnostic source, compiler and evidence controls',")
            change("'Preserve successful or failed matrix profile evidence'", "'Preserve successful or failed diagnostic evidence'")
            change("log.count('Ran 15 tests') == 1 and log.count('Ran 21 tests') == 1 and log.count('Ran 10 tests') == 2 and log.count('Ran 8 tests') == 1",
                "log.count('Ran 15 tests') == 1 and log.count('Ran 7 tests') == 1 and log.count('Ran 1 test in') == 1 and log.count('Ran 10 tests') == 2")
            change("'actual_compiler_control_tests': 15, 'actual_matrix_profile_control_tests': 21, 'actual_batch_profile_control_tests':10, 'actual_thread_clock_tests':8,",
                "'actual_compiler_control_tests': 15, 'actual_graph_source_parser_mock_tests': 7, 'actual_windows_cpu_api_tests': 1, 'actual_graph_diagnostic_tests': 10,")
        if name.startswith('prepared/replay-'):
            change("admission_identity = descriptor(work / 'qualification.json')",
                "admission_identity = parsed_inputs[str(work / 'qualification.json')]")
            change("assert descriptor(Path(__file__)) == identity",
                "assert descriptor(Path(__file__)) == identity\nassert all(descriptor(Path(name)) == expected for name, expected in parsed_inputs.items())")
        if name.startswith('prepared/collect-'):
            change("bound = {name: descriptor(path) for name, path in bound_paths.items()}",
                "bound = {**admission['metadata'],\n    'delivery-admission.json': parsed_inputs[str(bound_paths['delivery-admission.json'])],\n    'artifact-members.json': admission['artifact_members'],\n    'supervision-replay.json': parsed_inputs[str(bound_paths['supervision-replay.json'])],\n    'admit-windows-graph-cpu-profile-delivery-v1.py': admission['collector'],\n    'replay-windows-graph-cpu-profile-supervision-v1.py': supervision['collector']}\nassert set(bound) == set(bound_paths)")
            change("assert descriptor(Path(__file__)) == identity",
                "assert descriptor(Path(__file__)) == identity\nassert all(descriptor(Path(name)) == expected for name, expected in parsed_inputs.items())")
            change('windows_thread_cpu_profile_', 'windows_graph_cpu_profile_', 3)
            change('from ggml_thread_cpu_profile import MAX_LOG_BYTES, summarize',
                'from ggml_graph_cpu_profile import MAX_LOG_BYTES, summarize')
            change("pristine = cache / 'NeMo-Speech.cpp'", "pristine = cache / 'windows-graph-cpu-profile-source-admission-v1/stage/pristine/native'")
            change("native = cache / 'windows-graph-cpu-profile-source-admission-v1/source'",
                "native = cache / 'windows-graph-cpu-profile-source-admission-v1/stage/source'")
            change("cache / 'windows-graph-cpu-profile-source-admission-v1/evidence/bounded-inference.patch'",
                "cache / 'windows-graph-cpu-profile-source-admission-v1/stage/evidence/bounded-inference.patch'")
            change('thread_cpu_totals', 'graph_cpu_totals', 4)
            change("('cycles','kernel_100ns','user_100ns','final_wait_cycles')", "('kernel_100ns','user_100ns')")
            change("worker['thread_cpu'][name]", "worker['graph_cpu'][name]")
            change(", 'kernel_cycles':0, 'barrier_cycles':0", '')
            change(", 'kernel_cycles', 'barrier_cycles'", '')
            change(",'chunks':[0,0],\n                        **{name:0 for name in ('first_cycles','second_cycles','conversion_cycles','inner_wait_cycles','chunk_cycles','kernel_cycles')}", ",'chunks':[0,0]")
            change(",'first_cycles','second_cycles','conversion_cycles','inner_wait_cycles','chunk_cycles','kernel_cycles'", '')
            change("            assert sum(value['kernel_cycles']for value in groups.values())==mul_mat['kernel_cycles']\n", '')
            change("'per_worker_graph_cpu_totals':graph_cpu_totals,'cpu_time_units':'100ns','cycle_units':'raw-thread-cycles'",
                "'per_worker_graph_cpu_totals':graph_cpu_totals,'cpu_time_units':'100ns'")
            change('2026-10-17-thread-cpu-profile', '2026-10-17-graph-cpu-profile')
            change('Raw thread cycles and CPU100ns durations remain separate from wall microseconds; no frequency conversion, exact waits, pure arithmetic or causal scheduling inference. Preserve zeros and all original observer bounds.',
                'Graph-only kernel/user CPU100ns durations remain separate from wall microseconds. No per-region CPU attribution, cycles, frequency conversion, exact waits, pure arithmetic or causal scheduling inference. Preserve zeros and both original observer bounds.')
            change("(dest / 'collection-qualification.json').write_text(json.dumps(summary, indent=2) + '\\n')\n(dest / 'collection.json').write_text(json.dumps({'schema_version': 1, 'files': file_inventory(dest)}, indent=2) + '\\n')",
                PUBLICATION_CHECK.strip())
        ast.parse(raw)
        target = Path(name).name.replace('thread-cpu', 'graph-cpu')
        results[target], journal[target] = raw.encode(), edits
    if len(results) != 3:
        raise ValueError('all three saved-data receivers required')
    return results, journal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--run', type=int, required=True)
    parser.add_argument('--job', type=int, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if (args.output.is_symlink() or args.output.exists() or not args.output.parent.is_dir()
            or args.output.resolve().is_relative_to(repo)):
        parser.error('fresh external receiver preparation output required')
    originals = bound_origins(repo)
    programs, edits = derive(originals, args.source, args.run, args.job)
    programs = {name: relocate(raw, repo, args.cache.resolve()) for name, raw in programs.items()}
    args.output.mkdir()
    for name, raw in programs.items():
        with (args.output / name).open('xb') as stream:
            stream.write(raw)
    report = {'source_commit': args.source, 'run_id': args.run, 'job_id': args.job,
        'native_protocol': NATIVE_PROTOCOL, 'origins': ORIGIN_FILES,
        'programs': {name: identity(raw) for name, raw in programs.items()},
        'declared_edits': edits, 'artifact_consumed': False, 'receivers_invoked': False,
        'new_native_calls': 0,
        'limits': 'Source preparation only. Actual API/archive binding, original 8 GiB Linux floor, owned process supervision, full saved-data execution, bounded result-only return and independent local replay remain required.'}
    with (args.output / 'preparation.json').open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'programs': report['programs'], 'artifact_consumed': False, 'receivers_invoked': False}))


if __name__ == '__main__':
    main()
