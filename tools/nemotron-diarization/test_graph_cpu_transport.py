"""Graph-only transport identity, preparation, feature and actual quarantine controls."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import graph_cpu_transport as target

REPO = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parent


def fixture_protocol():
    """Explicit fixture identities; no real diagnostic/result identity is implied."""
    return {'native_source': 'a' * 40, 'native_run': 123, 'native_job': 456,
        'native_feature_run': 789, 'artifact_id': 111,
        'archive': {'bytes': 100, 'sha256': 'b' * 64}, 'remote_source': 'c' * 40,
        'remote_run_id': 222, 'remote_job_id': 333, 'result_commit': 'd' * 40,
        'delivery': {'id': 444, 'bytes': 200, 'sha256': 'e' * 64},
        'bundle': {'bytes': 150, 'sha256': 'f' * 64},
        'collection': {'members': 2, 'bytes_without_manifest': 1,
            'manifest': {'bytes': 100, 'sha256': '1' * 64}}}


class GraphCPUTransportTest(unittest.TestCase):
    def test_unset_recipes_reject_before_any_helper_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            tools = repo / 'tools/nemotron-diarization'
            tools.mkdir(parents=True)
            for local, filename in [(False, 'remote-graph-cpu-replay.protocol.json'),
                                    (True, 'graph-cpu-bundle-receive.protocol.json')]:
                (tools / filename).write_text(json.dumps({**fixture_protocol(), 'native_source': None}))
                with self.subTest(local=local), self.assertRaisesRegex(ValueError, 'freeze actual native'):
                    target.bind_route(repo, local)

    def test_complete_first_binding_conflicting_pins_and_later_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            tools = repo / 'tools/nemotron-diarization'
            tools.mkdir(parents=True)
            names = ['graph_cpu_transport.py', 'remote_q8_apack_replay.py', *target.ORIGINS]
            for name in names:
                (tools / name).write_bytes(b'inert fixture source ' + name.encode())
            native = json.dumps({'tools': {'remote_q8_apack_replay.py':
                target.identity((tools / 'remote_q8_apack_replay.py').read_bytes())}, 'repository_files': {}}).encode()
            recipe = b'{"files":{}}'
            (tools / 'windows-graph-cpu-profile.protocol.json').write_bytes(native)
            (tools / 'graph-cpu-receiver-sources.protocol.json').write_bytes(recipe)
            protocol = {**fixture_protocol(), 'schema_version': 1, 'native_protocol': target.identity(native),
                'receiver_recipe': target.identity(recipe), 'original_start_floor_bytes': 8 * 1024**3,
                'native_archive_cap_bytes': 256 * 1024**2, 'bundle_cap_bytes': 128 * 1024**2,
                'files': {'tools/nemotron-diarization/' + name: target.identity((tools / name).read_bytes())
                    for name in names}}
            path = tools / 'remote-graph-cpu-replay.protocol.json'
            path.write_text(json.dumps(protocol))
            with patch.multiple(target, NATIVE_PROTOCOL=target.identity(native), RECIPE=target.identity(recipe)):
                value, raw, frozen = target.bind_route(repo, False)
                self.assertEqual(value, protocol)
                self.assertEqual(raw, path.read_bytes())
                frozen()
                for name in names:
                    payload = tools / name
                    first = payload.read_bytes()
                    payload.write_bytes(first + b'changed')
                    with self.subTest(changed=name), self.assertRaises(ValueError):
                        frozen()
                    payload.write_bytes(first)
                key = 'tools/nemotron-diarization/remote_q8_apack_replay.py'
                protocol['files'][key] = target.identity(b'other')
                path.write_text(json.dumps(protocol))
                with self.assertRaisesRegex(ValueError, 'cannot replace'):
                    target.bind_route(repo, False)

    def test_exact_origins_and_whole_ast_outside_declared_seams(self):
        for local, name in [(False, 'remote_thread_cpu_replay.py'), (True, 'receive_thread_cpu_bundle.py')]:
            raw = (TOOLS / name).read_bytes()
            changed, edits = target.adapt_source(raw, local)
            expected = raw.decode()
            for old, new in edits:
                self.assertEqual(expected.count(old), 1)
                expected = expected.replace(old, new)
            # Compare the complete expected AST after only diagnostic string-literal changes.
            tree = ast.parse(expected)
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    node.value = node.value.replace('thread-cpu', 'graph-cpu')
            self.assertEqual(ast.dump(ast.parse(changed), include_attributes=False),
                ast.dump(tree, include_attributes=False))
            with self.assertRaises(ValueError):
                target.adapt_source(raw + b'\n# changed\n', local)

    def test_unset_bad_and_oversize_identity_mutations_reject_before_consumption(self):
        protocol = fixture_protocol()
        for local in (False, True):
            target.admit_identities(protocol, local)
            mutations = [('native_source', None), ('native_source', 'A' * 40),
                ('native_run', 0), ('native_job', True), ('native_feature_run', -1), ('artifact_id', None),
                ('archive', {'bytes': 128 * 1024**2 + 1, 'sha256': 'b' * 64})]
            mutations += ([('remote_source', None), ('remote_run_id', False), ('remote_job_id', -1),
                ('result_commit', 'c' * 39), ('delivery', {}), ('bundle', {}), ('collection', None),
                ('collection', {}), ('collection', {'members': 2}),
                ('collection', {**protocol['collection'], 'members': True}),
                ('collection', {**protocol['collection'], 'members': 10001}),
                ('collection', {**protocol['collection'], 'bytes_without_manifest': -1}),
                ('collection', {**protocol['collection'], 'bytes_without_manifest': 1024**3 + 1}),
                ('collection', {**protocol['collection'], 'manifest': {'bytes': 100, 'sha256': 'bad'}}),
                ('collection', {**protocol['collection'], 'manifest': {'bytes': 128 * 1024**2 + 1, 'sha256': '1' * 64}})]
                if local else [('artifact_id', None)])
            for key, value in mutations:
                with self.subTest(local=local, key=key), self.assertRaises(ValueError):
                    target.admit_identities({**protocol, key: value}, local)
        with self.assertRaises(ValueError):
            target.strict_json(b'{"native_run":123,"native_run":999}')

    def feature_fixture(self):
        protocol = fixture_protocol()
        rows = [{'name': 'evidence-tools'}]
        for system in ('ubuntu-latest', 'windows-latest', 'macos-latest'):
            for features in ('', 'backend-diar-nemo', 'asr-dynamic,backend-diar-nemo',
                    'backend-llamacpp,asr-dynamic,backend-diar-nemo'):
                rows.append({'name': 'features (' + system + (', ' + features if features else '') + ')'})
        for row in rows:
            row.update(run_id=789, head_sha=protocol['native_source'], status='completed', conclusion='success')
        run = {'id': 789, 'head_sha': protocol['native_source'], 'repository': {'full_name': 'saberra-ai/gen2'},
            'path': '.github/workflows/diarization.yml', 'status': 'completed', 'conclusion': 'success'}
        return run, {'total_count': 13, 'jobs': rows}

    def test_all_thirteen_feature_jobs_are_required_at_the_same_native_source(self):
        original = self.feature_fixture()
        self.assertEqual(target.admit_feature(*original, 'a' * 40, 789),
            {'source_commit': 'a' * 40, 'run_id': 789, 'jobs': 13, 'all_passed': True})
        mutations = [(0, ['head_sha'], 'b' * 40), (0, ['path'], 'other'),
            (0, ['status'], 'in_progress'), (0, ['conclusion'], 'failure'),
            (1, ['total_count'], 14), (1, ['jobs', 0, 'run_id'], 777),
            (1, ['jobs', 0, 'head_sha'], 'b' * 40), (1, ['jobs', 0, 'conclusion'], 'skipped'),
            (1, ['jobs', 0, 'name'], original[1]['jobs'][1]['name'])]
        for group, keys, value in mutations:
            changed = copy.deepcopy(original)
            parent = changed[group]
            for key in keys[:-1]:
                parent = parent[key]
            parent[keys[-1]] = value
            with self.subTest(keys=keys), self.assertRaises(ValueError):
                target.admit_feature(*changed, 'a' * 40, 789)

    def test_real_receiver_preparation_before_network_keeps_owned_source_stage(self):
        protocol = fixture_protocol()
        module = target.configured_module(REPO, protocol, b'fixture protocol', False)
        tree = ast.parse(target.adapt_source((TOOLS / 'remote_thread_cpu_replay.py').read_bytes(), False)[0])
        body = next(node.body for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        start = next(i for i, node in enumerate(body) if isinstance(node, ast.ImportFrom)
            and node.module == 'graph_cpu_receiver_sources')
        stop = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'source_q_raw' for t in node.targets))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, output = root / 'cache', root / 'output'
            cache.mkdir()
            output.mkdir()
            fixed, calls = {}, []

            def frozen():
                for name, expected in fixed.items():
                    self.assertEqual(module.descriptor(Path(name)), expected)

            def run(name, command, **kwargs):
                calls.append((name, command, kwargs))

            scope = {**vars(module), 'repo': REPO, 'tools': TOOLS, 'cache': cache, 'output': output,
                'base': 'fixture', 'fixed': fixed, 'frozen': frozen, 'run': run}
            exec(compile(ast.Module(body[start:stop], type_ignores=[]), 'actual-graph-receiver-preparation', 'exec'), scope)
            self.assertEqual(set(path.name for path in cache.iterdir()),
                set(target.EXECUTED) | {'windows-graph-cpu-profile-source-admission-v1'})
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], 'source-only-preparation')
            self.assertEqual(calls[0][1][-3:], ['--output', scope['source_root'], '--worker'])
            self.assertEqual(calls[0][2], {'owned_stage': True})
            for name in target.EXECUTED:
                self.assertIn(str(cache / name), fixed)
            report = json.loads((output / 'pre-execution.json').read_bytes())
            self.assertFalse(report['artifact_consumed'])
            self.assertEqual(len(report['relocated_files']), 3)
            path = cache / target.EXECUTED[0]
            path.write_bytes(path.read_bytes() + b'\n# changed\n')
            with self.assertRaises(AssertionError):
                scope['prepared_frozen']()

    def test_remote_native_api_uses_new_frozen_identity_and_artifact_name(self):
        module = target.configured_module(REPO, fixture_protocol(), b'fixture', False)
        run = {'id': 123, 'head_sha': 'a' * 40, 'repository': {'full_name': 'saberra-ai/gen2'},
            'status': 'completed', 'conclusion': 'success'}
        job = {'id': 456, 'run_id': 123, 'head_sha': 'a' * 40, 'status': 'completed', 'conclusion': 'success'}
        artifact = {'id': 111, 'name': 'nemotron-windows-graph-cpu-profile', 'expired': False,
            'workflow_run': {'id': 123, 'head_sha': 'a' * 40}, 'size_in_bytes': 100, 'digest': 'sha256:' + 'b' * 64}
        module.admit_api(run, job, {'total_count': 1, 'artifacts': [artifact]})
        for key, value in [('name', 'nemotron-windows-thread-cpu-profile'), ('digest', 'sha256:' + '0' * 64),
                           ('size_in_bytes', 101), ('id', 112)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                module.admit_api(run, job, {'total_count': 1, 'artifacts': [{**artifact, key: value}]})


# Reuse the complete existing raw-ZIP/quarantine regression suite against the
# configured graph-only module, including real result-only Git bundles. The
# retained test source is fixed; only its receiver import and name literals change.
def graph_return_tests():
    raw = (TOOLS / 'test_receive_thread_cpu_bundle.py').read_bytes()
    if target.identity(raw) != {'bytes': 11027, 'sha256': '964b878f4ae165016a84beec7a5f92eefe864ddb9eae6754f6b3b54f531f401f'}:
        raise ValueError('original local quarantine controls changed')
    source = raw.decode()
    if source.count('import receive_thread_cpu_bundle as receiver') != 1:
        raise ValueError('one receiver import seam required')
    source = source.replace('import receive_thread_cpu_bundle as receiver', 'receiver = CONFIGURED_RECEIVER')
    seam = 'current=ast.parse(Path(receiver.__file__).read_bytes())'
    if source.count(seam) != 1:
        raise ValueError('one derived transport AST inspection seam required')
    source = source.replace(seam, 'current=ast.parse(CONFIGURED_SOURCE)')
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace('thread-cpu', 'graph-cpu')
    namespace = {'__name__': 'gen2.graph_cpu_quarantine_controls', '__file__': str(Path(__file__)),
        'CONFIGURED_RECEIVER': target.configured_module(REPO, fixture_protocol(), b'fixture protocol', True),
        'CONFIGURED_SOURCE': target.adapt_source((TOOLS / 'receive_thread_cpu_bundle.py').read_bytes(), True)[0]}
    exec(compile(ast.fix_missing_locations(tree), 'graph-quarantine-controls', 'exec'), namespace)
    return namespace['BundleReceiverTest']


GraphCPUBundleReceiverTest = graph_return_tests()


if __name__ == '__main__':
    unittest.main()
