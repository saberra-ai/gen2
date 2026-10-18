"""Local graph-only replay derivation and actual AST/read/source-command controls."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import textwrap
import unittest

import graph_cpu_local_replay as target


class GraphCPULocalReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parents[2]
        cls.origin = (cls.repo / target.ORIGIN).read_bytes()
        cls.generated, cls.edits = target.derive(cls.origin)
        cls.tree = ast.parse(cls.generated)

    def pure_functions(self, *names):
        nodes = [node for node in self.tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        scope = {'ast': ast, 'Path': Path, 'json': json, 'hashlib': hashlib,
            'READ_EXPECTED': {}, 'READ_CAPTURED': {}}
        exec(compile(ast.Module(nodes, type_ignores=[]), 'actual-local-replay-functions', 'exec'), scope)
        return scope

    def test_whole_replay_outside_declared_seams_is_unchanged(self):
        expected = self.origin.decode()
        for old, new, count in self.edits:
            self.assertEqual(expected.count(old), count)
            expected = expected.replace(old, new)
        self.assertEqual(self.generated, expected.encode())
        with self.assertRaises(ValueError):
            target.derive(self.origin + b'\n# changed\n')
        for assertion in ["len(commands) == 59", 'len(database) == 97',
                "'model_free_vectors': 144", "'full_sequence_bit_exact': True",
                'scope[\'timing\']', 'original-exact-59-command-order',
                "timing == q['timing_comparison']", 'original-saved-12-graph-control-admission']:
            self.assertIn(assertion.encode(), self.generated)

    def test_all_three_actual_graph_receiver_asts_and_statement_assertion_path_mutations(self):
        from graph_cpu_receiver_sources import bound_origins, derive, relocate
        programs, _ = derive(bound_origins(self.repo), 'a' * 40, 123, 456)
        scope = self.pure_functions('require', 'reviewed_body')
        relocation = {'/Users/victor/workspace/gen2-nemotron3-n0': '/ci/repo',
            '/Users/victor/.cache/gen2-nemotron3-n0': '/ci/cache'}
        with tempfile.TemporaryDirectory() as directory:
            for name, original in programs.items():
                returned = relocate(original, Path('/ci/repo'), Path('/ci/cache'))
                path = Path(directory) / name
                path.write_bytes(returned)
                observed = scope['reviewed_body'](original, path, relocation)
                self.assertEqual(ast.dump(ast.Module(observed, type_ignores=[]), include_attributes=False),
                    ast.dump(ast.parse(original), include_attributes=False))
                variants = [returned + b'\nraise RuntimeError("must not execute")\n',
                    returned.replace(b"q['source_commit'] == source", b"q['source_commit'] != source", 1),
                    returned.replace(b'/ci/repo', b'/ci/other-repo', 1)]
                for mutation in variants:
                    self.assertNotEqual(mutation, returned)
                    path.write_bytes(mutation)
                    with self.subTest(receiver=name), self.assertRaises(ValueError):
                        scope['reviewed_body'](original, path, relocation)

    def test_actual_reader_parses_only_admitted_bytes_and_rejects_later_changes(self):
        scope = self.pure_functions('require', 'read', 'no_duplicate_keys')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / 'record.json'
            original = b'{"admitted":true}\n'
            path.write_bytes(original)
            scope['READ_EXPECTED'][str(path)] = target.identity(original)
            self.assertEqual(scope['read'](path), {'admitted': True})
            self.assertEqual(scope['READ_CAPTURED'], {str(path): target.identity(original)})
            path.write_bytes(b'{"admitted":false}\n')
            with self.assertRaisesRegex(ValueError, 'admitted bytes'):
                scope['read'](path)
            scope['READ_EXPECTED'].clear()
            with self.assertRaisesRegex(ValueError, 'previously parsed'):
                scope['read'](path)
            scope['READ_CAPTURED'].clear()
            path.write_bytes(b'{"a":1,"a":2}')
            with self.assertRaisesRegex(ValueError, 'duplicate JSON'):
                scope['read'](path)

    def test_actual_graph_only_observations_preserve_zero_units_and_separate_workers(self):
        from ggml_graph_cpu_profile import summarize
        from test_ggml_graph_cpu_profile import fixture
        profile = summarize('\n'.join(fixture()), True, expected_graphs=1, expected_threads=4, expected_matrices=1)
        scope = {'Counter': Counter, 'read': lambda path: profile, 'evidence': Path('/fixture')}
        exec(compile(textwrap.dedent(target.OBSERVATIONS), 'actual-local-graph-cpu-observations', 'exec'), scope)
        self.assertEqual(scope['zero_fields'], {'graph.kernel_100ns': 8, 'graph.user_100ns': 8})
        self.assertEqual(len(scope['worker_observations']), 2)
        for case in scope['worker_observations']:
            self.assertEqual(set(case['per_worker']), {'0', '1', '2', '3'})
            self.assertTrue(all(set(value) == {'graph_wall_us', 'kernel_100ns', 'user_100ns'}
                for value in case['per_worker'].values()))

    def test_source_command_reconstruction_matches_actual_historical_nineteen_command_proof(self):
        from pooled_runtime import BOUNDED_PROFILE, load_profile
        root = self.repo / 'benches/results/nemotron-diarization/linux-x86_64/2026-10-17-graph-cpu-source'
        manifest = json.loads((root / 'collection.json').read_bytes())['files']
        raw = target.capture(root / 'proof/stage/qualification.json', manifest['proof/stage/qualification.json'])
        proof = json.loads(raw)
        commands = proof['commands']
        repo = Path(commands[0]['cwd'])
        stage = Path(commands[0]['command'][-1]).parents[1]
        _, lock = load_profile(BOUNDED_PROFILE)
        expected = target.expected_source_commands(repo, stage, lock)
        self.assertEqual(len(expected), 19)
        self.assertEqual([(row['name'], row['command'], row['cwd']) for row in commands], expected)
        # Execute the actual derived command assertion on the real source receipt.
        source = ast.parse(textwrap.dedent(target.SOURCE_PROOF))
        assertion = next(node for node in source.body if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call) and 'expected_source_commands' in ast.unparse(node))
        scope = {'require': self.pure_functions('require')['require'], 'prep_commands': commands,
            'expected_source_commands': target.expected_source_commands, 'remote_repo': repo,
            'source_stage_path': stage, 'source_lock': lock}
        compiled = compile(ast.Module([assertion], type_ignores=[]), 'actual-nineteen-command-check', 'exec')
        exec(compiled, scope)
        for key, value in [('command', ['git', 'unexpected']), ('cwd', '/other/root')]:
            changed = [dict(row) for row in commands]
            changed[0][key] = value
            scope['prep_commands'] = changed
            with self.subTest(key=key), self.assertRaises(ValueError):
                exec(compiled, scope)

    def test_exact_local_sidecar_inventory_includes_direct_download_and_all_blobs(self):
        files = target.local_sidecars(Path('/fixture'), 463)
        self.assertEqual(len(files), 484)
        self.assertEqual(len(set(files)), 484)
        self.assertIn(Path('/fixture/delivery.zip.process.json'), files)
        self.assertNotIn(Path('/fixture/commands/download.stdout.process.json'), files)
        self.assertIn(Path('/fixture/commands/committed-graph_cpu_transport.py.stdout.process.json'), files)
        self.assertIn(Path('/fixture/blob-stage/462.process.json'), files)

    def test_actual_collection_provenance_assertion_rejects_wrong_native_identity(self):
        body = next(node.body for node in self.tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        assertion = next(node for node in body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and 'independent native collection provenance differs' in ast.unparse(node))
        pre = {'native_source': 'a' * 40, 'native_run': 123, 'native_job': 456, 'native_protocol': target.NATIVE_PROTOCOL}
        route = {'artifact_id': 789, 'archive': {'bytes': 100, 'sha256': 'b' * 64}}
        collection = {'source_commit': pre['native_source'], 'run_id': 123, 'job_id': 456,
            'artifact_id': 789, 'archive': route['archive'], 'protocol': target.NATIVE_PROTOCOL}
        scope = {'require': self.pure_functions('require')['require'], 'pre': pre, 'route': route, 'collection': collection}
        code = compile(ast.Module([assertion], type_ignores=[]), 'actual-native-provenance-check', 'exec')
        exec(code, scope)
        for key, value in [('source_commit', 'c' * 40), ('run_id', 124), ('job_id', 457),
                           ('artifact_id', 790), ('archive', {}), ('protocol', {})]:
            scope['collection'] = {**collection, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                exec(code, scope)


if __name__ == '__main__':
    unittest.main()
