"""Receiver derivation, graph-only data replay and closed publication controls."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import graph_cpu_receiver_sources as target


class GraphCPUReceiverSourcesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parents[2]
        cls.originals = target.bound_origins(cls.repo)
        # Fixture identities only: no API calls, artifact consumption or native execution.
        cls.programs, cls.edits = target.derive(cls.originals, 'a' * 40, 123, 456)

    def test_whole_receiver_bytes_are_reversible_outside_declared_edits(self):
        for name, raw in self.programs.items():
            restored = raw.decode()
            for edit in reversed(self.edits[name]):
                # Removed CPU-only fields are checked separately against the origin below.
                if not edit['new']:
                    continue
                self.assertEqual(restored.count(edit['new']), edit['count'], (name, edit['old']))
                restored = restored.replace(edit['new'], edit['old'])
            original_name = 'prepared/' + name.replace('graph-cpu', 'thread-cpu')
            # For removals, independently compare the origin after the same explicitly listed
            # CPU-only fields are removed; every other byte must survive the full derivation.
            expected = self.originals[original_name].decode()
            for edit in self.edits[name]:
                if not edit['new']:
                    expected = expected.replace(edit['old'], '')
            self.assertEqual(restored, expected, name)
            ast.parse(raw)

    def test_identity_origin_and_native_protocol_mutations_reject(self):
        for source, run_id, job_id in [('a' * 39, 123, 456), ('A' * 40, 123, 456),
                ('a' * 40, 0, 456), ('a' * 40, True, 456), ('a' * 40, 123, -1)]:
            with self.subTest(source=source, run=run_id, job=job_id), self.assertRaises(ValueError):
                target.derive(self.originals, source, run_id, job_id)
        changed = dict(self.originals)
        name = next(name for name in changed if name.startswith('prepared/'))
        changed[name] += b'\n# changed\n'
        with self.assertRaises(ValueError):
            target.derive(changed, 'a' * 40, 123, 456)
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            for name, raw in self.originals.items():
                path = repo / target.ORIGIN / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            protocol = repo / 'tools/nemotron-diarization/windows-graph-cpu-profile.protocol.json'
            protocol.parent.mkdir(parents=True)
            protocol.write_text('{}')
            with self.assertRaises(ValueError):
                target.bound_origins(repo)

    def test_relocation_changes_only_string_literal_path_prefixes(self):
        for name, raw in self.programs.items():
            changed = target.relocate(raw, Path('/fixture/repo'), Path('/fixture/cache'))
            expected = raw.decode().replace(target.ORIGINAL_REPO, '/fixture/repo').replace(
                target.ORIGINAL_CACHE, '/fixture/cache')
            self.assertEqual(ast.dump(ast.parse(changed), include_attributes=False),
                ast.dump(ast.parse(expected), include_attributes=False), name)
            self.assertNotIn(target.ORIGINAL_REPO.encode(), changed)
            self.assertNotIn(target.ORIGINAL_CACHE.encode(), changed)

    def test_graph_cpu_parser_and_totals_execute_on_saved_wall_fixture(self):
        from ggml_graph_cpu_profile import summarize
        from test_ggml_graph_cpu_profile import fixture
        log = '\n'.join(fixture())
        profile = summarize(log, True, expected_graphs=1, expected_threads=4, expected_matrices=1)
        # Execute the actual derived aggregation block; no SDK/model or collector top level.
        tree = ast.parse(self.programs['collect-windows-graph-cpu-profile-v1.py'])
        model_loop = next(node for node in tree.body if isinstance(node, ast.For)
            and isinstance(node.target, ast.Tuple) and [n.id for n in node.target.elts] == ['index', 'enabled'])
        enabled_block = next(node for node in model_loop.body if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name) and node.test.id == 'enabled')
        start = next(i for i, node in enumerate(enabled_block.body) if isinstance(node, ast.Assign)
            and any(isinstance(n, ast.Name) and n.id == 'cpu_by_worker' for n in node.targets))
        stop = next(i + 1 for i, node in enumerate(enabled_block.body) if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == 'graph_cpu_totals')
        scope = {'profile': profile, 'index': 1, 'graph_cpu_totals': []}
        exec(compile(ast.Module(enabled_block.body[start:stop], type_ignores=[]), 'derived-graph-totals', 'exec'), scope)
        self.assertEqual(scope['graph_cpu_totals'], [{'index': 1, 'per_worker': {
            str(i): {'kernel_100ns': 0, 'user_100ns': 0} for i in range(4)}}])
        # Missing/error/mixed cycle rows still fail in the exact parser imported by the receiver.
        imported = [node.module for node in tree.body if isinstance(node, ast.ImportFrom)]
        self.assertIn('ggml_graph_cpu_profile', imported)
        self.assertNotIn('ggml_thread_cpu_profile', imported)
        for mutation in [fixture()[:-1], fixture() + ['[DEBUG-thread-cpu-v1] old'],
                         [row.replace('errors=0', 'errors=1') for row in fixture()]]:
            with self.assertRaises(ValueError):
                summarize('\n'.join(mutation), True, expected_graphs=1, expected_threads=4, expected_matrices=1)
        text = self.programs['collect-windows-graph-cpu-profile-v1.py'].decode()
        for field in ['kernel_cycles', 'barrier_cycles', 'first_cycles', 'second_cycles', 'chunk_cycles',
                      'final_wait_cycles', "worker['thread_cpu']", 'cycle_units']:
            self.assertNotIn(field, text)
        self.assertIn("'per_worker_graph_cpu_totals':graph_cpu_totals", text)

    def publication(self, root):
        tree = ast.parse(target.PUBLICATION_CHECK)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
        scope = {'dest': root, 'Path': Path, 'descriptor': lambda path: target.identity(path.read_bytes())}
        exec(compile(ast.Module([function], type_ignores=[]), 'actual-publication-boundary', 'exec'), scope)
        return scope['closed_publication']

    def test_closed_publication_rejects_extra_changed_missing_and_nonregular_members(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'result'
            (root / 'sources').mkdir(parents=True)
            (root / 'sources/one.py').write_bytes(b'inert source')
            expected = {'sources/one.py': target.identity(b'inert source')}
            verify = self.publication(root)
            verify(expected)
            extra = root / 'extra'
            for kind in ['file', 'directory', 'symlink', 'dangling']:
                if kind == 'file':
                    extra.write_bytes(b'extra')
                elif kind == 'directory':
                    extra.mkdir()
                else:
                    extra.symlink_to(root / ('sources/one.py' if kind == 'symlink' else 'missing'))
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    verify(expected)
                extra.rmdir() if kind == 'directory' else extra.unlink()
            (root / 'sources/one.py').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                verify(expected)
            (root / 'sources/one.py').unlink()
            with self.assertRaises(ValueError):
                verify(expected)

    def test_source_compiler_supervision_graph_and_output_gates_remain_in_derived_program(self):
        text = self.programs['collect-windows-graph-cpu-profile-v1.py'].decode()
        for requirement in ["len(replayed) == 3425", 'len(database) == 97',
                "len(build['installed_files']) == 66", "len(control['inputs']) == 7",
                "len(control['cases']) == 12", 'len(control_files)==168',
                "raw == reference_raw and len(raw) == 192000", "actual['push_count'] == 58",
                "case['comparison']", "assert signatures[0] == signatures[1]",
                "timing == q['timing_comparison']", "'enabled_matrix_records_each':84*4*129",
                "'owned_tree_reaped_and_job_closed'", "assert control_build['command']==[",
                "assert model_command['command']==[", 'closed_publication(expected_publication)']:
            self.assertIn(requirement, text)
        delivery = self.programs['admit-windows-graph-cpu-profile-delivery-v1.py'].decode()
        self.assertIn('Actual Windows CPU API control', delivery)
        self.assertIn("log.count('Ran 1 test in') == 1", delivery)

    def test_actual_derived_readers_bind_parsed_bytes_and_reject_mutation_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'admission.json'
            original, changed = b'{"admitted": true}\n', b'{"admitted": false}\n'
            for name, raw in self.programs.items():
                with self.subTest(program=name):
                    tree = ast.parse(raw)
                    reader = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'read')
                    scope = {'parsed_inputs': {}, 'hashlib': hashlib, 'json': json, 'Path': Path,
                        'descriptor': lambda item: target.identity(item.read_bytes())}
                    exec(compile(ast.Module([reader], type_ignores=[]), 'actual-derived-reader', 'exec'), scope)
                    path.write_bytes(original)
                    self.assertEqual(scope['read'](path), {'admitted': True})
                    self.assertEqual(scope['parsed_inputs'], {str(path): target.identity(original)})
                    path.write_bytes(changed)
                    with self.assertRaises(ValueError):
                        scope['read'](path)
                    assertion = next(node for node in tree.body if isinstance(node, ast.Assert)
                        and 'parsed_inputs.items()' in ast.unparse(node))
                    with self.assertRaises(AssertionError):
                        exec(compile(ast.Module([assertion], type_ignores=[]), 'actual-first-binding-check', 'exec'), scope)

    def test_all_admitted_provenance_identities_survive_changes_before_collector_binding(self):
        tree = ast.parse(self.programs['collect-windows-graph-cpu-profile-v1.py'])
        reader = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'read')
        bound = next(node for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'bound' for t in node.targets))
        assertion = next(node for node in tree.body if isinstance(node, ast.Assert)
            and 'descriptor(path) == bound[name]' in ast.unparse(node))
        metadata = ['producer-run.json', 'artifacts.json', 'job.json', 'job.log']
        admission_helper = 'admit-windows-graph-cpu-profile-delivery-v1.py'
        supervision_helper = 'replay-windows-graph-cpu-profile-supervision-v1.py'
        names = metadata + ['delivery-admission.json', 'artifact-members.json',
            'supervision-replay.json', admission_helper, supervision_helper]
        with tempfile.TemporaryDirectory() as directory:
            paths = {name: Path(directory) / name for name in names}
            originals = {name: json.dumps({'fixture': name}).encode() for name in names}
            for name, path in paths.items():
                path.write_bytes(originals[name])
            expected = {name: target.identity(raw) for name, raw in originals.items()}
            scope = {'parsed_inputs': {}, 'hashlib': hashlib, 'json': json, 'Path': Path,
                'descriptor': lambda item: target.identity(item.read_bytes()), 'bound_paths': paths,
                'admission': {'metadata': {name: expected[name] for name in metadata},
                    'collector': expected[admission_helper], 'artifact_members': expected['artifact-members.json']},
                'supervision': {'collector': expected[supervision_helper]}}
            exec(compile(ast.Module([reader], type_ignores=[]), 'actual-collector-reader', 'exec'), scope)
            for name in ['delivery-admission.json', 'supervision-replay.json']:
                scope['read'](paths[name])
            bind_code = compile(ast.Module([bound], type_ignores=[]), 'actual-provenance-binding', 'exec')
            check_code = compile(ast.Module([assertion], type_ignores=[]), 'actual-provenance-final-check', 'exec')
            for name, path in paths.items():
                with self.subTest(changed=name):
                    path.write_bytes(b'changed after admission')
                    exec(bind_code, scope)
                    self.assertEqual(scope['bound'], expected)
                    with self.assertRaises(AssertionError):
                        exec(check_code, scope)
                    path.write_bytes(originals[name])
                    exec(check_code, scope)


if __name__ == '__main__':
    unittest.main()
