"""Reject wrong source identities, inadequate storage and overlapping output before acquisition."""
from pathlib import Path
import tempfile
import unittest

from graph_cpu_source_admission import START_FLOOR, admit_native, admit_output, merge_inputs


class GraphCPUSourceAdmissionTest(unittest.TestCase):
    def test_exact_native_bytes_required_before_parsing(self):
        raw = (Path(__file__).parent / 'windows-graph-cpu-profile.protocol.json').read_bytes()
        self.assertEqual(admit_native(raw)['expected_graphs'], 84)
        for damaged in (raw + b'\n', raw.replace(b'2048', b'4096', 1), b'not json'):
            with self.subTest(damaged=damaged[:30]), self.assertRaises(ValueError):
                admit_native(damaged)

    def test_original_floor_linux_and_fresh_disjoint_output_are_mandatory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / 'repo'
            repo.mkdir()
            output = root / 'source-proof'
            admit_output(repo, output, START_FLOOR, 'linux')
            for path, free, system in [(output, START_FLOOR - 1, 'linux'),
                                       (output, START_FLOOR, 'darwin'),
                                       (repo / 'inside', START_FLOOR, 'linux'),
                                       (root, START_FLOOR, 'linux')]:
                with self.subTest(path=path, free=free, system=system), self.assertRaises(ValueError):
                    admit_output(repo, path, free, system)
            output.mkdir()
            with self.assertRaises(ValueError):
                admit_output(repo, output, START_FLOOR, 'linux')
            if hasattr(Path, 'symlink_to'):
                link = root / 'linked'
                try:
                    link.symlink_to(root / 'missing')
                except OSError:
                    pass  # Windows unprivileged jobs may not create symlinks.
                else:
                    with self.assertRaises(ValueError):
                        admit_output(repo, link, START_FLOOR, 'linux')

    def test_receiver_bindings_cannot_override_native_inputs(self):
        wanted = {'bytes': 1, 'sha256': 'a' * 64}
        original = {Path('native.py'): wanted}
        merge_inputs(original, {Path('native.py'): wanted, Path('receiver.py'): wanted})
        self.assertEqual(len(original), 2)
        with self.assertRaises(ValueError):
            merge_inputs(original, {Path('native.py'): {**wanted, 'bytes': 2}})
        self.assertEqual(original[Path('native.py')], wanted)


if __name__ == '__main__':
    unittest.main()
