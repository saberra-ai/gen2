from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from windows_compat import WRITE, apply_windows_lf_compat, rewrite_materializer


class WindowsCompatibilityTests(unittest.TestCase):
    def test_rejects_changed_source_before_writing_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'cmake').mkdir()
            path = root / 'cmake/llama_cpp.cmake'
            path.write_bytes(WRITE)
            with self.assertRaises(ValueError):
                apply_windows_lf_compat(root, root)
            self.assertEqual(path.read_bytes(), WRITE)
            self.assertFalse((root / 'windows-compat.json').exists())

    def test_materialization_site_must_be_unique(self):
        for content in [b'no matching site', WRITE + b'\n' + WRITE]:
            with self.assertRaises(ValueError):
                rewrite_materializer(content)

    @unittest.skipUnless(shutil.which('cmake'), 'CMake required for byte-materialization regression')
    def test_cmake_preserves_literal_dollars_and_rejects_template_substitution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / 'test.cmake'
            for payload, succeeds in [(b'+literal ${VARIABLE}; text\n', True),
                                      (b'+literal @UNDEFINED@\n', False),
                                      (b'#cmakedefine UNDEFINED\n', False)]:
                (root / 'input.patch').write_bytes(payload)
                script.write_bytes(b'file(READ "' + (root / 'input.patch').as_posix().encode() + b'" content)\n'
                                   + b'set(normalized "' + (root / 'out.patch').as_posix().encode() + b'")\n'
                                   + rewrite_materializer(WRITE) + b'\n')
                result = subprocess.run(['cmake', '-P', str(script)], capture_output=True)
                if succeeds:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual((root / 'out.patch').read_bytes(), payload)
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(b'materialized patch bytes differ', result.stderr)


if __name__ == '__main__':
    unittest.main()
