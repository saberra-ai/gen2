"""Saved dyld traces must prove the real package, never just a reported version."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from macos_loaded_libraries import audit_loaded_libraries


class LoadedLibraryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        (self.root / 'lib').mkdir()
        self.exe = self.root / 'probe'
        self.exe.write_bytes(b'probe')
        self.library = self.root / 'lib/libgen2_nemo_runtime.dylib'
        self.library.write_bytes(b'qualified runtime')
        self.manifest = {'libraries': {self.library.name: {}}, 'files': {
            'lib/' + self.library.name: {'sha256': hashlib.sha256(self.library.read_bytes()).hexdigest()}}}
        self.prefix = f'dyld[42]: <0123-ABCD> {self.exe}\ndyld[42]: <FFFF> /usr/lib/libSystem.B.dylib\n'
        self.trace = self.prefix + f'test lifecycle ... dyld[42]: <ABCD> {self.library}\n'

    def audit(self, trace, require=True):
        return audit_loaded_libraries(trace, self.root, self.manifest, self.exe, require)

    def test_actual_images_and_repeated_loads_are_bound_to_bytes(self):
        self.assertEqual(self.audit(self.trace + self.trace)['private_library_count'], 1)
        self.library.write_bytes(b'mutated runtime')
        with self.assertRaisesRegex(ValueError, 'unexpected non-system'):
            self.audit(self.trace)

    def test_external_runtime_and_absent_or_incomplete_telemetry_fail(self):
        external = self.root / 'outside.dylib'
        external.write_bytes(b'unexpected')
        for trace in ['', self.prefix, self.trace + f'dyld[42]: <ABCD> {external}\n',
                      f'dyld[42]: <ABCD> {self.library}\n']:
            with self.subTest(trace=trace), self.assertRaises(ValueError):
                self.audit(trace)

    def test_model_free_listing_must_load_no_private_runtime(self):
        self.assertEqual(self.audit(self.prefix, False)['private_library_count'], 0)
        with self.assertRaisesRegex(ValueError, 'unexpectedly loaded'):
            self.audit(self.trace, False)


if __name__ == '__main__':
    unittest.main()
