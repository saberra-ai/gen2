from pathlib import Path
import tempfile
import unittest

from artifacts import digest
from macos_multiple_loaded_libraries import audit_multiple_loaded_libraries


class MultipleRuntimeTraceTests(unittest.TestCase):
    def test_requires_both_exact_complete_runtime_closures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            executable = root / 'test'; executable.write_bytes(b'test')
            packages = {}
            images = [executable, Path('/usr/lib/libSystem.B.dylib')]
            for name in ['cpu', 'metal']:
                package = root / name; (package / 'lib').mkdir(parents=True)
                binary = package / 'lib/private.dylib'; binary.write_bytes(name.encode())
                packages[name] = (package, {'libraries': {'private.dylib': {}},
                    'files': {'lib/private.dylib': {'sha256': digest(binary)}}})
                images.append(binary)
            trace = '\n'.join('dyld[123]: <ABCD-1234> ' + str(path) for path in images)
            result = audit_multiple_loaded_libraries(trace, packages, executable)
            self.assertEqual({name: audit['private_library_count'] for name, audit in result.items()}, {'cpu': 1, 'metal': 1})
            with self.assertRaisesRegex(ValueError, 'image set differs'):
                audit_multiple_loaded_libraries(trace.rsplit('\n', 1)[0], packages, executable)
            images[-1].write_bytes(b'wrong bytes')
            with self.assertRaisesRegex(ValueError, 'unexpected non-system'):
                audit_multiple_loaded_libraries(trace, packages, executable)

    def test_unknown_image_and_overlapping_packages_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); (root / 'lib').mkdir()
            executable = root / 'test'; executable.write_bytes(b'test')
            binary = root / 'lib/private.dylib'; binary.write_bytes(b'library')
            unknown = root / 'other.dylib'; unknown.write_bytes(b'unknown')
            manifest = {'libraries': {'private.dylib': {}}, 'files': {'lib/private.dylib': {'sha256': digest(binary)}}}
            trace = '\n'.join('dyld[123]: <ABCD-1234> ' + str(p) for p in [executable, binary, unknown])
            with self.assertRaisesRegex(ValueError, 'unexpected non-system'):
                audit_multiple_loaded_libraries(trace, {'cpu': (root, manifest)}, executable)
            with self.assertRaisesRegex(ValueError, 'packages overlap'):
                audit_multiple_loaded_libraries(trace, {'cpu': (root, manifest), 'metal': (root, manifest)}, executable)


if __name__ == '__main__':
    unittest.main()
