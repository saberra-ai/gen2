"""Auxiliary admission cannot hide an unknown or missing runtime image."""
from pathlib import Path
import tempfile
import unittest

from build_valid_mel_candidate import descriptor
from macos_auxiliary_loaded_libraries import audit_with_auxiliary


def line(path):
    return 'dyld[17]: <01234567-89AB-CDEF-0123-456789ABCDEF> ' + str(path)


class AuxiliaryImageTests(unittest.TestCase):
    def test_only_complete_bound_runtime_and_auxiliary_image_set_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); package = root / 'nemo'; (package / 'lib').mkdir(parents=True)
            native = package / 'lib/libgen2_nemo_test.dylib'; native.write_bytes(b'private fixture')
            executable = root / 'helper'; executable.write_bytes(b'executable fixture')
            auxiliary = root / 'provider.dylib'; auxiliary.write_bytes(b'auxiliary fixture')
            manifest = {'libraries': {native.name: {}}, 'files': {'lib/' + native.name: descriptor(native)}}
            bound = {str(auxiliary): descriptor(auxiliary)}
            trace = '\n'.join(map(line, [executable, native, auxiliary, '/usr/lib/dyld-shared-cache-fixture.dylib']))
            result = audit_with_auxiliary(trace, package, manifest, executable, bound)
            self.assertEqual(result['runtime']['private_library_count'], 1)
            self.assertEqual(result['auxiliary_library_count'], 1)
            for missing in [native, auxiliary]:
                with self.subTest(missing=missing), self.assertRaises(ValueError):
                    audit_with_auxiliary(trace.replace(line(missing), ''), package, manifest, executable, bound)
            auxiliary.write_bytes(b'changed auxiliary')
            with self.assertRaisesRegex(ValueError, 'auxiliary loaded image differs'):
                audit_with_auxiliary(trace, package, manifest, executable, bound)

    def test_unrecognized_external_image_cannot_be_filtered_as_auxiliary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); package = root / 'nemo'; (package / 'lib').mkdir(parents=True)
            native = package / 'lib/libgen2_nemo_test.dylib'; native.write_bytes(b'private fixture')
            executable = root / 'helper'; executable.write_bytes(b'executable fixture')
            auxiliary = root / 'provider.dylib'; auxiliary.write_bytes(b'auxiliary fixture')
            unknown = root / 'unbound/provider.dylib'; unknown.parent.mkdir(); unknown.write_bytes(auxiliary.read_bytes())
            manifest = {'libraries': {native.name: {}}, 'files': {'lib/' + native.name: descriptor(native)}}
            trace = '\n'.join(map(line, [executable, native, auxiliary, unknown]))
            with self.assertRaisesRegex(ValueError, 'unexpected non-system dyld image'):
                audit_with_auxiliary(trace, package, manifest, executable, {str(auxiliary): descriptor(auxiliary)})


if __name__ == '__main__':
    unittest.main()
