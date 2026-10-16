from pathlib import Path
import hashlib
import tempfile
import unittest
from unittest.mock import patch

from package_linux import apt_provenance, check_source_bundle, discover, inspect_elf, parse_ldd, qualified_entry, validate_library


class LinuxPackageTests(unittest.TestCase):
    def test_only_pinned_real_sdk_files_can_back_entry_alias(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            sdk = root / 'sdk'
            (sdk / 'lib').mkdir(parents=True)
            native = sdk / 'lib/libnemo_speech_asr_c.so.1'
            native.write_bytes(b'pinned native')
            record = {'sha256': hashlib.sha256(native.read_bytes()).hexdigest(), 'bytes': native.stat().st_size}
            build = {'installed_files': {'lib/' + native.name: record}}
            alias = sdk / 'lib/libnemo_speech_asr_c.so'
            alias.symlink_to(native.name)
            self.assertEqual(qualified_entry(sdk, build), native)
            native.write_bytes(b'changed native')
            with self.assertRaisesRegex(ValueError, 'receipt'):
                qualified_entry(sdk, build)
            native.write_bytes(b'pinned native')
            outside = root / native.name
            outside.write_bytes(native.read_bytes())
            alias.unlink()
            alias.symlink_to(outside)
            with self.assertRaises(ValueError):
                qualified_entry(sdk, build)

    def test_another_qualified_library_cannot_replace_the_c_abi_alias(self):
        with tempfile.TemporaryDirectory() as temporary:
            sdk = Path(temporary).resolve()
            (sdk / 'lib').mkdir()
            records = {}
            for name in ['libnemo_speech_asr_c.so.1', 'libggml-base.so.0.25.1']:
                path = sdk / 'lib' / name
                path.write_bytes(b'identical qualified bytes')
                records['lib/' + name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
            (sdk / 'lib/libnemo_speech_asr_c.so').symlink_to('libggml-base.so.0.25.1')
            with self.assertRaisesRegex(ValueError, 'C ABI'):
                qualified_entry(sdk, {'installed_files': records})

    def test_a_qualified_dependency_with_the_wrong_soname_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            sdk = Path(temporary).resolve()
            (sdk / 'lib').mkdir()
            entry = sdk / 'lib/libnemo_speech_asr_c.so.1'
            wrong = sdk / 'lib/libggml-base.so.0.25.1'
            records = {}
            for path in [entry, wrong]:
                path.write_bytes(b'qualified bytes')
                records['lib/' + path.name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
            metadata = {entry: {'soname': entry.name, 'needed': ['libggml.so.0']},
                        wrong: {'soname': 'libggml-base.so.0', 'needed': []}}
            with patch('package_linux.inspect_elf', side_effect=lambda p: metadata[p].copy()), patch(
                    'package_linux.resolved_dependencies', side_effect=lambda p, _: {'libggml.so.0': wrong} if p == entry else {}):
                with self.assertRaisesRegex(ValueError, 'identity'):
                    discover(entry, sdk, {'installed_files': records})

    def test_insecure_apt_overrides_cannot_back_source_authentication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = root / 'sources.list.d'
            parts.mkdir()
            source = parts / 'ubuntu.sources'
            source.write_text('Types: deb deb-src\nURIs: https://archive.ubuntu.com/ubuntu\nSigned-By: /keys/ubuntu.gpg\n')
            config = 'Dir "' + str(root) + '";\nDir::Etc ".";\n'
            with patch.dict('os.environ', {}, clear=True), patch('package_linux.capture', return_value=config):
                self.assertIn(str(source), apt_provenance()['source_files'])
                for setting in ['Trusted: yes', 'Trusted: on', 'Trusted: enable', 'Trusted: 01',
                                'Allow-Insecure: true', 'Allow-Weak: 1']:
                    source.write_text(setting + '\n')
                    with self.assertRaisesRegex(ValueError, 'authentication override'):
                        apt_provenance()
                source.unlink()
                legacy = parts / 'ubuntu.list'
                for value in ['on', 'with', 'enable', '01', '0x1', 'unexpected']:
                    legacy.write_text('deb [trusted=' + value + '] https://archive.ubuntu.com/ubuntu noble main\n')
                    with self.assertRaisesRegex(ValueError, 'authentication override'):
                        apt_provenance()
                legacy.unlink()
                source.write_text('Types: deb-src\n')
            with patch.dict('os.environ', {}, clear=True), patch('package_linux.capture', return_value=config + 'Acquire::AllowInsecureRepositories "true";\n'):
                with self.assertRaisesRegex(ValueError, 'authentication override'):
                    apt_provenance()

    def test_folded_signed_by_fields_bind_every_external_keyring(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = root / 'sources.list.d'
            parts.mkdir()
            keys = [root / 'one.gpg', root / 'two.gpg']
            for key in keys:
                key.write_bytes(key.name.encode())
            source = parts / 'ubuntu.sources'
            config = 'Dir "' + str(root) + '";\nDir::Etc ".";\n'
            expected = {str(key): hashlib.sha256(key.read_bytes()).hexdigest() for key in keys}
            with patch.dict('os.environ', {}, clear=True), patch('package_linux.capture', return_value=config):
                for value in ['\n ' + str(keys[0]) + '\n ' + str(keys[1]), str(keys[0]) + '\n ' + str(keys[1])]:
                    source.write_text('Types: deb-src\nSigned-By: ' + value + '\n')
                    self.assertEqual(apt_provenance()['keyrings'], expected)
                source.unlink()
                (parts / 'ubuntu.list').write_text('deb-src [signed-by=' + ','.join(map(str, keys)) + '] https://archive.ubuntu.com/ubuntu noble main\n')
                self.assertEqual(apt_provenance()['keyrings'], expected)

    def test_source_identity_and_each_downloaded_byte_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = b'exact corresponding source'
            source = root / 'source.orig.tar.xz'
            source.write_bytes(data)
            descriptor = root / 'source.dsc'
            descriptor.write_text('Format: 3.0 (quilt)\nSource: component\nVersion: 1:2.3-4\nChecksums-Sha256:\n ' + hashlib.sha256(data).hexdigest() + ' ' + str(len(data)) + ' ' + source.name + '\n')
            result = check_source_bundle(root, 'component', '1:2.3-4')
            self.assertEqual(result['source_version'], '1:2.3-4')
            with self.assertRaisesRegex(ValueError, 'identity'):
                check_source_bundle(root, 'component', '2.3-4')
            source.write_bytes(b'different bytes')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                check_source_bundle(root, 'component', '1:2.3-4')
            source.write_bytes(data)
            (root / 'unexpected').write_text('extra')
            with self.assertRaisesRegex(ValueError, 'unexpected'):
                check_source_bundle(root, 'component', '1:2.3-4')

    def test_ldd_handles_direct_loader_dependency_and_paths_with_spaces(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            native = root / 'library with spaces.so'
            loader = root / 'ld-linux-x86-64.so.2'
            native.touch()
            loader.touch()
            parsed = parse_ldd(f'  libnative.so => {native} (0x0012)\n  {loader} (0xabcd)\n  linux-vdso.so.1 (0x1234)\n')
            self.assertEqual(parsed, {'libnative.so': native, loader.name: loader})

    def test_only_relative_closed_elf_dependencies_pass(self):
        names = {'libgen2_nemo_linux_native.so', 'libstdc++.so.6'}
        info = {'soname': 'libgen2_nemo_linux_native.so', 'rpath': ['$ORIGIN'], 'runpath': [], 'needed': ['libstdc++.so.6', 'libc.so.6']}
        validate_library(info, names)
        for change in [{'rpath': ['/source/sdk']}, {'runpath': ['$ORIGIN']}, {'needed': ['libggml.so']}, {'soname': 'libggml.so'}]:
            with self.assertRaises(ValueError):
                validate_library({**info, **change}, names)

    def test_elf_architecture_and_unknown_or_new_glibc_requirements_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'native.so'
            header = bytearray(64)
            header[:6] = b'\x7fELF\x02\x01'
            header[18:20] = (62).to_bytes(2, 'little')
            path.write_bytes(header)
            dynamic = '(SONAME) Library soname: [libnative.so]\n(RPATH) Library rpath: [$ORIGIN]\n'
            for version, good in [('GLIBC_2.39', True), ('GLIBC_2.39.0', True), ('GLIBC_ABI_DT_RELR', True), ('GLIBC_2.39.0.1', False), ('GLIBC_2.40', False), ('GLIBC_PRIVATE', False)]:
                with patch('package_linux.capture', side_effect=[dynamic, 'Version needs section\n Name: ' + version]):
                    if good:
                        self.assertIn(version, inspect_elf(path)['required_symbol_versions'])
                    else:
                        with self.assertRaisesRegex(ValueError, 'glibc'):
                            inspect_elf(path)
            header[18:20] = (183).to_bytes(2, 'little')
            path.write_bytes(header)
            with self.assertRaisesRegex(ValueError, 'x86_64'):
                inspect_elf(path)


if __name__ == '__main__':
    unittest.main()
