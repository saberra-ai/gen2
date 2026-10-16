import hashlib
import json
from pathlib import Path
import tempfile
import struct
import unittest
import zipfile
from unittest.mock import patch

from windows_vcpkg_metadata import paragraphs, recover_package


class VcpkgMetadataTests(unittest.TestCase):
    def test_control_parser_preserves_continuations_and_rejects_duplicate_fields(self):
        self.assertEqual(paragraphs('Package: sample\nDescription: first\n second\n\n'),
                         [{'Package': 'sample', 'Description': 'first\nsecond'}])
        with self.assertRaises(ValueError):
            paragraphs('Package: a\nPackage: b\n')

    def test_archive_filename_does_not_substitute_for_actual_abi_info_or_notice_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            abi_info = b'features core\nportfile declared-port-hash\ntriplet x64-windows-static-md\n'
            abi = hashlib.sha256(abi_info).hexdigest()
            identity = {'Package': 'sample', 'Version': '1.0', 'Architecture': 'x64-windows-static-md', 'Abi': abi}
            archive = root / (abi + '.zip')
            def write(info, control_extra='', spdx=None):
                with zipfile.ZipFile(archive, 'w') as stream:
                    stream.writestr('CONTROL', '\n'.join(key + ': ' + value for key, value in identity.items()) + control_extra)
                    stream.writestr('BUILD_INFO', 'CRTLinkage: dynamic\nLibraryLinkage: static\n')
                    stream.writestr('share/sample/vcpkg_abi_info.txt', info)
                    stream.writestr('share/sample/vcpkg.spdx.json', json.dumps(spdx or {
                        'SPDXID': 'SPDXRef-DOCUMENT', 'spdxVersion': 'SPDX-2.3',
                        'name': 'sample:x64-windows-static-md@1.0 ' + abi,
                        'creationInfo': {'creators': ['Tool: vcpkg-2026-07-27-98d7cb0cf1f4686a3e43aa5672b6230c1d56bce8']},
                        'packages': [{'SPDXID': 'SPDXRef-port', 'name': 'sample', 'versionInfo': '1.0'},
                                     {'SPDXID': 'SPDXRef-binary', 'name': 'sample:x64-windows-static-md', 'versionInfo': abi}]}))
                    stream.writestr('share/sample/copyright', b'actual notice')
            write(abi_info)
            result = recover_package(archive, identity, root / 'valid', b'actual notice')
            self.assertEqual(result['abi_info_sha256'], abi)
            self.assertTrue(result['target_notice_matches_sdk'])
            write(b'different build identity')
            with self.assertRaises(ValueError):
                recover_package(archive, identity, root / 'wrong-abi', b'actual notice')
            write(abi_info)
            with self.assertRaises(ValueError):
                recover_package(archive, identity, root / 'wrong-notice', b'other notice')
            for label, control_extra, spdx in [('dependency', '\nDepends: unrelated-wrong-package', None),
                                               ('spdx', '', {'name': 'totally-unrelated', 'packages': []})]:
                write(abi_info, control_extra, spdx)
                with self.subTest(label=label), self.assertRaises(ValueError):
                    recover_package(archive, identity, root / label, b'actual notice')

    def test_zip64_cannot_override_small_classic_bounds_before_zipfile_allocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / 'forged.zip'
            with zipfile.ZipFile(archive, 'w') as stream:
                for index in range(20001):
                    stream.writestr(f'entry-{index}', b'')
            data = archive.read_bytes()
            end = struct.unpack('<4s4H2IH', data[-22:])
            prefix = data[:-22]
            record = struct.pack('<4sQ2H2I4Q', b'PK\x06\x06', 44, 45, 45, 0, 0, end[4], end[4], end[5], end[6])
            locator = struct.pack('<4sIQI', b'PK\x06\x07', 0, len(prefix), 1)
            forged = struct.pack('<4s4H2IH', b'PK\x05\x06', 0, 0, 1, 1, 76, len(prefix), 0)
            archive.write_bytes(prefix + record + locator + forged)
            with patch('windows_vcpkg_metadata.zipfile.ZipFile', side_effect=AssertionError('unbounded constructor reached')) as constructor:
                with self.assertRaises(ValueError):
                    recover_package(archive, {}, root / 'output')
                constructor.assert_not_called()

    def test_self_consistent_abi_hash_cannot_replace_the_qualified_dependency_graph(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dependency = {'Package': 'dependency', 'Architecture': 'x64-windows-static-md', 'Abi': 'd' * 64}
            for label, dependency_abi in [('valid', 'd' * 64), ('wrong', 'e' * 64)]:
                info = ('dependency ' + dependency_abi + '\nfeatures core\ntriplet x64-windows-static-md\n').encode()
                abi = hashlib.sha256(info).hexdigest()
                identity = {'Package': 'sample', 'Version': '1.0', 'Architecture': 'x64-windows-static-md',
                            'Abi': abi, 'Depends': 'dependency'}
                archive = root / (label + '.zip')
                with zipfile.ZipFile(archive, 'w') as stream:
                    stream.writestr('CONTROL', '\n'.join(key + ': ' + value for key, value in identity.items()))
                    stream.writestr('BUILD_INFO', 'CRTLinkage: dynamic\nLibraryLinkage: static\n')
                    stream.writestr('share/sample/vcpkg_abi_info.txt', info)
                    stream.writestr('share/sample/copyright', b'notice')
                    stream.writestr('share/sample/vcpkg.spdx.json', json.dumps({
                        'SPDXID': 'SPDXRef-DOCUMENT', 'spdxVersion': 'SPDX-2.3',
                        'name': 'sample:x64-windows-static-md@1.0 ' + abi,
                        'creationInfo': {'creators': ['Tool: vcpkg-2026-07-27-98d7cb0cf1f4686a3e43aa5672b6230c1d56bce8']},
                        'packages': [{'SPDXID': 'SPDXRef-port', 'name': 'sample', 'versionInfo': '1.0'},
                                     {'SPDXID': 'SPDXRef-binary', 'name': 'sample:x64-windows-static-md', 'versionInfo': abi}]}))
                if label == 'valid':
                    self.assertTrue(recover_package(archive, identity, root / label, identities=[identity, dependency])['abi_dependency_graph_verified'])
                else:
                    with self.assertRaisesRegex(ValueError, 'dependencies differ'):
                        recover_package(archive, identity, root / label, identities=[identity, dependency])


if __name__ == '__main__':
    unittest.main()
