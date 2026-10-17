import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from package_bounded_windows import BASELINE, CRT, NAMES, validate_libraries, verify
from windows_archive import byte_inventory
from bounded_platform_sdk_receipt import load_profile
from pooled_runtime import profile_provenance
from windows_notices import install_supplement


class WindowsPackageTests(unittest.TestCase):
    def test_historical_package_formats_cannot_pass_current_admission(self):
        for form in ['gen2-nemo-windows-cpu-v1', 'gen2-nemo-windows-cpu-central-v1',
                     'gen2-nemo-windows-cpu-v2', 'gen2-nemo-windows-cpu-central-v2']:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / 'package.json').write_text(json.dumps({'format': form}))
                with self.assertRaisesRegex(ValueError, 'unsupported Windows'):
                    verify(root)

    def test_bounded_v3_app_local_package_requires_supplemental_source_notices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {'format': 'gen2-nemo-windows-bounded-cpu-v3', 'platform_baseline': BASELINE,
                        'architecture': 'AMD64', 'backend': 'cpu', 'release_qualified': False,
                        'redistribution_qualified': False, 'files': byte_inventory(root)}
            manifest.update(profile_provenance(*load_profile()))
            (root / 'package.json').write_text(json.dumps(manifest))
            with patch('package_bounded_windows.inspect_pe', side_effect=AssertionError('unadmitted PE opened')):
                with self.assertRaisesRegex(ValueError, 'requires supplemental source notices'):
                    verify(root)

    def test_central_variant_ships_only_private_dlls_and_declares_external_crt_imports(self):
        private = {name: row for name, row in self.libraries().items() if name in NAMES.values()}
        entry = NAMES['nemo_speech_asr_c.dll']
        private[entry]['imports'] += ['msvcp140.dll', NAMES['ggml.dll']]
        closure = validate_libraries(private, central=True)
        self.assertEqual(closure['external_vc_imports'], ['msvcp140.dll'])
        self.assertEqual(closure['static_entry_closure'], sorted([entry, NAMES['ggml.dll']]))
        for invalid in [self.libraries(), {**private, 'unqualified.dll': private[entry]}]:
            with self.assertRaises(ValueError):
                validate_libraries(invalid, central=True)
        private[entry]['delay_imports'] = ['another-runtime.dll']
        with self.assertRaises(ValueError):
            validate_libraries(private, central=True)

    def test_central_manifest_rejects_added_crt_or_installer_before_opening_pe_images(self):
        for extra in ['vcruntime140.dll', 'VC_redist.x64.exe', 'package.msi', 'cab1.cab']:
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                supplement = install_supplement(root)
                (root / 'lib').mkdir()
                for name in NAMES.values():
                    (root / 'lib' / name).write_bytes(b'PE fixture must not be opened')
                (root / extra).write_bytes(b'undeclared publisher payload')
                requirement = json.loads(Path(__file__).with_name('windows-vc-runtime.lock.json').read_text())
                manifest = {'format': 'gen2-nemo-windows-bounded-cpu-central-v3', 'platform_baseline': BASELINE,
                            'architecture': 'AMD64', 'backend': 'cpu', 'release_qualified': False,
                            'redistribution_qualified': False, 'microsoft_payload_included': False,
                            'source_notice_supplement': supplement, 'required_vc_runtime': requirement,
                            'libraries': {name: {} for name in NAMES.values()}, 'files': byte_inventory(root)}
                manifest.update(profile_provenance(*load_profile()))
                (root / 'package.json').write_text(json.dumps(manifest))
                with patch('package_bounded_windows.inspect_pe', side_effect=AssertionError('unadmitted PE opened')):
                    with self.assertRaisesRegex(ValueError, 'central Windows contract'):
                        verify(root)

    def libraries(self):
        return {name: {'machine': 'AMD64', 'format': 'PE32+', 'is_dll': True, 'export_name': name,
                       'bound_imports_present': False, 'imports': ['kernel32.dll'],
                       'delay_imports': [], 'forwarder_dependencies': []}
                for name in set(NAMES.values()) | CRT}

    def test_closure_rejects_generic_missing_unknown_delay_forwarded_or_bound_dependencies(self):
        valid = self.libraries()
        entry = NAMES['nemo_speech_asr_c.dll']
        valid[entry]['imports'].append(NAMES['nemo_speech_asr.dll'])
        self.assertEqual(validate_libraries(valid)['static_entry_closure'], sorted([entry, NAMES['nemo_speech_asr.dll']]))
        for field, value in [('imports', 'ggml.dll'), ('delay_imports', 'unqualified.dll'),
                             ('forwarder_dependencies', 'api-ms-win-crt-invented-l1-1-0.dll')]:
            invalid = copy.deepcopy(valid)
            invalid[entry][field].append(value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_libraries(invalid)
        for mutation in ['missing', 'bound', 'identity']:
            invalid = copy.deepcopy(valid)
            if mutation == 'missing':
                invalid.pop('msvcp140_1.dll')
            elif mutation == 'bound':
                invalid[entry]['bound_imports_present'] = True
            else:
                invalid[entry]['export_name'] = 'generic.dll'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_libraries(invalid)

    def test_manifest_library_paths_are_rejected_before_opening_unlisted_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {'format': 'gen2-nemo-windows-bounded-cpu-v3', 'platform_baseline': BASELINE,
                        'architecture': 'AMD64', 'backend': 'cpu',
                        'release_qualified': False, 'redistribution_qualified': False,
                        'files': {'.': {'type': 'directory'}}, 'libraries': {'../../outside.dll': {}}}
            manifest.update(profile_provenance(*load_profile()))
            (root / 'package.json').write_text(json.dumps(manifest))
            with patch('package_bounded_windows.inspect_pe', side_effect=AssertionError('outside file opened')):
                with self.assertRaises(ValueError):
                    verify(root)


if __name__ == '__main__':
    unittest.main()
