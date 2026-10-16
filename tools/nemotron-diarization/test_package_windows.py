import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from package_windows import BASELINE, CRT, NAMES, validate_libraries, verify


class WindowsPackageTests(unittest.TestCase):
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
            manifest = {'format': 'gen2-nemo-windows-cpu-v1', 'platform_baseline': BASELINE,
                        'architecture': 'AMD64', 'backend': 'cpu',
                        'release_qualified': False, 'redistribution_qualified': False,
                        'files': {'.': {'type': 'directory'}}, 'libraries': {'../../outside.dll': {}}}
            (root / 'package.json').write_text(json.dumps(manifest))
            with patch('package_windows.inspect_pe', side_effect=AssertionError('outside file opened')):
                with self.assertRaises(ValueError):
                    verify(root)


if __name__ == '__main__':
    unittest.main()
