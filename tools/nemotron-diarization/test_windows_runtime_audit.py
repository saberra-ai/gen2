import copy
import unittest

from package_windows import CRT, NAMES
from windows_runtime_audit import loaded_modules


class WindowsRuntimeAuditTests(unittest.TestCase):
    def test_central_crt_and_openmp_must_match_the_verified_system32_state(self):
        libraries, observed = self.fixture()
        central = {name: {'bytes': 3, 'sha256': 'a' * 64} for name in CRT | {'vcomp140.dll'}}
        for row in observed['modules']:
            if row['path'].rsplit('\\', 1)[-1] in CRT:
                row['path'] = 'C:\\Windows\\System32\\' + row['path'].rsplit('\\', 1)[-1]
        observed['modules'].append({'path': 'C:\\Windows\\System32\\vcomp140.dll', **central['vcomp140.dll']})
        private = {name: record for name, record in libraries.items() if name in NAMES.values()}
        helper = {'bytes': 7, 'sha256': 'b' * 64}
        result = loaded_modules(observed, private, 'public.exe', helper, central_crt=central, openmp=central['vcomp140.dll'])
        self.assertEqual(len(result['crt']), 8)
        for name in ['vcruntime140_threads.dll', 'vcamp140.dll', 'vccorlib140.dll', 'mfc140u.dll']:
            invalid = copy.deepcopy(observed)
            invalid['modules'].append({'path': 'C:\\Windows\\System32\\' + name, 'bytes': 99, 'sha256': 'f' * 64})
            with self.subTest(name=name), self.assertRaises(ValueError):
                loaded_modules(invalid, private, 'public.exe', helper, central_crt=central, openmp=central['vcomp140.dll'])
        row = next(row for row in observed['modules'] if row['path'].endswith('vcruntime140.dll'))
        row['path'] = 'C:\\inputs\\bin\\vcruntime140.dll'
        with self.assertRaises(ValueError):
            loaded_modules(observed, private, 'public.exe', helper, central_crt=central, openmp=central['vcomp140.dll'])

    def test_extended_dos_module_paths_admit_only_the_same_c_drive_package_locations(self):
        libraries, observed = self.fixture()
        for row in observed['modules']:
            row['path'] = '\\\\?\\' + row['path']
        helper = {'bytes': 7, 'sha256': 'b' * 64}
        self.assertEqual(len(loaded_modules(observed, libraries, 'public.exe', helper)['private_native']), 5)
        for path in ['\\\\?\\GLOBALROOT\\Device\\HarddiskVolume1\\inputs\\bin\\public.exe',
                     '\\\\?\\UNC\\server\\share\\public.exe', '\\\\?\\C:\\inputs\\bin\\..\\public.exe',
                     '\\\\?\\D:\\inputs\\bin\\public.exe']:
            invalid = copy.deepcopy(observed)
            invalid['modules'][-2]['path'] = path
            with self.subTest(path=path), self.assertRaises(ValueError):
                loaded_modules(invalid, libraries, 'public.exe', helper)

    def fixture(self):
        libraries = {name: {'bytes': 3, 'sha256': 'a' * 64} for name in set(NAMES.values()) | CRT}
        modules = [{'path': 'C:\\inputs\\relocated runtime ü\\lib\\' + name, **record}
                   for name, record in libraries.items()]
        modules += [{'path': 'C:\\inputs\\bin\\public.exe', 'bytes': 7, 'sha256': 'b' * 64},
                    {'path': 'C:\\Windows\\System32\\kernel32.dll', 'bytes': 10, 'sha256': 'c' * 64}]
        return libraries, {'stable_snapshots': 2, 'modules': modules}

    def test_requires_all_private_dlls_at_relocated_package_and_exact_shared_crt_bytes(self):
        libraries, observed = self.fixture()
        result = loaded_modules(observed, libraries, 'public.exe', {'bytes': 7, 'sha256': 'b' * 64})
        self.assertEqual(len(result['private_native']), 5)
        self.assertEqual(len(result['crt']), 8)
        for name, mutation in [('wrong private directory', lambda rows: rows[0].update(path='C:\\outside\\' + rows[0]['path'].rsplit('\\', 1)[1])),
                               ('changed bytes', lambda rows: rows[0].update(sha256='d' * 64)),
                               ('generic DLL', lambda rows: rows.append({'path': 'C:\\Windows\\System32\\ggml.dll', 'bytes': 3, 'sha256': 'a' * 64})),
                               ('missing private', lambda rows: rows.__setitem__(slice(None), [row for row in rows if not row['path'].endswith(NAMES['ggml.dll'])]))]:
            invalid = copy.deepcopy(observed)
            mutation(invalid['modules'])
            with self.subTest(name=name), self.assertRaises(ValueError):
                loaded_modules(invalid, libraries, 'public.exe', {'bytes': 7, 'sha256': 'b' * 64})
        invalid = copy.deepcopy(observed)
        invalid['stable_snapshots'] = 0
        with self.assertRaises(ValueError):
            loaded_modules(invalid, libraries, 'public.exe', {'bytes': 7, 'sha256': 'b' * 64})

    def test_app_local_crt_is_allowed_only_with_qualified_bytes_and_no_ambiguous_aliases(self):
        libraries, observed = self.fixture()
        row = next(row for row in observed['modules'] if row['path'].endswith('vcruntime140.dll'))
        row['path'] = 'C:\\inputs\\bin\\vcruntime140.dll'
        loaded_modules(observed, libraries, 'public.exe', {'bytes': 7, 'sha256': 'b' * 64})
        row['path'] = 'C:\\Windows\\System32\\vcruntime140.dll'
        with self.assertRaises(ValueError):
            loaded_modules(observed, libraries, 'public.exe', {'bytes': 7, 'sha256': 'b' * 64})
        row['path'] = 'C:\\inputs\\bin\\vcruntime140.dll'
        observed['modules'].append({**row, 'path': row['path'].upper()})
        with self.assertRaises(ValueError):
            loaded_modules(observed, libraries, 'public.exe', {'bytes': 7, 'sha256': 'b' * 64})

    def test_isolated_chat_forbids_private_dlls_and_requires_exact_declared_openmp(self):
        libraries, observed = self.fixture()
        observed['modules'] = [row for row in observed['modules']
                               if row['path'].rsplit('\\', 1)[-1] not in NAMES.values()]
        openmp = {'bytes': 8, 'sha256': 'd' * 64}
        row = {'path': 'C:\\inputs\\bin\\vcomp140.dll', **openmp}
        observed['modules'].append(row)
        helper = {'bytes': 7, 'sha256': 'b' * 64}
        result = loaded_modules(observed, libraries, 'public.exe', helper, private_required=False, openmp=openmp)
        self.assertEqual(result['private_native'], [])
        self.assertEqual(result['chat_openmp'], [row])
        for mutation in [lambda rows: rows.pop(),
                         lambda rows: rows[-1].update(path='C:\\Windows\\System32\\vcomp140.dll'),
                         lambda rows: rows[-1].update(sha256='e' * 64),
                         lambda rows: rows.append({'path': 'C:\\inputs\\relocated runtime ü\\lib\\' + NAMES['ggml.dll'],
                                                  'bytes': 3, 'sha256': 'a' * 64})]:
            invalid = copy.deepcopy(observed)
            mutation(invalid['modules'])
            with self.assertRaises(ValueError):
                loaded_modules(invalid, libraries, 'public.exe', helper, private_required=False, openmp=openmp)


if __name__ == '__main__':
    unittest.main()
