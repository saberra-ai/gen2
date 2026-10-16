import copy
import unittest

from package_windows import CRT, NAMES
from windows_runtime_audit import loaded_modules


class WindowsRuntimeAuditTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
