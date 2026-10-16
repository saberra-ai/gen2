import copy
import unittest

from windows_central_state import audit_state, require_setup_success


class CentralStateTests(unittest.TestCase):
    def fixture(self):
        from package_windows import CRT
        expected = {name: {'bytes': 3, 'sha256': 'a' * 64} for name in CRT | {'vcomp140.dll'}}
        state = {'is_64_bit_process': True, 'pending_restart': {'cbs': False, 'windows_update': False, 'file_renames': []},
                 'registry': [{'view': 'Registry64', 'present': True, 'values': {'Installed': 1, 'Version': 'v14.51.36247.00'}},
                              {'view': 'Registry32', 'present': False, 'values': {}}],
                 'files': [{'path': 'C:\\Windows\\System32\\' + name, 'name': name, **row,
                            'file_version': '14.51.36247.0', 'product_version': '14.51.36247.0',
                            'signature': {'status': 'Valid', 'signer': {'subject': 'CN=Publisher, O=Microsoft Corporation, C=US'}}}
                           for name, row in expected.items()]}
        return expected, state

    def test_exact_central_runtime_state_and_real_restart_status_are_required(self):
        expected, state = self.fixture()
        audit_state(state, expected, '14.51.36247.0')
        require_setup_success({'installer_exit_code': 0, 'after': state}, expected, '14.51.36247.0')
        for code in [3010, 1638, 1603]:
            with self.subTest(code=code), self.assertRaises(ValueError):
                require_setup_success({'installer_exit_code': code, 'after': state}, expected, '14.51.36247.0')
        for mutation in [lambda s: s['files'][0].update(path='C:\\inputs\\bin\\' + s['files'][0]['name']),
                         lambda s: s['files'][0].update(sha256='b' * 64),
                         lambda s: s['files'][0]['signature'].update(status='NotTrusted'),
                         lambda s: s['files'].pop(),
                         lambda s: s['pending_restart'].update(cbs=True),
                         lambda s: s['registry'][0]['values'].update(Installed=0),
                         lambda s: s['registry'].pop(),
                         lambda s: s['registry'][1].update(view='Registry64'),
                         lambda s: s['registry'][1].update(view='unknown'),
                         lambda s: s['registry'][0].update(present=False)]:
            invalid = copy.deepcopy(state)
            mutation(invalid)
            with self.assertRaises(ValueError):
                audit_state(invalid, expected, '14.51.36247.0')
