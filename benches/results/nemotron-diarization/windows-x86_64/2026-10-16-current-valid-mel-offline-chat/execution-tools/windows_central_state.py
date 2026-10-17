"""Fail closed on partial VC setup, substituted DLLs and unverified restarts."""
from pathlib import PureWindowsPath
import re

from package_windows import CRT


def require_signature(record):
    if record['status'] != 'Valid' or 'O=Microsoft Corporation,' not in record['signer']['subject']:
        raise ValueError('actual valid Microsoft Windows signature observation required')


def audit_state(state, expected, version):
    if (set(expected) != CRT | {'vcomp140.dll'} or state['is_64_bit_process'] is not True
            or state['pending_restart'] != {'cbs': False, 'windows_update': False, 'file_renames': []}):
        raise ValueError('complete AMD64 central runtime with no pending restart required')
    rows = state['files']
    if len(rows) != len(expected) or {row['name'] for row in rows} != set(expected):
        raise ValueError('complete unique central CRT/OpenMP file observations required')
    for row in rows:
        if (PureWindowsPath(row['path']) != PureWindowsPath('C:/Windows/System32') / row['name']
                or {key: row[key] for key in ['bytes', 'sha256']} != expected[row['name']]
                or row['file_version'] != version or row['product_version'] != version):
            raise ValueError('installed central runtime path/bytes/version differs from pinned payload')
        require_signature(row['signature'])
    registry = state['registry']
    if (len(registry) != 2 or {row.get('view') for row in registry} != {'Registry32', 'Registry64'}
            or any(type(row.get('present')) is not bool or not isinstance(row.get('values'), dict)
                   or (row['present'] is False and row['values']) for row in registry)):
        raise ValueError('exact independent 32-bit and 64-bit registry observations required')

    def same_version(value):
        return bool(re.fullmatch(r'[vV]?\d+\.\d+\.\d+\.\d+', str(value))) and (
            tuple(map(int, str(value).lstrip('vV').split('.'))) == tuple(map(int, version.split('.'))))
    if not any(row['present'] is True and type(row['values'].get('Installed')) is int and row['values']['Installed'] == 1
               and same_version(row['values'].get('Version', ''))
               for row in registry):
        raise ValueError('actual installed x64 VC runtime registration required')
    return {'central_files': rows, 'registry': state['registry'], 'pending_restart': state['pending_restart']}


def require_setup_success(observation, expected, version):
    code = observation['installer_exit_code']
    if type(code) is not int or code != 0:
        if code == 3010:
            raise ValueError('VC setup requested restart (3010); no actual Windows restart has been verified')
        raise ValueError('VC setup did not return the required successful exit 0: ' + str(code))
    return audit_state(observation['after'], expected, version)
