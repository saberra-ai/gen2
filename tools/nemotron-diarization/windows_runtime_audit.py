"""Audit observed Windows helper modules; platform modules remain OS dependencies."""
from pathlib import PureWindowsPath
import re

from package_windows import CRT, NAMES

RUNTIME = PureWindowsPath('C:/inputs/relocated runtime ü/lib')
BIN = PureWindowsPath('C:/inputs/bin')
SYSTEM = PureWindowsPath('C:/Windows/System32')


def loaded_modules(observation, libraries, helper_name, helper, *, private_required=True, openmp=None):
    rows = observation['modules']
    if (type(observation['stable_snapshots']) is not int or observation['stable_snapshots'] < 1
            or not isinstance(rows, list) or not 1 <= len(rows) <= 512):
        raise ValueError('actual stable bounded Windows module observations required')
    seen, private, crt, platform, extra, executable = set(), [], [], [], [], False
    for row in rows:
        path = PureWindowsPath(row['path'])
        name = path.name.lower()
        if (not path.is_absolute() or path.drive.lower() != 'c:' or '..' in path.parts
                or name in seen or type(row['bytes']) is not int or row['bytes'] < 1
                or not re.fullmatch('[0-9a-f]{64}', row['sha256'])):
            raise ValueError('unsafe, duplicate or incomplete observed Windows module')
        seen.add(name)
        record = {'bytes': row['bytes'], 'sha256': row['sha256']}
        if name in libraries:
            if record != {field: libraries[name][field] for field in record}:
                raise ValueError('actual Windows native/CRT bytes differ from qualified payload')
            if name in CRT and path.parent in {BIN, RUNTIME}:
                crt.append(row)
            elif name in NAMES.values() and path.parent == RUNTIME:
                private.append(row)
            else:
                raise ValueError('actual Windows native/CRT module escaped packaged directories')
        elif name == 'vcomp140.dll':
            if openmp is None or path.parent != BIN or record != openmp:
                raise ValueError('observed chat OpenMP runtime differs from declared test payload')
            extra.append(row)
        elif name == helper_name and path.parent == BIN and record == helper:
            executable = True
        elif (path.parent == SYSTEM and path.suffix.lower() == '.dll'
              and not name.startswith(('ggml', 'nemo_', 'gen2_nemo_'))):
            platform.append(row)
        else:
            raise ValueError('unexpected Windows module outside declared helper/runtime/OS: ' + str(path))
    expected = set(NAMES.values()) if private_required else set()
    if {PureWindowsPath(row['path']).name.lower() for row in private} != expected or not executable:
        raise ValueError('actual helper/private Windows DLL observations differ from required load order')
    if openmp is not None and len(extra) != 1:
        raise ValueError('declared chat OpenMP runtime was not observed')
    result = {'private_native': private, 'crt': crt, 'platform': platform,
            'stable_snapshots': observation['stable_snapshots'],
            'scope': 'Observed modules during the recorded helper execution; no universal dynamic-load/namespace claim.'}
    if openmp is not None:
        result['chat_openmp'] = extra
    return result
