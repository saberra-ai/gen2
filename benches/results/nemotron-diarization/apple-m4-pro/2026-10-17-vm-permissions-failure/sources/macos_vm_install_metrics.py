"""Require callback, stopped state and normal owner exit before seed admission."""
import hashlib
import math
from pathlib import Path
import stat


def owner_metrics(value, command, events):
    expected = {'schema_version': 1, 'installation_qualified': True, 'virtual_machine_created': True,
        'installation_started': True, 'installer_callback_received': True, 'installer_callback_success': True,
        'cancellation_requested': False, 'cleanup_confirmed': True, 'final_state': 0,
        'normal_start_called': False, 'custom_listener_created': False, 'display_window_created': False,
        'logging_failed': False, 'errors': [], 'configuration_validated': True,
        'cpu_count': 4, 'memory_bytes': 8 * 1024**3}
    for key, want in expected.items():
        if type(value.get(key)) is not type(want) or value[key] != want:
            raise ValueError('installer terminal evidence differs: ' + key)
    if command.get('returncode') != 0 or any(key in command for key in [
            'abort_error', 'forced_process_group_termination', 'cancellation_signal', 'startup_error']):
        raise ValueError('successful observed owner exit without abort required')
    previous = -1
    for event in events:
        elapsed = event.get('elapsed_seconds')
        if type(elapsed) not in {int, float} or not math.isfinite(elapsed) or elapsed < previous:
            raise ValueError('ordered finite installer event times required')
        previous = elapsed
    names = [event['event'] for event in events]
    required = ['configuration_validated', 'virtual_machine_created', 'installation_started', 'installer_completion', 'terminal']
    if any(names.count(name) != 1 for name in required) or [names.index(name) for name in required] != sorted(names.index(name) for name in required):
        raise ValueError('complete ordered installer lifecycle required')
    configuration = events[names.index('configuration_validated')]
    if configuration.get('cpu_count') != 4 or configuration.get('memory_bytes') != 8 * 1024**3:
        raise ValueError('actual validated configuration event required before VM creation')
    if (not names or names[-1] != 'terminal' or events[-1].get('success') is not True
            or events[-1].get('cleanup_confirmed') is not True or events[-1].get('state') != 0
            or value.get('event_count') != len(events)
            or events[names.index('installer_completion')].get('success') is not True
            or any('cancel' in name or 'error' in name or 'expired' in name for name in names)):
        raise ValueError('uncancelled callback and stopped terminal event required')
    if type(value.get('stop_requested_after_installer_callback')) is not bool:
        raise ValueError('explicit post-callback stop disposition required')
    if value['stop_requested_after_installer_callback']:
        if (value.get('stop_callback_received') is not True or value.get('stop_callback_success') is not True
                or names.count('post_callback_destructive_stop_requested') != 1
                or names.count('post_callback_stop_completion') != 1
                or not names.index('installer_completion') < names.index('post_callback_destructive_stop_requested')
                    < names.index('post_callback_stop_completion') < names.index('terminal')):
            raise ValueError('successful supported stop strictly after installer callback required')
    elif any(name.startswith('post_callback_') for name in names) or value.get('stop_callback_received') is not False or value.get('stop_callback_success') is not False:
        raise ValueError('unexpected post-callback stop evidence')
    return value


def installed_seed_metrics(directory, original, *, uid):
    directory = Path(directory)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700 or info.st_uid != uid:
        raise ValueError('owned private installed seed directory required')
    if {path.name for path in directory.iterdir()} != set(original):
        raise ValueError('exact installed paired seed required')
    result = {}
    for name in sorted(original):
        path = directory / name
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != uid or info.st_nlink != 1:
            raise ValueError('private owned regular installed seed required: ' + name)
        row = {'bytes': info.st_size, 'allocated_bytes': info.st_blocks * 512, 'mode': 0o600, 'uid': uid, 'links': 1}
        if name == 'disk.img':
            if info.st_size != 100 * 1024**3 or not 0 < row['allocated_bytes'] <= info.st_size:
                raise ValueError('installed sparse disk geometry/allocation required')
            row['sha256'] = None
            row['content_evidence'] = 'OS clone followed by successful installer callback and observed stopped state; no whole-disk hash.'
        else:
            if not 0 < info.st_size <= 128 * 1024**2:
                raise ValueError('bounded installed metadata required')
            row['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            if name != 'auxiliary-storage.bin' and (row['bytes'] != original[name]['bytes'] or row['sha256'] != original[name]['sha256']):
                raise ValueError('installed platform identity changed: ' + name)
        result[name] = row
    return result
