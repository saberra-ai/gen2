"""Validate no-boot restore metadata and the four-file sparse VM seed."""
import hashlib
from pathlib import Path
import re
import stat


def image_metrics(value, *, operation):
    expected = {'schema_version': 1, 'operation': operation,
        'restore_image_loaded': True, 'image_supported': True,
        'supported_configuration_available': True, 'hardware_model_supported': True,
        'build_version': '25D125', 'os_version': [26, 3, 0],
        'virtual_machine_created': False, 'installation_started': False, 'guest_booted': False,
        'configuration_validated': operation == 'prepare'}
    for key, want in expected.items():
        got = value.get(key)
        if type(got) is not type(want) or got != want:
            raise ValueError('restore metadata differs: ' + key)
    if (type(value.get('minimum_cpu_count')) is not int
            or not 0 < value['minimum_cpu_count'] <= 4
            or type(value.get('minimum_memory_bytes')) is not int
            or not 0 < value['minimum_memory_bytes'] <= 8 * 1024**3
            or not re.fullmatch('[0-9a-f]{64}', value.get('hardware_model_sha256', ''))):
        raise ValueError('supported image geometry/model required')
    return value


def seed_metrics(directory, value, inspected, *, uid):
    image_metrics(value, operation='prepare')
    image_metrics(inspected, operation='inspect')
    for key in ['build_version', 'os_version', 'hardware_model_sha256', 'minimum_cpu_count', 'minimum_memory_bytes']:
        if value[key] != inspected[key]:
            raise ValueError('image changed between inspection and preparation: ' + key)
    geometry = {'cpu_count': 4, 'memory_bytes': 8 * 1024**3, 'disk_logical_bytes': 100 * 1024**3,
        'network_attachment': 'nat', 'graphics_displays': 1, 'audio_devices': 0, 'shared_directories': 0,
        'display_window_created': False, 'custom_listener_created': False}
    for key, want in geometry.items():
        if type(value.get(key)) is not type(want) or value[key] != want:
            raise ValueError('seed geometry differs: ' + key)
    mac = value.get('mac_address', '')
    if not re.fullmatch('(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}', mac) or int(mac[:2], 16) & 3 != 2:
        raise ValueError('locally administered unicast MAC required')
    directory = Path(directory)
    mode = directory.lstat()
    if not stat.S_ISDIR(mode.st_mode) or stat.S_IMODE(mode.st_mode) != 0o700 or mode.st_uid != uid:
        raise ValueError('owned private seed directory required')
    names = {'hardware-model.bin', 'machine-id.bin', 'auxiliary-storage.bin', 'disk.img'}
    if {path.name for path in directory.iterdir()} != names:
        raise ValueError('exact four-file seed required')
    result = {}
    for name in sorted(names):
        path = directory / name
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != uid or info.st_nlink != 1):
            raise ValueError('owned private single-link regular seed required: ' + name)
        row = {'bytes': info.st_size, 'allocated_bytes': info.st_blocks * 512,
            'mode': stat.S_IMODE(info.st_mode), 'uid': info.st_uid, 'links': info.st_nlink}
        if name == 'disk.img':
            if info.st_size != 100 * 1024**3 or info.st_blocks != 0:
                raise ValueError('fresh unallocated sparse 100GiB disk required')
            row['sha256'] = None
            row['content_evidence'] = 'Fresh O_EXCL/ftruncate-only creation; zero allocated blocks; no whole-disk hash.'
        else:
            if not 0 < info.st_size <= 128 * 1024**2:
                raise ValueError('bounded nonempty seed metadata required: ' + name)
            row['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        result[name] = row
    for name, key in [('hardware-model.bin', 'hardware_model_sha256'), ('machine-id.bin', 'machine_identifier_sha256')]:
        if result[name]['sha256'] != value.get(key):
            raise ValueError('persisted seed metadata differs: ' + name)
    return result
