"""Strict no-boot host probe and task-owned disk selection checks."""
from pathlib import Path
import re


def probe_metrics(value):
    if (type(value['schema_version']) is not int or value['schema_version'] != 1 or value['virtualization_supported'] is not True
            or value['geometry_within_host_limits'] is not True
            or value['proposed_cpu_count'] != 4 or value['proposed_memory_bytes'] != 8 * 1024**3
            or any(value[name] is not False for name in
                   ['virtual_machine_created', 'restore_image_loaded', 'guest_booted'])):
        raise ValueError('supported host-only virtualization probe required')
    for minimum, maximum, proposed in [('minimum_cpu_count', 'maximum_cpu_count', 'proposed_cpu_count'),
                                        ('minimum_memory_bytes', 'maximum_memory_bytes', 'proposed_memory_bytes')]:
        if (any(type(value[n]) is not int for n in [minimum, maximum, proposed])
                or not 0 < value[minimum] <= value[proposed] <= value[maximum]):
            raise ValueError('proposed geometry outside measured host limits')
    return value


def attached_image_device(value, image):
    """Never select a device by volume label, previous disk number or first image."""
    if not Path(image).is_absolute():
        raise ValueError('absolute task-owned image path required')
    if not isinstance(value.get('images'), list):
        raise ValueError('actual image inventory required before device selection')
    matches = [entry for entry in value['images'] if entry.get('image-path') == str(image)]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError('ambiguous task-owned image attachment')
    devices = [entry['dev-entry'] for entry in matches[0]['system-entities']
               if entry.get('content-hint') == 'GUID_partition_scheme']
    if len(devices) != 1 or not re.fullmatch(r'/dev/disk[0-9]+', devices[0]):
        raise ValueError('exact whole GUID device for task image required')
    return devices[0]


def apfs_probe_volume(value, whole, container):
    if not re.fullmatch(r'/dev/disk[0-9]+', whole) or not re.fullmatch(r'disk[0-9]+', container):
        raise ValueError('invalid exact image/container devices')
    rows = [row for row in value['Containers'] if row['ContainerReference'] == container]
    if len(rows) != 1 or [row['DeviceIdentifier'] for row in rows[0]['PhysicalStores']] != [whole[5:] + 's2']:
        raise ValueError('APFS container must belong only to the exact task image')
    volumes = rows[0]['Volumes']
    if (len(volumes) != 1 or volumes[0]['Name'] != 'Gen2VMHostProbe'
            or not re.fullmatch(re.escape(container) + r's[0-9]+', volumes[0]['DeviceIdentifier'])):
        raise ValueError('single blank probe volume required')
    return '/dev/' + volumes[0]['DeviceIdentifier']
