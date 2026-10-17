"""Derive a blank image's unique physical store; never assume partition s1/s2."""
from pathlib import Path
import re

from macos_vm_host_metrics import attached_image_device, probe_metrics


def apfs_image_partition(value, image, whole):
    if attached_image_device(value, image) != whole:
        raise ValueError('physical partition requires the same exact owned image')
    own = next(row for row in value['images'] if row.get('image-path') == str(Path(image)))
    candidates = [row['dev-entry'] for row in own['system-entities']
                  if re.fullmatch(re.escape(whole) + r's[1-9][0-9]*', row.get('dev-entry', ''))]
    if len(candidates) != 1:
        raise ValueError('single physical partition of the exact blank image required')
    return candidates[0]


def apfs_probe_volume(value, whole, container, physical_store):
    if (not re.fullmatch(r'/dev/disk[0-9]+', whole) or not re.fullmatch(r'disk[0-9]+', container)
            or not re.fullmatch(re.escape(whole) + r's[1-9][0-9]*', physical_store)):
        raise ValueError('invalid exact image/store/container devices')
    rows = [row for row in value['Containers'] if row['ContainerReference'] == container]
    if len(rows) != 1 or [row['DeviceIdentifier'] for row in rows[0]['PhysicalStores']] != [physical_store[5:]]:
        raise ValueError('APFS container must belong only to the observed task-image store')
    volumes = rows[0]['Volumes']
    if (len(volumes) != 1 or volumes[0]['Name'] != 'Gen2VMHostProbe'
            or not re.fullmatch(re.escape(container) + r's[0-9]+', volumes[0]['DeviceIdentifier'])):
        raise ValueError('single blank probe volume required')
    return '/dev/' + volumes[0]['DeviceIdentifier']
