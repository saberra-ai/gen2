"""Combine actual filesystem mount status with exact diskutil device identity."""
from pathlib import Path

from macos_vm_host_v3_metrics import attached_image_device, apfs_image_partition, apfs_probe_volume, probe_metrics


def mounted_volume_matches(value, mountpoint, volume, container, physical_store, status):
    if (value['MountPoint'] != str(Path(mountpoint)) or value['DeviceNode'] != volume
            or value['DeviceIdentifier'] != volume[5:] or value['FilesystemType'] != 'apfs'
            or value['APFSContainerReference'] != container
            or value['APFSPhysicalStores'] != [{'APFSPhysicalStore': physical_store[5:]}]
            or value['WritableVolume'] is not True
            or type(value['GlobalPermissionsEnabled']) is not bool
            or ('Mounted' in value and value['Mounted'] is not True)
            or status['is_mount'] is not True
            or type(status['stat_device']) is not int or type(status['parent_stat_device']) is not int
            or status['stat_device'] == status['parent_stat_device']):
        raise ValueError('actual task mount and exact owned APFS device identity required')
    return {**status, 'device': volume, 'physical_store': physical_store,
            'global_permissions_enabled': value['GlobalPermissionsEnabled']}
