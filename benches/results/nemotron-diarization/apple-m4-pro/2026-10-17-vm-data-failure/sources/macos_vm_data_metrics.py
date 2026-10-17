"""Bind one observed System/Data group and actual private mounts; no name guesses."""
from pathlib import Path
import re
from uuid import UUID

from macos_vm_boot_metrics import scoped_container


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'disk[0-9]+s[1-9][0-9]*', value):
        raise ValueError('observed APFS volume identifier required')
    return value


def uuid(value):
    if not isinstance(value, str):
        raise ValueError('actual UUID string required')
    return str(UUID(value)).upper()


def system_data_pair(listing, groups, container, partition, infos, physical_info):
    row = scoped_container(listing, container, partition)
    if len(listing['Containers']) != 1 or len(groups['Containers']) != 1:
        raise ValueError('container-scoped raw records required')
    group_container = groups['Containers'][0]
    container_uuid = uuid(row['APFSContainerUUID'])
    if (physical_info['DeviceIdentifier'] != partition[5:] or physical_info['DeviceNode'] != partition
            or physical_info['APFSContainerReference'] != container
            or uuid(physical_info['DiskUUID']) != uuid(row['PhysicalStores'][0]['DiskUUID'])):
        raise ValueError('physical partition UUID must bind its scoped APFS store')
    if (group_container['ContainerReference'] != container
            or uuid(group_container['APFSContainerUUID']) != container_uuid
            or len(group_container['VolumeGroups']) != 1):
        raise ValueError('one unambiguous group in the exact container required')
    group = group_container['VolumeGroups'][0]
    group_uuid = uuid(group['APFSVolumeGroupUUID'])
    members = group['Volumes']
    if len(members) != 2 or sorted(member['Role'] for member in members) != ['Data', 'System']:
        raise ValueError('exactly one System and one Data group member required')
    volumes = {identifier(volume['DeviceIdentifier']): volume for volume in row['Volumes']}
    if len(volumes) != len(row['Volumes']) or len({uuid(v['APFSVolumeUUID']) for v in volumes.values()}) != len(volumes):
        raise ValueError('unique observed volume identifiers and UUIDs required')
    result = {'container': container, 'physical_store': partition,
        'container_uuid': container_uuid, 'group_uuid': group_uuid}
    selected = set()
    for member in members:
        name = identifier(member['DeviceIdentifier'])
        if name in selected or name not in volumes:
            raise ValueError('distinct group members from scoped volume listing required')
        selected.add(name)
        volume, info = volumes[name], infos[name]
        role, volume_uuid = member['Role'], uuid(member['DiskUUID'])
        if (volume['Roles'] != [role] or uuid(volume['APFSVolumeUUID']) != volume_uuid
                or any(volume[key] is not False for key in ['Locked', 'Encryption', 'FileVault'])
                or info['DeviceIdentifier'] != name or info['DeviceNode'] != '/dev/' + name
                or info['APFSContainerReference'] != container or info['ParentWholeDisk'] != container
                or info['APFSPhysicalStores'] != [{'APFSPhysicalStore': partition[5:]}]
                or uuid(info['DiskUUID']) != volume_uuid or uuid(info['VolumeUUID']) != volume_uuid
                or uuid(info['APFSVolumeGroupID']) != group_uuid or info.get('MountPoint')):
            raise ValueError('group/list/info role, UUID and physical-store bridge required')
        result[role.lower()] = {'device': '/dev/' + name, 'uuid': volume_uuid}
    return result


def mounted_member(info, mount, member, pair, status, *, readonly):
    if (info['DeviceNode'] != member['device'] or info['DeviceIdentifier'] != member['device'][5:]
            or uuid(info['DiskUUID']) != member['uuid'] or uuid(info['VolumeUUID']) != member['uuid']
            or uuid(info['APFSVolumeGroupID']) != pair['group_uuid']
            or info['APFSContainerReference'] != pair['container'] or info['ParentWholeDisk'] != pair['container']
            or info['APFSPhysicalStores'] != [{'APFSPhysicalStore': pair['physical_store'][5:]}]
            or info['MountPoint'] != str(Path(mount)) or info['FilesystemType'] != 'apfs'
            or info['GlobalPermissionsEnabled'] is not False or info['WritableVolume'] is not (not readonly)
            or status['is_mount'] is not True or status['readonly'] is not readonly
            or type(status['stat_device']) is not int or type(status['parent_stat_device']) is not int
            or status['stat_device'] == status['parent_stat_device']):
        raise ValueError('actual private APFS mount, access flags and exact UUID/group bridge required')
    return {**status, 'device': member['device'], 'volume_uuid': member['uuid'],
        'group_uuid': pair['group_uuid'], 'global_permissions_enabled': False}
