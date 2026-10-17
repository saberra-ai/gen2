"""Reject confused group/UUID/mount bindings against real saved Apple plists."""
import copy
from pathlib import Path
import plistlib
import unittest

from macos_vm_data_metrics import mounted_member, system_data_pair


class DataMetricsTests(unittest.TestCase):
    def evidence(self):
        root = Path(__file__).resolve().parents[2] / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-vm-boot-stopped-readonly'
        load = lambda name: plistlib.loads((root / name).read_bytes())
        listing, groups = load('container-1-list.stdout.log'), load('container-1-groups.stdout.log')
        infos = {row['DeviceIdentifier']: load(f'volume-1-{index}-info.stdout.log')
            for index, row in enumerate(listing['Containers'][0]['Volumes'])}
        return listing, groups, 'disk15', '/dev/disk12s2', infos, load('partition-1-info.stdout.log')

    def test_actual_saved_group_uuid_and_two_role_bridge(self):
        pair = system_data_pair(*self.evidence())
        self.assertEqual(pair['system']['device'], '/dev/disk15s1')
        self.assertEqual(pair['data']['device'], '/dev/disk15s5')
        self.assertEqual(pair['group_uuid'], 'F3BC5BEA-3D05-45B2-940C-6EB51C65A411')

    def test_duplicate_groups_roles_or_devices_rejected(self):
        for mutation in ['group', 'role', 'device', 'missing']:
            args = list(self.evidence())
            group = args[1]['Containers'][0]['VolumeGroups'][0]
            if mutation == 'group': args[1]['Containers'][0]['VolumeGroups'].append(copy.deepcopy(group))
            elif mutation == 'role': group['Volumes'][0]['Role'] = 'Data'
            elif mutation == 'device': group['Volumes'][0]['DeviceIdentifier'] = group['Volumes'][1]['DeviceIdentifier']
            else: group['Volumes'].pop()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                system_data_pair(*args)

    def test_independent_container_member_info_uuids_and_foreign_store_rejected(self):
        for mutation in ['container', 'member', 'info', 'group', 'store', 'store_uuid', 'mounted', 'locked']:
            args = list(self.evidence())
            if mutation == 'container': args[1]['Containers'][0]['APFSContainerUUID'] = '00000000-0000-0000-0000-000000000000'
            elif mutation == 'member': args[1]['Containers'][0]['VolumeGroups'][0]['Volumes'][1]['DiskUUID'] = '00000000-0000-0000-0000-000000000000'
            elif mutation == 'info': args[4]['disk15s5']['VolumeUUID'] = '00000000-0000-0000-0000-000000000000'
            elif mutation == 'group': args[4]['disk15s5']['APFSVolumeGroupID'] = '00000000-0000-0000-0000-000000000000'
            elif mutation == 'store': args[0]['Containers'][0]['PhysicalStores'][0]['DeviceIdentifier'] = 'disk99s2'
            elif mutation == 'store_uuid': args[5]['DiskUUID'] = '00000000-0000-0000-0000-000000000000'
            elif mutation == 'mounted': args[4]['disk15s5']['MountPoint'] = '/host'
            else: args[0]['Containers'][0]['Volumes'][4]['Locked'] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                system_data_pair(*args)

    def test_mount_requires_actual_device_flags_uuid_and_exact_private_path(self):
        args = self.evidence()
        pair = system_data_pair(*args)
        info = copy.deepcopy(args[4]['disk15s5'])
        info.update(MountPoint='/task/data', WritableVolume=True)
        status = {'is_mount': True, 'stat_device': 7, 'parent_stat_device': 4, 'readonly': False}
        self.assertEqual(mounted_member(info, '/task/data', pair['data'], pair, status, readonly=False)['device'], '/dev/disk15s5')
        for mutation in ['path', 'uuid', 'readonly', 'device', 'ownership']:
            bad, observed = copy.deepcopy(info), dict(status)
            if mutation == 'path': bad['MountPoint'] = '/host'
            elif mutation == 'uuid': bad['VolumeUUID'] = '00000000-0000-0000-0000-000000000000'
            elif mutation == 'readonly': observed['readonly'] = True
            elif mutation == 'device': observed['stat_device'] = observed['parent_stat_device']
            else: bad['GlobalPermissionsEnabled'] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                mounted_member(bad, '/task/data', pair['data'], pair, observed, readonly=False)
