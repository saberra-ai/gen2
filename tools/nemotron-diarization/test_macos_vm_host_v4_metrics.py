"""Absent undocumented Mounted is acceptable only with real mount/device proof."""
import copy
from pathlib import Path
import unittest

from macos_vm_host_v4_metrics import mounted_volume_matches


class VMHostV4MetricsTests(unittest.TestCase):
    def test_supported_plist_requires_os_mount_and_exact_device_store_bridge(self):
        value = {'MountPoint': '/task/mount', 'DeviceNode': '/dev/disk13s1', 'DeviceIdentifier': 'disk13s1',
            'FilesystemType': 'apfs', 'APFSContainerReference': 'disk13',
            'APFSPhysicalStores': [{'APFSPhysicalStore': 'disk12s1'}],
            'WritableVolume': True, 'GlobalPermissionsEnabled': False}
        status = {'is_mount': True, 'stat_device': 42, 'parent_stat_device': 13}
        def check(v, s):
            return mounted_volume_matches(v, Path('/task/mount'), '/dev/disk13s1', 'disk13', '/dev/disk12s1', s)
        self.assertEqual(check(value, status)['physical_store'], '/dev/disk12s1')
        for mutation in ['not_mount', 'same_device', 'wrong_mountpoint', 'foreign_device',
                         'foreign_store', 'wrong_container', 'readonly', 'contradictory_mounted']:
            v, s = copy.deepcopy(value), copy.deepcopy(status)
            if mutation == 'not_mount':
                s['is_mount'] = False
            elif mutation == 'same_device':
                s['stat_device'] = s['parent_stat_device']
            elif mutation == 'wrong_mountpoint':
                v['MountPoint'] = '/other/mount'
            elif mutation == 'foreign_device':
                v['DeviceNode'] = '/dev/disk1s1'
            elif mutation == 'foreign_store':
                v['APFSPhysicalStores'][0]['APFSPhysicalStore'] = 'disk1s1'
            elif mutation == 'wrong_container':
                v['APFSContainerReference'] = 'disk1'
            elif mutation == 'readonly':
                v['WritableVolume'] = False
            else:
                v['Mounted'] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                check(v, s)
