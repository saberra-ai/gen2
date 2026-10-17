"""The actual single-s1 layout must bridge safely without admitting foreign disks."""
import copy
from pathlib import Path
import unittest

from macos_vm_host_v3_metrics import apfs_image_partition, apfs_probe_volume


class VMHostV3MetricsTests(unittest.TestCase):
    def observed(self, index):
        return {'images': [{'image-path': '/task/blank.dmg', 'system-entities': [
            {'content-hint': 'GUID_partition_scheme', 'dev-entry': '/dev/disk12'},
            {'content-hint': '7C3457EF-0000-11AA-AA11-00306543ECAC', 'dev-entry': '/dev/disk12s' + str(index)},
            {'content-hint': 'EF57347C-0000-11AA-AA11-00306543ECAC', 'dev-entry': '/dev/disk13'},
            {'content-hint': '41504653-0000-11AA-AA11-00306543ECAC', 'dev-entry': '/dev/disk13s1'}]}]}

    def test_actual_single_s1_and_s2_layouts_use_observed_store(self):
        for index in [1, 2]:
            observed = self.observed(index)
            store = apfs_image_partition(observed, Path('/task/blank.dmg'), '/dev/disk12')
            self.assertEqual(store, '/dev/disk12s' + str(index))
            listing = {'Containers': [{'ContainerReference': 'disk13',
                'PhysicalStores': [{'DeviceIdentifier': store[5:]}],
                'Volumes': [{'Name': 'Gen2VMHostProbe', 'DeviceIdentifier': 'disk13s1'}]}]}
            self.assertEqual(apfs_probe_volume(listing, '/dev/disk12', 'disk13', store), '/dev/disk13s1')
            wrong = copy.deepcopy(listing)
            wrong['Containers'][0]['PhysicalStores'][0]['DeviceIdentifier'] = 'disk1s1'
            with self.assertRaises(ValueError):
                apfs_probe_volume(wrong, '/dev/disk12', 'disk13', store)
            with self.assertRaises(ValueError):
                apfs_probe_volume(listing, '/dev/disk12', 'disk13', '/dev/disk1s1')

    def test_ambiguous_foreign_or_wrong_image_partition_rejected(self):
        for mutation in ['two_partitions', 'foreign_partition', 'wrong_image', 'wrong_whole']:
            value = self.observed(1)
            whole = '/dev/disk12'
            own = value['images'][0]
            if mutation == 'two_partitions':
                own['system-entities'].append({'dev-entry': '/dev/disk12s2'})
            elif mutation == 'foreign_partition':
                own['system-entities'][1]['dev-entry'] = '/dev/disk1s1'
            elif mutation == 'wrong_image':
                own['image-path'] = '/other/blank.dmg'
            else:
                whole = '/dev/disk1'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                apfs_image_partition(value, Path('/task/blank.dmg'), whole)
