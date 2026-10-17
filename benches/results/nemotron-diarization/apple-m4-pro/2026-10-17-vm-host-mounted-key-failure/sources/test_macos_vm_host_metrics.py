"""Reject wrong/ambiguous image devices before any mount or detach command."""
import copy
import unittest
from pathlib import Path

from macos_vm_host_metrics import attached_image_device, apfs_probe_volume, probe_metrics


class VMHostMetricsTests(unittest.TestCase):
    def test_attachment_selects_only_exact_task_image_and_whole_guid_device(self):
        own = {'image-path': '/task/blank.dmg', 'system-entities': [
            {'content-hint': 'GUID_partition_scheme', 'dev-entry': '/dev/disk9'},
            {'content-hint': 'Apple_APFS', 'dev-entry': '/dev/disk9s2'}]}
        other = {'image-path': '/other/blank.dmg', 'system-entities': [
            {'content-hint': 'GUID_partition_scheme', 'dev-entry': '/dev/disk1'}]}
        self.assertEqual(attached_image_device({'images': [other, own]}, Path('/task/blank.dmg')), '/dev/disk9')
        self.assertIsNone(attached_image_device({'images': [other]}, Path('/task/blank.dmg')))
        with self.assertRaises(ValueError):
            attached_image_device({}, Path('/task/blank.dmg'))
        for mutation in ['duplicate', 'partition', 'missing_guid', 'two_whole']:
            changed = copy.deepcopy(own)
            if mutation == 'partition':
                changed['system-entities'][0]['dev-entry'] = '/dev/disk9s2'
            elif mutation == 'missing_guid':
                changed['system-entities'][0]['content-hint'] = 'Apple_APFS'
            elif mutation == 'two_whole':
                changed['system-entities'].append(copy.deepcopy(changed['system-entities'][0]))
            rows = [changed, changed] if mutation == 'duplicate' else [changed]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                attached_image_device({'images': rows}, Path('/task/blank.dmg'))

    def test_volume_rejects_foreign_physical_store_or_ambiguous_volume(self):
        original = {'Containers': [{'ContainerReference': 'disk10', 'PhysicalStores': [{'DeviceIdentifier': 'disk9s2'}],
            'Volumes': [{'Name': 'Gen2VMHostProbe', 'DeviceIdentifier': 'disk10s1'}]}]}
        self.assertEqual(apfs_probe_volume(original, '/dev/disk9', 'disk10'), '/dev/disk10s1')
        for mutation in ['foreign_store', 'shared_store', 'foreign_volume', 'duplicate_volume']:
            changed = copy.deepcopy(original)
            row = changed['Containers'][0]
            if mutation == 'foreign_store':
                row['PhysicalStores'][0]['DeviceIdentifier'] = 'disk1s2'
            elif mutation == 'shared_store':
                row['PhysicalStores'].append({'DeviceIdentifier': 'disk1s2'})
            elif mutation == 'foreign_volume':
                row['Volumes'][0]['DeviceIdentifier'] = 'disk1s1'
            else:
                row['Volumes'].append(copy.deepcopy(row['Volumes'][0]))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                apfs_probe_volume(changed, '/dev/disk9', 'disk10')

    def test_host_support_is_not_guest_execution_or_only_self_reported_geometry(self):
        original = {'schema_version': 1, 'virtualization_supported': True, 'geometry_within_host_limits': True,
            'minimum_cpu_count': 1, 'maximum_cpu_count': 12, 'proposed_cpu_count': 4,
            'minimum_memory_bytes': 1024**3, 'maximum_memory_bytes': 32 * 1024**3,
            'proposed_memory_bytes': 8 * 1024**3, 'virtual_machine_created': False,
            'restore_image_loaded': False, 'guest_booted': False}
        probe_metrics(original)
        for key, value in [('schema_version', True), ('virtualization_supported', False), ('guest_booted', True),
                           ('maximum_cpu_count', 2), ('proposed_cpu_count', True),
                           ('maximum_memory_bytes', 4 * 1024**3)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                probe_metrics({**original, key: value})
