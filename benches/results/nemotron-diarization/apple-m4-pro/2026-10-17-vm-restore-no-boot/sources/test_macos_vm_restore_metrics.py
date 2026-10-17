"""Reject incompatible images, boot claims and unsafe/mutated seed files."""
import copy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest

from macos_vm_restore_metrics import image_metrics, seed_metrics


def metadata(operation):
    return {'schema_version': 1, 'operation': operation, 'restore_image_loaded': True,
        'image_supported': True, 'supported_configuration_available': True, 'hardware_model_supported': True,
        'build_version': '25D125', 'os_version': [26, 3, 0],
        'virtual_machine_created': False, 'installation_started': False, 'guest_booted': False,
        'configuration_validated': operation == 'prepare', 'minimum_cpu_count': 2,
        'minimum_memory_bytes': 4 * 1024**3, 'hardware_model_sha256': hashlib.sha256(b'model').hexdigest(),
        'machine_identifier_sha256': hashlib.sha256(b'id').hexdigest(),
        'cpu_count': 4, 'memory_bytes': 8 * 1024**3, 'disk_logical_bytes': 100 * 1024**3,
        'network_attachment': 'nat', 'graphics_displays': 1, 'audio_devices': 0, 'shared_directories': 0,
        'display_window_created': False, 'custom_listener_created': False, 'mac_address': '02:03:04:05:06:07'}


class RestoreMetricsTests(unittest.TestCase):
    def seed(self, parent):
        root = Path(parent) / 'seed'
        root.mkdir(mode=0o700)
        for name, data in [('hardware-model.bin', b'model'), ('machine-id.bin', b'id'), ('auxiliary-storage.bin', b'aux')]:
            path = root / name
            path.write_bytes(data)
            path.chmod(0o600)
        with (root / 'disk.img').open('xb') as stream:
            stream.truncate(100 * 1024**3)
        (root / 'disk.img').chmod(0o600)
        return root

    def test_sparse_disk_is_not_read_or_hashed(self):
        with tempfile.TemporaryDirectory() as parent:
            root = self.seed(parent)
            result = seed_metrics(root, metadata('prepare'), metadata('inspect'), uid=os.getuid())
            self.assertEqual(result['disk.img']['allocated_bytes'], 0)
            self.assertIsNone(result['disk.img']['sha256'])
            self.assertEqual(result['hardware-model.bin']['sha256'], metadata('prepare')['hardware_model_sha256'])

    def test_incompatible_image_or_boot_claim_rejected(self):
        for key, bad in [('build_version', '25D999'), ('os_version', [27, 0, 0]),
                ('guest_booted', True), ('virtual_machine_created', True), ('installation_started', True),
                ('image_supported', 1), ('minimum_cpu_count', 8), ('minimum_memory_bytes', 16 * 1024**3)]:
            value = metadata('inspect')
            value[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                image_metrics(value, operation='inspect')

    def test_seed_mutation_or_exposure_rejected(self):
        for mutation in ['allocated', 'short', 'symlink', 'mode', 'extra', 'model', 'hardlink']:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as parent:
                root = self.seed(parent)
                disk = root / 'disk.img'
                if mutation == 'allocated':
                    with disk.open('r+b') as stream:
                        stream.write(b'x')
                        stream.flush()
                        os.fsync(stream.fileno())
                elif mutation == 'short':
                    with disk.open('r+b') as stream:
                        stream.truncate(512)
                elif mutation == 'symlink':
                    (root / 'machine-id.bin').unlink()
                    (root / 'machine-id.bin').symlink_to(root / 'hardware-model.bin')
                elif mutation == 'mode':
                    (root / 'auxiliary-storage.bin').chmod(0o644)
                elif mutation == 'extra':
                    (root / 'unexpected').touch()
                elif mutation == 'model':
                    (root / 'hardware-model.bin').write_bytes(b'wrong')
                else:
                    os.link(root / 'machine-id.bin', Path(parent) / 'external')
                with self.assertRaises(ValueError):
                    seed_metrics(root, metadata('prepare'), metadata('inspect'), uid=os.getuid())

    def test_geometry_and_model_must_match_inspected_image(self):
        for key, bad in [('hardware_model_sha256', '0' * 64), ('cpu_count', 5),
                ('mac_address', '03:01:02:03:04:05'), ('network_attachment', 'bridged')]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as parent:
                root = self.seed(parent)
                value = copy.deepcopy(metadata('prepare'))
                value[key] = bad
                with self.assertRaises(ValueError):
                    seed_metrics(root, value, metadata('inspect'), uid=os.getuid())
