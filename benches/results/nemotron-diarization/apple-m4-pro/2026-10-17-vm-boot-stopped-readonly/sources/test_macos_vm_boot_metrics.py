"""Reject incomplete start/stop proof and foreign/ambiguous readonly layouts."""
import copy
from pathlib import Path
import unittest
from macos_vm_boot_metrics import boot_metrics, observed_partitions, scoped_container


class BootMetricsTests(unittest.TestCase):
    def evidence(self):
        value = {'schema_version': 1, 'boot_owner_qualified': True, 'virtual_machine_created': True,
            'normal_start_called': True, 'start_callback_received': True, 'start_callback_success': True,
            'run_window_completed': True, 'configured_run_seconds': 120.0, 'cancellation_requested': False,
            'cleanup_confirmed': True, 'final_state': 0, 'errors': [], 'logging_failed': False,
            'guest_os_ready_qualified': False, 'custom_listener_created': False, 'display_window_created': False,
            'running_seconds': 120.1, 'forced_stop_requested': False, 'forced_stop_completed': False,
            'forced_stop_success': False, 'guest_stop_observed': True, 'event_count': 8}
        names = ['configuration_validated', 'virtual_machine_created', 'normal_start_requested',
            'start_completion', 'run_window_completed', 'shutdown_began', 'guest_did_stop', 'terminal']
        events = [{'event': name, 'elapsed_seconds': elapsed, 'state': 0} for name, elapsed in
            zip(names, [0., .1, .2, 1., 121.1, 121.2, 122., 122.1])]
        events[0].update(cpu_count=4, memory_bytes=8 * 1024**3)
        events[3].update(success=True, state=1)
        events[4]['state'] = 1
        events[-1].update(success=True, cleanup_confirmed=True)
        return value, {'returncode': 0}, events

    def test_callback_full_interval_delegate_stopped_and_exit_passes(self):
        value, command, events = self.evidence()
        self.assertEqual(boot_metrics(value, command, events), value)

    def test_running_callback_alone_early_stop_cancel_or_readiness_promotion_rejected(self):
        for key, bad in [('running_seconds', 119.), ('cleanup_confirmed', False), ('guest_stop_observed', False),
                ('final_state', 1), ('guest_os_ready_qualified', True), ('cancellation_requested', True)]:
            value, command, events = self.evidence()
            value[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                boot_metrics(value, command, events)
        value, command, events = self.evidence()
        command['forced_process_group_termination'] = True
        with self.assertRaises(ValueError):
            boot_metrics(value, command, events)

    def test_forced_stop_needs_its_completion_after_shutdown(self):
        value, command, events = self.evidence()
        value.update(forced_stop_requested=True, forced_stop_completed=True, forced_stop_success=True, guest_stop_observed=False, event_count=9)
        events[6:7] = [{'event': 'destructive_stop_requested', 'elapsed_seconds': 122., 'state': 1},
            {'event': 'destructive_stop_completion', 'elapsed_seconds': 122.05, 'state': 0, 'success': True}]
        self.assertEqual(boot_metrics(value, command, events), value)
        value['forced_stop_success'] = False
        with self.assertRaises(ValueError):
            boot_metrics(value, command, events)

    def test_physical_partition_selection_uses_all_observed_children_only(self):
        image = Path('/task/disk.img')
        inventory = {'images': [{'image-path': str(image), 'writeable': False, 'system-entities': [
            {'dev-entry': '/dev/disk11', 'content-hint': 'GUID_partition_scheme'},
            {'dev-entry': '/dev/disk11s1'}, {'dev-entry': '/dev/disk11s3'},
            {'dev-entry': '/dev/disk12'}, {'dev-entry': '/dev/disk12s1'}]}]}
        self.assertEqual(observed_partitions(inventory, image, '/dev/disk11'), ['/dev/disk11s1', '/dev/disk11s3'])
        bad = copy.deepcopy(inventory)
        bad['images'][0]['writeable'] = True
        with self.assertRaises(ValueError):
            observed_partitions(bad, image, '/dev/disk11')

    def test_container_rejects_foreign_extra_store_or_unscoped_volume(self):
        value = {'Containers': [{'ContainerReference': 'disk12', 'PhysicalStores': [{'DeviceIdentifier': 'disk11s3'}],
            'Volumes': [{'DeviceIdentifier': 'disk12s1', 'Roles': ['System']}, {'DeviceIdentifier': 'disk12s5', 'Roles': ['Data']}]}]}
        self.assertEqual(len(scoped_container(value, 'disk12', '/dev/disk11s3')['Volumes']), 2)
        for mutation in ['foreign', 'extra', 'volume']:
            bad = copy.deepcopy(value)
            if mutation == 'foreign': bad['Containers'][0]['PhysicalStores'][0]['DeviceIdentifier'] = 'disk99s1'
            elif mutation == 'extra': bad['Containers'][0]['PhysicalStores'].append({'DeviceIdentifier': 'disk11s1'})
            else: bad['Containers'][0]['Volumes'][0]['DeviceIdentifier'] = 'disk99s1'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                scoped_container(bad, 'disk12', '/dev/disk11s3')
