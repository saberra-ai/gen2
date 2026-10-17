"""Normal-start owner evidence and scoped readonly discovery, never guest readiness."""
from pathlib import Path
import math
import re


def boot_metrics(value, command, events):
    expected = {'schema_version': 1, 'boot_owner_qualified': True, 'virtual_machine_created': True,
        'normal_start_called': True, 'start_callback_received': True, 'start_callback_success': True,
        'run_window_completed': True, 'cancellation_requested': False,
        'cleanup_confirmed': True, 'final_state': 0, 'errors': [], 'logging_failed': False,
        'guest_os_ready_qualified': False, 'custom_listener_created': False, 'display_window_created': False}
    for key, want in expected.items():
        if type(value.get(key)) is not type(want) or value[key] != want:
            raise ValueError('normal-start owner evidence differs: ' + key)
    if command.get('returncode') != 0 or any(key in command for key in ['abort_error', 'forced_process_group_termination', 'cancellation_signal', 'startup_error']):
        raise ValueError('normal owner exit without supervisor abort required')
    running = value.get('running_seconds')
    if (type(value.get('configured_run_seconds')) not in {int, float} or value['configured_run_seconds'] != 120
            or type(running) not in {int, float} or not math.isfinite(running) or running < 120):
        raise ValueError('complete fixed running interval required')
    names = [row['event'] for row in events]
    required = ['configuration_validated', 'virtual_machine_created', 'normal_start_requested',
        'start_completion', 'run_window_completed', 'shutdown_began', 'terminal']
    if any(names.count(name) != 1 for name in required) or [names.index(name) for name in required] != sorted(names.index(name) for name in required):
        raise ValueError('complete ordered normal-start lifecycle required')
    previous = -1
    for event in events:
        elapsed = event.get('elapsed_seconds')
        if type(elapsed) not in {int, float} or not math.isfinite(elapsed) or elapsed < previous:
            raise ValueError('finite ordered event times required')
        previous = elapsed
    configuration = events[names.index('configuration_validated')]
    start = events[names.index('start_completion')]
    end = events[names.index('run_window_completed')]
    if (configuration.get('cpu_count') != 4 or configuration.get('memory_bytes') != 8 * 1024**3
            or start.get('success') is not True or start.get('state') != 1 or end.get('state') != 1
            or end['elapsed_seconds'] - start['elapsed_seconds'] < 120
            or names[-1] != 'terminal' or events[-1].get('success') is not True
            or events[-1].get('cleanup_confirmed') is not True or events[-1].get('state') != 0
            or value.get('event_count') != len(events)
            or any('cancel' in name or 'error' in name or 'expired' in name for name in names)):
        raise ValueError('supported running and stopped terminal observations required')
    if value.get('forced_stop_requested') is True:
        if (value.get('forced_stop_completed') is not True or value.get('forced_stop_success') is not True
                or names.count('destructive_stop_requested') != 1 or names.count('destructive_stop_completion') != 1
                or not names.index('shutdown_began') < names.index('destructive_stop_requested')
                    < names.index('destructive_stop_completion') < names.index('terminal')):
            raise ValueError('successful supported destructive stop evidence required')
        request = events[names.index('destructive_stop_requested')]
        completion = events[names.index('destructive_stop_completion')]
        if request.get('state') not in {1, 2} or completion.get('success') is not True or completion.get('state') != 0:
            raise ValueError('supported forced stop and stopped completion required')
    elif value.get('forced_stop_requested') is False:
        if (value.get('guest_stop_observed') is not True or names.count('guest_did_stop') != 1
                or value.get('forced_stop_completed') is not False or value.get('forced_stop_success') is not False
                or any(name.startswith('destructive_stop_') for name in names)
                or not names.index('shutdown_began') < names.index('guest_did_stop') < names.index('terminal')
                or events[names.index('guest_did_stop')].get('state') != 0):
            raise ValueError('guest-stop delegate event required without forced stop')
    else:
        raise ValueError('explicit shutdown disposition required')
    return value


def observed_partitions(inventory, image, whole):
    from macos_vm_host_metrics import attached_image_device
    if attached_image_device(inventory, image) != whole:
        raise ValueError('same exact image required for readonly discovery')
    own = next(row for row in inventory['images'] if row.get('image-path') == str(Path(image)))
    if own.get('writeable') is not False:
        raise ValueError('observed read-only attachment required')
    candidates = [row['dev-entry'] for row in own['system-entities']
        if re.fullmatch(re.escape(whole) + r's[1-9][0-9]*', row.get('dev-entry', ''))]
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError('unique observed physical children required; no partition-index assumption')
    return candidates


def scoped_container(value, container, partition):
    if not re.fullmatch(r'disk[0-9]+', container):
        raise ValueError('observed container identifier required')
    rows = [row for row in value['Containers'] if row['ContainerReference'] == container]
    if len(rows) != 1 or [row['DeviceIdentifier'] for row in rows[0]['PhysicalStores']] != [partition[5:]]:
        raise ValueError('container must have only the exact observed task-image store')
    for row in rows[0]['Volumes']:
        if (not re.fullmatch(re.escape(container) + r's[0-9]+', row['DeviceIdentifier'])
                or not isinstance(row['Roles'], list) or any(not isinstance(role, str) for role in row['Roles'])):
            raise ValueError('actual scoped volumes/roles required')
    return rows[0]
