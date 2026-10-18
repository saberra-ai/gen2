"""Read-only snapshot of retained gate evidence; no runtime or model execution."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import stat


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def capture(path, expected):
    if not stat.S_ISREG(path.lstat().st_mode) or any(parent.is_symlink() for parent in path.parents):
        raise ValueError('regular nonsymlink audit input required')
    raw = path.read_bytes()
    if identity(raw) != expected:
        raise ValueError('first-bound audit input changed: ' + str(path))
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.protocol.read_bytes()
    protocol_identity = identity(raw)
    if protocol_identity['sha256'] != args.sha256:
        raise ValueError('audit protocol changed')
    pre = json.loads(raw)
    root = Path(pre['repository'])
    captured = {name: capture(Path(row['path']), row['identity']) for name, row in pre['inputs'].items()}
    if str(Path(__file__).resolve()) != pre['inputs']['audit_source']['path']:
        raise ValueError('first-bound audit source required')
    facts = {name: json.loads(data) for name, data in captured.items() if name not in
             ('audit_source', 'quality_investigation', 'segmentation_investigation', 'roadmap')}
    corpus, original = facts['current_corpus'], facts['original_nemo']
    public, windows = facts['bounded_public'], facts['windows_candidate']
    word, template = facts['word_preparation'], facts['review_template']
    vm, storage = facts['vm_data_failure'], facts['vm_boot_protocol']['storage']
    pnola, observer = facts['bounded_pnola'], facts['thread_cpu_replay']
    # These are consistency assertions on the fixed retained observations, not
    # substitutes for the original full numerical/command replay or new gates.
    if not (corpus['all_fixed_pairs_completed'] is True and
            corpus['all_full_valid_row_wrapper_comparisons_byte_exact'] is True and
            corpus['all_slice_quality_gates_passed'] is False and
            len(corpus['failed_slice_backend_pairs']) == 4):
        raise ValueError('expected fixed-corpus outcome changed')
    if not (original['both_fixed_original_cpu_calls_completed'] is True and
            original['original_ami_shared_quality_gates_passed'] is False and
            public['public_realtime_gate_passed'] is True and
            public['actual_model_reloads_in_existing_lifecycle_test'] == 100 and
            windows['functional_lifecycle_gates_passed'] is True and
            windows['realtime_gate_passed'] is False and
            pnola['automatic_packaged_resolution_gates_passed'] is True):
        raise ValueError('expected retained runtime outcomes changed')
    if not (word['human_annotators'] == word['human_words'] == 0 and
            word['human_gold_admitted'] is False and template['status'] == 'unreviewed' and
            template['annotators'] == template['words'] == [] and template['gold_admitted'] is False):
        raise ValueError('prepared review state changed; inspect actual human evidence separately')
    if not (vm['data_access_passed'] is False and vm['guest_os_ready_qualified'] is False and
            vm['fresh_machine_qualified'] is False and
            observer['timing_comparison']['bounded_observer_interpretation_supported'] is False):
        raise ValueError('expected incomplete VM/negative observer outcome changed')
    if storage != {'start_minimum': 120 * 1024**3, 'running_minimum': 80 * 1024**3,
                   'reserve_bytes': 256 * 1024**2}:
        raise ValueError('original VM storage floors changed')
    filesystem = os.statvfs(root)
    free = filesystem.f_bavail * filesystem.f_frsize
    metrics = [row for row in corpus['metrics'] if row['slice'].startswith('ami-')]
    result = {'schema_version': 1, 'observed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_commit': pre['source_commit'], 'pre_execution_protocol': protocol_identity,
        'scope': 'Fixed retained qualification-report audit plus current statvfs observation. No repeat of lower-level raw replays, no model/helper/VM/GUI/credential/network call and no completion claim.',
        'historical_verified_observations': {
            'mac_arm64_bounded_public_realtime': public['public_realtime_gate_passed'],
            'mac_arm64_bounded_public_warm_rtf': public['warm_rtf'],
            'actual_model_reloads': public['actual_model_reloads_in_existing_lifecycle_test'],
            'fixed_native_public_pairs_byte_exact': corpus['all_full_valid_row_wrapper_comparisons_byte_exact'],
            'q8_degradation_bounds_all_at_most_0_02': all(row['q8_minus_fp32_absolute_der'] <= 0.02 for row in corpus['metrics']),
            'same_host_bounded_pnola_cli_resolution': pnola['automatic_packaged_resolution_gates_passed']},
        'unmet_gates': {
            'original_per_slice_accuracy': {'status': 'failed', 'limit_der': 0.20, 'ami_metrics': metrics,
                'original_nemo_shared_der': original['original_shared_der'],
                'next_dependency': 'Product-appropriate acoustic boundary review and separate development/fresh validation; no policy selected by completed fixed contrasts. Preserve original failures.'},
            'human_word_time_attribution': {'status': 'deferred_by_user', 'annotators': 0, 'reviewed_words': 0,
                'human_gold_admitted': False, 'instruction': 'just document it for now',
                'next_dependency': 'Actual independent human annotation, review and adjudication; do not request again or fabricate gold.'},
            'physical_pnola_gui_mic_playback_ipc': {'status': 'not_established',
                'next_dependency': 'Accessible physical interactive test host and actual capture/display/playback observations. CLI parity is separate.'},
            'fresh_machine_complete_app': {'status': 'not_established', 'guest_os_ready_qualified': False,
                'data_access_passed': False, 'private_app_transfer': 'skipped_by_user',
                'next_dependency': 'Operational fresh receiver; local disposable guest also needs original storage admission and independently qualified guest access/provisioning.'},
            'windows_production_realtime': {'status': 'failed', 'functional_lifecycle_gates_passed': True,
                'rtfs': [row['rtf'] for row in windows['cases']],
                'next_dependency': 'New justified candidate with exact native/public, observer/control and original platform/package qualification. Do not repeat failed unchanged experiments.'},
            'independent_unseen_meeting_validation': {'status': 'not_established',
                'next_dependency': 'Separately frozen licensed recordings, demonstrable training exclusion and independent human labels; current corpus exposure limits remain.'},
            'advertised_platform_release_and_public_signing': {'status': 'not_established',
                'next_dependency': 'Actual platform/backend delivery and signing/notarization receipts; ad-hoc same-host construction is insufficient.'}},
        'current_vm_storage_observation': {'available_bytes': free, 'available_gib': free / 1024**3,
            'original_floors': storage, 'start_admissible': free >= storage['start_minimum'],
            'running_floor_satisfied': free >= storage['running_minimum'],
            'limits': 'Point-in-time filesystem observation only. No reserve, clone, disk read/hash, VM or cleanup was attempted.'},
        'full_goal_achieved': False, 'main_promotion_allowed': False,
        'new_model_helper_vm_or_gui_calls': 0, 'input_identities': pre['inputs']}
    for name, row in pre['inputs'].items():
        capture(Path(row['path']), row['identity'])
    capture(args.protocol, protocol_identity)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'full_goal_achieved': False,
        'main_promotion_allowed': False, 'vm_free_gib': free / 1024**3, 'unmet_gates': len(result['unmet_gates'])}))


if __name__ == '__main__':
    main()
