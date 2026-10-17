"""Validate a fixed endpoint-paced public trace without inventing label latency."""
import math
import uuid

from live_frontier_metrics import distribution


def nonnegative_integer(value):
    if type(value) is not int or value < 0:
        raise ValueError('nonnegative integer telemetry required')
    return value


def paced_timing(value):
    if (value['complete'] is not True or value['repeat_finish_cached'] is not True
            or value['accepted_samples'] != 1280640 or value['audio_duration_seconds'] != 80.04
            or value['native_frame_count'] != 8004 or len(value['pushes']) != 4002):
        raise ValueError('fixed completed 80.04s paced trace required')
    if not isinstance(value['session_id'], str):
        raise ValueError('session UUID required')
    uuid.UUID(value['session_id'])
    scheduling, admission = [], []
    previous_accepted = total_busy = 0
    for index, push in enumerate(value['pushes'], 1):
        endpoint = nonnegative_integer(push['endpoint_samples'])
        nominal = nonnegative_integer(push['nominal_endpoint_micros'])
        ready = nonnegative_integer(push['ready_micros'])
        accepted = nonnegative_integer(push['accepted_micros'])
        total_busy += nonnegative_integer(push['busy_retries'])
        if endpoint != index * 320 or nominal != index * 20000 or not previous_accepted <= ready <= accepted or ready < nominal:
            raise ValueError('missing, early or nonmonotonic paced input')
        scheduling.append((ready - nominal) / 1e6)
        admission.append((accepted - ready) / 1e6)
        previous_accepted = accepted
    requested = nonnegative_integer(value['finish_requested_micros'])
    returned = nonnegative_integer(value['finish_returned_micros'])
    compute = nonnegative_integer(value['native_compute_micros'])
    if (not previous_accepted <= requested <= returned or value['elapsed_micros'] != returned
            or value['finish_micros'] != returned - requested or compute > returned):
        raise ValueError('invalid public finish/compute clock')
    observations = value['observations']
    if not 16 <= len(observations) <= 17:
        raise ValueError('complete bounded provisional/final observation trace required')
    previous_elapsed = previous_frontier = previous_compute = -1
    ages, gaps, previous_ages = [], [], []
    for index, observation in enumerate(observations):
        final = index == len(observations) - 1
        observed = nonnegative_integer(observation['observed_micros'])
        before = nonnegative_integer(observation['poll_started_micros'])
        accepted = nonnegative_integer(observation['accepted_samples'])
        spent = nonnegative_integer(observation['native_compute_micros'])
        frontier = observation['processed_audio_seconds']
        revision = observation['revision']
        if (observation['session_id'] != value['session_id'] or observation['complete'] is not final
                or type(revision) is not int or revision != (17 if final else index + 1)
                or type(frontier) not in (int, float) or not math.isfinite(frontier)
                or not 0 <= before <= observed or observed <= previous_elapsed
                or not previous_frontier < frontier <= min(80.04, accepted / 16000., observed / 1e6) + 1e-9
                or not 0 <= accepted <= 1280640 or not previous_compute <= spent <= compute):
            raise ValueError('invalid, missing or impossible public observation')
        if ((final and (observed < returned or abs(frontier - 80.04) > 1e-9
                        or observation['turns'] != value['turns'] or spent != compute))
                or (not final and observed > requested)):
            raise ValueError('final delivery or provisional clock differs')
        previous_start = -1.
        if not isinstance(observation['turns'], list) or len(observation['turns']) > 100000:
            raise ValueError('bounded observed turns required')
        for turn in observation['turns']:
            start, end, speaker = turn['start_seconds'], turn['end_seconds'], turn['speaker']
            if (any(type(x) not in (int, float) or not math.isfinite(x) for x in [start, end])
                    or not 0 <= start <= end <= frontier + 1e-4 or start < previous_start
                    or type(speaker) is not int or not 1 <= speaker <= 8):
                raise ValueError('invalid observed provisional turns')
            previous_start = start
        if not final:
            ages.append(observed / 1e6 - frontier)
            if index:
                gaps.append((observed - previous_elapsed) / 1e6)
                previous_ages.append(observed / 1e6 - previous_frontier)
        previous_elapsed, previous_frontier, previous_compute = observed, frontier, spent
    polls = nonnegative_integer(value['poll_count'])
    poll_micros = nonnegative_integer(value['poll_micros'])
    if polls != 4003 + total_busy or poll_micros > observations[-1]['observed_micros']:
        raise ValueError('snapshot lookup accounting differs')
    return {'scope': 'Endpoint-paced public API replay; timing components overlap and must not be summed.',
        'pushes': 4002, 'provisional_observations': len(observations) - 1,
        'final_revision': 17, 'unobserved_revision_before_final': 17 - observations[-2]['revision'] - 1,
        'scheduling_overrun_seconds': distribution(scheduling),
        'normalization_hashing_and_admission_seconds': distribution(admission),
        'busy_retries': total_busy, 'snapshot_lookup_calls': polls, 'snapshot_lookup_wall_seconds': poll_micros / 1e6,
        'native_compute_seconds': compute / 1e6, 'paced_public_wall_seconds': returned / 1e6,
        'public_finish_drain_flush_delivery_seconds': (returned - requested) / 1e6,
        'final_observed_after_finish_return_seconds': (observations[-1]['observed_micros'] - returned) / 1e6,
        'provisional_frontier_age_seconds': distribution(ages), 'provisional_observation_gap_seconds': distribution(gaps),
        'previous_observed_frontier_age_at_next_provisional_seconds': distribution(previous_ages),
        'cold_load_seconds': nonnegative_integer(value['cold_load_micros']) / 1e6,
        'stream_open_seconds': nonnegative_integer(value['stream_open_micros']) / 1e6,
        'actual_speaker_or_word_label_latency_measured': False, 'gui_delivery_measured': False,
        'physical_microphone_measured': False, 'native_queue_lookahead_and_publication_delay_separated': False,
        'stable_prefix_or_label_finality_inferred': False}
