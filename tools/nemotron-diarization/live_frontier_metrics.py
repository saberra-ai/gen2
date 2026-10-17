"""Describe a saved CLI observer trace; never infer word or GUI label latency."""
import math
import statistics


def distribution(values):
    ordered = sorted(values)
    return {'samples': len(values), 'minimum': ordered[0], 'median': statistics.median(ordered),
            'p95_nearest_rank': ordered[math.ceil(.95 * len(ordered)) - 1], 'maximum': ordered[-1]}


def frontier_observations(revisions, duration):
    if not math.isfinite(duration) or duration <= 0 or not 2 <= len(revisions) <= 17:
        raise ValueError('bounded nonempty complete observer trace required')
    rows, gaps, previous_ages = [], [], []
    previous_elapsed = previous_processed = -1.
    for expected_revision, revision in enumerate(revisions, 1):
        elapsed, processed = revision['elapsed_seconds'], revision['processed_audio_seconds']
        if (type(revision['revision']) is not int or revision['revision'] != expected_revision
                or not math.isfinite(elapsed) or not math.isfinite(processed)
                or not 0 <= elapsed or not 0 < processed <= duration
                or elapsed <= previous_elapsed or processed <= previous_processed
                or processed > elapsed + .020 + 1e-9):
            raise ValueError('invalid, missing or nonmonotonic CLI observation')
        age = elapsed - processed
        rows.append({'revision': expected_revision, 'observed_elapsed_seconds': elapsed,
                     'processed_audio_seconds': processed, 'signed_frontier_age_seconds': age})
        if expected_revision > 1:
            gaps.append(elapsed - previous_elapsed)
            previous_ages.append(elapsed - previous_processed)
        previous_elapsed, previous_processed = elapsed, processed
    return {'observations': rows, 'first_observed_revision_elapsed_seconds': rows[0]['observed_elapsed_seconds'],
            'frontier_age_at_observation_seconds': distribution([row['signed_frontier_age_seconds'] for row in rows]),
            'inter_observation_seconds': distribution(gaps),
            'previous_observed_frontier_age_at_next_observation_seconds': distribution(previous_ages),
            'quantile_method': 'nearest-rank descriptive sample p95; not a population estimate',
            'nominal_feed_block_seconds': .020,
            'scope': 'CLI observation clock minus reported processed-audio frontier; full consecutive revision trace only',
            'actual_speaker_or_word_label_latency_measured': False, 'gui_delivery_measured': False,
            'physical_microphone_measured': False, 'finalization_delay_measured': False,
            'input_queue_native_compute_and_poll_delay_separated': False,
            'stable_prefix_or_label_finality_inferred': False}
