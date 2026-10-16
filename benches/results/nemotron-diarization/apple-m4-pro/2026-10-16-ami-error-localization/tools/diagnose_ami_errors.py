"""Localize saved AMI errors with a fixed whole-UEM official speaker mapping.

This post-failure diagnostic never changes inference, references or release gates.
Each partition is exhaustive and additive, including silence and empty bins.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from compare import segments
from diagnose_ami_reference import calibrate_engine, digest, official_der, official_engine


KEYS = ('SCORED_TIME', 'SCORED_SPEAKER', 'MISSED_SPEAKER', 'FALARM_SPEAKER', 'SPEAKER_ERROR')
BINS = {'occupancy': ('silence', 'single', 'overlap'),
        'reference_span': ('silence', 'short_only', 'long_only', 'mixed'),
        'crop_time': ('first_30_seconds', 'remaining_90_seconds')}


def summarize(stats, total_reference):
    row = {key: stats.get(key, 0.0) for key in KEYS}
    error = sum(row[key] for key in KEYS[2:])
    row['der'] = error / row['SCORED_SPEAKER'] if row['SCORED_SPEAKER'] else None
    row['whole_uem_der_contribution'] = error / total_reference if total_reference else None
    return row


def partition(engine, score_segments, reference, mapping, duration):
    """Score disjoint timeline strata using the *same* reference→system map.

    Short means a clipped input reference RTTM span <=1s. Mixed means short
    and long reference spans are simultaneously active, even for one speaker.
    These bins describe reference occupancy, not causal attribution of errors.
    """
    if duration != 120:
        raise ValueError('diagnostic requires the frozen 120-second UEM')
    stats = {axis: {name: {'TYPE': {}} for name in names} for axis, names in BINS.items()}
    total = {'TYPE': {}}
    # Break at 30s even when neither RTTM has an event there. Reference span
    # boundaries are already in the official timeline; include them explicitly
    # to preserve length strata if the engine ever coalesces equal speaker sets.
    boundaries = {30.0, *[row[key] for row in reference for key in ('start', 'end')]}
    for segment in score_segments:
        cuts = sorted({segment['TBEG'], segment['TEND'],
                       *[time for time in boundaries if segment['TBEG'] < time < segment['TEND']]})
        for left, right in zip(cuts, cuts[1:]):
            if right <= left:
                continue
            item = {**segment, 'TBEG': left, 'TEND': right, 'TDUR': right - left}
            midpoint = (left + right) / 2
            active = [row for row in reference if row['start'] <= midpoint < row['end']]
            if set(item['REF']) != {str(row['speaker']) for row in active}:
                raise ValueError('reference span classification differs from official timeline')
            occupancy = ('silence' if not item['REF'] else 'single' if len(item['REF']) == 1 else 'overlap')
            lengths = {row['end'] - row['start'] <= 1.0 for row in active}
            span = ('silence' if not lengths else 'short_only' if lengths == {True}
                    else 'long_only' if lengths == {False} else 'mixed')
            time = 'first_30_seconds' if midpoint < 30 else 'remaining_90_seconds'
            for target in [total, stats['occupancy'][occupancy], stats['reference_span'][span], stats['crop_time'][time]]:
                engine._score_speaker_segments(target, [item], [], mapping, {'REF': {}, 'SYS': {}})
    for axis in stats.values():
        for key in KEYS:
            if abs(sum(row.get(key, 0) for row in axis.values()) - total.get(key, 0)) > 1e-8:
                raise ValueError('partition components are not additive')
    denominator = total.get('SCORED_SPEAKER', 0)
    return {'total': summarize(total, denominator),
            'partitions': {axis: {name: summarize(row, denominator) for name, row in rows.items()}
                           for axis, rows in stats.items()}}


def localize(engine, reference, hypothesis, duration):
    with tempfile.TemporaryDirectory(prefix='gen2-ami-errors-') as temporary:
        root = Path(temporary)
        data = {}
        for name, entries in [('ref', reference), ('hyp', hypothesis)]:
            path = root / (name + '.rttm')
            path.write_text(''.join(
                f"SPEAKER ES2004a 1 {s['start']:.17g} {s['end'] - s['start']:.17g} <NA> <NA> {s['speaker']} <NA> <NA>\n"
                for s in entries))
            data[name] = {}
            engine.get_rttm_data(data[name], str(path))
        ref = data['ref']['ES2004a']['1']
        hyp = data['hyp'].get('ES2004a', {}).get('1', {}).get('SPEAKER', {})
        uem = [{'TBEG': 0.0, 'TEND': duration}]
        full, mapping = engine.score_speaker_diarization('ES2004a', '1', ref['SPEAKER'], hyp,
                                                        [], uem, ref['RTTM'], collar=0, opt_1=False)
        timeline = engine.create_speaker_segs(uem, ref['SPEAKER'], hyp)
        result = partition(engine, timeline, reference, mapping, duration)
        for key in KEYS:
            if abs(result['total'][key] - full[key]) > 1e-8:
                raise ValueError('partition total differs from official whole-UEM score')
        result['global_reference_to_system_mapping'] = mapping
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['evidence', 'reference-diagnostic', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    tools = Path(__file__).resolve().parent
    protocol_path = tools / 'ami-error-localization.protocol.json'
    protocol = json.loads(protocol_path.read_text())
    if args.output.exists():
        parser.error('refusing to overwrite diagnostic evidence')
    for name, expected in protocol['tools'].items():
        if digest(tools / name) != expected:
            raise ValueError('diagnostic tool changed after protocol freeze: ' + name)
    for group, root in [('inference_inputs', args.evidence), ('reference_diagnostic_inputs', args.reference_diagnostic)]:
        for name, expected in protocol[group].items():
            path = root / name
            if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
                raise ValueError('frozen diagnostic input differs: ' + name)
    engine = official_engine(args.reference_diagnostic / 'md_eval.py', protocol['independent_scorer'])
    calibrations = calibrate_engine(engine)
    old = json.loads((args.reference_diagnostic / 'result.json').read_text())
    reference = json.loads((args.reference_diagnostic / 'cropped-reference.json').read_text())
    rows = []
    for channel in protocol['channels']:
        identifier = 'ami-ES2004a-' + channel
        fp32 = np.load(args.evidence / (identifier + '-fp32.npy'), allow_pickle=False)
        for backend in protocol['backends']:
            prefix = args.evidence / (identifier + '-' + backend)
            q8 = np.fromfile(Path(str(prefix) + '-gen2.f32'), dtype='<f4').reshape(-1, 8)
            public = json.loads(Path(str(prefix) + '-gen2.json').read_text())
            hypotheses = {'fp32_shared_threshold': segments(fp32), 'q8_shared_threshold': segments(q8),
                          'q8_production_default': [{'start': x['start_seconds'], 'end': x['end_seconds'],
                                                      'speaker': x['speaker']} for x in public['turns']]}
            for profile, hypothesis in hypotheses.items():
                result = localize(engine, reference, hypothesis, 120)
                prior, = [row for row in old['results'] if row['slice'] == identifier and row['backend'] == backend and row['profile'] == profile]
                official = official_der(engine, reference, hypothesis, 120)
                if official != prior['official_published_forced_reference']:
                    raise ValueError('previous official result did not reproduce')
                if abs(result['total']['der'] - official['der']) > 1e-10:
                    raise ValueError('localized result differs from previous official result')
                rows.append({'slice': identifier, 'backend': backend, 'profile': profile, **result})
    args.output.mkdir(parents=True)
    (args.output / 'protocol.json').write_bytes(protocol_path.read_bytes())
    (args.output / 'result.json').write_text(json.dumps({
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=tools, text=True).strip(),
        'release_qualified': False, 'original_gate_status_changed': False,
        'inference_performed': False, 'reference_changed': False,
        'protocol_sha256': digest(protocol_path), 'purpose': protocol['purpose'],
        'independent_scorer_calibrations': calibrations,
        'mapping_rule': protocol['mapping_rule'], 'bin_definitions': protocol['bin_definitions'],
        'limitations': protocol['limitations'], 'results': rows}, indent=2) + '\n')
    print(json.dumps({'diagnostic_completed': True, 'original_gate_status_changed': False, 'results': len(rows)}))


if __name__ == '__main__':
    main()
