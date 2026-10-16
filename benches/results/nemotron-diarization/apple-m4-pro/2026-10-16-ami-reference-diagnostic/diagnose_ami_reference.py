"""Post-failure reference diagnosis; never replaces frozen qualification gates."""
import argparse
import hashlib
import importlib.util
import json
import logging
import math
from pathlib import Path
import sys
import tempfile
import types
import urllib.request

import numpy as np

from compare import segments
from score import der


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def crop_reference(text, recording, start, end):
    result = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 10 or fields[0] != 'SPEAKER' or fields[1] != recording:
            raise ValueError('unexpected reference RTTM record')
        onset, duration = float(fields[3]), float(fields[4])
        if not math.isfinite(onset) or not math.isfinite(duration) or onset < 0 or duration <= 0:
            raise ValueError('invalid reference interval')
        left, right = max(onset, start), min(onset + duration, end)
        if left < right:
            result.append({'start': left - start, 'end': right - start, 'speaker': fields[7]})
    if not result:
        raise ValueError('empty selected reference')
    return result


def official_engine(path, expected):
    import scipy

    if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
        raise ValueError('independent NeMo scorer differs from protocol')
    if np.__version__ != expected['numpy_version'] or scipy.__version__ != expected['scipy_version']:
        raise ValueError('independent scoring dependencies differ from diagnostic pins')
    # The unmodified scorer's only NeMo import is its logger. No model, torch,
    # Lhotse carrier or other NeMo package behavior is used by this invocation.
    nemo = types.ModuleType('nemo')
    utils = types.ModuleType('nemo.utils')
    utils.logging = logging
    nemo.utils = utils
    sys.modules['nemo'] = nemo
    sys.modules['nemo.utils'] = utils
    spec = importlib.util.spec_from_file_location('pinned_nemo_md_eval', path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    return engine


def official_der(engine, reference, hypothesis, duration):
    with tempfile.TemporaryDirectory(prefix='gen2-ami-score-') as temporary:
        root = Path(temporary)
        for name, entries in [('ref', reference), ('hyp', hypothesis)]:
            (root / (name + '.rttm')).write_text(''.join(
                f"SPEAKER ES2004a 1 {s['start']:.17g} {s['end'] - s['start']:.17g} <NA> <NA> {s['speaker']} <NA> <NA>\n"
                for s in entries))
        uem_path = root / 'full.uem'
        uem_path.write_text(f'ES2004a 1 0 {duration:.17g}\n')
        refs, hyps = {}, {}
        engine.get_rttm_data(refs, str(root / 'ref.rttm'))
        engine.get_rttm_data(hyps, str(root / 'hyp.rttm'))
        _, stats = engine.evaluate(refs, hyps, engine.get_uem_data(str(uem_path)),
                                   collar=0.0, opt_1=False, verbose=False)
    denominator = stats['SCORED_SPEAKER']
    return {'der': (stats['MISSED_SPEAKER'] + stats['FALARM_SPEAKER'] + stats['SPEAKER_ERROR']) / denominator,
            'missed_speaker_seconds': stats['MISSED_SPEAKER'],
            'false_alarm_speaker_seconds': stats['FALARM_SPEAKER'],
            'confusion_speaker_seconds': stats['SPEAKER_ERROR'],
            'reference_speaker_seconds': denominator, 'collar_seconds': 0,
            'overlap_included': True, 'scoring_duration_seconds': duration}


def calibrate_engine(engine):
    segment = lambda start, end, speaker: {'start': start, 'end': end, 'speaker': speaker}
    overlap = [segment(0, .6, 'A'), segment(.4, 1, 'B')]
    cases = [
        ('overlap_and_mapping', overlap, [segment(0, .6, '2'), segment(.4, 1, '1')], 0),
        ('missed_overlap', overlap, [segment(0, .6, '1')], .5),
        ('false_alarm', [segment(0, .4, 'A')], [segment(0, .6, '1')], .5),
        ('global_confusion', [segment(0, .5, 'A'), segment(.5, 1, 'B')], [segment(0, 1, '1')], .5),
    ]
    results = []
    for name, ref, hyp, expected in cases:
        official = official_der(engine, ref, hyp, 1)
        frame = der(ref, hyp, 1)
        if abs(official['der'] - expected) > 1e-9 or abs(frame['der'] - expected) > 1e-9:
            raise ValueError('independent scorer calibration failed: ' + name)
        results.append({'case': name, 'expected_der': expected, 'official': official, 'frame': frame})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--md-eval', type=Path, required=True)
    args = parser.parse_args()
    tools = Path(__file__).resolve().parent
    protocol_path = tools / 'ami-reference-diagnostic.protocol.json'
    protocol = json.loads(protocol_path.read_text())
    if args.output.exists():
        parser.error('refusing to overwrite diagnostic evidence')
    if digest(Path(__file__)) != protocol['adapter_sha256']:
        raise ValueError('diagnostic adapter changed after protocol freeze')
    engine = official_engine(args.md_eval, protocol['independent_scorer'])
    calibrations = calibrate_engine(engine)
    for name, expected in protocol['scoring']['tool_sha256'].items():
        if digest(tools / name) != expected:
            raise ValueError('diagnostic scoring tool changed: ' + name)
    if digest(args.corpus / 'manifest.json') != protocol['original_corpus_manifest_sha256']:
        raise ValueError('original corpus manifest changed')
    manifest = json.loads((args.corpus / 'manifest.json').read_text())
    for name, expected in protocol['original_inputs'].items():
        path = args.evidence / name
        if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
            raise ValueError('original inference output changed: ' + name)
    source = protocol['reference']
    # Bound the download before parsing or writing it; this is a tiny pinned RTTM.
    with urllib.request.urlopen(source['url'], timeout=60) as response:
        raw = response.read(source['bytes'] + 1)
    if len(raw) != source['bytes'] or hashlib.sha256(raw).hexdigest() != source['sha256']:
        raise ValueError('published reference differs from protocol')
    selection = protocol['selection']
    start, end = selection['source_interval_seconds']
    reference = crop_reference(raw.decode('utf-8'), selection['recording'], start, end)
    results = []
    for channel in selection['channels']:
        identifier = 'ami-ES2004a-' + channel
        item = next(x for x in manifest['slices'] if x['id'] == identifier)
        reference_path = args.corpus / item['reference']
        if digest(reference_path) != manifest['files'][item['reference']]['sha256']:
            raise ValueError('original reference changed')
        original = json.loads(reference_path.read_text())
        for backend in ['cpu', 'metal']:
            prefix = args.evidence / (identifier + '-' + backend)
            previous = json.loads(Path(str(prefix) + '-score.json').read_text())
            public = json.loads(Path(str(prefix) + '-gen2.json').read_text())
            fp32 = np.load(args.evidence / (identifier + '-fp32.npy'), allow_pickle=False)
            q8 = np.fromfile(Path(str(prefix) + '-gen2.f32'), dtype='<f4').reshape(-1, 8)
            hypotheses = {
                'fp32_shared_threshold': (segments(fp32), previous['fp32']),
                'q8_shared_threshold': (segments(q8), previous['q8_shared_threshold']),
                'q8_production_default': ([{'start': x['start_seconds'], 'end': x['end_seconds'],
                                          'speaker': x['speaker']} for x in public['turns']], previous['q8_production_default']),
            }
            for profile, (hypothesis, expected) in hypotheses.items():
                old = der(original, hypothesis, end - start)
                if old != expected:
                    raise ValueError('original metric did not reproduce: ' + identifier + '/' + profile)
                new = der(reference, hypothesis, end - start)
                official_old = official_der(engine, original, hypothesis, end - start)
                official_new = official_der(engine, reference, hypothesis, end - start)
                results.append({'slice': identifier, 'backend': backend, 'profile': profile,
                                'original_transcriber_reference': old, 'published_forced_reference': new,
                                'official_original_transcriber_reference': official_old,
                                'official_published_forced_reference': official_new,
                                'der_change': new['der'] - old['der']})
    args.output.mkdir(parents=True)
    (args.output / 'ES2004a.rttm').write_bytes(raw)
    (args.output / 'cropped-reference.json').write_text(json.dumps(reference, indent=2) + '\n')
    (args.output / 'protocol.json').write_bytes(protocol_path.read_bytes())
    (args.output / 'md_eval.py').write_bytes(args.md_eval.read_bytes())
    (args.output / 'result.json').write_text(json.dumps({
        'release_qualified': False, 'original_gate_status_changed': False,
        'scope': protocol['purpose'], 'protocol_sha256': digest(protocol_path),
        'script_sha256': digest(Path(__file__)), 'reference_sha256': source['sha256'],
        'scoring': '10ms midpoint, zero collar, overlap included; not exact official continuous scorer',
        'independent_scoring': protocol['independent_scorer'],
        'independent_calibrations': calibrations,
        'license': source['license'], 'reference_url': source['url'],
        'results': results, 'limitations': protocol['reporting'],
    }, indent=2) + '\n')
    print(json.dumps({'diagnostic_completed': True, 'original_gate_status_changed': False}))


if __name__ == '__main__':
    main()
