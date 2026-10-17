"""Fixed segmentation-operation ablations on saved ES2011a development output."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def descriptor(path):
    raw = Path(path).read_bytes()
    return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    args = parser.parse_args()
    raw = args.protocol.read_bytes()
    p = json.loads(raw)
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    output = Path(p['output_path'])
    if output.exists() or output.is_symlink():
        parser.error('fresh saved-output diagnostic destination required')

    def frozen():
        if args.protocol.read_bytes() != raw:
            raise ValueError('segmentation protocol changed')
        for name,expected in p['tools'].items():
            if descriptor(tools / name) != expected:
                raise ValueError('segmentation tool changed: '+name)
        for path,expected in p['inputs'].items():
            if descriptor(path) != expected:
                raise ValueError('segmentation saved input changed: '+path)

    # Admit every imported local module before executing any local module code.
    frozen()
    import numpy as np
    from ami_prefix_metrics import load_readonly_scorer
    from diagnose_ami_reference import calibrate_engine, official_der
    from frozen_git_sources import verify_git_sources
    from score import der
    from segmentation_ablation import CONTRASTS, compare_native_turns, native_segments

    source = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    if raw != subprocess.check_output(['git','show',source+':tools/nemotron-diarization/development-segmentation.protocol.json'],cwd=repo):
        raise ValueError('execution protocol must be committed before saved-output scoring')
    verify_git_sources(repo,source,{'tools/nemotron-diarization/'+name:value for name,value in p['tools'].items()})
    verify_git_sources(repo,source,{str(Path(name).relative_to(repo)):value for name,value in p['inputs'].items()
                                   if Path(name).is_relative_to(repo)})
    if CONTRASTS != p['contrasts']:
        raise ValueError('fixed segmentation operations differ')
    prior = json.loads(Path(p['prior_qualification']).read_text())
    prior_collection = json.loads(Path(p['prior_collection']).read_text())
    prior_manifest = json.loads((Path(p['prior_collection']).parent / 'collection.json').read_text())
    if prior['diagnostic_completed'] is not True or prior['release_qualified'] is not False:
        raise ValueError('completed development diagnostic required')
    work = Path(p['prior_work'])
    root = Path(p['prior_results'])
    engine = load_readonly_scorer(work / 'scoring/md_eval.py',p['independent_scorer'])
    calibrations = calibrate_engine(engine)
    references = {name:json.loads((work / 'references' / (name+'.json')).read_text())
                  for name in p['references']}
    result = {'schema_version':1,'source_commit':source,'diagnostic_completed':False,'model_calls':0,
              'meeting':'ES2011a','scoring_uem_seconds':[0,300],'training_exposed_development':True,
              'original_gate_status_changed':False,'production_policy_selected':False,'release_qualified':False,
              'calibrations':calibrations,'baseline_parity':{},'scores':{},'hypotheses':{},'deltas':{},'limits':p['limits']}
    for channel in p['channels']:
        for profile in p['profiles']:
            name = channel+'-'+profile
            data_path = root / (name+'.json.f32')
            if descriptor(root / (name+'.json')) != prior_manifest['files'][name+'.json']:
                raise ValueError('native JSON differs from committed prior receipt')
            if descriptor(data_path) != prior_collection['outside_git_probability_arrays'][data_path.name]:
                raise ValueError('raw probabilities differ from independently collected neural evidence')
            native = json.loads((root / (name+'.json')).read_text())
            if (native['audio_samples'] != 4800000 or native['native_frame_count'] != 30000
                    or native['retained_first_frame'] != 0 or native['retained_rows'] != 30000
                    or native['input_repeats'] != 1):
                raise ValueError('fixed full development timeline required')
            values = np.frombuffer(data_path.read_bytes(),dtype='<f4').reshape(30000,8)
            result['hypotheses'][name] = {mode:native_segments(values,config) for mode,config in CONTRASTS.items()}
            result['baseline_parity'][name] = compare_native_turns(result['hypotheses'][name]['native_default'],native['turns'])
            result['scores'][name] = {}
            result['deltas'][name] = {}
            for policy,reference in references.items():
                scores = {mode:{'frame':der(reference,turns,300.),'official':official_der(engine,reference,turns,300.)}
                          for mode,turns in result['hypotheses'][name].items()}
                if scores['native_default'] != prior['scores'][name][policy]['production']:
                    raise ValueError('native default development scores did not reproduce')
                result['scores'][name][policy] = scores
                result['deltas'][name][policy] = {mode:{scorer:{metric:value[scorer][metric]-scores['native_default'][scorer][metric]
                    for metric in ['der','missed_speaker_seconds','false_alarm_speaker_seconds','confusion_speaker_seconds']}
                    for scorer in ['frame','official']} for mode,value in scores.items()}
    frozen()
    result['diagnostic_completed'] = True
    output.mkdir()
    (output / 'protocol.json').write_bytes(raw)
    (output / 'qualification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'diagnostic_completed':True,'model_calls':0,'cells':len(result['scores']),
                      'contrasts_per_cell':len(CONTRASTS),'production_policy_selected':False}))


if __name__ == '__main__':
    main()
