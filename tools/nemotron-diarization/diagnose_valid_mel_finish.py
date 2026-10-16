"""One preregistered candidate V3 finish run; excluded calibration and fixed silence only."""
import json
from pathlib import Path
import subprocess

import numpy as np

from artifacts import digest
from frontend_admission import require_linked_sdk, run_probe
from valid_mel_finish_metrics import audit_outputs


def descriptor(path):
    return {'bytes': path.stat().st_size, 'sha256': digest(path)}


def main():
    tools = Path(__file__).resolve().parent
    repo = tools.parent.parent
    protocol_path = tools / 'valid-mel-finish.protocol.json'
    p = json.loads(protocol_path.read_text())
    for path, expected in p['bound_paths'].items():
        if descriptor(Path(path)) != expected:
            raise ValueError('candidate input/construction/source differs: ' + path)
    for name, expected in p['tools'].items():
        if descriptor(tools / name) != expected:
            raise ValueError('candidate tool differs: ' + name)
    audit = json.loads(Path(p['materialized_audit_path']).read_text())
    for name, expected in p['materialized_compiler_headers'].items():
        actual = Path(audit['compiled_tree_path']) / name
        if audit['files'].get(name) != expected or descriptor(actual) != expected:
            raise ValueError('actual compiler header differs from audited materialized source: ' + name)
    source = Path(p['candidate_source_path'])
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip() != p['native_base_revision']:
        raise ValueError('candidate native source base differs')
    delta = subprocess.check_output(['git', 'diff', '--binary', '--no-ext-diff', '--ignore-submodules=all'], cwd=source)
    if delta != (tools / 'patches/v3-finish-valid-mel.patch').read_bytes():
        raise ValueError('candidate has unrelated tracked source changes')
    build = json.loads(Path(p['probe_build_manifest_path']).read_text())
    if (build['returncode'] != 0 or build['probe_executed'] is not False
            or build['source'] != p['tools']['valid_mel_finish_probe.cpp']
            or build['binary'] != p['bound_paths'][p['probe_path']]
            or build['argv'] != p['probe_build_command']):
        raise ValueError('candidate helper construction binding differs')
    load = subprocess.check_output(['/usr/bin/otool', '-l', p['probe_path']], text=True)
    lines = load.splitlines()
    rpaths = [lines[i + 2].strip().removeprefix('path ').split(' (offset ')[0]
              for i, line in enumerate(lines) if line.strip() == 'cmd LC_RPATH']
    require_linked_sdk(Path(p['sdk_path']), p['sdk_path'], rpaths)
    output = Path(p['output_path'])
    if output.exists():
        raise ValueError('refusing prior candidate execution evidence overwrite')
    output.mkdir()
    (output / 'protocol.json').write_bytes(protocol_path.read_bytes())
    (output / 'helper-build.json').write_text(json.dumps(build, indent=2) + '\n')
    (output / 'helper-load-commands.txt').write_text(load)
    argv = [p['probe_path'], p['model_path'], p['audio_path'], p['mel_path'], str(output / 'probe')]
    stdout = run_probe(argv, output, 'candidate', {'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'})
    expected = {'fixed_candidate_finish_regression_completed': True, 'stream_sessions': 28,
                'direct_graph_calls': 4, 'boundary_cases': 21, 'tolerance': 0.0001}
    if json.loads(stdout) != expected:
        raise ValueError('candidate completion/count summary differs')
    replay = audit_outputs(output / 'probe')
    mel = np.fromfile(p['mel_path'], dtype='<f4').reshape(112, 128)[:111]
    for left in [0, 1]:
        label = 'direct-left' + str(left)
        for stage, rows in [('first', mel[:104]), ('final', mel[72 - left * 8:])]:
            if (output / 'probe' / (label + '-' + stage + '.mel.f32')).read_bytes() != rows.tobytes():
                raise ValueError('candidate direct Mel input differs')
    record = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
              'protocol_sha256': digest(protocol_path), **expected, **replay,
              'candidate_finish_executed': True, 'neural_model_instantiated': True,
              'stream_frontend_and_aosc_executed': True, 'hf_or_original_nemo_executed': False,
              'target_ami_executed': False, 'der_scoring_performed': False,
              'production_runtime_changed': False, 'original_gate_status_changed': False,
              'release_qualified': False, 'limits': p['limitations']}
    record['files'] = {path.relative_to(output).as_posix(): descriptor(path)
                       for path in output.rglob('*') if path.is_file()}
    (output / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(expected))


if __name__ == '__main__':
    main()
