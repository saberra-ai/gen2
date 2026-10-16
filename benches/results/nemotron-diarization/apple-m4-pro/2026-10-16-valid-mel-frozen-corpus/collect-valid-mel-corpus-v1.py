"""Recompute every saved current-profile corpus score; retain all quality failures."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0'); cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'; sys.path.insert(0, str(tools))
from build_valid_mel_candidate import descriptor
from corpus import pcm_header
from macos_loaded_libraries import audit_loaded_libraries
from qualify_corpus import validate_corpus
from qualify_valid_mel_corpus import admit_reference
from score_valid_mel_corpus import score_slice
from sdk_archive import inventory

base = cache / 'valid-mel-corpus-runtime-v1'
out = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-frozen-corpus'
assert not out.exists(); read = lambda path: json.loads(path.read_text())
p = read(base / 'protocol.json'); r = read(base / 'qualification.json'); commands = read(base / 'commands.json')
assert r['all_fixed_pairs_completed'] and not r['release_qualified'] and len(r['results']) == 16
assert len(commands) == 32 and all(c['returncode'] == 0 for c in commands)
assert r['storage']['released'] and r['storage']['reserve_reported_allocated_bytes'] == 256 * 1024**2
assert descriptor(base / 'protocol.json') == r['protocol'] == descriptor(tools / 'valid-mel-corpus.protocol.json')
work = Path(p['work_path']); assert inventory(work) == p['inventory']
for key, root, prefix in [('tools', tools, 'tools/nemotron-diarization/'), ('repository_files', repo, '')]:
    for name, expected in p[key].items():
        data = subprocess.check_output(['git', 'show', r['source_commit'] + ':' + prefix + name], cwd=repo)
        assert descriptor(root / name) == {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()} == expected, name
for name, expected in p['preparation_records'].items():
    assert descriptor(Path(name)) == expected, name
policy = '(version 1)\n(allow default)\n(deny network*)\n(deny file-write* (subpath ' + json.dumps(str(work)) + '))\n(deny file-read* ' + ' '.join('(subpath ' + json.dumps(path) + ')' for path in p['denied_paths']) + ')\n'
assert (base / 'runtime.sb').read_text() == policy
manifest = validate_corpus(work / 'corpus'); legacy = read(tools / 'corpus.protocol.json'); prior = read(Path(p['legacy_receipt_path']))
assert p['legacy_scoring'] == legacy['scoring'] and p['quality_gates'] == legacy['scoring']['per_slice_gates']
recomputed = []; image_counts = []; index = 0
for item in manifest['slices']:
    audio = (work / 'corpus' / item['wav']).read_bytes(); offset, samples = pcm_header(audio[:65536], len(audio))
    assert samples == item['audio_samples']
    pcm = np.frombuffer(audio, dtype='<i2', offset=offset, count=samples).astype(np.float32) / np.float32(32768)
    pcm_sha = hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest()
    fp32, metadata = admit_reference(item, work, legacy, prior)
    assert metadata['wav_sha256'] == manifest['files'][item['wav']]['sha256']
    for backend in ['cpu', 'metal']:
        name = item['id'] + '-' + backend; native_path = base / (name + '-native.json'); public_raw = base / (name + '-public.f32')
        public = read(base / (name + '-public.json')); native = read(native_path)
        assert json.loads((base / (name + '-public.stdout.log')).read_text()) == public
        score = score_slice(item, public, native, public_raw.read_bytes(), Path(str(native_path) + '.f32').read_bytes(),
                            fp32, metadata, read(work / 'corpus' / item['reference']), pcm_sha)
        score.update(all_slice_gates_passed=all(score['gates'].values()), input_pcm_f32le_sha256=pcm_sha,
                     oracle_array=descriptor(work / 'oracle' / (item['id'] + '-fp32.npy')),
                     oracle_metadata=descriptor(work / 'oracle' / (item['id'] + '-fp32.json')))
        assert score == read(base / (name + '-score.json')) == r['results'][index]; index += 1
        recomputed.append(score)
        manifest_package = read(work / backend / 'nemo/package.json')
        for kind in ['native', 'public']:
            command = commands[len(image_counts)]
            assert command['name'] == name + '-' + kind and command['cwd'] == str(base)
            assert command['environment'] == {'PATH': '/usr/bin:/bin', 'LC_ALL': 'C', 'TMPDIR': str(base / 'tmp') + '/'}
            executable = work / backend / 'bin' / (kind + '-stream')
            args = ([str(work / 'assets/Nemotron-3-Diarization.q8_0.gguf'), str(work / 'corpus' / item['wav']), '1', '-1' if backend == 'cpu' else '0', str(native_path)]
                    if kind == 'native' else (['--metal'] if backend == 'metal' else []) + [str(work / 'assets'), str(work / backend / 'nemo/lib/libnemo_speech_asr_c.dylib'), str(work / 'corpus' / item['wav']), '1', str(public_raw)])
            assert command['argv'] == ['/usr/bin/sandbox-exec', '-f', str(base / 'runtime.sb'), '/usr/bin/env', 'DYLD_PRINT_LIBRARIES=1', str(executable), *args]
            images = audit_loaded_libraries((base / (name + '-' + kind + '.stderr.log')).read_text(), work / backend / 'nemo', manifest_package, executable)
            assert images == read(base / (name + '-' + kind + '.images.json'))
            assert images['private_library_count'] == (92 if backend == 'cpu' else 93)
            image_counts.append(images['private_library_count'])
assert recomputed == read(base / 'scores.json')
assert r['all_slice_quality_gates_passed'] == all(s['all_slice_gates_passed'] for s in recomputed)
out.mkdir()
for path in base.iterdir():
    if path.is_file() and path.suffix in {'.json', '.log', '.sb', '.f32'}:
        shutil.copy2(path, out / path.name)
for name, expected in p['tools'].items():
    target = out / 'execution-tools' / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(tools / name, target)
shutil.copy2(cache / 'prepare-valid-mel-corpus-v1.py', out / 'prepare-valid-mel-corpus-v1.py')
shutil.copy2(cache / 'valid-mel-corpus-inputs-v1/preparation.json', out / 'preparation.json')
shutil.copy2(cache / 'valid-mel-corpus-inputs-v1/commands.json', out / 'preparation-commands.json')
shutil.copy2(cache / 'valid-mel-corpus-runtime-v1.launch.log', out / 'launch.log')
shutil.copy2(work / 'corpus/manifest.json', out / 'corpus-manifest.json')
for item in manifest['slices']:
    target = out / 'reused-oracle-metadata'; target.mkdir(exist_ok=True)
    shutil.copy2(work / 'oracle' / (item['id'] + '-fp32.json'), target / (item['id'] + '-fp32.json'))
shutil.copy2(__file__, out / Path(__file__).name)
record = {'schema_version': 1, 'all_fixed_pairs_completed': True, 'all_slice_quality_gates_passed': r['all_slice_quality_gates_passed'],
          'release_qualified': False, 'source_commit': r['source_commit'], 'protocol': r['protocol'],
          'scope': 'Actual current-profile CPU/Metal requalification on unchanged original frozen corpus, exact cached original official CPU FP32 references and unchanged quality gates.',
          'fixed_neural_child_processes': 32, 'independent_native_public_comparisons': 16, 'cached_fp32_arrays_reused': 8,
          'fresh_hf_or_original_nemo_calls': 0, 'private_native_image_records': sum(image_counts),
          'all_full_valid_row_wrapper_comparisons_passed': all(s['wrapper_parity']['passed'] for s in recomputed),
          'all_full_valid_row_wrapper_comparisons_byte_exact': all(s['wrapper_parity']['maximum_probability_error'] == 0 and s['wrapper_parity']['maximum_turn_error_seconds'] == 0 for s in recomputed),
          'passed_slice_backend_pairs': sum(s['all_slice_gates_passed'] for s in recomputed),
          'failed_slice_backend_pairs': [{'slice': s['slice'], 'backend': s['backend'], 'gates': s['gates']} for s in recomputed if not s['all_slice_gates_passed']],
          'metrics': [{'slice': s['slice'], 'backend': s['backend'], 'fp32_der': s['fp32']['der'], 'q8_shared_der': s['q8_shared_threshold']['der'],
                       'q8_production_der': s['q8_production_default']['der'], 'q8_minus_fp32_absolute_der': s['q8_minus_fp32_absolute_der']} for s in recomputed],
          'new_neural_or_native_helper_calls_during_collection': 0,
          'limits': ['Every newly emitted native/public probability is retained. Exact original cached FP32 array identities remain hash-described; reused array binaries, weights, SDKs, helpers and large audio remain local.',
                     'Original failed quality outcomes remain unchanged and unwaived. This reuses the same previously observed fixed corpus, not a fresh independent heldout or guaranteed checkpoint-unseen benchmark.',
                     'No original NeMo model execution, tuning from these results, diverse long-audio quality, GUI/microphone, clean performance, fresh-machine, public signing/notarization or release promotion.']}
record['files'] = {str(path.relative_to(out)): descriptor(path) for path in sorted(out.rglob('*')) if path.is_file()}
(out / 'receiver-qualification.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({'members': len(record['files']) + 1, 'passed_pairs': record['passed_slice_backend_pairs'], 'all_quality_passed': record['all_slice_quality_gates_passed'], 'failed': record['failed_slice_backend_pairs']}))
