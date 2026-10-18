"""Eight fixed corpus slices through qualified bounded CPU native/public helpers.

No model conversion, helper compilation, acquisition or policy selection. Old
FP32 oracles, audio, references and quality gates remain unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def descriptor(path):
    raw = Path(path).read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--admit-only', action='store_true')
    args = parser.parse_args()
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    raw_protocol = args.protocol.read_bytes()
    p = json.loads(raw_protocol)
    for records, root in [(p['tools'], tools), (p['repository_files'], repo)]:
        for name, wanted in records.items():
            if descriptor(root / name) != wanted:
                raise ValueError('bounded corpus source differs before imports: ' + name)
    import numpy as np
    from bounded_corpus_metrics import score_slice
    from bounded_cpu_metrics import VERSION
    from compare_macos_cpu_pools import model_free_policy
    from corpus import pcm_header
    from frozen_git_sources import verify_git_sources
    from macos_child_environment import sandboxed_command
    from macos_loaded_libraries import audit_loaded_libraries
    from package_macos import verify
    from qualification_storage import StorageGuard, atomic_json
    from qualify_corpus import validate_corpus
    from qualify_macos_portable_runtime import CONTROL
    from qualify_valid_mel_corpus import admit_reference
    from qualify_valid_mel_pnola import execute_guarded
    from sdk_archive import inventory

    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise ValueError('physical arm64 Mac required')
    work = Path(p['work_path'])
    output = Path(p['output_path'] + ('-admission' if args.admit_only else ''))
    if output.exists() or output.is_symlink() or output.is_relative_to(work) or work.is_relative_to(output):
        raise ValueError('fresh disjoint bounded corpus output required')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    pins = {**{'tools/nemotron-diarization/' + name: value for name, value in p['tools'].items()},
            **p['repository_files']}
    verify_git_sources(repo, source, pins)
    if raw_protocol != subprocess.check_output(['git', 'show', source + ':tools/nemotron-diarization/bounded-corpus.protocol.json'], cwd=repo):
        raise ValueError('committed bounded corpus protocol required')

    def frozen():
        if args.protocol.read_bytes() != raw_protocol:
            raise ValueError('bounded corpus protocol changed')
        for records, root in [(p['tools'], tools), (p['repository_files'], repo)]:
            for name, wanted in records.items():
                if descriptor(root / name) != wanted:
                    raise ValueError('bounded corpus source changed: ' + name)
        if inventory(work) != p['inventory']:
            raise ValueError('bounded corpus staged inputs changed')
        for name, wanted in p['preparation_records'].items():
            if descriptor(Path(name)) != wanted:
                raise ValueError('bounded corpus preparation changed: ' + name)
        environment = {'python': platform.python_version(), 'numpy': np.__version__,
                       'interpreter': descriptor(Path(sys.executable).resolve(strict=True))}
        if environment != p['scoring_environment']:
            raise ValueError('original corpus scoring environment required')

    frozen()
    legacy_protocol = json.loads((tools / 'corpus.protocol.json').read_text())
    legacy_receipt = json.loads(Path(p['legacy_receipt_path']).read_text())
    manifest = validate_corpus(work / 'corpus')
    if (descriptor(work / 'corpus/manifest.json')['sha256'] != legacy_protocol['corpus_manifest_sha256']
            or [item['id'] for item in manifest['slices']] != legacy_protocol['slice_ids']
            or p['quality_gates'] != legacy_protocol['scoring']['per_slice_gates']
            or p['legacy_scoring'] != legacy_protocol['scoring']
            or p['storage'] != {'start_minimum': 5 * 1024**3, 'running_minimum': 1024**3, 'reserve_bytes': 256 * 1024**2}
            or p['expected_fresh_child_processes'] != 16 or p['expected_scored_pairs'] != 8):
        raise ValueError('original corpus, scoring, gates and storage required')
    references = {item['id']: admit_reference(item, work, legacy_protocol, legacy_receipt)
                  for item in manifest['slices']}
    package = work / 'cpu/nemo'
    package_manifest = verify(package)
    if package_manifest['runtime_profile'] != 'gen2-valid-mel-pool-cpu-v2' or package_manifest['backend'] != 'cpu':
        raise ValueError('exact bounded CPU package required')
    reuse = p['helper_reuse']
    public_build = json.loads(Path(reuse['public_build_record']).read_text())
    verify_git_sources(repo, reuse['public_source_commit'], public_build['source_pins'])
    for name, wanted in public_build['source_pins'].items():
        if descriptor(repo / name) != wanted:
            raise ValueError('reused public helper source changed: ' + name)
    long = json.loads(Path(reuse['long_qualification']).read_text())
    helpers = json.loads(Path(reuse['helper_archive_manifest']).read_text())
    if (long['bounded_30m_1h_passed'] is not True or not long['storage']['released']
            or {k: v for k, v in helpers.items() if k != 'files'} != long['helper_archive']
            or public_build['model_or_frontend_execution_during_build'] is not False):
        raise ValueError('qualified reused helper construction/runtime required')
    for name in ('public-stream', 'native-stream', 'load-probe'):
        record = inventory(work / 'cpu/bin')[name]
        if record != helpers['files'][name]:
            raise ValueError('reused helper payload changed')
    if inventory(work / 'cpu/bin')['public-stream'] != public_build['compiled_helpers']['public-stream']:
        raise ValueError('public helper differs from actual source-bound compilation')
    frozen()
    output.mkdir()
    (output / 'protocol.json').write_bytes(raw_protocol)
    commands, scores, images = [], [], {}
    r = {'schema_version': 1, 'source_commit': source, 'admit_only': args.admit_only,
         'runtime_profile': 'gen2-valid-mel-pool-cpu-v2', 'runtime_version': VERSION,
         'input_admission_passed': True, 'all_fixed_pairs_completed': False,
         'all_slice_quality_gates_passed': False, 'release_qualified': False,
         'cached_original_fp32_oracles_reused': 8, 'fresh_hf_or_original_nemo_calls': 0,
         'protocol': descriptor(args.protocol), 'commands': commands, 'results': scores, 'loaded_images': images,
         'neural_commands_started': []}
    guard = None
    persist = lambda: atomic_json(output / 'qualification.json', r)
    try:
        guard = StorageGuard(output, **p['storage'])
        if args.admit_only:
            return
        cpu = subprocess.check_output(['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
        if cpu != p['expected_cpu']:
            raise ValueError('named physical M4 required')
        atomic_json(output / 'host.json', {'cpu': cpu, 'platform': platform.platform()})
        results, home, temporary = [output / name for name in ('results', 'home', 'tmp')]
        for path in (results, home, temporary):
            path.mkdir()
        environment = {'PATH': '/usr/bin:/bin', 'LC_ALL': 'C', 'HOME': str(home), 'TMPDIR': str(temporary) + '/'}
        policy = output / 'runtime.sb'
        deny = [*map(Path, p['denied_paths']), work / 'oracle',
                *[work / 'corpus' / name for name in manifest['files'] if not name.endswith('.wav')],
                work / 'corpus/manifest.json']
        text = '(version 1)\n(allow default)\n(deny network*)\n'
        text += '(deny file-write* (subpath ' + json.dumps(str(work)) + '))\n'
        text += '(deny file-read* ' + ' '.join('(subpath ' + json.dumps(str(path)) + ')' for path in deny) + ')\n'
        text += '(deny file-read-data (subpath ' + json.dumps(str(results)) + '))\n'
        policy.write_text(text)
        premodel = output / 'premodel.sb'
        premodel.write_text(model_free_policy(text))

        def run(name, argv, *, trace=False, inference=False, sandbox=None):
            frozen()
            guard.before_model()
            child = ['/usr/bin/time', '-l', '/usr/bin/env', 'DYLD_PRINT_LIBRARIES=1', *map(str, argv)] if trace else argv
            actual, variables = sandboxed_command(child, sandbox or policy, environment)
            row = {'name': name, 'argv': actual, 'environment': variables, 'cwd': str(output),
                   'inference_admitted': inference}
            commands.append(row)
            if inference:
                r['neural_commands_started'].append(name)
            persist()
            started = time.monotonic()
            try:
                with (output / (name + '.stdout.log')).open('xb') as stdout, (output / (name + '.stderr.log')).open('xb') as stderr:
                    execute_guarded(actual, cwd=output, environment=variables, stdout=stdout, stderr=stderr,
                        guard=guard, record=row, persist=persist, timeout=p['command_timeout_seconds'])
            finally:
                row['elapsed_seconds'] = time.monotonic() - started
                persist()
            if row['returncode'] != 0:
                raise ValueError('bounded corpus command failed: ' + name)
            if trace:
                images[name] = audit_loaded_libraries((output / (name + '.stderr.log')).read_text(),
                                                      package, package_manifest, argv[0])
                persist()
            return (output / (name + '.stdout.log')).read_text()

        prior = results / 'prior-output-control.txt'
        prior.write_bytes(b'not an oracle')
        reads = [work / 'oracle' / (manifest['slices'][0]['id'] + '-fp32.npy'),
                 work / 'corpus' / manifest['slices'][0]['reference'], repo / 'Cargo.toml', prior]
        for path in reads:
            path.read_bytes()
        run('runtime-controls', ['/usr/bin/python3', '-c', CONTROL,
            json.dumps({'reads': list(map(str, reads)), 'writes': [str(work / 'assets/Nemotron-3-Diarization.q8_0.gguf')]})])
        run('premodel-controls', ['/usr/bin/python3', '-c', CONTROL,
            json.dumps({'reads': [str(work / 'assets/Nemotron-3-Diarization.q8_0.gguf')], 'writes': []})], sandbox=premodel)
        reported = run('version-only', [work / 'cpu/bin/load-probe', package / 'lib' / package_manifest['entry_library'], VERSION], trace=True, sandbox=premodel)
        if reported.strip() != VERSION:
            raise ValueError('actual bounded runtime version differs')
        for item in manifest['slices']:
            wav = work / 'corpus' / item['wav']
            audio = wav.read_bytes()
            offset, samples = pcm_header(audio[:65536], len(audio))
            if samples != item['audio_samples']:
                raise ValueError('frozen PCM sample count differs')
            pcm = np.frombuffer(audio, dtype='<i2', offset=offset, count=samples).astype(np.float32) / np.float32(32768)
            pcm_sha = hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest()
            fp32, metadata = references[item['id']]
            if metadata['wav_sha256'] != manifest['files'][item['wav']]['sha256']:
                raise ValueError('original oracle WAV binding differs')
            name = item['id'] + '-cpu'
            native_path = results / (name + '-native.json')
            public_raw = results / (name + '-public.f32')
            run(name + '-native', [work / 'cpu/bin/native-stream', work / 'assets/Nemotron-3-Diarization.q8_0.gguf',
                wav, '1', '-1', native_path], trace=True, inference=True)
            public = json.loads(run(name + '-public', [work / 'cpu/bin/public-stream', work / 'assets',
                package / 'lib' / package_manifest['entry_library'], wav, '1', public_raw], trace=True, inference=True))
            atomic_json(results / (name + '-public.json'), public)
            score = score_slice(item, public, json.loads(native_path.read_text()), public_raw.read_bytes(),
                Path(str(native_path) + '.f32').read_bytes(), fp32, metadata,
                json.loads((work / 'corpus' / item['reference']).read_text()), pcm_sha)
            score.update(all_slice_gates_passed=all(score['gates'].values()), input_pcm_f32le_sha256=pcm_sha,
                oracle_array=descriptor(work / 'oracle' / (item['id'] + '-fp32.npy')),
                oracle_metadata=descriptor(work / 'oracle' / (item['id'] + '-fp32.json')))
            atomic_json(results / (name + '-score.json'), score)
            scores.append(score)
            persist()
        frozen()
        r['all_fixed_pairs_completed'] = len(scores) == 8
        r['all_slice_quality_gates_passed'] = len(scores) == 8 and all(row['all_slice_gates_passed'] for row in scores)
    except BaseException as error:
        r['error'] = repr(error)
        raise
    finally:
        if guard is not None:
            guard.release()
            r['storage'] = guard.record
        persist()


if __name__ == '__main__':
    main()
