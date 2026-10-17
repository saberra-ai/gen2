"""Four fresh owned-pool CPU streams at 30 minutes/one hour on the named M4.

Reuse independently qualified helper payloads; preserve every original long
acceptance rule. Neither helper can read the other helper's output tensors.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

from artifacts import matches
from build_valid_mel_candidate import descriptor
from compare_macos_cpu_pools import model_free_policy
from macos_loaded_libraries import audit_loaded_libraries
from macos_portable_delivery import admit_delivery
from package_macos import verify
from pooled_long_metrics import audit_case
from pooled_runtime import POOLED_PROFILE, load_profile, profile_provenance
from qualification_storage import StorageGuard, atomic_json
from qualify_macos_portable_runtime import CONTROL, policy
from qualify_pooled_native import result_policy
from qualify_valid_mel_pnola import execute_guarded
from sdk_archive import inventory, restore_archive, write_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['protocol', 'producer-zip', 'producer-metadata', 'native-zip', 'native-metadata',
                 'public-helpers', 'public-helpers-manifest', 'public-qualification', 'assets', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        parser.error('actual arm64 Mac required')
    if args.output.exists() or args.output.is_symlink():
        parser.error('fresh long-session output required')
    tools = Path(__file__).resolve().parent
    repo, output = tools.parents[1], args.output.resolve()
    inputs = [getattr(args, name).resolve(strict=True) for name in ['producer_zip', 'producer_metadata',
              'native_zip', 'native_metadata', 'public_helpers', 'public_helpers_manifest', 'public_qualification', 'assets']]
    if output.is_relative_to(repo) or any(path.is_relative_to(output) for path in inputs):
        parser.error('output must be separate from source and frozen inputs')
    protocol_bytes = args.protocol.read_bytes()
    p = json.loads(protocol_bytes)
    lock_path, lock = load_profile(POOLED_PROFILE)

    def frozen():
        original = json.loads((tools / 'valid-mel-cpu-long.protocol.json').read_text())
        if (args.protocol.read_bytes() != protocol_bytes
                or p['runtime_provenance'] != profile_provenance(lock_path, lock)
                or p['acceptance'] != original['acceptance'] or p['cases'] != original['cases']
                or p['command_timeout_seconds'] != original['command_timeout_seconds']):
            raise ValueError('long-session protocol/profile changed')
        for group, root in [('tools', tools), ('repository_files', repo)]:
            for name, expected in p[group].items():
                if descriptor(root / name) != expected:
                    raise ValueError('frozen long-session source differs: ' + name)
        for name in ['public_helpers', 'public_helpers_manifest', 'public_qualification']:
            if descriptor(getattr(args, name)) != p[name]:
                raise ValueError('qualified public helper input differs: ' + name)

    frozen()
    output.mkdir(parents=True)
    (output / 'protocol.json').write_bytes(protocol_bytes)
    r = {'schema_version': 1, 'source_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(), 'protocol': descriptor(args.protocol),
        'pooled_30m_1h_passed': False, 'neural_commands_started': [], 'commands': [],
        'loaded_images': {}, 'comparisons': {}, 'release_qualified': False, 'limits': p['limits']}
    persist = lambda: atomic_json(output / 'qualification.json', r)
    guard = None
    home, temporary = output / 'home', output / 'tmp'
    home.mkdir()
    temporary.mkdir()
    env = {'PATH': '/usr/bin:/bin', 'HOME': str(home), 'TMPDIR': str(temporary) + '/', 'LC_ALL': 'C'}

    def run(name, argv, sandbox, *, inference=False, audit=False):
        frozen()
        guard.before_model()
        if inference:
            r['neural_commands_started'].append(name)
        child = ['/usr/bin/time', '-l', '/usr/bin/env', 'DYLD_PRINT_LIBRARIES=1', *map(str, argv)] if audit else list(map(str, argv))
        row = {'name': name, 'argv': ['/usr/bin/sandbox-exec', '-f', str(sandbox), *child],
               'cwd': str(output), 'environment': env, 'inference_admitted': inference, 'load_before': os.getloadavg()}
        r['commands'].append(row)
        persist()
        with (output / (name + '.stdout.log')).open('xb') as stdout, (output / (name + '.stderr.log')).open('xb') as stderr:
            execute_guarded(row['argv'], cwd=output, environment=env, stdout=stdout, stderr=stderr,
                guard=guard, record=row, persist=persist, timeout=p['command_timeout_seconds'])
        row['load_after'] = os.getloadavg()
        persist()
        if row['returncode'] != 0:
            raise ValueError('long-session command failed: ' + name)
        if audit:
            r['loaded_images'][name] = audit_loaded_libraries((output / (name + '.stderr.log')).read_text(),
                                                             package, manifest, argv[0])
            persist()
        return (output / (name + '.stdout.log')).read_text()

    try:
        guard = StorageGuard(output, **p['storage'])
        producer, native_root = output / 'producer', output / 'native-delivery'
        r['producer_artifact'] = admit_delivery(args.producer_zip, args.producer_metadata, p['producer'], producer)
        r['native_artifact'] = admit_delivery(args.native_zip, args.native_metadata, p['native'], native_root)
        construction = json.loads((producer / 'qualification.json').read_text())
        native_q = json.loads((native_root / 'qualification.json').read_text())
        public_q = json.loads(args.public_qualification.read_text())
        if (construction['construction_and_negative_c_api_controls_passed'] is not True
                or construction['valid_models_loaded'] or construction['frontend_or_neural_calls']
                or native_q['native_parity_passed'] is not True
                or native_q['deliveries']['construction']['id'] != p['producer']['id']
                or public_q['pooled_public_runtime_passed'] is not True
                or public_q['source_commit'] != p['compiled_public_source_commit']
                or public_q['producer_artifact']['id'] != p['producer']['id']
                or public_q['lifecycle_tests_passed'] != p['lifecycle_tests']):
            raise ValueError('completed pooled construction/native/public lifecycle qualifications required')
        work = output / 'work'
        work.mkdir()
        package, assets, binaries = work / 'nemo', work / 'assets', work / 'bin'
        assets.mkdir()
        binaries.mkdir()
        package_files = json.loads((producer / 'pooled-cpu-runtime-files.json').read_text())
        restore_archive(producer / package_files['archive'], package_files, package)
        manifest = verify(package)
        if manifest['runtime_profile'] != POOLED_PROFILE:
            raise ValueError('owned-pool package required')
        atomic_json(output / 'package.json', manifest)
        host = {'cpu': subprocess.check_output(['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
                'platform': platform.platform(), 'cpu_counts': subprocess.check_output(
                    ['/usr/sbin/sysctl', 'hw.ncpu', 'hw.logicalcpu', 'hw.physicalcpu'], text=True), 'load_before': os.getloadavg()}
        atomic_json(output / 'host.json', host)
        if host['cpu'] != p['expected_cpu']:
            raise ValueError('named M4 host required; no M1 long/performance claim')
        native_helpers, public_helpers = output / 'native-helpers', output / 'public-helpers'
        native_files = json.loads((native_root / 'pooled-native-helpers-files.json').read_text())
        public_files = json.loads(args.public_helpers_manifest.read_text())
        if {k: v for k, v in public_files.items() if k != 'files'} != public_q['helper_archive']:
            raise ValueError('public helper archive is not the qualified archive')
        restore_archive(native_root / native_files['archive'], native_files, native_helpers)
        restore_archive(args.public_helpers, public_files, public_helpers)
        for name, source in [('native-stream', native_helpers / 'native-stream'),
                             ('public-stream', public_helpers / 'public-stream'), ('load-probe', public_helpers / 'load-probe')]:
            shutil.copy2(source, binaries / name)
            if descriptor(binaries / name) != descriptor(source):
                raise ValueError('qualified helper copy differs')
        r['helper_archive'] = write_archive(binaries, output, 'pooled-long-helpers')
        r['acquisitions'] = {}
        for entry in p['assets']:
            source = args.assets.resolve() / entry['name']
            if not matches(source, entry):
                raise ValueError('cached long-session asset differs: ' + entry['name'])
            shutil.copy2(source, assets / entry['name'])
            if not matches(assets / entry['name'], entry):
                raise ValueError('copied long-session asset differs')
            r['acquisitions'][entry['name']] = descriptor(assets / entry['name'])
        protected = {'package': inventory(package), 'binaries': inventory(binaries), 'assets': inventory(assets),
                     'producer': inventory(producer), 'native_delivery': inventory(native_root),
                     'native_helpers': inventory(native_helpers), 'public_helpers': inventory(public_helpers)}
        atomic_json(output / 'protected-inputs.json', protected)

        def unchanged():
            frozen()
            for key, root in [('package', package), ('binaries', binaries), ('assets', assets), ('producer', producer),
                              ('native_delivery', native_root), ('native_helpers', native_helpers), ('public_helpers', public_helpers)]:
                if inventory(root) != protected[key]:
                    raise ValueError('long-session protected input changed: ' + key)

        results = output / 'results'
        results.mkdir()
        denied = [repo, producer, native_root, native_helpers, public_helpers, '/opt/homebrew',
                  Path.home() / '.cargo', Path.home() / '.rustup', *[path.parent for path in inputs if path.is_file()],
                  args.assets.resolve()]
        readonly = [work, *denied]
        runtime, premodel = output / 'runtime.sb', output / 'premodel.sb'
        runtime.write_text(result_policy(policy(denied_reads=denied, readonly=readonly), results))
        premodel.write_text(model_free_policy(runtime.read_text()))
        probes = [output / ('denied.' + suffix) for suffix in ['nemo', 'gguf', 'safetensors', 'onnx', 'npy', 'f32']]
        for probe in probes:
            probe.write_bytes(b'not a model')
            if probe.read_bytes() != b'not a model':
                raise ValueError('weight-format positive control failed')
        run('model-free-controls', ['/usr/bin/python3', '-c', CONTROL,
            json.dumps({'reads': list(map(str, probes)) + [str(assets / p['model_name'])], 'writes': []})], premodel)
        prior = results / 'prior-output-control.txt'
        prior.write_bytes(b'not a model')
        read_controls = [repo / 'Cargo.toml', producer / 'protocol.json', native_root / 'results/native-60s.json',
                         args.public_helpers_manifest.resolve(), native_helpers / 'native-stream', prior]
        for path in read_controls:
            path.read_bytes()
        run('runtime-controls', ['/usr/bin/python3', '-c', CONTROL,
            json.dumps({'reads': list(map(str, read_controls)), 'writes': [str(assets / p['wav_name'])]})], runtime)
        version = run('version-only', [binaries / 'load-probe', package / 'lib/libnemo_speech_asr_c.dylib',
            lock['runtime']['reported_version']], premodel, audit=True)
        if version.strip() != lock['runtime']['reported_version']:
            raise ValueError('pooled C ABI identity differs')
        for case in p['cases']:
            unchanged()
            native_path = results / (case['name'] + '-native.json')
            public_path = results / (case['name'] + '-public.json')
            run(case['name'] + '-native', [binaries / 'native-stream', assets / p['model_name'],
                assets / p['wav_name'], str(case['repeats']), '-1', native_path], runtime, inference=True, audit=True)
            unchanged()
            public = json.loads(run(case['name'] + '-public', [binaries / 'public-stream', assets,
                package / 'lib/libnemo_speech_asr_c.dylib', assets / p['wav_name'], str(case['repeats']),
                results / (case['name'] + '-public.f32')], runtime, inference=True, audit=True))
            atomic_json(public_path, public)
            native = json.loads(native_path.read_text())
            r['comparisons'][case['name']] = audit_case(public, native, Path(str(native_path) + '.f32').read_bytes(),
                (results / (case['name'] + '-public.f32')).read_bytes(),
                (output / (case['name'] + '-native.stderr.log')).read_text(),
                (output / (case['name'] + '-public.stderr.log')).read_text(), case)
            persist()
        unchanged()
        r['load_after'] = os.getloadavg()
        r['pooled_30m_1h_passed'] = True
    except BaseException as error:
        r['error'] = repr(error)
        raise
    finally:
        if guard:
            guard.release()
            r['storage'] = guard.record
        persist()


if __name__ == '__main__':
    main()
