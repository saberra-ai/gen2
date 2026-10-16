"""One short, sampled run of the exact qualified portable CPU executable.

This diagnostic changes no SDK, thread count, preset or release criterion.
Sampling must work on a model-free control before neural admission.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import struct
import subprocess
import threading
import time
import wave

from artifacts import acquire
from build_valid_mel_candidate import descriptor
from macos_loaded_libraries import audit_loaded_libraries
from macos_portable_delivery import admit_delivery
from package_macos import verify
from qualification_storage import StorageGuard, atomic_json
from qualify_macos_portable_runtime import CONTROL, policy
from qualify_valid_mel_pnola import execute_guarded
from sdk_archive import inventory, restore_archive


CONTROL_SOURCE = '#include <unistd.h>\nint main(void) { sleep(10); return 0; }\n'


def traced_pid(text):
    values = set(re.findall(r'^dyld\[(\d+)\]: <[A-Fa-f0-9-]+> /', text, re.MULTILINE))
    if len(values) > 1:
        raise ValueError('multiple traced processes; sampling target is ambiguous')
    return int(next(iter(values))) if values else None


def sample_target(stderr, executable, output, result, stop, *, duration):
    """Observe the traced target, never a sandbox/time launcher or a guessed PID."""
    try:
        deadline = time.monotonic() + 20
        pid = None
        target_uuid = None
        while time.monotonic() < deadline and not stop.is_set():
            trace = stderr.read_text(errors='replace')
            pid = traced_pid(trace)
            images = re.findall(r'^dyld\[' + str(pid) + r'\]: <([A-Fa-f0-9-]+)> '
                                + re.escape(str(executable)) + r'$', trace, re.MULTILINE)
            if pid is not None and len(set(images)) == 1:
                target_uuid = images[0]
                break
            stop.wait(.1)
        if pid is None or target_uuid is None:
            raise ValueError('no live traced target available for sampling')
        identity = subprocess.check_output(
            ['/bin/ps', '-ww', '-p', str(pid), '-o', 'comm='], text=True, timeout=5).strip()
        if identity != str(executable):
            raise ValueError('live process executable differs from traced target: ' + identity)
        argv = ['/usr/bin/sample', str(pid), str(duration), '10', '-mayDie', '-file', str(output)]
        result.update(pid=pid, executable=identity, target_dyld_uuid=target_uuid, argv=argv)
        observed = subprocess.run(argv, capture_output=True, text=True, timeout=duration + 15)
        result.update(returncode=observed.returncode, stdout=observed.stdout, stderr=observed.stderr)
        if observed.returncode != 0 or not output.is_file():
            raise ValueError('sampler failed')
        text = output.read_text()
        if not re.search(r'^Process:\s+.*\[' + str(pid) + r'\]', text, re.MULTILINE):
            raise ValueError('sample report process identity differs')
        # macOS redacts Path and Binary Images paths. Bind the report to the
        # exact live PID/executable above and its observed dyld UUID instead.
        image = re.findall(r'^\s*0x[0-9a-fA-F]+ -\s*0x[0-9a-fA-F]+\s+\+?'
                           + re.escape(executable.name) + r' \([^\r\n]*\) <'
                           + re.escape(target_uuid) + r'>[^\r\n]*$', text, re.MULTILINE)
        if len(image) != 1:
            raise ValueError('sample report target image UUID differs')
        result['target_binary_image'] = image[0]
        result['report'] = descriptor(output)
    except Exception as error:
        result['error'] = repr(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['protocol', 'construction-zip', 'construction-metadata', 'receiver-zip', 'receiver-metadata', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        parser.error('actual Apple Silicon Mac required')
    if args.output.exists() or args.output.is_symlink():
        parser.error('fresh output required')
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    output = args.output.resolve()
    deliveries = [args.construction_zip, args.construction_metadata, args.receiver_zip, args.receiver_metadata]
    if output.is_relative_to(repo) or any(p.resolve().is_relative_to(output) for p in deliveries):
        parser.error('output must be separate from checkout and delivery inputs')
    protocol_bytes = args.protocol.read_bytes()
    p = json.loads(protocol_bytes)

    def frozen():
        if args.protocol.read_bytes() != protocol_bytes:
            raise ValueError('profile protocol changed')
        for group, root in [('tools', tools), ('repository_files', repo)]:
            for name, expected in p[group].items():
                if descriptor(root / name) != expected:
                    raise ValueError('frozen profile source differs: ' + name)

    frozen()
    output.mkdir(parents=True)
    (output / 'protocol.json').write_bytes(protocol_bytes)
    r = {'schema_version': 1, 'source_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        'protocol': descriptor(args.protocol), 'profile_completed': False,
        'model_execution_admitted': False, 'release_qualified': False, 'commands': [],
        'sampling': {}, 'scope': p['scope'], 'hypotheses': p['hypotheses']}
    persist = lambda: atomic_json(output / 'qualification.json', r)
    guard = None

    def run(name, argv, profile, *, executable=None, sample_seconds=0):
        frozen()
        guard.before_model()
        if name == 'native':
            r['model_execution_admitted'] = True
        env = {'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'HOME': os.environ['HOME'], 'LC_ALL': 'C'}
        command = {'name': name, 'argv': ['/usr/bin/sandbox-exec', '-f', str(profile), *map(str, argv)],
                   'cwd': str(output), 'environment': env}
        r['commands'].append(command)
        persist()
        stderr_path = output / (name + '.stderr.log')
        stop, observer, sampled = threading.Event(), None, None
        started = time.monotonic()
        try:
            with (output / (name + '.stdout.log')).open('xb') as stdout, stderr_path.open('xb') as stderr:
                if sample_seconds:
                    sampled = {}
                    observer = threading.Thread(target=sample_target,
                        args=(stderr_path, executable, output / (name + '.sample.txt'), sampled, stop),
                        kwargs={'duration': sample_seconds})
                    observer.start()
                execute_guarded(command['argv'], cwd=output, environment=env, stdout=stdout, stderr=stderr,
                    guard=guard, record=command, persist=persist, timeout=p['command_timeout_seconds'])
        finally:
            stop.set()
            if observer:
                observer.join(timeout=55)
                if observer.is_alive():
                    raise RuntimeError('sampler observer failed to terminate')
                r['sampling'][name] = sampled
            command['elapsed_seconds'] = time.monotonic() - started
            persist()
        if command['returncode'] != 0 or (sample_seconds and 'error' in r['sampling'][name]):
            raise ValueError('diagnostic command/sampling failed: ' + name)

    try:
        guard = StorageGuard(output, **p['storage'])
        r['deliveries'] = {}
        for name in ['construction', 'receiver']:
            r['deliveries'][name] = admit_delivery(getattr(args, name + '_zip'),
                getattr(args, name + '_metadata'), p[name], output / name)
        construction, receiver = output / 'construction', output / 'receiver'
        package, work = output / 'work/nemo', output / 'work'
        work.mkdir()
        restore_archive(construction / 'portable-cpu-runtime.tar.gz',
            json.loads((construction / 'portable-cpu-runtime-files.json').read_text()), package)
        manifest = verify(package)
        atomic_json(output / 'package.json', manifest)
        helpers = work / 'bin'
        helper_manifest = json.loads((receiver / 'receiver-helpers-files.json').read_text())
        restore_archive(receiver / helper_manifest['archive'], helper_manifest, helpers)
        if inventory(helpers) != p['helpers']:
            raise ValueError('actual qualified helper bytes differ')
        assets = work / 'assets'
        assets.mkdir()
        denied = [repo, construction, receiver, output / 'control.c', '/opt/homebrew',
                  Path.home() / '.cargo', Path.home() / '.rustup', *[f.resolve() for f in deliveries]]
        runtime = output / 'runtime.sb'
        runtime.write_text(policy(denied_reads=denied, readonly=[work, repo, construction, receiver]))
        compile_policy = output / 'compile.sb'
        compile_policy.write_text(policy(denied_reads=[assets], readonly=[repo, construction, receiver, package, helpers]))
        host = {'cpu': subprocess.check_output(['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
                'cpu_counts': subprocess.check_output(['/usr/sbin/sysctl', 'hw.ncpu', 'hw.logicalcpu', 'hw.physicalcpu'], text=True),
                'platform': platform.platform(), 'os': subprocess.check_output(['/usr/bin/sw_vers'], text=True),
                'load_before': os.getloadavg()}
        if host['cpu'] != p['expected_receiver_cpu']:
            raise ValueError('receiver CPU differs')
        atomic_json(output / 'host.json', host)
        control_source, control = output / 'control.c', work / 'control'
        control_source.write_text(CONTROL_SOURCE)
        run('compile-control', ['/usr/bin/clang', '-arch', 'arm64', '-march=armv8.2-a+dotprod+fp16',
            '-mmacosx-version-min=26.0', control_source, '-o', control], compile_policy)
        run('sampling-control', ['/usr/bin/env', 'DYLD_PRINT_LIBRARIES=1', control], runtime,
            executable=control, sample_seconds=1)
        r['control_images'] = audit_loaded_libraries((output / 'sampling-control.stderr.log').read_text(),
            package, manifest, control, False)
        r['acquisition_attempts'] = []
        for entry in p['assets']:
            guard.before_model()
            row = {'expected': entry}
            r['acquisition_attempts'].append(row)
            persist()
            acquire(entry, assets)
            row['verified'] = descriptor(assets / entry['name'])
            persist()
        wav_path = assets / 'profile-12s.wav'
        with wave.open(str(assets / 'ami_en2002d_2132.wav'), 'rb') as source:
            if (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getnframes()) != (1, 2, 16000, 960000):
                raise ValueError('source recording geometry differs')
            pcm = source.readframes(p['profile_samples'])
        with wave.open(str(wav_path), 'wb') as target:
            target.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            target.writeframes(pcm)
        pcm_float = b''.join(struct.pack('<f', n[0] / 32768) for n in struct.iter_unpack('<h', pcm))
        if hashlib.sha256(pcm_float).hexdigest() != p['pcm_f32le_sha256']:
            raise ValueError('cropped PCM differs')
        r['crop'] = descriptor(wav_path)
        run('runtime-controls', [os.sys.executable, '-c', CONTROL,
            json.dumps({'reads': [str(repo / 'Cargo.toml'), str(receiver / 'qualification.json')],
                        'writes': [str(wav_path)]})], runtime)
        executable = helpers / 'native-stream'
        result_path = output / 'native.json'
        run('native', ['/usr/bin/time', '-l', '/usr/bin/env', 'DYLD_PRINT_LIBRARIES=1', executable,
            assets / p['model_name'], wav_path, '1', '-1', result_path], runtime,
            executable=executable, sample_seconds=p['sample_seconds'])
        r['loaded_images'] = audit_loaded_libraries((output / 'native.stderr.log').read_text(),
            package, manifest, executable)
        result = json.loads(result_path.read_text())
        raw = Path(str(result_path) + '.f32').read_bytes()
        values = [n[0] for n in struct.iter_unpack('<f', raw)]
        if (result['audio_samples'] != p['profile_samples'] or result['native_frame_count'] != p['profile_samples'] // 160
                or result['retained_rows'] != p['profile_samples'] // 160
                or result['geometry'] != p['geometry'] or result['runtime_version'] != p['runtime_version']
                or result['input_pcm_f32le_sha256'] != p['pcm_f32le_sha256']
                or len(raw) != (p['profile_samples'] // 160) * 8 * 4
                or not all(math.isfinite(v) and 0 <= v <= 1 for v in values)):
            raise ValueError('profile result geometry/values differ')
        r['native_rtf'] = result['elapsed_micros'] / 1e6 / (p['profile_samples'] / 16000)
        r['load_after'] = os.getloadavg()
        r['profile_completed'] = True
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
