"""Construct one isolated ad-hoc bounded CPU Pnola app; never launch it.

The original private checkout and earlier apps stay read-only. Construction
copies/hashes model payloads but performs no SDK version query or inference.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import plistlib
import re
import shutil
import subprocess
import tempfile


def descriptor(path):
    raw = Path(path).read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['protocol', 'producer-zip', 'producer-metadata', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--admit-only', action='store_true')
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        parser.error('actual arm64 Mac required')
    tools = Path(__file__).resolve().parent
    repo, output = tools.parents[1], args.output.resolve()
    protocol_bytes = args.protocol.read_bytes()
    p = json.loads(protocol_bytes)
    private = Path(p['private_pnola'])
    inputs = [private, args.producer_zip.resolve(strict=True), args.producer_metadata.resolve(strict=True),
              *[Path(name) for name in p['asset_directories']], *[Path(name) for name in p['asset_files']]]
    if (args.output.exists() or args.output.is_symlink() or output.is_relative_to(repo)
            or any(path.is_relative_to(output) or output.is_relative_to(path) for path in inputs)):
        parser.error('fresh separate output outside source/input trees required')
    for group, base in [('tools', tools), ('repository_files', repo),
                        ('vendor_sources', repo / 'vendor/llama-cpp-sys-2/llama.cpp')]:
        for name, value in p[group].items():
            if descriptor(base / name) != value:
                raise ValueError('app construction source changed before imports: ' + name)
    from frozen_git_sources import verify_git_sources
    from macos_portable_delivery import admit_delivery
    from package_macos import verify
    from qualification_storage import StorageGuard, atomic_json
    from qualify_macos_pooled_mixed import verify_vendor_worktree
    from qualify_valid_mel_pnola import execute_guarded
    from sdk_archive import inventory, restore_archive
    from verify_bounded_pnola_app import verify_app

    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    pins = {'tools/nemotron-diarization/' + n: v for n, v in p['tools'].items()}
    pins.update(p['repository_files'])
    verify_git_sources(repo, source, pins)
    vendor = repo / 'vendor/llama-cpp-sys-2/llama.cpp'
    if subprocess.check_output(['git', 'rev-parse', source + ':vendor/llama-cpp-sys-2/llama.cpp'], cwd=repo, text=True).strip() != p['vendor_revision']:
        raise ValueError('app construction vendor gitlink changed')
    verify_git_sources(vendor, p['vendor_revision'], p['vendor_sources'])
    if protocol_bytes != subprocess.check_output(['git', 'show', source + ':tools/nemotron-diarization/macos-bounded-pnola-construction.protocol.json'], cwd=repo):
        raise ValueError('committed app construction protocol required')

    def frozen():
        if args.protocol.read_bytes() != protocol_bytes:
            raise ValueError('app construction protocol changed')
        for group, base in [('tools', tools), ('repository_files', repo), ('private_source_files', private),
                            ('vendor_sources', vendor)]:
            for name, value in p[group].items():
                path = base / name
                if path.is_symlink() or descriptor(path) != value:
                    raise ValueError('app construction frozen source changed: ' + name)
        verify_vendor_worktree(vendor, p['vendor_revision'])
        if inventory(private / 'node_modules') != p['node_modules_inventory']:
            raise ValueError('private npm dependencies changed')
        for name, value in p['asset_directories'].items():
            if inventory(Path(name)) != value:
                raise ValueError('app construction asset directory changed: ' + name)
        for name, value in p['asset_files'].items():
            if descriptor(Path(name)) != value:
                raise ValueError('app construction asset file changed: ' + name)
        for value in p['build_tools'].values():
            if descriptor(Path(value['path'])) != value['file']:
                raise ValueError('app construction build tool changed')
        for value in p['prior_qualifications'].values():
            if descriptor(Path(value['path'])) != value['file']:
                raise ValueError('app construction prerequisite changed')

    frozen()
    output.mkdir(parents=True)
    (output / 'protocol.json').write_bytes(protocol_bytes)
    r = {'schema_version': 1, 'source_commit': source, 'protocol': descriptor(args.protocol),
         'admit_only': args.admit_only, 'deliveries_admitted': False,
         'app_construction_passed': False, 'commands': [], 'apps': {},
         'native_version_queries_or_neural_calls': 0, 'models_loaded_for_inference': False,
         'release_qualified': False, 'limits': p['limits']}
    persist = lambda: atomic_json(output / 'qualification.json', r)
    guard = None

    def run(name, argv, environment, cwd, policy):
        frozen()
        guard.before_model()
        row = {'name': name, 'argv': ['/usr/bin/sandbox-exec', '-f', str(policy), *map(str, argv)],
               'cwd': str(cwd), 'environment': environment, 'inference_admitted': False}
        r['commands'].append(row)
        persist()
        with (output / (name + '.stdout.log')).open('xb') as stdout, (output / (name + '.stderr.log')).open('xb') as stderr:
            execute_guarded(row['argv'], cwd=cwd, environment=environment, stdout=stdout, stderr=stderr,
                guard=guard, record=row, persist=persist, timeout=p['command_timeout_seconds'])
        if row['returncode'] != 0:
            raise ValueError('app construction command failed: ' + name)

    try:
        guard = StorageGuard(output, **p['storage'])
        producer = output / 'producer'
        r['producer_artifact'] = admit_delivery(args.producer_zip, args.producer_metadata, p['producer'], producer)
        construction = json.loads((producer / 'qualification.json').read_text())
        if (construction['construction_and_negative_c_api_controls_passed'] is not True
                or construction['valid_models_loaded'] or construction['frontend_or_neural_calls']
                or construction['source_commit'] != p['producer']['source_commit']):
            raise ValueError('successful model-free bounded producer required')
        r['deliveries_admitted'] = True
        if args.admit_only:
            return
        for role, flag in [('long', 'bounded_30m_1h_passed'), ('mixed', 'bounded_cpu_mixed_residency_passed')]:
            value = p['prior_qualifications'][role]
            q = json.loads(Path(value['path']).read_text())
            if (q[flag] is not True or q['source_commit'] != value['source_commit']
                    or not q['storage']['released'] or 'error' in q
                    or any(c.get('returncode') != 0 for c in q['commands'])):
                raise ValueError('completed bounded ' + role + ' prerequisite required')
        cpu = subprocess.check_output(['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
        if cpu != p['expected_cpu']:
            raise ValueError('named physical M4 required')
        atomic_json(output / 'host.json', {'cpu': cpu, 'platform': platform.platform()})
        package = output / 'native-package'
        archive = json.loads((producer / 'pooled-cpu-runtime-files.json').read_text())
        if {k: v for k, v in archive.items() if k != 'files'} != construction['package_archive']:
            raise ValueError('qualified bounded package archive differs')
        restore_archive(producer / archive['archive'], archive, package)
        manifest = verify(package)
        if manifest['runtime_profile'] != 'gen2-valid-mel-pool-cpu-v2' or manifest['backend'] != 'cpu':
            raise ValueError('exact bounded CPU package required')
        snapshot = output / 'pnola-source'
        snapshot.mkdir()
        for name in p['private_source_files']:
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(private / name, target)
        temporary = output / 'tmp'
        temporary.mkdir()
        tempfile.tempdir = str(temporary)
        environment = {name: os.environ[name] for name in ['HOME', 'RUSTUP_HOME', 'CARGO_HOME', 'DEVELOPER_DIR'] if name in os.environ}
        environment.update(p['build_environment'])
        environment.update(TMPDIR=str(temporary) + '/', CARGO_TARGET_DIR=str(output / 'cargo-target'),
                           GEN2_ROOT=str(repo), GEN2_NEMO_PACKAGE=str(package))
        policy = output / 'construction.sb'
        readonly = [repo, private, producer, package, *map(Path, p['asset_directories']),
                    *map(Path, p['asset_files']), *[Path(v['path']) for v in p['prior_qualifications'].values()]]
        policy.write_text('(version 1)\n(allow default)\n(deny network*)\n(deny file-write* ' +
                          ' '.join('(subpath ' + json.dumps(str(path)) + ')' for path in readonly) + ')\n')
        run('node-dependencies-clone', ['/bin/cp', '-cR', private / 'node_modules', snapshot / 'node_modules'], environment, snapshot, policy)
        if inventory(snapshot / 'node_modules') != p['node_modules_inventory']:
            raise ValueError('isolated npm dependency clone differs')
        spec = importlib.util.spec_from_file_location('pnola_overlay_apply', repo / 'integrations/pnola/apply.py')
        overlay = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(overlay)
        planned = overlay.plan(snapshot, repo, repo / 'integrations/pnola')
        if {str(t.relative_to(snapshot)) for t, _, _ in planned} != set(p['expected_overlay_changes']):
            raise ValueError('isolated overlay change set differs')
        run('overlay-check', ['/usr/bin/python3', repo / 'integrations/pnola/apply.py', '--pnola-root', snapshot,
            '--gen2-root', repo, '--check'], environment, snapshot, policy)
        run('overlay-apply', ['/usr/bin/python3', repo / 'integrations/pnola/apply.py', '--pnola-root', snapshot,
            '--gen2-root', repo], environment, snapshot, policy)
        if overlay.plan(snapshot, repo, repo / 'integrations/pnola'):
            raise ValueError('isolated overlay is not idempotent')
        rendered = {name: descriptor(snapshot / name) for name in p['private_source_files']}
        if any(rendered[str(t.relative_to(snapshot))] != {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
               for t, _, data in planned):
            raise ValueError('isolated overlay rendering differs')
        atomic_json(output / 'rendered-source-files.json', rendered)
        run('cpu-build', [p['build_tools']['node']['path'], p['build_tools']['npm']['path'], 'run', 'desktop:build'], environment, snapshot, policy)
        if any(descriptor(snapshot / n) != v for n, v in rendered.items()):
            raise ValueError('source changed during app build')
        built = output / 'cargo-target/release/bundle/macos/Pnola.app'
        app = output / 'cpu/Pnola.app'
        app.parent.mkdir()
        run('cpu-clone', ['/bin/cp', '-cR', built, app], environment, snapshot, policy)
        run('cpu-signature', ['/usr/bin/codesign', '--verify', '--deep', '--strict', app], environment, snapshot, policy)
        static = verify_app(app)
        if verify(app / 'Contents/Resources/nemo') != manifest or len(static['frameworks']) != 94:
            raise ValueError('complete app package or framework closure differs')
        signing = subprocess.run(['/usr/bin/codesign', '--display', '--verbose=4', str(app)], check=True, capture_output=True, text=True).stderr
        if 'Signature=adhoc' not in signing or not re.search(r'flags=0x[0-9a-f]+\(.*runtime.*\)', signing):
            raise ValueError('local ad-hoc hardened signing required')
        entitlements = subprocess.check_output(['/usr/bin/codesign', '--display', '--entitlements', ':-', str(app)], stderr=subprocess.DEVNULL)
        if plistlib.loads(entitlements) != plistlib.loads((snapshot / 'src-tauri/Entitlements.local.plist').read_bytes()):
            raise ValueError('local entitlements changed')
        (output / 'cpu-signing-details.log').write_text(signing)
        (output / 'cpu-entitlements.plist').write_bytes(entitlements)
        atomic_json(output / 'cpu-static-verification.json', static)
        r['apps']['cpu'] = {'path': str(app), 'files': inventory(app),
            'static_verification': descriptor(output / 'cpu-static-verification.json'),
            'ad_hoc_hardened_signature_observed': True, 'local_entitlements_unchanged': True}
        frozen()
        r['app_construction_passed'] = True
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
