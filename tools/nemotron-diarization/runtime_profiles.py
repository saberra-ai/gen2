"""Bind the V3 valid-Mel build to its immutable base, patch and semantic profile."""
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

from artifacts import digest
from build_valid_mel_candidate import descriptor, source_inventory


PROFILE = 'gen2-valid-mel-v1'
FRAME_CONTRACT = {'name': 'v3-streaming-valid-mel-v1',
                  'scope': 'V3 streaming finish only; offline and V2 semantics unchanged',
                  'sample_rate': 16000, 'hop_samples': 160,
                  'final_frames': 'floor(real_input_samples / 160)', 'output_channels': 8}


def load_profile():
    tools = Path(__file__).resolve().parent
    path = tools / 'runtime-valid-mel.lock.json'
    lock = json.loads(path.read_text())
    parent = lock['historical_runtime_lock']
    if parent['path'] != 'runtime.lock.json' or descriptor(tools / parent['path']) != {k: parent[k] for k in ['bytes', 'sha256']}:
        raise ValueError('historical runtime lock differs')
    if lock['schema_version'] != 2 or lock['profile'] != PROFILE:
        raise ValueError('unsupported valid-Mel profile')
    runtime = lock['runtime']
    if (runtime['cmake_definitions'].get('NEMO_SPEECH_VERSION_METADATA') != PROFILE
            or runtime['reported_version'] != 'nemo-speech-asr 0.2.0+' + PROFILE
            or runtime['frame_contract'] != FRAME_CONTRACT):
        raise ValueError('valid-Mel semantic profile differs')
    patches = runtime['compatibility_patches']
    if len(patches) != 1 or patches[0]['path'] != 'patches/v3-finish-valid-mel.patch':
        raise ValueError('unexpected valid-Mel patch series')
    patch = patches[0]
    if descriptor(tools / patch['path']) != {k: patch[k] for k in ['bytes', 'sha256']}:
        raise ValueError('valid-Mel patch bytes differ')
    for key in ['source_sha256_before', 'source_sha256_after']:
        if set(patch.get(key, {})) != {'src/asr/diar/diar_pipeline.cpp'}:
            raise ValueError('valid-Mel patch source provenance is incomplete')
    historical = json.loads((tools / parent['path']).read_text())
    for key in ['revision', 'llama_revision', 'repository']:
        if runtime[key] != historical['runtime'][key]:
            raise ValueError('valid-Mel source base differs')
    if any(lock[key] != historical[key] for key in ['model', 'fixture', 'artifacts']):
        raise ValueError('valid-Mel model or fixture pins differ')
    for name, before in patch['source_sha256_before'].items():
        if historical['runtime']['source_sha256'].get(name) != before:
            raise ValueError('valid-Mel pre-patch source pin differs')
    for name, after in patch['source_sha256_after'].items():
        if runtime['source_sha256'].get(name) != after:
            raise ValueError('valid-Mel post-patch source pin differs')
    return path, lock


def verify_source(source, lock):
    """Require a fresh, isolated, already patched tree before configuration."""
    source = Path(source)
    runtime = lock['runtime']
    def capture(argv, root=source):
        return subprocess.check_output(argv, cwd=root)
    if capture(['git', 'rev-parse', 'HEAD']).decode().strip() != runtime['revision']:
        raise ValueError('valid-Mel native base differs')
    patch = Path(__file__).resolve().parent / runtime['compatibility_patches'][0]['path']
    if capture(['git', 'diff', '--binary', '--no-ext-diff', '--ignore-submodules=all']) != patch.read_bytes():
        raise ValueError('valid-Mel native diff differs from the sole compatibility patch')
    if capture(['git', 'diff', '--cached', '--binary', '--no-ext-diff', '--ignore-submodules=all']):
        raise ValueError('staged valid-Mel source changes')
    llama = source / 'llama.cpp'
    if (capture(['git', 'rev-parse', 'HEAD'], llama).decode().strip() != runtime['llama_revision']
            or capture(['git', 'status', '--porcelain', '--untracked-files=no'], llama)):
        raise ValueError('valid-Mel llama submodule differs')
    for tree in [source, llama]:
        for flags in [[], ['--ignored']]:
            if capture(['git', 'ls-files', '--others', *flags, '--exclude-standard'], tree):
                raise ValueError('extra files could enter valid-Mel compilation: ' + str(tree))
    for name, expected in runtime['source_sha256'].items():
        if digest(source / name) != expected:
            raise ValueError('valid-Mel source file differs: ' + name)


def audit_cpu_compiler_tree(source, output, run):
    """Independently replay the upstream GGML patch series before compilation."""
    source, output = Path(source), Path(output)
    _, lock = load_profile()
    archive = output / 'pristine-llama.tar'
    run('llama-archive', ['git', 'archive', '--format=tar', '-o', str(archive.resolve()),
                         lock['runtime']['llama_revision'], '--', '.', ':(exclude)models',
                         ':(exclude)docs', ':(exclude)media'], source / 'llama.cpp')
    with tempfile.TemporaryDirectory(prefix='llama-replay-', dir=output) as temporary:
        root = Path(temporary)
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter='data')
        series = []
        for line in (source / 'patches/series').read_text().splitlines():
            name = line.split('#', 1)[0].strip()
            if not name:
                continue
            patch = source / 'patches' / name
            normalized = output / 'current-llama.patch'
            normalized.write_bytes(patch.read_bytes().replace(b'\r\n', b'\n'))
            run('llama-patch-' + str(len(series)),
                ['git', 'apply', '--whitespace=nowarn', str(normalized.resolve())], root,
                {'GIT_CEILING_DIRECTORIES': str(root.parent.resolve())})
            series.append({'name': name, **descriptor(patch)})
        normalized.unlink()
        expected = source_inventory(root)
        materialized = source / 'build/cpu-diar/_deps/llama.cpp'
        if source_inventory(materialized) != expected:
            raise ValueError('materialized compiler tree differs from independent replay')
    audit = {'base_revision': lock['runtime']['llama_revision'], 'archive': descriptor(archive),
             'compiled_tree_path': str(materialized), 'exact_independent_replay_match': True,
             'patch_series': series, 'files': expected}
    (output / 'llama-materialized-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    return materialized, expected
