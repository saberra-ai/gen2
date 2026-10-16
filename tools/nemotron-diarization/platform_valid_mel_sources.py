"""Verify composed Windows source changes and replay platform CPU GGML inputs."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

from build_valid_mel_candidate import descriptor, source_inventory
from windows_compat import ORIGINAL_SHA256, rewrite_materializer


def verify_completed_source(source, build, lock, patch, windows):
    source, build = Path(source).resolve(), Path(build).resolve()
    if not build.is_relative_to(source / 'build'):
        raise ValueError('platform compiler tree must be inside the isolated source build directory')
    def capture(argv, root=source):
        return subprocess.check_output(argv, cwd=root)
    if capture(['git', 'rev-parse', 'HEAD']).decode().strip() != lock['runtime']['revision']:
        raise ValueError('platform native base changed during construction')
    if capture(['git', 'diff', '--cached', '--binary', '--no-ext-diff', '--ignore-submodules=all']):
        raise ValueError('staged platform source changes')
    changed = set(capture(['git', 'diff', '--name-only', '--ignore-submodules=all']).decode().splitlines())
    expected = {'src/asr/diar/diar_pipeline.cpp'} | ({'cmake/llama_cpp.cmake'} if windows else set())
    if changed != expected:
        raise ValueError('platform source changes exceed the valid-Mel and declared LF overlays')
    if capture(['git', 'diff', '--binary', '--no-ext-diff', '--', 'src/asr/diar/diar_pipeline.cpp']) != Path(patch).read_bytes():
        raise ValueError('platform inference patch changed during construction')
    if windows:
        original = capture(['git', 'show', 'HEAD:cmake/llama_cpp.cmake'])
        if hashlib.sha256(original).hexdigest() != ORIGINAL_SHA256 or (source / 'cmake/llama_cpp.cmake').read_bytes() != rewrite_materializer(original):
            raise ValueError('Windows source differs from the exact declared LF overlay')
    llama = source / 'llama.cpp'
    if capture(['git', 'rev-parse', 'HEAD'], llama).decode().strip() != lock['runtime']['llama_revision'] or capture(['git', 'status', '--porcelain', '--untracked-files=no'], llama):
        raise ValueError('platform pristine llama submodule changed')
    for tree in [source, llama]:
        for flags in [[], ['--ignored']]:
            extras = capture(['git', 'ls-files', '--others', *flags, '--exclude-standard'], tree).decode().splitlines()
            if any(not (tree == source and (source / path).resolve().is_relative_to(build)) for path in extras):
                raise ValueError('unrecorded platform compiler inputs')


def audit_cpu_tree(source, build, output, revision, run):
    """Audit saved compiler inputs after construction and before model admission."""
    source, build, output = Path(source), Path(build), Path(output)
    archive = output / 'platform-pristine-llama.tar'
    run('compiler-llama-archive', ['git', 'archive', '--format=tar', '-o', str(archive.resolve()), revision,
        '--', '.', ':(exclude)models', ':(exclude)docs', ':(exclude)media'], source / 'llama.cpp')
    with tempfile.TemporaryDirectory(prefix='platform-llama-replay-', dir=output) as temporary:
        root = Path(temporary)
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter='data')
        series = []
        for line in (source / 'patches/series').read_text().splitlines():
            name = line.split('#', 1)[0].strip()
            if not name:
                continue
            patch = source / 'patches' / name
            normalized = output / 'current-platform-llama.patch'
            normalized.write_bytes(patch.read_bytes().replace(b'\r\n', b'\n'))
            run('compiler-llama-patch-' + str(len(series)), ['git', 'apply', '--whitespace=nowarn', str(normalized.resolve())], root,
                {'GIT_CEILING_DIRECTORIES': str(root.parent.resolve())})
            series.append({'name': name, **descriptor(patch)})
        normalized.unlink()
        expected = source_inventory(root)
        materialized = build / '_deps/llama.cpp'
        if source_inventory(materialized) != expected:
            raise ValueError('platform CPU compiler tree differs from independent patch replay')
    values = dict(line.split('=', 1) for line in (build / 'CMakeCache.txt').read_text().splitlines()
                  if '=' in line and not line.startswith(('#', '//')))
    if (values.get('NEMO_SPEECH_GGML_PATCHED:BOOL') != 'ON'
            or values.get('NEMO_SPEECH_LLAMA_CPP_SOURCE_DIR:PATH') != ''
            or Path(values.get('ggml_SOURCE_DIR:STATIC', '')).resolve() != (materialized / 'ggml').resolve()):
        raise ValueError('platform CMake compiler source selection differs')
    tracked = {(source / name.decode()).resolve() for name in subprocess.check_output(['git', 'ls-files', '-z'], cwd=source).split(b'\0') if name}
    compiler_inputs, all_compiler_inputs = {}, {}
    for entry in json.loads((build / 'compile_commands.json').read_text()):
        path = Path(entry['file']).resolve()
        if path.is_relative_to((source / 'llama.cpp').resolve()):
            raise ValueError('CPU compilation unexpectedly used the pristine llama tree')
        if path.is_relative_to(materialized.resolve()):
            relative = str(path.relative_to(materialized.resolve())).replace('\\', '/')
            if descriptor(path) != expected.get(relative):
                raise ValueError('actual platform compiler input differs from replay')
            compiler_inputs[relative] = descriptor(path)
        elif path not in tracked or path.is_relative_to(build.resolve()) or not path.is_file():
            raise ValueError('unbound platform compiler source: ' + str(path))
        all_compiler_inputs[str(path)] = descriptor(path)
    if not compiler_inputs:
        raise ValueError('platform compile database lacks materialized GGML inputs')
    record = {'base_revision': revision, 'archive': descriptor(archive), 'compiled_tree_path': str(materialized),
        'audit_timing': 'after SDK build, before model admission; not a pre-compilation measurement',
        'exact_independent_replay_match': True, 'patch_series': series, 'files': expected,
        'actual_compiler_sources': compiler_inputs, 'all_compiler_sources': all_compiler_inputs,
        'compile_database': descriptor(build / 'compile_commands.json')}
    (output / 'platform-llama-materialized-audit.json').write_text(json.dumps(record, indent=2) + '\n')
    return record
