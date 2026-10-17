"""Actual SDK graph controls and compiler-bound SGEMM objects for the Q8 trial."""
import os
from pathlib import Path
import shutil

from build_valid_mel_candidate import descriptor
from platform_native import msvc_probe_command, msvc_tool
from windows_q8_model_metrics import PATH, graph_vectors

MAX_LOG_BYTES = 16 * 1024**2


def verify_bindings(bindings):
    if any(descriptor(Path(name)) != expected for name, expected in bindings.items()):
        raise ValueError('earlier Q8 model inputs or evidence changed')


def compiler_objects(runner, database):
    """Retain each actual translation unit's unique source/object association."""
    if len(database) != 97:
        raise ValueError('all 97 compiler translation units required')
    sources, objects, bindings, units = set(), set(), {}, []
    build = runner.build_directory.resolve(strict=True)
    for row in database:
        directory = Path(row['directory']).resolve(strict=True)
        source = Path(row['file']).resolve(strict=True)
        obj = Path(row['output'])
        if not obj.is_absolute():
            obj = directory / obj
        obj = obj.resolve(strict=True)
        if (directory != build or not obj.is_relative_to(build)
                or source in sources or obj in objects):
            raise ValueError('unique isolated source/object compiler bindings required')
        sources.add(source)
        objects.add(obj)
        target = runner.output / 'compiler-objects' / obj.relative_to(build)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            raise ValueError('fresh retained compiler object required')
        shutil.copyfile(obj, target)
        source_identity, object_identity = descriptor(source), descriptor(obj)
        if descriptor(target) != object_identity:
            raise ValueError('retained compiler object differs')
        bindings.update({str(source): source_identity, str(obj): object_identity, str(target): object_identity})
        units.append({'command': row, 'source': source_identity, 'object': object_identity,
                      'retained_object': target.relative_to(runner.output).as_posix()})
    verify_bindings(bindings)
    return units, bindings


def kernel_evidence(runner, database):
    units, bindings = compiler_objects(runner, database)
    rows = [row for row in database if Path(row['file']).resolve(strict=True)
            == (runner.build_directory / '_deps/llama.cpp' / PATH).resolve(strict=True)]
    if len(rows) != 1:
        raise ValueError('one actual SGEMM compiler input required')
    row, = rows
    obj = Path(row['output'])
    if not obj.is_absolute():
        obj = Path(row['directory']) / obj
    obj = obj.resolve(strict=True)
    if not obj.is_relative_to(runner.build_directory):
        raise ValueError('isolated actual SGEMM compiler object required')
    target = runner.output / 'kernel.obj'
    shutil.copyfile(obj, target)
    dumpbin = msvc_tool(runner.environment, 'dumpbin.exe')
    dumpbin_identity = descriptor(dumpbin)
    runner.run('kernel-disassembly', [dumpbin, '/nologo', '/disasm', '/symbols', target])
    disassembly = runner.output / 'kernel-disassembly.stdout.log'
    if 'llamafile_sgemm' not in disassembly.read_text():
        raise ValueError('actual SGEMM entry in compiler object required')
    paths = [Path(row['file']), obj, target, dumpbin, disassembly, runner.output / 'kernel-disassembly.log']
    bindings.update({str(path): descriptor(path) for path in paths})
    if descriptor(dumpbin) != dumpbin_identity or descriptor(obj) != descriptor(target):
        raise ValueError('SGEMM object or disassembler changed')
    verify_bindings(bindings)
    return {'compiler_command': row, 'translation_units': units, 'bindings': bindings}


def model_free_controls(runner, unchanged, reference=None):
    source = runner.tools / 'ggml_matrix_profile_control.cpp'
    binary = runner.output / 'matrix-profile-control.exe'
    libraries = []
    for stem in ('ggml', 'ggml-base', 'ggml-cpu'):
        path, = runner.build_directory.rglob('gen2_nemo_windows_' + stem + '.lib')
        libraries.append(path)
    inputs = {str(path): descriptor(path) for path in [source, *libraries]}
    command = msvc_probe_command(runner.environment, [source],
        [runner.build_directory / '_deps/llama.cpp/ggml/include'], libraries[0], binary) + libraries[1:]
    if Path(command[0]).resolve(strict=True) != runner.msvc:
        raise ValueError('fixed admitted MSVC graph compiler required')
    unchanged()
    runner.run('matrix-control-build', command, runner.output)
    bindings = {**inputs, str(binary): descriptor(binary)}
    mapping = {key: value for key, value in runner.environment.items() if key.upper() != 'PATH'}
    old_path = {key.upper(): value for key, value in runner.environment.items()}['PATH']
    mapping['PATH'] = str(runner.prefix / 'bin') + os.pathsep + old_path
    cases = []
    for threads in (1, 2, 4):
        unchanged()
        verify_bindings(bindings)
        phase = f'matrix-control-{threads}'
        target = runner.output / phase
        runner.run(phase, [binary, str(threads), target], environment=mapping)
        stdout, log = [runner.output / (phase + suffix) for suffix in ('.stdout.log', '.log')]
        if max(stdout.stat().st_size, log.stat().st_size) > MAX_LOG_BYTES:
            raise ValueError('bounded graph control logs required')
        if any(marker in log.read_text() for marker in ('[timing]', '[gen2-cpu-', '[gen2-matrix-')):
            raise ValueError('uninstrumented graph control required')
        paths = [Path(str(target) + f'.{shape}.{repeat}.f32') for shape in range(4) for repeat in range(3)]
        reference = graph_vectors(stdout.read_text(), threads, [path.read_bytes() for path in paths], reference)
        outputs = {str(path): descriptor(path) for path in [*paths, stdout, log]}
        bindings.update(outputs)
        cases.append({'threads': threads, 'outputs_and_logs': outputs})
    unchanged()
    verify_bindings(bindings)
    return {'cases': cases, 'bindings': bindings, 'vectors': 36,
            'limits': 'Three actual Windows processes; per-shape exact full vectors across workers and SDKs.'}, reference
