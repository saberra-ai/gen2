"""Retain the exact graph-link inputs before running production-candidate controls."""
from pathlib import Path
import shutil

from build_valid_mel_candidate import descriptor
from windows_q8_model_controls import model_free_controls as parent_controls, verify_bindings

LIBRARY_NAMES = {'gen2_nemo_windows_' + name + '.lib' for name in ('ggml', 'ggml-base', 'ggml-cpu')}


def retain_import_libraries(runner):
    build = runner.build_directory.resolve(strict=True)
    target_root = runner.output / 'graph-inputs'
    if target_root.exists() or target_root.is_symlink():
        raise ValueError('fresh graph input retention directory required')
    sources = []
    for name in sorted(LIBRARY_NAMES):
        paths = list(build.rglob(name))
        if len(paths) != 1:
            raise ValueError('one build-only graph import library required: ' + name)
        source = paths[0]
        if source.is_symlink() or not source.resolve(strict=True).is_relative_to(build):
            raise ValueError('isolated regular graph import library required')
        sources.append(source.resolve(strict=True))
    target_root.mkdir()
    bindings, records = {}, {}
    for source in sources:
        identity = descriptor(source)
        if identity['bytes'] <= 0:
            raise ValueError('nonempty graph import library required')
        target = target_root / source.name
        shutil.copyfile(source, target)
        if descriptor(target) != identity:
            raise ValueError('retained graph library differs')
        bindings.update({str(source): identity, str(target): identity})
        records[str(source)] = {'retained_path': target.relative_to(runner.output).as_posix(), **identity}
    verify_bindings(bindings)
    return records, bindings


def model_free_controls(runner, unchanged):
    libraries, bindings = retain_import_libraries(runner)
    def retained_unchanged():
        unchanged()
        verify_bindings(bindings)
    controls, reference = parent_controls(runner, retained_unchanged)
    retained_unchanged()
    if not all(controls['bindings'].get(name) == expected for name, expected in bindings.items()
               if name in libraries):
        raise ValueError('actual graph linker inputs differ from retained libraries')
    controls['bindings'].update(bindings)
    controls['retained_import_libraries'] = libraries
    controls['graph_input_payloads_fully_retained'] = True
    verify_bindings(controls['bindings'])
    return controls, reference
