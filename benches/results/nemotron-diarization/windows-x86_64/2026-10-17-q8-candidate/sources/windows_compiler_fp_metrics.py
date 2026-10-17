"""One explicit Clang contraction flag; preserve every other compiled input."""
import json
from pathlib import Path, PureWindowsPath
import re

from build_valid_mel_candidate import descriptor
from platform_native import powershell_literal
from windows_compiler_metrics import (ORDER, admit_compiler_paths, command_compiler,
    compiler_environment, compiler_kind, timing_evidence,
    build_command as parent_build_command, compiler_evidence as parent_compiler_evidence)

FLAG = '/clang:-ffp-contract=off'
REFERENCE = 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-compiler-parity-failure'


def baseline(kind):
    compiler_kind(kind)
    root = Path(__file__).resolve().parents[2] / REFERENCE
    manifest = json.loads((root / 'collection.json').read_text())['files']
    names = ['collection-qualification.json', 'evidence-' + kind + '/CMakeCache.txt',
             'evidence-' + kind + '/compile_commands.json']
    for name in names:
        if descriptor(root / name) != manifest[name]:
            raise ValueError('saved compiler baseline changed: ' + name)
    qualification = json.loads((root / names[0]).read_text())
    if qualification['source_commit'] != 'dd4a107e13b1f63a0cd796fd45511621fdba08cd':
        raise ValueError('actual failed compiler baseline required')
    return ((root / names[1]).read_text(), json.loads((root / names[2]).read_text()),
            qualification['builds'][kind]['compiler'])


def build_command(command, msvc, compiler, kind):
    result = parent_build_command(command, msvc, compiler, kind)
    for path, selected in [(msvc, 'msvc'), (compiler, kind)]:
        if descriptor(Path(path)) != baseline(selected)[2]['compiler_binary']:
            raise ValueError('exact scalar-tested compiler binary required')
    text = result[4]
    if re.search(r'-DCMAKE_(?:C|CXX)_FLAGS(?:[_:=])', text):
        raise ValueError('original build must not override global compiler flags')
    if kind == 'msvc':
        return result
    site = ' -CMakeArgs @('
    if text.count(site) != 1:
        raise ValueError('one original CMake argument list required')
    additions = ','.join(powershell_literal('-DCMAKE_' + language + '_FLAGS_INIT:STRING=' + FLAG)
                         for language in ('C', 'CXX'))
    return result[:4] + [text.replace(site, site + additions + ',')]


def cache_value(text, name):
    values = re.findall(r'^' + re.escape(name) + r':(?:STRING|UNINITIALIZED)=([^\r\n]*)\r?$', text, re.M)
    if len(values) != 1:
        raise ValueError('one actual compiler flag cache value required: ' + name)
    return values[0].split()


def normalized_commands(database, kind, remove_flag):
    compiler_kind(kind)
    directories = {PureWindowsPath(row['directory']) for row in database}
    if len(directories) != 1:
        raise ValueError('one original build directory required')
    directory, = directories
    if not directory.is_absolute() or directory.parts[-2:] != ('build', 'gen2-cpu-diar'):
        raise ValueError('original native build layout required')
    root = directory.parents[1]
    result = {}
    for row in database:
        path = PureWindowsPath(row['file'])
        if not path.is_relative_to(root):
            raise ValueError('compiled source outside original native tree')
        name = path.relative_to(root).as_posix()
        if name in result:
            raise ValueError('duplicate compiled source in original baseline')
        # The parent admission binds the leading executable to the actual selected
        # compiler file. Only its physical path alias and the isolated source root
        # vary here; preserve all other argument tokens and output identities.
        match = re.fullmatch(r'(?:"[^"\r\n]+"|\S+)\s+(.+)', row['command'])
        if not match:
            raise ValueError('explicit compiler command required')
        text = match[1].replace(str(root), '@SOURCE@').replace(root.as_posix(), '@SOURCE@')
        tokens = re.findall(r'"[^"]*"|\S+', text)
        if remove_flag:
            if tokens.count(FLAG) != 1:
                raise ValueError('exactly one Clang contraction flag required per compiler command')
            tokens.remove(FLAG)
        elif FLAG in tokens:
            raise ValueError('unmodified control must not contain contraction flag')
        result[name] = {'arguments': tokens, 'output': PureWindowsPath(row['output']).as_posix()}
    return result


def compiler_evidence(cache_text, database, compiler_files, compiler, kind, aliases=()):
    value = parent_compiler_evidence(cache_text, database, compiler_files, compiler, kind, aliases)
    prior_cache, prior_database, prior_compiler = baseline(kind)
    for language in ('C', 'CXX'):
        field = 'CMAKE_' + language + '_FLAGS'
        flags = cache_value(cache_text, field)
        if kind == 'clang-cl':
            if flags.count(FLAG) != 1 or cache_value(cache_text, field + '_INIT') != [FLAG]:
                raise ValueError('one explicit Clang contraction initializer required')
            flags.remove(FLAG)
        if flags != cache_value(prior_cache, field):
            raise ValueError('original platform compiler flags changed')
        if cache_value(cache_text, field + '_RELEASE') != cache_value(prior_cache, field + '_RELEASE'):
            raise ValueError('original Release compiler flags changed')
    actual = normalized_commands(database, kind, kind == 'clang-cl')
    expected = normalized_commands(prior_database, kind, False)
    if actual != expected:
        raise ValueError('compiler commands changed beyond explicit contraction flag')
    if descriptor(Path(value['linker'])) != prior_compiler['linker_binary']:
        raise ValueError('original selected linker binary required')
    value['implicit_contraction_admission'] = {
        'policy': 'MSVC default' if kind == 'msvc' else 'Clang explicit contraction off',
        'added_flag': None if kind == 'msvc' else FLAG,
        'compiler_commands_matching_original': len(actual),
        'normalization': 'Selected compiler path alias and isolated source root only; remove exactly one candidate flag.'}
    return value
