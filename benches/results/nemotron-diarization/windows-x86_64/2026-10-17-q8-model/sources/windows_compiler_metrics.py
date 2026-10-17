"""Guarded compiler-only diagnostic adaptation and actual build admission."""
from pathlib import Path, PureWindowsPath
import re

from build_valid_mel_candidate import descriptor
from platform_native import powershell_literal

ORDER = ['msvc', 'clang-cl', 'clang-cl', 'msvc']
AMBIENT_COMPILER_FLAGS = {'CL', '_CL_', 'CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'LDFLAGS', 'CC', 'CXX'}


def compiler_kind(kind):
    if kind not in ('msvc', 'clang-cl'):
        raise ValueError('explicit MSVC or clang-cl diagnostic required')
    return kind


def compiler_environment(environment):
    return {key: value for key, value in environment.items() if key.upper() not in AMBIENT_COMPILER_FLAGS}


def timing_evidence(cases):
    if len(cases) != 4 or [case['compiler'] for case in cases] != ORDER:
        raise ValueError('all four ordered compiler observations required')
    elapsed = [case['result']['elapsed_micros'] for case in cases]
    if any(type(value) is not int or value <= 0 for value in elapsed):
        raise ValueError('positive integer compiler timings required')
    drift_small = 10 * abs(elapsed[3] - elapsed[0]) <= elapsed[0]
    candidates_faster = 10 * max(elapsed[1:3]) <= 9 * min(elapsed[0], elapsed[3])
    return {'elapsed_micros': elapsed, 'native_rtfs': [value / 60e6 for value in elapsed],
            'control_drift_fraction': abs(elapsed[3] - elapsed[0]) / elapsed[0],
            'control_drift_at_most_10_percent': drift_small,
            'both_candidates_at_least_10_percent_faster_than_both_controls': candidates_faster,
            'compiler_speedup_prediction_supported': drift_small and candidates_faster,
            'limits': 'Small same-host diagnostic only; no timing average, release pass or causal operator attribution.'}


def build_command(command, msvc, compiler, kind):
    """Change only the existing compiler selections in the frozen build call."""
    compiler_kind(kind)
    if len(command) != 5 or command[:4] != ['pwsh', '-NoProfile', '-NonInteractive', '-Command']:
        raise ValueError('exact existing PowerShell build command required')
    text = command[4]
    site = ' -Backend cpu -Profile asr '
    if text.count(site) != 1 or ' -Compiler ' in text:
        raise ValueError('one original CPU build selection required')
    for language in ('C', 'CXX'):
        old = powershell_literal('-DCMAKE_' + language + '_COMPILER=' + str(msvc))
        if text.count(old) != 1:
            raise ValueError('one original absolute MSVC compiler selection required')
        text = text.replace(old, powershell_literal('-DCMAKE_' + language + '_COMPILER=' + str(compiler)))
    text = text.replace(site, ' -Backend cpu -Compiler ' + kind + ' -Profile asr ')
    return command[:4] + [text]


def cmake_value(text, name):
    values = re.findall(r'^set\(' + re.escape(name) + r' "([^"\r\n]*)"\)\s*$', text, re.MULTILINE)
    if len(values) != 1:
        raise ValueError('one actual CMake compiler value required: ' + name)
    return values[0]


def command_compiler(command):
    match = re.match(r'^(?:"([^"\r\n]+)"|(\S+))(?:\s|$)', command)
    if not match:
        raise ValueError('explicit actual compiler executable required')
    return match[1] or match[2]


def admit_compiler_paths(compiler, names):
    """Bind real Windows full/8.3 aliases by file identity, not by basename."""
    expected = descriptor(Path(compiler))
    result = {}
    for name in names:
        path = Path(name)
        if not path.samefile(compiler) or descriptor(path) != expected:
            raise ValueError('compiler alias does not identify selected binary')
        result[str(name)] = expected
    return result


def compiler_evidence(cache_text, database, compiler_files, compiler, kind, aliases=()):
    compiler_kind(kind)
    cache = {}
    for line in cache_text.splitlines():
        if line.startswith(('#', '//')) or '=' not in line or ':' not in line.split('=', 1)[0]:
            continue
        key, value = line.split('=', 1)
        key = key.split(':', 1)[0]
        if key in cache:
            raise ValueError('duplicate CMake compiler cache field')
        cache[key] = value
    required = {'CMAKE_GENERATOR': 'Ninja', 'CMAKE_BUILD_TYPE': 'Release', 'BUILD_SHARED_LIBS': 'ON',
        'GGML_NATIVE': 'OFF', 'GGML_AVX': 'ON', 'GGML_AVX2': 'ON', 'GGML_FMA': 'ON', 'GGML_F16C': 'ON',
        'GGML_AVX512': 'OFF', 'GGML_AVX_VNNI': 'OFF', 'GGML_OPENMP': 'OFF', 'GGML_BACKEND_DL': 'OFF',
        'GGML_METAL': 'OFF', 'GGML_CUDA': 'OFF', 'GGML_VULKAN': 'OFF',
        'NEMO_SPEECH_BUILD_ASR': 'OFF', 'NEMO_SPEECH_BUILD_TTS': 'OFF'}
    if any(cache.get(key) != value for key, value in required.items()):
        raise ValueError('compiler diagnostic portable CPU cache differs')
    linker = PureWindowsPath(cache.get('CMAKE_LINKER', ''))
    if not linker.is_absolute() or linker.name.lower() not in ('link.exe', 'lld-link.exe'):
        raise ValueError('actual Windows linker path required')
    identity = 'MSVC' if kind == 'msvc' else 'Clang'
    approved = {PureWindowsPath(compiler), *(PureWindowsPath(name) for name in aliases)}
    languages = {}
    for language in ('C', 'CXX'):
        text = compiler_files[language]
        if (cmake_value(text, 'CMAKE_' + language + '_COMPILER_ID') != identity
                or PureWindowsPath(cmake_value(text, 'CMAKE_' + language + '_COMPILER')) not in approved
                or PureWindowsPath(cache.get('CMAKE_' + language + '_COMPILER', '')) not in approved):
            raise ValueError('actual compiler identity/path differs')
        if kind == 'clang-cl' and cmake_value(text, 'CMAKE_' + language + '_SIMULATE_ID') != 'MSVC':
            raise ValueError('clang-cl must use the MSVC ABI')
        languages[language] = {'id': identity, 'version': cmake_value(text, 'CMAKE_' + language + '_COMPILER_VERSION')}
    if not isinstance(database, list) or not database:
        raise ValueError('complete actual compiler database required')
    cpu = []
    for entry in database:
        command = entry.get('command', '')
        if PureWindowsPath(command_compiler(command)) not in approved:
            raise ValueError('actual compiler command differs from selected binary')
        tokens = [token.strip('"') for token in re.findall(r'"[^"]*"|\S+', command)]
        if any((token.lower().startswith('/arch:') and token.lower() != '/arch:avx2')
               or token.lower().startswith(('-mavx512', '/clang:-mavx512', '-mavxvnni', '-mavx-vnni',
                    '-mamx', '-march=', '-mcpu=', '/clang:-march=', '/clang:-mcpu=')) for token in tokens):
            raise ValueError('compiler command escapes portable ISA baseline')
        if 'ggml-cpu' in PureWindowsPath(entry['file']).parts:
            if not {'-DGGML_AVX2', '-DGGML_FMA', '-DGGML_F16C', '/arch:AVX2'} <= set(tokens):
                raise ValueError('actual AVX2/FMA/F16C CPU compiler definitions required')
            cpu.append(entry['file'])
    if not cpu:
        raise ValueError('actual GGML CPU compiler commands required')
    return {'kind': kind, 'compiler': str(compiler), 'linker': str(linker), 'languages': languages,
            'compiler_entries': len(database), 'ggml_cpu_entries': len(cpu),
            'limits': 'Compiler/toolchain diagnostic only; no ISA widening, public runtime or package qualification.'}
