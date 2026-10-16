"""Explicit portable ARM compiler admission; no hardware/inference attestation."""
from pathlib import PurePosixPath
import shlex


PORTABLE_ARM64 = 'armv8.2-a+dotprod+fp16'


def portable_definitions():
    return {'GGML_NATIVE': 'OFF', 'GGML_CPU_ARM_ARCH': PORTABLE_ARM64,
            'CMAKE_C_FLAGS': '-march=' + PORTABLE_ARM64,
            'CMAKE_CXX_FLAGS': '-march=' + PORTABLE_ARM64,
            'CMAKE_OSX_ARCHITECTURES': 'arm64', 'CMAKE_OSX_DEPLOYMENT_TARGET': '26.0'}


def audit_portable_commands(entries):
    if not entries:
        raise ValueError('empty portable compiler database')
    sources = []
    for entry in entries:
        tokens = entry.get('arguments')
        if tokens is None:
            tokens = shlex.split(entry['command'])
        if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
            raise ValueError('invalid portable compiler arguments')
        if '-march=' + PORTABLE_ARM64 not in tokens:
            raise ValueError('portable target missing from compiler unit: ' + entry['file'])
        for token in tokens:
            if (token.startswith('-m') and token not in {'-march=' + PORTABLE_ARM64,
                                                        '-mmacosx-version-min=26.0'}
                    or token.startswith(('--target', '-target', '-Xclang', '-Xarch', '-mllvm'))
                    or token.startswith(('-D__ARM_FEATURE_', '-DGGML_USE_SME', '-DGGML_USE_MATMUL_INT8',
                                         '-DGGML_USE_SVE', '-DGGML_USE_SVE2'))):
                raise ValueError('unexpected portable CPU override: ' + token)
        sources.append(entry['file'])
    # Check the actual CPU kernel units, as well as every other SDK unit.
    required = {'ggml-cpu.cpp', 'arch/arm/quants.c', 'arch/arm/repack.cpp'}
    observed = {suffix for suffix in required for name in sources
                if PurePosixPath(name).as_posix().endswith('/ggml-cpu/' + suffix)}
    if observed != required:
        raise ValueError('portable CPU kernel compiler units missing')
    return {'mode': 'explicit-portable-arm64', 'arm_arch': PORTABLE_ARM64,
            'all_compiler_units_have_explicit_target': True, 'compiler_units': len(entries),
            'cpu_kernel_units_present': sorted(observed),
            'hardware_or_model_qualification': False}
