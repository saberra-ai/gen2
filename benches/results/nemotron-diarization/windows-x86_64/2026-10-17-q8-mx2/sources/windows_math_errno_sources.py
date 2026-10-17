"""Derive a single-flag Clang comparison from the frozen contraction controller."""
from windows_worker_metrics import replace_once


def derive_runner(raw):
    raw = replace_once(raw, b'from windows_compiler_fp_metrics import (', b'from windows_math_errno_metrics import (')
    raw = replace_once(raw, b'from windows_worker_metrics import PCM_SHA256, compare_native, telemetry',
        b'from windows_math_errno_metrics import PCM_SHA256, compare_native, telemetry')
    raw = replace_once(raw, b'compiler_environment, compiler_evidence, compiler_kind, timing_evidence)',
        b'compiler_environment, compiler_evidence, compiler_kind, timing_evidence, validate_variant, admit_math_driver, FLAGS)')
    raw = replace_once(raw, b'def __init__(self, source, prefix, output, jobs, kind, protocol, frozen):',
        b'def __init__(self, source, prefix, output, jobs, kind, protocol, frozen, variant):')
    raw = replace_once(raw, b'        super().__init__(source, prefix, output, jobs, 4, protocol, frozen)',
        b'        self.variant = validate_variant(variant)\n        super().__init__(source, prefix, output, jobs, 4, protocol, frozen)')
    raw = replace_once(raw, b'build_command(command, self.msvc, self.compiler, self.kind)',
        b'build_command(command, self.msvc, self.compiler, self.kind, self.variant)')
    raw = replace_once(raw, b'database, compiler_files, self.compiler, self.kind, self.compiler_aliases)',
        b'database, compiler_files, self.compiler, self.kind, self.compiler_aliases, self.variant)')
    site = b'        super().build()\n'
    raw = replace_once(raw, site,
        b"        flags = ['/nologo', '/c', '/O2', '/Ob2', '/DNDEBUG', '/arch:AVX2', '/std:c++17', *FLAGS[self.variant]]\n"
        b"        self.run('compiler-math-driver', [self.compiler, '-###', *flags,\n"
        b"            self.tools / 'fp_contract_probe.cpp', '/Fo' + str(self.output / 'unused-driver.obj')])\n"
        b"        self.math_driver = admit_math_driver((self.output / 'compiler-math-driver.log').read_text(), self.variant)\n" + site)
    prefix, body = raw.split(b'\n\ndef main():', 1)
    body = replace_once(body, b"'windows-compiler-fp.protocol.json'", b"'windows-math-errno.protocol.json'")
    body = body.replace(b"'msvc'", b"'control'").replace(b"'clang-cl'", b"'no-errno'")
    body = replace_once(body, b'evidence, args.jobs, kind, p, frozen)',
        b"evidence, args.jobs, 'clang-cl', p, frozen, kind)")
    body = replace_once(body, b'compiler_admission=admission,', b'compiler_admission=admission, math_driver=runner.math_driver,')
    body = replace_once(body, b"'CMakeCache.txt', 'compile_commands.json', 'platform-llama-materialized-audit.json']",
        b"'CMakeCache.txt', 'compile_commands.json', 'platform-llama-materialized-audit.json',\n                     'compiler-math-driver.log', 'compiler-math-driver.stdout.log']")
    return prefix + b'\n\ndef main():' + body
