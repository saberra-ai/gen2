"""Derive only the separately frozen contraction experiment's adapter/protocol."""


def derive_runner(original):
    replacements = [
        (b'from windows_compiler_metrics import (', b'from windows_compiler_fp_metrics import ('),
        (b"'windows-compiler-diagnostic-v2.protocol.json'", b"'windows-compiler-fp.protocol.json'")]
    for old, new in replacements:
        if original.count(old) != 1:
            raise ValueError('one exact original compiler controller site required')
        original = original.replace(old, new)
    return original
