"""Audit one private runtime plus explicitly bound auxiliary loaded images."""
from pathlib import Path
import re

from build_valid_mel_candidate import descriptor
from macos_loaded_libraries import audit_loaded_libraries


def audit_with_auxiliary(trace, package, manifest, executable, auxiliary):
    expected = {str(Path(path).resolve(strict=True)): record for path, record in auxiliary.items()}
    observed, retained = {}, []
    for line in trace.splitlines():
        match = re.search(r'dyld\[\d+\]: <[A-Fa-f0-9-]+> (/[^\r\n]+)', line)
        if match and not match[1].startswith(('/usr/lib/', '/System/Library/')):
            path = str(Path(match[1]).resolve(strict=True))
            if path in expected:
                actual = descriptor(Path(path))
                if actual != expected[path]:
                    raise ValueError('auxiliary loaded image differs from frozen bytes: ' + path)
                observed[path] = {'reported_path': match[1], **actual}
                continue
        # Unknown images must reach the original strict runtime auditor.
        retained.append(line)
    if set(observed) != set(expected):
        raise ValueError('required auxiliary loaded images missing')
    return {'runtime': audit_loaded_libraries('\n'.join(retained), package, manifest, executable),
            'auxiliary_libraries': observed, 'auxiliary_library_count': len(observed)}
