"""Audit one actual dyld trace against disjoint complete private packages."""
from pathlib import Path
import re

from macos_loaded_libraries import audit_loaded_libraries


def audit_multiple_loaded_libraries(trace, packages, executable):
    ownership = {}
    for label, (root, manifest) in packages.items():
        for name in manifest['libraries']:
            path = str((Path(root) / 'lib' / name).resolve(strict=True))
            if path in ownership:
                raise ValueError('private runtime packages overlap')
            ownership[path] = label
    filtered = {label: [] for label in packages}
    for match in re.finditer(r'dyld\[\d+\]: <[A-Fa-f0-9-]+> (/[^\r\n]+)', trace):
        image = match[1]
        label = None if image.startswith(('/usr/lib/', '/System/Library/')) else ownership.get(str(Path(image).resolve(strict=True)))
        for target in filtered:
            if label is None or label == target:
                filtered[target].append(match[0])
    return {label: audit_loaded_libraries('\n'.join(filtered[label]), root, manifest, executable)
            for label, (root, manifest) in packages.items()}
