"""Audit actual dyld image traces against one relocated package and executable."""
from pathlib import Path
import re

from artifacts import digest


def audit_loaded_libraries(trace, package, manifest, executable, require_runtime=True):
    images = set(re.findall(r'dyld\[\d+\]: <[A-Fa-f0-9-]+> (/[^\r\n]+)', trace))
    if not images:
        raise ValueError('dyld loaded-image telemetry is missing')
    executable = Path(executable).resolve(strict=True)
    root = (Path(package) / 'lib').resolve(strict=True)
    expected = {str(root / name): manifest['files']['lib/' + name]['sha256']
                for name in manifest['libraries']}
    observed = {}
    saw_executable = False
    for image in sorted(images):
        if image.startswith(('/usr/lib/', '/System/Library/')):
            continue
        path = Path(image).resolve(strict=True)
        if path == executable:
            saw_executable = True
            continue
        if str(path) not in expected or digest(path) != expected[str(path)]:
            raise ValueError('unexpected non-system dyld image: ' + image)
        observed[str(path)] = {'sha256': digest(path), 'reported_path': image}
    if not saw_executable:
        raise ValueError('dyld trace lacks the actual executable')
    if require_runtime and set(observed) != set(expected):
        raise ValueError('actual dyld runtime image set differs from the package closure')
    if not require_runtime and observed:
        raise ValueError('model-free command unexpectedly loaded the native runtime')
    return {'executable': str(executable), 'executable_sha256': digest(executable),
            'unique_reported_images': len(images), 'private_libraries': observed,
            'private_library_count': len(observed), 'entire_runtime_required': require_runtime}
