"""Audited build-only LF overlay for the pinned Windows CMake materializer."""
import difflib
import hashlib
import json


ORIGINAL_SHA256 = '22357d4a5d8a67716daa8b4e374abd63120432bd57540f97d572a525d5354c7b'
WRITE = b'        file(WRITE "${normalized}" "${content}")'
LF_WRITE = b'''        string(SHA256 normalized_expected "${content}")
        file(CONFIGURE OUTPUT "${normalized}" CONTENT "${content}" @ONLY NEWLINE_STYLE UNIX)
        file(SHA256 "${normalized}" normalized_actual)
        if(NOT "${normalized_actual}" STREQUAL "${normalized_expected}")
            message(FATAL_ERROR "materialized patch bytes differ from upstream normalized content")
        endif()'''


def rewrite_materializer(content):
    if content.count(WRITE) != 1:
        raise ValueError('expected exactly one pinned patch materialization site')
    return content.replace(WRITE, LF_WRITE)


def apply_windows_lf_compat(source, output):
    path = source / 'cmake/llama_cpp.cmake'
    original = path.read_bytes()
    before = hashlib.sha256(original).hexdigest()
    if before != ORIGINAL_SHA256:
        raise ValueError('Windows compatibility overlay requires exact pinned CMake source')
    updated = rewrite_materializer(original)
    patch = ''.join(difflib.unified_diff(original.decode().splitlines(keepends=True),
                                       updated.decode().splitlines(keepends=True),
                                       fromfile='a/cmake/llama_cpp.cmake', tofile='b/cmake/llama_cpp.cmake'))
    patch_path = output / 'windows-compat.patch.log'
    patch_path.write_bytes(patch.encode())
    record = {'scope': 'Pinned upstream plus explicit Windows build-only compatibility overlay; no inference or patch-payload edits',
              'path': 'cmake/llama_cpp.cmake', 'upstream_sha256': before,
              'updated_sha256': hashlib.sha256(updated).hexdigest(),
              'overlay_sha256': hashlib.sha256(patch.encode()).hexdigest(),
              'upstream_patch_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in sorted((source / 'patches').glob('*.patch'))},
              'guard': 'Each generated LF patch must hash exactly like its normalized upstream content before git apply.'}
    (output / 'windows-compat.json').write_text(json.dumps(record, indent=2) + '\n')
    path.write_bytes(updated)
    return record
