"""Copy first-bound local replay evidence into one closed repository receipt."""
import argparse
import hashlib
import json
from pathlib import Path
import stat


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def capture(path, expected):
    if not stat.S_ISREG(path.lstat().st_mode) or any(p.is_symlink() for p in path.parents):
        raise ValueError('nonsymlink regular evidence required')
    raw = path.read_bytes()
    if identity(raw) != expected:
        raise ValueError('first-bound evidence changed: ' + str(path))
    return raw


def relative_name(name):
    path = Path(name)
    if not path.parts or path.is_absolute() or '..' in path.parts or path.as_posix() != name:
        raise ValueError('canonical relative output required')
    return path


def unique_names(names):
    seen = {}
    for name in names:
        path = relative_name(name)
        for item in (path, *path.parents):
            if not item.parts:
                continue
            folded = item.as_posix().casefold()
            prior = seen.get(folded)
            if prior is not None and prior != item.as_posix():
                raise ValueError('case-colliding publication path')
            seen[folded] = item.as_posix()


def verify_tree(root, files):
    actual, directories, names = set(), set(), []
    expected_directories = {parent.as_posix() for name in files for parent in Path(name).parents if parent.parts}
    for path in root.rglob('*'):
        name = path.relative_to(root).as_posix()
        names.append(name)
        mode = path.lstat().st_mode
        if stat.S_ISREG(mode):
            actual.add(name)
        elif stat.S_ISDIR(mode):
            directories.add(name)
        else:
            raise ValueError('nonregular publication member')
    unique_names(names)
    if actual != set(files) or directories != expected_directories:
        raise ValueError('publication membership changed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    raw = args.protocol.read_bytes()
    first_protocol = identity(raw)
    if first_protocol['sha256'] != args.sha256:
        raise ValueError('publication protocol changed')
    pre = json.loads(raw)
    if pre['schema_version'] != 1:
        raise ValueError('fixed publication format required')
    targets = pre['files']
    if {'collection.json', 'qualification.json', 'pre-execution.json'} & set(targets):
        raise ValueError('reserved publication member')
    unique_names([*targets, 'collection.json', 'qualification.json', 'pre-execution.json'])
    if not any(row['source'] == str(Path(__file__).resolve()) for row in targets.values()):
        raise ValueError('publisher must be first-bound')
    # Admit every input and path before creating any destination.
    for name, row in targets.items():
        relative_name(name)
        capture(Path(row['source']), row['identity'])
    replay = json.loads(capture(Path(pre['replay']['source']), pre['replay']['identity']))
    if (replay['full_sequence_bit_exact'] is not True or
            replay['timing_comparison']['bounded_observer_interpretation_supported'] is not False or
            replay['production_or_release_qualified'] is not False or
            replay['saved_windows_job_commands_verified'] != 59 or
            replay['local_command_sidecars_verified'] != 470 or
            replay['source_only_command_sidecars_verified'] != 11):
        raise ValueError('exact independently replayed negative diagnostic required')
    dest = Path(pre['destination'])
    if dest.exists() or any(path.is_symlink() for path in dest.parents):
        raise ValueError('new nonsymlink destination required')
    dest.mkdir()
    copied = {}
    for name, row in targets.items():
        data = capture(Path(row['source']), row['identity'])
        target = dest / relative_name(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
        capture(target, row['identity'])
        copied[name] = row['identity']
    with (dest / 'pre-execution.json').open('xb') as stream:
        stream.write(raw)
    copied['pre-execution.json'] = first_protocol
    for name, row in targets.items():
        capture(Path(row['source']), row['identity'])
        capture(dest / name, row['identity'])
    capture(args.protocol, first_protocol)
    qualification = {'schema_version': 1, 'publication_completed': True,
        'source_commit_before_import': pre['source_commit'], 'result_commit_imported': pre['result_commit'],
        'pre_execution_protocol': first_protocol, 'replay': pre['replay']['identity'],
        'input_members_copied': len(targets), 'saved_windows_job_commands_verified': 59,
        'local_command_sidecars_verified': 470, 'source_only_command_sidecars_verified': 11,
        'model_free_vectors': 144, 'model_cases': 4, 'full_sequence_bit_exact': True,
        'timing_comparison': replay['timing_comparison'],
        'new_compiler_sdk_model_calls': 0, 'production_or_release_qualified': False,
        'limits': replay['limits']}
    data = (json.dumps(qualification, indent=2) + '\n').encode()
    with (dest / 'qualification.json').open('xb') as stream:
        stream.write(data)
    copied['qualification.json'] = identity(data)
    verify_tree(dest, copied)
    for name, expected in copied.items():
        capture(dest / name, expected)
    manifest = (json.dumps({'schema_version': 1, 'files': copied}, indent=2) + '\n').encode()
    with (dest / 'collection.json').open('xb') as stream:
        stream.write(manifest)
    print(json.dumps({'destination': str(dest), 'members': len(copied) + 1,
        'bytes_without_manifest': sum(value['bytes'] for value in copied.values()),
        'manifest': identity(manifest), 'production_or_release_qualified': False}))


if __name__ == '__main__':
    main()
