"""Verify frozen source bytes at a recorded commit without changing checkout."""
import hashlib
import io
import os
import re
import subprocess


def verify_git_sources(repo, commit, expected):
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('full recorded source commit required')
    environment = {**os.environ, 'GIT_NO_LAZY_FETCH': '1', 'GIT_NO_REPLACE_OBJECTS': '1'}
    kind = subprocess.run(['git', '-C', str(repo), 'cat-file', '-t', commit],
                          capture_output=True, env=environment)
    if kind.returncode or kind.stdout.strip() != b'commit':
        raise ValueError('recorded source commit unavailable locally; fetch it before offline admission: ' + commit)
    names = list(expected)
    if any('\n' in name or '\r' in name for name in names):
        raise ValueError('invalid frozen source path')
    queries = ''.join(commit + ':' + name + '\n' for name in names).encode()
    # Check sizes/types before requesting contents; only the frozen byte budget
    # can enter the content response, regardless of the supplied revision.
    checked = subprocess.run(['git', '-C', str(repo), 'cat-file', '--batch-check'],
                             input=queries, capture_output=True, env=environment)
    headers = checked.stdout.splitlines()
    if checked.returncode or len(headers) != len(names):
        raise ValueError('recorded source metadata unavailable')
    for name, header in zip(names, headers):
        fields = header.split()
        if len(fields) != 3 or fields[1] != b'blob' or fields[2] != str(expected[name]['bytes']).encode():
            raise ValueError('frozen source size/type differs at recorded commit: ' + name)
    result = subprocess.run(['git', '-C', str(repo), 'cat-file', '--batch'],
                            input=queries, capture_output=True, env=environment)
    if result.returncode:
        raise ValueError('recorded source contents unavailable')
    stream = io.BytesIO(result.stdout)
    for name, header in zip(names, headers):
        if stream.readline().rstrip(b'\n') != header:
            raise ValueError('recorded source metadata changed')
        raw = stream.read(expected[name]['bytes'])
        if stream.read(1) != b'\n' or hashlib.sha256(raw).hexdigest() != expected[name]['sha256']:
            raise ValueError('frozen source bytes differ at recorded commit: ' + name)
    if stream.read():
        raise ValueError('unexpected recorded source data')
