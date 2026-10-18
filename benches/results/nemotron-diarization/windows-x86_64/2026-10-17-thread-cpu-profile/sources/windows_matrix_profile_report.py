"""Retain complete decoded profiles once, with bounded compact saved evidence."""
import hashlib
import json
from pathlib import Path

MAX_PROFILE_BYTES=64*1024**2


def payload(profile):
    # The parser's CPU and matrix views share the same graph/worker objects.
    # Keep every graph/operation/matrix record once; raw logs remain separate.
    value={key:item for key,item in profile.items() if key!='cpu'}
    raw=(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n').encode()
    if len(raw)>MAX_PROFILE_BYTES:raise ValueError('bounded compact complete matrix profile required')
    return raw


def record(path,profile,raw):
    return {'file':Path(path).name,'enabled':profile['enabled'],'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
            'graphs':len(profile['graphs']),
            'worker_matrix_records':sum(len(worker['matrices']) for graph in profile['graphs']for worker in graph['workers'])}


def save_profile(path,profile):
    path=Path(path);raw=payload(profile)
    with path.open('xb')as stream:stream.write(raw)
    return record(path,profile,raw)


def admit_saved_profile(path,expected,profile):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>MAX_PROFILE_BYTES:
        raise ValueError('bounded regular saved matrix profile required')
    raw=payload(profile)
    if path.read_bytes()!=raw or record(path,profile,raw)!=expected:
        raise ValueError('saved complete matrix profile differs from raw-trace replay')
    return expected
