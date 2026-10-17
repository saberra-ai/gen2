"""Wait for the verified live long controller, then execute the frozen mixed CLI."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
long = cache / 'pooled-long-local-v1'
protocol = repo / 'tools/nemotron-diarization/macos-pooled-mixed.protocol.json'
assert hashlib.sha256(protocol.read_bytes()).hexdigest() == 'c40578e5fd6bc57003f6a556cd5838cbc5b12bede092437bb85b050bdfadf3ed'
record_path = cache / 'pooled-mixed-after-long-v1.json'
assert not record_path.exists()
record = {'schema_version': 1, 'state': 'waiting_for_live_long_controller', 'long_controller_pid': 69232,
          'long_output': str(long), 'mixed_protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest()}
def persist():
    temporary = record_path.with_suffix('.partial')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(record_path)
persist()
try:
    started = time.monotonic()
    while True:
        status = subprocess.run(['/bin/ps', '-p', '69232', '-o', 'command='], capture_output=True, text=True)
        if status.returncode != 0:
            break
        if ('qualify_macos_pooled_long.py ' not in status.stdout
                or str(long) not in status.stdout):
            raise ValueError('long-controller PID identity changed')
        if time.monotonic() - started > 2400:
            raise TimeoutError('long controller still live; no mixed run started')
        time.sleep(2)
    q = json.loads((long / 'qualification.json').read_text())
    assert q['source_commit'] == 'aeb6ba813d61653ebdac7273f2a4b404c606f142'
    assert q['pooled_30m_1h_passed'] is True and 'error' not in q and q['storage']['released']
    assert [c['name'] for c in q['commands']] == ['model-free-controls', 'runtime-controls', 'version-only',
        '30m-native', '30m-public', '60m-native', '60m-public']
    assert all(c['returncode'] == 0 for c in q['commands'])
    record['long_qualification_sha256'] = hashlib.sha256((long / 'qualification.json').read_bytes()).hexdigest()
    record['wait_seconds'] = time.monotonic() - started
    command = [sys.executable, str(repo / 'tools/nemotron-diarization/qualify_macos_pooled_mixed.py'),
        '--protocol', str(protocol),
        '--producer-zip', str(cache / 'pooled-macos-construction-ci-v1/artifact.zip'),
        '--producer-metadata', str(cache / 'pooled-macos-construction-ci-v1/download-identity.json'),
        '--public-helpers', str(cache / 'pooled-public-local-v1/pooled-public-helpers.tar.gz'),
        '--public-helpers-manifest', str(cache / 'pooled-public-local-v1/pooled-public-helpers-files.json'),
        '--public-qualification', str(cache / 'pooled-public-local-v1/qualification.json'),
        '--source-models', str(cache / 'valid-mel-mixed-inputs-v1/work'),
        '--output', str(cache / 'pooled-mixed-local-v1')]
    record.update(state='running_frozen_mixed_qualification', argv=command)
    persist()
    with (cache / 'pooled-mixed-local-v1.controller.log').open('xb') as output:
        process = subprocess.Popen(command, cwd=repo, stdout=output, stderr=subprocess.STDOUT)
        record['mixed_controller_pid'] = process.pid
        persist()
        record['returncode'] = process.wait()
    record['state'] = 'completed' if record['returncode'] == 0 else 'failed'
    if record['returncode']:
        raise RuntimeError('frozen mixed controller failed; no automatic retry')
except BaseException as error:
    record['state'] = 'failed'
    record['error'] = repr(error)
    raise
finally:
    persist()
