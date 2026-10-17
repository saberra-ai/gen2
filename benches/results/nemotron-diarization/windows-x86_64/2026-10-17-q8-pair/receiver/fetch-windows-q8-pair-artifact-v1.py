"""Fetch one terminal diagnostic artifact without native execution."""
import hashlib
import os
import selectors
import time
import json
from pathlib import Path
import shutil
import subprocess

root=Path('/Users/victor/.cache/gen2-nemotron3-n0/windows-q8-pair-ci-v1')
run_id=39066832336
job_id=117496146656
source='259966ceca113f50d7d172fa63269b0bcb17b37a'
api='repos/saberra-ai/gen2/actions/'
def get(endpoint):
    return subprocess.check_output(['gh','api',api+endpoint],timeout=180)
def save(name,raw):
    with(root/name).open('xb')as stream:stream.write(raw)
def descriptor(path):
    digest=hashlib.sha256()
    size=0
    with path.open('rb')as stream:
        while block:=stream.read(1024**2):
            size+=len(block)
            digest.update(block)
    return {'bytes':size,'sha256':digest.hexdigest()}

def bounded_download(command, output, stderr_path, report_path, expected_bytes, timeout=600):
    if type(expected_bytes) is not int or not 0 < expected_bytes <= 1024**3:
        raise ValueError('positive admitted archive size at most1GiB required')
    report = {'expected_bytes':expected_bytes,'bytes_received':0,'bytes_written':0,
              'stderr_bytes':0,'completed':False,'process_reaped':False}
    started = time.monotonic()
    process = None
    selector = selectors.DefaultSelector()
    try:
        with output.open('xb') as out, stderr_path.open('xb') as err:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
            report['pid'] = process.pid
            selector.register(process.stdout, selectors.EVENT_READ, 'archive')
            selector.register(process.stderr, selectors.EVENT_READ, 'stderr')
            deadline = started + timeout
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('archive download exceeded deadline')
                events = selector.select(remaining)
                if not events:
                    raise TimeoutError('archive download exceeded deadline')
                for key, event in events:
                    limit = expected_bytes if key.data == 'archive' else 1024**2
                    count = report['bytes_written'] if key.data == 'archive' else report['stderr_bytes']
                    block = os.read(key.fd, min(1024**2, limit-count+1))
                    if not block:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        continue
                    if key.data == 'archive':
                        report['bytes_received'] += len(block)
                    if count + len(block) > limit:
                        raise ValueError('download stream exceeded admitted byte limit: '+key.data)
                    if key.data == 'archive':
                        out.write(block)
                        report['bytes_written'] += len(block)
                    else:
                        err.write(block)
                        report['stderr_bytes'] += len(block)
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError('archive child did not exit before deadline')
            status = process.wait(timeout=remaining)
            report['returncode'] = status
            report['process_reaped'] = True
            if status != 0 or report['bytes_written'] != expected_bytes:
                raise ValueError('download child exit or complete archive size mismatch')
            report['completed'] = True
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
            report['returncode'] = process.wait(timeout=10)
            report['process_reaped'] = True
            for stream in (process.stdout, process.stderr):
                if not stream.closed:
                    stream.close()
        selector.close()
        report['elapsed_seconds'] = time.monotonic()-started
        with report_path.open('x') as stream:
            json.dump(report,stream,indent=2)
            stream.write('\n')

assert root.is_dir() and not root.is_symlink()
assert shutil.disk_usage(root).free>=8*1024**3
raw=get(f'runs/{run_id}')
run=json.loads(raw)
assert run['id']==run_id and run['head_sha']==source and run['status']=='completed'
assert run['conclusion']in('success','failure')
save('producer-run.json',raw)
raw=get(f'jobs/{job_id}')
job=json.loads(raw)
assert job['id']==job_id and job['run_id']==run_id and job['head_sha']==source
assert job['status']=='completed' and job['conclusion']==run['conclusion']
save('job.json',raw)
save('job.log',get(f'jobs/{job_id}/logs'))
raw=get(f'runs/{run_id}/artifacts')
artifacts=json.loads(raw)
artifact,=artifacts['artifacts']
assert artifact['name']=='nemotron-windows-q8-pair' and artifact['expired']is False
assert artifact['workflow_run']['id']==run_id and artifact['workflow_run']['head_sha']==source
assert 0<artifact['size_in_bytes']<=1024**3
save('artifacts.json',raw)
bounded_download(['gh','api',api+f"artifacts/{artifact['id']}/zip"],root/'artifact.zip',
    root/'artifact-download-v2.stderr.log',root/'artifact-download-v2.json',artifact['size_in_bytes'])
assert descriptor(root/'artifact.zip')=={'bytes':artifact['size_in_bytes'],'sha256':artifact['digest'].removeprefix('sha256:')}
print(json.dumps({'run':run_id,'job':job_id,'conclusion':run['conclusion'],'artifact_id':artifact['id'],'archive':descriptor(root/'artifact.zip')}))
