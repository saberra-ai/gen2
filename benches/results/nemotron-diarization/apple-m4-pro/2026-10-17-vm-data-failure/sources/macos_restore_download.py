"""Bounded one-shot acquisition of an exact public Apple restore image."""
import hashlib
import os
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        return None


def acquire_restore(artifact, directory, *, guard, record, persist, deadline_seconds, read_timeout_seconds):
    name = artifact['name']
    url = urllib.parse.urlsplit(artifact['url'])
    if (Path(name).name != name or name in {'.', '..'} or not name.endswith('.ipsw')
            or type(artifact['bytes']) is not int or artifact['bytes'] <= 0
            or not re.fullmatch('[0-9a-f]{64}', artifact['sha256'])
            or url.scheme != 'https' or url.netloc != 'updates.cdn-apple.com'
            or url.query or url.fragment or Path(url.path).name != name
            or not 0 < read_timeout_seconds <= 30 or deadline_seconds <= 0):
        raise ValueError('exact public Apple restore URL/size/hash and bounded deadlines required')
    target = Path(directory) / name
    partial = target.with_name(name + '.partial')
    if target.exists() or target.is_symlink() or partial.exists() or partial.is_symlink():
        raise FileExistsError('fresh restore destination required; no silent retry or overwrite')
    started = time.monotonic()
    deadline = started + deadline_seconds
    digest = hashlib.sha256()
    record.update({'url': artifact['url'], 'expected': {k: artifact[k] for k in ['bytes', 'sha256']},
                   'bytes_received': 0, 'bytes_written': 0, 'complete': False, 'partial': str(partial)})
    persist()

    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError('restore download total deadline exceeded')
        if guard.check() < guard.running_minimum:
            raise OSError('restore download running storage floor exceeded')

    try:
        check()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        request = urllib.request.Request(artifact['url'], headers={'Accept-Encoding': 'identity'}, method='GET')
        with opener.open(request, timeout=min(read_timeout_seconds, deadline_seconds)) as response:
            record['http_status'] = response.status
            record['headers'] = {key: response.headers.get(key) for key in
                ['Content-Length', 'Content-Type', 'Content-Encoding', 'ETag', 'x-amz-meta-digest-sha256']}
            if response.status != 200 or response.geturl() != artifact['url']:
                raise ValueError('non-200 or redirected restore response')
            length = response.headers.get('Content-Length')
            if length is not None and int(length) != artifact['bytes']:
                raise ValueError('restore response length differs from pin')
            if response.headers.get('Content-Encoding', 'identity') != 'identity':
                raise ValueError('encoded restore response refused')
            fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, 'wb', buffering=0) as output:
                published_bytes, published_at = 0, time.monotonic()
                while True:
                    check()
                    block = response.read1(min(1024**2, artifact['bytes'] - record['bytes_received'] + 1))
                    record['bytes_received'] += len(block)
                    check()
                    if not block:
                        break
                    if record['bytes_received'] > artifact['bytes']:
                        raise ValueError('restore body exceeds pinned size')
                    offset = 0
                    while offset < len(block):
                        count = output.write(block[offset:])
                        if not count:
                            raise OSError('restore file write made no progress')
                        digest.update(block[offset:offset + count])
                        offset += count
                        record['bytes_written'] += count
                    now = time.monotonic()
                    if now - published_at >= 1 or record['bytes_written'] - published_bytes >= 16 * 1024**2:
                        persist()
                        published_bytes, published_at = record['bytes_written'], now
                os.fsync(output.fileno())
        actual = {'bytes': record['bytes_written'], 'sha256': digest.hexdigest()}
        record['actual'] = actual
        if actual != record['expected']:
            raise ValueError('restore body size/hash differs from pin')
        check()
        # Hard-link publication refuses a concurrently created target instead of replacing it.
        os.link(partial, target)
        partial.unlink()
        record['complete'] = True
        record['path'] = str(target)
        record['wall_seconds'] = time.monotonic() - started
        persist()
        return target
    except BaseException as error:
        record['error'] = repr(error)
        record['prefix_sha256'] = digest.hexdigest()
        record['wall_seconds'] = time.monotonic() - started
        guard.release()
        persist()
        raise


def main():
    """Worker only: the parent owns the physical reserve and hard wall timeout."""
    import argparse
    import json
    import shutil
    from qualification_storage import atomic_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    output = args.output.resolve(strict=True)
    if output != Path(protocol['output_parent']).resolve(strict=True) / protocol['output_name']:
        raise ValueError('exact supervised output required')
    record = {'storage_checks': 0}

    class SpaceMonitor:
        running_minimum = protocol['storage']['running_minimum']

        def check(self):
            free = shutil.disk_usage(output).free
            record['storage_checks'] += 1
            record['minimum_observed_free_bytes'] = min(free, record.get('minimum_observed_free_bytes', free))
            return free

        def release(self):
            # Only the supervising owner may release its physical reserve.
            pass

    acquire_restore(protocol['restore_image'], output, guard=SpaceMonitor(), record=record,
        persist=lambda: atomic_json(output / 'download.json', record), **protocol['download_limits'])


if __name__ == '__main__':
    main()
