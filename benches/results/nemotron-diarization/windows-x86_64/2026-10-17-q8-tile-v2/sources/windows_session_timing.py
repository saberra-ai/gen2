"""Parse the existing upstream CPU session timers; no runtime source change."""
import hashlib
import math
import re
import stat
import zipfile

from build_valid_mel_candidate import descriptor
from windows_archive import MAX_ENTRIES, MAX_FILE_BYTES, MAX_TOTAL_BYTES, byte_inventory, safe_relative

ORDER = ['control', 'timing', 'timing', 'control']
EXPECTED_SESSIONS = 84  # ceil(6000 valid Mel rows / (9 coarse frames * 8))
EXPECTED_FRONTENDS = 59  # 58 pushes plus final zero-pad frontend call
NUMBER = r'([0-9]+\.[0-9]{2})'
ROW = re.compile(r'^\[timing\] session out=(.+?) nodes=([0-9]+) in=' + NUMBER
                 + r' alloc=' + NUMBER + r' compute=' + NUMBER + r' out=' + NUMBER
                 + r' total=' + NUMBER + r' ms$')
FRONTEND_ROW = re.compile(r'^\[timing\] fe path=cpu n_samples=([0-9]+) n_frames=([0-9]+) = ' + NUMBER + r' ms$')


def profiling_environment(environment, mode):
    if mode not in ('control', 'timing'):
        raise ValueError('explicit profiling mode required')
    result = {key: value for key, value in environment.items()
              if key.upper() not in {'NEMO_SPEECH_TIMING', 'GH_TOKEN', 'GITHUB_TOKEN'}}
    if mode == 'timing':
        result['NEMO_SPEECH_TIMING'] = '1'
    return result


def summarize(log, mode, elapsed_micros):
    profiling_environment({}, mode)
    if type(elapsed_micros) is not int or elapsed_micros <= 0:
        raise ValueError('positive native elapsed microseconds required')
    lines = [line for line in log.splitlines() if '[timing]' in line]
    if mode == 'control':
        if lines:
            raise ValueError('control unexpectedly emitted session timing')
        return {'mode': mode, 'session_timing_rows': 0, 'native_rtf': elapsed_micros / 60e6}
    session_lines = [line for line in lines if line.startswith('[timing] session ')]
    frontend_lines = [line for line in lines if line.startswith('[timing] fe ')]
    if (len(session_lines) != EXPECTED_SESSIONS or len(frontend_lines) != EXPECTED_FRONTENDS
            or len(lines) != EXPECTED_SESSIONS + EXPECTED_FRONTENDS):
        raise ValueError('all84 session and59 CPU frontend timing rows required')
    rows = []
    keys = ['input', 'allocation', 'compute', 'output_and_sync', 'total']
    for line in session_lines:
        match = ROW.fullmatch(line)
        if not match:
            raise ValueError('unexpected or malformed scheduled CPU timing row')
        name, nodes, *values = match.groups()
        numbers = list(map(float, values))
        if (not name.strip() or not 0 < int(nodes) <= 65536
                or not all(math.isfinite(value) and value >= 0 for value in numbers)
                # Four phases and their total each round to0.01ms.
                or numbers[-1] <= 0 or abs(sum(numbers[:-1]) - numbers[-1]) > 5 * .005 + 1e-8):
            raise ValueError('invalid/additively inconsistent native timing')
        rows.append({'output': name.strip(), 'nodes': int(nodes), **dict(zip(keys, numbers))})
    frontends = []
    for line in frontend_lines:
        match = FRONTEND_ROW.fullmatch(line)
        if not match:
            raise ValueError('unexpected or malformed CPU frontend timing row')
        samples, frames, elapsed = int(match[1]), int(match[2]), float(match[3])
        if samples <= 0 or frames <= 0 or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('invalid CPU frontend timing')
        frontends.append({'samples': samples, 'frames': frames, 'elapsed_ms': elapsed})
    totals = {key: sum(row[key] for row in rows) for key in keys}
    wall_ms = elapsed_micros / 1000
    # Each upstream total is rounded to0.01ms. This bounds representation only;
    # it never changes the original strict below-one performance requirement.
    frontend_total = sum(row['elapsed_ms'] for row in frontends)
    timed_total = totals['total'] + frontend_total
    rounding_bound = (EXPECTED_SESSIONS + EXPECTED_FRONTENDS) * .005
    if timed_total > wall_ms + rounding_bound + 1e-6:
        raise ValueError('summed timing regions exceed enclosing native elapsed time')
    return {'mode': mode, 'session_timing_rows': len(rows), 'rows': rows, 'totals_ms': totals,
            'frontend_rows': frontends, 'frontend_timing_rows': len(frontends), 'frontend_total_ms': frontend_total,
            'native_elapsed_ms': wall_ms, 'native_rtf': elapsed_micros / 60e6,
            'total_rounding_bound_ms': rounding_bound,
            'outside_timed_regions_ms': max(0, wall_ms - timed_total),
            'compute_and_output_fraction': (totals['compute'] + totals['output_and_sync']) / wall_ms,
            'frontend_fraction': frontend_total / wall_ms,
            'outside_timed_regions_fraction': max(0, wall_ms - timed_total) / wall_ms,
            'input_and_allocation_fraction': (totals['input'] + totals['allocation']) / wall_ms,
            'limits': 'Existing session timers exclude graph-cache construction and lock acquisition. CPU frontend is separately timed. Outside time also includes state work, queueing and logging. Output region includes synchronization. No per-operator attribution.'}


def file_inventory(root):
    return {name: {key: record[key] for key in ('bytes', 'sha256')}
            for name, record in byte_inventory(root).items() if record['type'] == 'file'}


def extract_delivery(archive, expected_archive, members, destination):
    """Restore exactly the already reviewed GitHub artifact, with Windows paths."""
    if destination.exists() or destination.is_symlink() or descriptor(archive) != expected_archive:
        raise ValueError('fresh destination and exact preregistered ZIP bytes required')
    if (not members or len(members) > MAX_ENTRIES
            or sum(row['bytes'] for row in members.values()) > MAX_TOTAL_BYTES
            or any(type(row['bytes']) is not int or not 0 <= row['bytes'] <= MAX_FILE_BYTES for row in members.values())):
        raise ValueError('bounded source artifact membership required')
    folded = set()
    for name in members:
        safe_relative(name)
        if name.casefold() in folded:
            raise ValueError('case-colliding artifact names')
        folded.add(name.casefold())
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        if len(entries) != len(members) or {row.filename for row in entries} != set(members):
            raise ValueError('exact source artifact member set required')
        for entry in entries:
            mode = stat.S_IFMT(entry.external_attr >> 16)
            if (entry.is_dir() or entry.flag_bits & 1 or mode not in (0, stat.S_IFREG)
                    or entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                    or entry.file_size != members[entry.filename]['bytes']):
                raise ValueError('regular exact-sized source artifact members required')
        destination.mkdir()
        for entry in entries:
            target = destination / entry.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            value, size = hashlib.sha256(), 0
            with bundle.open(entry) as source, target.open('xb') as out:
                while block := source.read(1024**2):
                    size += len(block)
                    if size > members[entry.filename]['bytes']:
                        raise ValueError('artifact member grew beyond declared size')
                    value.update(block)
                    out.write(block)
            if {'bytes': size, 'sha256': value.hexdigest()} != members[entry.filename]:
                raise ValueError('source artifact member bytes differ')
    if file_inventory(destination) != members:
        raise ValueError('restored source artifact inventory differs')
