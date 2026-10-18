"""Join graph-only CPU durations to the complete unchanged batched wall trace."""
import re

from ggml_graph_cpu_profile_source import ENVIRONMENT, PREFIX
from ggml_matrix_profile import summarize as wall_summary

MAX_LOG_BYTES = 64 * 1024**2
GRAPH = re.compile(re.escape(PREFIX) + r' graph pool=([0-9a-fA-Fx]+) generation=(\d+) thread=(\d+)'
                   r' tid=(\d+) kernel_100ns=(\d+) user_100ns=(\d+) errors=(\d+) error_bits=(\d+) first_error=(\d+)')
FIELDS = ('tid', 'kernel_100ns', 'user_100ns', 'errors', 'error_bits', 'first_error')
PROFILE_KEYS = {ENVIRONMENT, 'GEN2_DIAR_CPU_PROFILE', 'GEN2_DIAR_MATRIX_PROFILE',
                'GEN2_DIAR_THREAD_CPU_PROFILE'}


def environment(original, enabled):
    if type(enabled) is not bool:
        raise ValueError('explicit graph CPU observer mode required')
    result = {key: value for key, value in original.items() if key.upper() not in PROFILE_KEYS}
    if enabled:
        result[ENVIRONMENT] = '1'
    return result


def summarize(log, enabled, *, expected_graphs, expected_threads, expected_matrices):
    environment({}, enabled)
    if len(log.encode()) > MAX_LOG_BYTES:
        raise ValueError('bounded complete graph CPU trace required')
    if '[DEBUG-thread-cpu-v1]' in log or '[DEBUG-matrix-cpu-v1]' in log:
        raise ValueError('historical per-region CPU counters are not graph-only evidence')
    base = wall_summary(log, enabled, expected_graphs=expected_graphs,
                        expected_threads=expected_threads, expected_matrices=expected_matrices)
    lines = [line for line in log.splitlines() if PREFIX in line]
    if not enabled:
        if lines:
            raise ValueError('disabled observer emitted graph CPU counters')
        return base
    if len(lines) != expected_graphs * expected_threads:
        raise ValueError('one CPU row per expected worker graph required')
    observed = {}
    for line in lines:
        match = GRAPH.fullmatch(line)
        if not match:
            raise ValueError('malformed graph CPU row')
        key = (int(match[1], 16), int(match[2]), int(match[3]))
        values = dict(zip(FIELDS, map(int, match.groups()[3:]), strict=True))
        if (key in observed or not 0 < values['tid'] <= 0xffffffff
                or any(value > 2**64 - 1 for value in values.values())
                or any(values[name] for name in ('errors', 'error_bits', 'first_error'))
                or any(values[name] > 6_000_000_000 for name in ('kernel_100ns', 'user_100ns'))):
            raise ValueError('unique successful bounded graph CPU row required')
        observed[key] = values
    expected = set()
    for graph in base['graphs']:
        tids = set()
        for worker in graph['workers']:
            key = (graph['pool'], graph['generation'], worker['thread'])
            expected.add(key)
            if key not in observed:
                raise ValueError('CPU row is not joined to its wall worker graph')
            value = observed[key]
            if value['tid'] in tids:
                raise ValueError('distinct executing thread IDs required within each graph')
            tids.add(value['tid'])
            worker['graph_cpu'] = value
    if set(observed) != expected:
        raise ValueError('orphan graph CPU identities')
    base['limits'] = ('Graph-only kernel/user CPU durations retain Windows 100ns units. '
                      'No operation/matrix CPU attribution, cycles, exact descheduled time, '
                      'pure arithmetic or production speed inference. CPU/wall capture boundaries '
                      'and granularity differ. Overlapping workers and nested wall regions are not '
                      'added. Both original observer timing predicates must pass before interpretation.')
    return base
