"""Admit complete per-worker diagnostic timing without adding overlapping times."""
import re

from ggml_cpu_profile_source import ENVIRONMENT, PREFIX


IDENTITY = r' pool=([0-9A-Fa-fx]+) generation=(\d+) thread=(\d+)'
GRAPH = re.compile(re.escape(PREFIX) + r' graph' + IDENTITY
    + r' threads=(\d+) nodes=(\d+) roots=(\d+) fused=(\d+) skipped=(\d+)'
    + r' topology=([0-9a-f]{16}) execution=([0-9a-f]{16}) elapsed_us=(\d+) final_wait_us=(\d+)')
OP = re.compile(re.escape(PREFIX) + r' op' + IDENTITY
    + r' id=(\d+) name=([A-Z0-9_]+) roots=(\d+) fused=(\d+) kernel_us=(\d+) barrier_us=(\d+)')
MAX_LOG_BYTES = 16 * 1024**2

# GGML_OP_NAME in the canonical nineteen-patch ggml/src/ggml.c.
CANONICAL_GGML_C_SHA256 = 'c642c7ab0b3684717cd8ae5c1803e923b1a24736a1fc07b38b6db68744575c86'
OPERATION_NAMES = tuple((
    'NONE DUP ADD ADD_ID ADD1 ACC SUB MUL DIV SQR SQRT LOG SIN COS SUM SUM_ROWS CUMSUM MEAN ARGMAX '
    'COUNT_EQUAL REPEAT REPEAT_BACK CONCAT SILU_BACK NORM RMS_NORM RMS_NORM_BACK GROUP_NORM L2_NORM '
    'MUL_MAT MUL_MAT_ID OUT_PROD SCALE SET CPY CONT RESHAPE VIEW PERMUTE TRANSPOSE GET_ROWS '
    'GET_ROWS_BACK SET_ROWS DIAG DIAG_MASK_INF DIAG_MASK_ZERO SOFT_MAX SOFT_MAX_BACK ROPE ROPE_BACK '
    'CLAMP CONV_TRANSPOSE_1D IM2COL IM2COL_BACK IM2COL_3D COL2IM_1D CONV_2D CONV_3D CONV_2D_DW '
    'CONV_TRANSPOSE_2D POOL_1D POOL_2D POOL_2D_BACK UPSCALE PAD PAD_REFLECT_1D ROLL ARANGE '
    'TIMESTEP_EMBEDDING ARGSORT TOP_K LEAKY_RELU TRI FILL FLASH_ATTN_EXT FLASH_ATTN_BACK SSM_CONV '
    'SSM_SCAN WIN_PART WIN_UNPART GET_REL_POS ADD_REL_POS RWKV_WKV6 GATED_LINEAR_ATTN RWKV_WKV7 '
    'SOLVE_TRI GATED_DELTA_NET LIGHTNING_INDEXER DSV4_HC_COMB DSV4_HC_PRE DSV4_HC_POST UNARY '
    'MAP_CUSTOM1 MAP_CUSTOM2 MAP_CUSTOM3 CUSTOM CROSS_ENTROPY_LOSS CROSS_ENTROPY_LOSS_BACK '
    'OPT_STEP_ADAMW OPT_STEP_SGD GLU FUSED_ATTN CONV1D_FUSED '
).split())


def environment(original, enabled):
    if type(enabled) is not bool:
        raise ValueError('explicit boolean CPU timing mode required')
    value = {key: val for key, val in original.items() if key.upper() != ENVIRONMENT}
    if enabled:
        value[ENVIRONMENT] = '1'
    return value


def summarize(log, enabled, *, expected_graphs, expected_threads):
    """Require every worker/operation and retain times separately by worker."""
    environment({}, enabled)
    if (type(expected_graphs) is not int or not 1 <= expected_graphs <= 1024
            or type(expected_threads) is not int or not 1 <= expected_threads <= 4
            or len(log.encode('utf-8')) > MAX_LOG_BYTES):
        raise ValueError('bounded CPU profiling coverage required')
    lines = [line for line in log.splitlines() if PREFIX in line]
    if not enabled:
        if lines:
            raise ValueError('control unexpectedly emitted CPU operation timing')
        return {'enabled': False, 'graphs': []}
    if not 1 <= len(lines) <= expected_graphs * expected_threads * 1025:
        raise ValueError('bounded complete CPU profiling rows required')
    graphs, operations, order = {}, {}, []
    for line in lines:
        match = GRAPH.fullmatch(line)
        if match:
            pool, generation, thread, threads, nodes, roots, fused, skipped, topology, execution, elapsed, final = match.groups()
            key = (int(pool, 16), int(generation), int(thread))
            if (key in graphs or not 0 < key[0] <= 2**64 - 1
                    or not 0 <= key[1] <= 2**31 - 1 or not 0 <= key[2] < expected_threads
                    or int(threads) != expected_threads or not 1 <= int(nodes) <= 65536
                    or not 1 <= int(roots) <= int(nodes)
                    or int(roots) + int(fused) + int(skipped) != int(nodes)
                    or not 0 <= int(final) <= int(elapsed) <= 600 * 1000000):
                raise ValueError('invalid or duplicate CPU graph coverage')
            graphs[key] = {'nodes': int(nodes), 'roots': int(roots), 'fused': int(fused),
                'skipped': int(skipped), 'topology': topology, 'execution': execution,
                'elapsed_us': int(elapsed), 'final_wait_us': int(final)}
            if key[2] == 0:
                order.append(key[:2])
            continue
        match = OP.fullmatch(line)
        if not match:
            raise ValueError('malformed CPU profiling row')
        pool, generation, thread, op, name, roots, fused, kernel, barrier = match.groups()
        key = (int(pool, 16), int(generation), int(thread))
        op = int(op)
        if (not 0 <= op < len(OPERATION_NAMES) or OPERATION_NAMES[op] != name
                or not 0 < int(roots) <= 65536 or not 0 <= int(fused) <= 65536
                or not 0 <= int(kernel) <= 600 * 1000000 or not 0 <= int(barrier) <= 600 * 1000000
                or op in operations.setdefault(key, {})):
            raise ValueError('invalid or duplicate CPU operation row')
        operations[key][op] = {'id': op, 'name': name, 'roots': int(roots), 'fused': int(fused),
                               'kernel_us': int(kernel), 'barrier_us': int(barrier)}
    groups = {key[:2] for key in graphs}
    if (len(groups) != expected_graphs or len(order) != expected_graphs
            or len(graphs) != expected_graphs * expected_threads or set(operations) != set(graphs)):
        raise ValueError('missing graph, worker or operation coverage')
    result, names = [], {}
    for identity in order:
        workers, shape = [], None
        for thread in range(expected_threads):
            key = (*identity, thread)
            if key not in graphs:
                raise ValueError('missing CPU graph worker')
            graph, ops = graphs[key], list(operations[key].values())
            current = {name: graph[name] for name in ('nodes', 'roots', 'fused', 'skipped', 'topology', 'execution')}
            current['operations'] = sorted((op['id'], op['name'], op['roots'], op['fused']) for op in ops)
            if shape is not None and current != shape:
                raise ValueError('workers disagree on graph or fusion execution')
            shape = current
            if (sum(op['roots'] for op in ops) != graph['roots']
                    or sum(op['fused'] for op in ops) != graph['fused']
                    or sum(op['kernel_us'] + op['barrier_us'] for op in ops)
                        + graph['final_wait_us'] > graph['elapsed_us']):
                raise ValueError('CPU timing coverage exceeds its enclosing worker interval')
            for op in ops:
                if op['id'] in names and names[op['id']] != op['name']:
                    raise ValueError('CPU operation identity changed')
                names[op['id']] = op['name']
            workers.append({'thread': thread, **graph, 'operations': sorted(ops, key=lambda op: op['id'])})
        result.append({'pool': identity[0], 'generation': identity[1], 'workers': workers})
    return {'enabled': True, 'graph_count': len(result), 'threads': expected_threads, 'graphs': result,
        'limits': 'Worker wall intervals overlap. Kernel calls include internal synchronization and OS scheduling; outer barriers and final wait are separate. Do not sum workers into graph wall time or infer arithmetic cost.'}
