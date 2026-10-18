"""Join actual Windows CPU counters to all existing graph/matrix wall records."""
import re
from ggml_matrix_profile import summarize as wall_summary
from ggml_thread_cpu_profile_source import ENVIRONMENT,PREFIX,MATRIX_PREFIX

MAX_LOG_BYTES=64*1024**2
IDENTITY=r' pool=([0-9a-fA-Fx]+) generation=(\d+) thread=(\d+)'
GRAPH=re.compile(re.escape(PREFIX)+r' graph'+IDENTITY+r' tid=(\d+) cycles=(\d+) kernel_100ns=(\d+) user_100ns=(\d+) final_wait_cycles=(\d+) errors=(\d+) error_bits=(\d+) first_error=(\d+)')
OP=re.compile(re.escape(PREFIX)+r' op'+IDENTITY+r' id=(\d+) kernel_cycles=(\d+) barrier_cycles=(\d+)')
MATRIX=re.compile(re.escape(MATRIX_PREFIX)+r' matrix'+IDENTITY+r' node=(\d+) first_cycles=(\d+) second_cycles=(\d+) conversion_cycles=(\d+) inner_wait_cycles=(\d+) chunk_cycles=(\d+) kernel_cycles=(\d+)')
GRAPH_FIELDS=('tid','cycles','kernel_100ns','user_100ns','final_wait_cycles','errors','error_bits','first_error')
STAGES=('first_cycles','second_cycles','conversion_cycles','inner_wait_cycles','chunk_cycles','kernel_cycles')


def environment(original,enabled):
    if type(enabled)is not bool:raise ValueError('explicit CPU-clock observer mode required')
    result={k:v for k,v in original.items()if k.upper()not in
        (ENVIRONMENT,'GEN2_DIAR_CPU_PROFILE','GEN2_DIAR_MATRIX_PROFILE')}
    if enabled:result[ENVIRONMENT]='1'
    return result


def summarize(log,enabled,*,expected_graphs,expected_threads,expected_matrices):
    environment({},enabled)
    if len(log.encode())>MAX_LOG_BYTES:raise ValueError('bounded complete thread CPU trace required')
    base=wall_summary(log,enabled,expected_graphs=expected_graphs,expected_threads=expected_threads,expected_matrices=expected_matrices)
    lines=[line for line in log.splitlines()if PREFIX in line or MATRIX_PREFIX in line]
    if not enabled:
        if lines:raise ValueError('disabled observer emitted CPU counters')
        return base
    if not len(lines)<=expected_graphs*expected_threads*(1025+expected_matrices):
        raise ValueError('bounded CPU counter rows required')
    graphs={};operations={};matrices={}
    for line in lines:
        match=GRAPH.fullmatch(line)
        if match:
            pool,generation,thread,*raw=match.groups();key=(int(pool,16),int(generation),int(thread))
            values=dict(zip(GRAPH_FIELDS,map(int,raw)))
            if (key in graphs or not 0<values['tid']<=0xffffffff or any(not 0<=v<=2**64-1 for v in values.values())
                    or any(values[name]for name in ('errors','error_bits','first_error'))
                    or any(values[name]>6_000_000_000 for name in ('kernel_100ns','user_100ns'))):
                raise ValueError('complete successful graph CPU counter required')
            graphs[key]=values;continue
        match=OP.fullmatch(line)
        if match:
            # Windows %p can omit0x and include hexadecimal letters.
            key=(int(match[1],16),int(match[2]),int(match[3]));op=int(match[4])
            values={'kernel_cycles':int(match[5]),'barrier_cycles':int(match[6])}
            if op in operations.setdefault(key,{})or any(v>2**64-1 for v in values.values()):
                raise ValueError('unique bounded operation counters required')
            operations[key][op]=values;continue
        match=MATRIX.fullmatch(line)
        if not match:raise ValueError('malformed thread CPU row')
        key=(int(match[1],16),int(match[2]),int(match[3]));node=int(match[4]);values=dict(zip(STAGES,map(int,match.groups()[4:])))
        if node in matrices.setdefault(key,{})or any(v>2**64-1 for v in values.values()):
            raise ValueError('unique bounded matrix CPU counters required')
        if sum(values[name]for name in STAGES[:-1])>values['kernel_cycles']:
            raise ValueError('nested matrix cycle regions exceed enclosing kernel')
        matrices[key][node]=values
    expected=set()
    for graph in base['graphs']:
        tids=set()
        for worker in graph['workers']:
            key=(graph['pool'],graph['generation'],worker['thread']);expected.add(key)
            if key not in graphs or key not in operations or key not in matrices:
                raise ValueError('missing CPU graph/operation/matrix coverage')
            cpu=graphs[key]
            if cpu['tid']in tids:raise ValueError('distinct actual worker thread identities required')
            tids.add(cpu['tid'])
            if (set(operations[key])!={op['id']for op in worker['operations']}
                    or set(matrices[key])!={row['node']for row in worker['matrices']}):
                raise ValueError('exact wall/cycle record identity coverage required')
            if sum(sum(v.values())for v in operations[key].values())+cpu['final_wait_cycles']>cpu['cycles']:
                raise ValueError('worker cycle regions exceed enclosing graph')
            worker['thread_cpu']=cpu
            for op in worker['operations']:op.update(operations[key][op['id']])
            for row in worker['matrices']:
                observed=matrices[key][row['node']]
                if ((not row['first'][0]and observed['first_cycles'])or(not row['second'][0]and observed['second_cycles'])
                        or(not row['conversion'][0]and observed['conversion_cycles'])
                        or(not row['chunks'][0]and observed['chunk_cycles'])
                        or(row['route']not in (3,4,5)and observed['inner_wait_cycles'])):
                    raise ValueError('CPU counters without matching dispatched work')
                row.update(observed)
            matrix_op,=[op for op in worker['operations']if op['name']=='MUL_MAT']
            if sum(row['kernel_cycles']for row in worker['matrices'])!=matrix_op['kernel_cycles']:
                raise ValueError('matrix cycles must cover enclosing MUL_MAT roots')
    if any(set(rows)!=expected for rows in (graphs,operations,matrices)):
        raise ValueError('orphan CPU counter identities')
    base['limits']='Windows CPU kernel/user100ns units and cycle counts remain separate from wall microseconds. No cycles-to-time conversion. Nested cycles and overlapping worker walls are not added. CPU counters include stalls/spinning/kernel overhead; no pure arithmetic, causal scheduling, full-model speed or release inference. Observer bound remains mandatory.'
    return base
