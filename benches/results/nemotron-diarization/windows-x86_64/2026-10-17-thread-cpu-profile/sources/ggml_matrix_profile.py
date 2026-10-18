"""Admit complete per-root matrix paths while keeping worker walls separate."""
import re
from ggml_cpu_profile import PREFIX as CPU_PREFIX, summarize as cpu_summary
from ggml_matrix_profile_source import ENVIRONMENT, PREFIX, MAX_MATRICES

MAX_LOG_BYTES=64*1024**2
ROUTES={1:'hadamard',2:'sgemm_direct',3:'iq_panel',4:'sgemm_converted',5:'vec_dot',6:'extra',7:'fused'}
FIELDS=('types','names','a','b','c','sa','sb','sc','dot','route','first','second','conversion','inner_wait_us','chunks','kernel_us')
ROW=re.compile(re.escape(PREFIX)+r' matrix pool=([0-9a-fA-Fx]+) generation=(\d+) thread=(\d+) node=(\d+) (.+)')


def environment(original,enabled):
    if type(enabled) is not bool:raise ValueError('explicit matrix observer mode required')
    result={k:v for k,v in original.items() if k.upper() not in (ENVIRONMENT,'GEN2_DIAR_CPU_PROFILE')}
    if enabled:result[ENVIRONMENT]='1'
    return result


def integers(value,count,*,minimum=0,maximum=2**63-1):
    parts=value.split(',')
    if len(parts)!=count or any(not re.fullmatch(r'-?\d+',part) for part in parts):raise ValueError('complete integer tuple required')
    result=list(map(int,parts))
    if any(not minimum<=n<=maximum for n in result):raise ValueError('bounded integer tuple required')
    return result


def summarize(log,enabled,*,expected_graphs,expected_threads,expected_matrices):
    environment({},enabled)
    if (type(expected_matrices) is not int or not 1<=expected_matrices<=MAX_MATRICES
            or len(log.encode())>MAX_LOG_BYTES):raise ValueError('bounded complete matrix trace required')
    cpu=cpu_summary('\n'.join(line for line in log.splitlines() if CPU_PREFIX in line),enabled,
                    expected_graphs=expected_graphs,expected_threads=expected_threads)
    lines=[line for line in log.splitlines() if PREFIX in line]
    if not enabled:
        if lines:raise ValueError('disabled matrix observer emitted rows')
        return {'enabled':False,'cpu':cpu,'graphs':[]}
    if len(lines)!=expected_graphs*expected_threads*expected_matrices:raise ValueError('complete matrix count required')
    records={}
    for line in lines:
        match=ROW.fullmatch(line)
        if not match:raise ValueError('malformed or overflowing matrix row')
        pool,generation,thread,node,text=match.groups();key=(int(pool,16),int(generation),int(thread))
        pairs=[part.split('=',1)for part in text.split()]
        if any(len(pair)!=2 for pair in pairs) or [p[0]for p in pairs]!=list(FIELDS):raise ValueError('exact ordered matrix fields required')
        fields=dict(pairs)
        value={'node':int(node)}
        value['types']=integers(fields['types'],3,maximum=255)
        value['names']=fields['names'].split(',')
        if len(value['names'])!=3 or any(not re.fullmatch(r'[A-Za-z0-9_]+',n)for n in value['names']):raise ValueError('three actual type names required')
        for code,name in zip(value['types'],value['names']):
            if code in (0,1,8,30) and name!={0:'f32',1:'f16',8:'q8_0',30:'bf16'}[code]:raise ValueError('canonical common type identity required')
        for name in ('a','b','c','sa','sb','sc'):value[name]=integers(fields[name],4,minimum=1)
        for name in ('dot','route'):value[name]=integers(fields[name],1,minimum=-1,maximum=255)[0]
        if value['route'] not in ROUTES:raise ValueError('classified matrix route required')
        for name in ('first','second'):
            stage=integers(fields[name],4,maximum=600000000)
            if stage[0]!=stage[1]+stage[2] or stage[2]>1 or (stage[0]==0 and stage[3]!=0):raise ValueError('exact SGEMM attempt coverage required')
            value[name]=stage
        value['conversion']=integers(fields['conversion'],2,maximum=600000000)
        if value['conversion'][0] not in (0,1) or (value['conversion'][0]==0 and value['conversion'][1]):raise ValueError('exact conversion observation required')
        value['chunks']=integers(fields['chunks'],2,maximum=600000000)
        if value['chunks'][0]==0 and value['chunks'][1]:raise ValueError('chunk time without work')
        for name in ('kernel_us','inner_wait_us'):value[name]=integers(fields[name],1,maximum=600000000)[0]
        if sum([value['first'][3],value['second'][3],value['conversion'][1],value['inner_wait_us'],value['chunks'][1]])>value['kernel_us']:raise ValueError('matrix regions exceed enclosing kernel wall')
        if value['a'][0]!=value['b'][0] or value['c']!=[value['a'][1],*value['b'][1:]]:raise ValueError('matrix shape identity changed')
        if any(value['b'][d]%value['a'][d] for d in (2,3)):raise ValueError('invalid broadcast factors')
        route=value['route'];first=value['first'];second=value['second']
        batches=value['b'][2]*value['b'][3]
        if first[0]>batches or second[0]>batches:raise ValueError('SGEMM attempts exceed batch coverage')
        if route in (2,3,4,5) and value['dot']<0:raise ValueError('entered matrix routine requires observed dot type')
        if route==2 and (first[:3]!=[batches,batches,0] or second[0] or value['conversion'][0] or value['chunks'][0] or value['inner_wait_us']):raise ValueError('direct SGEMM route mismatch')
        if route==4 and (second[:3]!=[batches,batches,0] or value['conversion'][0]!=1 or value['dot']==value['types'][1] or value['chunks'][0]):raise ValueError('converted SGEMM route mismatch')
        if route in (3,5) and value['conversion'][0] != int(value['dot']!=value['types'][1]):raise ValueError('actual conversion does not match dot-type change')
        if route in (3,4,5) and first[0] and first[2]!=1:raise ValueError('first SGEMM fallback requires observed rejection')
        if route==5 and second[0] and second[2]!=1:raise ValueError('second SGEMM fallback requires observed rejection')
        current=records.setdefault(key,{})
        if value['node'] in current:raise ValueError('duplicate matrix root')
        current[value['node']]=value
    identities=set()
    for graph in cpu['graphs']:
        reference=None
        for worker in graph['workers']:
            key=(graph['pool'],graph['generation'],worker['thread']);identities.add(key)
            rows=sorted(records.get(key,{}).values(),key=lambda row:row['node'])
            operation,=[op for op in worker['operations'] if op['name']=='MUL_MAT']
            if (len(rows)!=expected_matrices or operation['roots']!=expected_matrices
                    or any(not 0<=row['node']<worker['nodes'] for row in rows)
                    or sum(row['kernel_us'] for row in rows)!=operation['kernel_us']):raise ValueError('matrix roots do not cover enclosing operation')
            signature=[{k:row[k]for k in ('node','types','names','a','b','c','sa','sb','sc')}for row in rows]
            # Worker-local paths and scheduling counters remain separate observations.
            if reference is not None and signature!=reference:raise ValueError('workers disagree on matrix shape/type')
            reference=signature
            worker['matrices']=rows
    if set(records)!=identities:raise ValueError('orphan matrix rows')
    return {'enabled':True,'cpu':cpu,'graphs':cpu['graphs'],'limits':'Per-worker wall regions overlap and include scheduling/internal waits/observer effects. Never sum workers or infer pure arithmetic or release performance.'}
