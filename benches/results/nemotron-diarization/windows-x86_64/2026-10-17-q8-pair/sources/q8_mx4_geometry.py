"""Freeze minimum/median/maximum distinct columns from admitted model roots."""

MODEL_RESULT = 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-matrix-batch-profile'
PRIMARY_A = [(2048,512,1,1), (512,2048,1,1), (512,1536,1,1), (512,512,1,1)]
COLUMNS = [13,292,535]
SHAPES = [(1,2,32),(2,3,64),(5,7,96),(7,5,160)]
SHAPES += [(a[1],n,a[0]) for a in PRIMARY_A for n in COLUMNS]
SHAPES += [(512,n,1024) for n in (3,12,13)] + [(192,n,512) for n in COLUMNS]
PRIMARY_SHAPES = list(range(4,16))


def admit_geometry(result):
    if (result.get('collection_completed') is not True or result.get('full_sequence_bit_exact') is not True
            or result.get('enabled_matrix_records_each') != 43344
            or result.get('timing_comparison', {}).get('bounded_observer_interpretation_supported') is not True):
        raise ValueError('complete bounded and exact full-model evidence required')
    groups = result.get('per_worker_matrix_groups')
    if not isinstance(groups, list) or len(groups) != 2:
        raise ValueError('both complete enabled model profiles required')
    expected = {a: (2604,63,COLUMNS) for a in PRIMARY_A}
    expected[(1024,512,1,1)] = (84,3,[3,12,13])
    expected[(512,192,1,1)] = (84,63,COLUMNS)
    reference = None
    for index, group in enumerate(groups, 1):
        if (not isinstance(group,dict) or type(group.get('index')) is not int
                or group['index'] != index or not isinstance(group.get('per_worker'),dict)
                or set(group['per_worker']) != {'0','1','2','3'}):
            raise ValueError('all four workers in both enabled profiles required')
        for rows in group['per_worker'].values():
            if (not isinstance(rows, list) or any(not isinstance(r,dict) or type(r.get('roots')) is not int or r['roots'] <= 0
                    or type(r.get('route')) is not int or not 1 <= r['route'] <= 7 for r in rows)
                    or sum(r['roots'] for r in rows) != 10836):
                raise ValueError('all actual matrix roots required')
            observed = {a: {'roots':0, 'columns':set()} for a in expected}
            for row in rows:
                if row.get('route') != 4: continue
                a_values = row.get('a', [])
                if not isinstance(a_values,list) or len(a_values) != 4 or any(type(v) is not int for v in a_values):
                    raise ValueError('exact captured A dimensions required')
                a = tuple(a_values)
                b, c = row.get('b'), row.get('c')
                if (a not in observed or row.get('types') != [8,0,0] or row.get('names') != ['q8_0','f32','f32']
                        or any(type(v) is not int for v in row['types'])
                        or type(row.get('dot')) is not int or row['dot'] != 8 or not isinstance(b,list) or len(b) != 4
                        or not isinstance(c,list) or len(c) != 4 or any(type(v) is not int for v in b+c)
                        or type(b[1]) is not int or b[1] <= 0
                        or b != [a[0],b[1],1,1] or c != [a[1],b[1],1,1]):
                    raise ValueError('exact captured converted-Q8 geometry required')
                observed[a]['roots'] += row['roots']
                observed[a]['columns'].add(b[1])
            for a, (roots,count,sample) in expected.items():
                value = observed[a]; columns = sorted(value['columns'])
                if (value['roots'] != roots or len(columns) != count
                        or [columns[0],columns[len(columns)//2],columns[-1]] != sample):
                    raise ValueError('fixed captured counts and column selection required')
                value['columns'] = columns
            if reference is None: reference = observed
            if observed != reference:
                raise ValueError('complete captured geometry differs across workers or profiles')
    return {'profiled_workers_checked':8, 'primary_columns':COLUMNS.copy(),
            'primary_a':[list(a) for a in PRIMARY_A], 'primary_roots_each_per_worker':2604,
            'selection':'Minimum, median and maximum of all 63 captured distinct primary column counts.'}


def once(raw, old, new):
    if raw.count(old) != 1: raise ValueError('one exact harness/project derivation site required')
    return raw.replace(old,new,1)


def derive_harness(raw):
    old = b'''constexpr std::array<Shape, 7> shapes = {{{1,2,32},{2,3,64},{5,7,96},{7,5,160},
    {512,33,512},{2048,33,512},{512,264,512}}};'''
    new = ('constexpr std::array<Shape, 22> shapes = {{' + ','.join('{'+','.join(map(str,s))+'}' for s in SHAPES) + '}};').encode()
    return once(once(raw,old,new), b'variant == "tile2"', b'variant == "mx4"')


def derive_project(raw):
    raw = once(raw,b'project(gen2_q8_tile_control ',b'project(gen2_q8_mx4_control ')
    return once(raw,b'target_compile_definitions(q8_candidate PRIVATE llamafile_sgemm=gen2_q8_tile_candidate_sgemm)',
                b'target_compile_definitions(q8_candidate PRIVATE llamafile_sgemm=gen2_q8_tile_candidate_sgemm GEN2_Q8_MX4_EXPECTED=1)')
