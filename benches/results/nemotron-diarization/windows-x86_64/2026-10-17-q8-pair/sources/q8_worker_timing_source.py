"""Exact import/observation-only derivation of the admitted Windows producer."""


def once(raw, old, new):
    if raw.count(old) != 1:
        raise ValueError('one exact worker timing derivation site required')
    return raw.replace(old, new)


def derive_probe(raw):
    replacements = [
        (b'from windows_q8_mx4_metrics import ', b'from windows_q8_worker_timing_metrics import '),
        (b"'windows-q8-mx4.protocol.json'", b"'windows-q8-worker-timing.protocol.json'"),
        (b"p['schema_version'] != 3", b"p['schema_version'] != 4"),
        (b"'sdk_or_model_qualified':False,'new_sdk_or_model_calls':0,'order':ORDER", b"'sdk_or_model_qualified':False,'new_sdk_or_model_calls':0,'kernel_order':p['kernel_order'],'order':ORDER"),
        (b"shutil.copyfile(tools/'q8_mx4_benchmark.cmake'", b"shutil.copyfile(tools/'q8_worker_timing_benchmark.cmake'"),
        (b"tools/'q8_mx4_benchmark.cpp']}", b"tools/'q8_worker_timing_benchmark.cpp']}"),
        (b"str(tools/'q8_mx4_benchmark.cpp')", b"str(tools/'q8_worker_timing_benchmark.cpp')"),
        (b"'index':index,'variant':variant,'command_started'", b"'index':index,'telemetry_mode':variant,'command_started'"),
        (b"paths=[Path(str(evidence/phase)+'.'+str(i)+'.f32') for i in range(len(SHAPES))]",
         b"paths=[Path(str(evidence/phase)+'.'+str(i)+'.'+str(position)+'.f32') for i in range(len(SHAPES)) for position in range(4)]"),
        (b"timing_evidence(q['cases'][-4:])", b"timing_evidence(q['cases'])"),
        (b"q['all_264_outputs_bit_exact']=True", b"q['all_1056_outputs_bit_exact']=True"),
    ]
    for old, new in replacements:
        raw = once(raw, old, new)
    raw = once(raw, b"if (p['schema_version'] != 4 or p['guard_codes'] != GUARD_CODES",
        b"if (p['schema_version'] != 4 or p['kernel_order'] != ['control','mx4','mx4','control'] or p['guard_codes'] != GUARD_CODES")
    return raw
