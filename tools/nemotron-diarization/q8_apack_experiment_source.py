"""Checked original same-process comparison derivations; clocks unchanged."""
from q8_worker_timing_source import once


def derive_metrics(raw):
    return once(raw, b"KERNEL_ORDER = ['control', 'row1', 'row1', 'control']",
        b"KERNEL_ORDER = ['control', 'apack', 'apack', 'control']")


def derive_benchmark(raw, controls):
    raw = once(raw, b'candidate ? "row1" : "control"', b'candidate ? "apack" : "control"')
    raw = once(raw, b'constexpr float guard = 0x1p80f;\n', b'constexpr float guard = 0x1p80f;\n\n' + controls)
    return once(raw, b'        Pool pool(threads);', b'        packing_controls(argv[3]);\n        Pool pool(threads);')


def derive_cmake(raw):
    raw = once(raw, b'project(gen2_q8_row_control', b'project(gen2_q8_apack_control')
    return once(raw, b'GEN2_Q8_ROW_EXPECTED=1', b'GEN2_Q8_APACK_EXPECTED=1')


def derive_probe(raw):
    replacements = [
        (b'from q8_row_source import ', b'from q8_apack_source import '),
        (b'from windows_q8_row_metrics import ', b'from windows_q8_apack_metrics import '),
        (b"'windows-q8-row.protocol.json'", b"'windows-q8-apack.protocol.json'"),
        (b"p['schema_version'] != 6", b"p['schema_version'] != 9"),
        (b"or p['finite_half_entries'] != 63488",
         b"or p['packing_blocks'] != 65536 or p['packing_payload_bytes'] != 2097152\n                or p['packing_scales_bytes'] != 262144 or p['packing_worker_cap_bytes'] != 67108864\n                or p['finite_half_entries'] != 63488"),
        (b"['control','row1','row1','control']", b"['control','apack','apack','control']"),
        (b"'sgemm-row1.cpp'", b"'sgemm-apack.cpp'"),
        (b"tools/'q8_row_benchmark.cmake'", b"tools/'q8_apack_benchmark.cmake'"),
        (b"[archive,control,candidate,project/'CMakeLists.txt',tools/'q8_row_benchmark.cpp']",
         b"[archive,control,candidate,project/'CMakeLists.txt',tools/'q8_apack_benchmark.cpp']"),
        (b"str(tools/'q8_row_benchmark.cpp')", b"str(tools/'q8_apack_benchmark.cpp')"),
        (b"[('control',control),('row1',candidate)]", b"[('control',control),('apack',candidate)]"),
        (b"'GEN2_Q8_ROW_EXPECTED'", b"'GEN2_Q8_APACK_EXPECTED'"),
        (b"{'control','row1'}", b"{'control','apack'}"),
    ]
    for old, new in replacements: raw = once(raw, old, new)
    raw = once(raw, b'from q8_mx4_geometry import MODEL_RESULT, admit_geometry',
        b'from q8_apack_metrics import admit_packing\nfrom q8_mx4_geometry import MODEL_RESULT, admit_geometry')
    raw = once(raw, b"case['half_table']=admit_half_table(half_raw)", b"""case['half_table']=admit_half_table(half_raw)
                payload_path=Path(str(evidence/phase)+'.apack-payload.i8')
                scales_path=Path(str(evidence/phase)+'.apack-scales.f32')
                payload_raw, scales_raw=payload_path.read_bytes(), scales_path.read_bytes()
                case['packing_control']=admit_packing(payload_raw,scales_raw)
                if scales_raw!=half_raw: raise ValueError('packed scales must preserve all original lookup bits')""")
    raw = once(raw, b"[half_path,*paths,evidence/(phase+'.stdout.log'),evidence/(phase+'.log')]",
        b"[half_path,payload_path,scales_path,*paths,evidence/(phase+'.stdout.log'),evidence/(phase+'.log')]")
    return once(raw, b"q['all_12_half_tables_bit_exact']=True",
        b"q['all_12_half_tables_bit_exact']=True\n        q['all_12_packing_controls_verified']=True")
