"""Checked derivation of the one-row experiment with exhaustive F16C outputs."""
from q8_worker_timing_source import once


def derive_metrics(raw):
    return once(raw, b"KERNEL_ORDER = ['control', 'row1', 'row1', 'control']",
        b"KERNEL_ORDER = ['control', 'scale', 'scale', 'control']")


def derive_benchmark(raw):
    raw = once(raw, b'candidate ? "row1" : "control"', b'candidate ? "scale" : "control"')
    raw = once(raw, b'// Original CPU lookup symbol and initialization formula; no kernel source edit.',
        b'extern "C" void gen2_q8_scale_broadcast_probe(uint16_t, float *);\nstatic std::array<float, 65536 * 4> scale_half_outputs;\n\n// Original CPU lookup symbol and initialization formula; retained unchanged.')
    raw = once(raw, b'        uint32_t bits = 0;\n', b'''        float * converted = scale_half_outputs.data() + i * 4;
        gen2_q8_scale_broadcast_probe(static_cast<uint16_t>(i), converted);
        for (int lane = 0; lane < 4; ++lane) {
            if (std::isnan(value) ? !std::isnan(converted[lane]) :
                    std::memcmp(converted + lane, &value, sizeof(float)) != 0)
                throw std::runtime_error("F16C A-scale broadcast differs from original half table");
        }
        uint32_t bits = 0;
''')
    return once(raw,
        b'        save_output(std::string(argv[3]) + ".half.f32", ggml_table_f32_f16, sizeof(ggml_table_f32_f16));',
        b'''        save_output(std::string(argv[3]) + ".half.f32", ggml_table_f32_f16, sizeof(ggml_table_f32_f16));
        save_output(std::string(argv[3]) + ".scale-half.f32", scale_half_outputs.data(), static_cast<DWORD>(sizeof(scale_half_outputs)));''')


def derive_cmake(raw):
    raw = once(raw, b'project(gen2_q8_row_control', b'project(gen2_q8_scale_control')
    return once(raw, b'GEN2_Q8_ROW_EXPECTED=1', b'GEN2_Q8_SCALE_EXPECTED=1')


def derive_probe(raw):
    replacements = [
        (b'from q8_row_source import ', b'from q8_scale_source import '),
        (b'from windows_q8_row_metrics import ', b'from windows_q8_scale_metrics import '),
        (b"'windows-q8-row.protocol.json'", b"'windows-q8-scale.protocol.json'"),
        (b"p['schema_version'] != 6", b"p['schema_version'] != 7"),
        (b"or p['finite_half_entries'] != 63488", b"or p['scale_half_entries'] != 65536 or p['scale_half_lanes'] != 4\n                or p['scale_half_bytes'] != 1048576 or p['finite_half_entries'] != 63488"),
        (b"['control','row1','row1','control']", b"['control','scale','scale','control']"),
        (b"'sgemm-row1.cpp'", b"'sgemm-scale.cpp'"),
        (b"tools/'q8_row_benchmark.cmake'", b"tools/'q8_scale_benchmark.cmake'"),
        (b"[archive,control,candidate,project/'CMakeLists.txt',tools/'q8_row_benchmark.cpp']",
         b"[archive,control,candidate,project/'CMakeLists.txt',tools/'q8_scale_benchmark.cpp']"),
        (b"str(tools/'q8_row_benchmark.cpp')", b"str(tools/'q8_scale_benchmark.cpp')"),
        (b"[('control',control),('row1',candidate)]", b"[('control',control),('scale',candidate)]"),
        (b"'GEN2_Q8_ROW_EXPECTED'", b"'GEN2_Q8_SCALE_EXPECTED'"),
        (b"{'control','row1'}", b"{'control','scale'}"),
        (b"reference_half=None", b"reference_half=None\n        reference_scale_half=None"),
        (b"case['half_table']=admit_half_table(half_raw)", b"case['half_table']=admit_half_table(half_raw)\n                scale_half_path=Path(str(evidence/phase)+'.scale-half.f32')\n                scale_half_raw=scale_half_path.read_bytes()\n                case['scale_half_table']=admit_scale_half_table(scale_half_raw)\n                if reference_scale_half is None: reference_scale_half=scale_half_raw\n                if reference_scale_half!=scale_half_raw: raise ValueError('all twelve complete A-scale broadcasts must match')"),
        (b"[half_path,*paths,evidence/(phase+'.stdout.log'),evidence/(phase+'.log')]",
         b"[half_path,scale_half_path,*paths,evidence/(phase+'.stdout.log'),evidence/(phase+'.log')]"),
        (b"q['all_12_half_tables_bit_exact']=True", b"q['all_12_half_tables_bit_exact']=True\n        q['all_12_scale_half_broadcasts_verified']=True"),
    ]
    for old, new in replacements:
        raw = once(raw, old, new)
    return once(raw, b'from q8_mx4_geometry import MODEL_RESULT, admit_geometry',
        b'from q8_scale_half_metrics import admit_scale_half_table\nfrom q8_mx4_geometry import MODEL_RESULT, admit_geometry')
