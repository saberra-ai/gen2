"""Exact derivations of the frozen worker experiment; new kernel inputs only."""
from q8_worker_timing_source import once


def derive_metrics(raw):
    return once(raw, b"KERNEL_ORDER = ['control', 'mx4', 'mx4', 'control']",
                b"KERNEL_ORDER = ['control', 'pair', 'pair', 'control']")


def derive_benchmark(raw):
    return once(raw, b'candidate ? "mx4" : "control"', b'candidate ? "pair" : "control"')


def derive_cmake(raw):
    replacements = [
        (b'project(gen2_q8_worker_timing_control', b'project(gen2_q8_pair_control'),
        (b'foreach(input GGML_SOURCE CANDIDATE_SOURCE BENCHMARK_SOURCE)',
         b'foreach(input GGML_SOURCE CONTROL_SOURCE CANDIDATE_SOURCE BENCHMARK_SOURCE)'),
        (b'add_library(q8_control OBJECT "${GGML_SOURCE}/ggml/src/ggml-cpu/llamafile/sgemm.cpp")',
         b'add_library(q8_control OBJECT "${CONTROL_SOURCE}")'),
        (b'target_compile_definitions(q8_control PRIVATE llamafile_sgemm=gen2_q8_tile_control_sgemm)',
         b'target_compile_definitions(q8_control PRIVATE llamafile_sgemm=gen2_q8_tile_control_sgemm GEN2_Q8_MX4_EXPECTED=1)'),
        (b'gen2_q8_tile_candidate_sgemm GEN2_Q8_MX4_EXPECTED=1)',
         b'gen2_q8_tile_candidate_sgemm GEN2_Q8_PAIR_EXPECTED=1)'),
    ]
    for old, new in replacements:
        raw = once(raw, old, new)
    return raw


def derive_probe(raw):
    replacements = [
        (b'from q8_mx4_source import PATH, ORIGINAL, transform',
         b'from q8_pair_source import PATH, ORIGINAL, transform, control_source'),
        (b'from windows_q8_worker_timing_metrics import ', b'from windows_q8_pair_metrics import '),
        (b"'windows-q8-worker-timing.protocol.json'", b"'windows-q8-pair.protocol.json'"),
        (b"p['schema_version'] != 4", b"p['schema_version'] != 5"),
        (b"['control','mx4','mx4','control']", b"['control','pair','pair','control']"),
        (b"candidate=evidence/'sgemm-mx4.cpp'", b"candidate=evidence/'sgemm-pair.cpp'"),
        (b"candidate.write_bytes(transform((canonical/PATH).read_bytes()))",
         b"candidate.write_bytes(transform((canonical/PATH).read_bytes()))\n        control=evidence/'sgemm-control.cpp'\n        control.write_bytes(control_source((canonical/PATH).read_bytes()))\n        if descriptor(control)!=p['control_source']:\n            raise ValueError('exact production Mx4 control source required')"),
        (b"tools/'q8_worker_timing_benchmark.cmake'", b"tools/'q8_pair_benchmark.cmake'"),
        (b"[archive,candidate,project/'CMakeLists.txt',tools/'q8_worker_timing_benchmark.cpp']",
         b"[archive,control,candidate,project/'CMakeLists.txt',tools/'q8_pair_benchmark.cpp']"),
        (b"'-DCANDIDATE_SOURCE:FILEPATH='+str(candidate)",
         b"'-DCONTROL_SOURCE:FILEPATH='+str(control),'-DCANDIDATE_SOURCE:FILEPATH='+str(candidate)"),
        (b"str(tools/'q8_worker_timing_benchmark.cpp')", b"str(tools/'q8_pair_benchmark.cpp')"),
        (b"[('control',canonical/PATH),('mx4',candidate)]", b"[('control',control),('pair',candidate)]"),
        (b"if variant=='mx4' and row['command'].count('GEN2_Q8_MX4_EXPECTED=1')!=1:",
         b"if row['command'].count(('GEN2_Q8_MX4_EXPECTED' if variant=='control' else 'GEN2_Q8_PAIR_EXPECTED')+'=1')!=1:"),
        (b"if set(kernels)!={'control','mx4'}:", b"if set(kernels)!={'control','pair'}:"),
    ]
    for old, new in replacements:
        raw = once(raw, old, new)
    return raw
