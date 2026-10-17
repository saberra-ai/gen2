"""Checked original same-process comparison derivations; clocks unchanged."""
from q8_worker_timing_source import once


def derive_metrics(raw):
    return once(raw, b"KERNEL_ORDER = ['control', 'row1', 'row1', 'control']",
        b"KERNEL_ORDER = ['control', 'mx2', 'mx2', 'control']")


def derive_benchmark(raw):
    return once(raw, b'candidate ? "row1" : "control"', b'candidate ? "mx2" : "control"')


def derive_cmake(raw):
    raw = once(raw, b'project(gen2_q8_row_control', b'project(gen2_q8_mx2_control')
    return once(raw, b'GEN2_Q8_ROW_EXPECTED=1', b'GEN2_Q8_MX2_EXPECTED=1')


def derive_probe(raw):
    replacements = [
        (b'from q8_row_source import ', b'from q8_mx2_source import '),
        (b'from windows_q8_row_metrics import ', b'from windows_q8_mx2_metrics import '),
        (b"'windows-q8-row.protocol.json'", b"'windows-q8-mx2.protocol.json'"),
        (b"p['schema_version'] != 6", b"p['schema_version'] != 8"),
        (b"['control','row1','row1','control']", b"['control','mx2','mx2','control']"),
        (b"'sgemm-row1.cpp'", b"'sgemm-mx2.cpp'"),
        (b"tools/'q8_row_benchmark.cmake'", b"tools/'q8_mx2_benchmark.cmake'"),
        (b"[archive,control,candidate,project/'CMakeLists.txt',tools/'q8_row_benchmark.cpp']",
         b"[archive,control,candidate,project/'CMakeLists.txt',tools/'q8_mx2_benchmark.cpp']"),
        (b"str(tools/'q8_row_benchmark.cpp')", b"str(tools/'q8_mx2_benchmark.cpp')"),
        (b"[('control',control),('row1',candidate)]", b"[('control',control),('mx2',candidate)]"),
        (b"'GEN2_Q8_ROW_EXPECTED'", b"'GEN2_Q8_MX2_EXPECTED'"),
        (b"{'control','row1'}", b"{'control','mx2'}"),
    ]
    for old, new in replacements: raw = once(raw, old, new)
    return raw
