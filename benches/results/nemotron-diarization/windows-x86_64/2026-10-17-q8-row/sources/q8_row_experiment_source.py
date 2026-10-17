"""Checked derivations of the original paired-kernel experiment; no clock edits."""
from q8_worker_timing_source import once


def derive_metrics(raw):
    return once(raw, b"KERNEL_ORDER = ['control', 'pair', 'pair', 'control']",
        b"KERNEL_ORDER = ['control', 'row1', 'row1', 'control']")


def derive_benchmark(raw):
    return once(raw, b'candidate ? "pair" : "control"', b'candidate ? "row1" : "control"')


def derive_cmake(raw):
    raw = once(raw, b'project(gen2_q8_pair_control', b'project(gen2_q8_row_control')
    return once(raw, b'GEN2_Q8_PAIR_EXPECTED=1', b'GEN2_Q8_ROW_EXPECTED=1')


def derive_probe(raw):
    replacements = [
        (b'from q8_pair_source import ', b'from q8_row_source import '),
        (b'from windows_q8_pair_metrics import ', b'from windows_q8_row_metrics import '),
        (b"'windows-q8-pair.protocol.json'", b"'windows-q8-row.protocol.json'"),
        (b"p['schema_version'] != 5", b"p['schema_version'] != 6"),
        (b"['control','pair','pair','control']", b"['control','row1','row1','control']"),
        (b"'sgemm-pair.cpp'", b"'sgemm-row1.cpp'"),
        (b"tools/'q8_pair_benchmark.cmake'", b"tools/'q8_row_benchmark.cmake'"),
        (b"[archive,control,candidate,project/'CMakeLists.txt',tools/'q8_pair_benchmark.cpp']",
         b"[archive,control,candidate,project/'CMakeLists.txt',tools/'q8_row_benchmark.cpp']"),
        (b"str(tools/'q8_pair_benchmark.cpp')", b"str(tools/'q8_row_benchmark.cpp')"),
        (b"[('control',control),('pair',candidate)]", b"[('control',control),('row1',candidate)]"),
        (b"'GEN2_Q8_PAIR_EXPECTED'", b"'GEN2_Q8_ROW_EXPECTED'"),
        (b"{'control','pair'}", b"{'control','row1'}"),
    ]
    for old, new in replacements:
        raw = once(raw, old, new)
    return raw
