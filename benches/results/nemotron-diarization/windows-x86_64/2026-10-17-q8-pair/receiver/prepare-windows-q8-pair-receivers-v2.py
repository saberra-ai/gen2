"""Repair uninvoked receivers: early failure and fixed compiler role identity."""
import hashlib
import json
from pathlib import Path

cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
previous = cache / 'windows-q8-pair-receivers-pre-execution-v1.json'
metadata = json.loads(previous.read_text())


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def once(raw, old, new):
    assert raw.count(old) == 1, old
    return raw.replace(old, new)


assert all(identity((cache / name).read_bytes()) == value for name, value in metadata['outputs'].items())
text = (cache / 'collect-windows-q8-pair-v1.py').read_text()
old = "control_matches = ((evidence/'sgemm-control.cpp').read_bytes() == control_source((tools/'fixtures/q8-tile-sgemm.cpp').read_bytes())\n                   and desc(evidence/'sgemm-control.cpp') == p['control_source'])"
text = once(text, old, "control_matches = matches_control_source(evidence/'sgemm-control.cpp', (tools/'fixtures/q8-tile-sgemm.cpp').read_bytes(), p['control_source'])")
functions = '''def matches_control_source(path, original, expected):
    return path.is_file() and path.read_bytes() == control_source(original) and desc(path) == expected

def admit_kernel_unit(unit, row, variant):
    expected = {'control': 'gen2_q8_tile_control_sgemm', 'pair': 'gen2_q8_tile_candidate_sgemm'}[variant]
    guard = ('GEN2_Q8_MX4_EXPECTED' if variant == 'control' else 'GEN2_Q8_PAIR_EXPECTED') + '=1'
    definitions = re.findall(r'(?:^|\\s)[-/]D([^\\s]+)', row['command'])
    assert unit['entry'] == expected and unit['command'] == row['command']
    assert [value for value in definitions if value.startswith('llamafile_sgemm=')] == ['llamafile_sgemm=' + expected]
    assert [value for value in definitions if value.startswith('GEN2_Q8_')] == [guard]

'''
text = once(text, 'def admit_kernel_command(', functions + 'def admit_kernel_command(')
text = once(text, "if path==PureWindowsPath(unit['source']):", "if path==PureWindowsPath(unit['source']):\n                admit_kernel_unit(unit,row,variant)")
text = once(text, "assert seen==set(compiled)", "assert seen==set(compiled) and {control,candidate} <= seen")
text = once(text, "collection-replay-v1.json", "collection-replay-v2.json")
compile(text, 'collect-windows-q8-pair-v2.py', 'exec')
outputs = {name: value for name, value in metadata['outputs'].items()
           if name not in ('collect-windows-q8-pair-v1.py', 'publish-windows-q8-pair-v1.py')}
with (cache / 'collect-windows-q8-pair-v2.py').open('x') as stream:
    stream.write(text)
outputs['collect-windows-q8-pair-v2.py'] = identity(text.encode())
text = (cache / 'publish-windows-q8-pair-v1.py').read_text()
text = text.replace('collect-windows-q8-pair-v1.py', 'collect-windows-q8-pair-v2.py').replace('collection-replay-v1.json', 'collection-replay-v2.json')
text = once(text, "'windows-q8-pair-receivers-pre-execution-v1.json',", "'windows-q8-pair-receivers-pre-execution-v1.json','windows-q8-pair-receivers-pre-execution-v2.json',\n        'prepare-windows-q8-pair-receivers-v1.py','prepare-windows-q8-pair-receivers-v2.py',")
compile(text, 'publish-windows-q8-pair-v2.py', 'exec')
with (cache / 'publish-windows-q8-pair-v2.py').open('x') as stream:
    stream.write(text)
outputs['publish-windows-q8-pair-v2.py'] = identity(text.encode())
report = {**metadata, 'outputs': outputs, 'superseded_uninvoked_v1': identity(previous.read_bytes()),
    'preparation': {name: identity((cache / name).read_bytes()) for name in (
        'prepare-windows-q8-pair-receivers-v1.py', 'prepare-windows-q8-pair-receivers-v2.py')}}
with (cache / 'windows-q8-pair-receivers-pre-execution-v2.json').open('x') as stream:
    stream.write(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
