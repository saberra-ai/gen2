"""Fail closed on interposed GGML calls and changed chat/native outcomes."""
import json
from pathlib import Path
import posixpath
import re

from compare_stream import compare
from qualify_linux_package import RUNTIME_PATH

CRITICAL_SYMBOLS = {'ggml_new_graph_custom', 'ggml_backend_sched_graph_compute_async'}


def ggml_bindings(trace, libraries):
    native = {RUNTIME_PATH + '/lib/' + name for name in libraries}
    bindings = set()
    for source, target, symbol in re.findall(
            r"binding file (.+?) \[[^\]]+\] to (.+?) \[[^\]]+\]:[^\n]*symbol `(ggml_[^']+)'", trace):
        source, target = posixpath.normpath(source), posixpath.normpath(target)
        if source in native:
            if target not in native:
                raise ValueError(f'GGML symbol interposed outside diarization package: {source}: {symbol}: {target}')
            bindings.add((source, target, symbol))
    if not CRITICAL_SYMBOLS <= {symbol for _, _, symbol in bindings}:
        raise ValueError('critical native GGML bindings missing from loader trace')
    return [{'source': a, 'target': b, 'symbol': c} for a, b, c in sorted(bindings)]


def audit_results(results, native, raw, fingerprints, *, comparator=compare):
    isolated = json.loads((results / 'isolated/summary.json').read_text())
    baseline = isolated['chat_outputs']['isolated']['text']
    if not isinstance(baseline, str) or not baseline:
        raise ValueError('isolated chat result is empty')
    system = isolated['settings']['system']
    if any(system[name] != value for name, value in {'gpu_layers': 0, 'threads': 2, 'threads_batch': 2, 'ctx_size': 1024}.items()):
        raise ValueError('chat CPU settings differ from the frozen protocol')
    comparisons = {}
    for order in ['isolated', 'chat-first', 'diar-first']:
        summary = json.loads((results / order / 'summary.json').read_text())
        if summary['order'] != order or summary['fingerprints'] != fingerprints:
            raise ValueError('chat/coexistence input identity differs')
        if summary['generation'] != {'max_tokens': 8, 'temperature': 0., 'seed': 0} or summary['prompt'] != 'Reply with the single word hello.':
            raise ValueError('chat generation protocol differs')
        if summary['settings'] != isolated['settings']:
            raise ValueError('chat backend settings differ from isolated baseline')
        expected = {'isolated'} if order == 'isolated' else {
            'both_loaded', 'concurrent', 'after_shutdown', 'after_reload'} | ({'before_diarizer'} if order == 'chat-first' else set())
        if set(summary['chat_outputs']) != expected:
            raise ValueError('required chat execution stages differ')
        for stage in summary['chat_outputs'].values():
            if stage['text'] != baseline or stage['offload']['layers'] != 0:
                raise ValueError('chat output or CPU placement changed during coexistence')
        if order == 'isolated':
            continue
        intervals = summary['concurrent_intervals']
        a, b = intervals['chat_micros'], intervals['diarization_micros']
        if not all(type(value) is int and value >= 0 for value in [*a, *b]) or not (
                len(a) == len(b) == 2 and a[0] < a[1] and b[0] < b[1] and max(a[0], b[0]) < min(a[1], b[1])):
            raise ValueError('chat and diarization did not execute concurrently')
        if summary['reservation_after_shutdown_mb'] != 0:
            raise ValueError('coexistence leaked a diarization reservation')
        if a != summary['chat_outputs']['concurrent']['api_call_micros']:
            raise ValueError('concurrent chat interval differs from actual API observation')
        for stage in ['first', 'concurrent', 'reloaded']:
            path = results / order / (stage + '.json')
            public = json.loads(path.read_text())
            if stage == 'concurrent' and b != public['api_call_micros']:
                raise ValueError('concurrent diarization interval differs from actual API observation')
            if public['input_pcm_f32le_sha256'] != native['input_pcm_f32le_sha256'] or public['input_repeats'] != 1:
                raise ValueError('mixed-model actual diarization PCM identity differs')
            comparisons[order + '/' + stage] = comparator(public, native, raw, path.with_suffix('.f32').read_bytes())
    return {'isolated_chat_text': baseline, 'comparisons': comparisons}
