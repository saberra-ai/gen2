"""Prepare frozen complete-app inputs/protocol; no helper, SDK or model runs."""
from pathlib import Path
import ast
import hashlib
import json
import shutil
import struct
import sys
import wave

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
tools = repo / 'tools/nemotron-diarization'
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
read = lambda path: json.loads(Path(path).read_text())


def descriptor(path):
    raw = Path(path).read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


constructor = read(tools / 'macos-bounded-pnola-construction.protocol.json')
for name, expected in constructor['tools'].items():
    assert descriptor(tools / name) == expected
sys.path.insert(0, str(tools))
from sdk_archive import inventory, restore_archive
from package_macos import verify

build = cache / 'bounded-pnola-construction-local-v1'
q = read(build / 'qualification.json')
receipt = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-bounded-pnola-app-construction'
assert q['source_commit'] == 'edf7b898b00df89d868761bfc80487d7c3bdf097'
assert q['app_construction_passed'] is True and q['native_version_queries_or_neural_calls'] == 0
assert q['models_loaded_for_inference'] is False and q['storage']['released'] and 'error' not in q
assert all(c['returncode'] == 0 for c in q['commands'])
assert (build / 'qualification.json').read_bytes() == (receipt / 'qualification.json').read_bytes()
assert read(receipt / 'collection-qualification.json')['app_construction_passed'] is True
for name, expected in read(receipt / 'collection.json')['files'].items():
    assert descriptor(receipt / name) == expected
app = build / 'cpu/Pnola.app'
assert inventory(app) == q['apps']['cpu']['files']
work = cache / 'bounded-pnola-complete-app-inputs-v1'
assert not work.exists()
work.mkdir()
long = cache / 'bounded-long-local-v1'
long_q = read(long / 'qualification.json')
assert long_q['bounded_30m_1h_passed'] is True and long_q['storage']['released']
helpers = read(long / 'bounded-long-helpers-files.json')
assert {k: v for k, v in helpers.items() if k != 'files'} == long_q['helper_archive']
restore_archive(long / helpers['archive'], helpers, work / 'bin')
shutil.copytree(app / 'Contents/Resources/nemo', work / 'nemo', symlinks=True)
manifest = verify(work / 'nemo')
assert manifest == verify(app / 'Contents/Resources/nemo')
assert manifest['runtime_profile'] == 'gen2-valid-mel-pool-cpu-v2'
assets = work / 'assets'
assets.mkdir()
shutil.copy2(app / 'Contents/Resources/nemotron/Nemotron-3-Diarization.q8_0.gguf', assets / 'model.gguf')
source = cache / 'valid-mel-pnola-inputs-v2/work/assets/ami.wav'
assert descriptor(source) == {'bytes': 1920044, 'sha256': 'f00f92e53115a4a6724aec0ddaf9c675df6d5fabd3a489b43fffe56ba52a3ae9'}
assert source.read_bytes() == (app / 'Contents/Resources/sample.wav').read_bytes()
shutil.copy2(source, assets / 'ami.wav')
with wave.open(str(source), 'rb') as reader:
    assert reader.getnchannels() == 1 and reader.getsampwidth() == 2 and reader.getframerate() == 16000 and reader.getnframes() == 960000
    raw = reader.readframes(reader.getnframes())
live = (raw * 2)[:1280640 * 2]
with wave.open(str(assets / 'live-80s.wav'), 'wb') as writer:
    writer.setnchannels(1)
    writer.setsampwidth(2)
    writer.setframerate(16000)
    writer.writeframes(live)
pcm = {}
for name, data in [('ami.wav', raw), ('live-80s.wav', live)]:
    values = struct.unpack('<' + 'h' * (len(data) // 2), data)
    digest = hashlib.sha256()
    for value in values:
        digest.update(struct.pack('<f', value / 32768.))
    pcm[name] = digest.hexdigest()
old = read(tools / 'macos-pooled-pnola-app-v2.protocol.json')
assert pcm == old['pcm_sha256']
names = set(constructor['tools']) | set(old['tools']) | {
    'qualify_bounded_pnola_app.py', 'bounded_pnola_metrics.py', 'test_bounded_pnola_metrics.py'}
checked = set()
while names - checked:
    for name in sorted(names - checked):
        checked.add(name)
        if not name.endswith('.py'):
            continue
        for node in ast.walk(ast.parse((tools / name).read_text())):
            modules = [node.module] if isinstance(node, ast.ImportFrom) and node.module else [v.name for v in node.names] if isinstance(node, ast.Import) else []
            for module in modules:
                local = module.split('.')[0] + '.py'
                if (tools / local).is_file():
                    names.add(local)
denied = set(old['denied_paths'])
denied.update(str(path) for path in cache.iterdir() if path.is_dir() and path not in [work, build])
denied.update(str(path) for path in build.iterdir() if path.name != 'cpu')
records = [build / 'qualification.json', build / 'protocol.json', build / 'cpu-static-verification.json',
    receipt / 'collection-qualification.json', receipt / 'collection.json', long / 'qualification.json',
    long / 'bounded-long-helpers-files.json', Path(__file__)]
p = {'schema_version': 1,
    'scope': 'Five fresh neural processes: independent qualified bounded CPU public C ABI60s/80.04s, then automatic complete-app CPU recording/live and explicit Metal rejection while preserving Bark. No old app output is an oracle. No GUI/microphone/fresh-machine/accuracy/public-signing/release claim.',
    'output_parent': str(cache), 'output_name': 'bounded-pnola-complete-app-runtime-v1',
    'work_path': str(work), 'app_path': str(app), 'expected_cpu': 'Apple M4 Pro',
    'construction': {'path': str(build / 'qualification.json'), 'file': descriptor(build / 'qualification.json'),
                     'source_commit': q['source_commit']},
    'tools': {n: descriptor(tools / n) for n in sorted(names)},
    'repository_files': constructor['repository_files'],
    'work_inventory': inventory(work), 'app_inventory': inventory(app),
    'preparation_records': {str(path): descriptor(path) for path in records},
    'cases': old['cases'], 'bark_text': old['bark_text'], 'wav_sha256': descriptor(source)['sha256'],
    'pcm_sha256': pcm, 'denied_paths': sorted(denied), 'storage': old['storage'],
    'command_timeout_seconds': old['command_timeout_seconds'],
    'limits': [
        'Require committed controller/protocol, current tool/root bytes before imports, exact historical root Git blobs, completed model-free construction and independently collected app inventory. Original complete-app numerical/session/export/revision/cleanup gates and five fresh processes remain unchanged.',
        'After reviewed commit run actual --admit-only first: static package/model/signature checks and frozen inventory/preparation identities only. No model/version/helper/app execution, output tensor or child home/tmp/result tree is created; retain/review its receipt before full qualification. Static metadata tools may inspect code and temporary unsigned copies.',
        'Reuse only the exact three qualified bounded long helpers/archives; no compiler or private C++ ABI access. Construct live input by PCM16 repetition to1280640samples/80.04s with frozen normalized-F32 digests. Native references require all6000/8004rows, finite bounded8channel arrays and every final turn; app timelines compare within1e-6 and internal public API parity is also required.',
        'CPU automatic packaged resolution must supply exact Bark text/117recordingwords and152livewords/11sections/803exportbytes, bounded ordered revisions and final timeline. Metal selection on the CPU package must preserve Bark with explicit speaker unavailability. No artifact-path overrides, physical microphone or inference-time network/download is allowed.',
        'Keep the existing runtime/model-denial controls, six weight formats, source/reference/prior-result denial and immutable app/work/preparation inputs. Audit exact92 private native images for version/native helpers and complete actual process-local add-image history for allthree hardened app cases, including ORT/source aliases.',
        'Original5GiB start/1GiB running floors,256MiBphysicalreserve,600s child deadlines/groupcleanup/durablefailure and no automatic neural retry remain unchanged. Observe the construction controller terminal before preparation/qualification; never overlap inference with compilation or other model workloads.',
        'Same named-M4 hardened ad-hoc complete-app CLI only. No GUI/microphone/fresh-machine/public trust/signing/notarization, acoustic timing, held-out quality/human word gold, other-platform or release pass. Original20%DER and deferred human review remain open; no production/default promotion.']}
target = tools / 'macos-bounded-pnola-app.protocol.json'
assert not target.exists()
target.write_text(json.dumps(p, indent=2) + '\n')
assert inventory(app) == q['apps']['cpu']['files']
print(descriptor(target), 'tools', len(p['tools']), 'root', len(p['repository_files']),
      'work', len(p['work_inventory']), 'app', len(p['app_inventory']), 'pcm', pcm)
