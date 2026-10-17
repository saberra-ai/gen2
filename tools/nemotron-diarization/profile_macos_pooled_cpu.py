"""Sample one unchanged owned-pool C ABI call before changing CPU thread policy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time


def descriptor(path):
    raw = Path(path).read_bytes()
    return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def compare_sampled_history(value,raw,expected,golden):
    from qualify_pooled_native import compare_history
    version = 'nemo-speech-asr 0.2.0+gen2-valid-mel-pool-v1'
    if value['runtime_version'] != version or expected['runtime_version'] != version:
        raise ValueError('exact pooled identities required')
    legacy = {**expected,'runtime_version':'nemo-speech-asr 0.2.0+gen2-valid-mel-v1'}
    return compare_history(value,raw,legacy,golden)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['protocol','producer-zip','producer-metadata','native-reference-zip','native-reference-metadata','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    output = args.output.resolve()
    deliveries = [args.producer_zip,args.producer_metadata,args.native_reference_zip,args.native_reference_metadata]
    if (platform.system() != 'Darwin' or platform.machine() != 'arm64' or output.exists()
            or args.output.is_symlink() or output.is_relative_to(repo)
            or any(path.resolve().is_relative_to(output) for path in deliveries)):
        parser.error('actual Apple Silicon and fresh disjoint output required')
    raw_protocol = args.protocol.read_bytes()
    p = json.loads(raw_protocol)

    def frozen():
        if args.protocol.read_bytes() != raw_protocol:
            raise ValueError('pooled sampling protocol changed')
        for names,base in [(p['tools'],tools),(p['repository_files'],repo)]:
            for name,expected in names.items():
                if descriptor(base / name) != expected:
                    raise ValueError('pooled sampling source changed: '+name)

    frozen()  # Admit the complete local import closure before importing it.
    import struct
    import wave
    from artifacts import acquire
    from frozen_git_sources import verify_git_sources
    from macos_child_environment import sandboxed_command
    from macos_loaded_libraries import audit_loaded_libraries
    from macos_portable_delivery import admit_delivery
    from package_macos import verify
    from profile_macos_portable_cpu import CONTROL_SOURCE, sample_target
    from qualification_storage import StorageGuard, atomic_json
    from sdk_archive import inventory, restore_archive, write_archive

    source = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    verify_git_sources(repo,source,{**{'tools/nemotron-diarization/'+name:value for name,value in p['tools'].items()},
                                   **p['repository_files']})
    if raw_protocol != subprocess.check_output(['git','show',source+':tools/nemotron-diarization/macos-pooled-cpu-profile.protocol.json'],cwd=repo):
        raise ValueError('pooled profile protocol must be committed before execution')
    output.mkdir(parents=True)
    (output / 'protocol.json').write_bytes(raw_protocol)
    r = {'schema_version':1,'source_commit':source,'profile_completed':False,'model_execution_admitted':False,
         'neural_commands_started':[],'commands':[],'sampling':{},'loaded_images':{},'release_qualified':False,
         'scope':p['scope'],'hypotheses':p['hypotheses'],'limits':p['limits']}
    persist = lambda:atomic_json(output / 'qualification.json',r)
    guard = None
    protected = {}
    env = {'PATH':'/usr/bin:/bin:/usr/sbin:/sbin','HOME':str(output / 'home'),
           'TMPDIR':str(output / 'tmp')+'/','LC_ALL':'C'}

    def unchanged():
        frozen()
        if 'control_executable' in r and descriptor(output / 'work/control') != r['control_executable']:
            raise ValueError('compiled sampling control changed')
        for path,expected in protected.items():
            if inventory(Path(path)) != expected:
                raise ValueError('pooled profile immutable input changed: '+path)

    def run(name,argv,sandbox,*,executable=None,sample_seconds=0,inference=False):
        unchanged()
        guard.before_model()
        actual,environment = sandboxed_command(argv,sandbox,env)
        row = {'name':name,'argv':actual,'environment':environment,'cwd':str(output),'inference_admitted':inference}
        r['commands'].append(row)
        if inference:
            r['model_execution_admitted'] = True
            r['neural_commands_started'].append(name)
        persist()
        log = output / (name+'.stderr.log')
        stop,observer,sampled = threading.Event(),None,None
        started = time.monotonic()
        try:
            with (output / (name+'.stdout.log')).open('xb') as stdout,log.open('xb') as stderr:
                if sample_seconds:
                    sampled = {}
                    observer = threading.Thread(target=sample_target,args=(log,executable,output / (name+'.sample.txt'),sampled,stop),
                        kwargs={'duration':sample_seconds})
                    observer.start()
                process = subprocess.Popen(actual,cwd=output,env=environment,stdout=stdout,stderr=stderr,start_new_session=True)
                row['returncode'] = guard.wait(process,row,p['command_timeout_seconds'])
        finally:
            stop.set()
            if observer:
                observer.join(timeout=55)
                if observer.is_alive():
                    raise RuntimeError('pooled sampler observer did not terminate')
                r['sampling'][name] = sampled
            row['elapsed_seconds'] = time.monotonic()-started
            persist()
        if row['returncode'] != 0 or (sample_seconds and 'error' in sampled):
            raise ValueError('pooled sampling command failed: '+name)

    try:
        guard = StorageGuard(output,**p['storage'])
        for name in ['home','tmp','work','results']:
            (output / name).mkdir()
        r['deliveries'] = {}
        for name in ['producer','native_reference']:
            r['deliveries'][name] = admit_delivery(getattr(args,name+'_zip'),getattr(args,name+'_metadata'),
                p[name],output / name)
        producer,reference = output / 'producer',output / 'native_reference'
        pq,nq = [json.loads((path / 'qualification.json').read_text()) for path in [producer,reference]]
        if (pq['construction_and_negative_c_api_controls_passed'] is not True or pq['valid_models_loaded'] != 0
                or pq['frontend_or_neural_calls'] != 0 or pq['source_commit'] != p['producer']['source_commit']
                or nq['native_parity_passed'] is not True or nq['source_commit'] != p['native_reference']['source_commit']
                or nq['neural_commands_started'] != ['native-12s','native-60s']
                or any(row['returncode'] != 0 for row in nq['commands'])):
            raise ValueError('complete qualified pooled producer/native deliveries required')
        origin = nq['deliveries']['construction']
        if (origin['id'] != p['producer']['id'] or origin['workflow_run']['head_sha'] != p['producer']['source_commit']
                or origin['digest'] != 'sha256:'+p['producer']['zip']['sha256']):
            raise ValueError('native reference used another construction')
        package,helpers,assets = [output / 'work' / name for name in ['nemo','bin','assets']]
        package_files = json.loads((producer / 'pooled-cpu-runtime-files.json').read_text())
        if {k:v for k,v in package_files.items() if k != 'files'} != pq['package_archive']:
            raise ValueError('producer archive declaration differs')
        restore_archive(producer / package_files['archive'],package_files,package)
        manifest = verify(package)
        if (manifest['runtime_profile'] != 'gen2-valid-mel-pool-v1' or manifest['runtime_reported_version'] != p['runtime_version']
                or manifest['backend'] != 'cpu'):
            raise ValueError('exact existing pooled CPU package required')
        atomic_json(output / 'package.json',manifest)
        helper_files = json.loads((reference / 'pooled-native-helpers-files.json').read_text())
        restore_archive(reference / helper_files['archive'],helper_files,helpers)
        if inventory(helpers) != p['helpers']:
            raise ValueError('archived qualified public C ABI helpers differ')
        host = {'cpu':subprocess.check_output(['/usr/sbin/sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
                'cpu_counts':subprocess.check_output(['/usr/sbin/sysctl','hw.ncpu','hw.logicalcpu','hw.physicalcpu'],text=True),
                'platform':platform.platform(),'python':sys.version,'os':subprocess.check_output(['/usr/bin/sw_vers'],text=True),
                'load_before':os.getloadavg()}
        atomic_json(output / 'host.json',host)
        if host['cpu'] != p['expected_receiver_cpu']:
            raise ValueError('unexpected pooled profile receiver CPU')
        assets.mkdir()
        quote = lambda path:json.dumps(str(Path(path).resolve()),ensure_ascii=False)
        runtime = output / 'runtime.sb'
        runtime.write_text('(version 1)\n(allow default)\n(deny network*)\n(deny file-write*)\n'
            +'(allow file-write* '+ ' '.join('(subpath '+quote(output / name)+')' for name in ['results','home','tmp'])+' (literal "/dev/null"))\n'
            +'(deny file-read-data (subpath '+quote(output / 'results')+'))\n'
            +''.join('(deny file-read* (subpath '+quote(path)+'))\n' for path in
                [repo,producer,reference,'/opt/homebrew',output / 'control.c',*[path.resolve() for path in deliveries]]))
        premodel = output / 'premodel.sb'
        premodel.write_text(runtime.read_text()+'(deny file-read* (subpath '+quote(assets)+'))\n'
            +'(deny file-read* (regex #".*\\.(gguf|nemo|onnx|safetensors|npy|f32)$"))\n')
        build_policy = output / 'control-build.sb'
        build_policy.write_text('(version 1)(allow default)(deny network*)\n'
            +''.join('(deny file-read* (subpath '+quote(path)+'))\n' for path in [assets,repo,package,helpers,producer,reference])
            +'(deny file-read* (regex #".*\\.(gguf|nemo|onnx|safetensors|npy|f32)$"))\n')
        (output / 'control.c').write_text(CONTROL_SOURCE)
        protected.update({str(path):inventory(path) for path in [producer,reference,package,helpers]})
        atomic_json(output / 'protected-before.json',protected)
        control = output / 'work/control'
        run('compile-control',['/usr/bin/clang','-arch','arm64','-march=armv8.2-a+dotprod+fp16',
            '-mmacosx-version-min=26.0',output / 'control.c','-o',control],build_policy)
        r['control_executable'] = descriptor(control)
        run('sampling-control',['/usr/bin/env','DYLD_PRINT_LIBRARIES=1',control],premodel,executable=control,sample_seconds=1)
        r['loaded_images']['sampling-control'] = audit_loaded_libraries((output / 'sampling-control.stderr.log').read_text(),
            package,manifest,control,require_runtime=False)
        # The actual model-free sampler control must pass before asset acquisition.
        r['acquisitions'] = []
        for entry in p['assets']:
            guard.before_model()
            acquire(entry,assets)
            r['acquisitions'].append({'asset':entry['name'],**descriptor(assets / entry['name'])})
            persist()
        with wave.open(str(assets / p['source_wav']),'rb') as audio:
            if (audio.getnchannels(),audio.getsampwidth(),audio.getframerate(),audio.getnframes()) != (1,2,16000,960000):
                raise ValueError('pooled source WAV differs')
            pcm = audio.readframes(192000)
        wav = assets / 'profile-12s.wav'
        with wave.open(str(wav),'wb') as target:
            target.setparams((1,2,16000,0,'NONE','not compressed'))
            target.writeframes(pcm)
        normalized = b''.join(struct.pack('<f',v[0]/32768.) for v in struct.iter_unpack('<h',pcm))
        if hashlib.sha256(normalized).hexdigest() != p['pcm_f32le_sha256']:
            raise ValueError('pooled actual input PCM differs')
        protected[str(assets)] = inventory(assets)
        atomic_json(output / 'protected-before.json',protected)
        marker = output / 'results/prior.txt'
        marker.write_text('not a model')
        denied = [[str(repo / 'Cargo.toml'),'rb'],[str(reference / 'results/native-12s.json.f32'),'rb'],
                  [str(marker),'rb'],[str(wav),'wb']]
        control_code = 'import json,sys\nfor path,mode in json.loads(sys.argv[1]):\n try:open(path,mode).close()\n except PermissionError:continue\n raise RuntimeError("protected access allowed: "+path)\nprint("protected reads and writes denied")\n'
        run('runtime-controls',['/usr/bin/python3','-B','-c',control_code,json.dumps(denied)],runtime)
        executable,target = helpers / 'native-stream',output / 'results/native.json'
        run('native',['/usr/bin/time','-l','/usr/bin/env','DYLD_PRINT_LIBRARIES=1',executable,
            assets / p['model_name'],wav,'1','-1',target],runtime,executable=executable,sample_seconds=5,inference=True)
        r['loaded_images']['native'] = audit_loaded_libraries((output / 'native.stderr.log').read_text(),package,manifest,executable)
        value = json.loads(target.read_text())
        expected = json.loads((reference / 'results/native-12s.json').read_text())
        # The immutable pool-v1 reference is admitted explicitly; reuse the same
        # complete-history mathematical comparator without weakening identities.
        r['comparison'] = compare_sampled_history(value,Path(str(target)+'.f32').read_bytes(),expected,
            (reference / 'results/native-12s.json.f32').read_bytes())
        if value['input_pcm_f32le_sha256'] != p['pcm_f32le_sha256'] or value['audio_samples'] != 192000:
            raise ValueError('sampled result uses different PCM')
        unchanged()
        r['results'] = {path.name:descriptor(path) for path in (output / 'results').iterdir()}
        r['helper_archive'] = write_archive(helpers,output,'profile-helpers')
        r['native_rtf'] = value['elapsed_micros']/1e6/12.
        r['load_after'] = os.getloadavg()
        if r['neural_commands_started'] != ['native'] or [row['name'] for row in r['commands']] != p['command_order']:
            raise ValueError('fixed sampling command sequence differs')
        r['profile_completed'] = True
    except BaseException as error:
        r['error'] = repr(error)
        raise
    finally:
        if guard:
            guard.release()
            r['storage'] = guard.record
        persist()


if __name__ == '__main__':
    main()
