from pathlib import Path
import json,subprocess,sys
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');sys.path.insert(0,str(tools))
from build_valid_mel_candidate import descriptor
from sdk_archive import inventory
from macos_loaded_libraries import audit_loaded_libraries
from package_macos import verify_files,verify_package_provenance
previous=cache/'pooled-cleanup-settled-control-v2';old=json.loads((previous/'qualification.json').read_text());out=cache/'pooled-cleanup-final-control-v3';assert not out.exists();out.mkdir();work=out/'work';(work/'bin').mkdir(parents=True);candidate=cache/'pooled-cpu-construction-local-v1';sdk=candidate/'sdk';package=candidate/'runtime';(work/'nemo').symlink_to(package,target_is_directory=True)
for name in ['compile.sb','runtime.sb']:(out/name).write_text((previous/name).read_text().replace(str(previous),str(out)))
manifest=json.loads((package/'package.json').read_text());verify_files(package,manifest['files']);verify_package_provenance(package,manifest);assert inventory(sdk)==old['protected_inputs']['sdk'] and inventory(package)==old['protected_inputs']['package']
q={**{k:v for k,v in old.items() if k not in ['commands','result','loaded_images','binary']},'scope':'Final canonical public failed-load control plus failure-log preservation fault injection against unchanged constructed SDK/package; no valid weights/frontend/neural execution.','source':descriptor(tools/'native_failed_load_probe.cpp'),'commands':[],'prior_settled_control':descriptor(previous/'qualification.json')}
base=old['commands'][0]['argv'];compile=[s.replace(str(previous),str(out)) for s in base];runtime=[s.replace(str(previous),str(out)) for s in old['commands'][1]['argv']]
def run(name,argv,expected):
 with (out/(name+'.stdout.log')).open('xb') as stdout,(out/(name+'.stderr.log')).open('xb') as stderr:
  p=subprocess.run(argv,cwd=out,env={'PATH':'/usr/bin:/bin:/usr/sbin:/sbin','LC_ALL':'C'},stdout=stdout,stderr=stderr,timeout=60)
 q['commands'].append({'name':name,'argv':argv,'returncode':p.returncode,'expected_returncode':expected});(out/'qualification.json').write_text(json.dumps(q,indent=2)+'\n');assert p.returncode==expected,(out/(name+'.stderr.log')).read_text()[-2000:]
run('compile',compile,0);run('failed-loads',runtime,0)
r=json.loads((out/'failed-loads.stdout.log').read_text());assert len(r['threads_after_each_failure'])==101 and max(r['threads_after_each_failure'])<=r['threads_after_each_failure'][0]+2 and len(r['requested_settling_wait_milliseconds'])==len(r['immediate_thread_counts'])==101;q['result']=r
binary=work/'bin/failed-load-probe';q['binary']=descriptor(binary);q['loaded_images']=audit_loaded_libraries((out/'failed-loads.stderr.log').read_text(),package,manifest,binary)
fault=out/'fault.cpp';text=(tools/'native_failed_load_probe.cpp').read_text();needle='            waits.push_back(requested_wait);';assert text.count(needle)==1;fault.write_text(text.replace(needle,needle+'\n            if (index == 4) throw std::runtime_error("forced observation diagnostic");'));fault_binary=work/'bin/fault-probe';fault_compile=[s.replace(str(tools/'native_failed_load_probe.cpp'),str(fault)).replace(str(binary),str(fault_binary)) for s in compile];run('fault-build',fault_compile,0);fault_runtime=[s.replace(str(binary),str(fault_binary)) for s in runtime];run('fault-observations',fault_runtime,1)
log=(out/'fault-observations.stderr.log').read_text();line=next(s for s in log.splitlines() if s.startswith('failed_load_thread_observations:'))
for label,next_label in [('immediate=',' settled='),('settled=',' requested_wait_ms='),('requested_wait_ms=',None)]:
 part=line.split(label,1)[1]
 if next_label:part=part.split(next_label,1)[0]
 assert len([s for s in part.split(',') if s])==5
assert 'forced observation diagnostic' in log;q['fault_injection']={'source':descriptor(fault),'binary':descriptor(fault_binary),'change':'Throw after observation index4; no SDK/runtime changes.','all_three_partial_histories_retained':True,'observations_retained':5,'loaded_images':audit_loaded_libraries(log,package,manifest,fault_binary)}
assert inventory(sdk)==old['protected_inputs']['sdk'] and inventory(package)==old['protected_inputs']['package'];q['sdk_and_package_unchanged']=True;(out/'qualification.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({'canonical_failed_attempts':101,'settled_threads':sorted(set(r['threads_after_each_failure'])),'immediate_threads':sorted(set(r['immediate_thread_counts'])),'wait_ms':sorted(set(r['requested_settling_wait_milliseconds'])),'fault_partial_histories':5,'private_images_each':92}))
