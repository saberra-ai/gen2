from pathlib import Path
import json,subprocess,sys
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');sys.path.insert(0,str(tools))
from build_valid_mel_candidate import descriptor
from compare_macos_cpu_pools import model_free_policy
from qualify_macos_portable_runtime import policy
from macos_loaded_libraries import audit_loaded_libraries
from package_macos import verify_files,verify_package_provenance
out=cache/'pooled-cleanup-diagnostic-v1';assert not out.exists();out.mkdir();work=out/'work';(work/'bin').mkdir(parents=True);candidate=cache/'pooled-cpu-construction-local-v1';package=candidate/'runtime';sdk=candidate/'sdk';(work/'nemo').symlink_to(package,target_is_directory=True)
source=out/'diagnostic.cpp';text=(tools/'native_failed_load_probe.cpp').read_text().replace('#include <cstdint>','#include <cstdint>\n#include <thread>\n#include <chrono>')
old='''            if (counts.back() > baseline + 2)
                throw std::runtime_error("failed model loads accumulate native threads");'''
new='''            if (counts.back() > baseline + 2) {
                std::cerr << "[DEBUG-pool-cleanup] failure_index=" << index << " baseline=" << baseline << " count=" << counts.back() << " history=";
                for (auto value : counts) std::cerr << value << ",";
                std::cerr << "\\n";
                for (int tick = 0; tick < 10; ++tick) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    std::cerr << "[DEBUG-pool-cleanup] quiet_millis=" << (tick + 1) * 10 << " count=" << thread_count() << "\\n";
                }
                thread_act_array_t threads = nullptr; mach_msg_type_number_t total = 0;
                if (task_threads(mach_task_self(), &threads, &total) == KERN_SUCCESS) {
                    for (unsigned k = 0; k < total; ++k) {
                        thread_extended_info_data_t info{}; mach_msg_type_number_t size = THREAD_EXTENDED_INFO_COUNT;
                        auto status = thread_info(threads[k], THREAD_EXTENDED_INFO, reinterpret_cast<thread_info_t>(&info), &size);
                        std::cerr << "[DEBUG-pool-cleanup] thread=" << k << " status=" << status << " name=" << info.pth_name << " run_state=" << info.pth_run_state << "\\n";
                        mach_port_deallocate(mach_task_self(), threads[k]);
                    }
                    vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(threads), total * sizeof(thread_t));
                }
                throw std::runtime_error("failed model loads accumulate native threads");
            }'''
assert old in text;source.write_text(text.replace(old,new));binary=work/'bin/diagnostic';manifest=json.loads((package/'package.json').read_text());verify_files(package,manifest['files']);verify_package_provenance(package,manifest)
compile_profile=out/'compile.sb';compile_profile.write_text(model_free_policy(policy(denied_reads=[out/'absent-weights'],readonly=[repo,sdk,package])))
runtime_profile=out/'runtime.sb';runtime_profile.write_text(model_free_policy(policy(denied_reads=[repo,sdk,candidate/'source'],readonly=[work,repo,sdk,package])))
q={'scope':'Instrumented failure-only missing-model cleanup diagnosis; original guard unchanged, not qualification. No valid model/frontend/neural execution.','source':descriptor(source),'commands':[]}
commands=[('compile',['/usr/bin/sandbox-exec','-f',str(compile_profile),'/usr/bin/clang++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-arch','arm64','-march=armv8.2-a+dotprod+fp16','-mmacosx-version-min=26.0',str(source),'-I'+str(sdk/'include'),'-L'+str(package/'lib'),'-lnemo_speech_asr_c','-Wl,-rpath,@loader_path/../nemo/lib','-o',str(binary)]),('diagnostic',['/usr/bin/sandbox-exec','-f',str(runtime_profile),'/usr/bin/env','DYLD_PRINT_LIBRARIES=1',str(binary),'nemo-speech-asr 0.2.0+gen2-valid-mel-pool-v1',str(work/'missing-model')])]
for name,argv in commands:
 with (out/(name+'.stdout.log')).open('xb') as stdout,(out/(name+'.stderr.log')).open('xb') as stderr:
  done=subprocess.run(argv,cwd=out,env={'PATH':'/usr/bin:/bin:/usr/sbin:/sbin','LC_ALL':'C'},stdout=stdout,stderr=stderr,timeout=60)
 q['commands'].append({'name':name,'argv':argv,'returncode':done.returncode});(out/'qualification.json').write_text(json.dumps(q,indent=2)+'\n')
 if name=='compile':assert done.returncode==0,(out/(name+'.stderr.log')).read_text()
q['loaded_images']=audit_loaded_libraries((out/'diagnostic.stderr.log').read_text(),package,manifest,binary);q['binary']=descriptor(binary);verify_files(package,manifest['files']);(out/'qualification.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({'diagnostic_returncode':done.returncode,'private_libraries':q['loaded_images']['private_library_count']}));print('\n'.join(line for line in (out/'diagnostic.stderr.log').read_text().splitlines() if '[DEBUG-pool-cleanup]' in line));print((out/'diagnostic.stdout.log').read_text())
