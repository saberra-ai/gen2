"""Create an isolated source-only candidate; no builds or inference."""
import hashlib,json,os,pathlib,subprocess
cache=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0');original=cache/'NeMo-Speech.cpp';candidate=cache/'NeMo-Speech-valid-mel-v1';output=cache/'valid-mel-source-v1';repo=pathlib.Path('/Users/victor/workspace/gen2-nemotron3-n0');base='083861c241061562b8ec29eb06d7f6cdc3af8775';llama='bd4f514db14d87fded667787a7a963bfbaa98e89';relative='src/asr/diar/diar_pipeline.cpp'
assert not candidate.exists()and not output.exists();output.mkdir()
env=os.environ.copy();env['GIT_LFS_SKIP_SMUDGE']='1';commands=[]
def run(argv,cwd=None):
 result=subprocess.run(argv,cwd=cwd,env=env,capture_output=True);index=len(commands);(output/(str(index)+'.stdout.log')).write_bytes(result.stdout);(output/(str(index)+'.stderr.log')).write_bytes(result.stderr);commands.append({'argv':argv,'cwd':str(cwd)if cwd else None,'returncode':result.returncode});(output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');result.check_returncode();return result.stdout
assert run(['git','rev-parse','HEAD'],original).decode().strip()==base
assert run(['git','rev-parse','HEAD'],original/'llama.cpp').decode().strip()==llama
run(['git','clone','--no-hardlinks','--no-checkout',str(original),str(candidate)])
run(['git','checkout','--detach',base],candidate)
run(['git','clone','--no-hardlinks','--no-checkout',str(original/'llama.cpp'),str(candidate/'llama.cpp')])
run(['git','checkout','--detach',llama],candidate/'llama.cpp')
assert not run(['git','status','--porcelain','--ignore-submodules=all','--untracked-files=no'],candidate)
path=candidate/relative;raw=path.read_text();assert hashlib.sha256(path.read_bytes()).hexdigest()=='4143ca8e6d6f14ee454263702510cb163fe8e170aa928f236c0b4b540a7a7da9'
old='''    if (finished_)
        return;
    finished_ = true;
'''
new='''    if (finished_)
        return;
    const size_t original_audio_end = audio_base_ + audio_buf_.size();
    finished_ = true;
'''
assert raw.count(old)==1;raw=raw.replace(old,new)
old='''    run_ready_chunks(/*end_of_stream=*/true);
'''
new='''    if (m_.cfg().is_v3()) {
        // NeMo masks the extra centered-STFT row before feature stacking.
        const int64_t valid_mel_end = static_cast<int64_t>(
            original_audio_end / static_cast<size_t>(m_.fe().hop_length()));
        if (valid_mel_end < mel_base_ || valid_mel_end < mel_consumed_ ||
            valid_mel_end > mel_produced()) {
            throw std::runtime_error("DiarStream: invalid v3 final mel bounds");
        }
        mel_buf_.resize(static_cast<size_t>(valid_mel_end - mel_base_) * n_mels_);
    }
    run_ready_chunks(/*end_of_stream=*/true);
'''
assert raw.count(old)==1;path.write_text(raw.replace(old,new))
patch=run(['git','diff','--binary','--no-ext-diff','--ignore-submodules=all'],candidate)
patch_path=repo/'tools/nemotron-diarization/patches/v3-finish-valid-mel.patch';assert not patch_path.exists();patch_path.parent.mkdir(exist_ok=True);patch_path.write_bytes(patch)
assert run(['git','status','--porcelain','--ignore-submodules=all','--untracked-files=no'],candidate).decode()==' M '+relative+'\n'
record=lambda p:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
report={'source_base_revision':base,'llama_revision':llama,'source_original':str(original),'source_candidate':str(candidate),'patched_file':relative,'patched_file_identity':record(path),'patch':record(patch_path),'network_acquisition_performed':False,'build_performed':False,'neural_inference_performed':False,'commands':commands};(output/'prepared.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
