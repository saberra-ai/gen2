"""Saved-source audit only; no SDK build, version query, helper or model execution."""
import json
from pathlib import Path
import subprocess,sys
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';source=cache/'NeMo-Speech-valid-mel-tagged-v1';build=source/'build/cpu-diar';out=cache/'valid-mel-platform-source-preflight-v2'
assert not out.exists();out.mkdir();sys.path.insert(0,str(tools))
from platform_valid_mel_sources import audit_cpu_tree,verify_completed_source
from runtime_profiles import load_profile
lockpath,lock=load_profile();verify_completed_source(source,build,lock,tools/lock['runtime']['compatibility_patches'][0]['path'],False)
profile=out/'source-audit.sb';profile.write_text('(version 1)\n(allow default)\n(deny network*)\n(deny file-read* '+ ' '.join('(subpath '+json.dumps(str(cache/name))+')'for name in ['assets','native-f32-conversion-v1','tail-framing-v1'])+')\n(deny file-write* '+ ' '.join('(subpath '+json.dumps(str(path))+')'for path in [source,cache/'sdk-valid-mel-tagged-cpu-v1',Path('/opt/homebrew')])+')\n')
commands=[];base={'PATH':'/usr/bin:/bin','LC_ALL':'C','GIT_CONFIG_COUNT':'2','GIT_CONFIG_KEY_0':'core.autocrlf','GIT_CONFIG_VALUE_0':'false','GIT_CONFIG_KEY_1':'core.eol','GIT_CONFIG_VALUE_1':'lf'}
def run(name,argv,cwd,overrides=None):
 actual=['/usr/bin/sandbox-exec','-f',str(profile),*map(str,argv)];env={**base,**(overrides or {})};record={'name':name,'argv':actual,'cwd':str(cwd),'environment':env};commands.append(record)
 with(out/(name+'.stdout.log')).open('xb')as stdout,(out/(name+'.stderr.log')).open('xb')as stderr:r=subprocess.run(actual,cwd=cwd,env=env,stdout=stdout,stderr=stderr)
 record['returncode']=r.returncode;(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');r.check_returncode()
a=audit_cpu_tree(source,build,out,lock['runtime']['llama_revision'],run)
print(json.dumps({'static_source_audit_passed':True,'patches':len(a['patch_series']),'materialized_files':len(a['files']),'actual_compiler_files':len(a['actual_compiler_sources']),'all_compiler_files':len(a['all_compiler_sources']),'new_native_or_neural_calls':0}))
