"""Pure receiver branch controls; no actual collector, native tool or model runs."""
import ast,copy,hashlib,json,tempfile
from pathlib import Path,PureWindowsPath
c=Path('/Users/victor/.cache/gen2-nemotron3-n0')
helpers=json.loads((c/'q8-candidate-receivers-pre-execution-v2.json').read_text())
def desc(path):
 raw=path.read_bytes();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
for name,value in helpers.items():assert desc(c/name)==value
raw=(c/'replay-windows-q8-candidate-supervision-v1.py').read_text()
body=raw[raw.index("commands=q.get('commands',[])"):raw.index('assert descriptor(Path(__file__)) == identity')]
record={'name':'unit-control','process_supervision':'Windows kill-on-close Job v2 with scoped snapshot and owned-background cleanup',
 'process_tree_reaped':True,'returncode':0,'job_observation':{'worker_assigned':True,'normal_worker_exit_observed':True,
 'process_tree_reaped':True,'job_handle_closed':True,'cleanup_errors':[],'observed_worker_exit_code':0,'reaped_worker_exit_code':0,
 'worker_pid':123,'remaining_before_cleanup':{'processes':[],'assigned_count':0}}}
rows=[]
for label,complete,mode,should_pass in [('empty-failure',False,'empty',True),('partial-failure',False,'partial',True),
 ('nonzero-failure',False,'nonzero',True),('complete-zero',True,'clean',True),('complete-partial-reject',True,'partial',False),
 ('complete-nonzero-reject',True,'nonzero',False)]:
 with tempfile.TemporaryDirectory()as folder:
  root=Path(folder);groups={}if mode=='empty'else{'candidate':[copy.deepcopy(record)]}
  members={}
  for kind,commands in groups.items():
   row=commands[0]
   if mode=='partial':row={'name':'unit-control','process_supervision':record['process_supervision']};commands[0]=row
   if mode=='nonzero':
    row['returncode']=1;row['job_observation']['observed_worker_exit_code']=1;row['job_observation']['reaped_worker_exit_code']=1
   folder=root/'evidence';folder.mkdir();(folder/'commands.json').write_text(json.dumps(commands))
   for suffix in('.stdout.log','.log'):members['evidence/unit-control'+suffix]={}
  context={'q':{'development_native_gates_passed':complete,'commands':groups.get('candidate',[])},'delivered':root,'read':lambda p:json.loads(p.read_text()),
           'members':members,'summaries':[],'PureWindowsPath':PureWindowsPath}
  error=None
  try:exec(compile(body,'supervision-branch-control','exec'),context)
  except AssertionError as exc:error=type(exc).__name__
  assert (error is None)==should_pass,label
  if error is None and mode=='partial':assert all(r['owned_tree_reaped_and_job_closed']is False and r['observed_exit_preserved']is False for r in context['summaries'])
  if error is None and mode=='clean':assert all(r['owned_tree_reaped_and_job_closed']is True and r['observed_exit_preserved']is True for r in context['summaries'])
  rows.append({'name':label,'passed':True,'expected_acceptance':should_pass,'observed_error':error})
segments=[]
for name in ['fetch-windows-q8-worker-timing-artifact-v2.py','fetch-windows-q8-candidate-artifact-v2.py']:
 source=(c/name).read_text();tree=ast.parse(source)
 node,=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='bounded_download']
 segments.append(ast.get_source_segment(source,node))
assert segments[0]==segments[1]

source=(c/'replay-windows-q8-candidate-result-v2.py').read_text();tree=ast.parse(source)
node,=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='admit_execution_commands']
context={'PureWindowsPath':PureWindowsPath};exec(compile(ast.get_source_segment(source,node),'execution-command-control','exec'),context)
admit=context['admit_execution_commands'];ci=PureWindowsPath('D:/controlled/trial');repo=PureWindowsPath('D:/controlled/repo');ev=ci/'evidence';assets=ci/'assets'
public=PureWindowsPath('D:/controlled/target/release/examples/nemotron_valid_mel_stream.exe')
order=['native','public','public','native'];lifecycle=['first_gate','second_gate'];commands=[]
for index,kind in enumerate(order):
 name=f'case-{index}-{kind}';target=ev/(name+'.json')
 if kind=='native':argv=[ci/'sdk/bin/gen2_native_stream_probe_valid_mel.exe',assets/'Nemotron-3-Diarization.q8_0.gguf',assets/'ami_en2002d_2132.wav','1','-1',target]
 else:argv=[public,assets,ci/'sdk/bin/gen2_nemo_windows_nemo_speech_asr_c.dll',assets/'ami_en2002d_2132.wav','1',PureWindowsPath(str(target)+'.f32')]
 commands.append({'name':name,'command':list(map(str,argv)),'cwd':str(repo)})
cargo=['cargo','test','--locked','--release','--no-default-features','--features','backend-diar-nemo','--test','diarization_native','--']
commands.append({'name':'native-lifecycle-list','command':cargo+['--ignored','--list'],'cwd':str(repo)})
for index,test in enumerate(lifecycle):commands.append({'name':'native-lifecycle-'+str(index),'command':cargo+['--ignored','--exact',test,'--test-threads=1'],'cwd':str(repo)})
for label in ['clean','history-order','lifecycle-before-history','native-executable','public-executable','other-sdk','other-audio','other-model','repeat-count','lifecycle-features','lifecycle-test','extra-argument','other-cwd','duplicate-history']:
 actual=copy.deepcopy(commands)
 if label=='history-order':actual[0],actual[1]=actual[1],actual[0]
 elif label=='lifecycle-before-history':actual.insert(0,actual.pop(4))
 elif label=='native-executable':actual[0]['command'][0]='D:\\wrong-probe.exe'
 elif label=='public-executable':actual[1]['command'][0]='D:\\wrong-public.exe'
 elif label=='other-sdk':actual[1]['command'][2]='D:\\other-sdk\\library.dll'
 elif label=='other-audio':actual[1]['command'][3]='D:\\other.wav'
 elif label=='other-model':actual[0]['command'][1]='D:\\other.gguf'
 elif label=='repeat-count':actual[0]['command'][3]='2'
 elif label=='lifecycle-features':actual[5]['command'][6]='different-feature'
 elif label=='lifecycle-test':actual[5]['command'][-2]='different-test'
 elif label=='extra-argument':actual[5]['command'].append('--skip')
 elif label=='other-cwd':actual[0]['cwd']='D:\\other-repo'
 elif label=='duplicate-history':actual.insert(1,copy.deepcopy(actual[0]))
 error=None
 try:value=admit(actual,ci,repo,{str(public):{}},order,lifecycle)
 except AssertionError as exc:error=type(exc).__name__
 assert(error is None)==(label=='clean'),label
 rows.append({'name':'execution-'+label,'passed':True,'expected_acceptance':label=='clean','observed_error':error})
result={'controls':rows,'all_passed':True,'bounded_download_body_equals_prior_six_control_receiver':True,'helpers':helpers,
 'new_compiler_sdk_model_calls':0,'scope':__doc__}
target=c/'q8-candidate-receiver-controls-local-v2.json';assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'controls':len(rows),'all_passed':True,'bounded_download_body_unchanged':True}))
