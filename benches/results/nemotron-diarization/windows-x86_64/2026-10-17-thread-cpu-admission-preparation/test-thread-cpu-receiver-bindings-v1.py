"""Exercise actual prepared binding ASTs with historical commands and synthetic retained-copy descriptors."""
import ast
import copy
import hashlib
import json
from pathlib import Path, PureWindowsPath

CACHE=Path('/Users/victor/.cache/gen2-nemotron3-n0')
REPO=Path('/Users/victor/workspace/gen2-nemotron3-n0')
BASE=REPO/'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-matrix-batch-profile'
PREPARED=CACHE/'thread-cpu-receivers-prepared-v3'
source_path=PREPARED/'collect-windows-thread-cpu-profile-v1.py'
raw=source_path.read_bytes()
identity={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
assert identity==json.loads((PREPARED/'preparation.json').read_bytes())['prepared'][source_path.name]['prepared']
source=raw.decode()
sequence,=[n for n in ast.parse(source).body if isinstance(n,ast.Assert)and isinstance(n.test,ast.Compare)and isinstance(n.test.left,ast.ListComp)]
sequence_code=compile(ast.Module([sequence],type_ignores=[]),'actual-prepared-sequence','exec')
start=source.index("assert len(control['inputs']) == 7\n")
end=source.index('control_reference={};control_files={}\n',start)
graph_code=compile(source[start:end],'actual-prepared-graph-bindings','exec')
start=source.index("    model_command,=[row for row in q['commands']if row['name']==phase]\n")
end=source.index("    path = evidence / (phase + '.json')\n",start)
model_code=compile('\n'.join(line[4:]for line in source[start:end].splitlines()),'actual-prepared-model-bindings','exec')
q=json.loads((BASE/'qualification.json').read_bytes())
audit=json.loads((BASE/'evidence/platform-llama-materialized-audit.json').read_bytes())
materialized=PureWindowsPath(audit['compiled_tree_path'])
control_build=next(row for row in q['commands']if row['name']=='matrix-control-build')
win_evidence=PureWindowsPath(control_build['cwd'])
for name,value in list(q['model_free_controls']['inputs'].items()):
    path=PureWindowsPath(name)
    if path.suffix=='.lib':q['model_free_controls']['inputs'][str(win_evidence/'matrix-control-inputs'/path.name)]=value.copy()
assert len(q['model_free_controls']['inputs'])==7
tools=Path('/fixture/tools');evidence=Path('/fixture/evidence')
descriptors={}
for name,value in q['model_free_controls']['inputs'].items():
    path=PureWindowsPath(name)
    if path.name=='ggml_matrix_profile_control.cpp':descriptors[tools/path.name]=value.copy()
    elif path.parent.name=='matrix-control-inputs':descriptors[evidence/'matrix-control-inputs'/path.name]=value.copy()

def validate(value):
    scope={'q':value,'control':value['model_free_controls'],'build':value['build'],
        'materialized':materialized,'tools':tools,'evidence':evidence,'PureWindowsPath':PureWindowsPath,
        'descriptor':lambda path:descriptors[path]}
    exec(sequence_code,scope);exec(graph_code,scope)
    for index in range(4):
        scope.update(index=index,phase='case-'+str(index));exec(model_code,scope)

validate(q)
mutations=[]
def changed_command(name,index,replacement):
    value=copy.deepcopy(q);row=next(r for r in value['commands']if r['name']==name)
    row['command'][index]=replacement;return value
mutations.append(('wrong-compiler',changed_command('matrix-control-build',0,'C:\\wrong-cl.exe')))
mutations.append(('wrong-include',changed_command('matrix-control-build',7,'/IC:\\other-include')))
mutations.append(('wrong-library',changed_command('matrix-control-build',9,'C:\\other.lib')))
mutations.append(('wrong-output',changed_command('matrix-control-build',10,'/OUT:C:\\other.exe')))
mutations.append(('wrong-probe',changed_command('case-0',0,'C:\\other.exe')))
mutations.append(('wrong-model',changed_command('case-1',1,'C:\\other.gguf')))
mutations.append(('wrong-pcm',changed_command('case-2',2,'C:\\other.wav')))
mutations.append(('wrong-cut',changed_command('case-3',3,'2')))
mutations.append(('wrong-target',changed_command('case-0',5,str(win_evidence/'case-1.json'))))
value=copy.deepcopy(q);value['commands'][-1]['cwd']='C:\\other';mutations.append(('wrong-cwd',value))
value=copy.deepcopy(q);value['commands'][-1],value['commands'][-2]=value['commands'][-2],value['commands'][-1];mutations.append(('model-order',value))
value=copy.deepcopy(q);value['commands'].append({**value['commands'][-1],'name':'extra-model'});mutations.append(('extra-attempt',value))
value=copy.deepcopy(q);value['commands'].insert(-1,value['commands'].pop(-5));mutations.append(('graph-after-model',value))
value=copy.deepcopy(q)
name=next(n for n in value['model_free_controls']['inputs']if 'matrix-control-inputs' in n)
value['model_free_controls']['inputs'][name]['sha256']='0'*64
mutations.append(('retained-library-mismatch',value))
value=copy.deepcopy(q);inputs=value['model_free_controls']['inputs']
name=next(n for n in inputs if 'matrix-control-inputs' in n);inputs[str(PureWindowsPath(name).parent.parent/'foreign'/PureWindowsPath(name).name)]=inputs.pop(name)
mutations.append(('retained-library-path',value))
for label,value in mutations:
    try:validate(value)
    except (AssertionError,StopIteration,ValueError):continue
    raise AssertionError('mutation accepted: '+label)
assert source_path.read_bytes()==raw
report={'prepared_collector':identity,'complete_binding_fixture_accepted':True,
    'rejected_mutations':[name for name,_ in mutations],
    'fixture':'Actual retained historical command/source metadata with synthetic added retained-copy descriptors; no Windows binary payload or new artifact read, no native calls. This checks binding predicates only.',
    'source_commit':'15c905d8815190a54f25452e7e51378c24f85914','new_native_calls':0}
with(CACHE/'thread-cpu-receiver-binding-controls-v1.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'binding_controls':len(mutations),'new_native_calls':0}))
