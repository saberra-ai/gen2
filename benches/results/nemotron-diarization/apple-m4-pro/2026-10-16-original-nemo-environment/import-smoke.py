"""Model-free import only: never restore a checkpoint or instantiate a module."""
import hashlib,importlib.metadata,json,pathlib,sys,time
start=time.monotonic();out=pathlib.Path(__file__).resolve().parent
import numpy,torch
from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel
from nemo.collections.asr.modules.audio_preprocessing import AudioToMelSpectrogramPreprocessor
import nemo
mapping=json.loads((out/'wheel-source-mapping.json').read_text());site=pathlib.Path(nemo.__file__).parent.parent
for name,value in mapping.items():
 p=site/name;b=p.read_bytes();assert len(b)==value['bytes'] and hashlib.sha256(b).hexdigest()==value['sha256'],name
result={'passed':True,'stage':'unmodified original NeMo model-class and frontend-class imports only','model_constructed':False,'model_restored':False,'model_executed':False,'frontend_executed':False,'elapsed_seconds':time.monotonic()-start,'python':sys.version,'executable':sys.executable,'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},'source_files_verified':len(mapping),'model_class_module':str(sys.modules[SortformerEncLabelModel.__module__].__file__),'frontend_class_module':str(sys.modules[AudioToMelSpectrogramPreprocessor.__module__].__file__),'torch_cuda_available':torch.cuda.is_available()}
(out/'import-smoke-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='packages'}))
