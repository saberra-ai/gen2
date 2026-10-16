import hashlib,json
from pathlib import Path
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';cache=Path('/Users/victor/.cache/gen2-nemotron3-n0')
def rec(p):
 d=p.read_bytes();return {'bytes':len(d),'sha256':hashlib.sha256(d).hexdigest()}
p=json.loads((tools/'valid-mel-metal-long.protocol.json').read_text())
p['scope']='Guarded continuation after preserved storage-failed Metal attempt: exactly one independent native and one public one-hour stream; prior completed30m pair reused as saved evidence only.'
p['output_path']=str(cache/'valid-mel-metal-long-v2')
failure=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-metal-long-failure'
p['prior_failed_attempt_path']=str(failure/'qualification.json')
q=json.loads((failure/'qualification.json').read_text());assert q['completed_30m_pair_passed']and not q['one_hour_qualified']
p['cases']=p['cases'][1:];assert len(p['cases'])==1 and p['cases'][0]['seconds']==3600
p['storage_guard']={'start_minimum':5*1024**3,'running_minimum':1024**3,'reserve_bytes':256*1024**2}
for name in ['qualification_storage.py','test_qualification_storage.py','qualify_valid_mel_metal_long_v2.py']:p['tools'][name]=rec(tools/name)
for f in failure.rglob('*'):
 if f.is_file():p['inputs'][str(f)]=rec(f)
for name in ['run-valid-mel-metal-long-v2.py','freeze-valid-mel-metal-long-v2.py']:p['inputs'][str(cache/name)]=rec(cache/name)
p['external_launcher']=rec(cache/'run-valid-mel-metal-long-v2.py')
p['preflights']=[{'name':'storage-preflight','pattern':'test_qualification_storage.py','tests':4}]+p['preflights']
p['execution_order']=['35 model-free preflights and all frozen source/input/failed-attempt/prior30m evidence checks. No repeat30m neural run.','Require5GiB free before each native admission, allocate and verify256MiB reported diagnostic reserve.','One one-hour independent C API GPU0 stream, then one one-hour public Rust --metal stream, with one-second disk-space observations and1GiB running floor.','Compare all retained rows/turns, compaction, processRSS/RTF/revisions and93privateimages perprocess; recheck frozen bytes. Atomic phase metadata, reserve released on success/failure.']
p['limits'][0]='Exactly two new model instantiations/stream runs: one1h native and one1h public. Priorcompleted30m pair is reused from unchanged saved evidence; failedV1one-hour native command remains failed/unqualified. No additional helper/reload/HF/originalNeMo/DER/qualitytargets or retries. Internalforwardcounts notinstrumented;1800secondtimeout percommand killsentireprocessgroup.'
p['limits'][1]=p['limits'][1].replace('31preflights','35preflights')
p['limits']+=['StorageinterruptionV1isretainedunchanged, includingunrecordedchildstatus andmissing1h terminalarrays. Thiscontinuationchangesqualificationstoragehandlingonly, notSDK/model/pipeline/profile/thresholds. Reportfour successful30m/1h streams acrossattempts onlyifthisnewpairpasses; do notclaimV1itselfcompleted.','Beforeeachnewnativecommand require>=5GiBfree. Pollobservedfreebytes atmostoncepersecond whilechildruns; <1GiB aborts thequalificationprocessgroup andfailscase. A256MiB newlyallocatedreserve (reportedst_blocks) isreleased onerror toretain diagnostics. Thisreducesstoragerisk, notaguaranteeagainstexternaldiskconsumers.','Commandsandouterstatusarewrittenviafsyncedtemporaryfiles+atomicreplace; completepreviousstatussurvivesreplacementfailure. Originalstdout/stderr/tensorsremainraw. No restartfrompartialmodelstate oroutputmetadatarepair. Cleanup/disksnapshot observationsarenotcleanperformanceclaims.']
path=tools/'valid-mel-metal-long-v2.protocol.json';assert not path.exists();path.write_text(json.dumps(p,indent=2)+'\n');print(rec(path))
