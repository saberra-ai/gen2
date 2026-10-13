"""Freeze independent fluent/negation fixtures, noise controls and WAV variants.

LibriSpeech validation excerpt is frozen before any native comparison. It is
supplementary domain coverage, not a claim of universal dictation accuracy.
"""
import argparse
import hashlib
import io
import json
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
import soundfile as sf
import soxr
from onnx_asr.loader import Manager

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bundle",type=Path,required=True)
parser.add_argument("--fluent",type=Path,required=True)
parser.add_argument("--disfluent",type=Path,required=True)
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()
assert hashlib.sha256(args.fluent.read_bytes()).hexdigest()=="4e69a06fa5edc90921e5e7e39a7084881f8b3ed9c805c574f4f39c6fde27c603"
opts=ort.SessionOptions(); opts.intra_op_num_threads=4; opts.inter_op_num_threads=1
asr=Manager(providers=["CPUExecutionProvider"],sess_options=opts,preprocessor_config={"use_numpy_preprocessors":True}).create_asr("nemo-conformer-tdt",args.bundle,offline=True)
args.output.mkdir(parents=True,exist_ok=True)
records=[]
def record(wave,rate,name,human,category,channels=1,subtype="FLOAT"):
    path=args.output/(name+".wav")
    sf.write(path,wave,rate,subtype=subtype)
    decoded,_=sf.read(path,dtype="float32",always_2d=True)
    normalized=decoded.mean(axis=1)
    if rate!=16000: normalized=soxr.resample(normalized,rate,16000,quality="HQ")
    text=asr.recognize(normalized,sample_rate=16000).strip()
    records.append({"wav":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"text":text,"human":human,"category":category,"rate":rate,"channels":channels,"encoding":subtype})
    print(name,text,flush=True)
fluent=pq.read_table(args.fluent).to_pylist()
selected=[0,1,3,11,12,20,21,23,24,26,33,35,36,38,44,45,50,56]
for i in selected:
    row=fluent[i]; wave,rate=sf.read(io.BytesIO(row["audio"]["bytes"]),dtype="float32")
    record(wave,rate,"fluent-"+row["id"],row["text"],"quiet-fluent")
    rng=np.random.default_rng(100+i)
    noise=rng.normal(0,float(np.sqrt(np.mean(wave**2)))/10,wave.shape).astype(np.float32)
    record(wave+noise,rate,"noisy-"+row["id"],row["text"],"20dB-white-noise")
disfluent=pq.read_table(args.disfluent).to_pylist()
for i in range(3):
    source,rate=sf.read(io.BytesIO(disfluent[i]["audio"]["bytes"]),dtype="float32")
    for target in [8000,16000,44100,48000]:
        mono=soxr.resample(source,rate,target,quality="HQ") if target!=rate else source
        for channels in [1,2,8]:
            wave=mono if channels==1 else np.repeat(mono[:,None],channels,axis=1)
            for subtype in ["PCM_16","FLOAT"]:
                record(wave,target,f"wav-{i}-{target}-{channels}-{subtype}",disfluent[i]["transcript_c"],"wav-conformance",channels,subtype)
for length in [320,16000,960000]:
    record(np.zeros(length,np.float32),16000,f"silence-{length}","","silence")
record(np.random.default_rng(7).normal(0,.01,16000).astype(np.float32),16000,"noise-16000","","noise")
(args.output/"audio-reference.json").write_text(json.dumps({"dataset_revision":"5be91486e11a2d616f4ec5db8d3fd248585ac07a","records":records},indent=2)+"\n")
