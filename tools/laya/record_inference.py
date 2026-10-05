"""Record independent eager/ONNX outputs for the preprocessing corpus.
Run eager and onnx as separate processes to avoid retaining both weight sets.
"""
import argparse,json,sys,subprocess,hashlib
from pathlib import Path
import numpy as np

def main(a):
    source_sha=subprocess.check_output(["git","-C",str(a.source),"rev-parse","HEAD"],text=True).strip()
    if source_sha!="8a6e1328cce2460a0e5aa348ad465bb1b5821cd2":raise ValueError("wrong source revision")
    sys.path.insert(0,str(a.source.resolve()))
    from laya.common import temp_bucket,clamp_temperature,unpermute_probs
    fixture=json.loads(a.fixture.read_text(encoding="utf-8"))
    config=json.loads((a.bundle/"rl_agent_config.json").read_text())
    manifest_bytes=(a.bundle/"manifest.json").read_bytes()
    manifest=json.loads(manifest_bytes)
    if fixture["source"].get("checkpoint","english")!=manifest["checkpoint"]:raise ValueError("fixture checkpoint mismatch")
    temperatures=[clamp_temperature(t) for t in config.get("temperature",[1]*3)]
    buckets={k:clamp_temperature(v) for k,v in config.get("temperature_by_options",{}).items()}
    if a.mode=="eager":
        import torch
        from laya.agent import Agent
        torch.set_num_threads(2)
        agent=Agent(str(a.bundle/"source"),device="cpu",compile=False,backend="eager")
        agent.model.float().eval()
        def run(inputs):
            with torch.inference_mode():
                return [v.detach().numpy() for v in agent.model(**{k:torch.from_numpy(v) for k,v in inputs.items()})]
    else:
        import onnxruntime as ort
        opts=ort.SessionOptions();opts.intra_op_num_threads=2
        session=ort.InferenceSession(str(a.bundle/"model.onnx"),sess_options=opts,providers=["CPUExecutionProvider"])
        def run(inputs):return session.run(["logits","act_logits"],inputs)
    cases=[]
    for case in fixture["cases"]:
        rows=case["expected"];n=len(rows)
        if not n:cases.append(dict(name=case["name"],request=case["request"],options=case["options"],inputs=[],probabilities=[]));continue
        length=max(len(r["input_ids"]) for r in rows);k=max(2,max(len(r["marker_pos"]) for r in rows))
        inputs=dict(input_ids=np.full((n,length),fixture["special_tokens"]["pad"],dtype=np.int64),attention_mask=np.zeros((n,length),dtype=np.int64),marker_pos=np.zeros((n,k),dtype=np.int64),marker_mask=np.zeros((n,k),dtype=bool),qtype=np.array([r["qtype"] for r in rows],dtype=np.int64))
        for i,r in enumerate(rows):
            inputs["input_ids"][i,:len(r["input_ids"])]=r["input_ids"]
            inputs["attention_mask"][i,:len(r["input_ids"])]=1
            inputs["marker_pos"][i,:len(r["marker_pos"])]=r["marker_pos"]
            inputs["marker_mask"][i,:len(r["marker_pos"])]=True
        logits,actions=run(inputs);probabilities=[];action_probabilities=[]
        for i,r in enumerate(rows):
            width=len(r["marker_pos"]);qt=r["qtype"]
            z=logits[i,:width]/buckets.get(temp_bucket(qt,width),temperatures[qt]);p=np.exp(z-z.max());p/=p.sum()
            p=unpermute_probs(p,case["request"]["questions"][i][1].get("option_order"))
            action=np.exp(actions[i]-actions[i].max());action/=action.sum()
            probabilities.append(p.tolist());action_probabilities.append(float(action[0]))
        cases.append(dict(name=case["name"],request=case["request"],options=case["options"],inputs=rows,probabilities=probabilities,act_probabilities=action_probabilities))
        print(f'{a.mode}: {case["name"]} ({n}, {length}, {k})',flush=True)
    a.output.write_text(json.dumps(dict(mode=a.mode,source_revision=source_sha,checkpoint=manifest["checkpoint"],manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),cases=cases),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ["source","bundle","fixture","output"]:p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--mode",choices=["eager","onnx"],required=True)
    main(p.parse_args())
