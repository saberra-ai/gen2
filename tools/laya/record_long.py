"""Record a real upstream long-state scan for the native integration gate."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path

def main(a):
    sha=subprocess.check_output(["git","-C",str(a.source),"rev-parse","HEAD"],text=True).strip()
    if sha!="8a6e1328cce2460a0e5aa348ad465bb1b5821cd2":raise ValueError("wrong source revision")
    sys.path.insert(0,str(a.source.resolve()))
    import onnxruntime as ort
    original=ort.InferenceSession
    options=ort.SessionOptions();options.intra_op_num_threads=2
    def session(path,**kwargs):
        selected=kwargs.get("sess_options",options);selected.intra_op_num_threads=2
        kwargs["sess_options"]=selected
        return original(path,**kwargs)
    ort.InferenceSession=session
    from laya.onnx_agent import ONNXAgent
    agent=ONNXAgent(str(a.bundle/"source"),onnx_path=str(a.bundle/"model.onnx"))
    state="Routine product question . "*12+"Our invoice was charged twice . Please refund it ! "+"The account is blocked and this is urgent . "*3
    questions={"team":{"type":"choice","instructions":"Who handles this?","criteria":{"billing":"Payments","support":"Product issues"}},"urgency":{"type":"score","instructions":"How urgent?","criteria":["Routine","Soon","Blocking"]},"refund":{"type":"noul","instructions":"Is a refund requested?"}}
    request=dict(state=dict(kind="text",value=state),questions=[
        ["team",dict(type="choice",instructions="Who handles this?",options=[dict(label=k,description=v) for k,v in questions["team"]["criteria"].items()])],
        ["urgency",dict(type="score",instructions="How urgent?",levels=questions["urgency"]["criteria"])],
        ["refund",dict(type="yes_no",instructions="Is a refund requested?",false_label="false",true_label="true",false_description=None,true_description=None)]])
    result=agent.predict_long(state,questions,window=32,stride=16,batch_size=4)
    a.output.write_text(json.dumps(dict(source_revision=sha,manifest_sha256=hashlib.sha256((a.bundle/"manifest.json").read_bytes()).hexdigest(),request=request,window_tokens=32,stride_tokens=16,result=result),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f'Upstream long scan: {result["usage"]["windows"]} windows',flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["source","bundle","output"]:p.add_argument("--"+name,type=Path,required=True)
    main(p.parse_args())
