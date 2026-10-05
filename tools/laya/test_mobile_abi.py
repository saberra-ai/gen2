"""Exercise the mobile host boundary on a desktop dynamic-library build."""
import argparse,ctypes,json
from pathlib import Path

def main(a):
    lib=ctypes.CDLL(str(a.library.resolve()))
    lib.gen2_laya_open.argtypes=[ctypes.c_char_p]
    lib.gen2_laya_decide.argtypes=[ctypes.c_uint64,ctypes.c_char_p]
    lib.gen2_laya_invoke.argtypes=[ctypes.c_uint64,ctypes.c_char_p]
    for name in ["suspend","resume","close"]:getattr(lib,"gen2_laya_"+name).argtypes=[ctypes.c_uint64]
    for name in ["open","decide","invoke","suspend","resume","close"]:getattr(lib,"gen2_laya_"+name).restype=ctypes.c_void_p
    lib.gen2_laya_free.argtypes=[ctypes.c_void_p]
    def call(name,*args):
        pointer=getattr(lib,"gen2_laya_"+name)(*args)
        try:return json.loads(ctypes.string_at(pointer).decode("utf-8"))
        finally:lib.gen2_laya_free(pointer)
    config=json.dumps(dict(bundle=str(a.bundle.resolve()),native_library=str(a.runtime.resolve()),resident_budget_mb=16,
                           max_input_bytes=4096,intra_threads=1,queue_capacity=2)).encode()
    opened=call("open",config);assert opened["ok"],opened
    handle=opened["value"]["handle"]
    request=json.dumps(dict(state=dict(kind="text",value="Unicode 👋 café 返金"),questions=[["refund",dict(type="yes_no",instructions="a?",false_label="false",true_label="true",false_description=None,true_description=None)]]),ensure_ascii=False).encode("utf-8")
    first=call("decide",handle,request);assert first["ok"],first
    def invoke(operation,**fields):
        return call("invoke",handle,json.dumps(dict(operation=operation,**fields),ensure_ascii=False).encode("utf-8"))
    described=invoke("describe");assert described["ok"],described
    caps=described["value"]["capabilities"]
    assert (caps["intra_threads"],caps["max_input_bytes"],caps["queue_capacity"])==(1,4096,2)
    assert caps["multi_state_batch"] and caps["long_state_windows"]
    def semantic(result):
        result=dict(result)
        assert result.pop("queue_micros")>=0
        assert result.pop("execution_micros")>=0
        return result
    decoded=json.loads(request)
    configured=invoke("decide",request=decoded,options=dict(language="en",timeout_ms=60000))
    assert configured["ok"],configured
    assert semantic(configured["value"])==semantic(first["value"])
    empty=dict(state=dict(kind="text",value=""),questions=[])
    batch=invoke("batch",requests=[decoded,empty,decoded])
    assert batch["ok"],batch
    assert len(batch["value"])==3 and batch["value"][1]["answers"]==[]
    assert semantic(batch["value"][0])==semantic(first["value"])
    assert semantic(batch["value"][2])==semantic(first["value"])
    long_request=dict(decoded,state=dict(kind="text",value="a "*40))
    scan=invoke("long",request=long_request,scan=dict(window_tokens=10,stride_tokens=5,max_windows=16))
    assert scan["ok"],scan
    assert scan["value"]["state_tokens"]==40 and len(scan["value"]["windows"])==7
    assert scan["value"]["answers"][0]["window_index"]==0
    for operation,payload in [("decide",dict(request=decoded)),("batch",dict(requests=[decoded])),("long",dict(request=long_request))]:
        expired=invoke(operation,**payload,options=dict(timeout_ms=0))
        assert not expired["ok"] and "deadline" in expired["error"].lower(),expired
    assert not invoke("decide",request=decoded,options=dict(typo=1))["ok"]
    assert not invoke("long",request=long_request,scan=dict(window_tokens=10,stride_tokens=5,max_windows=1))["ok"]
    assert not invoke("unsupported")["ok"]
    assert not call("invoke",handle,None)["ok"]
    oversized=dict(decoded,state=dict(kind="text",value="x"*5000))
    assert not invoke("decide",request=oversized)["ok"]
    overflowing=dict(decoded,state=dict(kind="text",value="a "*600))
    assert not invoke("decide",request=overflowing)["ok"]
    truncated=invoke("decide",request=overflowing,options=dict(encode=dict(
        max_len=512,head_max_len=192,overflow="allow_with_diagnostics")))
    assert truncated["ok"] and truncated["value"]["inputs"][0]["state_tokens_dropped"]>0,truncated
    assert call("suspend",handle)["ok"]
    assert not call("decide",handle,request)["ok"]
    assert not invoke("batch",requests=[decoded])["ok"]
    assert call("resume",handle)["ok"]
    second=call("decide",handle,request);assert second["ok"],second
    # Job timings vary; semantic output, provenance and execution policy must not.
    for value in (first["value"], second["value"]):
        assert value.pop("queue_micros") >= 0
        assert value.pop("execution_micros") > 0
    assert second==first
    assert call("close",handle)["ok"]
    assert not call("decide",handle,request)["ok"]
    assert not call("close",handle)["ok"]
    assert not call("open",b"not JSON")["ok"]
    for field in ("queue_capacity","intra_threads","max_input_bytes"):
        invalid=dict(json.loads(config),**{field:0})
        assert not call("open",json.dumps(invalid).encode())["ok"],field
    print("C ABI: budgets/capabilities, Unicode inference, options, batches, seven-window scan, deadlines, invalid input, suspend/resume/close passed")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["library","runtime","bundle"]:p.add_argument("--"+name,type=Path,required=True)
    main(p.parse_args())
