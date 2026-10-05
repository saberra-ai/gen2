"""Exercise the full declared CPU shape envelope and record host measurements.
This desktop runner does not replace iOS/Android physical-device host tests.
"""
import argparse,hashlib,json,math,platform,time
from pathlib import Path
import numpy as np
import onnxruntime as ort

def peak_bytes():
    if platform.system()=="Windows":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[("cb",wintypes.DWORD),("faults",wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in
                ["peak_working_set","working_set","peak_paged","paged","peak_nonpaged","nonpaged","pagefile","peak_pagefile"]]
        kernel=ctypes.WinDLL("kernel32",use_last_error=True);psapi=ctypes.WinDLL("psapi",use_last_error=True)
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        c=Counters();c.cb=ctypes.sizeof(c)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(c),c.cb):raise ctypes.WinError(ctypes.get_last_error())
        return c.peak_working_set
    import resource
    n=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return n if platform.system()=="Darwin" else n*1024

def main(a):
    manifest_bytes=(a.bundle/"manifest.json").read_bytes();m=json.loads(manifest_bytes);e=m["envelope"]
    options=ort.SessionOptions();options.intra_op_num_threads=a.threads
    start=time.perf_counter();session=ort.InferenceSession(str(a.bundle/m["graph"]),sess_options=options,providers=["CPUExecutionProvider"]);load=time.perf_counter()-start
    results=[];rng=np.random.default_rng(20261004)
    shapes=[(1,17,2),(2,31,3),(e["max_batch"],e["max_len"],max(2,e["max_options"]))]
    for b,s,k in shapes:
        if k>s:raise ValueError("qualified marker envelope exceeds sequence capacity")
        data=dict(input_ids=rng.integers(5,100,(b,s),dtype=np.int64),attention_mask=np.ones((b,s),dtype=np.int64),marker_pos=np.tile(np.linspace(1,s-2,k,dtype=np.int64),(b,1)),marker_mask=np.ones((b,k),dtype=bool),qtype=np.arange(b,dtype=np.int64)%3)
        if k==2:data["marker_mask"][0,1]=False
        durations=[]
        for _ in range(a.runs):
            start=time.perf_counter();logits,act=session.run(["logits","act_logits"],data);durations.append(time.perf_counter()-start)
            if logits.shape!=(b,k) or act.shape!=(b,2) or not np.isfinite(logits).all() or not np.isfinite(act).all():raise ValueError("invalid output at declared shape")
        results.append(dict(batch=b,sequence=s,markers=k,runs=a.runs,median_seconds=float(np.median(durations)),p95_seconds=float(np.percentile(durations,95))))
        print(results[-1],flush=True)
    peak=peak_bytes();passed=peak<=e["reservation_mb"]*1024*1024
    report=dict(platform=platform.platform(),architecture=platform.machine(),runtime=ort.__version__,provider="CPUExecutionProvider",threads=a.threads,manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),checkpoint=m["checkpoint"],load_seconds=load,peak_process_bytes=peak,reservation_mb=e["reservation_mb"],reservation_covers_measured_peak=passed,shapes=results)
    a.output.write_text(json.dumps(report,indent=2)+"\n");print(f"peak process: {peak/1024**2:.1f} MiB; reservation covered: {passed}",flush=True)
    if not passed:raise SystemExit("reservation is below measured peak")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--bundle",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--threads",type=int,default=2);p.add_argument("--runs",type=int,default=5);main(p.parse_args())
