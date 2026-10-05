"""Record linked Android ELF evidence; this does not execute on a device."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def main(a):
    artifacts=[]
    for path in sorted(a.libraries.glob("*.so")):
        text=subprocess.check_output([str(a.readelf),"-h","-l","-d","--wide",str(path)],text=True)
        exports=subprocess.check_output([str(a.readelf),"--dyn-syms","--wide",str(path)],text=True)
        needed=re.findall(r'\(NEEDED\).*?\[(.*?)\]',text)
        alignments=[int(line.split()[-1],16) for line in text.splitlines() if line.strip().startswith("LOAD ")]
        machine = "AArch64" if a.abi == "arm64-v8a" else "Advanced Micro Devices X86-64"
        if machine not in text or not alignments or any(n<16384 for n in alignments):
            raise ValueError(f"{path.name}: wrong architecture or insufficient LOAD alignment")
        if any("/" in n or "\\" in n or ":" in n for n in needed):
            raise ValueError("absolute build path leaked into DT_NEEDED")
        expected = ["gen2_laya_open", "gen2_laya_decide", "gen2_laya_invoke", "gen2_laya_suspend", "gen2_laya_resume", "gen2_laya_close", "gen2_laya_free"] if path.name=="libgen2_laya_mobile.so" else ["Java_example_gen2_LayaNative_"+name+"Bytes" for name in ("open","decide","invoke","suspend","resume","close")] if path.name=="libgen2_laya_jni.so" else ["OrtGetApiBase"]
        for symbol in expected:
            if not any(line.split() and line.split()[-1].split("@")[0]==symbol and not re.search(r'\bUND\b',line) for line in exports.splitlines()):
                raise ValueError(f"missing exported ABI symbol {symbol}")
        artifacts.append(dict(file=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              needed=needed,load_alignments=alignments,checked_exports=expected))
    if {item["file"] for item in artifacts} != {"libgen2_laya_mobile.so","libgen2_laya_jni.so","libonnxruntime.so"}:
        raise ValueError("expected Rust, JNI and ONNX libraries")
    system_libraries={"libc.so","libdl.so","libm.so","liblog.so","libandroid.so"}
    packaged={item["file"] for item in artifacts}
    for item in artifacts:
        missing=set(item["needed"])-packaged-system_libraries
        if missing:raise ValueError(f"unpackaged native dependencies: {missing}")
    workspace=Path(__file__).resolve().parents[2]
    sources=sorted(list((workspace/"src").rglob("*.rs"))+list((workspace/"examples/laya-mobile").rglob("*.rs"))+
                   [workspace/name for name in ("Cargo.toml","Cargo.lock","examples/laya-mobile/Cargo.toml",
                                                "examples/laya-mobile/android/laya_jni.cpp","examples/laya-mobile/gen2_laya.h")])
    inventory=[(p.relative_to(workspace).as_posix(),hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources]
    target = "aarch64-linux-android" if a.abi == "arm64-v8a" else "x86_64-linux-android"
    result=dict(status="cross-compiled and linked; device execution not verified",target=target,abi=a.abi,android_api=a.api,
                ndk=(a.ndk/"source.properties").read_text(),ort_android_version="1.24.3",artifacts=artifacts,
                source_tree_sha256=hashlib.sha256(json.dumps(inventory,separators=(",", ":")).encode()).hexdigest(),
                source_file_count=len(inventory),
                sources={"ndk":"https://github.com/android/ndk/releases/tag/r27d",
                         "runtime":"https://repo.maven.apache.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.24.3/"})
    a.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(result["status"])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("libraries","readelf","ndk","output"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--api",type=int,default=24)
    p.add_argument("--abi",choices=["arm64-v8a","x86_64"],default="arm64-v8a")
    main(p.parse_args())
