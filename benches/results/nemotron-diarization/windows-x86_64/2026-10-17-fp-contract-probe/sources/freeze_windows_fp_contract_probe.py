"""Freeze the standalone arithmetic probe and complete local import closure."""
import ast
import hashlib
import json
from pathlib import Path


def descriptor(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    names = {'fp_contract_probe.cpp'}
    pending = ['windows_fp_contract_probe.py', 'test_windows_fp_contract_probe.py',
               'test_windows_job_v2.py', 'freeze_windows_fp_contract_probe.py']
    while pending:
        name = pending.pop()
        if name in names:
            continue
        names.add(name)
        for node in ast.walk(ast.parse((tools / name).read_bytes())):
            imports = ([node.module] if isinstance(node, ast.ImportFrom) else
                       [item.name for item in node.names] if isinstance(node, ast.Import) else [])
            for module in imports:
                local = (module or '').split('.')[0] + '.py'
                if (tools / local).is_file() and local not in names:
                    pending.append(local)
    references = ['.github/workflows/diarization-windows-fp-contract-probe.yml',
                  'docs/research/nemotron-windows-fp-contract-probe.md',
                  'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-compiler-parity-failure/collection-qualification.json']
    p = {'schema_version': 1, 'tools': {name: descriptor(tools / name) for name in sorted(names)},
         'repository_files': {name: descriptor(repo / name) for name in references},
         'limits': ['No SDK construction/model execution or qualification; scalar compiler hypothesis only.']}
    target = tools / 'windows-fp-contract-probe.protocol.json'
    target.write_text(json.dumps(p, indent=2) + '\n')
    print(json.dumps({'protocol': descriptor(target), 'tools': len(names), 'repository_files': len(references)}))


if __name__ == '__main__':
    main()
