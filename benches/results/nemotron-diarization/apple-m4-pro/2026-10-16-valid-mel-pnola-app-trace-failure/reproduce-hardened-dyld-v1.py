"""Model-free reproduction of hardened runtime suppression of dyld environment diagnostics."""
import json
import os
from pathlib import Path
import subprocess
import time

cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
out = cache / 'hardened-dyld-diagnostics-v1'; out.mkdir(exist_ok=False)
source = out / 'probe.c'
source.write_text('#include <stdio.h>\n#include <stdlib.h>\nint main(void) { const char *p=getenv("DYLD_PRINT_LIBRARIES"); printf("DYLD_PRINT_LIBRARIES=%s\\n", p?p:"<absent>"); return 0; }\n')
commands = []; observations = {}
def run(name, argv, env=None):
    started = time.monotonic()
    result = subprocess.run(argv, env=env, capture_output=True)
    (out / (name + '.stdout.log')).write_bytes(result.stdout)
    (out / (name + '.stderr.log')).write_bytes(result.stderr)
    commands.append({'name': name, 'argv': argv, 'returncode': result.returncode, 'elapsed_seconds': time.monotonic() - started})
    (out / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    result.check_returncode()
    return result
for kind in ['plain', 'hardened']:
    binary = out / kind
    run(kind + '-compile', ['/usr/bin/clang', '-Wall', '-Wextra', '-Werror', str(source), '-o', str(binary)])
    run(kind + '-sign', ['/usr/bin/codesign', '--sign', '-', *(['--options', 'runtime'] if kind == 'hardened' else []), str(binary)])
    run(kind + '-signature', ['/usr/bin/codesign', '--display', '--verbose=4', str(binary)])
    result = run(kind + '-run', [str(binary)], {'PATH': '/usr/bin:/bin', 'DYLD_PRINT_LIBRARIES': '1'})
    observations[kind] = {'stdout': result.stdout.decode(), 'stderr_bytes': len(result.stderr), 'dyld_trace_observed': b'dyld[' in result.stderr}
assert observations['plain']['stdout'] == 'DYLD_PRINT_LIBRARIES=1\n' and observations['plain']['dyld_trace_observed']
assert observations['hardened']['stdout'] == 'DYLD_PRINT_LIBRARIES=<absent>\n' and not observations['hardened']['dyld_trace_observed']
(out / 'result.json').write_text(json.dumps({'scope': __doc__, 'reproduced': True, 'observations': observations, 'app_or_native_model_calls': 0}, indent=2) + '\n')
print(json.dumps(observations))
