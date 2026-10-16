"""Compile the actual verification image module in a hardened model-free Rust probe."""
import hashlib
import json
from pathlib import Path
import subprocess

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0'); cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
out = cache / 'process-image-callback-v1'; out.mkdir(exist_ok=False)
module = repo / 'integrations/pnola/overlay/src-tauri/src/verification_images.rs'
source = out / 'probe.rs'
source.write_text('#[path=' + json.dumps(str(module)) + ']\nmod verification_images;\n'
                  'unsafe extern "C" { fn dlopen(name: *const std::ffi::c_char, flags: i32) -> *mut std::ffi::c_void; fn dlclose(handle: *mut std::ffi::c_void) -> i32; }\n'
                  'fn main() { verification_images::start(); verification_images::start(); eprintln!("BEFORE_DLOPEN"); '
                  'let handle = unsafe { dlopen(c"/usr/lib/libiconv.2.dylib".as_ptr(), 2) }; assert!(!handle.is_null()); '
                  'assert_eq!(unsafe { dlclose(handle) }, 0); eprintln!("AFTER_DLCLOSE"); '
                  'println!("DYLD_PRINT_LIBRARIES={}", std::env::var("DYLD_PRINT_LIBRARIES").unwrap_or("<absent>".into())); }\n')
commands = []
for name, argv in [('compile', ['/Users/victor/.cargo/bin/rustc', '--edition=2024', '-Dwarnings', str(source), '-o', str(out / 'probe')]),
                   ('sign', ['/usr/bin/codesign', '--sign', '-', '--options', 'runtime', str(out / 'probe')]),
                   ('signature', ['/usr/bin/codesign', '--display', '--verbose=4', str(out / 'probe')]),
                   ('run', [str(out / 'probe')])]:
    env = {'PATH': '/usr/bin:/bin', 'HOME': '/Users/victor', 'DYLD_PRINT_LIBRARIES': '1'}
    result = subprocess.run(argv, env=env, capture_output=True)
    (out / (name + '.stdout.log')).write_bytes(result.stdout); (out / (name + '.stderr.log')).write_bytes(result.stderr)
    commands.append({'name': name, 'argv': argv, 'environment': env, 'returncode': result.returncode})
    (out / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n'); result.check_returncode()
trace = (out / 'run.stderr.log').read_text(); lines = trace.splitlines()
assert 'PNOLA_LOADED_IMAGE_ERROR' not in trace
images = [(index, bytes.fromhex(line.removeprefix('PNOLA_LOADED_IMAGE_HEX ')).decode()) for index, line in enumerate(lines) if line.startswith('PNOLA_LOADED_IMAGE_HEX ')]
assert len(images) == len({path for _, path in images})
assert str(out / 'probe') in {path for _, path in images}
assert any(lines.index('BEFORE_DLOPEN') < index < lines.index('AFTER_DLCLOSE') and path == '/usr/lib/libiconv.2.dylib' for index, path in images)
assert (out / 'run.stdout.log').read_text() == 'DYLD_PRINT_LIBRARIES=<absent>\n'
assert '(adhoc,runtime)' in (out / 'signature.stderr.log').read_text()
desc = lambda p: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
r = {'scope': __doc__, 'passed': True, 'module': desc(module), 'binary': desc(out / 'probe'), 'current_and_future_images': len(images),
     'registration_idempotent': True, 'future_dlopen_and_dlclose_history_observed': True,
     'hardened_signing_and_dyld_environment_suppression_retained': True, 'app_or_model_calls': 0}
(out / 'result.json').write_text(json.dumps(r, indent=2) + '\n'); print(json.dumps(r))
