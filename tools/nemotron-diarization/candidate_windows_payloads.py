"""Replay the complete delivered Windows Q8 producer payload; never load a DLL."""
import json
from pathlib import Path, PureWindowsPath
import re
import tempfile

from build_valid_mel_candidate import descriptor
from candidate_windows_compiler_receipt import verify_compiler_records
from candidate_windows_sdk_receipt import validate_observation_metadata, validate_producer_metadata, PROTOCOL_IDENTITY
from ctest_output import require_single_test_pass
from platform_bounded_metrics import derive_native_probe
from pooled_runtime import compatibility_patch_bytes
from windows_naming import verify_private_outputs
from windows_compiler_fp_metrics import baseline
from windows_q8_candidate import LIFECYCLE, ORDER, PROTOCOL, require_host, require_uninstrumented
from windows_q8_candidate_controls import LIBRARY_NAMES
from windows_q8_candidate_metrics import admit_candidate_audit, compare_candidate_pair, both_pairs_realtime
from windows_q8_candidate_sources import load_recipe
from windows_q8_model_metrics import graph_vectors
from windows_q8_model_sources import Q8_PATCH, q8_patch, PATH
from windows_session_timing import file_inventory


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def verify_output_bindings(evidence, record, ci_evidence):
    actual = {str(ci_evidence / name): value for name, value in file_inventory(evidence).items()
              if name != 'commands.json'}
    require(actual == record.get('retained_output_bindings'), 'complete retained producer outputs differ')
    require(read(evidence / 'commands.json') == record['commands'], 'producer command ledger differs')


def verify_objects(evidence, build, database, audit, ci_source, ci_evidence):
    ci_build = ci_source / 'build/gen2-cpu-diar'
    inputs = {PureWindowsPath(name): value for name, value in audit['all_compiler_sources'].items()}
    require(len(database) == 97 and {PureWindowsPath(row['file']) for row in database} == set(inputs),
            'all97 audited compiler sources required')
    kernel = build['kernel']
    units = kernel['translation_units']
    require(len(units) == 97, 'all97 retained compiler objects required')
    bindings, seen = {}, set()
    for unit, row in zip(units, database):
        source, obj = PureWindowsPath(row['file']), PureWindowsPath(row['output'])
        if not obj.is_absolute():
            obj = ci_build / obj
        require(unit['command'] == row and PureWindowsPath(row['directory']) == ci_build
                and source.is_relative_to(ci_source) and obj.is_relative_to(ci_build) and obj not in seen,
                'unique isolated source/command/object associations required')
        seen.add(obj)
        retained = 'compiler-objects/' + obj.relative_to(ci_build).as_posix()
        require(unit['retained_object'] == retained and descriptor(evidence / retained) == unit['object']
                and unit['source'] == inputs[source], 'retained compiler object or audited source differs')
        bindings.update({str(source): unit['source'], str(obj): unit['object'], str(ci_evidence / retained): unit['object']})
    selected = [row for row in database if PureWindowsPath(row['file']) == ci_build / '_deps/llama.cpp' / PATH]
    require(len(selected) == 1 and kernel['compiler_command'] == selected[0], 'one actual SGEMM compiler input required')
    unit, = [unit for unit in units if unit['command'] == selected[0]]
    require(descriptor(evidence / 'kernel.obj') == unit['object'], 'SGEMM disassembly object differs')
    return bindings


def verify_graph_payloads(evidence, controls, ci_build, ci_evidence, ci_repo):
    libraries = controls['retained_import_libraries']
    require(controls['graph_input_payloads_fully_retained'] is True and len(libraries) == 3
            and {PureWindowsPath(name).name for name in libraries} == LIBRARY_NAMES,
            'all three retained graph import libraries required')
    for name, value in libraries.items():
        path = PureWindowsPath(name)
        require(path.is_relative_to(ci_build) and value['retained_path'] == 'graph-inputs/' + path.name,
                'isolated original and retained graph library paths required')
        identity = {key: value[key] for key in ('bytes', 'sha256')}
        require(identity['bytes'] > 0 and descriptor(evidence / value['retained_path']) == identity
                and controls['bindings'][name] == identity, 'retained graph library payload differs')
    for name, identity in controls['bindings'].items():
        path = PureWindowsPath(name)
        if path.is_relative_to(ci_evidence):
            actual = evidence / path.relative_to(ci_evidence).as_posix()
        elif path == ci_repo / 'tools/nemotron-diarization/ggml_matrix_profile_control.cpp':
            actual = Path(__file__).resolve().parent / 'ggml_matrix_profile_control.cpp'
        else:
            require(name in libraries, 'unexpected graph input binding')
            actual = evidence / libraries[name]['retained_path']
        require(descriptor(actual) == identity, 'actual retained graph binding differs')
    require(len(controls['bindings']) == 50 and controls['vectors'] == 36
            and [case['threads'] for case in controls['cases']] == [1, 2, 4], 'all36 graph vectors and50 bindings required')
    reference = None
    for case in controls['cases']:
        threads = case['threads']
        phase = f'matrix-control-{threads}'
        paths = [evidence / (phase + f'.{shape}.{repeat}.f32') for shape in range(4) for repeat in range(3)]
        reference = graph_vectors((evidence / (phase + '.stdout.log')).read_text(), threads,
                                  [path.read_bytes() for path in paths], reference)
        outputs = {str(ci_evidence / path.name): descriptor(path)
                   for path in [*paths, evidence / (phase + '.stdout.log'), evidence / (phase + '.log')]}
        require(outputs == case['outputs_and_logs']
                and all(controls['bindings'][name] == value for name, value in outputs.items()),
                'complete graph output/log associations differ')
        require_uninstrumented((evidence / (phase + '.log')).read_text())


def verify_probe_and_graph_commands(commands, controls, ci_source, ci_evidence, ci_repo):
    root, build = ci_source.parent, ci_source / 'build/gen2-cpu-diar'
    probe, = [row for row in commands if row['name'] == 'native-probe-build']
    compiler = probe['command'][0]
    require(compiler == baseline('msvc')[2]['compiler'], 'original pinned MSVC probe compiler path required')
    hashing = ci_source / 'llama.cpp/vendor/hash'
    entry = 'gen2_nemo_windows_nemo_speech_asr_c'
    expected_probe = [compiler, '/nologo', '/std:c++17', '/O2', '/EHsc', '/TP',
        str(ci_evidence / 'native_stream_probe_bounded.cpp'), str(hashing / 'sha256/sha256.c'),
        '/I' + str(root / 'sdk/include'), '/I' + str(ci_source / 'examples'), '/I' + str(hashing),
        '/link', str(root / ('sdk/lib/' + entry + '.lib')),
        '/OUT:' + str(root / 'sdk/bin/gen2_native_stream_probe_valid_mel.exe')]
    require(probe['command'] == expected_probe and PureWindowsPath(probe['cwd']) == ci_evidence,
            'complete original native probe build command required')
    libraries = controls['retained_import_libraries']
    selected = []
    for stem in ('ggml', 'ggml-base', 'ggml-cpu'):
        paths = [name for name in libraries if PureWindowsPath(name).name == 'gen2_nemo_windows_' + stem + '.lib']
        require(len(paths) == 1, 'one actual graph linker input per original target required')
        selected.append(paths[0])
    matrix, = [row for row in commands if row['name'] == 'matrix-control-build']
    expected_matrix = [compiler, '/nologo', '/std:c++17', '/O2', '/EHsc', '/TP',
        str(ci_repo / 'tools/nemotron-diarization/ggml_matrix_profile_control.cpp'),
        '/I' + str(build / '_deps/llama.cpp/ggml/include'), '/link', selected[0],
        '/OUT:' + str(ci_evidence / 'matrix-profile-control.exe'), *selected[1:]]
    require(matrix['command'] == expected_matrix and PureWindowsPath(matrix['cwd']) == ci_evidence,
            'actual graph compilation must select the retained linker inputs')
    for threads in (1, 2, 4):
        name = 'matrix-control-' + str(threads)
        call, = [row for row in commands if row['name'] == name]
        require(call['command'] == [str(ci_evidence / 'matrix-profile-control.exe'), str(threads), str(ci_evidence / name)]
                and PureWindowsPath(call['cwd']) == ci_repo, 'complete original graph execution command required')


def replay_histories(evidence, record):
    cases = record['cases']
    comparisons = []
    for index, case in enumerate(cases):
        phase = f'case-{index}-{ORDER[index]}'
        target, raw = evidence / (phase + '.json'), evidence / (phase + '.json.f32')
        require(read(target) == case['result'] and descriptor(target) == case['result_file']
                and descriptor(raw) == case['probabilities'], 'retained full-history bytes or metadata differ')
        if ORDER[index] == 'public':
            require(read(evidence / (phase + '.stdout.log')) == case['result'], 'public process stdout differs')
        require_uninstrumented((evidence / (phase + '.log')).read_text())
        if index in (1, 3):
            first, second = cases[index - 1:index + 1]
            native, public = (first, second) if first['kind'] == 'native' else (second, first)
            comparisons.append(compare_candidate_pair(public['result'], native['result'],
                (evidence / native['probability_path']).read_bytes(), (evidence / public['probability_path']).read_bytes()))
    require(comparisons == record['comparisons'], 'actual full-history timings and frozen probability bytes differ')
    return comparisons


def verify_histories(evidence, record):
    comparisons = replay_histories(evidence, record)
    require(both_pairs_realtime(comparisons), 'all four original full-history RTFs must be below one')
    return comparisons


def verify_lifecycle_logs(evidence, record):
    native = record['cases'][0]
    expected = {'segments': [{'start': turn['start_seconds'], 'end': turn['end_seconds'], 'speaker': turn['speaker']}
                             for turn in native['result']['turns']]}
    require(read(evidence / 'reference/ami.json') == expected
            and (evidence / 'reference/ami.json.f32').read_bytes() == (evidence / native['probability_path']).read_bytes(),
            'lifecycle reference differs from complete native history')
    listed = {line.removesuffix(': test') for line in (evidence / 'native-lifecycle-list.stdout.log').read_text().splitlines()
              if line.endswith(': test')}
    require(set(LIFECYCLE) <= listed, 'all seven lifecycle tests must be compiled')
    for index, test in enumerate(LIFECYCLE):
        phase = 'native-lifecycle-' + str(index)
        stdout = (evidence / (phase + '.stdout.log')).read_text()
        require(re.search(r'^test ' + re.escape(test) + r' \.\.\. ok$', stdout, re.MULTILINE)
                and re.search(r'^test result: ok\. 1 passed; 0 failed; 0 ignored;[^\r\n]*$', stdout, re.MULTILINE),
                'one actual passing lifecycle execution required: ' + test)
        require_uninstrumented((evidence / (phase + '.log')).read_text())


def replay_candidate_delivery(directory):
    """Replay complete retained observations, including an unsuccessful timing gate."""
    directory = Path(directory)
    require(not directory.is_symlink(), 'regular candidate delivery root required')
    root = directory.resolve(strict=True)
    before = file_inventory(root)
    require({path.name for path in root.iterdir()} == {'qualification.json', 'protocol.json', 'evidence', 'sdk'},
            'closed producer delivery required')
    record = read(root / 'qualification.json')
    _, lock = validate_observation_metadata(record)
    tools = Path(__file__).resolve().parent
    require(descriptor(root / 'protocol.json') == PROTOCOL_IDENTITY
            and (root / 'protocol.json').read_bytes() == (tools / PROTOCOL).read_bytes(), 'retained producer protocol differs')
    protocol = read(root / 'protocol.json')
    for group, base in (('tools', tools), ('repository_files', tools.parents[1])):
        for name, identity in protocol[group].items():
            require(descriptor(base / name) == identity, 'current frozen producer input differs: ' + name)
    evidence, sdk, build = root / 'evidence', root / 'sdk', record['build']
    ci_source = PureWindowsPath(record['commands'][0]['command'][-1])
    ci_root, ci_evidence = ci_source.parent, ci_source.parent / 'evidence'
    ci_build, ci_repo = ci_source / 'build/gen2-cpu-diar', PureWindowsPath(record['commands'][0]['cwd'])
    require(ci_repo.is_absolute(), 'absolute recorded consumer repository required')
    verify_output_bindings(evidence, record, ci_evidence)
    require(file_inventory(sdk) == build['installed_files'], 'actual SDK inventory differs from qualified producer')
    require(record['assets'] == {item['name']: {'bytes': item['size'], 'sha256': item['sha256']}
                                for item in lock['artifacts'] if item['role'] != 'reference'},
            'original frozen model and audio identities required')
    require(read(evidence / 'build-recipe.json') == record['build_recipe']
            and read(evidence / 'runtime-profile.json') == lock, 'retained compiler/runtime recipe differs')
    require((evidence / 'version-only.stdout.log').read_text().strip() == lock['runtime']['reported_version'],
            'observed bounded runtime identity differs')
    require(read(evidence / 'receiver-host.stdout.log') == record['host'], 'actual receiver host report differs')
    require_host(record['host'])
    loader = read(evidence / 'windows-default-loader-control.stdout.log')
    require(loader == {'passed': True, 'library': str(ci_root / 'sdk/bin/gen2_nemo_windows_nemo_speech_asr_c.dll'),
                       'load_flags': 0, 'native_win32_error': 126, 'default_loader_resolves_siblings': False},
            'original native Windows default-loader negative control required')
    require_single_test_pass((evidence / 'model-free-state-test.stdout.log').read_text(), 'diar_state')
    require((evidence / 'bounded-inference.patch').read_bytes() == compatibility_patch_bytes(lock), 'bounded inference patch differs')
    for patch in lock['runtime']['compatibility_patches']:
        require(descriptor(evidence / Path(patch['path']).name) == {key: patch[key] for key in ('bytes', 'sha256')},
                'retained bounded native patch differs')
    require((evidence / Q8_PATCH).read_bytes() == q8_patch(), 'retained exact SGEMM patch differs')
    probe = build['native_probe_source']
    require(probe['original'] == descriptor(tools / 'native_stream_probe_valid_mel.cpp')
            and probe['derived'] == descriptor(evidence / 'native_stream_probe_bounded.cpp')
            and (evidence / 'native_stream_probe_bounded.cpp').read_bytes() == derive_native_probe(
                (tools / 'native_stream_probe_valid_mel.cpp').read_bytes()), 'retained bounded native probe differs')
    with tempfile.TemporaryDirectory(prefix='q8-sdk-pe-') as folder:
        verify_private_outputs(sdk, Path(folder))
        require(read(Path(folder) / 'windows-private-pe.json') == read(evidence / 'windows-private-pe.json'),
                'actual private SDK PE identities differ')
    audit = read(evidence / 'platform-llama-materialized-audit.json')
    require(admit_candidate_audit(audit) == build['exact_source'], 'actual bounded/SGEMM source audit differs')
    require(descriptor(evidence / 'platform-pristine-llama.tar') == audit['archive']
            and descriptor(evidence / 'compile_commands.json') == audit['compile_database'], 'compiler archive/database differs')
    database = read(evidence / 'compile_commands.json')
    compiler = read(evidence / 'compiler-admission.json')
    require(compiler == build['compiler_admission'], 'retained compiler admission differs')
    _, recipe = load_recipe()
    verify_compiler_records((evidence / 'CMakeCache.txt').read_text(), database,
        {language: (evidence / ('compiler-' + language + '.cmake')).read_text() for language in ('C', 'CXX')}, compiler, recipe)
    bindings = verify_objects(evidence, build, database, audit, ci_source, ci_evidence)
    dump, = [row for row in record['commands'] if row['name'] == 'kernel-disassembly']
    require(dump['command'][1:] == ['/nologo', '/disasm', '/symbols', str(ci_evidence / 'kernel.obj')]
            and PureWindowsPath(dump['command'][0]).name.lower() == 'dumpbin.exe', 'actual SGEMM disassembly command differs')
    bindings[str(PureWindowsPath(dump['command'][0]))] = build['kernel']['bindings'][str(PureWindowsPath(dump['command'][0]))]
    for name in ('kernel.obj', 'kernel-disassembly.stdout.log', 'kernel-disassembly.log'):
        bindings[str(ci_evidence / name)] = descriptor(evidence / name)
    require(bindings == build['kernel']['bindings']
            and 'llamafile_sgemm' in (evidence / 'kernel-disassembly.stdout.log').read_text(), 'SGEMM object/disassembly bindings differ')
    verify_graph_payloads(evidence, build['model_free_controls'], ci_build, ci_evidence, ci_repo)
    verify_probe_and_graph_commands(record['commands'], build['model_free_controls'], ci_source, ci_evidence, ci_repo)
    public = build['public_executables']
    require(len(public) == 4 and len({PureWindowsPath(name).name for name in public}) == 2,
            'both original and retained public/test executables required')
    for name, identity in public.items():
        path = PureWindowsPath(name)
        require((path.is_relative_to(ci_evidence / 'public-binaries')
                 or path.parts[-3:-1] in (('release', 'examples'), ('release', 'deps')))
                and descriptor(evidence / 'public-binaries' / path.name) == identity, 'retained public executable differs')
    comparisons = replay_histories(evidence, record)
    realtime = both_pairs_realtime(comparisons)
    require(record['realtime_gate_passed'] is realtime, 'recorded timing outcome differs from actual histories')
    verify_lifecycle_logs(evidence, record)
    require(file_inventory(root) == before, 'candidate payload changed during package admission')
    return {'producer_receipt': descriptor(root / 'qualification.json'), 'source_commit': record['source_commit'],
            'protocol': PROTOCOL_IDENTITY, 'retained_compiler_objects_verified': 97,
            'retained_graph_libraries_verified': 3, 'graph_vectors_verified': 36, 'comparisons': comparisons,
            'lifecycle_tests_verified': LIFECYCLE, 'payload_replay_passed': True, 'package_qualified': False,
            'functional_lifecycle_gates_passed': True, 'realtime_gate_passed': realtime,
            'production_candidate_sdk_gates_passed': realtime, 'release_qualified': False,
            'limits': 'Retained producer SDK payload replay only. Compiler binaries/aliases are CI observations. Package construction, offline/coexistence/desktop, quality and release remain separate.'}


def verify_candidate_delivery(directory):
    """Require a passing producer and complete payloads before package construction."""
    validate_producer_metadata(read(Path(directory) / 'qualification.json'))
    result = replay_candidate_delivery(directory)
    require(result['production_candidate_sdk_gates_passed'] is True, 'passing candidate SDK required for packaging')
    return result
