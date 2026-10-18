"""Local saved-evidence replay. No compiler, model, subprocess or Git mutation."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import stat
import struct
import sys


def descriptor(path):
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError('regular nonsymlink input required: ' + str(path))
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        while block := stream.read(1024 ** 2):
            digest.update(block)
            size += len(block)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=no_duplicate_keys)


def require(test, message):
    if not test:
        raise ValueError(message)


def verify_files(root, files):
    for name, wanted in files.items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'relative member required')
        path = root / relative
        require(all(not parent.is_symlink() for parent in [path, *path.parents]), 'no symlink input')
        require(descriptor(path) == wanted, 'input changed: ' + str(path))


def verify_collection(root, expected):
    require(descriptor(root / 'collection.json') == expected['manifest'], 'first-bound manifest changed')
    manifest = read(root / 'collection.json')
    require(set(manifest) == {'files'} or
            (set(manifest) == {'schema_version', 'files'} and manifest['schema_version'] == 1),
            'closed collection format required')
    files = manifest['files']
    require('collection.json' not in files, 'self manifest forbidden')
    actual = set()
    for path in root.rglob('*'):
        require(not path.is_symlink(), 'symlink collection member')
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
        else:
            require(path.is_dir(), 'special collection member')
    require(actual == set(files) | {'collection.json'}, 'collection membership changed')
    verify_files(root, files)
    require(len(actual) == expected['members'] and
            sum(value['bytes'] for value in files.values()) == expected['bytes_without_manifest'],
            'collection size changed')
    return files


def execute(nodes, scope, label):
    tree = ast.Module(body=nodes, type_ignores=[])
    identity = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
    exec(compile(tree, label, 'exec'), scope)
    return identity


def assignment(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name
                                               for target in node.targets)


def reviewed_body(original, returned, relocation):
    """Accept only the exact reviewed AST with the two fixed CI path prefixes."""
    tree = ast.parse(original.read_bytes())
    class Paths(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                value = node.value
                for old, new in relocation.items():
                    value = value.replace(old, new)
                return ast.copy_location(ast.Constant(value), node)
            return node
    relocated = Paths().visit(ast.parse(original.read_bytes()))
    require(ast.dump(relocated, include_attributes=False) ==
            ast.dump(ast.parse(returned.read_bytes()), include_attributes=False),
            'returned validation AST differs from reviewed original')
    return tree.body


def section(body, start, stop):
    first, = [i for i, node in enumerate(body) if assignment(node, start)]
    last, = [i for i, node in enumerate(body) if assignment(node, stop)]
    require(first < last, 'ordered original AST block required')
    return body[first:last]


def main():
    require(sys.flags.optimize == 0, 'assertions must remain enabled')
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    first_protocol = descriptor(args.protocol)
    require(first_protocol['sha256'] == args.protocol_sha256, 'pre-execution protocol digest changed')
    pre = read(args.protocol)
    require(pre['schema_version'] == 1 and pre['new_compiler_sdk_model_calls'] == 0,
            'saved replay protocol required')
    repo, local = Path(pre['repo']), Path(pre['local'])
    tools = repo / 'tools/nemotron-diarization'
    delivered = local / 'materialized' / pre['result_path']
    require(str(Path(__file__).resolve()) in pre['files'], 'replayer must be first-bound')
    for name, wanted in pre['files'].items():
        require(descriptor(Path(name)) == wanted, 'first-bound input changed: ' + name)
    admission = read(local / 'admission.json')
    require(admission['remote_source_commit'] == 'c7bc18994fb8247fd3723ce3c3645039cd14dcaf' and
            admission['remote_result_commit'] == '580e79ac6943384a3ec545c0c9f2a540e48e9e6d' and
            admission['remote_run_id'] == 39084790919 and admission['remote_job_id'] == 117548983776,
            'fixed remote provenance required')
    require(admission['source_refs_or_index_changed'] is False and
            admission['production_or_release_qualified'] is False,
            'quarantine-only admission required')
    require(admission['collections'][pre['result_path']] == pre['result_collection'] and
            admission['source_preparation_proof'] == pre['source_preparation_collection'],
            'original admitted collection descriptors required')
    retained = verify_collection(delivered, pre['result_collection'])
    prep_root = local / 'delivery/source-preparation'
    verify_collection(prep_root, pre['source_preparation_collection'])
    verify_files(local / 'delivery', admission['delivery_members'])
    require(descriptor(local / 'delivery.zip') == admission['archive'], 'return ZIP changed')
    p = read(delivered / 'protocol.json')
    require(descriptor(delivered / 'protocol.json') == pre['native_protocol'] and
            descriptor(tools / 'windows-thread-cpu-profile.protocol.json') == pre['native_protocol'],
            'original native protocol required')
    require(len(p['tools']) == 135 and len(p['repository_files']) == 2880,
            'original native source closure required')
    verify_files(tools, p['tools'])
    verify_files(repo, p['repository_files'])
    verify_files(delivered / 'sources', p['tools'])
    route = read(repo / 'tools/nemotron-diarization/remote-thread-cpu-replay.protocol.json')
    require(len(route['files']) == 98, 'reviewed remote route closure required')
    for name, relative in pre['reviewed_originals'].items():
        require(pre['files'][str(repo / relative)] == route['files'][relative],
                'reviewed original must retain its remote-protocol pin')
    reviewed = {name: reviewed_body(repo / relative, delivered / name, pre['literal_relocation'])
                for name, relative in pre['reviewed_originals'].items()}
    print('bootstrap: 449 result members, 42 preparation members, 135 tools and 2880 references verified', flush=True)

    # All executable project code is local and admitted above. Returned Python
    # modules are never imported; only the fixed original validation AST blocks
    # selected below are compiled after their complete payload admission.
    sys.path.insert(0, str(tools))
    from ctest_output import require_single_test_pass
    from ggml_thread_cpu_profile import MAX_LOG_BYTES, summarize
    from windows_compiler_metrics import compiler_evidence as parent_compiler_evidence
    from windows_compiler_fp_metrics import FLAG, baseline, cache_value, normalized_commands
    from windows_matrix_profile_report import admit_saved_profile
    from windows_thread_cpu_profile_metrics import (ORDER, admit_profile_audit, compare_native,
        derive_probe, metadata, profile_signature, telemetry, timing_evidence, version)
    from windows_thread_cpu_profile_sources import PROFILE_PATCH, profile_patch
    from windows_worker_metrics import version as worker_version

    q = read(delivered / 'qualification.json')
    collection = read(delivered / 'collection-qualification.json')
    members = read(delivered / 'artifact-members.json')
    evidence = delivered / 'evidence'
    build = q['build']
    require(q['source_commit'] == '15c905d8815190a54f25452e7e51378c24f85914' and
            q['protocol'] == pre['native_protocol'] and q['diagnostic_completed'] is True and
            q['order'] == p['order'] == ORDER and q['limits'] == p['limits'], 'native diagnostic binding changed')
    require(all(q[key] is False for key in ('release_qualified','package_qualified','public_or_lifecycle_qualified')),
            'diagnostic must not qualify release')
    commands = q['commands']
    require(len(commands) == 59 and commands == read(evidence / 'commands.json'), 'complete native ledger required')
    for row in commands:
        for suffix in ('.stdout.log', '.log'):
            name = 'evidence/' + row['name'] + suffix
            require(descriptor(delivered / name) == members[name], 'native log binding changed')
    require(q['host'] == read(evidence / 'receiver-host.stdout.log') and
            q['host']['avx2'] is True and q['host']['fma'] is True, 'actual native host evidence required')

    scope = {**globals(), **locals()}
    body = reviewed['replay-windows-thread-cpu-profile-supervision-v1.py']
    start, = [i for i, node in enumerate(body) if assignment(node, 'summaries')]
    stop, = [i for i, node in enumerate(body) if isinstance(node, ast.Assert) and
             ast.unparse(node.test).startswith('descriptor(Path(__file__)) ==')]
    supervision_ast = execute(body[start:stop], scope, 'original-saved-Windows-Job-supervision')
    require(scope['summaries'] == read(delivered / 'supervision-replay.json')['commands'],
            'independent Windows Job reconstruction differs')

    body = reviewed['collector.py']
    order_checks = [node for node in body if isinstance(node, ast.Assert) and
                    ast.unparse(node.test).startswith("[row['name'] for row in q['commands']] ==")]
    require(len(order_checks) == 1, 'exact original 59-command order assertion required')
    command_order_ast = execute(order_checks, scope, 'original-exact-59-command-order')
    audit = read(evidence / 'platform-llama-materialized-audit.json')
    require(descriptor(evidence / 'platform-llama-materialized-audit.json') == build['compiler_audit'],
            'compiler audit binding changed')
    require(build['exact_profile_source'] == admit_profile_audit(audit), 'declared source delta changed')
    database = read(evidence / 'compile_commands.json')
    require(descriptor(evidence / 'compile_commands.json') == audit['compile_database'] and len(database) == 97,
            'complete compiler database binding required')
    materialized = PureWindowsPath(audit['compiled_tree_path'])
    require({PureWindowsPath(row['file']) for row in database} ==
            {PureWindowsPath(name) for name in audit['all_compiler_sources']}, 'compiler source identities changed')
    require(all(descriptor(evidence / name) == wanted for name, wanted in build['compiler_evidence_files'].items()),
            'retained compiler evidence changed')
    require_single_test_pass((evidence / 'model-free-state-test.stdout.log').read_text(), 'diar_state')
    require((evidence / 'version-only.stdout.log').read_text().strip() == version(4), 'diagnostic version changed')
    original, derived = tools / 'native_stream_probe_valid_mel.cpp', evidence / 'native_stream_probe_bounded.cpp'
    require(descriptor(original) == build['native_probe_source']['original'] and
            descriptor(derived) == build['native_probe_source']['derived'] and
            derived.read_bytes() == derive_probe(original.read_bytes(), 4), 'derived probe source changed')
    require((evidence / PROFILE_PATCH).read_bytes() == profile_patch(), 'retained CPU observer patch changed')
    scope.update(locals())
    compiler_ast = execute(section(body, 'aliases', 'control'), scope, 'original-saved-97-compiler-command-admission')
    compiler = scope['compiler']
    require(compiler == collection['compiler'], 'independent compiler metadata reconstruction differs')

    nodes = section(body, 'control', 'reference_path')
    # The original collector admitted this binary remotely before constructing
    # the result. It was deliberately not returned. Keep that remote-only check
    # explicit; execute every other original control assertion unchanged.
    excluded = [node for node in nodes if isinstance(node, ast.Assert) and
                ast.unparse(node.test) == "descriptor(evidence / 'matrix-profile-control.exe') == control['binary']"]
    require(len(excluded) == 1 and not (evidence / 'matrix-profile-control.exe').exists(),
            'exact single unavailable-binary assertion expected')
    control_ast = execute([node for node in nodes if node not in excluded], scope, 'original-saved-12-graph-control-admission')
    # End before full original-archive checks and publication, neither of which
    # is possible or appropriate for this bounded local returned subset.
    first, = [i for i, node in enumerate(body) if assignment(node, 'reference_path')]
    last, = [i for i, node in enumerate(body) if isinstance(node, ast.Assert) and
             ast.unparse(node.test) == "timing == q['timing_comparison']"]
    model_ast = execute(body[first:last + 1], scope, 'original-saved-model-counter-profile-timing-admission')
    for key, name in [('cases','cases'),('operations','per_worker_operations'),('matrix_groups','per_worker_matrix_groups'),
                      ('thread_cpu_totals','per_worker_thread_cpu_totals'),('timing','timing_comparison')]:
        require(scope[key] == collection[name], 'independent raw replay differs: ' + name)
    print('raw replay: 59 Job records, 97 compiler commands, 144 graph vectors and four model cases match', flush=True)

    local_commands = read(local / 'commands.json')
    sidecars = pre['local_sidecars']
    require(len(local_commands) == len(sidecars) == 470, 'complete local command proof required')
    canonical = lambda row: json.dumps(row, sort_keys=True, separators=(',', ':'))
    require(Counter(map(canonical, local_commands)) ==
            Counter(canonical(read(Path(name))) for name in sidecars), 'local ledger/sidecar mismatch')
    require(all(row.get('completed') is True and row.get('leader_reaped') is True and
                row.get('group_gone') is True and type(row.get('returncode')) is int and row['returncode'] == 0
                for row in local_commands), 'local command cleanup failure')
    prep = read(prep_root / 'qualification.json')
    prep_commands = prep['commands']
    require(prep['completed'] is True and prep['compiled_sdk_or_model'] is False and
            len(prep_commands) == 11 and len(prep['materialized_files']) == 3425 and
            prep['materialized_files'] == audit['files'], 'source-only preparation scope changed')
    prep_sidecars = list((prep_root / 'evidence').glob('*.process.json'))
    require(len(prep_sidecars) == 11 and
            Counter(canonical({key: value for key, value in row.items() if key != 'name'}) for row in prep_commands) ==
            Counter(canonical(read(path)) for path in prep_sidecars), 'source-only command sidecar mismatch')
    require(all(row['completed'] is True and row['leader_reaped'] is True and row['returncode'] == 0
                for row in prep_commands), 'source-only command completion failure')

    zero_fields, worker_observations = Counter(), []
    for index in (1, 2):
        saved = read(evidence / ('case-' + str(index) + '.matrix-profile.json'))
        totals = {}
        for graph in saved['graphs']:
            for worker in graph['workers']:
                slot = totals.setdefault(str(worker['thread']), {'graph_wall_us': 0, 'graph_cycles': 0,
                    'kernel_100ns': 0, 'user_100ns': 0, 'operation_kernel_cycles': 0,
                    'operation_barrier_cycles': 0, 'final_wait_cycles': 0, 'matrix_kernel_cycles': 0,
                    'matrix_second_cycles': 0, 'matrix_conversion_cycles': 0, 'matrix_inner_wait_cycles': 0})
                slot['graph_wall_us'] += worker['elapsed_us']
                for name in ('kernel_100ns', 'user_100ns', 'final_wait_cycles'):
                    slot[name] += worker['thread_cpu'][name]
                slot['graph_cycles'] += worker['thread_cpu']['cycles']
                for category, records in [('graph', [worker['thread_cpu']]),
                                          ('operation', worker['operations']), ('matrix', worker['matrices'])]:
                    for record in records:
                        for key, value in record.items():
                            if key == 'cycles' or key.endswith(('_cycles', '_100ns')):
                                zero_fields[category + '.' + key] += value == 0
                for op in worker['operations']:
                    slot['operation_kernel_cycles'] += op['kernel_cycles']
                    slot['operation_barrier_cycles'] += op['barrier_cycles']
                for matrix in worker['matrices']:
                    for key in ('kernel_cycles', 'second_cycles', 'conversion_cycles', 'inner_wait_cycles'):
                        slot['matrix_' + key] += matrix[key]
        worker_observations.append({'index': index, 'per_worker': totals})

    result = {'schema_version': 1, 'source_commit': q['source_commit'], 'run_id': collection['run_id'],
        'job_id': collection['job_id'], 'local_bundle_admission': pre['files'][str(local / 'admission.json')],
        'pre_execution_protocol': first_protocol, 'native_protocol': pre['native_protocol'],
        'result_collection': pre['result_collection'], 'source_preparation_collection': pre['source_preparation_collection'],
        'current_frozen_tools_verified': 135, 'current_frozen_repository_references_verified': 2880,
        'retained_frozen_tools_verified': 135, 'saved_windows_job_commands_verified': 59,
        'compiler_metadata_commands_verified': 97, 'model_free_processes': 12, 'model_free_vectors': 144,
        'model_cases': scope['cases'], 'full_sequence_bit_exact': True, 'complete_compact_profiles_replayed': 4,
        'enabled_matrix_records_each': 43344, 'timing_comparison': scope['timing'],
        'per_worker_thread_cpu_totals': scope['thread_cpu_totals'], 'per_worker_operations': scope['operations'],
        'per_worker_matrix_groups': scope['matrix_groups'], 'host': q['host'],
        'local_command_sidecars_verified': 470, 'source_only_command_sidecars_verified': 11,
        'zero_counter_fields': dict(zero_fields), 'worker_observation_totals': worker_observations,
        'unchanged_source_ast': {'supervision': supervision_ast, 'command_order': command_order_ast, 'compiler': compiler_ast,
            'graph_controls_except_unavailable_exe_stat': control_ast, 'models_counters_timing': model_ast},
        'excluded_remote_only_assertion': ast.unparse(excluded[0]),
        'new_compiler_sdk_model_calls': 0, 'production_or_release_qualified': False,
        'limits': [
            'Saved raw-output/counter/timing replay only. No new SDK, model, public runtime or release qualification.',
            'Original full native ZIP, SDK binaries, controller executable and pristine archive were admitted remotely and are unavailable for local re-verification. Compiler binary aliases/source materialization remain remote observations; local compiler metadata checks do not rehash those binaries or the full compiled tree.',
            'Remote native ZIP-download owned-process sidecar stayed in the remote cache and was not returned. Native59WindowsJob, returned source11 and local470 records are separately retained and checked.',
            'Raw thread cycles and CPU100ns units stay separate from wall microseconds. No frequency conversion, exact wait, pure arithmetic or causal scheduling inference. Preserve genuine zeros; timing interpretation requires both original observer bounds.']}
    # Recheck the original first-bound identities, never a newly captured set.
    require(descriptor(args.protocol) == first_protocol, 'pre-execution protocol changed during replay')
    for name, wanted in pre['files'].items():
        require(descriptor(Path(name)) == wanted, 'first-bound input changed during replay: ' + name)
    verify_files(tools, p['tools'])
    verify_files(repo, p['repository_files'])
    verify_files(delivered / 'sources', p['tools'])
    verify_collection(delivered, pre['result_collection'])
    verify_collection(prep_root, pre['source_preparation_collection'])
    verify_files(local / 'delivery', admission['delivery_members'])
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ('saved_windows_job_commands_verified',
        'compiler_metadata_commands_verified','model_free_vectors','full_sequence_bit_exact',
        'timing_comparison','local_command_sidecars_verified','source_only_command_sidecars_verified',
        'new_compiler_sdk_model_calls','production_or_release_qualified')}))


if __name__ == '__main__':
    main()
