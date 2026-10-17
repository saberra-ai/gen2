"""Exact controller-only derivation; preserve the original inference experiment."""


def derive_runner(original):
    replacements = [
        (b'from windows_job import run_in_job', b'from windows_job_v2 import run_in_job'),
        (b"'windows-compiler-diagnostic.protocol.json'", b"'windows-compiler-diagnostic-v2.protocol.json'"),
        (b"'Windows kill-on-close Job with gated worker'",
         b"'Windows kill-on-close Job v2 with scoped snapshot and owned-background cleanup'"),
        (b"            record['process_tree_reaped'] = result.process_tree_reaped\n",
         b"            record['process_tree_reaped'] = result.process_tree_reaped\n"
         b"            record['job_observation'] = result.job_observation\n"),
        (b"            record['process_tree_reaped'] = getattr(error, 'process_tree_reaped', record['process_tree_reaped'])\n",
         b"            record['process_tree_reaped'] = getattr(error, 'process_tree_reaped', record['process_tree_reaped'])\n"
         b"            record['job_observation'] = getattr(error, 'job_observation', record.get('job_observation'))\n")]
    result = original
    for old, new in replacements:
        if result.count(old) != 1:
            raise ValueError('one exact original controller site required')
        result = result.replace(old, new)
    return result
