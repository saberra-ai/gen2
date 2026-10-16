"""Bind the diagnostic to its loader tree and retain failed native commands."""
import json
from pathlib import Path
import subprocess


def require_linked_sdk(sdk, frozen_path, rpaths):
    actual, expected = Path(sdk).resolve(strict=True), Path(frozen_path).resolve(strict=True)
    if actual != expected or rpaths != [str(expected / 'lib')]:
        raise ValueError('SDK differs from the native probe loader directory')


def run_probe(argv, output, name, environment):
    record = {'argv': list(map(str, argv)), 'environment': environment}
    command = output / (name + '.command.json')
    command.write_text(json.dumps(record, indent=2) + '\n')
    # Open streams first so even failed startup/termination retains diagnostics.
    with (output / (name + '.stdout.log')).open('x') as stdout, (output / (name + '.stderr.log')).open('x') as stderr:
        try:
            completed = subprocess.run(record['argv'], stdout=stdout, stderr=stderr, env=environment, check=False)
        except OSError as error:
            record.update(returncode=None, startup_error=str(error))
            command.write_text(json.dumps(record, indent=2) + '\n')
            raise
    record['returncode'] = completed.returncode
    command.write_text(json.dumps(record, indent=2) + '\n')
    if completed.returncode:
        raise ValueError('native frontend probe failed with exit ' + str(completed.returncode))
    return (output / (name + '.stdout.log')).read_text()
