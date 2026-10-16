"""Retain commands, separate output streams and exit statuses for qualification."""
import json
from pathlib import Path
import subprocess
import time


class CommandRunner:
    def __init__(self, output, repo):
        self.output, self.repo = Path(output), Path(repo)
        self.commands = []

    def run(self, name, command, expected_failure=False):
        args = list(map(str, command))
        record = {'name': name, 'argv': args}
        self.commands.append(record)
        (self.output / 'commands.json').write_text(json.dumps(self.commands, indent=2) + '\n')
        print('START ' + name, flush=True)
        started = time.monotonic()
        with (self.output / (name + '.stdout.log')).open('x') as out, (self.output / (name + '.stderr.log')).open('x') as err:
            result = subprocess.run(args, cwd=self.repo, stdout=out, stderr=err, timeout=3600)
        record.update(returncode=result.returncode, elapsed_seconds=time.monotonic() - started)
        (self.output / 'commands.json').write_text(json.dumps(self.commands, indent=2) + '\n')
        if (result.returncode == 0) == expected_failure:
            raise ValueError(f'{name}: unexpected exit status {result.returncode}')
        print('PASS ' + name, flush=True)
        return (self.output / (name + '.stdout.log')).read_text()

