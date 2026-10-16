"""Create an isolated local valid-Mel source clone; no network, build or model execution."""
import argparse
import json
from pathlib import Path
import subprocess

from pooled_runtime import PROFILE, POOLED_PROFILE, load_profile, verify_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['upstream', 'destination', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--runtime-profile', choices=[PROFILE, POOLED_PROFILE], default=PROFILE)
    args = parser.parse_args()
    destination, output = args.destination.resolve(), args.output.resolve()
    if destination.is_relative_to(output) or output.is_relative_to(destination):
        parser.error('source and preparation evidence paths must be disjoint')
    if args.destination.exists() or args.output.exists():
        parser.error('refusing existing source or preparation evidence')
    upstream, destination = args.upstream.resolve(), args.destination.resolve()
    lock_path, lock = load_profile(args.runtime_profile)
    runtime = lock['runtime']
    for root, revision in [(upstream, runtime['revision']), (upstream / 'llama.cpp', runtime['llama_revision'])]:
        if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != revision:
            parser.error('local source revision differs')
    args.output.mkdir(parents=True)
    env = {'PATH': '/usr/bin:/bin', 'LC_ALL': 'C', 'GIT_LFS_SKIP_SMUDGE': '1'}
    commands = []
    def run(argv, cwd):
        name = str(len(commands))
        record = {'argv': list(map(str, argv)), 'cwd': str(cwd), 'environment': env}
        commands.append(record)
        (args.output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        with (args.output / (name + '.stdout.log')).open('xb') as stdout, (args.output / (name + '.stderr.log')).open('xb') as stderr:
            result = subprocess.run(record['argv'], cwd=cwd, env=env, stdout=stdout, stderr=stderr)
        record['returncode'] = result.returncode
        (args.output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        result.check_returncode()
    run(['git', 'clone', '--no-hardlinks', '--no-checkout', '--', upstream, destination], upstream.parent)
    run(['git', 'checkout', '--detach', runtime['revision']], destination)
    run(['git', 'clone', '--no-hardlinks', '--no-checkout', '--', upstream / 'llama.cpp', destination / 'llama.cpp'], destination)
    run(['git', 'checkout', '--detach', runtime['llama_revision']], destination / 'llama.cpp')
    for item in runtime['compatibility_patches']:
        patch = lock_path.parent / item['path']
        run(['git', 'apply', '--check', patch], destination)
        run(['git', 'apply', patch], destination)
    verify_source(destination, lock)
    (args.output / 'prepared.json').write_text(json.dumps({'profile': lock['profile'],
        'upstream': str(upstream), 'destination': str(destination), 'commands': commands,
        'base_revision': runtime['revision'], 'llama_revision': runtime['llama_revision'],
        'compatibility_patches': runtime['compatibility_patches'], 'network_or_model_execution': False}, indent=2) + '\n')
    print(json.dumps({'prepared': True, 'profile': lock['profile'], 'network_or_model_execution': False}))


if __name__ == '__main__':
    main()
