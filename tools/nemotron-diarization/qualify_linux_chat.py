"""Qualify actual Gen2 CPU chat plus the delivered diarizer in offline containers.

Requires a passing same-source Linux container run first. Chat's existing
OpenMP runtime is a separate test dependency; this is not a chat/Bark/Laya
release package or general ELF namespace guarantee.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from artifacts import acquire, matches
from platform_valid_mel_native import compare_complete_60s as compare
from runtime_profiles import load_profile, profile_provenance, validate_provenance
from evidence_commands import CommandRunner
from linux_chat_audit import audit_results, ggml_bindings
from package_linux import (PLATFORM_LIBRARIES, check_source_bundle, digest, inspect_elf,
                           owner, resolved_dependencies, verify)
from qualify_linux_package import RUNTIME_PATH, container_command, loader_paths
from sdk_archive import inventory

CHAT_PATH = '/work/chat'


class Qualification(CommandRunner):
    def __init__(self, args):
        self.args = args
        self.tools = Path(__file__).resolve().parent
        super().__init__(args.output.resolve(), self.tools.parents[1])
        self.work, self.results = self.output / 'inputs', self.output / 'results'
        lock_path, lock = load_profile()
        self.receipt = {**profile_provenance(lock_path, lock), 'schema_version': 2, 'release_qualified': False,
                        'chat_diarization_gates_passed': False,
                        'scope': 'Actual CPU Gen2 chat/OpenMP plus packaged diarization, both load orders, repeated/reloaded/concurrent calls and GGML bindings in fresh offline Ubuntu 24.04 containers.',
                        'limits': ['No Bark/Laya coexistence, other Linux baseline, GPU or physical-machine qualification.',
                                   'No independent chat runtime distribution, held-out quality or uncontended performance claim.',
                                   'Binding observations cover the recorded binaries/calls, not a general ELF namespace isolation guarantee.']}

    def prepare(self):
        base = self.args.container_evidence.resolve()
        prior = json.loads((base / 'receipt.json').read_text())
        validate_provenance(prior)
        frozen = json.loads((base / 'frozen-inputs.json').read_text())
        current = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.repo, text=True).strip()
        if prior.get('offline_container_gates_passed') is not True or prior['source_commit'] != current or frozen['source_commit'] != current:
            raise ValueError('a passing same-source offline-container prerequisite is required')
        if digest(base / 'frozen-inputs.json') != prior['frozen_inputs_sha256'] or inventory(base / 'inputs') != frozen['inputs']:
            raise ValueError('offline prerequisite input identity changed')
        self.image = prior['image_id']
        if self.image != frozen['image']['Id'] or prior['image_repo_digests'] != frozen['image']['RepoDigests']:
            raise ValueError('prerequisite image identity differs')
        shutil.copytree(base / 'inputs', self.work, symlinks=True)
        self.results.mkdir()
        self.runtime = self.work / 'relocated runtime ü'
        self.package = verify(self.runtime)
        if digest(self.runtime / 'package.json') != prior['package_manifest_sha256']:
            raise ValueError('coexistence package differs from offline-qualified package')
        self.reference = json.loads((base / 'results/native.json').read_text())
        self.raw = (base / 'results/native.json.f32').read_bytes()
        public = json.loads((base / 'results/public.json').read_text())
        if compare(public, self.reference, self.raw, (base / 'results/public.f32').read_bytes()) != prior['native_public_comparison'] or public['input_pcm_f32le_sha256'] != self.reference['input_pcm_f32le_sha256']:
            raise ValueError('offline prerequisite output comparison differs')
        shutil.copy2(base / 'results/native.json', self.output / 'independent-native.json')
        shutil.copy2(base / 'results/native.json.f32', self.output / 'independent-native.f32')
        submodule = self.repo / 'vendor/llama-cpp-sys-2/llama.cpp'
        self.run('initialize-chat-source', ['git', 'submodule', 'update', '--init', '--depth', '1', 'vendor/llama-cpp-sys-2/llama.cpp'])
        expected = subprocess.check_output(['git', 'ls-tree', 'HEAD', 'vendor/llama-cpp-sys-2/llama.cpp'], cwd=self.repo, text=True).split()[2]
        actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=submodule, text=True).strip()
        if actual != expected or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=submodule, text=True):
            raise ValueError('chat native source differs from the repository gitlink')
        self.run('build-chat-diagnostic', ['cargo', 'build', '--locked', '--release', '--no-default-features',
                 '--features', 'backend-llamacpp,backend-diar-nemo', '--example', 'nemotron_chat_coexistence'])
        metadata = json.loads(self.run('cargo-metadata', ['cargo', 'metadata', '--locked', '--no-deps', '--format-version', '1']))
        source = Path(metadata['target_directory']) / 'release/examples/nemotron_chat_coexistence'
        executable = self.work / 'bin/chat-coexistence'
        shutil.copy2(source, executable)
        original = inspect_elf(executable, require_soname=False)
        if 'libgomp.so.1' not in original['needed']:
            raise ValueError('regular CPU chat backend did not link its expected OpenMP runtime')
        gomp = resolved_dependencies(executable, ['libgomp.so.1'])['libgomp.so.1']
        component = owner(gomp)
        if component['binary:Package'].split(':')[0] != 'libgomp1':
            raise ValueError('chat OpenMP runtime has an unexpected package owner')
        gcc = self.package['components']['libgcc-s1']['source_bundle']
        if (component['source:Package'], component['source:Version']) != (gcc['source_package'], gcc['source_version']):
            raise ValueError('chat OpenMP runtime requires a separately acquired exact source bundle')
        if check_source_bundle(self.runtime / 'sources' / gcc['source_package'], gcc['source_package'], gcc['source_version']) != gcc:
            raise ValueError('shared GCC source bundle changed')
        chat = self.work / 'chat'
        (chat / 'lib').mkdir(parents=True)
        shutil.copy2(gomp, chat / 'lib/libgomp.so.1')
        gomp_elf = inspect_elf(chat / 'lib/libgomp.so.1')
        if gomp_elf['soname'] != 'libgomp.so.1' or gomp_elf['rpath'] or gomp_elf['runpath'] or any(name not in PLATFORM_LIBRARIES | {'libgcc_s.so.1', 'libstdc++.so.6'} for name in gomp_elf['needed']):
            raise ValueError('OpenMP test runtime has an unqualified dependency')
        copyright = Path('/usr/share/doc/libgomp1/copyright').resolve(strict=True)
        copyright_owner = owner(copyright)
        if (copyright_owner['source:Package'], copyright_owner['source:Version']) != (gcc['source_package'], gcc['source_version']):
            raise ValueError('OpenMP copyright differs from binary source identity')
        (chat / 'notices').mkdir()
        shutil.copy2(copyright, chat / 'notices/libgomp1-copyright')
        shutil.copytree(self.runtime / 'notices/common-licenses', chat / 'notices/common-licenses')
        lock = json.loads((self.tools / 'linux-chat.lock.json').read_text())
        for item in lock['artifacts']:
            acquire(item, chat)
            if not matches(chat / item['name'], item):
                raise ValueError('chat fixture identity differs: ' + item['name'])
        rpath = '$ORIGIN/../relocated runtime ü/lib:$ORIGIN/../chat/lib'
        self.run('chat-private-rpath', ['patchelf', '--force-rpath', '--set-rpath', rpath, executable])
        elf = inspect_elf(executable, require_soname=False)
        if elf['rpath'] != [rpath] or elf['runpath'] or any(
                name not in PLATFORM_LIBRARIES | {'libstdc++.so.6', 'libgcc_s.so.1', 'libgomp.so.1'} for name in elf['needed']):
            raise ValueError('chat diagnostic helper has an unqualified dependency/search path')
        self.run('chat-dynamic-symbols', ['nm', '--dynamic', '--defined-only', executable])
        self.run('chat-static-ggml-symbols', ['nm', '--defined-only', '--extern-only', executable])
        if 'ggml_new_graph_custom' not in (self.output / 'chat-static-ggml-symbols.stdout.log').read_text():
            raise ValueError('chat helper does not contain the actual statically linked GGML implementation')
        for name, command in [('rust-version', ['rustc', '--version', '--verbose']),
                              ('cargo-version', ['cargo', '--version']), ('docker-version', ['docker', 'version'])]:
            self.run(name, command)
        library = self.runtime / 'lib' / self.package['entry_library']
        self.fingerprints = {'executable': digest(executable), 'chat_model': digest(chat / 'Qwen3-0.6B-Q4_K_M.gguf'),
                             'diarization_model': digest(self.work / 'assets/Nemotron-3-Diarization.q8_0.gguf'),
                             'library': digest(library), 'wav': digest(self.work / 'assets/ami_en2002d_2132.wav')}
        inputs = {'source_commit': current, 'offline_receipt_sha256': digest(base / 'receipt.json'),
                  'offline_frozen_inputs_sha256': digest(base / 'frozen-inputs.json'),
                  'image_id': self.image, 'image_repo_digests': prior['image_repo_digests'],
                  'package_manifest_sha256': digest(self.runtime / 'package.json'),
                  'chat_llama_revision': actual, 'chat_features': ['backend-llamacpp', 'backend-diar-nemo'],
                  'chat_protocol': {'prompt': 'Reply with the single word hello.', 'max_tokens': 8,
                                    'temperature': 0., 'seed': 0, 'gpu_layers': 0, 'threads': 2, 'threads_batch': 2, 'ctx_size': 1024,
                                    'load_orders': ['isolated', 'chat-first', 'diar-first']},
                  'openmp_test_runtime': {'package': component, 'source': str(gomp), 'sha256': digest(gomp),
                                          'copyright_owner': copyright_owner, 'shared_gcc_source_bundle': gcc, 'elf': gomp_elf},
                  'original_helper_sha256': digest(source), 'helper_elf': elf,
                  'fingerprints': self.fingerprints, 'files': inventory(self.work),
                  'native_reference_sha256': digest(self.output / 'independent-native.json'),
                  'native_probabilities_sha256': digest(self.output / 'independent-native.f32'),
                  'tools': {name: digest(self.tools / name) for name in ['qualify_linux_chat.py', 'linux_chat_audit.py',
                            'qualify_linux_package.py', 'evidence_commands.py', 'package_linux.py', 'sdk_archive.py',
                            'compare_stream_valid_mel.py', 'platform_valid_mel_native.py', 'platform_valid_mel_sources.py', 'platform_native.py', 'platform_sdk_receipt.py', 'runtime_profiles.py', 'build_valid_mel_candidate.py', 'qualification_storage.py', 'runtime-valid-mel.lock.json', 'runtime.lock.json', 'valid-mel-platform-native.protocol.json', 'linux-chat.lock.json']},
                  'example_sha256': digest(self.repo / 'examples/nemotron_chat_coexistence.rs')}
        (self.output / 'frozen-inputs.json').write_text(json.dumps(inputs, indent=2) + '\n')
        self.receipt.update(source_commit=current, frozen_inputs_sha256=digest(self.output / 'frozen-inputs.json'),
                            image_id=self.image, image_repo_digests=prior['image_repo_digests'])

    def exercise(self):
        mounts = [(self.work / 'bin', '/work/bin'), (self.work / 'assets', '/work/assets'),
                  (self.work / 'chat', CHAT_PATH), (self.runtime, RUNTIME_PATH)]
        library = RUNTIME_PATH + '/lib/' + self.package['entry_library']
        for order in ['isolated', 'chat-first', 'diar-first']:
            command = ['/work/bin/chat-coexistence', order, CHAT_PATH + '/Qwen3-0.6B-Q4_K_M.gguf',
                       '/work/assets', library, '/work/assets/ami_en2002d_2132.wav', '/results/' + order]
            self.run(order, container_command(self.image, mounts, self.results, command,
                     {'LD_DEBUG': 'libs,bindings', 'LD_BIND_NOW': '1', 'OMP_NUM_THREADS': '2'}))
        results = audit_results(self.results, self.reference, self.raw, self.fingerprints, comparator=compare)
        loaded, bindings = {}, {}
        gcc = {RUNTIME_PATH + '/lib/libstdc++.so.6', RUNTIME_PATH + '/lib/libgcc_s.so.1'}
        for order in ['isolated', 'chat-first', 'diar-first']:
            trace = (self.output / (order + '.stderr.log')).read_text()
            expected = [] if order == 'isolated' else self.package['libraries']
            extra = {CHAT_PATH + '/lib/libgomp.so.1'} | (gcc if order == 'isolated' else set())
            loaded[order] = loader_paths(trace, expected, extra)
            if order != 'isolated':
                bindings[order] = ggml_bindings(trace, self.package['libraries'])
        (self.output / 'loaded-libraries.json').write_text(json.dumps(loaded, indent=2) + '\n')
        (self.output / 'ggml-bindings.json').write_text(json.dumps(bindings, indent=2) + '\n')
        (self.output / 'comparison.json').write_text(json.dumps(results, indent=2) + '\n')
        frozen = json.loads((self.output / 'frozen-inputs.json').read_text())
        if inventory(self.work) != frozen['files'] or verify(self.runtime) != self.package:
            raise ValueError('coexistence input files changed during execution')
        self.receipt.update(chat_diarization_gates_passed=True, comparisons=results,
                            native_ggml_binding_counts={name: len(items) for name, items in bindings.items()})

    def main(self):
        try:
            self.prepare()
            self.exercise()
        except Exception as error:
            self.receipt['error'] = str(error)
            raise
        finally:
            (self.output / 'receipt.json').write_text(json.dumps(self.receipt, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container-evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    Qualification(args).main()


if __name__ == '__main__':
    main()
