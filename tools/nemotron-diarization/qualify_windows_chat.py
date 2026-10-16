"""Actual Windows CPU chat/diarization qualification after same-source offline gates.

The large immutable chat model is a separately verified read-only mount. Runtime
ZIP byte bounds and original package contents remain unchanged.
"""
import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess

from artifacts import acquire
from linux_chat_audit import audit_results
from qualify_windows_package import Qualification as OfflineQualification
from windows_central_setup import MINIMUM
from windows_archive import byte_inventory, digest, write_archive
from windows_chat_inputs import model_inventory
from windows_pe import inspect_pe
from windows_runtime_audit import loaded_modules


class Qualification(OfflineQualification):
    def __init__(self, args):
        super().__init__(args)
        self.receipt['chat_diarization_gates_passed'] = False
        self.receipt['scope'] = 'Windows Server 2022 AMD64 CPU offline native/public parity and lifecycle, followed by isolated chat and both chat/diarization load orders in fresh process containers.'
        self.receipt['limits'][0] = 'No Windows desktop/physical-machine, Bark/Laya coexistence or universal namespace guarantee.'
        self.receipt['limits'].append('Chat helper uses an additional exact VS OpenMP test dependency; no redistribution grant or universal namespace qualification.')

    def prepare(self):
        super().prepare()
        self.chat_lock = json.loads((self.tools / 'linux-chat.lock.json').read_text())
        self.chat = self.output / 'locked chat fixture ü'
        self.chat.mkdir()
        for item in self.chat_lock['artifacts']:
            acquire(item, self.chat)
        self.chat_inventory = model_inventory(self.chat, self.chat_lock)
        submodule = self.repo / 'vendor/llama-cpp-sys-2/llama.cpp'
        self.run('initialize-chat-source', ['git', 'submodule', 'update', '--init', '--depth', '1', 'vendor/llama-cpp-sys-2/llama.cpp'])
        expected = subprocess.check_output(['git', 'ls-tree', 'HEAD', 'vendor/llama-cpp-sys-2/llama.cpp'], cwd=self.repo, text=True).split()[2]
        actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=submodule, text=True).strip()
        if actual != expected or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=submodule, text=True):
            raise ValueError('chat source differs from repository gitlink')
        self.run('build-chat-helper', ['cargo', 'build', '--locked', '--release', '--no-default-features',
                 '--features', 'backend-llamacpp,backend-diar-nemo', '--example', 'nemotron_chat_coexistence'])
        metadata = json.loads(self.run('chat-cargo-metadata', ['cargo', 'metadata', '--locked', '--no-deps', '--format-version', '1']))
        helper = Path(metadata['target_directory']) / 'release/examples/nemotron_chat_coexistence.exe'
        self.chat_helper = 'chat-coexistence.exe'
        destination = self.work / 'bin' / self.chat_helper
        shutil.copy2(helper, destination)
        pe = inspect_pe(destination)
        if (pe['machine'] != 'AMD64' or pe['is_dll'] or pe['bound_imports_present']
                or 'vcomp140.dll' not in {name.lower() for name in pe['imports']}):
            raise ValueError('normal CPU chat helper must import MSVC OpenMP')
        self.helpers[self.chat_helper] = {key: pe[key] for key in ['bytes', 'sha256']}
        (self.output / 'chat-helper-pe.json').write_text(json.dumps(pe, indent=2) + '\n', encoding='utf-8')
        if self.central_crt is None:
            self.run('chat-openmp-provenance', ['pwsh', '-NoLogo', '-NoProfile', '-NonInteractive', '-File',
                     self.tools / 'windows_chat_openmp.ps1', '-NativeProvenance', self.args.native_evidence / 'windows-provenance.json',
                     '-Output', self.output / 'chat-openmp-provenance.json'])
            provenance = json.loads((self.output / 'chat-openmp-provenance.json').read_text(encoding='utf-8-sig'))
            origin = Path(provenance['path']).resolve(strict=True)
            if (provenance['signature_status'] != 'Valid' or 'O=Microsoft Corporation,' not in provenance['signer_subject']
                    or origin.name.lower() != 'vcomp140.dll'
                    or [part.lower() for part in origin.parts[-3:-1]] != ['x64', 'microsoft.vc143.openmp']):
                raise ValueError('actual signed OpenMP source differs from expected VS redist tree')
            self.openmp = {key: provenance[key] for key in ['bytes', 'sha256']}
            if {'bytes': origin.stat().st_size, 'sha256': digest(origin)} != self.openmp:
                raise ValueError('OpenMP source bytes differ from connected-host observation')
            shutil.copy2(origin, self.work / 'bin/vcomp140.dll')
            openmp_pe = inspect_pe(self.work / 'bin/vcomp140.dll')
            if openmp_pe['machine'] != 'AMD64' or not openmp_pe['is_dll'] or openmp_pe['bound_imports_present']:
                raise ValueError('unsupported OpenMP test image')
        else:
            row, = [row for row in self.central_state['files'] if row['name'] == 'vcomp140.dll']
            self.openmp = self.central_crt['vcomp140.dll']
            provenance = {**row, 'signature_status': row['signature']['status'],
                          'signer_subject': row['signature']['signer']['subject'],
                          'source': 'Exact central System32 observation from the qualified publisher-installer image.',
                          'central_setup_receipt_sha256': digest(self.args.central_setup / 'receipt.json')}
            (self.output / 'chat-openmp-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
            minimum = json.loads((self.repo / MINIMUM).read_text())
            openmp_pe, = [row for row in minimum['selected_dlls'] if row['runtime_name'] == 'vcomp140.dll']
            openmp_pe = {**openmp_pe, 'scope': 'Pinned CAB static PE metadata for bytes matching the actual installed System32 observation; no app-local OpenMP payload.'}
            self.receipt['limits'][-1] = 'Chat uses the exact central System32 OpenMP bytes supplied by the qualified publisher-installer image; no app-local CRT/OpenMP copies or universal namespace claim.'
        (self.output / 'chat-openmp-pe.json').write_text(json.dumps(openmp_pe, indent=2) + '\n', encoding='utf-8')
        archive = write_archive(self.work / 'bin', self.output, 'development-windows-chat-helpers')
        self.fingerprints = {'executable': digest(destination),
                             'chat_model': self.chat_inventory['Qwen3-0.6B-Q4_K_M.gguf']['sha256'],
                             'diarization_model': digest(self.work / 'assets/Nemotron-3-Diarization.q8_0.gguf'),
                             'library': digest(self.runtime / 'lib' / self.package['entry_library']),
                             'wav': digest(self.work / 'assets/ami_en2002d_2132.wav')}
        frozen = {'source_commit': self.source, 'chat_llama_revision': actual,
                  'chat_features': ['backend-llamacpp', 'backend-diar-nemo'],
                  'fingerprints': self.fingerprints, 'chat_fixture_files': self.chat_inventory,
                  'files': byte_inventory(self.work), 'chat_helper_archive': archive,
                  'openmp_provenance_sha256': digest(self.output / 'chat-openmp-provenance.json'),
                  'example_sha256': digest(self.repo / 'examples/nemotron_chat_coexistence.rs'),
                  'tools': {name: digest(self.tools / name) for name in ['qualify_windows_chat.py', 'windows_chat_inputs.py',
                            'windows_chat_openmp.ps1', 'windows_runtime_audit.py', 'qualify_windows_package.py',
                            'linux_chat_audit.py', 'linux-chat.lock.json']}}
        (self.output / 'chat-frozen-inputs.json').write_text(json.dumps(frozen, indent=2) + '\n', encoding='utf-8')
        self.receipt['chat_frozen_inputs_sha256'] = digest(self.output / 'chat-frozen-inputs.json')

    def audit_modules(self, observation, executable):
        if executable != self.chat_helper:
            return super().audit_modules(observation, executable)
        return loaded_modules(observation, self.package['libraries'], executable, self.helpers[executable],
                              private_required=self.current_order != 'isolated', openmp=self.openmp, central_crt=self.central_crt)

    def exercise(self):
        super().exercise()
        collected = self.output / 'chat-results'
        collected.mkdir()
        stages = {}
        for order in ['isolated', 'chat-first', 'diar-first']:
            self.current_order = order
            results = self.container(order, self.chat_helper,
                [order, 'C:\\chat\\Qwen3-0.6B-Q4_K_M.gguf', 'C:\\inputs\\assets',
                 'C:\\inputs\\relocated runtime ü\\lib\\' + self.package['entry_library'],
                 'C:\\inputs\\assets\\ami_en2002d_2132.wav', 'C:\\results\\' + order],
                readonly_mounts=[(self.chat, 'C:\\chat')])
            output = results / order
            shutil.copytree(output, collected / order)
            stages[order] = {}
            for stage in (['isolated'] if order == 'isolated' else ['both-loaded', 'reloaded']):
                modules = json.loads((output / (stage + '.modules.json')).read_text(encoding='utf-8-sig'))
                stages[order][stage] = self.audit_modules({'stable_snapshots': 1, 'modules': modules}, self.chat_helper)
        native_path = self.output / 'native/results/native.json'
        native = json.loads(native_path.read_text(encoding='utf-8-sig'))
        comparisons = audit_results(collected, native, native_path.with_suffix('.json.f32').read_bytes(), self.fingerprints)
        (self.output / 'chat-comparison.json').write_text(json.dumps(comparisons, indent=2) + '\n', encoding='utf-8')
        (self.output / 'chat-module-stages.json').write_text(json.dumps(stages, indent=2) + '\n', encoding='utf-8')
        if model_inventory(self.chat, self.chat_lock) != self.chat_inventory or byte_inventory(self.work) != self.with_reference_inventory:
            raise ValueError('Windows coexistence inputs changed during execution')
        self.receipt.update(chat_diarization_gates_passed=True, chat_comparisons=comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['sdk-prefix', 'native-evidence', 'package-evidence', 'prerequisites', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--central-setup', type=Path, help='Successful same-source central prerequisite setup evidence.')
    args = parser.parse_args()
    if platform.system() != 'Windows':
        parser.error('requires actual Windows Docker host')
    args.output.mkdir(parents=True, exist_ok=False)
    Qualification(args).main()


if __name__ == '__main__':
    main()
