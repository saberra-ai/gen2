"""Replay actual compiler baselines and reject any change beyond contraction."""
from pathlib import Path, PureWindowsPath
import unittest
from unittest.mock import patch

from build_valid_mel_candidate import descriptor as real_descriptor
from platform_native import powershell_literal
import windows_compiler_fp_metrics as metrics
from windows_compiler_metrics import build_command as parent_build_command
from windows_compiler_fp_sources import derive_runner


class CompilerContractionTest(unittest.TestCase):
    def fixture(self, kind):
        cache, database, compiler = metrics.baseline(kind)
        directory = Path(__file__).resolve().parents[2] / metrics.REFERENCE / ('evidence-' + kind)
        files = {language: (directory / ('compiler-' + language + '.cmake')).read_text() for language in ('C', 'CXX')}
        if kind == 'clang-cl':
            for language in ('C', 'CXX'):
                field = 'CMAKE_' + language + '_FLAGS'
                cache = cache.replace(field + ':STRING=', field + ':STRING=' + metrics.FLAG + ' ')
                cache += '\n' + field + '_INIT:STRING=' + metrics.FLAG + '\n'
            database = [{**row, 'command': row['command'].replace(' /DWIN32 ', ' ' + metrics.FLAG + ' /DWIN32 ')} for row in database]
        return cache, database, files, compiler

    def admit(self, kind, cache, database, files, compiler):
        def descriptor(path):
            if str(path) == compiler['linker']:
                return compiler['linker_binary']
            return real_descriptor(path)
        with patch.object(metrics, 'descriptor', side_effect=descriptor):
            return metrics.compiler_evidence(cache, database, files, compiler['compiler'], kind,
                                             compiler['compiler_path_aliases'])

    def test_controller_changes_only_adapter_and_protocol(self):
        tools = Path(__file__).resolve().parent
        raw = (tools / 'windows_compiler_diagnostic_v2.py').read_bytes()
        self.assertEqual(derive_runner(raw), (tools / 'windows_compiler_fp_diagnostic.py').read_bytes())
        for invalid in [raw + raw, raw.replace(b'from windows_compiler_metrics import (', b'unknown')]:
            with self.assertRaises(ValueError):
                derive_runner(invalid)

    def test_build_adds_initializers_and_requires_exact_compiler_binaries(self):
        compilers = {kind: metrics.baseline(kind)[2] for kind in ('msvc', 'clang-cl')}
        msvc, clang = (compilers[kind]['compiler'] for kind in ('msvc', 'clang-cl'))
        source = "& 'native/build.ps1' -Backend cpu -Profile asr -Jobs 2 -CMakeArgs @(" + ','.join(
            powershell_literal('-DCMAKE_' + language + '_COMPILER=' + msvc) for language in ('C', 'CXX')) + ')'
        command = ['pwsh', '-NoProfile', '-NonInteractive', '-Command', source]
        identities = {record['compiler']: record['compiler_binary'] for record in compilers.values()}
        def descriptor(path):
            return identities[str(path)] if str(path) in identities else real_descriptor(path)
        with patch.object(metrics, 'descriptor', side_effect=descriptor):
            self.assertEqual(metrics.build_command(command, msvc, msvc, 'msvc'), parent_build_command(command, msvc, msvc, 'msvc'))
            result = metrics.build_command(command, msvc, clang, 'clang-cl')
            additions = ','.join(powershell_literal('-DCMAKE_' + language + '_FLAGS_INIT:STRING=' + metrics.FLAG) for language in ('C', 'CXX'))
            self.assertEqual(result[4], parent_build_command(command, msvc, clang, 'clang-cl')[4].replace(' -CMakeArgs @(', ' -CMakeArgs @(' + additions + ','))
            for field in ('CMAKE_C_FLAGS', 'CMAKE_CXX_FLAGS_INIT', 'CMAKE_C_FLAGS_RELEASE'):
                with self.assertRaises(ValueError):
                    metrics.build_command(command[:4] + [source.replace(' -CMakeArgs @(', " -CMakeArgs @('-D" + field + "=/fp:fast',")], msvc, clang, 'clang-cl')
        with patch.object(metrics, 'descriptor', side_effect=lambda path: {'bytes': 1, 'sha256': '0' * 64}
                          if str(path) in identities else real_descriptor(path)), self.assertRaisesRegex(ValueError, 'compiler binary'):
            metrics.build_command(command, msvc, clang, 'clang-cl')

    def test_all_actual_97_commands_admit_with_only_candidate_flag(self):
        for kind in ('msvc', 'clang-cl'):
            cache, database, files, compiler = self.fixture(kind)
            result = self.admit(kind, cache, database, files, compiler)
            self.assertEqual(result['implicit_contraction_admission']['compiler_commands_matching_original'], 97)
            self.assertEqual(result['implicit_contraction_admission']['added_flag'], metrics.FLAG if kind == 'clang-cl' else None)

    def test_changed_flags_missing_duplicate_or_conflicting_option_reject(self):
        cache, database, files, compiler = self.fixture('clang-cl')
        for command in [database[0]['command'].replace(metrics.FLAG + ' ', ''),
                        database[0]['command'] + ' ' + metrics.FLAG,
                        database[0]['command'] + ' /fp:fast', database[0]['command'] + ' /Od']:
            changed = [{**database[0], 'command': command}, *database[1:]]
            with self.subTest(command=command), self.assertRaises(ValueError):
                self.admit('clang-cl', cache, changed, files, compiler)
        for changed_cache in [cache.replace('/DWIN32 /D_WINDOWS', '/D_WINDOWS'),
                              cache.replace('_FLAGS_RELEASE:STRING=/O2', '_FLAGS_RELEASE:STRING=/Od')]:
            with self.assertRaises(ValueError):
                self.admit('clang-cl', changed_cache, database, files, compiler)

    def test_only_temporary_source_root_and_compiler_alias_may_normalize(self):
        _, database, _ = metrics.baseline('clang-cl')
        old = PureWindowsPath(database[0]['directory']).parents[1]
        new = PureWindowsPath(r'Z:\isolated\source-clang-cl')
        moved = [{key: value.replace(str(old), str(new)).replace(old.as_posix(), new.as_posix())
                  for key, value in row.items()} for row in database]
        expected = metrics.normalized_commands(database, 'clang-cl', False)
        self.assertEqual(metrics.normalized_commands(moved, 'clang-cl', False), expected)
        for changed in [[*moved, moved[0]],
                        [{**moved[0], 'file': r'C:\outside\source.c'}, *moved[1:]],
                        [{**moved[0], 'output': 'different.obj'}, *moved[1:]]]:
            try:
                actual = metrics.normalized_commands(changed, 'clang-cl', False)
            except ValueError:
                continue
            self.assertNotEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
