from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from platform_native import msvc_compiler_definitions, msvc_environment, pinned_git_environment, powershell_literal


class PlatformToolTests(unittest.TestCase):
    def test_windows_compiler_selection_is_explicit_and_case_insensitive(self):
        with tempfile.TemporaryDirectory() as temporary:
            compiler = Path(temporary) / 'cl.exe'
            compiler.write_bytes(b'compiler fixture')
            environment = {'Path': 'active MSVC path'}
            with patch('platform_native.shutil.which', return_value=str(compiler)) as which:
                definitions = msvc_compiler_definitions(environment)
            which.assert_called_once_with('cl.exe', path='active MSVC path')
            self.assertEqual(definitions, ['-DCMAKE_C_COMPILER=' + str(compiler.resolve()),
                                           '-DCMAKE_CXX_COMPILER=' + str(compiler.resolve())])
            self.assertEqual(environment, {'Path': 'active MSVC path'})
        with patch('platform_native.shutil.which', return_value=None):
            with self.assertRaisesRegex(ValueError, 'MSVC compiler'):
                msvc_compiler_definitions({'PATH': 'missing'})

    def test_windows_environment_lookup_and_merge_are_case_insensitive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vcvars = root / 'VC/Auxiliary/Build/vcvars64.bat'
            vcvars.parent.mkdir(parents=True)
            vcvars.write_text('fixture')
            environment = {'PROGRAMFILES(X86)': str(root), 'PATH': 'old', 'KEEP': 'unchanged'}
            with patch('platform_native.capture', return_value=str(root)), patch(
                    'platform_native.subprocess.check_output', return_value='Path=new\r\nLIB=libdir\r\n=C:=ignored\r\n'.encode('utf-16-le')) as run:
                result = msvc_environment(environment)
            self.assertEqual({k: v for k, v in result.items() if k.lower() == 'path'}, {'Path': 'new'})
            self.assertEqual(result['KEEP'], 'unchanged')
            self.assertEqual(result['LIB'], 'libdir')
            self.assertNotIn('', result)
            self.assertEqual(environment['PATH'], 'old')
            self.assertIn('/d /u /s /c ""', run.call_args.args[0])

    def test_powershell_literal_preserves_metacharacters(self):
        self.assertEqual(powershell_literal("C:/a'b/$(literal)"), "'C:/a''b/$(literal)'")

    def test_pinned_git_configuration_preserves_existing_entries(self):
        original = {'GIT_CONFIG_COUNT': '1', 'GIT_CONFIG_KEY_0': 'safe.directory',
                    'GIT_CONFIG_VALUE_0': '/existing'}
        result = pinned_git_environment(original)
        self.assertEqual(result['GIT_CONFIG_COUNT'], '3')
        self.assertEqual(result['GIT_CONFIG_VALUE_0'], '/existing')
        self.assertEqual(result['GIT_CONFIG_KEY_1'], 'core.autocrlf')
        self.assertEqual(result['GIT_CONFIG_VALUE_1'], 'false')
        self.assertEqual(result['GIT_CONFIG_KEY_2'], 'core.eol')
        self.assertEqual(result['GIT_CONFIG_VALUE_2'], 'lf')
        self.assertEqual(original['GIT_CONFIG_COUNT'], '1')


if __name__ == '__main__':
    unittest.main()
