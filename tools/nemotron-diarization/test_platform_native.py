from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from platform_native import msvc_environment, powershell_literal


class PlatformToolTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
