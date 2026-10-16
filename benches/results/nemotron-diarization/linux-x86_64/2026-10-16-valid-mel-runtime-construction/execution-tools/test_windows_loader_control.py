from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from windows_loader_control import default_loader_control


class WindowsLoaderControlTests(unittest.TestCase):
    def test_only_native_missing_module_error_proves_the_negative_control(self):
        with tempfile.TemporaryDirectory() as temporary:
            library = Path(temporary).resolve() / 'native.dll'
            library.write_bytes(b'DLL fixture')
            kernel = Mock()
            kernel.LoadLibraryExW.return_value = None
            with patch('windows_loader_control.ctypes.WinDLL', return_value=kernel, create=True), patch(
                    'windows_loader_control.ctypes.set_last_error', create=True) as reset, patch(
                    'windows_loader_control.ctypes.get_last_error', return_value=126, create=True):
                result = default_loader_control(library)
                self.assertEqual(result['native_win32_error'], 126)
                self.assertFalse(result['default_loader_resolves_siblings'])
                reset.assert_called_once_with(0)
                kernel.LoadLibraryExW.assert_called_once_with(str(library), None, 0)
                kernel.FreeLibrary.assert_not_called()
            for code in [0, 2, 5, 193]:
                with patch('windows_loader_control.ctypes.WinDLL', return_value=kernel, create=True), patch(
                        'windows_loader_control.ctypes.set_last_error', create=True), patch(
                        'windows_loader_control.ctypes.get_last_error', return_value=code, create=True):
                    with self.assertRaisesRegex(ValueError, 'unexpected Win32 error'):
                        default_loader_control(library)
            kernel.LoadLibraryExW.return_value = 1234
            with patch('windows_loader_control.ctypes.WinDLL', return_value=kernel, create=True), patch(
                    'windows_loader_control.ctypes.set_last_error', create=True), patch(
                    'windows_loader_control.ctypes.get_last_error', return_value=0, create=True):
                with self.assertRaisesRegex(ValueError, 'unexpectedly resolved'):
                    default_loader_control(library)
                kernel.FreeLibrary.assert_called_once_with(1234)


if __name__ == '__main__':
    unittest.main()
