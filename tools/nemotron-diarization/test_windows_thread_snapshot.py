"""Execute the C++ observer with a fake API, separately from real Windows controls."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest


class WindowsThreadSnapshotTests(unittest.TestCase):
    def test_cpp_observer_filters_owns_handles_and_rejects_incomplete_queries(self):
        compiler = shutil.which('c++') or shutil.which('clang++')
        if os.name != 'nt' and not compiler:
            self.skipTest('C++ compiler required for fake-API regression; actual Windows control remains mandatory')
        tools = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / ('snapshot-test.exe' if os.name == 'nt' else 'snapshot-test')
            environment = dict(os.environ)
            if os.name == 'nt':
                from platform_native import msvc_environment, msvc_tool
                environment = msvc_environment(environment)
                command = [str(msvc_tool(environment, 'cl.exe')), '/nologo', '/std:c++17', '/EHsc', '/O2', '/W4', '/WX',
                           str(tools / 'windows_thread_snapshot_test.cpp'), '/link', '/OUT:' + str(binary)]
            else:
                command = [compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                           str(tools / 'windows_thread_snapshot_test.cpp'), '-o', str(binary)]
            compiled = subprocess.run(command, capture_output=True, text=True, cwd=directory, env=environment)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            completed = subprocess.run([str(binary)], capture_output=True, text=True, env=environment)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn('9 fake snapshot cases passed', completed.stdout)


if __name__ == '__main__':
    unittest.main()
