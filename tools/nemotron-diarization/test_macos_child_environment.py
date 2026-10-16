"""Trace the target process without admitting launcher images to its closure."""
from pathlib import Path
import platform
import re
import subprocess
import tempfile
import unittest

from macos_child_environment import sandboxed_command


@unittest.skipUnless(platform.system() == 'Darwin', 'actual macOS loader boundary')
class ChildEnvironment(unittest.TestCase):
    def test_loader_trace_contains_target_without_launcher_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source, binary, policy = root / 'probe.c', root / 'probe', root / 'offline.sb'
            source.write_text('#include <stdio.h>\n#include <stdlib.h>\n'
                              'int main(void) { const char *v = getenv("DYLD_PRINT_LIBRARIES"); '
                              'puts(v ? v : "missing"); return 0; }\n')
            subprocess.run(['/usr/bin/clang', str(source), '-o', str(binary)], check=True,
                           capture_output=True, text=True, timeout=30)
            policy.write_text('(version 1)\n(allow default)\n(deny network*)\n')
            environment = {'PATH': '/usr/bin:/bin', 'DYLD_PRINT_LIBRARIES': '1'}
            command, parent = sandboxed_command([binary], policy, environment)
            result = subprocess.run(command, env=parent, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), '1')
            images = set(re.findall(r'dyld\[\d+\]: <[A-Fa-f0-9-]+> (/[^\r\n]+)', result.stderr))
            self.assertIn(str(binary), images)
            self.assertNotIn('/usr/bin/sandbox-exec', images)
            self.assertNotIn('/usr/bin/env', images)
            self.assertEqual(environment['DYLD_PRINT_LIBRARIES'], '1')


if __name__ == '__main__':
    unittest.main()
