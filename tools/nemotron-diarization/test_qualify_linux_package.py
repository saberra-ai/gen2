from pathlib import Path
import os
import tempfile
import unittest

from qualify_linux_package import LIFECYCLE_GATES, RUNTIME_PATH, container_command, loader_paths, require_lifecycle


class LinuxContainerGateTests(unittest.TestCase):
    def test_loader_trace_requires_every_bundled_library_including_gcc(self):
        names = {'libgen2_nemo_linux_native.so', 'libstdc++.so.6', 'libgcc_s.so.1'}
        paths = [RUNTIME_PATH + '/lib/' + name for name in sorted(names)]
        trace = '\n'.join('   12: calling init: ' + path for path in [*paths, '/lib/x86_64-linux-gnu/libc.so.6'])
        result = loader_paths(trace, names)
        self.assertEqual(result['bundled'], paths)
        self.assertEqual(result['platform'], ['/lib/x86_64-linux-gnu/libc.so.6'])
        helper_rpath_trace = trace.replace(RUNTIME_PATH, '/work/bin/../relocated runtime ü')
        relocated = loader_paths(helper_rpath_trace, names)
        self.assertEqual(relocated['bundled'], paths)
        self.assertIn('/work/bin/../relocated runtime ü/lib/libstdc++.so.6', relocated['raw_initialization_paths'])
        for changed in [trace.replace(RUNTIME_PATH + '/lib/libstdc++.so.6', '/usr/lib/libstdc++.so.6'),
                        trace.replace(RUNTIME_PATH + '/lib/libgcc_s.so.1', ''),
                        trace + '\n 12: calling init: /usr/lib/libsentencepiece.so.0',
                        trace + '\n 12: calling init: /source/libc.so.6']:
            with self.assertRaises(ValueError):
                loader_paths(changed, names)

    def test_lifecycle_listing_and_actual_passes_must_match_the_full_gate_set(self):
        listed = '\n'.join(name + ': test' for name in LIFECYCLE_GATES)
        completed = '\n'.join('test ' + name + ' ... ok' for name in LIFECYCLE_GATES)
        require_lifecycle(listed)
        require_lifecycle(completed, completed=True)
        for changed in [completed.replace('... ok', '... ignored', 1), completed.replace('... ok', '... FAILED', 1), completed + '\ntest unaccounted ... ok']:
            with self.assertRaises(ValueError):
                require_lifecycle(changed, completed=True)

    def test_negative_runtime_is_the_only_package_mounted_at_the_positive_search_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            negative = root / 'missing library ü'
            negative.mkdir()
            results = root / 'results'
            results.mkdir()
            image = 'sha256:' + '1' * 64
            command = container_command(image, [(negative, RUNTIME_PATH)], results, ['/work/bin/public-stream'])
            self.assertIn('type=bind,src=' + str(negative) + ',dst=' + RUNTIME_PATH + ',readonly', command)
            self.assertIn('type=bind,src=' + str(results) + ',dst=/results', command)
            self.assertEqual(command[command.index('--network') + 1], 'none')
            self.assertEqual(command[command.index('--pull') + 1], 'never')
            self.assertIn('--read-only', command)
            # Acquired fixtures are 0600; capability-free root cannot read the
            # CI user's files or write its results. Preserve the owning UID/GID.
            self.assertEqual(command[command.index('--user') + 1], f'{os.getuid()}:{os.getgid()}')
            self.assertEqual(command[command.index(image) + 1:command.index(image) + 3], ['/usr/bin/env', '-i'])
            self.assertEqual(sum(item.startswith('type=bind,') for item in command), 2)
            for variable in ['LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_AUDIT']:
                with self.assertRaises(ValueError):
                    container_command(image, [], results, ['true'], {variable: '/host/sdk'})


if __name__ == '__main__':
    unittest.main()
