import tempfile
from pathlib import Path
import unittest

from build_valid_mel_candidate import descriptor
from verify_valid_mel_pnola_app import audit_app_images, require_closed_dependencies, ENTRY_ALIAS, ORT


class AppImageTest(unittest.TestCase):
    def test_dependencies_require_exact_framework_siblings(self):
        name = 'libgen2_dep.dylib'
        require_closed_dependencies(['@loader_path/' + name, '/usr/lib/libSystem.B.dylib',
                                     '/System/Library/Frameworks/Foundation.framework/Versions/C/Foundation'], {name})
        for dependency in ['@loader_path/../../outside/' + name, '@loader_path/./' + name,
                           '@loader_path/subdir/' + name, '@rpath/' + name,
                           '/usr/lib/../../outside/' + name, '@loader_path/unknown.dylib']:
            with self.subTest(dependency=dependency), self.assertRaises(ValueError):
                require_closed_dependencies([dependency], {name})

    def test_entry_copy_is_one_logical_component_and_every_provider_required(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Path(directory) / 'Pnola.app'; frameworks = app / 'Contents/Frameworks'; frameworks.mkdir(parents=True)
            app = app.resolve(); frameworks = app / 'Contents/Frameworks'
            executable = app / 'Contents/MacOS/pnola-desktop'; executable.parent.mkdir(); executable.write_bytes(b'exe')
            private = 'libgen2_entry.dylib'; dep = 'libgen2_dep.dylib'
            for name in [private, ENTRY_ALIAS, dep, ORT]:
                (frameworks / name).write_bytes(name.encode())
            verified = {'app': str(app), 'executable': descriptor(executable), 'entry_library': private,
                        'frameworks': {name: descriptor(frameworks / name) for name in [private, ENTRY_ALIAS, dep, ORT]}}
            trace = lambda paths: '\n'.join('dyld[1]: <AB-CD> ' + str(path) for path in paths)
            positive = [executable, frameworks / ENTRY_ALIAS, frameworks / dep, frameworks / ORT]
            result = audit_app_images(trace(positive), app, verified)
            self.assertEqual(result['logical_runtime_library_count'], 2)
            self.assertEqual(result['entry_image'], ENTRY_ALIAS)
            callback_trace = '\n'.join('PNOLA_LOADED_IMAGE_HEX ' + str(path).encode().hex() for path in positive)
            self.assertTrue(audit_app_images(callback_trace, app, verified)['process_local_add_image_history_observed'])
            for bad in ['PNOLA_LOADED_IMAGE_HEX zz', 'PNOLA_LOADED_IMAGE_HEX 72656c6174697665', 'PNOLA_LOADED_IMAGE_ERROR',
                        'PNOLA_LOADED_IMAGE_HEX', 'PNOLA_LOADED_IMAGE_HEXzz', 'PNOLA_LOADED_IMAGE_HEX 2f 61',
                        'PNOLA_LOADED_IMAGE_HEX 2F61', 'PNOLA_LOADED_IMAGE_HEX 2f6']:
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    audit_app_images(callback_trace + '\n' + bad, app, verified)
            for paths in [positive + [frameworks / private], positive[:-1], positive[:-2]]:
                with self.assertRaises(ValueError):
                    audit_app_images(trace(paths), app, verified)
            outside = Path(directory) / dep; outside.write_bytes(dep.encode())
            with self.assertRaises(ValueError):
                audit_app_images(trace(positive + [outside]), app, verified)
            (frameworks / dep).write_bytes(b'changed')
            with self.assertRaises(ValueError):
                audit_app_images(trace(positive), app, verified)


if __name__ == '__main__':
    unittest.main()
