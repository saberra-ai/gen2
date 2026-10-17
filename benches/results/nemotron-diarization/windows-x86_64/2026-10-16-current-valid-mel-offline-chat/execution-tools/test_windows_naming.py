from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from test_windows_pe import fixture
from windows_naming import NAMES, verify_private_outputs


class PrivateOutputTests(unittest.TestCase):
    def prepare(self, root):
        import struct
        (root / 'bin').mkdir()
        (root / 'lib').mkdir()
        for name in NAMES.values():
            data = fixture()
            # Use the fixture's ample string area for longer private names.
            dependency = NAMES['ggml-base.dll'].encode() + b'\0'
            data[0x300:0x300 + len(dependency)] = dependency
            struct.pack_into('<I', data, 0x60c, 0x1600)
            encoded = name.encode() + b'\0'
            data[0x800:0x800 + len(encoded)] = encoded
            (root / 'bin' / name).write_bytes(data)
            # Upstream installs only the public C ABI import library. Internal
            # target import libraries remain build artifacts.
            if name == NAMES['nemo_speech_asr_c.dll']:
                (root / 'lib' / name.replace('.dll', '.lib')).write_bytes(b'import fixture')

    def test_accepts_private_outputs_but_rejects_generic_imports_aliases_and_missing_libraries(self):
        for mutation in ['none', 'generic-import', 'generic-alias', 'missing-import-library']:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.prepare(root)
                if mutation == 'generic-import':
                    (root / 'bin' / NAMES['ggml.dll']).write_bytes(fixture())
                elif mutation == 'generic-alias':
                    (root / 'bin/ggml.dll').write_bytes(fixture())
                elif mutation == 'missing-import-library':
                    (root / 'lib' / NAMES['nemo_speech_asr_c.dll'].replace('.dll', '.lib')).unlink()
                if mutation == 'none':
                    self.assertTrue(verify_private_outputs(root, root)['generic_imports_absent'])
                else:
                    with self.assertRaises(ValueError):
                        verify_private_outputs(root, root)


@unittest.skipUnless(shutil.which('cmake') and shutil.which('ninja'), 'CMake/Ninja required for deferred target naming')
class WindowsNamingTests(unittest.TestCase):
    def configure(self, root, extra='', ggml_kind='SHARED'):
        hook = Path(__file__).with_name('windows_private_dlls.cmake')
        # Exercise real deferred target/install generation on this host, without
        # pretending this proves PE output or Windows execution.
        (root / 'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.24)
project(naming C)
set(WIN32 TRUE)
set(CMAKE_SIZEOF_VOID_P 8)
set(CMAKE_BUILD_TYPE Release)
set(BUILD_SHARED_LIBS ON)
foreach(target IN ITEMS nemo_speech_asr_c nemo_speech_asr ggml ggml-base ggml-cpu)
  set(kind SHARED)
  if(target STREQUAL "ggml")
    set(kind GGML_KIND)
  endif()
  add_library(${target} ${kind} empty.c)
  install(TARGETS ${target} DESTINATION lib)
  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/${target}.txt" CONTENT "$<TARGET_PROPERTY:${target},OUTPUT_NAME>\\n$<TARGET_FILE_NAME:${target}>\\n$<TARGET_LINKER_FILE_NAME:${target}>\\n")
endforeach()
'''.replace('GGML_KIND', ggml_kind) + extra)
        (root / 'empty.c').write_text('int example(void) { return 0; }\n')
        return subprocess.run(['cmake', '-S', str(root), '-B', str(root / 'build'),
                               '-G', 'Ninja',
                               '-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=' + str(hook)],
                              capture_output=True, text=True)

    def test_deferred_hook_changes_generated_targets_and_installed_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.configure(root)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            install = (root / 'build/cmake_install.cmake').read_text()
            for target in ['nemo_speech_asr_c', 'nemo_speech_asr', 'ggml', 'ggml-base', 'ggml-cpu']:
                lines = (root / f'build/{target}.txt').read_text().splitlines()
                self.assertEqual(lines[0], 'gen2_nemo_windows_' + target)
                self.assertTrue(all('gen2_nemo_windows_' + target in name for name in lines[1:]))
                self.assertIn('gen2_nemo_windows_' + target, install)

    def test_rejects_existing_configuration_names_and_wrong_target_types(self):
        for extra in ['set_target_properties(ggml PROPERTIES RUNTIME_OUTPUT_NAME_RELEASE generic)',
                      'set_target_properties(ggml PROPERTIES OUTPUT_NAME generic)',
                      'set(WIN32 FALSE)']:
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as temporary:
                result = self.configure(Path(temporary), extra)
                self.assertNotEqual(result.returncode, 0, result.stdout)
        with tempfile.TemporaryDirectory() as temporary:
            result = self.configure(Path(temporary), ggml_kind='STATIC')
            self.assertNotEqual(result.returncode, 0, result.stdout)


if __name__ == '__main__':
    unittest.main()
