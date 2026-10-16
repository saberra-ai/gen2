"""Reject host CPU leakage anywhere in a portable SDK compiler database."""
import unittest
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

from macos_cpu_architecture import PORTABLE_ARM64, audit_portable_commands
from platform_macos_portable import SANDBOX_CONTROL, construction_policy, construction_roots


class PortableMacCpu(unittest.TestCase):
    @unittest.skipUnless(platform.system() == 'Darwin', 'Mac construction destination guard')
    def test_dangling_destination_symlinks_rejected_before_resolution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for destination in range(1, 5):
                paths = [root / (str(destination) + '-' + name)
                         for name in ['upstream', 'source', 'sdk', 'package', 'output']]
                target = root / ('absent-' + str(destination))
                paths[destination].symlink_to(target)
                with self.assertRaisesRegex(ValueError, 'fresh construction destinations'):
                    construction_roots(paths)
                self.assertFalse(target.exists())

    def rows(self):
        return [{'file': '/source/' + name,
                 'arguments': ['clang++', '-march=' + PORTABLE_ARM64, '-c', '/source/' + name]}
                for name in ['ggml-cpu/ggml-cpu.cpp', 'ggml-cpu/arch/arm/quants.c',
                             'ggml-cpu/arch/arm/repack.cpp', 'asr/c_api.cpp']]

    def test_all_sdk_units_and_cpu_kernels_are_admitted(self):
        rows = self.rows()
        # Exercise both forms emitted by CMake.
        rows[0]['command'] = ' '.join(rows[0].pop('arguments'))
        result = audit_portable_commands(rows)
        self.assertEqual(result['compiler_units'], 4)
        self.assertFalse(result['hardware_or_model_qualification'])

    def test_host_or_extra_isa_overrides_in_other_sdk_unit_rejected(self):
        for flag in ['-mcpu=native', '-march=armv8.8-a', '-Xclang', '--target=arm64-apple-macos26',
                     '-D__ARM_FEATURE_MATMUL_INT8=1', '-DGGML_USE_SME=1']:
            rows = self.rows()
            rows[-1]['arguments'].append(flag)
            with self.assertRaises(ValueError):
                audit_portable_commands(rows)

    def test_missing_target_in_non_kernel_unit_rejected(self):
        rows = self.rows()
        rows[-1]['arguments'].remove('-march=' + PORTABLE_ARM64)
        with self.assertRaises(ValueError):
            audit_portable_commands(rows)

    def test_missing_kernel_or_empty_database_rejected(self):
        for rows in [[], self.rows()[1:]]:
            with self.assertRaises(ValueError):
                audit_portable_commands(rows)

    @unittest.skipUnless(platform.system() == 'Darwin', 'actual macOS sandbox control')
    def test_actual_construction_sandbox_denies_weight_read_and_loopback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            probes = [root / ('denied.' + suffix)
                      for suffix in ['nemo', 'gguf', 'safetensors', 'onnx', 'npy', 'f32']]
            for probe in probes:
                probe.write_bytes(b'not a model')
                self.assertEqual(probe.read_bytes(), b'not a model')
            policy = root / 'construction.sb'
            policy.write_text(construction_policy(root / 'upstream', root / 'repository'))
            result = subprocess.run(['/usr/bin/sandbox-exec', '-f', str(policy), sys.executable,
                                     '-c', SANDBOX_CONTROL, *map(str, probes)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
