"""Model-free controls for diagnostic identity, instrumentation and full parity."""
import copy
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

from windows_worker_metrics import (DIARIZER, BACKEND, PCM_SHA256, compare_native,
                                    derive_probe, metadata, telemetry, transform, version)
from windows_worker_sources_v2 import checkout_environment


def result(cap):
    return {'audio_samples': 960000, 'audio_duration_seconds': 60, 'runtime_version': version(cap),
            'input_pcm_f32le_sha256': PCM_SHA256, 'input_repeats': 1, 'requested_gpu_index': -1,
            'geometry': {'chunk': 9, 'right': 4, 'left': 0, 'fifo': 264, 'cache': 264, 'update': 222},
            'native_frame_count': 6000, 'retained_first_frame': 0, 'retained_rows': 6000,
            'elapsed_micros': 60000001, 'turns': [{'start_seconds': 0, 'end_seconds': 1, 'speaker': 1}]}


class WorkerDiagnosticTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which('git-lfs'), 'actual Git LFS required for checkout regression')
    def test_missing_unrelated_lfs_asset_does_not_break_source_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination = root / 'source', root / 'destination'
            environment = dict(os.environ)
            environment.pop('GIT_LFS_SKIP_SMUDGE', None)
            # Isolate Git configuration and configure the real installed filter
            # in this child's mapping only. The sole remote is a local path.
            for name in list(environment):
                if name.startswith('GIT_CONFIG_'):
                    del environment[name]
            environment.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
                               GIT_CONFIG_COUNT='2', GIT_CONFIG_KEY_0='filter.lfs.process',
                               GIT_CONFIG_VALUE_0='git-lfs filter-process',
                               GIT_CONFIG_KEY_1='filter.lfs.required', GIT_CONFIG_VALUE_1='true')
            def git(*args, env=environment, check=True):
                return subprocess.run(['git', *map(str, args)], env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, check=check)
            git('init', source)
            (source / '.gitattributes').write_text('unrelated.bin filter=lfs -text\n')
            pointer = b'version https://git-lfs.github.com/spec/v1\noid sha256:' + b'a' * 64 + b'\nsize 1\n'
            (source / 'unrelated.bin').write_bytes(pointer)
            git('-C', source, 'add', '.gitattributes', 'unrelated.bin')
            git('-C', source, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                '-c', 'commit.gpgsign=false', 'commit', '-m', 'Missing unrelated LFS fixture')
            git('clone', '--no-hardlinks', '--no-checkout', source, destination)
            broken = git('-C', destination, 'checkout', '--detach', 'HEAD', check=False)
            self.assertNotEqual(broken.returncode, 0)
            self.assertIn(b'smudge', broken.stderr.lower())
            fixed = checkout_environment(environment)
            fixed_destination = root / 'fixed-destination'
            git('clone', '--no-hardlinks', '--no-checkout', source, fixed_destination, env=fixed)
            git('-C', fixed_destination, 'checkout', '--detach', 'HEAD', env=fixed)
            self.assertEqual((fixed_destination / 'unrelated.bin').read_bytes(), pointer)
            self.assertNotIn('GIT_LFS_SKIP_SMUDGE', environment)

    def test_complete_arrays_before_timing(self):
        raw = struct.pack('<48000f', *([.5] * 48000))
        compared = compare_native(result(4), result(2), raw, raw, 4, 2)
        self.assertTrue(compared['passed'])
        self.assertTrue(compared['probabilities_bit_exact'])
        self.assertGreater(compared['actual_rtf'], 1)
        changed = raw[:-4] + struct.pack('<f', .50001)
        with self.assertRaisesRegex(ValueError, 'probability values'):
            compare_native(result(4), result(2), raw, changed, 4, 2)
        with self.assertRaisesRegex(ValueError, '48000'):
            compare_native(result(4), result(2), raw, raw[:-4], 4, 2)
        with self.assertRaisesRegex(ValueError, 'invalid native/Gen2 probabilities'):
            compare_native(result(4), result(2), raw, raw[:-4] + struct.pack('<f', float('nan')), 4, 2)

    def test_identity_geometry_and_turn_fail_closed(self):
        raw = struct.pack('<48000f', *([.5] * 48000))
        mutations = {'runtime_version': 'nemo-speech-asr 0.2.0+gen2-valid-mel-pool-cpu-v2',
                     'input_pcm_f32le_sha256': '0' * 64, 'input_repeats': 2, 'requested_gpu_index': 0,
                     'retained_first_frame': 1, 'retained_rows': 5999, 'audio_samples': 959999,
                     'geometry': {}, 'elapsed_micros': 0,
                     'turns': [{'start_seconds': 0, 'end_seconds': 2, 'speaker': 1}]}
        for key, value in mutations.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                actual = copy.deepcopy(result(2))
                actual[key] = value
                compare_native(result(4), actual, raw, raw, 4, 2)
        for end in (100, float('inf'), float('nan')):
            actual = result(2)
            actual['turns'][0]['end_seconds'] = end
            with self.assertRaises(ValueError):
                compare_native(result(4), actual, raw, raw, 4, 2)

    def test_actual_worker_telemetry_is_required(self):
        log = '[DEBUG-worker-v1] cap=2 available=4 effective=2\n[DEBUG-worker-v1] attached_pool_threads=2\n'
        self.assertEqual(telemetry(log, 2)['attached_pool_threads'], 2)
        for invalid in [log + log, '', log.splitlines()[0], log.replace('effective=2', 'effective=4'),
                        log.replace('attached_pool_threads=2', 'attached_pool_threads=1'), log.replace('cap=2', 'cap=4')]:
            with self.subTest(log=invalid), self.assertRaises(ValueError):
                telemetry(invalid, 2)
        self.assertEqual(telemetry(log.replace('available=4', 'available=0').replace('effective=2', 'effective=1')
                                  .replace('attached_pool_threads=2', 'attached_pool_threads=1'), 2)['effective_threads'], 1)

    def test_exact_source_sites_and_distinct_versions(self):
        diarizer = b'#include <algorithm>\n        params.n_threads = bounded_cpu_threads(params.n_threads, std::thread::hardware_concurrency());'
        backend = b'#include <cstdlib>\n        set_pool(backend, pool);'
        for cap in (1, 2, 4):
            self.assertIn(f'bounded_cpu_threads({cap}, available)'.encode(), transform(DIARIZER, diarizer, cap))
            self.assertIn(b'attached_pool_threads=%d', transform(BACKEND, backend, cap))
            probe = b'before "nemo-speech-asr 0.2.0+gen2-valid-mel-v1" after'
            self.assertEqual(derive_probe(probe, cap), b'before "' + version(cap).encode() + b'" after')
            for invalid in [probe + probe, b'wrong']:
                with self.assertRaises(ValueError):
                    derive_probe(invalid, cap)
        for invalid in [0, 3, True, '2']:
            with self.assertRaises(ValueError):
                metadata(invalid)
        with self.assertRaises(ValueError):
            transform(DIARIZER, diarizer + diarizer, 2)
        with self.assertRaises(ValueError):
            transform(BACKEND, backend.replace(b'set_pool', b'other'), 2)


if __name__ == '__main__':
    unittest.main()
