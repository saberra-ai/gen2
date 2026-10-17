"""Model-free session timing coverage, arithmetic and delivery controls."""
import hashlib
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile

from build_valid_mel_candidate import descriptor
from windows_session_timing import EXPECTED_SESSIONS, extract_delivery, profiling_environment, summarize


LINE = '[timing] session out=output.chunk_embs     nodes=3000 in=1.00 alloc=2.00 compute=90.00 out=3.00 total=96.00 ms'
FRONTEND = '[timing] fe path=cpu n_samples=16640 n_frames=105 = 1.00 ms'


def full_log(session=LINE):
    return '\n'.join([session] * EXPECTED_SESSIONS + [FRONTEND] * 59)


class SessionProfileTest(unittest.TestCase):
    def test_complete_timings_and_untimed_residual(self):
        log = full_log()
        result = summarize(log, 'timing', 10000000)
        self.assertEqual(result['session_timing_rows'], 84)
        self.assertEqual(result['totals_ms']['total'], 8064)
        self.assertEqual(result['outside_timed_regions_ms'], 1877)
        self.assertEqual(result['frontend_total_ms'], 59)
        self.assertAlmostEqual(result['compute_and_output_fraction'], .7812)
        self.assertAlmostEqual(result['total_rounding_bound_ms'], .715)

    def test_missing_malformed_extra_and_inconsistent_regions_fail(self):
        log = full_log()
        for invalid in ['\n'.join([LINE] * 83), log + '\n' + LINE, log.replace('compute=90.00', 'compute=nan'),
                        log.replace('alloc=2.00', 'alloc=20.00'), log.replace('nodes=3000', 'nodes=0'),
                        log.replace('in=1.00 alloc=2.00', 'in=1.00 direct'), log.replace('total=96.00', 'total=0.00'),
                        log.replace('fe path=cpu', 'fe path=gpu'), log.replace('n_samples=16640', 'n_samples=0')]:
            with self.subTest(log=invalid[:100]), self.assertRaises(ValueError):
                summarize(invalid, 'timing', 10000000)
        with self.assertRaisesRegex(ValueError, 'exceed'):
            summarize(log, 'timing', 8000000)

    def test_control_environment_has_no_timing_and_does_not_mutate_parent(self):
        parent = {'PATH': 'fixture', 'nemo_speech_timing': '0', 'GH_TOKEN': 'fixture-token'}
        self.assertEqual(profiling_environment(parent, 'control'), {'PATH': 'fixture'})
        self.assertEqual(profiling_environment(parent, 'timing'), {'PATH': 'fixture', 'NEMO_SPEECH_TIMING': '1'})
        self.assertEqual(parent['nemo_speech_timing'], '0')
        self.assertEqual(parent['GH_TOKEN'], 'fixture-token')
        self.assertEqual(summarize('ordinary native log', 'control', 60000000)['session_timing_rows'], 0)
        with self.assertRaises(ValueError):
            summarize(LINE, 'control', 60000000)
        for invalid in ['auto', None]:
            with self.assertRaises(ValueError):
                profiling_environment(parent, invalid)

    def test_rounding_bound_is_representational(self):
        for total in ['95.98', '95.99', '96.01', '96.02']:
            with self.subTest(total=total):
                summarize(full_log(LINE.replace('total=96.00', 'total=' + total)), 'timing', 10000000)
        for total in ['95.97', '96.03']:
            with self.subTest(total=total), self.assertRaisesRegex(ValueError, 'inconsistent'):
                summarize(full_log(LINE.replace('total=96.00', 'total=' + total)), 'timing', 10000000)
        log = full_log(LINE.replace('total=96.00', 'total=96.01'))
        self.assertEqual(summarize(log, 'timing', 8123125)['native_elapsed_ms'], 8123.125)
        with self.assertRaises(ValueError):
            summarize(log, 'timing', 8123000)

    def test_exact_delivery_and_mismatched_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / 'source.zip'
            raw = b'fixture SDK bytes'
            with zipfile.ZipFile(archive, 'w') as bundle:
                bundle.writestr('sdk-4/bin/probe.exe', raw)
            members = {'sdk-4/bin/probe.exe': {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}}
            extract_delivery(archive, descriptor(archive), members, root / 'valid')
            self.assertEqual((root / 'valid/sdk-4/bin/probe.exe').read_bytes(), raw)
            with self.assertRaises(ValueError):
                extract_delivery(archive, descriptor(archive), members, root / 'valid')
            wrong = {'sdk-4/bin/probe.exe': {**members['sdk-4/bin/probe.exe'], 'sha256': '0' * 64}}
            with self.assertRaisesRegex(ValueError, 'bytes differ'):
                extract_delivery(archive, descriptor(archive), wrong, root / 'wrong')

    def test_unsafe_names_and_aliases_fail_before_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, name in enumerate(['../escape', 'C:/escape', 'sdk-4/CON.exe', 'sdk-4/probe.exe.']):
                archive = root / (str(index) + '.zip')
                with zipfile.ZipFile(archive, 'w') as bundle:
                    bundle.writestr(name, b'x')
                members = {name: {'bytes': 1, 'sha256': hashlib.sha256(b'x').hexdigest()}}
                with self.assertRaises(ValueError):
                    extract_delivery(archive, descriptor(archive), members, root / ('out-' + str(index)))
            archive = root / 'alias.zip'
            info = zipfile.ZipInfo('sdk-4/alias')
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(archive, 'w') as bundle:
                bundle.writestr(info, b'target')
            with self.assertRaises(ValueError):
                extract_delivery(archive, descriptor(archive), {'sdk-4/alias': {
                    'bytes': 6, 'sha256': hashlib.sha256(b'target').hexdigest()}}, root / 'alias-output')

    def test_exact_archive_members_and_case_collision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / 'extra.zip'
            row = {'bytes': 1, 'sha256': hashlib.sha256(b'x').hexdigest()}
            with zipfile.ZipFile(archive, 'w') as bundle:
                bundle.writestr('sdk-4/A', b'x')
                bundle.writestr('sdk-4/a', b'x')
            with self.assertRaisesRegex(ValueError, 'case-colliding'):
                extract_delivery(archive, descriptor(archive), {'sdk-4/A': row, 'sdk-4/a': row}, root / 'colliding')
            with self.assertRaisesRegex(ValueError, 'member set'):
                extract_delivery(archive, descriptor(archive), {'sdk-4/A': row}, root / 'extra')
            with self.assertRaisesRegex(ValueError, 'ZIP bytes'):
                extract_delivery(archive, {'bytes': 1, 'sha256': '0' * 64}, {'sdk-4/A': row}, root / 'digest')


if __name__ == '__main__':
    unittest.main()
