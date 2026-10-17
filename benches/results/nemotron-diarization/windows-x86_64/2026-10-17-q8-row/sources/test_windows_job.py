"""Owned-process controls; actual Windows cases execute no compiler or model."""
import ctypes
import io
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from windows_job import Accounting, BasicLimits, ExtendedLimits, Job, _child, run_in_job


class JobControlTest(unittest.TestCase):
    def test_x64_abi_layout(self):
        if ctypes.sizeof(ctypes.c_void_p) != 8:
            self.skipTest('64-bit Job layout')
        self.assertEqual(ctypes.sizeof(BasicLimits), 64)
        self.assertEqual(ctypes.sizeof(ExtendedLimits), 144)
        self.assertEqual(ctypes.sizeof(Accounting), 48)
        self.assertEqual(BasicLimits.flags.offset, 16)

    def test_invalid_gate_payload_cannot_spawn(self):
        for payload in [b'', b'{}\n', b'{"command":["python"],"cwd":null}\n',
                        b'{"command":[5],"cwd":"fixture"}\n', b'x' * 4097]:
            with self.subTest(payload=payload[:50]), patch('sys.stdin', io.TextIOWrapper(io.BytesIO(payload))), \
                    patch('windows_job.subprocess.run', side_effect=AssertionError('unexpected spawn')):
                with self.assertRaises(ValueError):
                    _child()

    @unittest.skipUnless(platform.system() == 'Windows', 'actual Windows Job API required')
    def test_normal_exit_and_nonzero_status_are_reaped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for status in (0, 9):
                with (root / ('out-' + str(status))).open('wb') as out, (root / ('err-' + str(status))).open('wb') as err:
                    result = run_in_job([sys.executable, '-c', 'import sys;print("owned child");sys.exit(' + str(status) + ')'],
                        cwd=root, env=os.environ.copy(), stdout=out, stderr=err, timeout=15)
                self.assertEqual(result.returncode, status)
                self.assertTrue(result.process_tree_reaped)
                self.assertIn('owned child', (root / ('out-' + str(status))).read_text())

    @unittest.skipUnless(platform.system() == 'Windows', 'actual Windows Job API required')
    def test_timeout_kills_real_grandchild_before_late_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            late, spawned = root / 'late-write', root / 'spawned-pid'
            grandchild = 'import time,pathlib;time.sleep(4);pathlib.Path(' + repr(str(late)) + ').write_text("escaped")'
            command = 'import subprocess,sys,pathlib,time;p=subprocess.Popen([sys.executable,"-c",' + repr(grandchild) + ']);pathlib.Path(' + repr(str(spawned)) + ').write_text(str(p.pid));time.sleep(30)'
            with (root / 'out').open('wb') as out, (root / 'err').open('wb') as err:
                with self.assertRaises(subprocess.TimeoutExpired) as caught:
                    run_in_job([sys.executable, '-c', command], cwd=root, env=os.environ.copy(), stdout=out, stderr=err, timeout=2)
            self.assertTrue(caught.exception.process_tree_reaped)
            self.assertTrue(spawned.is_file(), 'test must establish an actual descendant before timeout')
            time.sleep(4.2)
            self.assertFalse(late.exists())

    @unittest.skipUnless(platform.system() == 'Windows', 'actual Windows Job API required')
    def test_assignment_failure_reaps_gated_worker_without_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / 'must-not-run'
            command = 'import pathlib;pathlib.Path(' + repr(str(marker)) + ').write_text("ran")'
            with (root / 'out').open('wb') as out, (root / 'err').open('wb') as err, \
                    patch.object(Job, 'assign', side_effect=OSError('fixture assignment failure')):
                with self.assertRaises(OSError) as caught:
                    run_in_job([sys.executable, '-c', command], cwd=root, env=os.environ.copy(), stdout=out, stderr=err, timeout=15)
            self.assertTrue(caught.exception.process_tree_reaped)
            self.assertFalse(marker.exists())


if __name__ == '__main__':
    unittest.main()
