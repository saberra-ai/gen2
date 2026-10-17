"""Check bounded ACL ABI and metadata retention before a denied open."""
import errno
import os
from pathlib import Path
import struct
import tempfile
import unittest

from macos_vm_permissions_probe import CAPACITY, TARGETS, inspect_paths, packed_security


class PermissionProbeTests(unittest.TestCase):
    def test_canonical_empty_reference_retains_raw_without_policy_claim(self):
        result = packed_security(struct.pack('<IiI', 12, 8, 0))
        self.assertEqual(result['attribute_bytes'], 0)
        self.assertEqual(result['attribute_offset'], 12)
        self.assertFalse(result['acl_policy_qualified'])
        self.assertTrue(result['abi_shape_qualified'])
        null = packed_security(struct.pack('<IiI', 12, 0, 0))
        self.assertTrue(null['null_reference'])
        self.assertFalse(null['recognized_filesec'])
        self.assertFalse(null['abi_shape_qualified'])
        opaque = packed_security(struct.pack('<IiI', 16, 8, 4) + b'abcd')
        self.assertFalse(opaque['abi_shape_qualified'])
        padded = packed_security(struct.pack('<IiI', 16, 12, 0) + bytes(4))
        self.assertFalse(padded['canonical_empty'])
        self.assertFalse(padded['abi_shape_qualified'])

    def test_empty_acl_and_noacl_sentinel_remain_distinct(self):
        for count in [0, 0xFFFFFFFF]:
            body = struct.pack('<I', 0x012CC16D) + bytes(32) + struct.pack('<II', count, 0)
            result = packed_security(struct.pack('<IiI', 12 + len(body), 8, len(body)) + body)
            self.assertTrue(result['recognized_filesec'])
            self.assertEqual(result['acl_entrycount'], count)
            self.assertFalse(result['acl_policy_qualified'])

    def test_silent_truncation_negative_alignment_and_out_of_buffer_rejected(self):
        for raw in [b'', struct.pack('<IiI', 4096, 8, 44), struct.pack('<IiI', 12, -4, 4),
                struct.pack('<IiI', 12, 9, 0), struct.pack('<IiI', 12, 8, 1),
                struct.pack('<IiI', 12, 0, 1), bytes(CAPACITY + 1),
                struct.pack('<IiII', 16, 8, 4, 0x012CC16D)]:
            with self.subTest(size=len(raw)), self.assertRaises(ValueError):
                packed_security(raw)

    def tree(self, root):
        for path in TARGETS:
            (root / path).mkdir(parents=True, exist_ok=True)

    def test_users_metadata_and_acl_are_retained_before_denied_open(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.tree(root)
            seen = []
            def query(parent, name):
                seen.append(name)
                return {'status': 'fixture', 'leaf': name}
            def denied(name, flags, *, dir_fd):
                if name == 'users':
                    self.assertIn('users', seen)
                    raise PermissionError(errno.EACCES, 'fixture denied', name)
                return os.open(name, flags, dir_fd=dir_fd)
            fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                rows = inspect_paths(fd, query=query, opener=denied)
                users = rows['private/var/db/dslocal/nodes/Default/users']
                self.assertEqual(users['stat']['inode'], (root / 'private/var/db/dslocal/nodes/Default/users').stat().st_ino)
                self.assertEqual(users['security']['leaf'], 'users')
                self.assertEqual(users['open']['errno'], errno.EACCES)
            finally:
                os.close(fd)

    def test_symlink_entry_cannot_escape_to_external_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'guest').mkdir()
            (root / 'outside').mkdir()
            (root / 'outside/secret').write_bytes(b'unchanged')
            (root / 'guest/Users').symlink_to(root / 'outside', target_is_directory=True)
            fd = os.open(root / 'guest', os.O_RDONLY | os.O_DIRECTORY)
            rows = {}
            try:
                with self.assertRaises(ValueError):
                    inspect_paths(fd, query=lambda parent, name: {'status': 'fixture'}, rows=rows)
                self.assertEqual(rows['Users']['status'], 'metadata_observed')
                self.assertNotIn('security', rows['Users'])
                self.assertEqual((root / 'outside/secret').read_bytes(), b'unchanged')
            finally:
                os.close(fd)

    def test_unexpected_security_failure_keeps_partial_metadata_and_closes_fds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.tree(root)
            opened = []
            def opener(name, flags, *, dir_fd):
                fd = os.open(name, flags, dir_fd=dir_fd)
                opened.append(fd)
                return fd
            def query(parent, name):
                if name == 'users':
                    raise RuntimeError('fixture unexpected metadata failure')
                return {'status': 'fixture'}
            fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            rows = {}
            try:
                with self.assertRaises(RuntimeError):
                    inspect_paths(fd, query=query, opener=opener, rows=rows)
                self.assertIn('stat', rows['private/var/db/dslocal/nodes/Default/users'])
                for closed in opened:
                    with self.assertRaises(OSError):
                        os.fstat(closed)
                self.assertEqual(os.fstat(fd).st_ino, root.stat().st_ino)
            finally:
                os.close(fd)
