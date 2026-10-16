"""Reject incomplete/ambiguous dependency evidence before packaging PE images."""
from pathlib import Path
import struct
import tempfile
import unittest

from windows_pe import inspect_pe


def fixture():
    data = bytearray(0x1200)
    data[:2] = b'MZ'
    struct.pack_into('<I', data, 0x3c, 0x80)
    data[0x80:0x84] = b'PE\0\0'
    struct.pack_into('<HHIIIHH', data, 0x84, 0x8664, 1, 0, 0, 0, 240, 0x2022)
    optional = 0x98
    struct.pack_into('<H', data, optional, 0x20b)
    struct.pack_into('<Q', data, optional + 24, 0x180000000)
    struct.pack_into('<II', data, optional + 56, 0x3000, 0x200)
    struct.pack_into('<I', data, optional + 108, 16)
    # One real file-backed section, with RVAs deliberately unlike file offsets.
    struct.pack_into('<8sIIIIIIHHI', data, optional + 240, b'.rdata\0\0',
                     0x1000, 0x1000, 0x1000, 0x200, 0, 0, 0, 0, 0x40000040)
    put = lambda rva, value: data.__setitem__(slice(rva - 0xe00, rva - 0xe00 + len(value)), value)
    def directory(index, rva, size):
        struct.pack_into('<II', data, optional + 112 + 8 * index, rva, size)
    directory(1, 0x1000, 40)
    put(0x1000, struct.pack('<IIIII', 0, 0, 0, 0x1100, 0x1180))
    put(0x1100, b'ggml-base.dll\0')
    directory(13, 0x1200, 64)
    put(0x1200, struct.pack('<IIIIIIII', 1, 0x1300, 0, 0x1380, 0x1388, 0, 0, 0))
    put(0x1300, b'VCRUNTIME140.dll\0')
    directory(0, 0x1400, 0x100)
    put(0x1400, struct.pack('<IIHHIIIIIII', 0, 0, 0, 0, 0x1510, 1, 2, 1,
                           0x1520, 0x1530, 0x1540))
    put(0x1450, b'KERNEL32.Sleep\0')
    put(0x1510, b'test.dll\0')
    put(0x1520, struct.pack('<II', 0x1800, 0x1450))
    put(0x1530, struct.pack('<I', 0x1550))
    put(0x1540, struct.pack('<H', 0))
    put(0x1550, b'nemo_stream_push\0')
    return data


class PeEvidenceTests(unittest.TestCase):
    def inspect(self, data):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'test.dll'
            path.write_bytes(data)
            return inspect_pe(path)

    def test_collects_normal_delay_and_ordinal_forwarder_dependencies(self):
        report = self.inspect(fixture())
        self.assertEqual(report['imports'], ['ggml-base.dll'])
        self.assertEqual(report['delay_imports'], ['vcruntime140.dll'])
        self.assertEqual(report['forwarder_dependencies'], ['kernel32.dll'])
        self.assertEqual(report['export_name'], 'test.dll')
        self.assertEqual(report['exports'], ['nemo_stream_push'])
        self.assertEqual(report['export_count'], 2)
        self.assertFalse(report['bound_imports_present'])

    def test_executable_export_identity_does_not_relax_dll_dependency_names(self):
        data = fixture()
        struct.pack_into('<H', data, 0x96, 0x22)  # executable image without IMAGE_FILE_DLL
        data[0x710:0x719] = b'test.exe\0'
        report = self.inspect(data)
        self.assertFalse(report['is_dll'])
        self.assertEqual(report['export_name'], 'test.exe')
        data[0x300:0x30e] = b'dependency.exe'
        data[0x30e] = 0
        with self.assertRaises(ValueError):
            self.inspect(data)
        dll = fixture()
        dll[0x710:0x719] = b'test.exe\0'
        with self.assertRaises(ValueError):
            self.inspect(dll)

    def test_rejects_truncated_wrong_architecture_and_ambiguous_mappings(self):
        cases = [fixture()[:200], fixture()]
        struct.pack_into('<H', cases[1], 0x84, 0x14c)
        overlap = fixture()
        struct.pack_into('<H', overlap, 0x86, 2)
        overlap[0x1b0:0x1d8] = overlap[0x188:0x1b0]
        cases.append(overlap)
        zero_fill = fixture()
        struct.pack_into('<I', zero_fill, 0x200 + 12, 0x2100)
        cases.append(zero_fill)
        for data in cases:
            with self.subTest(size=len(data)), self.assertRaises(ValueError):
                self.inspect(data)

    def test_rejects_unterminated_tables_unsafe_names_and_legacy_delay_addresses(self):
        cases = []
        truncated = fixture()
        struct.pack_into('<I', truncated, 0x98 + 112 + 8 + 4, 20)
        cases.append(truncated)
        unsafe = fixture()
        unsafe[0x300:0x30e] = b'../escape.dll\0\0'
        cases.append(unsafe)
        delay_va = fixture()
        struct.pack_into('<I', delay_va, 0x400, 0)
        cases.append(delay_va)
        bad_ordinal = fixture()
        struct.pack_into('<H', bad_ordinal, 0x740, 2)
        cases.append(bad_ordinal)
        for data in cases:
            with self.subTest(), self.assertRaises(ValueError):
                self.inspect(data)

    def test_reports_bound_imports_without_claiming_to_resolve_them(self):
        for offset, layout, values in [
                (0x98 + 112 + 8 * 11, '<II', (0x1900, 16)),
                (0x204, '<I', (123,)),  # ordinary import binding timestamp
                (0x414, '<I', (0x13a0,)),  # delay bound IAT without directory 11
                (0x41c, '<I', (123,))]:  # delay binding timestamp
            data = fixture()
            struct.pack_into(layout, data, offset, *values)
            with self.subTest(offset=offset):
                self.assertTrue(self.inspect(data)['bound_imports_present'])


if __name__ == '__main__':
    unittest.main()
