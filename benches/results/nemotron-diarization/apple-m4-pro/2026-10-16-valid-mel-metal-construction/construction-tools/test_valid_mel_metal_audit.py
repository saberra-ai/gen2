"""Shader provenance must bind physical sources, flattening and installed bytes."""
from pathlib import Path
import struct
import tempfile
import unittest

from valid_mel_metal_audit import embedded_payloads, pristine_files, shader_text


def one_payload_dylib(payload, escape=False):
    address, position = 0x1000, 32 + 72 + 80 + 24
    names = b'\0_ggml_metallib_test_start\0_ggml_metallib_test_end\0'
    symbol_offset = position + len(payload)
    strings_offset = symbol_offset + 32
    header = struct.pack('<8I', 0xFEEDFACF, 0x0100000C, 0, 6, 2, 176, 0, 0)
    segment = struct.pack('<II16sQQQQiiII', 0x19, 152, b'__DATA', address, len(payload),
                          position, len(payload), 7, 3, 1, 0)
    section = struct.pack('<16s16sQQIIIIIIII', b'__ggml_metallib', b'__DATA', address,
                          len(payload), position, 0, 0, 0, 0, 0, 0, 0)
    symtab = struct.pack('<6I', 2, 24, symbol_offset, 2, strings_offset, len(names))
    symbols = struct.pack('<IBBHQ', 1, 0x0F, 1, 0, address)
    symbols += struct.pack('<IBBHQ', names.index(b'_ggml_metallib_test_end'), 0x0F, 1, 0,
                           address + len(payload) + (1000 if escape else 0))
    return header + segment + section + symtab + payload + symbols + names


class MetalCompilerAuditTests(unittest.TestCase):
    def test_flattening_matches_order_and_replaces_includes_after_filtering(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / 'metal'
            (root / 'kernels').mkdir(parents=True)
            (root / 'kernels/common.h').write_bytes(b'#pragma once\n#include "ggml-metal-impl.h"\nCOMMON\n')
            (root / 'kernels/dequantize.h').write_bytes(b'__embed_ggml-common.h__\nDEQUANTIZE\n')
            (root / 'kernels/test.metal').write_bytes(b'#include "common.h"\n#include "dequantize.h"\nKERNEL\n')
            (parent / 'ggml-common.h').write_bytes(b'TYPES\n')
            (root / 'ggml-metal-impl.h').write_bytes(b'#pragma once\nIMPLEMENTATION\n')
            self.assertEqual(shader_text(root, 'test'), b'#pragma once\nIMPLEMENTATION\nCOMMON\nTYPES\nDEQUANTIZE\nKERNEL\n')
            (root / 'kernels/test.metal').write_bytes(b'no newline')
            with self.assertRaises(ValueError):
                shader_text(root, 'test')

    def test_installed_payload_symbols_must_bound_exact_source_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'metal.dylib'
            payload = b'kernel source\n'
            path.write_bytes(one_payload_dylib(payload))
            record = embedded_payloads(path, {'test': payload})
            self.assertEqual(record['test']['bytes'], len(payload))
            with self.assertRaisesRegex(ValueError, 'payload differs'):
                embedded_payloads(path, {'test': b'mutated source'})
            with self.assertRaisesRegex(ValueError, 'symbol set'):
                embedded_payloads(path, {'wrong': payload})
            for data in [one_payload_dylib(payload, True), b'wrong Mach-O', one_payload_dylib(payload)[:-1]]:
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    embedded_payloads(path, {'test': payload})

    def test_pristine_inventory_rejects_symlinks_and_tracks_mode_and_extra_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / '.git').mkdir()
            (root / '.git/metadata').write_bytes(b'not a compiler input')
            path = root / 'source.cpp'
            path.write_bytes(b'actual source')
            before = pristine_files(root)
            self.assertEqual(set(before), {'source.cpp'})
            path.chmod(0o755)
            self.assertNotEqual(pristine_files(root), before)
            (root / 'extra.cpp').write_bytes(b'could enter a glob')
            self.assertEqual(set(pristine_files(root)), {'source.cpp', 'extra.cpp'})
            (root / 'alias').symlink_to(path)
            with self.assertRaises(ValueError):
                pristine_files(root)


if __name__ == '__main__':
    unittest.main()
