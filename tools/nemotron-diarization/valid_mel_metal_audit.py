"""Audit pristine Metal compiler inputs and actual embedded shader text bytes."""
import json
import os
from pathlib import Path
import re
import struct
import tarfile
import tempfile

from build_valid_mel_candidate import descriptor


KINDS = ('fa', 'mul_mv', 'mul_mm', 'quantize', 'softmax', 'norm', 'unary', 'binbcast', 'reduce',
         'tri', 'ssm', 'wkv', 'gated_delta_net', 'solve_tri', 'rope', 'conv', 'upscale', 'argsort', 'pool', 'misc')


def pristine_files(root):
    root = Path(root)
    result = {}
    for directory, folders, files in os.walk(root):
        if Path(directory) == root:
            folders[:] = [name for name in folders if name != '.git']
        for name in folders:
            if (Path(directory) / name).is_symlink():
                raise ValueError('symlink in pristine Metal compiler tree')
        for name in files:
            path = Path(directory) / name
            if path == root / '.git':
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError('non-regular Metal compiler input')
            result[str(path.relative_to(root))] = {**descriptor(path),
                'executable': bool(path.stat().st_mode & 0o111)}
    return result


def audit_pristine_llama(source, output, revision, run):
    root = Path(source) / 'llama.cpp'
    output = Path(output)
    archive = output / 'pristine-metal-llama.tar'
    run('pristine-llama-archive', ['git', 'archive', '--format=tar', '-o', str(archive), revision], root)
    with tempfile.TemporaryDirectory(prefix='metal-llama-replay-', dir=output) as temporary:
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            if any(not (item.isfile() or item.isdir()) for item in members):
                raise ValueError('non-regular content in pristine Metal source archive')
            tar.extractall(temporary, filter='data')
        expected = pristine_files(temporary)
    actual = pristine_files(root)
    if actual != expected:
        raise ValueError('physical Metal compiler tree differs from independent Git archive')
    record = {'revision': revision, 'source_path': str(root), 'archive': descriptor(archive),
              'files': actual, 'source_is_pristine_submodule': True,
              'cpu_ggml_patches_applied': False, 'independent_archive_match': True}
    (output / 'pristine-metal-llama-audit.json').write_text(json.dumps(record, indent=2) + '\n')
    return actual


def shader_text(source, kind):
    """Independently reproduce the pinned CMake cat/sed flattening recipe."""
    root = Path(source)
    kernel = (root / 'kernels' / (kind + '.metal')).read_bytes()
    paths = [root / 'kernels/common.h']
    for name in ['dequantize.h', 'quantize.h']:
        if ('#include "' + name + '"').encode() in kernel:
            paths.append(root / 'kernels' / name)
    paths.append(root / 'kernels' / (kind + '.metal'))
    chunks = [path.read_bytes() for path in paths]
    if any(not data.endswith(b'\n') for data in chunks):
        raise ValueError('requalify shader flattening for missing terminal newline')
    removed = [b'#include "common.h"', b'#include "dequantize.h"',
               b'#include "quantize.h"', b'#pragma once']
    lines = [line for line in b''.join(chunks).splitlines(keepends=True)
             if not any(pattern in line for pattern in removed)]
    common = (root.parent / 'ggml-common.h').read_bytes()
    implementation = (root / 'ggml-metal-impl.h').read_bytes()
    stage = [common if b'__embed_ggml-common.h__' in line else line for line in lines]
    return b''.join(implementation if b'#include "ggml-metal-impl.h"' in line else line
                    for line in b''.join(stage).splitlines(keepends=True))


def embedded_payloads(binary, payloads):
    """Read Mach-O symbols/sections directly; never dlopen or execute the library."""
    data = Path(binary).read_bytes()
    if len(data) < 32 or struct.unpack_from('<I', data)[0] != 0xFEEDFACF:
        raise ValueError('expected little-endian 64-bit Mach-O')
    _, cpu, _, filetype, commands, command_bytes, _, _ = struct.unpack_from('<8I', data)
    if cpu != 0x0100000C or filetype != 6 or 32 + command_bytes > len(data):
        raise ValueError('expected a complete arm64 Mach-O dylib')
    offset, section_index, section, symtab = 32, 0, None, None
    for _ in range(commands):
        if offset + 8 > 32 + command_bytes:
            raise ValueError('truncated Mach-O load commands')
        command, size = struct.unpack_from('<II', data, offset)
        if size < 8 or offset + size > 32 + command_bytes:
            raise ValueError('invalid Mach-O load command size')
        if command == 0x19:
            if size < 72:
                raise ValueError('truncated Mach-O segment')
            sections = struct.unpack_from('<I', data, offset + 64)[0]
            if 72 + sections * 80 > size:
                raise ValueError('truncated Mach-O section table')
            for i in range(sections):
                section_index += 1
                base = offset + 72 + i * 80
                name, segment, address, length, position = struct.unpack_from('<16s16sQQI', data, base)
                if name.rstrip(b'\0') == b'__ggml_metallib' and segment.rstrip(b'\0') == b'__DATA':
                    if section is not None or position + length > len(data):
                        raise ValueError('ambiguous or truncated embedded Metal section')
                    section = (section_index, address, length, position)
        elif command == 2:
            if size != 24 or symtab is not None:
                raise ValueError('ambiguous Mach-O symbol table')
            symtab = struct.unpack_from('<4I', data, offset + 8)
        offset += size
    if offset != 32 + command_bytes or section is None or symtab is None:
        raise ValueError('missing embedded Metal section or symbol table')
    symbol_offset, count, string_offset, string_size = symtab
    if symbol_offset + count * 16 > len(data) or string_offset + string_size > len(data):
        raise ValueError('truncated Mach-O symbols or strings')
    index, address, length, position = section
    strings, symbols = data[string_offset:string_offset + string_size], {}
    for i in range(count):
        name_offset, flags, ordinal, _, value = struct.unpack_from('<IBBHQ', data, symbol_offset + i * 16)
        if name_offset >= string_size:
            raise ValueError('invalid Mach-O symbol name offset')
        end = strings.find(b'\0', name_offset)
        if end < 0:
            raise ValueError('unterminated Mach-O symbol name')
        name = strings[name_offset:end].decode('utf-8')
        if name.startswith('_ggml_metallib_'):
            if name in symbols or flags & 0x0E != 0x0E or ordinal != index:
                raise ValueError('invalid embedded Metal payload symbol')
            symbols[name] = value
    expected = {'_ggml_metallib_' + kind + '_' + bound for kind in payloads for bound in ['start', 'end']}
    if set(symbols) != expected:
        raise ValueError('embedded Metal payload symbol set differs')
    records = {}
    for kind, payload in payloads.items():
        prefix = '_ggml_metallib_' + kind + '_'
        start, end = symbols[prefix + 'start'], symbols[prefix + 'end']
        if not address <= start < end <= address + length:
            raise ValueError('embedded Metal payload escapes its section')
        actual = data[position + start - address:position + end - address]
        if actual != payload:
            raise ValueError('installed embedded Metal payload differs: ' + kind)
        records[kind] = {'start_symbol_address': start, 'end_symbol_address': end,
                         'bytes': len(actual), 'file_offset': position + start - address,
                         'independent_source_bytes_match': True}
    return records


def audit_shaders(source, build, library, output):
    root = Path(source) / 'llama.cpp/ggml/src/ggml-metal'
    generated = Path(build) / 'ggml/src/ggml-metal/autogenerated'
    cmake = (root / 'CMakeLists.txt').read_text()
    selection = re.search(r'set\(METALLIB_KERNEL_SOURCES\s+(.*?)\n\)', cmake, re.DOTALL)
    if not selection or re.findall(r'kernels/([a-z_]+)\.metal', selection[1]) != list(KINDS):
        raise ValueError('pinned Metal shader selection differs')
    payloads, files = {}, {}
    target = Path(output) / 'metal-payloads'
    target.mkdir()
    for kind in KINDS:
        path = generated / ('ggml-metal-embed-' + kind + '.metal')
        assembly = path.with_suffix('.s')
        payload = shader_text(root, kind)
        if path.read_bytes() != payload:
            raise ValueError('generated Metal text differs from independent flattening: ' + kind)
        symbol = kind.replace('-', '_')
        expected = ('.section __DATA,__ggml_metallib\n.globl _ggml_metallib_' + symbol + '_start\n'
            + '_ggml_metallib_' + symbol + '_start:\n.incbin "' + str(path) + '"\n'
            + '.globl _ggml_metallib_' + symbol + '_end\n_ggml_metallib_' + symbol + '_end:\n').encode()
        if assembly.read_bytes() != expected:
            raise ValueError('Metal embedding assembly differs: ' + kind)
        for member in [path, assembly]:
            (target / member.name).write_bytes(member.read_bytes())
            files[member.name] = descriptor(member)
        payloads[symbol] = payload
    record = {'shader_text_bundles': len(payloads), 'precompiled_gpu_machine_code': False,
              'runtime_compiles_embedded_source': True, 'generated_files': files,
              'installed_library': descriptor(Path(library)),
              'actual_installed_payloads': embedded_payloads(library, payloads)}
    (Path(output) / 'metal-shader-audit.json').write_text(json.dumps(record, indent=2) + '\n')
    return record
