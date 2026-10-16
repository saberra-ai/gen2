"""Bounded, read-only AMD64 PE dependency inspection for packaging evidence.

Implements the ordinary import, RVA delay-import and export-forwarder tables
described by Microsoft's PE/COFF specification. It never loads or edits an image.
Bound imports are reported, not resolved; packaging must reject unsupported
forms. Dynamic LoadLibrary calls, signatures and Windows API-set resolution
require separate runtime/provenance checks.
https://learn.microsoft.com/en-us/windows/win32/debug/pe-format
"""
import hashlib
import re
import struct


MAX_IMAGE_BYTES = 256 * 1024 * 1024
MAX_EXPORTS = 65536


def image_name(value, extension):
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.+-]*\.' + extension, value, re.IGNORECASE):
        raise ValueError('unsafe or unsupported PE ' + extension + ' name: ' + repr(value))
    return value.lower()


class PeImage:
    def __init__(self, data):
        self.data = data
        self.descriptor_bindings = False
        if len(data) > MAX_IMAGE_BYTES or len(data) < 64 or data[:2] != b'MZ':
            raise ValueError('expected bounded DOS/PE image')
        pe = self.unpack('<I', 0x3c)[0]
        if pe < 64 or self.slice(pe, 4) != b'PE\0\0':
            raise ValueError('invalid PE signature')
        machine, sections, _, _, _, optional_size, characteristics = self.unpack('<HHIIIHH', pe + 4)
        if machine != 0x8664 or not 1 <= sections <= 96 or not characteristics & 2:
            raise ValueError('expected executable AMD64 PE image')
        self.is_dll = bool(characteristics & 0x2000)
        optional = pe + 24
        self.slice(optional, optional_size)
        if optional_size < 112 or self.unpack('<H', optional)[0] != 0x20b:
            raise ValueError('expected PE32+ optional header')
        self.image_size, self.headers_size = self.unpack('<II', optional + 56)
        count = self.unpack('<I', optional + 108)[0]
        section_table = optional + optional_size
        if count > 16 or optional_size < 112 + count * 8 or not section_table + sections * 40 <= self.headers_size <= len(data):
            raise ValueError('invalid PE directory/header bounds')
        self.directories = [self.unpack('<II', optional + 112 + index * 8) for index in range(count)]
        self.sections = []
        raw_ranges = [(0, self.headers_size)]
        virtual_ranges = [(0, self.headers_size)]
        for index in range(sections):
            record = section_table + index * 40
            virtual_size, rva, raw_size, offset = self.unpack('<IIII', record + 8)
            size = max(virtual_size, raw_size)
            if not size or rva + size > self.image_size:
                raise ValueError('invalid PE section virtual bounds')
            if any(rva < end and start < rva + size for start, end in virtual_ranges):
                raise ValueError('ambiguous overlapping PE virtual sections')
            virtual_ranges.append((rva, rva + size))
            if raw_size:
                self.slice(offset, raw_size)
                if any(offset < end and start < offset + raw_size for start, end in raw_ranges):
                    raise ValueError('ambiguous overlapping PE raw sections')
                raw_ranges.append((offset, offset + raw_size))
            self.sections.append((rva, raw_size, offset))

    def slice(self, offset, size):
        if offset < 0 or size < 0 or offset + size > len(self.data):
            raise ValueError('PE data extends beyond file')
        return self.data[offset:offset + size]

    def unpack(self, layout, offset):
        return struct.unpack(layout, self.slice(offset, struct.calcsize(layout)))

    def offset(self, rva, size):
        if rva < 0 or size < 0 or rva + size > self.image_size:
            raise ValueError('PE RVA extends beyond image')
        if rva + size <= self.headers_size:
            return rva
        for start, raw_size, offset in self.sections:
            if start <= rva and rva + size <= start + raw_size:
                return offset + rva - start
        raise ValueError('PE RVA is unmapped or requires zero-filled data')

    def rva_unpack(self, layout, rva):
        return self.unpack(layout, self.offset(rva, struct.calcsize(layout)))

    def string(self, rva, limit=None):
        result = bytearray()
        for index in range(4096):
            if limit is not None and rva + index >= limit:
                break
            byte = self.data[self.offset(rva + index, 1)]
            if byte == 0:
                try:
                    value = result.decode('ascii')
                except UnicodeDecodeError as error:
                    raise ValueError('non-ASCII PE dependency/export string') from error
                if not value:
                    raise ValueError('empty PE dependency/export string')
                return value
            result.append(byte)
        raise ValueError('unterminated or oversized PE string')

    def directory(self, index):
        rva, size = self.directories[index] if index < len(self.directories) else (0, 0)
        if bool(rva) != bool(size):
            raise ValueError('incomplete PE data directory')
        if rva:
            self.offset(rva, size)
        return rva, size

    def imports(self, delay=False):
        rva, size = self.directory(13 if delay else 1)
        if not rva:
            return []
        width = 32 if delay else 20
        if size < width or size // width > 4096:
            raise ValueError('invalid PE import directory length')
        names = []
        for index in range(size // width):
            record = self.rva_unpack('<8I' if delay else '<5I', rva + index * width)
            if not any(record):
                return sorted(set(names))
            if delay and record[0] != 1:
                raise ValueError('only RVA delay-import descriptors are supported')
            if (delay and (record[5] or record[7])) or (not delay and record[1]):
                self.descriptor_bindings = True
            name = record[1] if delay else record[3]
            names.append(image_name(self.string(name), 'dll'))
        raise ValueError('unterminated PE import directory')

    def exports(self):
        rva, size = self.directory(0)
        if not rva:
            return {'export_name': None, 'exports': [], 'export_count': 0, 'forwarders': [], 'forwarder_dependencies': []}
        if size < 40:
            raise ValueError('incomplete PE export directory')
        _, _, _, _, name, base, functions, names, addresses, strings, ordinals = self.rva_unpack('<IIHHIIIIIII', rva)
        if functions > MAX_EXPORTS or names > functions:
            raise ValueError('oversized or inconsistent PE export table')
        self.offset(addresses, functions * 4)
        self.offset(strings, names * 4)
        self.offset(ordinals, names * 2)
        exports = []
        for index in range(names):
            ordinal = self.rva_unpack('<H', ordinals + index * 2)[0]
            if ordinal >= functions:
                raise ValueError('PE export ordinal exceeds address table')
            exports.append(self.string(self.rva_unpack('<I', strings + index * 4)[0]))
        forwarders, dependencies = [], set()
        for index in range(functions):
            address = self.rva_unpack('<I', addresses + index * 4)[0]
            if rva <= address < rva + size:
                value = self.string(address, rva + size)
                module, separator, symbol = value.rpartition('.')
                if not separator or not re.fullmatch(r'(?:#[0-9]+|[A-Za-z_?@$][A-Za-z0-9_?@$]*)', symbol):
                    raise ValueError('unsupported PE export forwarder: ' + repr(value))
                dependencies.add(image_name(module if module.lower().endswith('.dll') else module + '.dll', 'dll'))
                forwarders.append({'ordinal': base + index, 'target': value})
        return {'export_name': image_name(self.string(name), 'dll' if self.is_dll else 'exe'), 'exports': sorted(exports),
                'export_count': functions, 'forwarders': forwarders, 'forwarder_dependencies': sorted(dependencies)}


def inspect_pe(path):
    if path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError('PE file exceeds inspection limit')
    data = path.read_bytes()
    image = PeImage(data)
    imports, delay_imports = image.imports(), image.imports(delay=True)
    return {'machine': 'AMD64', 'format': 'PE32+', 'sha256': hashlib.sha256(data).hexdigest(),
            'bytes': len(data), 'is_dll': image.is_dll,
            'imports': imports, 'delay_imports': delay_imports,
            'bound_imports_present': bool(image.directory(11)[0]) or image.descriptor_bindings, **image.exports()}
