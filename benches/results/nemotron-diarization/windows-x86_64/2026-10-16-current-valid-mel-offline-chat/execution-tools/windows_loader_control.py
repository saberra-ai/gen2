"""Check default Windows loading in a fresh process without ctypes error rewriting."""
import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path


def default_loader_control(library):
    library = Path(library).resolve(strict=True)
    if not library.is_file():
        raise ValueError('loader control requires an existing DLL')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    load = kernel.LoadLibraryExW
    load.argtypes = [wintypes.LPCWSTR, wintypes.HANDLE, wintypes.DWORD]
    load.restype = wintypes.HMODULE
    release = kernel.FreeLibrary
    release.argtypes = [wintypes.HMODULE]
    release.restype = wintypes.BOOL
    ctypes.set_last_error(0)
    handle = load(str(library), None, 0)
    code = ctypes.get_last_error()
    if handle:
        release(handle)
        raise ValueError('default Windows loader unexpectedly resolved SDK dependencies; control invalid')
    if code != 126:  # ERROR_MOD_NOT_FOUND; every other failure invalidates this control.
        raise ValueError(f'default loader returned unexpected Win32 error {code}')
    return {'passed': True, 'library': str(library), 'load_flags': 0,
            'native_win32_error': code, 'default_loader_resolves_siblings': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('library', type=Path)
    args = parser.parse_args()
    print(json.dumps(default_loader_control(args.library), indent=2))


if __name__ == '__main__':
    main()
