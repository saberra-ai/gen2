"""Reuse the frozen bounded SDK build with only its source verifier replaced."""
import ast


def derive_build(raw):
    tree = ast.parse(raw)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Qualification']
    if len(classes) != 1:
        raise ValueError('one original bounded qualification class required')
    methods = [node for node in classes[0].body if isinstance(node, ast.FunctionDef) and node.name == 'build']
    if len(methods) != 1:
        raise ValueError('one original bounded build required')
    method, = methods
    body = b''.join(raw.splitlines(keepends=True)[method.lineno - 1:method.end_lineno])
    header = b'''"""Exact bounded SDK build mechanics; candidate source admission only."""
import json
import shutil
import sys

from platform_bounded_windows_native import (Qualification as BoundedQualification,
    msvc_compiler_definitions, msvc_probe_command, msvc_tool, powershell_literal,
    prepare_private_names, apply_windows_lf_compat, digest, audit_cpu_tree,
    require_single_test_pass, derive_native_probe, descriptor, verify_private_outputs)
from windows_q8_candidate_verify import verify_completed_source


class Qualification(BoundedQualification):
'''
    if body.count(b'verify_completed_source(') != 2:
        raise ValueError('two original bounded source checks required')
    return header + body.rstrip(b'\n') + b'\n'
