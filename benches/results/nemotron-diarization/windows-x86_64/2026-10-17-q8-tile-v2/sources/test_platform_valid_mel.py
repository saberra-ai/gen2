"""Platform migration must reject stale compiler inputs and partial smoke output."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from platform_native import pinned_git_environment
from platform_valid_mel_native import compare_complete_60s
from platform_valid_mel_sources import audit_cpu_tree, verify_completed_source
from windows_compat import WRITE, rewrite_materializer
import test_stream_comparison_valid_mel as fixtures


def git(root, *args):
    return subprocess.check_output(['git', '-c', 'core.autocrlf=false', '-c', 'core.hooksPath=' + str(root.parent / 'empty-hooks'), *args], cwd=root, stderr=subprocess.PIPE)


def initialize(root):
    root.mkdir(parents=True)
    git(root, 'init', '-q')


def commit(root):
    git(root, 'add', '.')
    git(root, '-c', 'user.name=QualificationFixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture')
    return git(root, 'rev-parse', 'HEAD').decode().strip()


class PlatformValidMelTests(unittest.TestCase):
    def source_fixture(self, root):
        source = root / 'source'; initialize(source)
        llama = source / 'llama.cpp'; initialize(llama)
        (llama / 'tracked.h').write_bytes(b'pristine compiler input\n')
        llama_revision = commit(llama)
        (source / '.gitignore').write_text('build/\n')
        pipeline = source / 'src/asr/diar/diar_pipeline.cpp'; pipeline.parent.mkdir(parents=True)
        pipeline.write_bytes(b'original inference\n')
        cmake = source / 'cmake/llama_cpp.cmake'; cmake.parent.mkdir()
        original = WRITE + b'\n'; cmake.write_bytes(original)
        revision = commit(source)
        pipeline.write_bytes(b'valid Mel inference\n')
        recorded_patch = root / 'expected.patch'
        recorded_patch.write_bytes(git(source, 'diff', '--binary', '--no-ext-diff', '--', 'src/asr/diar/diar_pipeline.cpp'))
        build = source / 'build/cpu-diar'; build.mkdir(parents=True)
        return source, build, {'runtime': {'revision': revision, 'llama_revision': llama_revision}}, recorded_patch, cmake, original

    def test_composed_source_rejects_unrecorded_files_or_inference_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source, build, lock, recorded_patch, _, _ = self.source_fixture(root)
            verify_completed_source(source, build, lock, recorded_patch, False)
            extra = source / 'undeclared.h'; extra.write_bytes(b'extra')
            with self.assertRaisesRegex(ValueError, 'unrecorded platform'):
                verify_completed_source(source, build, lock, recorded_patch, False)
            extra.unlink()
            (source / 'src/asr/diar/diar_pipeline.cpp').write_bytes(b'other inference\n')
            with self.assertRaisesRegex(ValueError, 'inference patch changed'):
                verify_completed_source(source, build, lock, recorded_patch, False)

    def test_windows_accepts_only_exact_recorded_lf_overlay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source, build, lock, recorded_patch, cmake, original = self.source_fixture(root)
            cmake.write_bytes(rewrite_materializer(original))
            with patch('platform_valid_mel_sources.ORIGINAL_SHA256', hashlib.sha256(original).hexdigest()):
                verify_completed_source(source, build, lock, recorded_patch, True)
                cmake.write_bytes(cmake.read_bytes() + b'# unrecorded change\n')
                with self.assertRaisesRegex(ValueError, 'exact declared LF'):
                    verify_completed_source(source, build, lock, recorded_patch, True)

    def test_independent_patch_replay_rejects_stale_materialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); source = root / 'source'; initialize(source)
            llama = source / 'llama.cpp'; initialize(llama)
            code = llama / 'ggml/foo.c'; code.parent.mkdir(); code.write_bytes(b'int foo = 1;\n')
            revision = commit(llama); code.write_bytes(b'int foo = 2;\n')
            payload = git(llama, 'diff', '--binary', '--no-ext-diff')
            git(llama, 'checkout', '--', 'ggml/foo.c')
            patches = source / 'patches'; patches.mkdir()
            (patches / 'series').write_text('first.patch\n'); (patches / 'first.patch').write_bytes(payload)
            (source / '.gitignore').write_text('build/\n'); commit(source)
            build = source / 'build/cpu-diar'; materialized = build / '_deps/llama.cpp'
            actual = materialized / 'ggml/foo.c'; actual.parent.mkdir(parents=True); actual.write_bytes(b'int foo = 2;\n')
            (build / 'CMakeCache.txt').write_text('NEMO_SPEECH_GGML_PATCHED:BOOL=ON\nNEMO_SPEECH_LLAMA_CPP_SOURCE_DIR:PATH=\nggml_SOURCE_DIR:STATIC=' + str(materialized / 'ggml') + '\n')
            (build / 'compile_commands.json').write_text(json.dumps([{'file': str(actual)}]))
            def run(name, argv, cwd, overrides=None):
                subprocess.run(list(map(str, argv)), cwd=cwd, env={**pinned_git_environment(dict(os.environ)), **(overrides or {})}, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output = root / 'first'; output.mkdir()
            audit = audit_cpu_tree(source, build, output, revision, run)
            self.assertEqual(len(audit['patch_series']), 1)
            self.assertIn('ggml/foo.c', audit['actual_compiler_sources'])
            self.assertEqual(len(audit['all_compiler_sources']), 1)
            for index, foreign in enumerate([root / 'foreign/ggml-cpu.c', build / 'generated.cpp']):
                foreign.parent.mkdir(parents=True, exist_ok=True); foreign.write_bytes(b'int unbound;\n')
                (build / 'compile_commands.json').write_text(json.dumps([{'file': str(actual)}, {'file': str(foreign)}]))
                rejected = root / ('unbound-' + str(index)); rejected.mkdir()
                with self.assertRaisesRegex(ValueError, 'unbound platform compiler'):
                    audit_cpu_tree(source, build, rejected, revision, run)
            (build / 'compile_commands.json').write_text(json.dumps([{'file': str(actual)}]))
            actual.write_bytes(b'int foo = 1;\n')
            output = root / 'second'; output.mkdir()
            with self.assertRaisesRegex(ValueError, 'independent patch replay'):
                audit_cpu_tree(source, build, output, revision, run)

    def test_complete_fixture_cannot_be_replaced_by_matching_truncated_tails(self):
        public, native, _ = fixtures.ValidMelReference().fixture()
        raw = struct.pack('<f', .5) * 6000 * 8
        public.update(accepted_samples=960000, audio_duration_seconds=60, native_frame_count=6000,
            retained_probabilities={'first_frame': 0, 'rows': 6000}, retained_probability_sha256=hashlib.sha256(raw).hexdigest())
        native.update(audio_samples=960000, audio_duration_seconds=60, native_frame_count=6000,
            retained_first_frame=0, retained_rows=6000)
        self.assertTrue(compare_complete_60s(public, native, raw, raw)['passed'])
        public['retained_probabilities'] = {'first_frame': 5900, 'rows': 100}
        native.update(retained_first_frame=5900, retained_rows=100)
        short = raw[-3200:]; public['retained_probability_sha256'] = hashlib.sha256(short).hexdigest()
        with self.assertRaisesRegex(ValueError, 'complete6000row'):
            compare_complete_60s(public, native, short, short)


if __name__ == '__main__':
    unittest.main()
