"""Exercise receipt acceptance through the command-line boundary."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


COLLECTOR = Path(__file__).with_name("capture_receipts.py").resolve()


class ReceiptAcceptance(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="gen2-bark-receipts-")
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name)
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run([
            "git", "-c", "user.name=Gen2 Verification", "-c",
            "user.email=verification@example.invalid", "-c", "commit.gpgsign=false",
            "commit", "--allow-empty", "-qm", "Receipt test fixture",
        ], cwd=self.repo, check=True)
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.repo, text=True).strip()
        self.root = self.repo / "verification"

        def write(relative, content):
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

        for path in [
            "Cargo.toml", "Cargo.lock", "build.rs", "tools/bark/reference.lock",
            "tools/bark/runtime123.lock", "src/memory/runtime_memory.rs",
            "src/onnx_runtime.rs", "src/api/runtime.rs", "assets/bark/nemo128.f32",
            "tests/fixtures/bark/smoke.json", "tools/bark/critical-content.json",
            "runtime.dylib", "verification/chat/Qwen3-0.6B-Q4_K_M.gguf",
        ]:
            write(path, "Fixture bytes, not model/runtime evidence\n")
        for path in [
            "oracle124/reference.json", "native.json", "audio/audio-reference.json",
            "audio/native-audio.json", "audio/normalization-errors.json",
            "performance-comparison.json", "audio-quality.json", "qualification.json",
            "performance.json", "mixed-performance.json",
        ]:
            write(f"verification/{path}", json.dumps({"git_sha": sha, "passed": True}))
        compatibility = {"passed": True, "records": [
            {"pcm_equal": True, "text_equal": True, "tokens_equal": True}
            for _ in range(25)
        ]}
        write("verification/runtime-compatibility.json", json.dumps(compatibility))
        logs = {
            "numeric.log": [
                "speech::frontend::tests::native_frontend_matches_pinned_numpy_reference",
                "speech::onnx::tests::native_graphs_match_independent_encoder_decoder_tensors",
            ],
            "integration.log": [
                "real_bark_matches_reference_and_reuses_resident_weights",
                "real_audio_workflow_matches_qualified_reference",
                "real_bundle_integrity_and_warm_offline_cache",
                "real_worker_bounds_queue_cancellation_shutdown_and_admission",
                "coexistence_laya_first", "coexistence_speech_first",
            ],
            "runtime-rejection-green.log": [
                "incompatible_runtime_is_rejected_before_binding_initialization",
            ],
            "live-memory-green.log": [
                "memory::runtime_memory::tests::snapshot_tracks_released_pages_not_lifetime_peak",
            ],
        }
        for name, tests in logs.items():
            write(f"verification/{name}", "\n".join(f"test {test} ... ok" for test in tests)
                  + f"\ntest result: ok. {len(tests)} passed; 0 failed; 0 ignored\n")
        write("verification/provisioning.log",
              "test real_provisioner_recovers_incomplete_and_corrupt_entries ... ok\n"
              "test result: ok. 1 passed; 0 failed; 0 ignored\n")

    def collect(self):
        return subprocess.run([
            sys.executable, str(COLLECTOR), "--root", str(self.root),
            "--library", str(self.repo / "runtime.dylib"),
            "--output", str(self.repo / "receipts"),
        ], cwd=self.repo, text=True, capture_output=True)

    def test_skipped_runtime_regression_cannot_qualify_receipts(self):
        (self.root / "runtime-rejection-green.log").write_text(
            "test incompatible_runtime_is_rejected_before_binding_initialization ... ignored\n"
            "test result: ok. 0 passed; 0 failed; 1 ignored\n"
        )
        result = self.collect()
        self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_executed_named_gates_qualify_receipts(self):
        result = self.collect()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.repo / "receipts/provenance.json").is_file())

    def test_all_filtered_run_cannot_qualify_receipts(self):
        (self.root / "live-memory-green.log").write_text(
            "test result: ok. 0 passed; 0 failed; 0 ignored; 1 filtered out\n"
        )
        result = self.collect()
        self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_unrelated_passing_test_cannot_replace_required_gate(self):
        (self.root / "runtime-rejection-green.log").write_text(
            "test unrelated_test ... ok\n"
            "test result: ok. 1 passed; 0 failed; 0 ignored\n"
        )
        result = self.collect()
        self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_zero_count_cannot_qualify_a_named_pass_line(self):
        (self.root / "runtime-rejection-green.log").write_text(
            "test incompatible_runtime_is_rejected_before_binding_initialization ... ok\n"
            "test result: ok. 0 passed; 0 failed; 0 ignored\n"
        )
        result = self.collect()
        self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_failed_invocation_after_a_pass_cannot_qualify_receipts(self):
        path = self.root / "runtime-rejection-green.log"
        path.write_text(path.read_text() + "test result: FAILED. 0 passed; 1 failed\n")
        result = self.collect()
        self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_skipped_provisioning_cannot_qualify_receipts(self):
        (self.root / "provisioning.log").write_text(
            "test real_provisioner_recovers_incomplete_and_corrupt_entries ... ignored\n"
            "test result: ok. 0 passed; 0 failed; 1 ignored\n"
        )
        result = self.collect()
        self.assertNotEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
