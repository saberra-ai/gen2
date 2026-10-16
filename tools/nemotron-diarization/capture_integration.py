"""Capture N1–N3 development evidence without claiming a supported desktop release."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import subprocess


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def validate_soak(result, time_output, seconds):
    assert result["complete"] is True
    assert result["accepted_samples"] == seconds * 16000
    assert result["audio_duration_seconds"] == seconds
    assert result["native_frame_count"] == seconds * 100 + 1
    retained = result["retained_probabilities"]
    assert retained["first_frame"] > 0, "native probability compaction did not occur"
    assert retained["first_frame"] + retained["rows"] == result["native_frame_count"]
    assert result["backend"] == "cpu"
    assert result["elapsed_micros"] > 0
    assert result["elapsed_micros"] / 1e6 / seconds < 1, "not faster than real time"
    assert len(result["turns"]) <= 100000
    previous = 0.
    for turn in result["turns"]:
        start, end = turn["start_seconds"], turn["end_seconds"]
        assert math.isfinite(start) and math.isfinite(end)
        assert previous <= start <= end <= seconds and 1 <= turn["speaker"] <= 8
        previous = start
    revision, position = 0, 0.
    for snapshot in result["snapshots"]:
        assert snapshot["revision"] > revision
        assert position <= snapshot["processed_audio_seconds"] <= seconds
        revision, position = snapshot["revision"], snapshot["processed_audio_seconds"]
    match = re.search(r"(\d+)\s+maximum resident set size", time_output)
    assert match, "missing actual macOS peak RSS"
    rss = int(match[1])
    assert rss <= 1024 * 1024 * 1024, "peak exceeds the admission reservation"
    return {"audio_seconds": seconds, "elapsed_seconds": result["elapsed_micros"] / 1e6,
            "real_time_factor": result["elapsed_micros"] / 1e6 / seconds,
            "peak_rss_bytes": rss, "turn_count": len(result["turns"]),
            "observed_provisional_revisions": len(result["snapshots"]),
            "retained_probabilities": retained, "native_frame_count": result["native_frame_count"]}


def require_pass(log, names):
    text = log.read_text()
    assert "test result: FAILED" not in text
    for name in names:
        assert re.search(rf"^test {re.escape(name)} \.\.\. ok$", text, re.M), f"missing executed passing test: {name}"
    return {"sha256": digest(log), "required_passed_tests": names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--stream-executable", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.evidence
    native = ["native_admission_cancellation_and_failed_load_release_reservations",
              "real_diarizer_matches_native_reference_and_resets_recordings",
              "repeated_reload_full_queue_cancel_and_dropped_stream_release_resources",
              "silence_short_tail_and_idle_deadline_have_explicit_terminal_results",
              "streamed_uneven_pcm_matches_recording_and_finish_is_idempotent",
              "unsupported_abi_and_missing_symbols_fail_before_ready",
              "verified_provisioned_cache_is_reused_offline"]
    logs = {"n3-native-final.log": require_pass(root / "n3-native-final.log", native)}
    for name, test in [("n2-mixed-chat-first.log", "mixed_chat_first"), ("n2-mixed-diar-first.log", "mixed_diarization_first"),
                       ("n2-mixed-real-laya-chat-first.log", "mixed_real_laya_chat_first"),
                       ("n2-mixed-real-laya-diar-first.log", "mixed_real_laya_diarization_first")]:
        logs[name] = require_pass(root / name, [test])
    soaks = []
    for minutes in [30, 60]:
        path = root / f"n3-soak-{minutes}m.json"
        time_path = path.with_suffix(".time")
        result = json.loads(path.read_text())
        summary = validate_soak(result, time_path.read_text(), minutes * 60)
        summary |= {"result_sha256": digest(path), "time_sha256": digest(time_path),
                    "geometry": result["geometry"], "runtime_version": result["runtime_version"],
                    "model_revision": result["model_revision"]}
        soaks.append(summary)
    provision = json.loads((root / "n2-provision-receipt.json").read_text())
    assert provision["offline_warm_reuse"] and provision["corrupt_entry_quarantined_and_repaired"]
    assert provision["interrupted_staging_ignored"] and len(provision["concurrent_cold_processes"]) == 3
    assert provision["resumed_after_process_termination"] and provision["interrupted_process_exit_code"] != 0
    assert provision["interrupted_download"]["bytes"] > 0
    project = Path(__file__).resolve().parents[2]
    sources = [project / "Cargo.toml", project / "src/api/runtime.rs", project / "src/api/error.rs",
               *sorted((project / "src/diarization").glob("*.rs")), *sorted((project / "tests").glob("diarization*.rs"))]
    result = {"schema_version": 1, "scope": "N1–N3 macOS CPU development qualification",
              "release_qualified": False, "machine": platform.platform(),
              "commit_at_collection": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip(),
              "source_hashes_at_collection": {str(p.relative_to(project)): digest(p) for p in sources},
              "stream_executable_sha256": digest(args.stream_executable),
              "stream_build_provenance": "Preserved development executable, prior to review allocation-capacity and diagnostic-message fixes; clean-commit release rerun remains required.",
              "workload": "Repeated 60-second licensed AMI clip, one native session; development builds/tests also ran on the host.",
              "logs": logs, "provisioning_receipt_sha256": digest(root / "n2-provision-receipt.json"),
              "soaks": soaks, "reservation_mb": 1024,
              "unmet_gates": ["N4 alignment and Pnola integration", "N5 native deployment and platform/backend release qualification", "held-out corpus quality and attribution gates", "clean committed release build performance/reload receipts"]}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
