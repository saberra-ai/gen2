import Foundation
import SwiftUI
import UIKit

private enum SmokeFailure: Error, CustomStringConvertible {
    case check(String)
    var description: String { switch self { case let .check(message): return message } }
}

// The Rust registry owns native resources. The lock only protects our handle ID;
// never hold it across inference, resume or shutdown.
private final class NativeRun: @unchecked Sendable {
    private let lock = NSLock()
    private var handle: UInt64?
    func set(_ value: UInt64?) { lock.lock(); handle = value; lock.unlock() }
    func suspend() {
        lock.lock(); let current = handle; lock.unlock()
        if let current { _ = try? LayaSmoke.suspend(handle: current) }
    }
    func execute(family: String) -> [String: Any] {
        var checks: [String] = []
        let started = Date()
        func require(_ valid: Bool, _ message: String) throws {
            guard valid else { throw SmokeFailure.check(message) }
            checks.append(message)
        }
        func json(_ value: Any) throws -> String {
            String(decoding: try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]), as: UTF8.self)
        }
        func envelope(_ text: String) throws -> [String: Any] {
            guard let object = try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any] else {
                throw SmokeFailure.check("Native response was not an object")
            }
            return object
        }
        func value(_ text: String) throws -> Any {
            let object = try envelope(text)
            guard object["ok"] as? Bool == true, let result = object["value"] else {
                throw SmokeFailure.check(String(describing: object["error"] ?? "Missing native result"))
            }
            return result
        }
        func semantic(_ result: [String: Any]) -> [String: Any] {
            var copy = result; copy.removeValue(forKey: "queue_micros"); copy.removeValue(forKey: "execution_micros")
            return copy
        }
        func equivalent(_ lhs: Any, _ rhs: Any) -> Bool {
            if let a = lhs as? [String: Any], let b = rhs as? [String: Any] {
                return Set(a.keys) == Set(b.keys) && a.allSatisfy { key, value in equivalent(value, b[key]!) }
            }
            if let a = lhs as? [Any], let b = rhs as? [Any] {
                return a.count == b.count && zip(a, b).allSatisfy { equivalent($0, $1) }
            }
            if let a = lhs as? NSNumber, let b = rhs as? NSNumber {
                return abs(a.doubleValue - b.doubleValue) <= 0.0001
            }
            if let a = lhs as? String, let b = rhs as? String { return a == b }
            return lhs is NSNull && rhs is NSNull
        }
        var report: [String: Any] = ["family": family, "synthetic": family == "synthetic",
            "physical_device": !Self.isSimulator, "application_quality_claim": false]
        do {
            let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let bundle = family == "synthetic"
                ? Bundle.main.bundleURL.appendingPathComponent("smoke")
                : documents.appendingPathComponent("laya").appendingPathComponent(family)
            let config: [String: Any] = ["bundle": bundle.path, "resident_budget_mb": family == "synthetic" ? 16 : 4096,
                "queue_capacity": 2, "intra_threads": 1, "max_input_bytes": 1048576]
            let opened = try value(LayaSmoke.open(configJSON: json(config))) as? [String: Any]
            guard let id = (opened?["handle"] as? NSNumber)?.uint64Value else { throw SmokeFailure.check("Missing handle") }
            set(id)
            defer { set(nil); _ = try? LayaSmoke.close(handle: id) }
            checks.append("open")
            func invoke(_ operation: String, _ fields: [String: Any] = [:]) throws -> Any {
                var payload = fields; payload["operation"] = operation
                return try value(LayaSmoke.invoke(handle: id, invocationJSON: json(payload)))
            }
            let described = try invoke("describe") as? [String: Any]
            let caps = described?["capabilities"] as? [String: Any]
            try require(caps?["multi_state_batch"] as? Bool == true && caps?["long_state_windows"] as? Bool == true, "capabilities")
            report["description"] = described
            let request: [String: Any] = ["state": ["kind": "text", "value": "Unicode 👋 café 返金. Please refund the duplicate payment."],
                "questions": [["refund", ["type": "yes_no", "instructions": "Is a refund requested?", "false_label": "false", "true_label": "true"]]]]
            guard let first = try invoke("decide", ["request": request]) as? [String: Any] else {
                throw SmokeFailure.check("Missing single result")
            }
            try require((first["answers"] as? [Any])?.count == 1, "Unicode single decision")
            let empty: [String: Any] = ["state": ["kind": "text", "value": ""], "questions": []]
            let batch = try invoke("batch", ["requests": [request, empty, request]]) as? [[String: Any]]
            try require(batch?.count == 3 && (batch?[1]["answers"] as? [Any])?.isEmpty == true, "batch ordering and empty state")
            try require(equivalent(semantic(batch![0]), semantic(first)) && equivalent(semantic(batch![2]), semantic(first)), "batch/single parity")
            var long = request; long["state"] = ["kind": "text", "value": String(repeating: "a ", count: 40)]
            let scan = try invoke("long", ["request": long, "scan": ["window_tokens": 10, "stride_tokens": 5, "max_windows": 32]]) as? [String: Any]
            try require(((scan?["windows"] as? [Any])?.count ?? 0) > 1, "multi-window decision")
            let expired = try envelope(LayaSmoke.invoke(handle: id, invocationJSON: json([
                "operation": "decide", "request": request, "options": ["timeout_ms": 0]])))
            try require(expired["ok"] as? Bool == false && (expired["error"] as? String)?.lowercased().contains("deadline") == true, "expired deadline rejected")
            _ = try value(LayaSmoke.suspend(handle: id))
            let suspended = try envelope(LayaSmoke.decide(handle: id, requestJSON: json(request)))
            try require(suspended["ok"] as? Bool == false, "suspended inference rejected")
            _ = try value(LayaSmoke.resume(handle: id))
            guard let resumed = try invoke("decide", ["request": request]) as? [String: Any] else {
                throw SmokeFailure.check("Missing resumed result")
            }
            try require(equivalent(semantic(first), semantic(resumed)), "resume parity")
            _ = try value(LayaSmoke.close(handle: id)); set(nil)
            let closed = try envelope(LayaSmoke.decide(handle: id, requestJSON: json(request)))
            try require(closed["ok"] as? Bool == false, "closed handle rejected")
            report["ok"] = true; report["single_result"] = first
        } catch {
            report["ok"] = false; report["error"] = String(describing: error)
        }
        report["checks"] = checks
        report["elapsed_seconds"] = Date().timeIntervalSince(started)
        report["os"] = ProcessInfo.processInfo.operatingSystemVersionString
        return report
    }
    private static var isSimulator: Bool {
        #if targetEnvironment(simulator)
        return true
        #else
        return false
        #endif
    }
}

@MainActor private final class SmokeViewModel: ObservableObject {
    @Published var family = "synthetic"
    @Published var running = false
    @Published var report = "Choose a bundle and run. The included synthetic graph checks integration only."
    private let native = NativeRun()
    func suspend() { native.suspend() }
    func run() {
        guard !running else { return }
        running = true; report = "Running offline on a background queue…"
        let selected = family
        let native = native
        DispatchQueue.global(qos: .userInitiated).async {
            let result = native.execute(family: selected)
            var text: String
            do {
                let data = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
                let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                try FileManager.default.createDirectory(at: documents, withIntermediateDirectories: true)
                try data.write(to: documents.appendingPathComponent("laya-smoke-report.json"), options: [.atomic])
                text = String(decoding: data, as: UTF8.self)
            } catch { text = "Could not save smoke report: \(error)" }
            let rendered = text
            DispatchQueue.main.async { self.report = rendered; self.running = false }
        }
    }
}

@main struct LayaSmokeApp: App {
    @StateObject private var model = SmokeViewModel()
    @Environment(\.scenePhase) private var phase
    var body: some Scene {
        WindowGroup {
            NavigationView {
                VStack(alignment: .leading, spacing: 16) {
                    Text("Offline Laya integration checks").font(.headline)
                    Picker("Bundle", selection: $model.family) {
                        ForEach(["synthetic", "english", "multilingual", "typed-decisions"], id: \.self) { Text($0) }
                    }.disabled(model.running)
                    Text("Real bundles go in Documents/laya/<family>. The included graph does not measure model quality.").font(.caption)
                    Button(model.running ? "Running…" : "Run checks") { model.run() }.disabled(model.running)
                    ScrollView { Text(model.report).font(.system(.caption, design: .monospaced)).textSelection(.enabled) }
                }.padding().navigationTitle("Gen2 Laya")
            }.task {
                if ProcessInfo.processInfo.arguments.contains("--ci-smoke") { model.run() }
            }.onReceive(NotificationCenter.default.publisher(for: UIApplication.didReceiveMemoryWarningNotification)) { _ in model.suspend() }
        }.onChange(of: phase) { newPhase in if newPhase == .background { model.suspend() } }
    }
}
