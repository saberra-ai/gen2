// Apple API usage adapted from the official macOS VM sample. See macos_vm_apple_LICENSE.txt.
import Darwin
import Foundation
import Virtualization

func errorChain(_ error: Error) -> [[String: Any]] {
    var result = [[String: Any]]()
    var pending: [(NSError, Int, String)] = [(error as NSError, -1, "root")]
    var visited = [ObjectIdentifier: Int]()
    while let (value, parent, relation) = pending.popLast() {
        var row: [String: Any] = ["parent_node": parent, "relation": relation]
        if let reference = visited[ObjectIdentifier(value)] {
            row["reference_node"] = reference
        } else {
            let index = result.count
            visited[ObjectIdentifier(value)] = index
            row["domain"] = value.domain
            row["code"] = value.code
            row["description"] = value.localizedDescription
            if let underlying = value.userInfo[NSUnderlyingErrorKey] as? NSError {
                pending.append((underlying, index, "underlying"))
            }
            if let multiple = value.userInfo[NSMultipleUnderlyingErrorsKey] as? [NSError] {
                for (offset, underlying) in multiple.enumerated() {
                    pending.append((underlying, index, "multiple[\(offset)]"))
                }
            }
        }
        result.append(row)
    }
    return result
}

enum BootFailure: Error { case invalidInput, invalidPlatform, unexpectedState }

func graceElapsed(now: TimeInterval, shutdownAt: TimeInterval, requestReturnedAt: TimeInterval?, allowance: TimeInterval) -> Bool {
    return now - (requestReturnedAt ?? shutdownAt) >= allowance
}

struct BootInput: Decodable {
    let seed: String
    let result: String
    let mac_address: String
    let run_seconds: Double
    let operation_seconds: Double
    let graceful_seconds: Double
    let cleanup_seconds: Double
}

final class BootOwner: NSObject, VZVirtualMachineDelegate {
    let input: BootInput
    let began = ProcessInfo.processInfo.systemUptime
    var machine: VZVirtualMachine?
    var timer: DispatchSourceTimer?
    var signals = [DispatchSourceSignal]()
    var startCalled = false
    var startActive = false
    var startCallbackReceived = false
    var startCallbackSuccess = false
    var startedAt: TimeInterval?
    var shutdownAt: TimeInterval?
    var cleanupDeadline: TimeInterval?
    var gracefulRequested = false
    var gracefulRequestedAt: TimeInterval?
    var guestStopped = false
    var forcedStopRequested = false
    var forcedStopCompleted = false
    var forcedStopSuccess = false
    var runWindowCompleted = false
    var cancelled = false
    var finished = false
    var loggingFailed = false
    var eventCount = 0
    var lastHeartbeat = -1
    var errors = [[[String: Any]]]()

    init(input: BootInput) { self.input = input; super.init() }

    func record(_ event: String, _ details: [String: Any] = [:]) {
        guard !finished else { return }
        eventCount += 1
        guard eventCount <= 20_000 else { loggingFailed = true; return }
        var row = details
        row["event"] = event
        row["elapsed_seconds"] = ProcessInfo.processInfo.systemUptime - began
        row["state"] = machine.map { $0.state.rawValue } ?? -1
        do {
            var data = try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys])
            data.append(10)
            try FileHandle.standardOutput.write(contentsOf: data)
        } catch { loggingFailed = true }
    }

    func finish(cleanupConfirmed: Bool) -> Never {
        let stoppedEvidence = forcedStopRequested ? (forcedStopCompleted && forcedStopSuccess) : guestStopped
        var success = startCallbackSuccess && runWindowCompleted && cleanupConfirmed && stoppedEvidence
            && !cancelled && !loggingFailed && errors.isEmpty
        record("terminal", ["success": success, "cleanup_confirmed": cleanupConfirmed])
        success = success && !loggingFailed
        finished = true
        timer?.cancel()
        let row: [String: Any] = ["schema_version": 1, "boot_owner_qualified": success,
            "virtual_machine_created": machine != nil, "normal_start_called": startCalled,
            "start_callback_received": startCallbackReceived, "start_callback_success": startCallbackSuccess,
            "run_window_completed": runWindowCompleted,
            "running_seconds": (shutdownAt ?? began) - (startedAt ?? began),
            "configured_run_seconds": input.run_seconds, "cancellation_requested": cancelled,
            "graceful_stop_requested": gracefulRequested, "guest_stop_observed": guestStopped,
            "forced_stop_requested": forcedStopRequested, "forced_stop_completed": forcedStopCompleted,
            "forced_stop_success": forcedStopSuccess, "cleanup_confirmed": cleanupConfirmed,
            "final_state": machine.map { $0.state.rawValue } ?? -1, "errors": errors,
            "logging_failed": loggingFailed, "event_count": eventCount,
            "guest_os_ready_qualified": false, "custom_listener_created": false, "display_window_created": false,
            "elapsed_seconds": ProcessInfo.processInfo.systemUptime - began]
        do {
            let url = URL(fileURLWithPath: input.result)
            try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys]).write(to: url, options: .atomic)
            let handle = try FileHandle(forWritingTo: url)
            try handle.synchronize()
            try handle.close()
            try FileHandle.standardOutput.synchronize()
        } catch { exit(1) }
        exit(success ? 0 : 1)
    }

    func cancel(_ reason: String) {
        if !cancelled { cancelled = true; record("cancellation_requested", ["reason": reason]) }
        beginShutdown()
    }

    func beginShutdown() {
        guard !finished else { return }
        if shutdownAt == nil {
            shutdownAt = ProcessInfo.processInfo.systemUptime
            cleanupDeadline = shutdownAt! + input.cleanup_seconds
            record("shutdown_began")
        }
        inspectShutdown()
    }

    func inspectShutdown() {
        guard !finished else { return }
        guard let machine = machine else { finish(cleanupConfirmed: true) }
        if startActive { return }
        if machine.state == .stopped {
            if !startCalled || !startCallbackSuccess { finish(cleanupConfirmed: true) }
            if forcedStopRequested {
                if forcedStopCompleted { finish(cleanupConfirmed: forcedStopSuccess) }
            } else if guestStopped { finish(cleanupConfirmed: true) }
            return
        }
        if machine.state == .error { finish(cleanupConfirmed: false) }
        guard shutdownAt != nil else { return }
        if !gracefulRequested && !forcedStopRequested && machine.canRequestStop {
            gracefulRequested = true
            record("graceful_stop_requested")
            do {
                try machine.requestStop()
                gracefulRequestedAt = ProcessInfo.processInfo.systemUptime
                record("graceful_stop_request_returned")
            }
            catch { errors.append(errorChain(error)); record("graceful_stop_error", ["error": errorChain(error)]) }
        }
        if graceElapsed(now: ProcessInfo.processInfo.systemUptime, shutdownAt: shutdownAt!,
                        requestReturnedAt: gracefulRequestedAt, allowance: input.graceful_seconds),
           !forcedStopRequested, (machine.state == .running || machine.state == .paused), machine.canStop {
            forcedStopRequested = true
            record("destructive_stop_requested")
            machine.stop(completionHandler: { [self] error in
                forcedStopCompleted = true
                forcedStopSuccess = error == nil
                if let error = error { errors.append(errorChain(error)) }
                record("destructive_stop_completion", ["success": forcedStopSuccess])
                if error != nil { finish(cleanupConfirmed: machine.state == .stopped) }
                inspectShutdown()
            })
        }
    }

    func tick() {
        guard !finished else { return }
        let now = ProcessInfo.processInfo.systemUptime
        if loggingFailed { cancel("diagnostic-write-failure") }
        if now - began >= input.operation_seconds { cancel("operation-deadline") }
        if let started = startedAt, shutdownAt == nil {
            guard machine?.state == .running else { cancel("unexpected-running-state"); return }
            let heartbeat = Int((now - started) / 10)
            if heartbeat != lastHeartbeat {
                lastHeartbeat = heartbeat
                record("running_observed", ["running_seconds": now - started])
            }
            if now - started >= input.run_seconds {
                runWindowCompleted = true
                record("run_window_completed", ["running_seconds": now - started])
                beginShutdown()
            }
        }
        if let deadline = cleanupDeadline, now >= deadline {
            record("cleanup_deadline_expired")
            finish(cleanupConfirmed: false)
        }
        if shutdownAt != nil { inspectShutdown() }
    }

    func begin() {
        dispatchPrecondition(condition: .onQueue(.main))
        do {
            guard input.run_seconds > 0, input.operation_seconds > input.run_seconds,
                  input.cleanup_seconds > input.graceful_seconds, input.graceful_seconds > 0 else { throw BootFailure.invalidInput }
            let root = URL(fileURLWithPath: input.seed, isDirectory: true)
            guard let hardware = VZMacHardwareModel(dataRepresentation: try Data(contentsOf: root.appendingPathComponent("hardware-model.bin"))), hardware.isSupported,
                  let identifier = VZMacMachineIdentifier(dataRepresentation: try Data(contentsOf: root.appendingPathComponent("machine-id.bin"))),
                  let address = VZMACAddress(string: input.mac_address) else { throw BootFailure.invalidPlatform }
            let platform = VZMacPlatformConfiguration()
            platform.hardwareModel = hardware
            platform.machineIdentifier = identifier
            platform.auxiliaryStorage = VZMacAuxiliaryStorage(url: root.appendingPathComponent("auxiliary-storage.bin"))
            let configuration = VZVirtualMachineConfiguration()
            configuration.platform = platform
            configuration.bootLoader = VZMacOSBootLoader()
            configuration.cpuCount = 4
            configuration.memorySize = 8 * 1024 * 1024 * 1024
            let graphics = VZMacGraphicsDeviceConfiguration()
            graphics.displays = [VZMacGraphicsDisplayConfiguration(widthInPixels: 1280, heightInPixels: 800, pixelsPerInch: 80)]
            configuration.graphicsDevices = [graphics]
            configuration.keyboards = [VZUSBKeyboardConfiguration()]
            configuration.pointingDevices = [VZUSBScreenCoordinatePointingDeviceConfiguration()]
            configuration.audioDevices = []
            let network = VZVirtioNetworkDeviceConfiguration()
            network.macAddress = address
            network.attachment = VZNATNetworkDeviceAttachment()
            configuration.networkDevices = [network]
            configuration.storageDevices = [VZVirtioBlockDeviceConfiguration(
                attachment: try VZDiskImageStorageDeviceAttachment(url: root.appendingPathComponent("disk.img"), readOnly: false))]
            try configuration.validate()
            record("configuration_validated", ["cpu_count": configuration.cpuCount, "memory_bytes": configuration.memorySize])
            guard !loggingFailed else { finish(cleanupConfirmed: true) }
            let virtualMachine = VZVirtualMachine(configuration: configuration)
            machine = virtualMachine
            virtualMachine.delegate = self
            record("virtual_machine_created")
            guard !loggingFailed else { finish(cleanupConfirmed: virtualMachine.state == .stopped) }
            guard virtualMachine.state == .stopped else { throw BootFailure.unexpectedState }
            for number in [SIGTERM, SIGINT] {
                signal(number, SIG_IGN)
                let source = DispatchSource.makeSignalSource(signal: number, queue: .main)
                source.setEventHandler { [self] in cancel("signal-\(number)") }
                source.resume()
                signals.append(source)
            }
            let clock = DispatchSource.makeTimerSource(queue: .main)
            clock.schedule(deadline: .now(), repeating: 1)
            clock.setEventHandler { [self] in tick() }
            timer = clock
            clock.resume()
            DispatchQueue.main.async { [self] in start() }
        } catch {
            errors.append(errorChain(error))
            record("pre_start_error", ["errors": errors])
            finish(cleanupConfirmed: machine == nil || machine?.state == .stopped)
        }
    }

    func start() {
        guard !cancelled, let machine = machine else { finish(cleanupConfirmed: machine?.state == .stopped) }
        guard !loggingFailed else { cancel("pre-start-log-failure"); return }
        record("normal_start_requested")
        guard !loggingFailed else { cancel("start-marker-write-failure"); return }
        startCalled = true
        startActive = true
        machine.start(completionHandler: { [self] result in
            startActive = false
            startCallbackReceived = true
            switch result {
            case .success: startCallbackSuccess = true; startedAt = ProcessInfo.processInfo.systemUptime
            case .failure(let error): errors.append(errorChain(error))
            }
            record("start_completion", ["success": startCallbackSuccess])
            if !startCallbackSuccess { cancel("start-failed") }
            else if cancelled || shutdownAt != nil { beginShutdown() }
        })
    }

    func guestDidStop(_ virtualMachine: VZVirtualMachine) {
        guestStopped = true
        record("guest_did_stop")
        if shutdownAt == nil { cancel("unexpected-guest-stop") }
        else { inspectShutdown() }
    }

    func virtualMachine(_ virtualMachine: VZVirtualMachine, didStopWithError error: Error) {
        errors.append(errorChain(error))
        record("delegate_stop_error", ["error": errorChain(error)])
        cancel("delegate-stop-error")
    }
}

if CommandLine.arguments.count == 2, CommandLine.arguments[1] == "shutdown-control" {
    let checks = [!graceElapsed(now: 40, shutdownAt: 0, requestReturnedAt: 40, allowance: 30),
        !graceElapsed(now: 69.9, shutdownAt: 0, requestReturnedAt: 40, allowance: 30),
        graceElapsed(now: 70, shutdownAt: 0, requestReturnedAt: 40, allowance: 30),
        !graceElapsed(now: 29.9, shutdownAt: 0, requestReturnedAt: nil, allowance: 30),
        graceElapsed(now: 30, shutdownAt: 0, requestReturnedAt: nil, allowance: 30)]
    guard checks.allSatisfy({ $0 }) else { exit(1) }
    try FileHandle.standardOutput.write(contentsOf: JSONSerialization.data(withJSONObject: ["checks": 5, "passed": true], options: [.sortedKeys]))
    exit(0)
}
if CommandLine.arguments.count == 2, CommandLine.arguments[1] == "error-control" {
    let shared = NSError(domain: "Gen2OwnerControl", code: 3)
    let branch = NSError(domain: "Gen2OwnerControl", code: 2, userInfo: [NSUnderlyingErrorKey: shared])
    let other = NSError(domain: "Gen2OwnerControl", code: 4)
    let root = NSError(domain: "Gen2OwnerControl", code: 1, userInfo: [NSUnderlyingErrorKey: branch, NSMultipleUnderlyingErrorsKey: [shared, other]])
    try FileHandle.standardOutput.write(contentsOf: JSONSerialization.data(withJSONObject: errorChain(root), options: [.sortedKeys]))
    exit(0)
}
guard CommandLine.arguments.count == 3, CommandLine.arguments[1] == "boot" else { exit(64) }
let input = try JSONDecoder().decode(BootInput.self, from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[2])))
let owner = BootOwner(input: input)
DispatchQueue.main.async { owner.begin() }
dispatchMain()
