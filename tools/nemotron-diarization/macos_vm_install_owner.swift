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

enum OwnerFailure: Error {
    case invalidConfiguration
    case invalidPlatform
    case initialStateNotStopped
}

struct OwnerInput: Decodable {
    let seed: String
    let image: String
    let result: String
    let mac_address: String
    let operation_seconds: Double
    let cleanup_seconds: Double
}

final class InstallOwner: NSObject, VZVirtualMachineDelegate {
    let input: OwnerInput
    let began = ProcessInfo.processInfo.systemUptime
    var machine: VZVirtualMachine?
    var installer: VZMacOSInstaller?
    var timer: DispatchSourceTimer?
    var signals = [DispatchSourceSignal]()
    var installStarted = false
    var installActive = false
    var callbackReceived = false
    var callbackSuccess = false
    var cancelled = false
    var stopRequested = false
    var stopCallbackReceived = false
    var stopCallbackSuccess = false
    var cleanupDeadline: TimeInterval?
    var finished = false
    var loggingFailed = false
    var eventCount = 0
    var lastState = -1
    var lastProgress = -1
    var errors = [[[String: Any]]]()
    var configurationValidated = false
    var actualCPU = 0
    var actualMemory: UInt64 = 0

    init(input: OwnerInput) {
        self.input = input
        super.init()
    }

    func record(_ event: String, _ details: [String: Any] = [:]) {
        guard !finished else { return }
        eventCount += 1
        guard eventCount <= 20_000 else { loggingFailed = true; return }
        var value = details
        value["event"] = event
        value["elapsed_seconds"] = ProcessInfo.processInfo.systemUptime - began
        value["state"] = machine.map { $0.state.rawValue } ?? -1
        do {
            var bytes = try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
            bytes.append(10)
            try FileHandle.standardOutput.write(contentsOf: bytes)
        } catch { loggingFailed = true }
    }

    func finish(cleanupConfirmed: Bool) -> Never {
        var success = callbackReceived && callbackSuccess && cleanupConfirmed && !cancelled
            && errors.isEmpty && !loggingFailed && (!stopRequested || stopCallbackSuccess)
        record("terminal", ["success": success, "cleanup_confirmed": cleanupConfirmed])
        success = success && !loggingFailed
        finished = true
        timer?.cancel()
        let result: [String: Any] = [
            "schema_version": 1, "installation_qualified": success,
            "virtual_machine_created": machine != nil, "installation_started": installStarted,
            "installer_callback_received": callbackReceived, "installer_callback_success": callbackSuccess,
            "cancellation_requested": cancelled, "cleanup_confirmed": cleanupConfirmed,
            "final_state": machine.map { $0.state.rawValue } ?? -1,
            "stop_requested_after_installer_callback": stopRequested,
            "stop_callback_received": stopCallbackReceived, "stop_callback_success": stopCallbackSuccess,
            "normal_start_called": false, "custom_listener_created": false, "display_window_created": false,
            "logging_failed": loggingFailed, "errors": errors, "event_count": eventCount,
            "configuration_validated": configurationValidated, "cpu_count": actualCPU, "memory_bytes": actualMemory,
            "elapsed_seconds": ProcessInfo.processInfo.systemUptime - began
        ]
        do {
            let url = URL(fileURLWithPath: input.result)
            try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys]).write(to: url, options: .atomic)
            let handle = try FileHandle(forWritingTo: url)
            try handle.synchronize()
            try handle.close()
            try FileHandle.standardOutput.synchronize()
        } catch { exit(1) }
        exit(success ? 0 : 1)
    }

    func cancel(_ reason: String) {
        guard !finished, !cancelled else { return }
        cancelled = true
        cleanupDeadline = ProcessInfo.processInfo.systemUptime + input.cleanup_seconds
        record("cancellation_requested", ["reason": reason])
        if installActive {
            // No pause/stop while the installer owns the VM. Never cancel before start.
            installer?.progress.cancel()
            record("installer_progress_cancelled")
        } else if !installStarted {
            finish(cleanupConfirmed: machine == nil || machine?.state == .stopped)
        } else {
            inspectTerminalState()
        }
    }

    func inspectTerminalState() {
        guard !finished, callbackReceived, !installActive, let machine = machine else { return }
        if machine.state == .stopped {
            if !stopRequested || stopCallbackReceived { finish(cleanupConfirmed: true) }
        } else if (machine.state == .running || machine.state == .paused) && machine.canStop && !stopRequested {
            stopRequested = true
            record("post_callback_destructive_stop_requested")
            machine.stop(completionHandler: { [self] error in
                stopCallbackReceived = true
                stopCallbackSuccess = error == nil
                if let error = error { errors.append(errorChain(error)) }
                record("post_callback_stop_completion", ["success": stopCallbackSuccess])
                if error != nil { finish(cleanupConfirmed: machine.state == .stopped) }
                inspectTerminalState()
            })
        } else if machine.state == .error {
            record("terminal_error_state")
            finish(cleanupConfirmed: false)
        }
    }

    func tick() {
        guard !finished else { return }
        let now = ProcessInfo.processInfo.systemUptime
        if let machine = machine, lastState != machine.state.rawValue {
            lastState = machine.state.rawValue
            record("state_observed")
        }
        if installActive, let installer = installer {
            let fraction = installer.progress.fractionCompleted
            guard fraction.isFinite else { cancel("nonfinite-progress"); return }
            let progress = Int(max(0, min(1, fraction)) * 10_000)
            if progress != lastProgress {
                lastProgress = progress
                record("installation_progress", ["fraction": fraction])
            }
        }
        if loggingFailed { cancel("diagnostic-write-failure") }
        if !callbackReceived && now - began >= input.operation_seconds { cancel("operation-deadline") }
        if let deadline = cleanupDeadline, now >= deadline {
            record("cleanup_deadline_expired")
            // Owner exit is bounded; it does not prove framework teardown.
            finish(cleanupConfirmed: callbackReceived && !installActive && machine?.state == .stopped
                && (!stopRequested || stopCallbackReceived))
        }
        if callbackReceived { inspectTerminalState() }
    }

    func begin() {
        dispatchPrecondition(condition: .onQueue(.main))
        do {
            guard input.operation_seconds > 0, input.cleanup_seconds > 0 else { throw OwnerFailure.invalidConfiguration }
            let root = URL(fileURLWithPath: input.seed, isDirectory: true)
            guard let hardware = VZMacHardwareModel(dataRepresentation: try Data(contentsOf: root.appendingPathComponent("hardware-model.bin"))),
                  hardware.isSupported,
                  let identifier = VZMacMachineIdentifier(dataRepresentation: try Data(contentsOf: root.appendingPathComponent("machine-id.bin"))),
                  let address = VZMACAddress(string: input.mac_address) else { throw OwnerFailure.invalidPlatform }
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
            configurationValidated = true
            actualCPU = configuration.cpuCount
            actualMemory = configuration.memorySize
            record("configuration_validated", ["cpu_count": actualCPU, "memory_bytes": actualMemory])
            guard !loggingFailed else { finish(cleanupConfirmed: true) }
            let virtualMachine = VZVirtualMachine(configuration: configuration)
            machine = virtualMachine
            virtualMachine.delegate = self
            record("virtual_machine_created")
            guard !loggingFailed else { finish(cleanupConfirmed: virtualMachine.state == .stopped) }
            guard virtualMachine.state == .stopped else { throw OwnerFailure.initialStateNotStopped }
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
            installer = VZMacOSInstaller(virtualMachine: virtualMachine, restoringFromImageAt: URL(fileURLWithPath: input.image))
            // Give queued signal/timer handlers an opportunity before the one install call.
            DispatchQueue.main.async { [self] in startInstallation() }
        } catch {
            errors.append(errorChain(error))
            record("pre_installation_error", ["errors": errors])
            finish(cleanupConfirmed: machine == nil || machine?.state == .stopped)
        }
    }

    func startInstallation() {
        guard !cancelled, let installer = installer else { finish(cleanupConfirmed: machine?.state == .stopped) }
        guard !loggingFailed else { cancel("pre-installation-log-failure"); return }
        record("installation_started")
        // Do not start guest execution after a failed synchronous marker write.
        guard !loggingFailed else { cancel("installation-marker-write-failure"); return }
        installStarted = true
        installActive = true
        installer.install(completionHandler: { [self] result in
            installActive = false
            callbackReceived = true
            switch result {
            case .success: callbackSuccess = true
            case .failure(let error): errors.append(errorChain(error))
            }
            record("installer_completion", ["success": callbackSuccess])
            if ProcessInfo.processInfo.systemUptime - began >= input.operation_seconds {
                cancel("completion-after-operation-deadline")
            }
            if cleanupDeadline == nil {
                cleanupDeadline = ProcessInfo.processInfo.systemUptime + input.cleanup_seconds
            }
            inspectTerminalState()
        })
    }

    func guestDidStop(_ virtualMachine: VZVirtualMachine) {
        record("guest_did_stop")
        if callbackReceived { inspectTerminalState() }
    }

    func virtualMachine(_ virtualMachine: VZVirtualMachine, didStopWithError error: Error) {
        errors.append(errorChain(error))
        record("delegate_stop_error", ["error": errorChain(error)])
        cancel("delegate-stop-error")
    }
}

if CommandLine.arguments.count == 2, CommandLine.arguments[1] == "error-control" {
    let shared = NSError(domain: "Gen2OwnerControl", code: 3)
    let branch = NSError(domain: "Gen2OwnerControl", code: 2, userInfo: [NSUnderlyingErrorKey: shared])
    let other = NSError(domain: "Gen2OwnerControl", code: 4)
    let root = NSError(domain: "Gen2OwnerControl", code: 1, userInfo: [NSUnderlyingErrorKey: branch,
        NSMultipleUnderlyingErrorsKey: [shared, other]])
    try FileHandle.standardOutput.write(contentsOf: JSONSerialization.data(withJSONObject: errorChain(root), options: [.sortedKeys]))
    exit(0)
}
guard CommandLine.arguments.count == 3, CommandLine.arguments[1] == "install" else { exit(64) }
let input = try JSONDecoder().decode(OwnerInput.self, from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[2])))
let owner = InstallOwner(input: input)
DispatchQueue.main.async { owner.begin() }
dispatchMain()
