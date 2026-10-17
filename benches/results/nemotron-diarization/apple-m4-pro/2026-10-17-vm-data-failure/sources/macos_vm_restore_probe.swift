// Apple API usage adapted from the official macOS VM sample. See macos_vm_apple_LICENSE.txt.
import CryptoKit
import Darwin
import Foundation
import Virtualization

let arguments = CommandLine.arguments
guard (arguments.count == 3 && arguments[1] == "inspect")
    || (arguments.count == 4 && arguments[1] == "prepare") else {
    FileHandle.standardError.write(Data("usage: restore-probe inspect IPSW | prepare IPSW NEW_DIRECTORY\n".utf8))
    exit(64)
}

func digest(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

func finish(_ value: [String: Any], code: Int32) -> Never {
    do {
        FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]))
        FileHandle.standardOutput.write(Data("\n".utf8))
    } catch {
        FileHandle.standardError.write(Data("JSON encoding failed: \(error)\n".utf8))
        exit(1)
    }
    exit(code)
}

enum ProbeFailure: Error {
    case unsupportedGeometry
    case diskCreation(Int32)
    case diskResize(Int32)
}

func prepare(_ requirements: VZMacOSConfigurationRequirements, directory: URL) throws -> [String: Any] {
    let cpuCount = 4
    let memory: UInt64 = 8 * 1024 * 1024 * 1024
    let diskBytes: Int64 = 100 * 1024 * 1024 * 1024
    guard requirements.minimumSupportedCPUCount <= cpuCount,
          requirements.minimumSupportedMemorySize <= memory,
          VZVirtualMachineConfiguration.minimumAllowedCPUCount <= cpuCount,
          cpuCount <= VZVirtualMachineConfiguration.maximumAllowedCPUCount,
          VZVirtualMachineConfiguration.minimumAllowedMemorySize <= memory,
          memory <= VZVirtualMachineConfiguration.maximumAllowedMemorySize else {
        throw ProbeFailure.unsupportedGeometry
    }
    let files = FileManager.default
    try files.createDirectory(at: directory, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
    let hardware = requirements.hardwareModel
    let identifier = VZMacMachineIdentifier()
    let modelURL = directory.appendingPathComponent("hardware-model.bin")
    let identifierURL = directory.appendingPathComponent("machine-id.bin")
    let auxiliaryURL = directory.appendingPathComponent("auxiliary-storage.bin")
    let diskURL = directory.appendingPathComponent("disk.img")
    try hardware.dataRepresentation.write(to: modelURL, options: .withoutOverwriting)
    try identifier.dataRepresentation.write(to: identifierURL, options: .withoutOverwriting)
    let auxiliary = try VZMacAuxiliaryStorage(creatingStorageAt: auxiliaryURL, hardwareModel: hardware, options: [])
    let descriptor = Darwin.open(diskURL.path, O_CREAT | O_EXCL | O_RDWR | O_NOFOLLOW, S_IRUSR | S_IWUSR)
    guard descriptor >= 0 else { throw ProbeFailure.diskCreation(errno) }
    defer { Darwin.close(descriptor) }
    guard ftruncate(descriptor, diskBytes) == 0 else { throw ProbeFailure.diskResize(errno) }
    let platform = VZMacPlatformConfiguration()
    platform.hardwareModel = hardware
    platform.machineIdentifier = identifier
    platform.auxiliaryStorage = auxiliary
    let configuration = VZVirtualMachineConfiguration()
    configuration.platform = platform
    configuration.bootLoader = VZMacOSBootLoader()
    configuration.cpuCount = cpuCount
    configuration.memorySize = memory
    let graphics = VZMacGraphicsDeviceConfiguration()
    graphics.displays = [VZMacGraphicsDisplayConfiguration(widthInPixels: 1280, heightInPixels: 800, pixelsPerInch: 80)]
    configuration.graphicsDevices = [graphics]
    configuration.keyboards = [VZUSBKeyboardConfiguration()]
    configuration.pointingDevices = [VZUSBScreenCoordinatePointingDeviceConfiguration()]
    configuration.audioDevices = []
    let network = VZVirtioNetworkDeviceConfiguration()
    network.macAddress = VZMACAddress.randomLocallyAdministered()
    network.attachment = VZNATNetworkDeviceAttachment()
    configuration.networkDevices = [network]
    configuration.storageDevices = [VZVirtioBlockDeviceConfiguration(
        attachment: try VZDiskImageStorageDeviceAttachment(url: diskURL, readOnly: false))]
    try configuration.validate()
    for path in [modelURL, identifierURL, auxiliaryURL, diskURL] {
        try files.setAttributes([.posixPermissions: 0o600], ofItemAtPath: path.path)
    }
    return [
        "configuration_validated": true,
        "cpu_count": cpuCount, "memory_bytes": memory, "disk_logical_bytes": diskBytes,
        "hardware_model_sha256": digest(hardware.dataRepresentation),
        "machine_identifier_sha256": digest(identifier.dataRepresentation),
        "network_attachment": "nat", "mac_address": network.macAddress.string,
        "graphics_displays": 1, "audio_devices": 0, "shared_directories": 0,
        "display_window_created": false, "custom_listener_created": false
    ]
}

let imageURL = URL(fileURLWithPath: arguments[2])
VZMacOSRestoreImage.load(from: imageURL) { result in
    DispatchQueue.main.async {
        do {
            let image = try result.get()
            let version = image.operatingSystemVersion
            var value: [String: Any] = [
                "schema_version": 1, "operation": arguments[1], "restore_image_loaded": true,
                "image_supported": image.isSupported,
                "build_version": image.buildVersion,
                "os_version": [version.majorVersion, version.minorVersion, version.patchVersion],
                "configuration_validated": false,
                "virtual_machine_created": false, "installation_started": false, "guest_booted": false
            ]
            guard let requirements = image.mostFeaturefulSupportedConfiguration else {
                value["supported_configuration_available"] = false
                finish(value, code: 2)
            }
            value["supported_configuration_available"] = true
            value["hardware_model_supported"] = requirements.hardwareModel.isSupported
            value["hardware_model_sha256"] = digest(requirements.hardwareModel.dataRepresentation)
            value["minimum_cpu_count"] = requirements.minimumSupportedCPUCount
            value["minimum_memory_bytes"] = requirements.minimumSupportedMemorySize
            if arguments[1] == "prepare" {
                let prepared = try prepare(requirements, directory: URL(fileURLWithPath: arguments[3], isDirectory: true))
                value.merge(prepared) { _, new in new }
            }
            finish(value, code: image.isSupported && requirements.hardwareModel.isSupported ? 0 : 2)
        } catch {
            finish(["schema_version": 1, "operation": arguments[1], "error": String(describing: error),
                    "virtual_machine_created": false, "installation_started": false, "guest_booted": false], code: 1)
        }
    }
}
dispatchMain()
