// Apple API usage adapted from the official macOS VM sample. See macos_vm_apple_LICENSE.txt.
import Foundation
import Virtualization

guard CommandLine.arguments == [CommandLine.arguments[0], "probe"] else {
    FileHandle.standardError.write(Data("usage: vm-host-probe probe\n".utf8))
    exit(64)
}

let supported = VZVirtualMachine.isSupported
let minimumCPU = VZVirtualMachineConfiguration.minimumAllowedCPUCount
let maximumCPU = VZVirtualMachineConfiguration.maximumAllowedCPUCount
let minimumMemory = VZVirtualMachineConfiguration.minimumAllowedMemorySize
let maximumMemory = VZVirtualMachineConfiguration.maximumAllowedMemorySize
let requestedMemory: UInt64 = 8 * 1024 * 1024 * 1024
let geometryAllowed = minimumCPU <= 4 && 4 <= maximumCPU
    && minimumMemory <= requestedMemory && requestedMemory <= maximumMemory
let result: [String: Any] = [
    "schema_version": 1,
    "virtualization_supported": supported,
    "minimum_cpu_count": minimumCPU,
    "maximum_cpu_count": maximumCPU,
    "minimum_memory_bytes": minimumMemory,
    "maximum_memory_bytes": maximumMemory,
    "proposed_cpu_count": 4,
    "proposed_memory_bytes": requestedMemory,
    "geometry_within_host_limits": geometryAllowed,
    "virtual_machine_created": false,
    "restore_image_loaded": false,
    "guest_booted": false
]
let data = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
FileHandle.standardOutput.write(data)
FileHandle.standardOutput.write(Data("\n".utf8))
exit(supported && geometryAllowed ? 0 : 2)
