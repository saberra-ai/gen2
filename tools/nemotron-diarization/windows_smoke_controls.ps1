$ErrorActionPreference = 'Stop'
$expected = 'Gen2 offline read-only input control'
if ([IO.File]::ReadAllText('C:\inputs\control.txt') -ne $expected) {
    throw 'Read-only input content differs'
}
$writeDenied = $false
$writeError = $null
try {
    [IO.File]::WriteAllText('C:\inputs\forbidden.txt', 'must not succeed')
} catch {
    $cause = $_.Exception
    while ($null -ne $cause.InnerException) { $cause = $cause.InnerException }
    $writeError = $cause.HResult -band 65535
    $writeDenied = $writeError -eq 5
}
if (-not $writeDenied -or (Test-Path -LiteralPath 'C:\inputs\forbidden.txt')) {
    throw 'Read-only input control did not fail with access denied'
}
$tools = @(Get-Command cl.exe,cmake.exe,python.exe,dumpbin.exe,git.exe -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty Source)
if ($tools.Count -ne 0 -or (Test-Path -LiteralPath 'C:\Program Files\Microsoft Visual Studio')) {
    throw 'Development tools unexpectedly available in runtime image'
}
$network = @([Net.NetworkInformation.NetworkInterface]::GetAllNetworkInterfaces() |
    Where-Object { $_.OperationalStatus -eq 'Up' -and $_.NetworkInterfaceType -ne 'Loopback' } |
    ForEach-Object { $_.Name })
if ($network.Count -ne 0) { throw 'Operational non-loopback interface in network-none container' }
$record = [ordered]@{
    schema_version = 1
    input_read = $true
    input_write_denied = $writeDenied
    input_write_win32_error = $writeError
    development_tools_absent_from_path = $true
    visual_studio_directory_absent = $true
    operational_non_loopback_interfaces = $network
    os_version = [Environment]::OSVersion.Version.ToString()
    process_architecture = $env:PROCESSOR_ARCHITECTURE
    input_sha256 = (Get-FileHash -LiteralPath 'C:\inputs\control.txt' -Algorithm SHA256).Hash.ToLowerInvariant()
}
$record | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath 'C:\results\controls.json' -Encoding UTF8
