# Read-only build provenance. This does not establish redistribution entitlement.
param(
    [Parameter(Mandatory=$true)][string]$SdkRoot,
    [Parameter(Mandatory=$true)][string]$CMakeCache,
    [Parameter(Mandatory=$true)][string]$Compiler,
    [Parameter(Mandatory=$true)][string]$Output
)
$ErrorActionPreference = 'Stop'

function FileRecord([string]$Path) {
    $item = Get-Item -LiteralPath $Path
    if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Expected regular provenance file: $Path"
    }
    return [ordered]@{
        path = $item.FullName
        bytes = $item.Length
        sha256 = (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        file_version = $item.VersionInfo.FileVersion
        product_version = $item.VersionInfo.ProductVersion
    }
}

function CertificateRecord($Certificate) {
    if ($null -eq $Certificate) { return $null }
    return [ordered]@{
        subject = $Certificate.Subject
        issuer = $Certificate.Issuer
        thumbprint = $Certificate.Thumbprint
        serial_number = $Certificate.SerialNumber
        not_before_utc = $Certificate.NotBefore.ToUniversalTime().ToString('o')
        not_after_utc = $Certificate.NotAfter.ToUniversalTime().ToString('o')
    }
}

function CachePath([string]$Name) {
    $prefix = $Name + ':PATH='
    $lines = @(Get-Content -LiteralPath $CMakeCache | Where-Object { $_.StartsWith($prefix) })
    if ($lines.Count -ne 1) { throw "Expected one CMake provenance path: $Name" }
    return (Get-Item -LiteralPath $lines[0].Substring($prefix.Length)).FullName
}

$redist = CachePath 'MSVC_REDIST_DIR'
$x64 = Join-Path $redist 'x64'
$files = @()
foreach ($name in @('concrt140.dll', 'msvcp140.dll', 'msvcp140_1.dll', 'msvcp140_2.dll',
                    'msvcp140_atomic_wait.dll', 'msvcp140_codecvt_ids.dll',
                    'vcruntime140.dll', 'vcruntime140_1.dll')) {
    $sources = @(Get-ChildItem -LiteralPath $x64 -Recurse -File -Filter $name)
    if ($sources.Count -ne 1) { throw "Ambiguous or absent x64 redistributable origin: $name" }
    $source = FileRecord $sources[0].FullName
    $installed = FileRecord (Join-Path (Join-Path $SdkRoot 'bin') $name)
    if ($source.sha256 -ne $installed.sha256 -or $source.bytes -ne $installed.bytes) {
        throw "Installed redistributable differs from actual redist source: $name"
    }
    $signature = Get-AuthenticodeSignature -LiteralPath $sources[0].FullName
    $files += [ordered]@{
        name = $name
        source = $source
        installed = $installed
        signature = [ordered]@{
            status = $signature.Status.ToString()
            status_message = $signature.StatusMessage
            signer = CertificateRecord $signature.SignerCertificate
            timestamp_signer = CertificateRecord $signature.TimeStamperCertificate
        }
    }
}

# Capture only actual candidates; absence is a visible unresolved license gate.
$licenseCandidates = @(Get-ChildItem -LiteralPath $redist -Recurse -File |
    Where-Object { $_.Name -match '(?i)(license|licence|eula|notice)' } |
    ForEach-Object { FileRecord $_.FullName })
$vcpkg = CachePath 'VCPKG_INSTALLED_DIR'
$status = Join-Path $vcpkg 'vcpkg/status'
$metadata = @()
if (Test-Path -LiteralPath $status -PathType Leaf) {
    $destination = Join-Path $Output 'windows-vcpkg-status.log'
    Copy-Item -LiteralPath $status -Destination $destination
    $metadata += FileRecord $destination
}
$info = Join-Path $vcpkg 'vcpkg/info'
if (Test-Path -LiteralPath $info -PathType Container) {
    $metadata = @($metadata) + @(Get-ChildItem -LiteralPath $info -File |
        Where-Object { $_.Name -match '(?i)(\.list|\.spdx\.json)$' } |
        ForEach-Object { FileRecord $_.FullName })
}
$os = Get-CimInstance Win32_OperatingSystem
$cpu = @(Get-CimInstance Win32_Processor | Select-Object Name, Manufacturer, Architecture,
    NumberOfCores, NumberOfLogicalProcessors, ProcessorId)
$record = [ordered]@{
    schema_version = 1
    scope = 'Read-only actual build, source/installed redist bytes, versions and Authenticode observations.'
    redistribution_qualified = $false
    limits = @('No distribution entitlement or applicable license selection established.',
               'Signature status is an observation on the connected build host, not an offline trust validation.',
               'vcpkg installed metadata does not by itself preserve complete source/build provenance.',
               'No fresh-machine, runtime closure or coexistence qualification.')
    os = [ordered]@{ caption = $os.Caption; version = $os.Version; build = $os.BuildNumber; architecture = $os.OSArchitecture }
    cpu = $cpu
    compiler = FileRecord $Compiler
    cmake_cache = FileRecord $CMakeCache
    redist_root = $redist
    redistributables = $files
    redist_license_candidates = $licenseCandidates
    vcpkg_installed_root = $vcpkg
    vcpkg_metadata = $metadata
}
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $Output 'windows-provenance.json') -Encoding utf8NoBOM
