# Disposable qualification image only; no host installation or image publishing.
param([Parameter(Mandatory=$true)][ValidateSet('Install','Observe')][string]$Mode)
$ErrorActionPreference = 'Stop'
& 'C:\inputs\controls.ps1'
$utf8 = New-Object Text.UTF8Encoding($false)
$expected = Get-Content -Raw -LiteralPath 'C:\inputs\expected.json' -Encoding UTF8 | ConvertFrom-Json
function CertificateRecord($value) {
    if ($null -eq $value) { return $null }
    return [ordered]@{subject=$value.Subject;issuer=$value.Issuer;thumbprint=$value.Thumbprint
        serial_number=$value.SerialNumber;not_before_utc=$value.NotBefore.ToUniversalTime().ToString('o')
        not_after_utc=$value.NotAfter.ToUniversalTime().ToString('o')}
}
function Snapshot {
    $files = @()
    foreach ($property in $expected.files.PSObject.Properties) {
        $path = Join-Path 'C:\Windows\System32' $property.Name
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            $files += [ordered]@{name=$property.Name;path=$path;present=$false}
            continue
        }
        $item = Get-Item -LiteralPath $path
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Unexpected central runtime reparse point' }
        $sig = Get-AuthenticodeSignature -LiteralPath $path
        $files += [ordered]@{name=$property.Name;path=$item.FullName;present=$true;bytes=$item.Length
            sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
            file_version=$item.VersionInfo.FileVersion;product_version=$item.VersionInfo.ProductVersion
            signature=[ordered]@{status=$sig.Status.ToString();status_message=$sig.StatusMessage
                signer=(CertificateRecord $sig.SignerCertificate);timestamp_signer=(CertificateRecord $sig.TimeStamperCertificate)}}
    }
    $registry = @()
    foreach ($view in @([Microsoft.Win32.RegistryView]::Registry64,[Microsoft.Win32.RegistryView]::Registry32)) {
        $base = [Microsoft.Win32.RegistryKey]::OpenBaseKey([Microsoft.Win32.RegistryHive]::LocalMachine,$view)
        $key = $null
        try {
            $key = $base.OpenSubKey('SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64')
            $values = [ordered]@{}
            if ($null -ne $key) { foreach ($name in $key.GetValueNames()) { $values[$name] = $key.GetValue($name) } }
            $registry += [ordered]@{view=$view.ToString();present=($null -ne $key);values=$values}
        } finally { if ($null -ne $key) { $key.Dispose() }; $base.Dispose() }
    }
    $renames = @()
    $key = Get-ItemProperty -LiteralPath 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager' -ErrorAction Stop
    if ($null -ne $key.PendingFileRenameOperations) { $renames = @($key.PendingFileRenameOperations) }
    return [ordered]@{is_64_bit_process=[Environment]::Is64BitProcess;files=$files;registry=$registry
        pending_restart=[ordered]@{
            cbs=(Test-Path -LiteralPath 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending')
            windows_update=(Test-Path -LiteralPath 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired')
            file_renames=$renames}}
}
$result = [ordered]@{schema_version=1;mode=$Mode;installer_executed=$false;completed=$false;before=(Snapshot)}
try {
    if ($Mode -eq 'Install') {
        $path = 'C:\inputs\VC_redist.x64.exe'
        $item = Get-Item -LiteralPath $path
        $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($item.Length -ne $expected.installer.size -or $hash -cne $expected.installer.sha256) { throw 'Pinned installer bytes differ inside setup container' }
        New-Item -Path 'C:\results\tmp' -ItemType Directory -Force | Out-Null
        $env:TEMP = 'C:\results\tmp'; $env:TMP = 'C:\results\tmp'
        $process = Start-Process -FilePath $path -ArgumentList '/install /quiet /norestart /log C:\results\vc-install.log' -Wait -PassThru
        $result.installer_executed = $true
        $result.installer_exit_code = $process.ExitCode
        $process.Dispose()
    }
    $result.after = Snapshot
    $result.completed = $true
} catch {
    $result.error = $_.Exception.Message
    throw
} finally {
    [IO.File]::WriteAllText('C:\results\state.json', ($result | ConvertTo-Json -Depth 20), $utf8)
}
