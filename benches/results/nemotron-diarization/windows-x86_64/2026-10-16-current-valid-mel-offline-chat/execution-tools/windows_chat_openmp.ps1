# Test-only acquisition from this build host's existing VS redist tree.
param([Parameter(Mandatory=$true)][string]$NativeProvenance,
      [Parameter(Mandatory=$true)][string]$Output)
$ErrorActionPreference = 'Stop'
$native = Get-Content -Raw -LiteralPath $NativeProvenance -Encoding UTF8 | ConvertFrom-Json
$path = Join-Path $native.redist_root 'x64/Microsoft.VC143.OpenMP/vcomp140.dll'
$item = Get-Item -LiteralPath $path
if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Expected regular OpenMP DLL' }
$signature = Get-AuthenticodeSignature -LiteralPath $path
$record = [ordered]@{
    schema_version = 1; redistribution_qualified = $false
    scope = 'Connected build-host observation of existing VS OpenMP bytes for isolated qualification helpers only.'
    path = $item.FullName; bytes = $item.Length
    sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    file_version = $item.VersionInfo.FileVersion; product_version = $item.VersionInfo.ProductVersion
    signature_status = $signature.Status.ToString()
    signer_subject = $signature.SignerCertificate.Subject
    signer_thumbprint = $signature.SignerCertificate.Thumbprint
    timestamp_subject = $signature.TimeStamperCertificate.Subject
}
$record | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $Output -Encoding utf8NoBOM
if ($record.signature_status -ne 'Valid' -or $record.signer_subject -notmatch '(?:^|, )O=Microsoft Corporation(?:,|$)') {
    throw 'Expected connected-host valid Microsoft OpenMP signature'
}
