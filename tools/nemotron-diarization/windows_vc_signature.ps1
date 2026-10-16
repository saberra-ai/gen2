param([Parameter(Mandatory=$true)][string]$InputFile,
      [Parameter(Mandatory=$true)][string]$Output)
$ErrorActionPreference = 'Stop'
function CertificateRecord($value) {
    if ($null -eq $value) { return $null }
    return [ordered]@{subject=$value.Subject;issuer=$value.Issuer;thumbprint=$value.Thumbprint
        serial_number=$value.SerialNumber;not_before_utc=$value.NotBefore.ToUniversalTime().ToString('o')
        not_after_utc=$value.NotAfter.ToUniversalTime().ToString('o')}
}
$item = Get-Item -LiteralPath $InputFile
if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Expected regular prerequisite file' }
$sig = Get-AuthenticodeSignature -LiteralPath $item.FullName
[ordered]@{path=$item.FullName;bytes=$item.Length
    sha256=(Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    observed_at_utc=[DateTime]::UtcNow.ToString('o');status=$sig.Status.ToString();status_message=$sig.StatusMessage
    signer=(CertificateRecord $sig.SignerCertificate);timestamp_signer=(CertificateRecord $sig.TimeStamperCertificate)
    scope='Actual connected Windows build-host trust observation before installer execution; no independent offline or future trust guarantee.'
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Output -Encoding utf8NoBOM
