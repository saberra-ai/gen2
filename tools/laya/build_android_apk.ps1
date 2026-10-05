param(
    [Parameter(Mandatory=$true)][string]$Sdk,
    [Parameter(Mandatory=$true)][string]$JavaHome,
    [Parameter(Mandatory=$true)][string]$GradleHome,
    [switch]$AllowDependencyDownload
)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$env:JAVA_HOME = (Resolve-Path -LiteralPath $JavaHome).Path
$env:ANDROID_HOME = (Resolve-Path -LiteralPath $Sdk).Path
$env:ANDROID_USER_HOME = Join-Path $workspace 'target/laya/android/user-home'
$env:GRADLE_USER_HOME = Join-Path $workspace 'target/laya/android/gradle-home'
$gradle = Join-Path (Resolve-Path -LiteralPath $GradleHome).Path 'bin/gradle.bat'
$key = Join-Path $workspace 'target/laya/android/debug.keystore'
if (!(Test-Path -LiteralPath $key)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $key) | Out-Null
    & (Join-Path $env:JAVA_HOME 'bin/keytool.exe') -genkeypair -keystore $key -storepass android -keypass android -alias gen2debug -dname 'CN=Gen2 local debug' -keyalg RSA -keysize 2048 -validity 10000 -noprompt
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
$arguments = @('--no-daemon', '--max-workers=2', '-p', (Join-Path $workspace 'examples/laya-mobile/android'), ':app:assembleDebug', ':app:lintDebug')
if (!$AllowDependencyDownload) { $arguments += '--offline' }
& $gradle @arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$apk = Join-Path $workspace 'examples/laya-mobile/android/app/build/outputs/apk/debug/app-debug.apk'
& (Join-Path $env:ANDROID_HOME 'build-tools/35.0.0/apksigner.bat') verify --verbose $apk
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Output "Built and signature-verified $apk; install and inference on a device remain unverified."
