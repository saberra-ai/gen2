# Qualification-only launcher. The child uses the exact recorded command line;
# async output reads prevent a full pipe from blocking native inference.
$ErrorActionPreference = 'Stop'
& 'C:\inputs\controls.ps1'
$utf8 = New-Object Text.UTF8Encoding($false)
$spec = Get-Content -Raw -LiteralPath 'C:\inputs\command.json' -Encoding UTF8 | ConvertFrom-Json
$start = New-Object Diagnostics.ProcessStartInfo
$start.FileName = $spec.executable
$start.Arguments = $spec.command_line
$start.UseShellExecute = $false
$start.CreateNoWindow = $true
$start.RedirectStandardOutput = $true
$start.RedirectStandardError = $true
$start.StandardOutputEncoding = $utf8
$start.StandardErrorEncoding = $utf8
$start.WorkingDirectory = 'C:\results'
foreach ($key in @($start.EnvironmentVariables.Keys)) { $start.EnvironmentVariables.Remove($key) }
foreach ($entry in $spec.environment.PSObject.Properties) { $start.EnvironmentVariables[$entry.Name] = [string]$entry.Value }
$child = New-Object Diagnostics.Process
$child.StartInfo = $start
$started = $false
$modules = @{}
$previous = $null
$stable = 0
$result = [ordered]@{ schema_version = 1; stable_snapshots = 0; modules = @(); completed = $false }
try {
    if (-not $child.Start()) { throw 'Child process did not start' }
    $started = $true
    $stdout = $child.StandardOutput.ReadToEndAsync()
    $stderr = $child.StandardError.ReadToEndAsync()
    $timer = [Diagnostics.Stopwatch]::StartNew()
    while (-not $child.HasExited) {
        if ($timer.Elapsed.TotalSeconds -gt 1800) { throw 'Qualification helper exceeded 1800 seconds' }
        if ($spec.capture_modules) {
            try {
                $child.Refresh()
                $paths = @($child.Modules | ForEach-Object { $_.FileName } | Sort-Object)
                if ($paths.Count -gt 512) { throw 'Excessive child module count' }
                $current = $paths -join "`n"
                if ($paths.Count -gt 0 -and $current -ceq $previous) {
                    $stable++
                    foreach ($path in $paths) {
                        if (-not $modules.ContainsKey($path)) {
                            $item = Get-Item -LiteralPath $path
                            $modules[$path] = [ordered]@{ path = $path; bytes = $item.Length
                                sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() }
                        }
                    }
                }
                $previous = $current
            } catch {
                if (-not $child.HasExited) { throw }
            }
        }
        Start-Sleep -Milliseconds 100
    }
    $child.WaitForExit()
    [IO.File]::WriteAllText('C:\results\child.stdout.log', $stdout.GetAwaiter().GetResult(), $utf8)
    [IO.File]::WriteAllText('C:\results\child.stderr.log', $stderr.GetAwaiter().GetResult(), $utf8)
    $result.returncode = $child.ExitCode
    $result.stable_snapshots = $stable
    $result.modules = @($modules.Values | Sort-Object { $_.path })
    $result.completed = $true
    if (($child.ExitCode -eq 0) -eq [bool]$spec.expected_failure) { throw 'Unexpected child exit status' }
    if ($spec.capture_modules -and $stable -eq 0) { throw 'No stable child module snapshot observed' }
} catch {
    $result.error = $_.Exception.Message
    throw
} finally {
    if ($started -and -not $child.HasExited) { $child.Kill(); $child.WaitForExit() }
    $result | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath 'C:\results\child.json' -Encoding UTF8
    $child.Dispose()
}
