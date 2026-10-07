[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $root

$files = @(& git ls-files --cached --others --exclude-standard) |
    Where-Object { $_ -and -not $_.StartsWith('.git/') } |
    Sort-Object -Unique
if ($LASTEXITCODE -ne 0) { throw 'Could not enumerate the Git publication surface.' }

$findings = [System.Collections.Generic.List[string]]::new()
$forbiddenExtensions = @('.pt', '.onnx', '.bin', '.wav', '.mp3', '.mp4', '.mov', '.avi')
$windowsUsers = 'C:' + [char]92 + 'Users' + [char]92
$macUsers = '/' + 'Users' + '/'
$linuxHome = '/' + 'home' + '/'
$emailPattern = '[A-Z0-9._%+-]+' + [regex]::Escape([string][char]64) + '[A-Z0-9.-]+\.[A-Z]{2,}'
$privateKeyMarker = 'BEGIN ' + 'PRIVATE KEY'
$secretPrefix = 'sk' + '-[A-Za-z0-9]{20,}'
$contentChecks = @(
    @{ Name = 'absolute Windows user path'; Pattern = [regex]::Escape($windowsUsers) },
    @{ Name = 'absolute macOS user path'; Pattern = [regex]::Escape($macUsers) + '[^/\s]+' },
    @{ Name = 'absolute Linux home path'; Pattern = [regex]::Escape($linuxHome) + '[^/\s]+' },
    @{ Name = 'email address'; Pattern = $emailPattern },
    @{ Name = 'private key'; Pattern = [regex]::Escape($privateKeyMarker) },
    @{ Name = 'API-key-shaped token'; Pattern = $secretPrefix }
)

foreach ($relative in $files) {
    $path = Join-Path $root $relative
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { continue }
    $item = Get-Item -LiteralPath $path
    $extension = [IO.Path]::GetExtension($relative).ToLowerInvariant()
    if ($forbiddenExtensions -contains $extension) {
        $findings.Add("$relative - model/media binary is not allowed in the public repository")
        continue
    }
    if ($item.Length -gt 10MB) {
        $findings.Add("$relative - file exceeds the 10 MB public-source limit")
        continue
    }
    if ($extension -in @('.png', '.jpg', '.jpeg', '.gif', '.ico')) { continue }

    try {
        $content = Get-Content -LiteralPath $path -Raw -Encoding UTF8
    } catch {
        $findings.Add("$relative - could not be inspected as UTF-8 text")
        continue
    }
    foreach ($check in $contentChecks) {
        if ($content -match $check.Pattern) {
            $findings.Add("$relative - contains $($check.Name)")
        }
    }
}

Write-Host "Public Git candidate files: $($files.Count)"
$files | ForEach-Object { Write-Host "  $_" }
if ($findings.Count -gt 0) {
    Write-Host "`nPUBLICATION BLOCKED" -ForegroundColor Red
    $findings | Sort-Object -Unique | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    exit 1
}

Write-Host "`nPUBLICATION SURFACE PASSED: no personal paths, email addresses, common secrets, runtime data, or model binaries found." -ForegroundColor Green
