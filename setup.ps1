[CmdletBinding()]
param(
    [switch]$NoLaunch,
    [switch]$SkipSelfTest,
    [switch]$PrerequisitesOnly
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$dataRoot = Join-Path $root 'SolomonPocketAIData'
$venvRoot = Join-Path $root '.venv'
$venvPython = Join-Path $venvRoot 'Scripts\python.exe'

function Write-Step([string]$Text) {
    Write-Host "`n==> $Text" -ForegroundColor Cyan
}

function Warn-LowDiskSpace {
    $driveRoot = [System.IO.Path]::GetPathRoot($root)
    try {
        $drive = New-Object System.IO.DriveInfo($driveRoot)
        $recommendedBytes = 7GB
        if ($drive.AvailableFreeSpace -lt $recommendedBytes) {
            $freeGiB = [math]::Round($drive.AvailableFreeSpace / 1GB, 1)
            Write-Warning "Only $freeGiB GB is free on $driveRoot. A first setup may need about 7 GB."
        }
    } catch {
        Write-Warning 'Could not check free disk space; setup will continue.'
    }
}

function Refresh-ProcessPath {
    $machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = (@($machinePath, $userPath) | Where-Object { $_ }) -join ';'
}

function Get-WinGetPath {
    $command = Get-Command winget.exe -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($command) {
        return $command.Source
    }
    $windowsAppsWinget = Join-Path $env:LOCALAPPDATA 'Microsoft\WindowsApps\winget.exe'
    if (Test-Path -LiteralPath $windowsAppsWinget -PathType Leaf) {
        return $windowsAppsWinget
    }
    return $null
}

function Install-WinGetPackage(
    [string]$Id,
    [string]$DisplayName,
    [switch]$UserScope
) {
    $winget = Get-WinGetPath
    if (-not $winget) {
        throw "$DisplayName is missing and Windows Package Manager (winget) is unavailable. Install or update 'App Installer' from Microsoft Store, then double-click setup.bat again."
    }

    Write-Host "Installing $DisplayName with Windows Package Manager..."
    $arguments = @(
        'install', '--id', $Id, '--exact',
        '--accept-package-agreements', '--accept-source-agreements',
        '--silent'
    )
    if ($UserScope) {
        $arguments += @('--scope', 'user')
    }
    & $winget @arguments
    $installExitCode = $LASTEXITCODE
    if ($installExitCode -notin @(0, 3010)) {
        throw "$DisplayName installation failed (winget exit code $installExitCode)."
    }
    Refresh-ProcessPath
}

function Resolve-Git {
    $command = Get-Command git.exe -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($command) {
        return $command.Source
    }
    $candidates = @(
        (Join-Path $env:ProgramFiles 'Git\cmd\git.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Git\cmd\git.exe')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }
    return $null
}

function Test-OllamaApi {
    try {
        Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -Method Get -TimeoutSec 2 | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Test-Python311Command(
    [string]$Launcher,
    [string[]]$Prefix
) {
    # Windows PowerShell 5.1 converts a native program's stderr into an error
    # record. A missing `py -3.11` is an expected negative probe, so it must not
    # inherit the script-wide Stop preference and abort before auto-install.
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Launcher @Prefix -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)' 2>$null | Out-Null
        $pythonExitCode = $LASTEXITCODE
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    return ($pythonExitCode -eq 0)
}

function Resolve-Python311 {
    $launcher = Get-Command py.exe -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($launcher) {
        if (Test-Python311Command -Launcher $launcher.Source -Prefix @('-3.11')) {
            return @{ Launcher = $launcher.Source; Prefix = @('-3.11') }
        }
    }
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe'),
        (Join-Path $env:ProgramFiles 'Python311\python.exe'),
        (Get-Command python.exe -CommandType Application -ErrorAction SilentlyContinue |
            Select-Object -First 1).Source
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
    foreach ($candidate in $candidates) {
        if (Test-Python311Command -Launcher $candidate -Prefix @()) {
            return @{ Launcher = $candidate; Prefix = @() }
        }
    }
    return $null
}

Set-Location -LiteralPath $root
Write-Host 'Solomon Pocket AI - private local setup' -ForegroundColor Green
Write-Host 'Models and conversations stay inside SolomonPocketAIData, which Git ignores.'
Warn-LowDiskSpace

Write-Step 'Checking Git'
$git = Resolve-Git
if (-not $git) {
    Install-WinGetPackage -Id 'Git.Git' -DisplayName 'Git'
    $git = Resolve-Git
}
if (-not $git) {
    throw 'Git was installed but could not be found. Double-click setup.bat again; a Windows restart should not normally be necessary.'
}
$gitVersion = & $git --version
if ($LASTEXITCODE -ne 0) {
    throw 'Git was found but could not be started.'
}
Write-Host "Found $gitVersion"

Write-Step 'Checking the tested Python 3.11 runtime'
$pythonInfo = Resolve-Python311
if (-not $pythonInfo) {
    Install-WinGetPackage -Id 'Python.Python.3.11' -DisplayName 'Python 3.11' -UserScope
    $pythonInfo = Resolve-Python311
}
$pythonLauncher = $pythonInfo.Launcher
$pythonPrefix = @($pythonInfo.Prefix)
if (-not $pythonLauncher) { throw 'Python 3.11 was installed but could not be found. Double-click setup.bat again; a Windows restart should not normally be necessary.' }
$versionText = & $pythonLauncher @pythonPrefix -c 'import sys; print(sys.version_info.major,sys.version_info.minor)'
if ($LASTEXITCODE -ne 0) {
    throw 'Python 3.11 could not be started. Double-click setup.bat again.'
}
$versionParts = $versionText.Trim().Split(' ', [System.StringSplitOptions]::RemoveEmptyEntries)
$version = [version]("$($versionParts[0]).$($versionParts[1])")
if ($version.Major -ne 3 -or $version.Minor -ne 11) {
    throw "The tested runtime is Python 3.11; found $version instead."
}
Write-Host "Found Python $version"

Write-Step 'Checking the local Ollama runtime'
$ollama = (Get-Command ollama.exe -CommandType Application -ErrorAction SilentlyContinue |
    Select-Object -First 1).Source
if (-not $ollama) {
    $standardOllama = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'
    if (Test-Path -LiteralPath $standardOllama -PathType Leaf) {
        $ollama = $standardOllama
    }
}
if (-not $ollama) {
    Install-WinGetPackage -Id 'Ollama.Ollama' -DisplayName 'Ollama'
    $standardOllama = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'
    if (Test-Path -LiteralPath $standardOllama -PathType Leaf) {
        $ollama = $standardOllama
    } else {
        $ollama = (Get-Command ollama.exe -CommandType Application -ErrorAction SilentlyContinue |
            Select-Object -First 1).Source
    }
}
if (-not $ollama) { throw 'Ollama was installed but could not be found. Double-click setup.bat again.' }
$ollamaVersion = & $ollama --version
if ($LASTEXITCODE -ne 0) { throw 'Ollama was found but could not be started.' }
Write-Host "Found $ollamaVersion"

if ($PrerequisitesOnly) {
    Write-Host "`nGit, Python 3.11, and Ollama are ready." -ForegroundColor Green
    exit 0
}

Write-Step 'Creating the private Python environment'
if (Test-Path -LiteralPath $venvRoot) {
    $venvIsCompatible = $false
    if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
        $venvIsCompatible = Test-Python311Command -Launcher $venvPython -Prefix @()
    }
    if (-not $venvIsCompatible) {
        $backupName = '.venv-incompatible-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
        $backupPath = Join-Path $root $backupName
        $suffix = 1
        while (Test-Path -LiteralPath $backupPath) {
            $backupPath = Join-Path $root ($backupName + '-' + $suffix)
            $suffix += 1
        }
        Move-Item -LiteralPath $venvRoot -Destination $backupPath
        Write-Host "Moved the incompatible Python environment to $backupPath"
    }
}
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    & $pythonLauncher @pythonPrefix -m venv $venvRoot
    if ($LASTEXITCODE -ne 0) { throw 'Could not create .venv.' }
}
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Could not update pip.' }
& $venvPython -m pip install -r (Join-Path $root 'desktop-requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Could not install Python dependencies.' }

Write-Step 'Downloading local speech models with checksum verification'
& $venvPython (Join-Path $root 'scripts\download_models.py') --data-root $dataRoot
if ($LASTEXITCODE -ne 0) { throw 'Could not prepare local speech models.' }

if (-not (Test-OllamaApi)) {
    Start-Process -FilePath $ollama -ArgumentList 'serve' -WindowStyle Hidden
    foreach ($attempt in 1..20) {
        Start-Sleep -Milliseconds 500
        if (Test-OllamaApi) { break }
    }
}
if (-not (Test-OllamaApi)) { throw 'Ollama did not start on its local loopback address.' }

Write-Step 'Downloading the local qwen3.5:4b conversation model'
& $ollama pull qwen3.5:4b
if ($LASTEXITCODE -ne 0) { throw 'Could not download qwen3.5:4b.' }

if (-not $SkipSelfTest) {
    Write-Step 'Running the offline engine self-test'
    & $venvPython (Join-Path $root 'SolomonPocketAI.py') --self-test
    if ($LASTEXITCODE -ne 0) { throw 'The self-test did not pass.' }
}

Write-Host "`nSolomon Pocket AI is ready." -ForegroundColor Green
if (-not $NoLaunch) {
    Start-Process -FilePath (Join-Path $root 'start-solomon-pocket-ai.bat') -WorkingDirectory $root
}
