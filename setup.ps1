[CmdletBinding()]
param(
    [switch]$NoLaunch,
    [switch]$SkipSelfTest
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$dataRoot = Join-Path $root 'SolomonPocketAIData'
$venvRoot = Join-Path $root '.venv'
$venvPython = Join-Path $venvRoot 'Scripts\python.exe'

function Write-Step([string]$Text) {
    Write-Host "`n==> $Text" -ForegroundColor Cyan
}

function Test-OllamaApi {
    try {
        Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -Method Get -TimeoutSec 2 | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Resolve-Python311 {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.11 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>$null
        if ($LASTEXITCODE -eq 0) {
            return @{ Launcher = 'py'; Prefix = @('-3.11') }
        }
    }
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe'),
        (Get-Command python -ErrorAction SilentlyContinue).Source
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
    foreach ($candidate in $candidates) {
        & $candidate -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>$null
        if ($LASTEXITCODE -eq 0) {
            return @{ Launcher = $candidate; Prefix = @() }
        }
    }
    return $null
}

Set-Location -LiteralPath $root
Write-Host 'Solomon Pocket AI - private local setup' -ForegroundColor Green
Write-Host 'Models and conversations stay inside SolomonPocketAIData, which Git ignores.'

Write-Step 'Checking Python 3.11+'
$pythonInfo = Resolve-Python311
if (-not $pythonInfo) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw 'Python 3.11 is required. Install it from https://www.python.org/downloads/ and rerun setup.bat.'
    }
    Write-Host 'Installing Python 3.11 with Windows Package Manager...'
    & winget install --id Python.Python.3.11 --exact --accept-package-agreements --accept-source-agreements --silent
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 installation failed.' }
    $pythonInfo = Resolve-Python311
}
$pythonLauncher = $pythonInfo.Launcher
$pythonPrefix = @($pythonInfo.Prefix)
if (-not $pythonLauncher) { throw 'Python 3.11 was installed but could not be started. Restart Windows and rerun setup.bat.' }
$versionText = & $pythonLauncher @pythonPrefix -c 'import sys; print(sys.version_info.major,sys.version_info.minor)'
if ($LASTEXITCODE -ne 0) {
    throw 'Python 3.11 could not be started. Install Python 3.11+ and rerun setup.bat.'
}
$versionParts = $versionText.Trim().Split(' ', [System.StringSplitOptions]::RemoveEmptyEntries)
$version = [version]("$($versionParts[0]).$($versionParts[1])")
if ($version -lt [version]'3.11') {
    throw "Python 3.11 or newer is required; found $version."
}

Write-Step 'Creating the private Python environment'
if (-not (Test-Path -LiteralPath $venvPython)) {
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

Write-Step 'Checking the local Ollama runtime'
$ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
if (-not $ollama) {
    $standardOllama = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'
    if (Test-Path -LiteralPath $standardOllama) {
        $ollama = $standardOllama
    }
}
if (-not $ollama) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw 'Ollama is required. Install it from https://ollama.com/download/windows and rerun setup.bat.'
    }
    Write-Host 'Installing Ollama with Windows Package Manager...'
    & winget install --id Ollama.Ollama --exact --accept-package-agreements --accept-source-agreements --silent
    if ($LASTEXITCODE -ne 0) { throw 'Ollama installation failed.' }
    $standardOllama = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'
    if (Test-Path -LiteralPath $standardOllama) {
        $ollama = $standardOllama
    } else {
        $ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
    }
}
if (-not $ollama) { throw 'Ollama was installed but its executable could not be found. Restart Windows and rerun setup.bat.' }

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
