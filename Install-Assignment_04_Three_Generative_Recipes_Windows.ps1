#Requires -Version 5.1

param(
    [string]$ScriptDir = $PSScriptRoot,
    # Skip the interactive menu: 'quick' (smoke test), 'full' (complete run) or 'none' (setup only)
    [ValidateSet('', 'quick', 'full', 'none')]
    [string]$Launch = ''
)

#=============================================================================
#region SCRIPT DETAILS
#=============================================================================
<#
.SYNOPSIS
Checks for Windows OS, verifies prerequisites, and installs the required
Python packages for Assignment 4 (Three Generative Recipes) into a local .venv.

.DESCRIPTION
- Verifies Python 3.10+ is installed and on PATH
- Creates (or reuses) a project-local virtual environment in .venv
- Upgrades pip inside the venv
- Runs env_setup.py, which detects imports from the assignment script,
  installs any missing packages, and reports the compute device
- Reports NVIDIA GPU status (the graded run uses a Google Colab T4)
- Offers a quick smoke test or the full run after setup

.NOTES
Run from the project folder:
  powershell -ExecutionPolicy Bypass -File .\Install-Assignment_04_Three_Generative_Recipes_Windows.ps1
Unattended:
  ... -File .\Install-Assignment_04_Three_Generative_Recipes_Windows.ps1 -Launch none
#>
#=============================================================================
#endregion
#=============================================================================
#region Prerequisites
#=============================================================================

$OS = (Get-CimInstance -ClassName Win32_OperatingSystem).Caption
$Windows = ($OS -match 'Windows')
if (!$Windows) {
    Write-Output 'OS is not Windows. This script is only intended for Windows devices'
    exit 666
}

#=============================================================================
#endregion
#=============================================================================
#region VARIABLES
#=============================================================================

$EnableLogging = $False
$LogFileName = 'Install-Assignment_04_Three_Generative_Recipes_Windows.log'
$LogFile = Join-Path -Path $ScriptDir -ChildPath $LogFileName
$AssignmentFile = Join-Path $ScriptDir 'scripts\Assignment_04_Three_Generative_Recipes.py'
$EnvSetupFile = Join-Path $ScriptDir 'env_setup.py'
$VenvDir = Join-Path $ScriptDir '.venv'
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
$MinMajor = 3
$MinMinor = 10

#=============================================================================
#endregion
#=============================================================================
#region FUNCTIONS
#=============================================================================

function Write-Log {
    [CmdletBinding()]
    param (
        [AllowEmptyString()]
        [Parameter(Mandatory = $true,Position = 0)][String]$LogText,
        [Alias('ForegroundColor')]
        [Parameter(Mandatory = $false,Position = 1)][System.ConsoleColor]$Color = [System.ConsoleColor]::White
    )
    if ($EnableLogging) {
        $CurrentTime = Get-Date
        Add-Content $LogFile "$CurrentTime - $LogText"
    }
    Write-Host $LogText -ForegroundColor $Color
}

function Stop-WithError {
    param([string]$Message)
    Write-Log $Message -ForegroundColor Red
    if (-not $Launch) { Read-Host '  Press Enter to exit' | Out-Null }
    exit 1
}

#=============================================================================
#endregion
#=============================================================================
#region EXECUTION
#=============================================================================

if ($EnableLogging) {
    if (!(Test-Path $LogFile)) { New-Item -ItemType File -Path $LogFile -Force | Out-Null }
}

if (-not $Launch) { Clear-Host }
Write-Log 'Assignment 4 Three Generative Recipes - Windows Installer' -ForegroundColor Cyan
Write-Log ''
Write-Log '  This script will verify your Python environment, create a local .venv,'
Write-Log '  install the dependencies detected from the assignment script, and prepare:'
Write-Log '    * scripts\Assignment_04_Three_Generative_Recipes.py'

#=============================================================================
#1. Locate Python
#=============================================================================
Write-Log ''
Write-Log "Checking for Python $MinMajor.$MinMinor+..." -ForegroundColor Cyan

$pythonCmd = $null
$candidates = @(@('py', '-3'), @('python'), @('python3'))

foreach ($cand in $candidates) {
    $exe = $cand[0]
    $preArgs = @($cand | Select-Object -Skip 1)
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
    try {
        $ver = (& $exe @preArgs --version 2>&1) | Out-String
        if ($ver -match 'Python (\d+)\.(\d+)') {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            if ($major -gt $MinMajor -or ($major -eq $MinMajor -and $minor -ge $MinMinor)) {
                $pythonCmd = $cand
                Write-Log "Found: $($ver.Trim())  (using '$($cand -join ' ')')" -ForegroundColor Green
                break
            } else {
                Write-Log "Found $($ver.Trim()) but Python $MinMajor.$MinMinor+ is required - skipping." -ForegroundColor Yellow
            }
        }
    } catch { }
}

if (-not $pythonCmd) {
    Write-Log "Python $MinMajor.$MinMinor+ was not found on this machine." -ForegroundColor Red
    Write-Log ''
    Write-Log "  IMPORTANT: During install, tick 'Add Python to PATH'."
    if (-not $Launch) {
        $open = Read-Host '  Open the Python download page now? [Y/N]'
        if ($open -match '^[Yy]') { Start-Process 'https://www.python.org/downloads/' }
    }
    Write-Log '  Re-run this installer after Python is installed.'
    exit 1
}

#=============================================================================
#2. Check application files
#=============================================================================
Write-Log ''
Write-Log 'Checking application files...' -ForegroundColor Cyan

foreach ($f in @($AssignmentFile, $EnvSetupFile)) {
    if (Test-Path $f) {
        Write-Log "$(Split-Path $f -Leaf) found" -ForegroundColor Green
    } else {
        Stop-WithError "$(Split-Path $f -Leaf) not found under $ScriptDir. Run the installer from the project folder."
    }
}

#=============================================================================
#3. Create / reuse the virtual environment
#=============================================================================
Write-Log ''
Write-Log 'Preparing virtual environment (.venv)...' -ForegroundColor Cyan

if (Test-Path $VenvPython) {
    Write-Log 'Existing .venv found - reusing it.' -ForegroundColor Green
} else {
    $exe = $pythonCmd[0]
    $preArgs = @($pythonCmd | Select-Object -Skip 1)
    & $exe @preArgs -m venv $VenvDir
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $VenvPython)) { Stop-WithError 'Failed to create .venv.' }
    Write-Log 'Created .venv' -ForegroundColor Green
}

#=============================================================================
#4. Verify & upgrade pip
#=============================================================================
Write-Log ''
Write-Log 'Checking pip...' -ForegroundColor Cyan

& $VenvPython -m ensurepip --upgrade 2>$null | Out-Null
& $VenvPython -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -eq 0) { Write-Log 'pip is ready and up to date.' -ForegroundColor Green }
else { Stop-WithError 'pip upgrade failed (check your internet connection / proxy).' }

#=============================================================================
#5. Detect & install required packages (env_setup.py parses the assignment imports)
#=============================================================================
Write-Log ''
Write-Log 'Detecting & installing packages (this can take a few minutes the first time: torch is large)...' -ForegroundColor Cyan

Push-Location $ScriptDir
& $VenvPython $EnvSetupFile
$setupExit = $LASTEXITCODE
Pop-Location
if ($setupExit -ne 0) { Stop-WithError 'env_setup.py failed - see the messages above.' }

#=============================================================================
#6. Verify imports
#=============================================================================
Write-Log ''
Write-Log 'Verifying imports...' -ForegroundColor Cyan

$allGood = $true
foreach ($importName in @('torch', 'numpy', 'pandas', 'matplotlib')) {
    & $VenvPython -c "import $importName" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Log "Import verification failed: $importName" -ForegroundColor Red
        $allGood = $false
    } else { Write-Log "$importName imports successfully" -ForegroundColor Green }
}

#=============================================================================
#7. GPU report
#=============================================================================
Write-Log ''
Write-Log 'Checking for an NVIDIA GPU...' -ForegroundColor Cyan

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpu = (& nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null | Select-Object -First 1)
    Write-Log "NVIDIA GPU detected: $gpu" -ForegroundColor Green
    Write-Log '  The default Windows torch wheel is CPU-only. To use this GPU locally, install the CUDA' -ForegroundColor Yellow
    Write-Log '  build into .venv with the command from https://pytorch.org/get-started/locally/' -ForegroundColor Yellow
} else {
    Write-Log 'No NVIDIA GPU - the local run uses the CPU (about 15 min for the full run).' -ForegroundColor Gray
}
Write-Log '  The graded run is on Google Colab with a T4 GPU - see COLAB.md.' -ForegroundColor Gray

#=============================================================================
#8. Summary & launch prompt
#=============================================================================
Write-Log ''
Write-Log 'Setup Complete' -ForegroundColor Cyan

if (-not $allGood) {
    Write-Log ''
    Stop-WithError 'One or more packages could not be verified.'
}

$choice = $Launch
if (-not $choice) {
    Write-Log ''
    Write-Log '  Everything is ready. What would you like to run?' -ForegroundColor White
    Write-Log ''
    Write-Log '  [1] Quick smoke test (~2-3 min, short training - NOT submission numbers)' -ForegroundColor Gray
    Write-Log '  [2] Full run (~15 min on CPU; writes results\*_cpu.*)' -ForegroundColor Gray
    Write-Log '  [3] Exit' -ForegroundColor Gray
    Write-Log ''
    switch (Read-Host '  Enter choice [1/2/3]') {
        '1' { $choice = 'quick' }
        '2' { $choice = 'full' }
        default { $choice = 'none' }
    }
}

Push-Location $ScriptDir
switch ($choice) {
    'quick' {
        Write-Log ''
        Write-Log 'Running quick smoke test...' -ForegroundColor Cyan
        $env:HALDEN_QUICK = '1'
        & $VenvPython $AssignmentFile
        Remove-Item Env:\HALDEN_QUICK
    }
    'full' {
        Write-Log ''
        Write-Log 'Running full pipeline...' -ForegroundColor Cyan
        & $VenvPython $AssignmentFile
    }
    default {
        Write-Log ''
        Write-Log '  To run manually:' -ForegroundColor White
        Write-Log "    .venv\Scripts\python.exe scripts\Assignment_04_Three_Generative_Recipes.py" -ForegroundColor Cyan
        Write-Log '  Quick smoke test:' -ForegroundColor White
        Write-Log "    `$env:HALDEN_QUICK='1'; .venv\Scripts\python.exe scripts\Assignment_04_Three_Generative_Recipes.py" -ForegroundColor Cyan
    }
}
Pop-Location

Write-Log ''
#=============================================================================
#endregion
#=============================================================================
