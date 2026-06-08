param(
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"

function Add-Candidate {
    param([System.Collections.Generic.List[string]]$Candidates, [string]$Path)
    if ($Path -and (Test-Path -LiteralPath $Path) -and -not $Candidates.Contains($Path)) {
        $Candidates.Add($Path)
    }
}

$candidates = [System.Collections.Generic.List[string]]::new()
Add-Candidate $candidates $Python

try {
    $pyLauncherPath = (& py -3.12 -c "import sys; print(sys.executable)" 2>$null).Trim()
    if ($LASTEXITCODE -eq 0) {
        Add-Candidate $candidates $pyLauncherPath
    }
} catch {
}

Add-Candidate $candidates "$env:APPDATA\uv\python\cpython-3.12-windows-x86_64-none\python.exe"
Add-Candidate $candidates "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
Add-Candidate $candidates "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"

$selectedPython = $null
foreach ($candidate in $candidates) {
    try {
        $version = (& $candidate -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
        if ($version -eq "3.12") {
            $selectedPython = $candidate
            break
        }
    } catch {
    }
}

if (-not $selectedPython) {
    throw "Python 3.12 was not found. Install Python 3.12, or run: .\scripts\setup_env.ps1 -Python C:\Path\To\python.exe"
}

Write-Host "Using Python: $selectedPython"
& $selectedPython --version

$workspace = (Resolve-Path ".").Path
$venvPath = Join-Path $workspace ".venv"
if (Test-Path -LiteralPath $venvPath) {
    $resolvedVenv = (Resolve-Path -LiteralPath $venvPath).Path
    if (-not $resolvedVenv.StartsWith($workspace, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove virtual environment outside workspace: $resolvedVenv"
    }
    Remove-Item -LiteralPath $resolvedVenv -Recurse -Force
}

& $selectedPython -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip config --site set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
.\.venv\Scripts\python.exe -m pip config --site set global.timeout 120
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt

Write-Host "Environment ready."
.\.venv\Scripts\python.exe --version
