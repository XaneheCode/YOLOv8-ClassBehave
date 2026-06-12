param(
    [string]$OutputDir = "dist",
    [string]$PackageName = "backend-student-sleep-server",
    [string]$ModelPath = "models\student_behaviour_v6_6cls_img960_e50_best.pt"
)

$ErrorActionPreference = "Stop"

function Test-IsSubPath {
    param(
        [Parameter(Mandatory=$true)][string]$Parent,
        [Parameter(Mandatory=$true)][string]$Child
    )
    $parentFull = [System.IO.Path]::GetFullPath($Parent).TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
    $childFull = [System.IO.Path]::GetFullPath($Child).TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
    return $childFull.Equals($parentFull.TrimEnd([System.IO.Path]::DirectorySeparatorChar), [System.StringComparison]::OrdinalIgnoreCase) -or $childFull.StartsWith($parentFull, [System.StringComparison]::OrdinalIgnoreCase)
}

$workspace = (Resolve-Path ".").Path
$modelFullPath = (Resolve-Path -LiteralPath $ModelPath).Path
$distDir = Join-Path $workspace $OutputDir
$packageDir = Join-Path $distDir $PackageName
$zipPath = Join-Path $distDir "$PackageName.zip"
$readmePath = Join-Path $packageDir "README_BACKEND.md"
$startBackendPath = Join-Path $packageDir "START_BACKEND.ps1"

if (-not (Test-IsSubPath -Parent $workspace -Child $modelFullPath)) {
    throw "Refusing to package model outside workspace: $modelFullPath"
}

New-Item -ItemType Directory -Force -Path $distDir | Out-Null

if (Test-Path -LiteralPath $packageDir) {
    $resolvedPackage = (Resolve-Path -LiteralPath $packageDir).Path
    if (-not (Test-IsSubPath -Parent $distDir -Child $resolvedPackage)) {
        throw "Refusing to remove package outside dist directory: $resolvedPackage"
    }
    Remove-Item -LiteralPath $resolvedPackage -Recurse -Force
}

if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}

New-Item -ItemType Directory -Force -Path $packageDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $packageDir "src") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $packageDir "models") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $packageDir "scripts") | Out-Null

Copy-Item -LiteralPath "src\__init__.py" -Destination (Join-Path $packageDir "src\__init__.py")
Copy-Item -LiteralPath "src\backend" -Destination (Join-Path $packageDir "src") -Recurse
Copy-Item -LiteralPath "src\common" -Destination (Join-Path $packageDir "src") -Recurse
Copy-Item -LiteralPath "requirements.txt" -Destination (Join-Path $packageDir "requirements.txt")
Copy-Item -LiteralPath "scripts\setup_env.ps1" -Destination (Join-Path $packageDir "scripts\setup_env.ps1")
Copy-Item -LiteralPath "START_BACKEND_GUI.ps1" -Destination (Join-Path $packageDir "START_BACKEND_GUI.ps1")
Copy-Item -LiteralPath $modelFullPath -Destination (Join-Path $packageDir "models\student_behaviour_v6_6cls_img960_e50_best.pt")

Get-ChildItem -LiteralPath $packageDir -Recurse -Directory -Filter "__pycache__" |
    Remove-Item -Recurse -Force

$readmeContent = @'
# Backend runtime package

## Install

```powershell
.\scripts\setup_env.ps1
```

## Start backend GUI

```powershell
.\START_BACKEND_GUI.ps1
```

## Start backend service

```powershell
.\START_BACKEND.ps1
```

Default listen address: `0.0.0.0:5001`.

Default 50-epoch six-class classroom behaviour model:

```txt
models\student_behaviour_v6_6cls_img960_e50_best.pt
```

Abnormal people are highlighted in red. Normal people stay green.

After startup, connect from the frontend computer with this backend computer's IPv4 address.
'@
Set-Content -Path $readmePath -Encoding UTF8 -Value $readmeContent

$startBackendContent = @'
$ErrorActionPreference = "Stop"
.\.venv\Scripts\python.exe -m src.backend.app --host 0.0.0.0 --port 5001 --model models\student_behaviour_v6_6cls_img960_e50_best.pt
'@
Set-Content -Path $startBackendPath -Encoding UTF8 -Value $startBackendContent

Compress-Archive -Path (Join-Path $packageDir "*") -DestinationPath $zipPath -Force

Write-Host "Backend package created:"
Write-Host "Directory: $packageDir"
Write-Host "Zip: $zipPath"
