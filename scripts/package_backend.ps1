param(
    [string]$OutputDir = "dist",
    [string]$PackageName = "backend-student-sleep-server",
    [string]$ModelPath = "output\training\student_behaviour_yolov8n_e3\weights\best.pt"
)

$ErrorActionPreference = "Stop"

$workspace = (Resolve-Path ".").Path
$modelFullPath = (Resolve-Path -LiteralPath $ModelPath).Path
$distDir = Join-Path $workspace $OutputDir
$packageDir = Join-Path $distDir $PackageName
$zipPath = Join-Path $distDir "$PackageName.zip"

if (-not $modelFullPath.StartsWith($workspace, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to package model outside workspace: $modelFullPath"
}

New-Item -ItemType Directory -Force -Path $distDir | Out-Null

if (Test-Path -LiteralPath $packageDir) {
    $resolvedPackage = (Resolve-Path -LiteralPath $packageDir).Path
    if (-not $resolvedPackage.StartsWith($distDir, [System.StringComparison]::OrdinalIgnoreCase)) {
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
Copy-Item -LiteralPath $modelFullPath -Destination (Join-Path $packageDir "models\student_behaviour_yolov8n_best.pt")

Get-ChildItem -LiteralPath $packageDir -Recurse -Directory -Filter "__pycache__" |
    Remove-Item -Recurse -Force

@'
# 后端运行包

## 安装环境

```powershell
.\scripts\setup_env.ps1
```

## 启动后端

```powershell
.\START_BACKEND.ps1
```

默认监听 `0.0.0.0:5001`，加载 12 类课堂行为模型：

```txt
models\student_behaviour_yolov8n_best.pt
```

异常状态框显示为红色，正常状态框显示为绿色。

启动后，在前端电脑使用后端电脑的 IPv4 地址连接。
'@ | Set-Content -Encoding UTF8 (Join-Path $packageDir "README_BACKEND.md")

@'
$ErrorActionPreference = "Stop"
.\.venv\Scripts\python.exe -m src.backend.app --host 0.0.0.0 --port 5001 --model models\student_behaviour_yolov8n_best.pt
'@ | Set-Content -Encoding UTF8 (Join-Path $packageDir "START_BACKEND.ps1")

Compress-Archive -Path (Join-Path $packageDir "*") -DestinationPath $zipPath -Force

Write-Host "Backend package created:"
Write-Host "Directory: $packageDir"
Write-Host "Zip: $zipPath"
