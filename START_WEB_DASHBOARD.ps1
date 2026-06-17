$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) {
            return
        }
        $key, $value = $line.Split("=", 2)
        [Environment]::SetEnvironmentVariable($key.Trim(), $value.Trim().Trim('"'), "Process")
    }
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "未找到 .venv，请先运行 scripts\setup_env.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "浏览器控制台：http://127.0.0.1:8765" -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m src.web.server --host 127.0.0.1 --port 8765
