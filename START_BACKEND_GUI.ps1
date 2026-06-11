$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
.\.venv\Scripts\python.exe -m src.backend.gui_app
