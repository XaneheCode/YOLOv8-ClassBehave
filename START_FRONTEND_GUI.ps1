$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
.\.venv\Scripts\python.exe -m src.frontend.gui_client
