# run-dev.ps1 — launch the ML Constructor backend + built frontend locally (Windows).
#
#   powershell -ExecutionPolicy Bypass -File run-dev.ps1
#
# Backend (FastAPI + PyTorch) -> http://localhost:8050
# Frontend (static build)     -> http://localhost:4200
#
# Node and Python must be available on PATH (see README).
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

$venvPy = Join-Path $root 'backend\.venv\Scripts\python.exe'
if (-not (Test-Path $venvPy)) {
  Write-Error "backend venv not found at $venvPy — see README 'Quickstart — backend'."
}

# 1) backend in its own window
Write-Host '[1/3] starting backend on http://localhost:8050 ...'
Start-Process -FilePath $venvPy `
  -ArgumentList '-m', 'uvicorn', 'app:app', '--host', '0.0.0.0', '--port', '8050' `
  -WorkingDirectory (Join-Path $root 'backend')

# 2) build the frontend if it has not been built yet
$browser = Join-Path $root 'frontend\dist\mlconstructor\browser'
if (-not (Test-Path (Join-Path $browser 'index.html'))) {
  Write-Host '[2/3] building frontend (first run) ...'
  Push-Location (Join-Path $root 'frontend')
  try { npx ng build } finally { Pop-Location }
} else {
  Write-Host '[2/3] frontend already built (delete frontend\dist to rebuild).'
}

# 3) serve the built frontend (foreground; Ctrl+C to stop)
Write-Host '[3/3] serving frontend on http://localhost:4200  (Ctrl+C to stop)'
python -m http.server 4200 --directory $browser
