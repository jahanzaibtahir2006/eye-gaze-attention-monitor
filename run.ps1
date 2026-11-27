# Eye Gaze Attention Monitor launcher.
# First run: creates .venv and installs dependencies. Any arguments pass through, e.g.
#   .\run.ps1 --record
#   .\run.ps1 --source samples\desk-worker.mp4 --calibrate
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
    & $py -m pip install --upgrade pip
    & $py -m pip install -r requirements.txt
}

$models = "face_det_lite.onnx", "facemap_3dmm.onnx", "eyegaze.onnx"
foreach ($m in $models) {
    if (-not (Test-Path (Join-Path $PSScriptRoot "models\$m"))) {
        Write-Host "Missing models\$m. See README: 'Re-exporting the models'." -ForegroundColor Red
        exit 1
    }
}

$env:PYTHONPATH = Join-Path $PSScriptRoot "src"
# Native programs log to stderr; with "Stop" PowerShell would treat that as a failure.
$ErrorActionPreference = "Continue"
& $py -m gaze_monitor @args
exit $LASTEXITCODE
