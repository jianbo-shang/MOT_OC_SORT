$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PreferredPython = 'D:\miniconda\envs\PJT_1\python.exe'
$PythonExe = if (Test-Path -LiteralPath $PreferredPython) { $PreferredPython } else { 'python' }

$env:YOLO_CONFIG_DIR = Join-Path $ProjectRoot '.runtime\ultralytics'
$env:YOLO_AUTOINSTALL = 'False'
$env:YOLO_VERBOSE = 'False'

Set-Location -LiteralPath $ProjectRoot
& $PythonExe (Join-Path $ProjectRoot 'mot_app.py') @args
exit $LASTEXITCODE
