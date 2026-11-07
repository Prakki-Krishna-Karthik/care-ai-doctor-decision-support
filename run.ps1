$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:MIMIC_IMAGE_ROOT = if ($env:MIMIC_IMAGE_ROOT) { $env:MIMIC_IMAGE_ROOT } else { 'D:\MIMIC\jpg' }
$env:MIMIC_DATA_ROOT = if ($env:MIMIC_DATA_ROOT) { $env:MIMIC_DATA_ROOT } else { 'C:\Users\Krishna Karthik\Downloads' }
python "$project\backend\app.py"
