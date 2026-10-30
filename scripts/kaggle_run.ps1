# Push kaggle/train.py to Kaggle, wait for it, download outputs to out/<stage>/.
# Usage (PowerShell, from project root):
#   .\scripts\kaggle_run.ps1 -Stage probe
#   .\scripts\kaggle_run.ps1 -Stage baseline
#   .\scripts\kaggle_run.ps1 -Stage all [-Steps 600] [-Corti]
param(
  [ValidateSet("probe","baseline","all")][string]$Stage = "probe",
  [int]$Steps = 600,
  [switch]$Corti,
  [int]$PollSeconds = 60,
  [int]$TimeoutMinutes = 180
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# --- resolve Kaggle username (env var, else ~/.kaggle/kaggle.json) ---
$user = $env:KAGGLE_USERNAME
if (-not $user) {
  $kj = Join-Path $HOME ".kaggle\kaggle.json"
  if (Test-Path $kj) { $user = (Get-Content $kj | ConvertFrom-Json).username }
}
if (-not $user) { throw "Kaggle username not found. Set `$env:KAGGLE_USERNAME or create ~/.kaggle/kaggle.json" }
$slug = "$user/whisper-medasr-lora"

# --- stamp config into train.py and metadata ---
$py = Get-Content kaggle\train.py -Raw
$py = [regex]::Replace($py, '(?m)^STAGE = "[^"]*"', "STAGE = `"$Stage`"")
$py = [regex]::Replace($py, '(?m)^MAX_STEPS = \d+', "MAX_STEPS = $Steps")
$cv = if ($Corti) { "True" } else { "False" }
$py = [regex]::Replace($py, '(?m)^RUN_CORTI = \w+', "RUN_CORTI = $cv")
# PS 5.1 "-Encoding utf8" writes a BOM, which the Kaggle CLI rejects in kernel-metadata.json -> write BOM-less
$noBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText((Join-Path $root "kaggle\train.py"), $py, $noBom)
$meta = Get-Content kaggle\kernel-metadata.json -Raw
$meta = [regex]::Replace($meta, '"id": "[^"]*"', "`"id`": `"$slug`"")
[System.IO.File]::WriteAllText((Join-Path $root "kaggle\kernel-metadata.json"), $meta, $noBom)

Write-Host ">> pushing $slug  stage=$Stage steps=$Steps corti=$cv"
kaggle kernels push -p kaggle
if ($LASTEXITCODE -ne 0) { throw "kaggle push failed" }

# --- poll ---
$deadline = (Get-Date).AddMinutes($TimeoutMinutes)
Start-Sleep -Seconds 20
while ($true) {
  $s = (kaggle kernels status $slug) 2>&1 | Out-String
  Write-Host ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $s.Trim())
  if ($s -match "COMPLETE") { break }
  if ($s -match "ERROR|CANCEL") { Write-Host "!! kernel failed - downloading log anyway"; break }
  if ((Get-Date) -gt $deadline) { Write-Host "!! local timeout; kernel may still be running"; break }
  Start-Sleep -Seconds $PollSeconds
}

# --- download ---
$out = "out\$Stage"
New-Item -ItemType Directory -Force $out | Out-Null
kaggle kernels output $slug -p $out
Write-Host ">> outputs in $out"
Get-ChildItem $out | Select-Object Name, Length
