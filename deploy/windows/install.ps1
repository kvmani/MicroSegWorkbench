# MicroSegWorkbench - offline install for a Windows workstation (per user, no administrator rights needed).
#
# Usage (PowerShell, from the folder that holds this script, the wheelhouse and the source):
#   powershell -ExecutionPolicy Bypass -File install.ps1 -Wheelhouse ..\..\wheelhouse -Source ..\..
#
# Creates %LOCALAPPDATA%\MicroSegWorkbench\venv, installs from the wheelhouse only (never the internet),
# and puts a "MicroSegWorkbench" shortcut on the desktop. Re-running upgrades in place; workspaces and
# settings are not touched.
param(
    [Parameter(Mandatory = $true)][string]$Wheelhouse,
    [Parameter(Mandatory = $true)][string]$Source,
    [string]$Python = "py -3.12"
)
$ErrorActionPreference = "Stop"
$Root = Join-Path $env:LOCALAPPDATA "MicroSegWorkbench"
$Venv = Join-Path $Root "venv"
New-Item -ItemType Directory -Force -Path $Root | Out-Null

if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
    Write-Host "Creating Python environment in $Venv"
    Invoke-Expression "$Python -m venv `"$Venv`""
}
$Py = Join-Path $Venv "Scripts\python.exe"
function Invoke-Checked { param([scriptblock]$Block, [string]$What)
    & $Block
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit code $LASTEXITCODE)" }
}
Invoke-Checked { & $Py -m pip install --no-index --find-links "$Wheelhouse" --upgrade pip } "Upgrading pip"
Invoke-Checked { & $Py -m pip install --no-index --find-links "$Wheelhouse" -r (Join-Path $Source "deploy\requirements-desktop.txt") } "Installing dependencies"
Invoke-Checked { & $Py -m pip install --no-index --find-links "$Wheelhouse" --no-deps "$Source" } "Installing MicroSegWorkbench"

Write-Host "Checking the installation..."
Invoke-Checked { & $Py -m microseg_workbench devices } "Checking the installation"

$Launcher = Join-Path $Venv "Scripts\pythonw.exe"
$Desktop = [Environment]::GetFolderPath("Desktop")
if (-not $Desktop) { $Desktop = Join-Path $env:USERPROFILE "Desktop" }
New-Item -ItemType Directory -Force -Path $Desktop | Out-Null
$Shell = New-Object -ComObject WScript.Shell
$Link = $Shell.CreateShortcut((Join-Path $Desktop "MicroSegWorkbench.lnk"))
$Link.TargetPath = $Launcher
$Link.Arguments = "-m microseg_workbench.gui.app"
$Link.WorkingDirectory = $Root
$Link.Description = "MicroSegWorkbench - microstructure segmentation"
$Link.Save()
Write-Host "Shortcut: $(Join-Path $Desktop 'MicroSegWorkbench.lnk')"
Write-Host "Done. Start MicroSegWorkbench from the desktop shortcut."
