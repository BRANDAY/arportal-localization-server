# Puts a shortcut "AR Portal Server" on the desktop of the current user. It points to "Start AR Portal Server.cmd"
# in this folder. Run it by a double click on "Create Desktop Shortcut.cmd"; nothing else on the PC is changed.
# Remove the shortcut by deleting it from the desktop.
param([string]$Folder = "")   # another folder instead of the desktop (used by the test)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$target = Join-Path $repo "Start AR Portal Server.cmd"
if (-not (Test-Path $target)) { Write-Host "Not found: $target" -ForegroundColor Red; exit 1 }
$desktop = if ($Folder) { $Folder } else { [Environment]::GetFolderPath("Desktop") }
$link = Join-Path $desktop "AR Portal Server.lnk"
$shell = New-Object -ComObject WScript.Shell
$s = $shell.CreateShortcut($link)
$s.TargetPath = $target
$s.WorkingDirectory = $repo
$s.Description = "Start the AR Portal localization server and its panel"
$s.IconLocation = "$env:SystemRoot\System32\imageres.dll,170"
$s.Save()
Write-Host ""
Write-Host "  Shortcut created: $link" -ForegroundColor Green
Write-Host "  Double click 'AR Portal Server' on the desktop to start the server."
