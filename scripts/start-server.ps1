# Starts the localization server and opens its panel. Meant to be run by "Start AR Portal Server.cmd" (double click);
# from a PowerShell window:  powershell -ExecutionPolicy Bypass -File scripts\start-server.ps1 [-Port 8077] [-NoPanel]
#
# What it does, in this order - and it says in plain words what is wrong if a step fails:
#   1. loads env.ps1 (and through it the untracked env.local.ps1) - where the large files are;
#   2. checks that uv and the Python environment are there;
#   3. if a server already answers on the port: says so and only opens the panel;
#   4. opens the panel (it shows "Server is starting..." and turns into the panel when the server answers);
#   5. runs the server in this window. Closing the window stops the server.
# Everything the phones send is kept (LOCSERVER_KEEP_INCOMING, default <work>\incoming) and everything shown in this
# window is also written to <work>\server.log, so that a failure can be looked at afterwards.
param(
    [int]$Port = 0,
    [switch]$NoPanel
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot   # not `$root`: env.ps1 has a parameter of that name
Set-Location $repo
try { $Host.UI.RawUI.WindowTitle = "AR Portal Server" } catch {}

function Say([string]$text, [string]$color = "Gray") { Write-Host $text -ForegroundColor $color }
function Fail([string]$what, [string[]]$todo) {
    Say ""
    Say "  COULD NOT START THE SERVER" Red
    Say "  $what" Red
    Say ""
    foreach ($line in $todo) { Say "  $line" Yellow }
    Say ""
    exit 1
}

Say ""
Say "  AR Portal localization server" Cyan
Say "  folder: $repo"
Say ""

# 1. where the large files are (optional file; without it everything lives inside this folder)
try { . (Join-Path $repo "env.ps1") } catch { Fail "env.ps1 could not be loaded: $($_.Exception.Message)" @("Check the file env.local.ps1 in this folder: it should only set `$env:LOCSERVER_ROOT.") }
if (-not $Port) { $Port = if ($env:LOCSERVER_PORT) { [int]$env:LOCSERVER_PORT } else { 8077 } }
if (-not $env:LOCSERVER_KEEP_INCOMING) { $env:LOCSERVER_KEEP_INCOMING = "1" }

# 2. the Python environment (uv is only needed to install it, not to run the server)
$venv = if ($env:UV_PROJECT_ENVIRONMENT) { $env:UV_PROJECT_ENVIRONMENT } else { Join-Path $repo ".venv" }
$python = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $python)) {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Fail "The Python environment is not installed yet, and the program 'uv' that installs it was not found." @(
            "1. Install uv (README.md, section 'Setup (Windows)').",
            "2. Open PowerShell in this folder and run:",
            "       powershell -ExecutionPolicy Bypass -Command `". .\env.ps1; uv sync`"",
            "   (it downloads several GB, mostly PyTorch). Then start again.")
    }
    Fail "The Python environment is not installed yet (looked for $python)." @(
        "Install it once: open PowerShell in this folder and run",
        "    powershell -ExecutionPolicy Bypass -Command `". .\env.ps1; uv sync`"",
        "(it downloads several GB, mostly PyTorch). Then start again.",
        "More: README.md, section 'Setup (Windows)'.")
}

function Get-Health {
    try { return Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 3 } catch { return $null }
}
function Open-Panel([switch]$Wait) {
    if ($NoPanel) { return }
    $more = if ($Wait) { @("--wait") } else { @() }
    try { & $python -m locserver.panel --port $Port @more | Out-Null } catch { Say "  (the panel could not be opened: open http://127.0.0.1:$Port/ui in a browser)" Yellow }
}

# 3. already running?
$health = Get-Health
if ($health -and $health.service -eq "arportal-localization-server") {
    Say "  THE SERVER IS ALREADY RUNNING (version $($health.server_version)) - nothing was started." Green
    Say ""
    Say "  Panel on this PC:  http://127.0.0.1:$Port/ui"
    Say "  Address for the phone:"
    try { & $python -c "from locserver import netinfo; print(*['      http://%s:$Port' % a for a in netinfo.lan_ipv4_addresses()] or ['      none - this PC is not on a network'], sep=chr(10))" } catch {}
    Say "  To stop that server, close the window it was started in."
    Open-Panel
    exit 0
}
$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($listener) {
    $who = try { (Get-Process -Id $listener.OwningProcess).ProcessName } catch { "unknown" }
    Fail "Port $Port is used by another program ($who, process $($listener.OwningProcess))." @(
        "Close that program, or start the server on another port:",
        "    `"Start AR Portal Server.cmd`" -Port $($Port + 1)",
        "(the phone then needs the address with that port typed in by hand).")
}

# 4. + 5. the panel waits for the server; the server runs here
$work = if ($env:LOCSERVER_WORK) { $env:LOCSERVER_WORK } else { Join-Path $repo "data\work" }
if (-not (Test-Path $work)) { New-Item -ItemType Directory -Force $work | Out-Null }
$log = Join-Path $work "server.log"
Say "  STARTING ...  (the panel opens now and shows when the server is ready)" Yellow
Say "  this window's text is also saved in $log"
Open-Panel -Wait
$ErrorActionPreference = "Continue"
# through cmd so that the server's messages arrive as plain text (PowerShell would wrap every line of the error stream)
cmd /c "`"$python`" -m locserver.server --port $Port 2>&1" | Tee-Object -FilePath $log
$code = $LASTEXITCODE
Say ""
if ($code -eq 0) {
    Say "  The server has stopped." Yellow
} else {
    Say "  THE SERVER STOPPED WITH AN ERROR (code $code)." Red
    Say "  The reason is in the lines above (look for 'COULD NOT START' or the last 'Error'); they are also in" Red
    Say "  $log" Red
    Say "  Typical reasons: the graphics card is full (close games), or the environment is incomplete (run 'uv sync')." Yellow
}
exit $code
