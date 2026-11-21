"""Starting the server the way a Windows user does: a double click, a window that says what goes on, a panel that waits."""

from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path

import pytest

from locserver import netinfo, panel, server

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "src" / "locserver" / "ui"


def test_the_window_says_that_the_server_runs_what_to_type_and_how_to_stop():
    text = "\n".join(netinfo.ready_lines("0.0.0.0", 8077, ["192.168.1.50", "10.0.0.7"], "PC", "0.7.2"))
    assert "SERVER IS RUNNING" in text and '"PC", version 0.7.2' in text
    assert "ADDRESS FOR THE PHONE" in text and "http://192.168.1.50:8077" in text and "http://10.0.0.7:8077" in text
    assert "same network as the phone" in text and "http://192.168.1.50:8077/health" in text
    assert "http://127.0.0.1:8077/ui" in text and "TO STOP THE SERVER: close this window" in text
    one = "\n".join(netinfo.ready_lines("0.0.0.0", 9000, ["192.168.1.50"], "PC"))
    assert "http://192.168.1.50:9000" in one and "same network as the phone" not in one
    assert "NO NETWORK" in "\n".join(netinfo.ready_lines("0.0.0.0", 8077, [], "PC"))
    assert "THIS PC ONLY" in "\n".join(netinfo.ready_lines("127.0.0.1", 8077, ["192.168.1.50"], "PC"))


def test_a_busy_port_is_said_in_plain_words_before_anything_is_loaded(monkeypatch, capsys):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        assert not netinfo.port_free("127.0.0.1", port)
        monkeypatch.setattr(server, "create_app", lambda *a, **k: pytest.fail("the models must not be loaded"))
        assert server.main(["--host", "127.0.0.1", "--port", str(port)]) == 3
    out = capsys.readouterr().out
    assert "STARTING" in out and f"COULD NOT START: port {port} is already in use" in out and "Another program listens" in out
    assert netinfo.port_free("127.0.0.1", port)


def test_the_panel_can_be_opened_before_the_server_answers(monkeypatch):
    started = []
    monkeypatch.setattr(panel, "find_app_browser", lambda: "browser.exe")
    monkeypatch.setattr(panel.subprocess, "Popen", lambda cmd, **kw: started.append(cmd))
    assert panel.main(["--port", "8078", "--wait"]) == 0          # no check: the server is not there yet
    url = started[0][1].removeprefix("--app=")
    assert url.startswith("file:///") and url.endswith("/ui/starting.html?port=8078")
    page = (UI / "starting.html").read_text(encoding="utf-8")
    # the page asks the server on this PC and goes over to its panel; it is self-contained (works without internet)
    assert '"/health"' in page and 'location.replace(base + "/ui")' in page and "Server is starting" in page
    assert "Server is not running" in page and "https://" not in page and "src=" not in page


def test_the_panel_has_a_strip_that_tells_running_from_not_running():
    page = (UI / "index.html").read_text(encoding="utf-8")
    for needle in ('id="status"', "Server is running", "Server is not running", "Address for the phone:", "Phones connected now:",
                   "Map build:", "location.reload()"):
        assert needle in page, needle


def test_the_files_for_a_double_click_are_what_windows_needs():
    for name in ("Start AR Portal Server.cmd", "Create Desktop Shortcut.cmd"):
        raw = (ROOT / name).read_bytes()
        assert b"-ExecutionPolicy Bypass" in raw and b"%~dp0scripts\\" in raw and b"pause" in raw   # works with the default script policy
        assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"") and raw.isascii()            # cmd.exe wants CRLF
    ps = (ROOT / "scripts" / "start-server.ps1").read_text(encoding="utf-8")
    assert "$root" not in ps.replace("`$root`", "")       # env.ps1 has a parameter $Root: the same name here would be overwritten
    for needle in ("env.ps1", "ALREADY RUNNING", "COULD NOT START THE SERVER", "uv sync", "--wait", "server.log", "Get-NetTCPConnection"):
        assert needle in ps, needle
    assert (ROOT / ".gitattributes").read_text(encoding="utf-8").count("eol=crlf") >= 2


@pytest.mark.skipif(sys.platform != "win32", reason="Windows shortcut")
def test_the_shortcut_script_makes_a_shortcut_to_the_start_file(tmp_path):
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "scripts" / "make-desktop-shortcut.ps1"),
                        "-Folder", str(tmp_path)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "AR Portal Server.lnk").stat().st_size > 100 and "Shortcut created" in r.stdout
