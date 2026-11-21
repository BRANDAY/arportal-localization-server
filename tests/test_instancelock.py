"""One server per data folder: the second one must not start, and a dead one must not stand in the way."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys

import pytest

from locserver import instancelock, server
from locserver.instancelock import FolderInUse, InstanceLock, lock_folders, release_all

# what a second process does: try the lock of the folder, say who holds it
TRY = """
import json, sys
from locserver.instancelock import FolderInUse, InstanceLock
try:
    InstanceLock(sys.argv[1], port=9001).acquire()
except FolderInUse as e:
    print(json.dumps(e.holder))
    raise SystemExit(7)
print("taken")
"""
# a process that takes the lock and dies without giving it up
CRASH = """
import os, sys
from locserver.instancelock import InstanceLock
InstanceLock(sys.argv[1], port=9002).acquire()
os._exit(1)
"""


def run(code, folder):
    return subprocess.run([sys.executable, "-c", code, str(folder)], capture_output=True, text=True, timeout=60)


def test_two_locks_on_one_folder_conflict_and_a_released_one_can_be_taken_again(tmp_path):
    folder = tmp_path / "data" / "maps"   # does not exist yet: a fresh clone
    first = InstanceLock(folder, port=8077).acquire()
    assert first.held and first.acquire() is first and (folder / instancelock.LOCK_FILE).exists()
    with pytest.raises(FolderInUse) as e:
        InstanceLock(folder, port=9000).acquire()
    assert e.value.folder == folder.resolve() and e.value.holder["pid"] == os.getpid() and e.value.holder["port"] == 8077
    text = instancelock.describe(e.value.holder)
    assert f"process {os.getpid()} on port 8077, started 20" in text and str(folder.resolve()) in str(e.value)
    first.release()
    first.release()   # twice does no harm
    assert not first.held
    with InstanceLock(folder, port=9000) as second:   # the file of the first is still there and does not block
        assert second.held and json.loads((folder / instancelock.LOCK_FILE).read_text())["port"] == 9000
        with pytest.raises(FolderInUse):
            InstanceLock(folder).acquire()
    assert not second.held
    assert instancelock.describe({}) == "another process"


def test_several_folders_are_locked_all_or_none(tmp_path):
    work, maps = tmp_path / "work", tmp_path / "maps"
    locks = lock_folders([work, maps, tmp_path / "work" / ".." / "work"], port=8077)   # the same folder named twice
    assert [x.folder for x in locks] == [work.resolve(), maps.resolve()] and all(x.held for x in locks)
    release_all(locks)
    with InstanceLock(maps):
        with pytest.raises(FolderInUse) as e:
            lock_folders([work, maps])
        assert e.value.folder == maps.resolve()
        InstanceLock(work).acquire().release()   # the one that was taken first has been given back


def test_a_second_process_cannot_take_the_lock_and_a_crashed_one_leaves_it_free(tmp_path):
    with InstanceLock(tmp_path, port=8077):
        r = run(TRY, tmp_path)
        assert r.returncode == 7, r.stdout + r.stderr
        holder = json.loads(r.stdout)
        assert holder["pid"] == os.getpid() and holder["port"] == 8077 and holder["started"]
    r = run(TRY, tmp_path)   # given up: the other process gets it, and gives it up by ending
    assert r.returncode == 0 and "taken" in r.stdout, r.stdout + r.stderr
    assert run(CRASH, tmp_path).returncode == 1
    assert json.loads((tmp_path / instancelock.LOCK_FILE).read_text())["port"] == 9002   # the dead one's file is still there
    InstanceLock(tmp_path).acquire().release()


# ------------------------------------------------------------- the server --- #
def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_a_second_server_on_the_same_folders_refuses_to_start(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LOCSERVER_WORK", str(tmp_path / "work"))
    monkeypatch.setattr(server, "create_app", lambda *a, **k: pytest.fail("nothing may be tidied up or loaded"))
    args = ["--host", "127.0.0.1", "--port", str(free_port()), "--maps-dir", str(tmp_path / "maps")]
    for folder in ("maps", "work"):   # either folder in use is enough
        with InstanceLock(tmp_path / folder, port=8077):
            assert server.main(args) == 4
        out = capsys.readouterr().out
        assert "COULD NOT START: another localization server already works with the data folder" in out
        assert str((tmp_path / folder).resolve()) in out and f"process {os.getpid()} on port 8077, started" in out
        assert "http://127.0.0.1:8077/ui" in out
        for other in ("maps", "work"):   # and nothing stays locked by the one that did not start
            InstanceLock(tmp_path / other).acquire().release()


def test_a_server_that_fails_while_loading_gives_its_folders_back(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LOCSERVER_WORK", str(tmp_path / "work"))
    held = []

    def no_models(*a, **k):
        with pytest.raises(FolderInUse):   # while it loads, the folders are its own
            InstanceLock(tmp_path / "maps").acquire()
        held.append(True)
        raise RuntimeError("no graphics card")

    monkeypatch.setattr(server, "create_app", no_models)
    with pytest.raises(RuntimeError):
        server.main(["--host", "127.0.0.1", "--port", str(free_port()), "--maps-dir", str(tmp_path / "maps")])
    assert held and "COULD NOT START: RuntimeError: no graphics card" in capsys.readouterr().out
    for folder in ("maps", "work"):
        InstanceLock(tmp_path / folder).acquire().release()
