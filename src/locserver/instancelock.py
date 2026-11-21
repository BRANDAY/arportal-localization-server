"""One server per data folder.

At its start a server tidies up unfinished builds in its maps and work
folders. A second server on the same folders would take the running build of
the first one for a leftover and delete it. So a server locks its data folders
before it touches them, and a second one refuses to start.

The lock is a lock of the operating system on an open file, held for the life
of the process. When the process ends - or crashes - the system gives the lock
up, so the file a dead server left behind never blocks the next start. The
file also says who holds the lock (process id, port, start time); that text is
only for the message.
"""

from __future__ import annotations

import errno
import json
import os
import sys
import time
from pathlib import Path

LOCK_FILE = ".locserver.lock"
# Windows keeps other processes from reading a locked byte. So the byte that is locked lies far behind the text.
_LOCK_BYTE = 2**30
_BUSY = (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK, errno.EDEADLK)


class FolderInUse(RuntimeError):
    """Another running process holds the lock of ``folder``; ``holder`` is what it wrote about itself (may be empty)."""

    def __init__(self, folder: Path, holder: dict):
        self.folder, self.holder = folder, holder
        super().__init__(f"{folder} is in use by {describe(holder)}")


def describe(holder: dict) -> str:
    """``process 1234 on port 8077, started 2026-11-21 14:03:22`` - as far as the holder said it."""
    text = f"process {holder['pid']}" if holder.get("pid") else "another process"
    if holder.get("port"):
        text += f" on port {holder['port']}"
    if holder.get("started"):
        text += f", started {holder['started']}"
    return text


def _try_lock(fd: int) -> bool:
    try:
        if sys.platform == "win32":
            import msvcrt

            os.lseek(fd, _LOCK_BYTE, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # flock, not lockf: a second open file in the same process conflicts too
    except OSError as e:
        if e.errno in _BUSY:
            return False
        raise
    return True


def _unlock(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, _LOCK_BYTE, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


def _read_holder(fd: int) -> dict:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        holder = json.loads(os.read(fd, 4096).decode("utf-8"))
        return holder if isinstance(holder, dict) else {}
    except (OSError, ValueError):  # the holder is between locking and writing, or the text is damaged: the lock still counts
        return {}


class InstanceLock:
    """The lock of one folder. ``info`` (e.g. ``port=8077``) is written into the file next to pid and start time."""

    def __init__(self, folder: Path, **info):
        self.folder = Path(folder).resolve()
        self.path = self.folder / LOCK_FILE
        self.info = {"pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S"), **info}
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self) -> InstanceLock:
        """Take the lock or raise ``FolderInUse``. Never waits."""
        if self._fd is not None:
            return self
        self.folder.mkdir(parents=True, exist_ok=True)
        # os.open gives a handle that child processes do not inherit: a build worker must not keep the lock of a dead server
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o644)
        try:
            if not _try_lock(fd):
                raise FolderInUse(self.folder, _read_holder(fd))
            os.ftruncate(fd, 0)  # what a dead server wrote
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, json.dumps(self.info).encode("utf-8"))
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        return self

    def release(self) -> None:
        """Give the lock up. The file stays: removing it would let two processes lock two different files of one name."""
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            _unlock(fd)
        except OSError:
            pass  # closing the file gives the lock up as well
        finally:
            os.close(fd)

    def __enter__(self) -> InstanceLock:
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()


def lock_folders(folders, **info) -> list[InstanceLock]:
    """Lock every folder, all or none. A folder named twice (maps and work in one place) is locked once."""
    taken: list[InstanceLock] = []
    seen = set()
    try:
        for folder in folders:
            lock = InstanceLock(folder, **info)
            key = os.path.normcase(str(lock.folder))
            if key in seen:
                continue
            seen.add(key)
            taken.append(lock.acquire())
    except BaseException:
        release_all(taken)
        raise
    return taken


def release_all(locks: list[InstanceLock]) -> None:
    for lock in locks:
        lock.release()
