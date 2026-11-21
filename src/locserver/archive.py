"""Keep what the phones send, so that a failure can be looked at without a second walk through the room.

Off by default. Switched on with the environment variable
``LOCSERVER_KEEP_INCOMING`` (``1`` = ``<work>/incoming``, or a folder). Then:

* every ``/localize`` request leaves a folder with the clip (or image), its
  ``session.jsonl``, the form fields (``request.json``) and the answer
  (``response.json``);
* every recording upload keeps its job folder (video, session log, spec,
  worker log) - also when the recording was refused or the build failed;
* ``requests.jsonl`` gets one line per request with its status, duration and,
  for a refused one, the reason;
* ``localizations.csv`` gets one row per answered ``/localize``: result,
  quality, matches, inliers, times, alignment, and how far the door moved
  since the earlier attempts of the same AR session.

Recordings of a real room are private: the folder lies outside the repository
and nothing here is ever sent anywhere.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import shutil
import threading
import time
from pathlib import Path

log = logging.getLogger("locserver.archive")
_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe(text) -> str:
    return _SAFE.sub("-", str(text or "unknown"))[:40] or "unknown"


class Archive:
    def __init__(self, root: Path | None):
        self.root = Path(root) if root else None
        self._lock = threading.Lock()
        self._n = 0
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)
            log.info("keeping everything the phones send in %s", self.root)

    @property
    def enabled(self) -> bool:
        return self.root is not None

    # every method is a no-op when the archive is off, and never lets a disk problem fail a request
    def new(self, kind: str, device: str | None = None, session: str | None = None) -> Path | None:
        if not self.root:
            return None
        with self._lock:
            self._n += 1
            n = self._n
        d = self.root / time.strftime("%Y%m%d") / f"{time.strftime('%H%M%S')}_{n:04d}_{_safe(kind)}_{_safe(device)}_{_safe(session)}"
        try:
            d.mkdir(parents=True, exist_ok=True)
            return d
        except OSError:
            log.warning("could not create %s", d, exc_info=True)
            return None

    @staticmethod
    def write_json(folder: Path | None, name: str, obj) -> None:
        if folder is None:
            return
        try:
            (folder / name).write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
        except (OSError, TypeError, ValueError):
            log.warning("could not write %s", name, exc_info=True)

    @staticmethod
    def write_bytes(folder: Path | None, name: str, data: bytes) -> None:
        if folder is None:
            return
        try:
            (folder / name).write_bytes(data)
        except OSError:
            log.warning("could not write %s", name, exc_info=True)

    @staticmethod
    def copy(folder: Path | None, src: Path, name: str | None = None) -> None:
        if folder is None or not Path(src).exists():
            return
        try:
            shutil.copy2(src, folder / (name or Path(src).name))
        except OSError:
            log.warning("could not keep %s", src, exc_info=True)

    def take(self, src: Path, kind: str, device: str | None = None, note: dict | None = None) -> Path | None:
        """Move a whole folder in (a refused recording that would otherwise be deleted)."""
        d = self.new(kind, device)
        if d is None:
            return None
        try:
            for f in Path(src).iterdir():
                shutil.move(str(f), str(d / f.name))
        except OSError:
            log.warning("could not keep %s", src, exc_info=True)
        if note:
            self.write_json(d, "note.json", note)
        return d

    def csv_line(self, name: str, row: dict) -> None:
        """One row of a table that opens in a spreadsheet; the header is written with the first row."""
        if not self.root:
            return
        try:
            path = self.root / name
            with self._lock:
                fresh = not path.exists() or path.stat().st_size == 0
                with open(path, "a", encoding="utf-8", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(row), extrasaction="ignore")
                    if fresh:
                        w.writeheader()
                    w.writerow({k: ("" if v is None else v) for k, v in row.items()})
        except (OSError, ValueError):
            log.warning("could not append to %s", name, exc_info=True)

    def request_line(self, entry: dict) -> None:
        if not self.root:
            return
        try:
            with self._lock, open(self.root / "requests.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            log.warning("could not append to requests.jsonl", exc_info=True)
