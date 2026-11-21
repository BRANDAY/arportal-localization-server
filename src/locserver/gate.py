"""The line in front of the one graphics card.

Several phones may look at the same door at once; the card does one
localisation at a time. The gate lets them through in the order they came
(a plain lock gives no such promise) and knows how long the line is, so that
a request can do less work when others wait and the answer can say how long
it stood in line.

Sized for a handful of phones (owner, 2026-11-21: "five for now"). For many
more the answer is not a longer line but another design.
"""

from __future__ import annotations

import threading
import time
from collections import deque


class Gate:
    def __init__(self, history: int = 60):
        self._mutex = threading.Lock()
        self._line: deque[threading.Event] = deque()
        self._busy = False
        self._waits_ms: deque[float] = deque(maxlen=history)
        self._work_ms: deque[float] = deque(maxlen=history)
        self._entered = 0.0
        self.served = 0

    # ------------------------------------------------------------------ #
    def enter(self) -> dict:
        """Wait for the turn. Returns ``{"ahead": requests that were before this one, "waited_ms": ...}``."""
        turn = threading.Event()
        with self._mutex:
            ahead = len(self._line) + int(self._busy)
            if ahead == 0:
                self._busy = True
                turn.set()
            else:
                self._line.append(turn)
        t = time.perf_counter()
        turn.wait()
        waited = (time.perf_counter() - t) * 1000.0
        with self._mutex:
            self._waits_ms.append(waited)
            self._entered = time.perf_counter()
        return {"ahead": ahead, "waited_ms": round(waited, 1)}

    def leave(self) -> None:
        with self._mutex:
            self._work_ms.append((time.perf_counter() - self._entered) * 1000.0)
            self.served += 1
            if self._line:
                self._line.popleft().set()  # the card goes straight to the next in line
            else:
                self._busy = False

    def __enter__(self) -> dict:
        return self.enter()

    def __exit__(self, *exc) -> None:
        self.leave()

    # ------------------------------------------------------------------ #
    @property
    def waiting(self) -> int:
        """Requests standing in line right now (the one being served is not counted)."""
        return len(self._line)

    def status(self) -> dict:
        with self._mutex:
            waits, work = sorted(self._waits_ms), sorted(self._work_ms)
            return {
                "running": int(self._busy), "waiting": len(self._line), "served": self.served,
                "recent_wait_ms": round(waits[len(waits) // 2], 1) if waits else None,
                "recent_wait_ms_longest": round(waits[-1], 1) if waits else None,
                "recent_work_ms": round(work[len(work) // 2], 1) if work else None,
            }


def frames_for(asked: int, waiting: int, aligned_frames: int) -> int:
    """How many frames of a clip to localise when ``waiting`` other requests stand in line.

    With nobody waiting: what the client asked for. With others waiting, a
    session that is already aligned gets one frame (it only refines), one that
    still looks for the room gets two (a single frame cannot give an alignment
    a second opinion). Fewer frames per turn means shorter turns for all.
    """
    if waiting <= 0:
        return asked
    return max(1, min(asked, 1 if aligned_frames >= 3 else 2))
