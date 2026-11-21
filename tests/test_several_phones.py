"""A handful of phones looking at one door: a fair line in front of the one graphics card, less work when others wait."""

from __future__ import annotations

import json
import threading
import time

import cv2
import numpy as np

from locserver import geometry as g
from locserver.gate import Gate, frames_for
from test_staff_door import jpeg, make_server

POSE = g.make_T(np.eye(3), [0.0, 0.0, 0.0])


# ------------------------------------------------------------------- the gate --- #
def test_the_line_is_served_in_the_order_of_arrival_and_knows_its_length():
    gate, order, inside = Gate(), [], []
    first = gate.enter()
    assert first["ahead"] == 0 and first["waited_ms"] < 50 and gate.status()["running"] == 1 and gate.waiting == 0

    def phone(i):
        info = gate.enter()
        inside.append(len(inside))
        order.append((i, info["ahead"]))
        time.sleep(0.01)
        gate.leave()

    threads = []
    for i in range(5):
        t = threading.Thread(target=phone, args=(i,))
        t.start()
        threads.append(t)
        while gate.waiting < i + 1:  # each one is in line before the next arrives
            time.sleep(0.001)
    assert gate.status()["waiting"] == 5
    gate.leave()
    for t in threads:
        t.join(timeout=5)
    assert [i for i, _ in order] == [0, 1, 2, 3, 4]          # nobody jumped the line
    assert [ahead for _, ahead in order] == [1, 2, 3, 4, 5]  # and each knew how many were before
    s = gate.status()
    assert s == {**s, "running": 0, "waiting": 0, "served": 6} and s["recent_wait_ms"] is not None and s["recent_work_ms"] is not None
    with gate as turn:  # usable as a plain lock too
        assert turn["ahead"] == 0 and gate.status()["running"] == 1
    assert gate.status()["running"] == 0


def test_a_request_does_less_when_others_wait():
    assert frames_for(3, 0, 0) == 3 and frames_for(3, 0, 12) == 3    # nobody waits: what was asked for
    assert frames_for(3, 1, 0) == 2 and frames_for(3, 4, 2) == 2     # still looking for the room: two frames
    assert frames_for(3, 1, 3) == 1 and frames_for(8, 2, 20) == 1    # aligned already: one frame refines
    assert frames_for(1, 3, 0) == 1 and frames_for(2, 1, 0) == 2     # never more than asked


# -------------------------------------------------------------------- the API --- #
def clip(tmp_path, frames=12):
    path = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    rng = np.random.default_rng(0)
    for _ in range(frames):
        w.write(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))
    w.release()
    return path.read_bytes()


class BusyGate(Gate):
    """A gate that says others are waiting (as when several phones send clips at once)."""

    def __init__(self, waiting: int):
        super().__init__()
        self._pretend = waiting

    @property
    def waiting(self) -> int:
        return self._pretend


def send(c, video, **data):
    r = c.post("/localize", files={"video": ("video.mp4", video, "video/mp4")},
               data={"map_id": "metric_room", "ar_pose": json.dumps({"position": [0, 1.5, 0], "rotation": [0, 0, 0]}), **data})
    assert r.status_code == 200, r.text
    return r.json()


def test_localize_says_how_long_it_waited_and_uses_fewer_frames_under_load(tmp_path):
    c = make_server(tmp_path)
    loc, video = c.state_obj.localizer, clip(tmp_path)
    loc.queue += [POSE] * 3
    out = send(c, video, session_id="v1")
    assert len(out["frames"]) == 3 and out["load"] == {**out["load"], "frames_asked": 3, "frames_used": 3, "ahead": 0}
    assert out["load"]["waited_ms"] < 1000 and out["load"]["next_request_after_s"] == 1.0
    h = c.get("/health").json()
    assert h["localize"]["running"] == 0 and h["localize"]["waiting"] == 0 and h["localize"]["served"] == 1
    for _ in range(2):  # a clip with one AR pose gives one observation: three requests make the session "aligned"
        loc.queue += [POSE] * 3
        send(c, video, session_id="v1")
    assert "several_phones" in h["capabilities"] and h["api_revision"] >= 6

    c.state_obj.gpu_lock = BusyGate(waiting=2)  # two other phones stand in line
    loc.queue += [POSE] * 2
    new = send(c, video, session_id="v2")       # a session that still looks for the room: two frames
    assert len(new["frames"]) == 2 and new["load"]["frames_used"] == 2 and new["load"]["frames_asked"] == 3
    n = len(loc.calls)
    loc.queue += [POSE]
    again = send(c, video, session_id="v1")     # session v1 has three observations already: one frame refines
    assert len(again["frames"]) == 1 and len(loc.calls) == n + 1 and again["session"]["num_observations"] == 4
    assert again["alignment"]["status"] == "ok"
    # a single picture is never cut, and the image path reports its load too
    loc.queue += [POSE]
    r = c.post("/localize", files={"image": ("q.jpg", jpeg(), "image/jpeg")}, data={"map_id": "metric_room"}).json()
    assert r["load"]["frames_used"] == 1 and r["success"]


def test_an_aligned_session_is_told_to_slow_down(tmp_path):
    c = make_server(tmp_path)
    loc, video = c.state_obj.localizer, clip(tmp_path)
    for _ in range(6):
        loc.queue += [POSE] * 3
        out = send(c, video, session_id="v1")
    assert out["alignment"]["observations_used"] >= 6 and out["load"]["next_request_after_s"] == 5.0


def test_five_phones_at_once_all_get_their_answer(tmp_path):
    c = make_server(tmp_path)
    loc, video = c.state_obj.localizer, clip(tmp_path)
    real = loc.localize

    def slow(*a, **k):  # a localisation takes time; meanwhile the others queue up
        time.sleep(0.05)
        return real(*a, **k)

    loc.localize = slow
    loc.queue += [POSE] * 40
    answers, errors = [], []

    def phone(i):
        try:
            for _ in range(2):
                answers.append(send(c, video, session_id=f"phone-{i}"))
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    threads = [threading.Thread(target=phone, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors and len(answers) == 10 and all(a["success"] and a["alignment"]["status"] == "ok" for a in answers)
    assert max(a["load"]["ahead"] for a in answers) >= 1          # they did stand in line
    assert any(a["load"]["frames_used"] < 3 for a in answers)     # and some did less because of it
    s = c.get("/health").json()["localize"]
    assert s["served"] == 10 and s["waiting"] == 0 and s["running"] == 0
    assert c.get("/monitor/state").json()["server"]["localize"]["served"] == 10
