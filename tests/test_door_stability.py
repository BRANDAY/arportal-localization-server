"""Numbers for a report: how far the door moves between the requests of one AR session."""

from __future__ import annotations

import csv

import numpy as np

from locserver import geometry as g
from locserver.responses import door_stability
from test_archive import server
from test_staff_door import localize


def alignment(door=None, origin=(0.0, 0.0, 0.0), status="ok", updated="t1"):
    portal = {"source": "manual", "position": list(door), "updated_utc": updated} if door is not None else {"source": "none", "position": None}
    return {"status": status, "T_arworld_from_map": np.eye(4).tolist() if status == "ok" else None, "translation": list(origin), "portal": portal}


def test_the_door_is_followed_from_request_to_request():
    track: list = []
    first = door_stability(track, alignment(door=[1.0, 0.0, -2.0]), "scan-1")
    assert first == {**first, "point": "door", "attempt": 1, "shift_from_previous_m": None, "spread_m": None}
    second = door_stability(track, alignment(door=[1.0, 0.0, -1.9]), "scan-1")
    assert second["attempt"] == 2 and second["shift_from_previous_m"] == 0.1 and second["shift_from_median_m"] == 0.05
    third = door_stability(track, alignment(door=[1.0, 0.0, -1.9]), "scan-1")
    assert third["shift_from_previous_m"] == 0.0 and third["shift_from_median_m"] == 0.0   # the middle is where it settled
    assert third["spread_max_m"] == 0.1 and 0.05 < third["spread_m"] < 0.06                 # the first attempt was the outlier
    # no usable alignment: nothing is said, and the series is not disturbed
    assert door_stability(track, alignment(status="failed"), "scan-1") is None and door_stability(track, None, "scan-1") is None
    assert door_stability(track, alignment(door=[1.0, 0.0, -1.9]), "scan-1")["attempt"] == 4


def test_a_new_door_or_a_new_scan_starts_a_new_series():
    track: list = []
    door_stability(track, alignment(door=[1.0, 0.0, -2.0]), "scan-1")
    door_stability(track, alignment(door=[1.0, 0.0, -2.0]), "scan-1")
    moved = door_stability(track, alignment(door=[3.0, 0.0, 0.0], updated="t2"), "scan-1")  # the staff put the door elsewhere
    assert moved["attempt"] == 1 and moved["shift_from_previous_m"] is None
    assert door_stability(track, alignment(door=[3.0, 0.0, 0.0], updated="t2"), "scan-2")["attempt"] == 1   # the room was scanned again
    # a room without a door: the origin of the map is followed instead
    plain = door_stability(track, alignment(origin=[0.2, 0.0, 0.0]), "scan-2")
    assert plain["point"] == "map_origin" and plain["attempt"] == 1
    assert door_stability(track, alignment(origin=[0.2, 0.0, 0.3]), "scan-2")["shift_from_previous_m"] == 0.3


def test_localize_reports_the_shift_and_the_table_keeps_it(tmp_path):
    c = server(tmp_path)
    true = g.Sim3(1.0, g.viro_euler_to_rotmat([0, 25, 0]), np.array([0.5, 0.0, -1.0]))
    off = g.Sim3(1.0, g.viro_euler_to_rotmat([0, 25, 0]), np.array([0.5, 0.0, -0.94]))  # as if the phone's world slipped 6 cm
    head = {"X-Device-Id": "phone-1"}
    a = localize(c, "metric_room", "ar-1", true, [0.0, 1.5, 0.0], [0, 0, 0], head)
    b = localize(c, "metric_room", "ar-1", true, [0.6, 1.5, 0.2], [0, 30, 0], head)
    d = localize(c, "metric_room", "ar-1", off, [1.0, 1.5, -0.4], [0, -20, 0], head)
    assert a["door_stability"]["attempt"] == 1 and a["door_stability"]["shift_from_previous_m"] is None
    assert b["door_stability"]["attempt"] == 2 and b["door_stability"]["shift_from_previous_m"] < 0.001
    assert d["door_stability"]["attempt"] == 3 and 0.005 < d["door_stability"]["shift_from_previous_m"] < 0.06  # it moved, and it shows
    other = localize(c, "metric_room", "ar-2", true, [0.0, 1.5, 0.0], [0, 0, 0], {"X-Device-Id": "phone-2"})
    assert other["door_stability"]["attempt"] == 1                                 # each AR session has its own series
    # a request without an AR session has nothing to compare with
    c.state_obj.localizer.queue.append(g.make_T(np.eye(3), [0, 0, 0]))
    from test_staff_door import jpeg
    single = c.post("/localize", files={"image": ("q.jpg", jpeg(), "image/jpeg")}, data={"map_id": "metric_room"}).json()
    assert single["door_stability"] is None

    locs = c.get("/monitor/state").json()["localizations"]  # newest first
    assert [x["door_attempt"] for x in locs] == [None, 1, 3, 2, 1]
    assert locs[2]["door_shift_m"] == d["door_stability"]["shift_from_previous_m"] and locs[2]["session_id"] == "ar-1"
    assert locs[2]["num_matches"] is not None and locs[2]["observations_used"] >= 2 and locs[2]["door_point"] == "map_origin"

    with open(tmp_path / "incoming" / "localizations.csv", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 5 and rows[0]["door_attempt"] == "1" and rows[0]["door_shift_m"] == "" and rows[4]["session_id"] == ""
    assert float(rows[2]["door_shift_m"]) == d["door_stability"]["shift_from_previous_m"] and rows[2]["success"] == "True"
    assert {"time", "device_id", "map_id", "quality", "num_matches", "num_inliers", "elapsed_ms", "alignment", "position_rms_m",
            "door_spread_m", "folder"} <= set(rows[0])
    assert (tmp_path / "incoming" / rows[2]["folder"]).name == rows[2]["folder"] and any(
        p.name == rows[2]["folder"] for p in (tmp_path / "incoming").glob("*/*"))
