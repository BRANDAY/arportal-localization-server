"""Two lessons of the first run on a phone: one good frame of a clip is enough, and a weak frame never places a door.

Measured there: good frames put the door within 0-11 cm of each other, weak ones 1.6-2.4 m off; and in a clip that
is recognised nearly every frame is, so localising the others only costs time.
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from conftest import weak_result
from locserver import geometry as g

POSE = g.make_T(np.eye(3), [0.0, 0.0, 0.0])
AR = json.dumps({"position": [0, 1.5, 0], "rotation": [0, 0, 0]})


def clip(tmp_path, frames=12):
    path = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    rng = np.random.default_rng(0)
    for _ in range(frames):
        w.write(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))
    w.release()
    return path.read_bytes()


def poses_of(frames=12) -> bytes:
    """A session.jsonl with one AR pose per frame of ``clip``."""
    lines = [json.dumps({"type": "header", "intrinsics": {"fx": 50, "fy": 50, "cx": 32, "cy": 24, "width": 64, "height": 48}})]
    lines += [json.dumps({"type": "pose", "t": i * 100_000_000, "orientation": [0, 0, 0, 1], "position": [0.05 * i, 1.4, 0]}) for i in range(frames)]
    return ("\n".join(lines) + "\n").encode()


def send(api, video, **data):
    """A clip with one AR pose, as the app sends it."""
    r = api.post("/localize", files={"video": ("video.mp4", video, "video/mp4")}, data={"map_id": "metric_room", "ar_pose": AR, **data})
    assert r.status_code == 200, r.text
    return r.json()


def look(api, result, **data):
    """One picture with its AR pose; ``result`` is what the stub answers for it."""
    api.stub.queue[:] = [result]
    jpeg = cv2.imencode(".jpg", np.zeros((48, 64, 3), np.uint8))[1].tobytes()
    r = api.post("/localize", files={"image": ("q.jpg", jpeg, "image/jpeg")}, data={"map_id": "metric_room", "ar_pose": AR, **data})
    assert r.status_code == 200, r.text
    return r.json()


# ------------------------------------------------- one good frame is enough --- #
def test_a_clip_is_answered_at_its_first_good_frame(api, tmp_path):
    loc, video = api.stub, clip(tmp_path)
    loc.queue[:] = [POSE] * 3
    out = send(api, video)
    assert len(out["frames"]) == 1 and len(loc.calls) == 1 and len(loc.queue) == 2  # two frames were never looked at
    assert out["load"] == {**out["load"], "frames_asked": 3, "frames_used": 1, "stopped_early": True}
    assert out["success"] and out["alignment"]["status"] == "ok"
    # a weak frame does not end it; the good one after it does - and is the answer, whatever the inlier counts
    loc.queue[:] = [weak_result(POSE, inliers=250), POSE, POSE]
    out = send(api, video)
    assert [f["quality"] for f in out["frames"]] == ["weak", "good"] and out["quality"] == "good"
    assert out["best_frame_index"] == out["frames"][1]["index"] and out["alignment"]["status"] == "ok"
    assert out["load"]["frames_used"] == 2 and out["load"]["stopped_early"] is True
    # no good frame in the clip: every picked frame is tried, in the order of the clip
    loc.queue[:] = [weak_result(POSE), None, weak_result(POSE)]
    out = send(api, video)
    assert [f["quality"] for f in out["frames"]] == ["weak", "failed", "weak"]
    assert out["load"]["frames_used"] == 3 and out["load"]["stopped_early"] is False
    assert [f["index"] for f in out["frames"]] == sorted(f["index"] for f in out["frames"])
    assert out["alignment"]["reason_code"] == "weak_only"


def test_all_frames_switches_the_early_answer_off_and_a_picture_knows_nothing_of_it(api, tmp_path):
    loc, video = api.stub, clip(tmp_path)
    for flag in ("1", "true"):
        loc.queue[:] = [POSE] * 3
        out = send(api, video, all_frames=flag)
        assert len(out["frames"]) == 3 and not loc.queue
        assert out["load"]["frames_used"] == 3 and out["load"]["stopped_early"] is False
    out = look(api, POSE)
    assert out["load"]["frames_asked"] == 1 and out["load"]["frames_used"] == 1 and out["load"]["stopped_early"] is False


def test_the_frame_that_finds_the_room_can_be_the_one_good_frame(api, tmp_path):
    loc, video = api.stub, clip(tmp_path)
    files = lambda: {"video": ("video.mp4", video, "video/mp4")}  # noqa: E731
    loc.queue[:] = [POSE] * 3
    out = api.post("/localize", files=files()).json()   # two rooms, none named
    assert out["room_choice"]["mode"] == "auto" and out["room_choice"]["rooms_tried"] == 1
    assert len(out["frames"]) == 1 and len(loc.calls) == 1 and out["load"]["stopped_early"] is True
    # not this room, but the next one: still no second frame
    loc.calls.clear()
    loc.queue[:] = [None, POSE, POSE]
    out = api.post("/localize", files=files()).json()
    assert out["room_choice"]["rooms_tried"] == 2 and out["success"] and len(out["frames"]) == 1 and len(loc.calls) == 2


def test_a_map_without_scale_gets_every_frame_of_a_clip_that_brings_its_poses(api, tmp_path):
    loc, video = api.stub, clip(tmp_path)
    files = lambda: {"video": ("video.mp4", video, "video/mp4"), "session_jsonl": ("session.jsonl", poses_of(), "application/x-ndjson")}  # noqa: E731
    loc.queue[:] = [POSE] * 3
    r = api.post("/localize", files=files(), data={"map_id": "scaleless_room"})
    assert r.status_code == 200, r.text
    out = r.json()   # its scale comes from frames taken apart: one frame cannot give it
    assert len(out["frames"]) == 3 and all(f["has_ar_pose"] for f in out["frames"]) and out["load"]["stopped_early"] is False
    assert out["alignment"]["observations_total"] == 3
    loc.queue[:] = [POSE] * 3
    out = api.post("/localize", files=files(), data={"map_id": "metric_room"}).json()   # a metric map needs one
    assert len(out["frames"]) == 1 and out["load"]["stopped_early"] is True and out["alignment"]["observations_total"] == 1


# --------------------------------------------- a weak frame places no door --- #
def test_weak_frames_alone_give_no_alignment_and_no_door(api):
    api.put("/maps/metric_room/anchors/portal", json={"position": [2.0, 0.0, -3.0], "quaternion_xyzw": [0, 0, 0, 1]})
    for data in ({"session_id": "s1"}, {}):   # with a session and without one
        out = look(api, weak_result(POSE), **data)
        al = out["alignment"]
        assert out["success"] and out["quality"] == "weak" and out["pose"] is not None   # the localisation itself is reported as before
        assert al["status"] == "failed" and al["reason_code"] == "weak_only" and al["T_arworld_from_map"] is None
        assert "weak" in al["reason"] and "eye height" in al["reason"]
        assert "portal" not in al and "anchors" not in al and "from_weak_localization_only" not in al
    assert out["door_stability"] is None
    again = look(api, weak_result(POSE), session_id="s1")
    assert again["session"]["num_observations"] == 0 and again["door_stability"] is None   # nothing was kept
    assert api.get("/sessions/s1").json()["alignment"]["status"] == "failed"
    # so nothing can be placed in the room from that session: not the door, not the bridge to a service
    door = {"ar_pose": {"position": [1.0, 0.0, -2.0], "rotation": [0, 0, 0]}, "session_id": "s1"}
    r = api.put("/maps/metric_room/anchors/portal", json=door)
    assert r.status_code == 422 and "/localize" in r.json()["detail"]
    assert api.get("/maps/metric_room/anchors/portal").json()["position"] == [2.0, 0.0, -3.0]
    r = api.post("/maps/metric_room/vps/bridge",
                 json={"session_id": "s1", "location_id": "loc_42", "samples": [{"T_arworld_from_service": np.eye(4).tolist()}]})
    assert r.status_code == 422 and "/localize" in r.json()["detail"]
    # a frame that was not recognised at all is another failure, and says so
    al = look(api, None, session_id="s1")["alignment"]
    assert al["status"] == "failed" and "reason_code" not in al and "no successfully localised frame" in al["reason"]
    # the first good frame gives the door
    good = look(api, POSE, session_id="s1")
    assert good["alignment"]["status"] == "ok" and good["alignment"]["observations_total"] == 1
    assert good["alignment"]["portal"]["source"] == "manual" and good["door_stability"]["attempt"] == 1
    assert api.put("/maps/metric_room/anchors/portal", json=door).status_code == 200


def test_a_weak_frame_does_not_move_a_door_that_good_frames_placed(api):
    api.put("/maps/metric_room/anchors/portal", json={"position": [2.0, 0.0, -3.0], "quaternion_xyzw": [0, 0, 0, 1]})
    first = look(api, POSE, session_id="s1")
    off = g.make_T(np.eye(3), [2.0, 0.0, 0.0])   # a weak frame that sees the room two metres from where it is
    second = look(api, weak_result(off), session_id="s1")
    a, b = first["alignment"], second["alignment"]
    assert second["quality"] == "weak" and second["session"]["num_observations"] == 1
    assert b["status"] == "ok" and "reason_code" not in b and b["observations_total"] == 1
    assert b["T_arworld_from_map"] == a["T_arworld_from_map"] and b["portal"]["position"] == a["portal"]["position"]
    # the door stood still, and the numbers say so
    assert second["door_stability"]["attempt"] == 2 and second["door_stability"]["shift_from_previous_m"] == 0.0
    assert look(api, POSE, session_id="s1")["session"]["num_observations"] == 2


def test_a_weak_frame_does_not_tie_a_session_to_a_room(api):
    jpeg = cv2.imencode(".jpg", np.zeros((48, 64, 3), np.uint8))[1].tobytes()

    def ask():   # no room named: the server looks for it
        r = api.post("/localize", files={"image": ("q.jpg", jpeg, "image/jpeg")}, data={"session_id": "s1", "ar_pose": AR})
        assert r.status_code == 200, r.text
        return r.json()

    api.stub.queue[:] = [weak_result(POSE)]
    first = ask()
    assert first["quality"] == "weak" and first["room_choice"]["mode"] == "auto" and first["room_choice"]["rooms_tried"] == 2
    assert first["alignment"]["reason_code"] == "weak_only"
    api.stub.queue[:] = [None, POSE]
    second = ask()   # the session has no good frame yet, so the room is looked for again
    assert second["room_choice"]["mode"] == "auto" and second["quality"] == "good" and second["session"]["num_observations"] == 1
    api.stub.queue[:] = [POSE]
    assert ask()["room_choice"]["mode"] == "session"
