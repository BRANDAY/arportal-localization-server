"""With LOCSERVER_KEEP_INCOMING the server keeps what the phones send, so a failure can be studied afterwards."""

from __future__ import annotations

import json

import numpy as np
from fastapi.testclient import TestClient

from conftest import StubLocalizer, StubPipe, make_map
from locserver import config
from locserver import geometry as g
from locserver.archive import Archive
from locserver.server import State, create_app
from test_api_v2 import session_jsonl
from test_several_phones import clip
from test_staff_door import jpeg


def server(tmp_path, keep=True):
    maps = tmp_path / "maps"
    make_map(maps / "metric_room", "metric_room", metric=True)
    state = State(maps, pipe=StubPipe(), localizer=StubLocalizer(), calibration_dir=tmp_path / "calib", work_dir=tmp_path / "work",
                  job_runner=lambda job: {"map_id": job.map_id}, keep_incoming=tmp_path / "incoming" if keep else None)
    c = TestClient(create_app(state))
    c.state_obj = state
    return c


def lines(tmp_path):
    return [json.loads(x) for x in (tmp_path / "incoming" / "requests.jsonl").read_text(encoding="utf-8").splitlines()]


def test_a_localize_request_leaves_its_upload_the_form_and_the_answer(tmp_path):
    c = server(tmp_path)
    c.state_obj.localizer.queue += [g.make_T(np.eye(3), [0, 0, 0])] * 4
    head = {"X-Device-Id": "phone-1", "X-Session-Id": "run-1", "X-Device-Role": "visitor"}
    pose = json.dumps({"position": [0, 1.5, 0], "rotation": [0, 0, 0]})
    r = c.post("/localize", files={"image": ("q.jpg", jpeg(), "image/jpeg")}, headers=head,
               data={"map_id": "metric_room", "session_id": "ar/../1", "ar_pose": pose})
    assert r.status_code == 422  # a bad session id: no folder is needed to see why
    r = c.post("/localize", files={"image": ("q.jpg", jpeg(), "image/jpeg")}, headers=head,
               data={"map_id": "metric_room", "session_id": "ar-1", "ar_pose": pose})
    assert r.status_code == 200
    video = clip(tmp_path)
    r2 = c.post("/localize", files={"video": ("video.mp4", video, "video/mp4"), "session_jsonl": ("session.jsonl", b"not a log", "text/plain")},
                headers=head, data={"map_id": "metric_room", "session_id": "ar-1"})
    assert r2.status_code == 422  # the clip could not be read - and exactly that upload is what has to be kept
    assert c.post("/localize", files={"video": ("video.mp4", video, "video/mp4")}, headers=head,
                  data={"map_id": "metric_room", "session_id": "ar-1", "ar_pose": pose}).status_code == 200
    kept = sorted(p for p in (tmp_path / "incoming").glob("*/*") if p.is_dir())
    names = [sorted(f.name for f in d.iterdir()) for d in kept]
    assert ["image.jpg", "request.json", "response.json"] in names
    assert ["request.json", "session.jsonl", "video.mp4"] in names          # the failed one: upload kept, no answer
    assert ["request.json", "response.json", "video.mp4"] in names
    assert all("localize_phone-1_ar-" in d.name for d in kept)
    good = next(d for d in kept if (d / "image.jpg").exists() and (d / "response.json").exists())
    req, resp = json.loads((good / "request.json").read_text()), json.loads((good / "response.json").read_text())
    assert req["map_id"] == "metric_room" and req["headers"]["x-device-role"] == "visitor" and json.loads(req["ar_pose"])["position"] == [0, 1.5, 0]
    assert resp["success"] is True and resp["alignment"]["status"] == "ok" and (good / "image.jpg").read_bytes() == jpeg()
    assert next(d for d in kept if (d / "session.jsonl").exists()).joinpath("video.mp4").read_bytes() == video
    # one line per request, with the reason of a refusal; polling is left out
    c.get("/health", headers=head)
    c.get("/maps/nope", headers=head)
    log = lines(tmp_path)
    assert [x["status"] for x in log if x["path"] == "/localize"] == [422, 200, 422, 200]
    bad = [x for x in log if x["status"] == 422]
    assert "session_id" in bad[0]["error"] and "cannot read the clip" in bad[1]["error"] and bad[0]["device"] == "phone-1"
    assert [x["path"] for x in log if x["status"] == 404] == ["/maps/nope"] and not [x for x in log if x["path"] == "/health"]
    assert c.get("/health").json()["keeps_incoming"] is True


def test_recordings_are_kept_even_when_refused_or_replaced(tmp_path):
    c = server(tmp_path)
    video = clip(tmp_path)
    files = lambda log: {"video": ("video.mp4", video, "video/mp4"), "session_jsonl": ("session.jsonl", log, "application/x-ndjson")}  # noqa: E731
    r = c.post("/maps/hall/recording", files=files(b'{"type": "header"}\n'), data={"device_id": "staff-phone"})
    assert r.status_code == 422
    refused = [d for d in (tmp_path / "incoming").glob("*/*recording-refused-hall*")]
    assert len(refused) == 1 and (refused[0] / "video.mp4").read_bytes() == video and (refused[0] / "session.jsonl").exists()
    assert json.loads((refused[0] / "note.json").read_text())["status"] == 422
    # two accepted recordings of one room: both job folders stay (normally only the newest is kept)
    jobs = []
    for _ in range(2):
        r = c.post("/maps/hall/recording", files=files(session_jsonl()), data={"replace": "true"})
        assert r.status_code == 202, r.text
        jobs.append(r.json()["job"]["job_id"])
        c.state_obj.jobs.wait(jobs[-1])
    for j in jobs:
        d = c.state_obj.jobs.job_dir(j)
        assert (d / "video.mp4").exists() and (d / "session.jsonl").exists()


def test_nothing_is_kept_unless_asked_for(tmp_path, monkeypatch):
    c = server(tmp_path, keep=False)
    c.state_obj.localizer.queue.append(g.make_T(np.eye(3), [0, 0, 0]))
    assert c.post("/localize", files={"image": ("q.jpg", jpeg(), "image/jpeg")}, data={"map_id": "metric_room"}).status_code == 200
    assert not (tmp_path / "incoming").exists() and c.get("/health").json()["keeps_incoming"] is False
    a = Archive(None)
    assert a.new("localize") is None and a.take(tmp_path, "x") is None
    a.write_json(None, "x.json", {})
    a.request_line({"x": 1})
    # the switch: off by default, 1 = a folder in the work directory, anything else = that folder
    monkeypatch.delenv("LOCSERVER_KEEP_INCOMING", raising=False)
    assert config.keep_incoming_dir() is None
    monkeypatch.setenv("LOCSERVER_KEEP_INCOMING", "1")
    assert config.keep_incoming_dir() == config.work_dir() / "incoming"
    monkeypatch.setenv("LOCSERVER_KEEP_INCOMING", str(tmp_path / "there"))
    assert config.keep_incoming_dir() == tmp_path / "there"
    monkeypatch.setenv("LOCSERVER_KEEP_INCOMING", "off")
    assert config.keep_incoming_dir() is None
