"""The room model and the scan verdict through the API: rows of /maps, the model file, its numbers."""

import json

import numpy as np

from locserver import roommodel as rm


def put_model(map_dir):
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [9, 9, 9]], np.uint8)
    normals = np.array([[0, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0]], float)
    tris = np.array([[0, 1, 2], [0, 2, 3]], np.uint32)
    rm.write_model(map_dir / rm.MODEL_BIN, verts, colors, normals, tris, 0.03)
    info = {"version": 1, "frame": "map", "created_utc": "2026-11-21T20:00:00Z", "format": "LRM1", "vertices": 4, "triangles": 2,
            "area_m2": 1.0, "voxel_m": 0.03, "bbox_min": [0, 0, 0], "bbox_max": [1, 1, 1], "images_used": 3, "bytes": 99}
    (map_dir / rm.MODEL_INFO).write_text(json.dumps(info), encoding="utf-8")


def row(api, map_id):
    return next(m for m in api.get("/maps").json()["maps"] if m["map_id"] == map_id)


def test_the_model_is_announced_and_served(api, maps_dir):
    h = api.get("/health").json()
    assert {"room_model", "scan_quality"} <= set(h["capabilities"]) and h["api_revision"] == 7
    assert row(api, "metric_room")["model"] == {"available": False}
    assert api.get("/maps/metric_room/model").status_code == 404 and api.get("/maps/metric_room/model/info").status_code == 404

    put_model(maps_dir / "metric_room")
    r = row(api, "metric_room")["model"]
    assert r["available"] and r["vertices"] == 4 and r["triangles"] == 2 and r["created_utc"] == "2026-11-21T20:00:00Z"
    b = api.get("/maps/metric_room/model")
    assert b.status_code == 200 and b.content == (maps_dir / "metric_room" / rm.MODEL_BIN).read_bytes() and b.content[:4] == b"LRM1"
    ply = api.get("/maps/metric_room/model?format=ply")
    assert ply.status_code == 200 and ply.content.startswith(b"ply") and b"element vertex 4" in ply.content
    assert api.get("/maps/metric_room/model?format=obj").status_code == 422
    info = api.get("/maps/metric_room/model/info").json()
    assert info["model"]["triangles"] == 2 and info["scale_m_per_unit"] == 1.0 and info["frame_id"]
    assert api.get("/monitor/state").json()["maps"][0]["model"] is not None


def test_a_model_without_its_numbers_is_no_model(api, maps_dir):
    put_model(maps_dir / "metric_room")
    (maps_dir / "metric_room" / rm.MODEL_INFO).unlink()            # a build that died between the two files
    assert row(api, "metric_room")["model"] == {"available": False}
    assert api.get("/maps/metric_room/model").status_code == 404


def test_the_scan_verdict_is_in_the_rows_and_whole_in_the_map(api):
    assert row(api, "metric_room")["scan"] is None
    q = {"version": 1, "verdict": "rescan", "summary": "Scan the room again: ...",
         "issues": [{"code": "one_spot", "value": 1.1, "limit": 1.5, "text": "t", "do": "d"}], "motion": {"footprint_m": [1.1, 0.8]}}
    api.state_obj.maps["metric_room"].set_meta({"scan_quality": q})
    assert row(api, "metric_room")["scan"] == {"verdict": "rescan", "summary": "Scan the room again: ...", "issues": ["one_spot"]}
    assert api.get("/maps/metric_room").json()["scan_quality"] == q
