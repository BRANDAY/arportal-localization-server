"""Worker process of a background job: ``python -m locserver.jobworker <spec.json>``.

Started by :class:`locserver.jobs.JobManager`. Talks to it through stdout, one
JSON object per line::

    {"stage": "matching", "p": 0.41, "msg": "matched 410 of 1000 image pairs"}
    {"result": {...}}            <- last line on success
    {"error": {"code": "sfm_failed", "message": "..."}}

Everything else (logging, COLMAP's own output) goes to stderr, which the
manager writes to ``worker.log`` in the job directory.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

log = logging.getLogger("locserver.jobworker")


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


class Reporter:
    """Progress lines, at most a few per second."""

    def __init__(self):
        self.last = (None, 0.0)

    def __call__(self, stage: str, fraction: float, message: str = "") -> None:
        now = time.time()
        if stage == self.last[0] and fraction < 1.0 and now - self.last[1] < 0.3:
            return
        self.last = (stage, now)
        emit({"stage": stage, "p": round(float(fraction), 4), "msg": message})

    @staticmethod
    def plan(frames: int, pairs: int) -> None:
        """How much work the build is: the server sets the shares of the stages and its estimate of the time left by it."""
        emit({"plan": {"frames": int(frames), "pairs": int(pairs)}})


def classify(e: BaseException) -> str:
    text = str(e).lower()
    name = type(e).__name__
    if "outofmemory" in name.lower() or isinstance(e, MemoryError) or "out of memory" in text:
        return "out_of_memory"
    if "usable frames" in text:
        return "too_few_frames"
    if "sfm" in text:
        return "sfm_failed"
    if isinstance(e, (OSError, ValueError, KeyError)):
        return "bad_recording"
    return "internal"


def room_summary(room: dict) -> dict:
    return {"status": room["status"], "reason": room.get("reason"), "num_walls": len(room["walls"]),
            "num_corners": len(room["corners"]), "num_openings": len(room["openings"]),
            "best_opening_id": room.get("best_opening_id")}


def portal_summary(room: dict) -> dict:
    from .roomplan import opening_by_id

    o = opening_by_id(room, room.get("best_opening_id"))
    if o is None:
        return {"source": "none", "opening_id": None, "confidence": None}
    return {"source": "auto", "opening_id": o["id"], "confidence": o["confidence"]}


def _depth(lmap, spec: dict, report, warnings: list[str]) -> tuple[list, list]:
    """Metric depth of every image that shows something new: the room plan and the room model are both made from it."""
    from . import roommodel
    from .roomstage import map_up

    if lmap.scale_m_per_unit is None or map_up(lmap) is None:   # no plan without scale and gravity; test maps
        return [], []
    try:
        return roommodel.prepare_all(lmap, spec.get("device", "auto"), report)
    except Exception as e:  # the old way of the plan (its own depth of forty images) still works without it
        log.exception("depth for the room model failed")
        warnings.append(f"no room model: depth failed ({type(e).__name__}: {e})")
        return [], []


def _room(lmap, spec: dict, report, warnings: list[str], frames: list | None = None) -> dict:
    from . import roommodel
    from .roomplan import empty_room
    from .roomstage import compute_room_for_map, save_room

    mesh = spec.get("mesh")
    try:
        room, warn = compute_room_for_map(lmap, spec.get("device", "auto"), mesh, spec.get("mesh_format"),
                                          bool(spec.get("detect_openings", True)), report,
                                          views_in=roommodel.plan_views(frames, lmap) if frames else None)
        warnings += warn
    except Exception as e:  # the map itself is fine: report, do not fail the build
        log.exception("room plan failed")
        room = empty_room(f"room plan failed: {type(e).__name__}: {e}")
        warnings.append(room["reason"])
    if room["status"] == "unavailable":
        warnings.append(f"no room plan: {room['reason']}")
    elif room["status"] == "partial" and room.get("reason"):
        warnings.append(f"room plan is partial: {room['reason']}")
    save_room(lmap.path, room)
    return room


def _model(lmap, spec: dict, report, warnings: list[str], frames: list, infos: list) -> dict | None:
    """The room as one coloured surface (``model.bin``). A failure here never fails the build: the map works without it."""
    from . import roommodel

    for name in (roommodel.MODEL_BIN, roommodel.MODEL_INFO):  # a model of an earlier plan run must not outlive a failure
        (lmap.path / name).unlink(missing_ok=True)
    if not frames:
        report("model", 1.0, "no room model")
        return None
    try:
        return roommodel.build_room_model(lmap, frames, infos, spec.get("device", "auto"), report)
    except Exception as e:
        log.exception("room model failed")
        warnings.append(f"no room model: {type(e).__name__}: {e}")
        report("model", 1.0, "no room model")
        return None


def _store_meta(lmap, key: str, value) -> None:
    """One field of ``map.json``, written into the file as it is now: the server may have renamed the room meanwhile."""
    f = lmap.path / "map.json"
    meta = json.loads(f.read_text(encoding="utf-8-sig"))
    meta[key] = lmap.meta[key] = value
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp.replace(f)


def model_summary(info: dict | None) -> dict:
    if not info:
        return {"available": False}
    return {"available": True, **{k: info[k] for k in ("vertices", "triangles", "area_m2", "voxel_m", "bbox_min", "bbox_max",
                                                        "images_used", "bytes", "created_utc")}}


def _scan_quality(lmap, room: dict, session=None) -> dict | None:
    """How good the scan was (:mod:`locserver.scanquality`), kept in ``map.json``. Never fails a build."""
    from . import scanquality
    from .roomstage import map_up

    try:
        counts = {"registered": lmap.meta.get("num_images_registered", len(lmap.images)),
                  "selected": lmap.meta.get("num_images_selected")}
        if session is not None and len(session.poses) >= 5:
            q = scanquality.from_session_log(session, room=room, **counts)
        else:
            up = map_up(lmap)
            q = scanquality.from_map_images(lmap, up if up is not None else (0.0, 1.0, 0.0), room=room, **counts)
        _store_meta(lmap, "scan_quality", q)
        return q
    except Exception:
        log.exception("scan quality failed")
        return None


def _scan_quality_walls(lmap, quality: dict, room: dict) -> dict:
    """A new plan of a room whose recording was judged already: the motion numbers stay, the walls are judged anew."""
    from . import scanquality

    try:
        walls = scanquality.wall_numbers(room)
        verdict, issues, line = scanquality.judge(quality.get("motion"), walls, (quality.get("frames") or {}).get("share"))
        quality = dict(quality, walls=walls, verdict=verdict, issues=issues, summary=line)
        _store_meta(lmap, "scan_quality", quality)
    except Exception:
        log.exception("scan quality failed")
    return quality


def run(spec: dict, report) -> dict:
    from .mapstore import load_map

    t0 = time.time()
    warnings: list[str] = []
    if spec["kind"] == "room_plan":
        lmap = load_map(Path(spec["map_dir"]))
        frames, infos = _depth(lmap, spec, report, warnings)
        room = _room(lmap, spec, report, warnings, frames)
        model = _model(lmap, spec, report, warnings, frames, infos)
        quality = lmap.meta.get("scan_quality")      # of the recording; a plan run alone has only the map's own cameras
        if not quality or quality.get("poses_from") != "recording":
            quality = _scan_quality(lmap, room)
        else:
            quality = _scan_quality_walls(lmap, quality, room)
    elif spec["kind"] == "build_map":
        from .features import FeaturePipeline
        from .mapping import BuildOptions, build_map
        from .recording import ViroRecordingCapture

        rec_dir, out_dir = Path(spec["job_dir"]), Path(spec["out_dir"])
        report("frames", 0.0, "opening the recording")
        cap = ViroRecordingCapture(rec_dir, display_rotation_cw=spec.get("display_rotation_cw"))
        pipe = FeaturePipeline(spec.get("device", "auto"))
        opts = BuildOptions(max_frames=int(spec.get("max_frames", 200)))
        lmap = build_map(rec_dir, out_dir, spec["map_id"], pipe, opts, capture=cap, progress=report)
        del pipe
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        extra = {k: spec[k] for k in ("label", "device_id", "recording_session_id", "wifi") if spec.get(k)}
        extra["built_from"] = "recording upload"
        lmap.meta.update(extra)
        (lmap.path / "map.json").write_text(json.dumps(lmap.meta, indent=2), encoding="utf-8")
        warnings += lmap.meta.get("warnings") or []
        if not lmap.metric:
            warnings.append("the reconstruction could not be aligned to the recorded AR poses: the map is not metric")
        t1 = time.time()
        frames, infos = _depth(lmap, spec, report, warnings)
        t2 = time.time()
        room = _room(lmap, spec, report, warnings, frames)
        t3 = time.time()
        model = _model(lmap, spec, report, warnings, frames, infos)
        t4 = time.time()
        quality = _scan_quality(lmap, room, cap.session)
        log.info("map %.0f s, depth of %d images %.0f s, room plan %.0f s, model %.0f s, scan quality %.1f s",
                 t1 - t0, len(frames), t2 - t1, t3 - t2, t4 - t3, time.time() - t4)
    else:
        raise ValueError(f"unknown job kind {spec['kind']!r}")
    report("finalize", 0.0, "writing the map")
    return {
        "map_id": spec["map_id"], "num_images": len(lmap.images), "num_points3D": lmap.meta.get("num_points3D"),
        "metric": lmap.metric, "gravity_aligned": bool(lmap.meta.get("up")) or room["status"] != "unavailable",
        "build_time_s": round(time.time() - t0, 1), "room": room_summary(room), "portal": portal_summary(room),
        "model": model_summary(model), "scan_quality": quality, "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    from . import config

    argv = sys.argv[1:] if argv is None else argv
    config.setup_logging("INFO")
    spec = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    try:
        result = run(spec, Reporter())
    except BaseException as e:  # noqa: BLE001 - everything is reported to the manager
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        log.exception("job failed")
        emit({"error": {"code": classify(e), "message": f"{type(e).__name__}: {e}"}})
        return 1
    emit({"result": result})
    # The answer is out: leave at once. The interpreter's own shutdown after a build (CUDA, the depth images, the
    # fused grid) took more than three minutes once (2026-11-21), and the server waits for this process to end.
    logging.shutdown()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
