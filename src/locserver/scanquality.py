"""How good was the scan? Numbers and a plain verdict, from the phone's own poses and from what was built.

A map can be built from a poor scan - and then recognises the room only from
where the scan stood, or only when the phone is held the way it was held then.
The staff member cannot see that from the map. This module says it: how large
the patch of floor was that the scan was made from, how far the phone looked
up and down, how much of the recording only repeats views it already had, how
much of the walls was seen - and, from that, "good", "limited" (works, with the
limits listed) or "rescan", each with the reason and what to do differently.

The thresholds are choices, set from the first real scan (2026-11-21: 100 s
from a patch of 1 x 1 m, about three turns on the spot, the last 35 s standing
still) and from what failed afterwards (looking down at a table was not
recognised 2 m from the scan spot). They are in ``LIMITS`` and in the answer.
"""

from __future__ import annotations

import cv2
import numpy as np

LIMITS = {
    "footprint_min_m": 1.5,          # the longer side of the patch the scan was made from
    "look_down_deg": -30.0,          # the phone should have looked at least this far down ...
    "look_up_deg": 15.0,             # ... and this far up, at some time
    "repeated_share_max": 0.35,      # share of the recording that shows views it already had
    "walls_seen_min": 0.45,          # share of the walls' area that was seen
    "registered_min": 0.70,          # share of the chosen frames that made it into the map
    "directions_min": 0.75,          # share of the compass directions looked at
}
VIEW_YAW_DEG, VIEW_PITCH_DEG, VIEW_CELL_M, VIEW_FIRST_S = 20.0, 20.0, 0.5, 1.5


def _horizontal(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    up = np.asarray(up, float) / np.linalg.norm(up)
    a = np.cross(up, [1.0, 0.0, 0.0])
    if np.linalg.norm(a) < 0.3:
        a = np.cross(up, [0.0, 0.0, 1.0])
    a /= np.linalg.norm(a)
    return a, np.cross(up, a)


def motion_numbers(positions, forwards, times_s, up=(0.0, 1.0, 0.0)) -> dict:
    """Numbers of the camera's path: ``positions`` in metres, ``forwards`` unit viewing directions, ``times_s`` seconds."""
    P, F, t = np.asarray(positions, float), np.asarray(forwards, float), np.asarray(times_s, float)
    up = np.asarray(up, float) / np.linalg.norm(up)
    a, b = _horizontal(up)
    xy = np.stack([P @ a, P @ b], 1)
    pitch = np.degrees(np.arcsin(np.clip(F @ up, -1, 1)))
    yaw = np.degrees(np.arctan2(F @ b, F @ a))
    dt = np.diff(t, append=t[-1] + (np.median(np.diff(t)) if len(t) > 1 else 0.0))
    dt = np.clip(dt, 0.0, 1.0)                         # a gap in the recording is not time spent looking
    total = float(dt.sum()) or 1.0

    # the patch of floor the scan was made from: extent along its own long and short direction
    c = xy - xy.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    q = c @ vt.T
    ext = np.percentile(q, 98, axis=0) - np.percentile(q, 2, axis=0)
    hull = cv2.convexHull(xy.astype(np.float32))
    area = float(cv2.contourArea(hull)) if len(hull) >= 3 else 0.0

    # views: where the phone stood (cell), which way it looked (yaw, pitch); time in a view after its first 1.5 s is a repeat
    key = np.stack([np.floor(xy[:, 0] / VIEW_CELL_M), np.floor(xy[:, 1] / VIEW_CELL_M),
                    np.floor(yaw / VIEW_YAW_DEG), np.floor(pitch / VIEW_PITCH_DEG)], 1).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    spent = np.bincount(inv, weights=dt)
    repeated = float(np.maximum(spent - VIEW_FIRST_S, 0.0).sum())
    cells = len(np.unique(key[:, :2], axis=0))
    directions = len(np.unique(np.floor(yaw / VIEW_YAW_DEG))) / (360.0 / VIEW_YAW_DEG)

    # standing still: within a second the phone moved less than 3 cm and turned less than 3 degrees
    still = np.zeros(len(t), bool)
    j = 0
    for i in range(len(t)):
        while j < len(t) - 1 and t[j] < t[i] + 1.0:
            j += 1
        if t[j] - t[i] >= 0.8:
            still[i] = np.linalg.norm(P[j] - P[i]) < 0.03 and float(F[j] @ F[i]) > np.cos(np.radians(3.0))
    step = np.linalg.norm(np.diff(P, axis=0), axis=1)
    dyaw = np.abs((np.diff(yaw) + 180.0) % 360.0 - 180.0)
    return {
        "duration_s": round(float(t[-1] - t[0]), 1), "poses": int(len(t)),
        "footprint_m": [round(float(ext[0]), 2), round(float(ext[1]), 2)], "footprint_area_m2": round(area, 2),
        "floor_cells_visited": int(cells), "path_m": round(float(step.sum()), 1),
        "turned_deg": int(round(float(dyaw.sum()))), "directions_seen_share": round(float(min(1.0, directions)), 2),
        "pitch_deg": {"lowest": round(float(np.percentile(pitch, 2)), 1), "median": round(float(np.median(pitch)), 1),
                      "highest": round(float(np.percentile(pitch, 98)), 1)},
        "distinct_views": int(len(spent)), "repeated_share": round(repeated / total, 2),
        "standing_still_share": round(float(dt[still].sum()) / total, 2),
    }


def wall_numbers(room: dict | None) -> dict:
    walls = (room or {}).get("walls") or []
    if not walls:
        return {"walls": 0, "corners": 0, "walls_seen_share": None}
    area = np.array([w["width"] * w["height"] for w in walls], float)
    cov = np.array([w.get("coverage") or 0.0 for w in walls], float)
    return {"walls": len(walls), "corners": len((room or {}).get("corners") or []),
            "walls_seen_share": round(float((area * cov).sum() / max(area.sum(), 1e-9)), 2),
            "least_seen_wall_share": round(float(cov.min()), 2)}


def judge(motion: dict | None, walls: dict, registered_share: float | None) -> tuple[str, list[dict], str]:
    """(verdict, issues, one sentence). ``limited``: usable where and how it was scanned; ``rescan``: do it again."""
    issues: list[dict] = []

    def issue(code, value, limit, text, do):
        issues.append({"code": code, "value": value, "limit": limit, "text": text, "do": do})

    if motion:
        fp = max(motion["footprint_m"])
        if fp < LIMITS["footprint_min_m"]:
            issue("one_spot", fp, LIMITS["footprint_min_m"],
                  f"The scan was made from one spot: the phone stayed within {motion['footprint_m'][0]:.1f} x {motion['footprint_m'][1]:.1f} m. "
                  "The room is known only as seen from there.",
                  "Walk through the room while recording: along the walls, and through the places where visitors will stand.")
        if motion["pitch_deg"]["lowest"] > LIMITS["look_down_deg"]:
            issue("never_looked_down", motion["pitch_deg"]["lowest"], LIMITS["look_down_deg"],
                  f"The phone never looked down further than {abs(motion['pitch_deg']['lowest']):.0f} degrees. People hold a phone tilted "
                  "down; that view will not be recognised.",
                  "Do one round with the phone tilted down 30 - 45 degrees, the way it is normally held.")
        if motion["pitch_deg"]["highest"] < LIMITS["look_up_deg"]:
            issue("never_looked_up", motion["pitch_deg"]["highest"], LIMITS["look_up_deg"],
                  f"The phone never looked up (highest {motion['pitch_deg']['highest']:.0f} degrees): the walls were not seen up to the ceiling.",
                  "In every corner, move the phone once slowly from the floor to the ceiling.")
        if motion["repeated_share"] > LIMITS["repeated_share_max"]:
            still = motion["standing_still_share"]
            issue("repeated_views", motion["repeated_share"], LIMITS["repeated_share_max"],
                  f"{motion['repeated_share'] * 100:.0f} % of the recording shows views it already had"
                  + (f" ({still * 100:.0f} % of the time the phone stood still)." if still > 0.1 else " (turning on the spot again and again)."),
                  "Keep moving to new places; five seconds on one view are enough.")
        if motion["directions_seen_share"] < LIMITS["directions_min"]:
            issue("not_all_around", motion["directions_seen_share"], LIMITS["directions_min"],
                  f"Only {motion['directions_seen_share'] * 100:.0f} % of the directions around were looked at.",
                  "Look all the way around in several places of the room.")
    if walls.get("walls_seen_share") is not None and walls["walls_seen_share"] < LIMITS["walls_seen_min"]:
        issue("walls_poorly_seen", walls["walls_seen_share"], LIMITS["walls_seen_min"],
              f"Only {walls['walls_seen_share'] * 100:.0f} % of the walls' area was seen (furniture in front, or not looked at).",
              "Walk along the walls about a metre away from them, looking across the room at the opposite wall.")
    if registered_share is not None and registered_share < LIMITS["registered_min"]:
        issue("frames_lost", round(registered_share, 2), LIMITS["registered_min"],
              f"Only {registered_share * 100:.0f} % of the chosen frames could be joined into the map (blur, bare walls or too fast).",
              "Move slower, with more light, and keep furniture or door frames in view.")
    hard = {"frames_lost"} & {i["code"] for i in issues} and registered_share is not None and registered_share < 0.5
    verdict = "good" if not issues else "rescan" if (hard or len(issues) >= 3) else "limited"
    line = {"good": "The scan is good: the room was seen from several places, up and down, all around.",
            "limited": "The scan is usable with limits: " + " ".join(i["text"] for i in issues),
            "rescan": "Scan the room again: " + " ".join(i["text"] for i in issues)}[verdict]
    return verdict, issues, line


def scan_quality(positions=None, forwards=None, times_s=None, up=(0.0, 1.0, 0.0), room: dict | None = None,
                 registered: int | None = None, selected: int | None = None, source: str = "recording") -> dict:
    """The whole answer. Without poses (``positions`` None) only the walls and the frame counts are judged."""
    motion = motion_numbers(positions, forwards, times_s, up) if positions is not None and len(positions) >= 5 else None
    walls = wall_numbers(room)
    share = (registered / selected) if registered is not None and selected else None
    verdict, issues, line = judge(motion, walls, share)
    return {"version": 1, "verdict": verdict, "summary": line, "issues": issues, "poses_from": source if motion else None,
            "motion": motion, "walls": walls,
            "frames": {"selected": selected, "in_map": registered, "share": round(share, 2) if share is not None else None},
            "limits": LIMITS}


def from_session_log(slog, **kw) -> dict:
    """From the phone's pose lines (``recording.parse_session_jsonl``): AR camera looks along its -Z, the world's up is +Y."""
    P = np.array([p.T_arworld_from_arcam[:3, 3] for p in slog.poses])
    F = np.array([-p.T_arworld_from_arcam[:3, 2] for p in slog.poses])
    t = np.array([p.t_ns for p in slog.poses], float) / 1e9
    return scan_quality(P, F, t - t[0], (0.0, 1.0, 0.0), source="recording", **kw)


def from_map_images(lmap, up, **kw) -> dict:
    """From the cameras of the map itself (no recording at hand): fewer, and without real times - half a second each."""
    unit = lmap.scale_m_per_unit or 1.0
    P = np.array([im.center for im in lmap.images]) * unit
    F = np.array([im.T_cvcam_from_map[2, :3] for im in lmap.images])
    return scan_quality(P, F, np.arange(len(P)) * 0.5, up, source="map_images", **kw)
