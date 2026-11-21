"""Pictures of a room model without a browser: a view from above and four views from where the scan stood.

    uv run python scripts\\render_room_model.py data\\maps\\demo_room --out data\\work\\demo_room_model.png

For reports and for checking a build from the command line; the panel (``/ui``) shows the same model live.
The model is drawn as coloured dots (its vertices), nearest first; walls of ``room.json`` and the door are drawn in
the view from above.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from locserver import roommodel as rm  # noqa: E402


def splat(img: np.ndarray, zbuf: np.ndarray, u, v, z, colors, radius: int) -> None:
    order = np.argsort(-z)                               # far first, near last: near wins
    u, v, colors = u[order], v[order], colors[order]
    h, w = zbuf.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            x, y = u + dx, v + dy
            ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            img[y[ok], x[ok]] = colors[ok]


def perspective(verts, colors, eye, yaw_deg, pitch_deg, up_axis: int, size=(640, 480), fov_deg=75.0) -> np.ndarray:
    """What a camera at ``eye`` sees; the map's "up" is +Y (ARCore) or +Z (test data)."""
    up = np.zeros(3)
    up[up_axis] = 1.0
    a, b = [ax for ax in range(3) if ax != up_axis]
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    fwd = np.zeros(3)
    fwd[a], fwd[b], fwd[up_axis] = np.cos(y) * np.cos(p), np.sin(y) * np.cos(p), np.sin(p)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    cam = (verts - eye) @ np.stack([right, down, fwd], 1)
    w, h = size
    f = 0.5 * w / np.tan(np.radians(fov_deg) / 2)
    ok = cam[:, 2] > 0.05
    u = (cam[ok, 0] / cam[ok, 2] * f + w / 2).astype(int)
    v = (cam[ok, 1] / cam[ok, 2] * f + h / 2).astype(int)
    img = np.full((h, w, 3), 24, np.uint8)
    splat(img, np.zeros((h, w)), u, v, cam[ok, 2], colors[ok][:, ::-1], 1)
    return img


def top(verts, colors, up_axis: int, room: dict | None, portal: dict | None, centers: np.ndarray | None, px_per_m=90.0,
        cut_below_top_m=0.5):
    """From above, with the ceiling (and everything within ``cut_below_top_m`` of the top) taken off."""
    a, b = [ax for ax in range(3) if ax != up_axis]
    lo, hi = verts.min(0), verts.max(0)
    keep = verts[:, up_axis] < hi[up_axis] - cut_below_top_m
    w, h = int((hi[a] - lo[a]) * px_per_m) + 40, int((hi[b] - lo[b]) * px_per_m) + 40
    to_px = lambda p: (((np.asarray(p, float)[..., a] - lo[a]) * px_per_m + 20).astype(int),  # noqa: E731
                       ((np.asarray(p, float)[..., b] - lo[b]) * px_per_m + 20).astype(int))
    img = np.full((h, w, 3), 24, np.uint8)
    u, v = to_px(verts[keep])
    splat(img, np.zeros((h, w)), u, v, -verts[keep, up_axis], colors[keep][:, ::-1], 1)
    if centers is not None and len(centers):
        cu, cv_ = to_px(centers)
        for x, y in zip(cu, cv_):
            cv2.circle(img, (int(x), int(y)), 2, (80, 220, 80), -1)
    for wall in (room or {}).get("walls", []):
        p0, p1 = to_px(np.array(wall["corners"][0])), to_px(np.array(wall["corners"][1]))
        cv2.line(img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (255, 160, 60), 2)
    if portal and portal.get("position"):
        x, y = to_px(np.array(portal["position"]))
        cv2.circle(img, (int(x), int(y)), 7, (40, 140, 255), 2)
    cv2.line(img, (20, h - 12), (20 + int(px_per_m), h - 12), (230, 230, 230), 2)
    cv2.putText(img, "1 m", (26 + int(px_per_m), h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
    return img


def sheet(map_dir: Path, out: Path) -> dict:
    model = rm.read_model(map_dir / rm.MODEL_BIN)
    verts, colors = model["vertices"].astype(float), model["colors"]
    meta = json.loads((map_dir / "map.json").read_text(encoding="utf-8"))
    up = np.asarray(meta.get("up") or [0, 1, 0], float)
    up_axis = int(np.argmax(np.abs(up)))
    room = json.loads((map_dir / "room.json").read_text(encoding="utf-8")) if (map_dir / "room.json").exists() else None
    anchors = json.loads((map_dir / "anchors.json").read_text(encoding="utf-8")) if (map_dir / "anchors.json").exists() else {}
    portal = (anchors.get("anchors") or anchors).get("portal")
    eye = np.median(verts, axis=0)                      # about where the scan stood: the middle of what was seen
    eye[up_axis] = verts[:, up_axis].min() + 1.4
    views = [perspective(verts, colors, eye, yaw, -8, up_axis) for yaw in (0, 90, 180, 270)]
    t = top(verts, colors, up_axis, room, portal, None)
    row = np.concatenate(views[:2], 1), np.concatenate(views[2:], 1)
    grid = np.concatenate(row, 0)
    s = grid.shape[0] / t.shape[0]
    t = cv2.resize(t, (max(1, int(t.shape[1] * s)), grid.shape[0]), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out), np.concatenate([t, grid], 1))
    return {"vertices": len(verts), "triangles": len(model["triangles"]), "out": str(out)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("map_dir")
    ap.add_argument("--out", default="room_model.png")
    a = ap.parse_args()
    print(sheet(Path(a.map_dir), Path(a.out)))
