"""A coloured 3D model of the scanned room: walls, floor, furniture - what the phone saw, as one surface.

The sparse map (a few thousand points per image) is what the room is
*recognised* by; it is not something a person can look at. This module makes
the thing to look at, from the same images and poses:

1. **Depth per image.** Depth Anything V2 gives relative inverse depth; it is
   made metric with the map's own 3D points seen in that image
   (``1/z = a*d + b``, robust), and then *bent* to them: a smooth field of
   corrections (a few per cent, varying slowly over the image) takes out what
   the two numbers cannot. Without it the depth of neighbouring images
   disagrees by 5 - 15 cm at three metres and the fused walls are doubled.
2. **Fusion.** All depth images go into one voxel grid of truncated signed
   distances (TSDF, as in KinectFusion): every voxel averages what the images
   say about the nearest surface along their rays. One wrong image is
   outvoted; free space seen by one image removes what another put there.
3. **Surface.** One vertex per grid cell the surface passes through, a quad
   per grid edge it crosses ("surface nets"), colours from the images, small
   floating pieces dropped.

The result is in the map frame (the frame of ``room.json`` and of the door),
written next to the map as ``model.bin`` (+ ``model.json`` with numbers).

What it cannot do: it shows only what was seen. Behind furniture, and in every
direction the phone did not look, there are holes; a scan from one spot gives
a room seen from that spot.
"""

from __future__ import annotations

import json
import logging
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import geometry as g
from .mapstore import LocMap, utc_now

log = logging.getLogger(__name__)

MAGIC = b"LRM1"
MODEL_BIN, MODEL_INFO = "model.bin", "model.json"
Progress = Callable[[str, float, str], None]


@dataclass
class ModelOptions:
    voxel_m: float = 0.03            # edge of a voxel; enlarged when the room would need more than `max_voxels`
    max_voxels: int = 12_000_000
    trunc_voxels: float = 4.0        # the distance band around a surface, in voxels
    max_depth_m: float = 7.0         # farther than this the depth of one image is not trusted
    depth_long_side: int = 384       # size of the depth images that are fused
    max_frames: int = 220
    min_move_m: float = 0.04         # an image this close to one already taken ...
    min_turn_deg: float = 3.0        # ... and turned this little adds nothing: skipped
    min_weight: float = 1.2          # a voxel counts when seen about twice from 1.5 m (weight of a view: 1 / max(z, 1)^2)
    edge_rel: float = 0.07           # relative depth step between neighbouring pixels that counts as an occlusion edge
    min_piece_m2: float = 0.08       # floating pieces smaller than this are dropped
    field_cells: int = 20            # the correction field: cells along the long side ...
    field_sigma: float = 0.10        # ... smoothing radius as a share of the long side ...
    field_prior: float = 1.0         # ... and how many points' worth of "no correction" every place starts with
    min_inlier_fraction: float = 0.35
    max_median_rel_error: float = 0.12


@dataclass
class PreparedDepth:
    """One map image with metric depth (metres, nan = unknown) at fusion resolution."""

    name: str
    T_cvcam_from_map: np.ndarray  # map units, as in the map
    fx: float
    fy: float
    cx: float
    cy: float
    depth: np.ndarray             # float32 HxW
    color: np.ndarray             # uint8 HxWx3, BGR
    info: dict


# ------------------------------------------------------------------ depth --- #
def correction_field(xy: np.ndarray, log_ratio: np.ndarray, width: int, height: int, opts: ModelOptions) -> np.ndarray:
    """A smooth field (coarse grid, float32) of ``log(true depth / predicted depth)`` from scattered points.

    Kernel regression with a prior of "no correction": where there are no map
    points the field goes back to zero instead of extrapolating.
    """
    long_side = max(width, height)
    cell = long_side / opts.field_cells
    gw, gh = max(2, int(np.ceil(width / cell))), max(2, int(np.ceil(height / cell)))
    gx = (np.arange(gw) + 0.5) * width / gw
    gy = (np.arange(gh) + 0.5) * height / gh
    sigma = opts.field_sigma * long_side
    wx = np.exp(-0.5 * ((gx[:, None] - xy[None, :, 0]) / sigma) ** 2)   # gw x N
    wy = np.exp(-0.5 * ((gy[:, None] - xy[None, :, 1]) / sigma) ** 2)   # gh x N
    num = (wy * log_ratio[None, :]) @ wx.T                              # gh x gw
    den = wy @ wx.T
    return (num / (den + opts.field_prior)).astype(np.float32)


def _edge_mask(depth: np.ndarray, rel: float) -> np.ndarray:
    """Pixels next to a jump in depth: there the surface between two pixels does not exist."""
    z = np.where(np.isfinite(depth), depth, 0.0)
    bad = np.zeros(z.shape, bool)
    dx = np.abs(z[:, 1:] - z[:, :-1]) > rel * np.minimum(z[:, 1:], z[:, :-1])
    dy = np.abs(z[1:, :] - z[:-1, :]) > rel * np.minimum(z[1:, :], z[:-1, :])
    bad[:, 1:] |= dx
    bad[:, :-1] |= dx
    bad[1:, :] |= dy
    bad[:-1, :] |= dy
    return cv2.dilate(bad.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


def _resized(bgr: np.ndarray, long_side: int) -> tuple[np.ndarray, float, float]:
    h, w = bgr.shape[:2]
    s = long_side / max(h, w)
    out = cv2.resize(bgr, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    return out, out.shape[1] / w, out.shape[0] / h


def prepare_depth(disp: np.ndarray, im, cam: dict, bgr: np.ndarray, unit_m: float = 1.0,
                  opts: ModelOptions | None = None) -> tuple[PreparedDepth | None, dict]:
    """Metric, corrected depth of map image ``im`` from its relative inverse depth ``disp`` (same size as the image)."""
    from .roomstage import STRICT_INLIER_FRACTION, STRICT_MEDIAN_REL_ERROR, fit_inverse_depth

    opts = opts or ModelOptions()
    h, w = disp.shape
    has = np.isfinite(im.xyz).all(1)
    kp = im.features.keypoints[has].astype(float)
    Xc = im.xyz[has].astype(float) @ im.T_cvcam_from_map[:3, :3].T + im.T_cvcam_from_map[:3, 3]
    z = Xc[:, 2] * unit_m
    xi = np.clip(np.round(kp[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(kp[:, 1]).astype(int), 0, h - 1)
    d = disp[yi, xi].astype(float)
    info = {"image": im.name, "points": int(has.sum())}
    fit = fit_inverse_depth(d, z)
    if fit is None:
        return None, dict(info, used=False, reason="too few 3D points to scale the depth")
    a, b, frac, err = fit
    info.update(inlier_fraction=round(frac, 3), median_rel_error=round(err, 4),
                strict=bool(frac >= STRICT_INLIER_FRACTION and err <= STRICT_MEDIAN_REL_ERROR))
    if frac < opts.min_inlier_fraction or err > opts.max_median_rel_error:
        return None, dict(info, used=False, reason="depth does not agree with the map points")

    # bend the depth to the map points: a smooth field of log(true / predicted)
    pred = a * d + b
    zp = 1.0 / np.where(pred > 1e-9, pred, np.nan)
    ok = np.isfinite(zp) & (z > 0.15) & (np.abs(zp - z) < 0.3 * z)
    field = None
    if ok.sum() >= 30:
        lr = np.log(z[ok] / zp[ok])
        field = correction_field(kp[ok], lr, w, h, opts)
        fy_i = np.clip((kp[ok, 1] * field.shape[0] / h).astype(int), 0, field.shape[0] - 1)
        fx_i = np.clip((kp[ok, 0] * field.shape[1] / w).astype(int), 0, field.shape[1] - 1)
        info["median_rel_error_corrected"] = round(float(np.median(np.abs(np.expm1(lr - field[fy_i, fx_i])))), 4)

    color, sx, sy = _resized(bgr, opts.depth_long_side)
    hh, ww = color.shape[:2]
    small = cv2.resize(disp, (ww, hh), interpolation=cv2.INTER_AREA)
    inv = a * small + b
    depth = np.where(inv > 1.0 / (opts.max_depth_m * 1.5), 1.0 / np.maximum(inv, 1e-9), np.nan)
    if field is not None:
        depth = depth * np.exp(cv2.resize(field, (ww, hh), interpolation=cv2.INTER_CUBIC))
    depth = np.where((depth > 0.15) & (depth < opts.max_depth_m), depth, np.nan)
    depth[_edge_mask(depth, opts.edge_rel)] = np.nan
    return PreparedDepth(im.name, im.T_cvcam_from_map.copy(), cam["fx"] * sx, cam["fy"] * sy, cam["cx"] * sx, cam["cy"] * sy,
                         depth.astype(np.float32), color, info), dict(info, used=True)


def prepared_from_metric(depth_z: np.ndarray, im, cam: dict, bgr: np.ndarray, opts: ModelOptions | None = None) -> PreparedDepth:
    """The same from a measured depth image (metres, at the size of the image) - tests and evaluation."""
    opts = opts or ModelOptions()
    color, sx, sy = _resized(bgr, opts.depth_long_side)
    d = np.where(np.isfinite(depth_z) & (depth_z > 0.15) & (depth_z < opts.max_depth_m), depth_z, np.nan).astype(np.float32)
    d = cv2.resize(d, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)
    d[_edge_mask(d, opts.edge_rel)] = np.nan
    return PreparedDepth(im.name, im.T_cvcam_from_map.copy(), cam["fx"] * sx, cam["fy"] * sy, cam["cx"] * sx, cam["cy"] * sy,
                         d, color, {"image": im.name, "used": True})


def pick_model_images(images: list, opts: ModelOptions, unit_m: float = 1.0) -> list:
    """Images that show something new: one that stands and looks like an earlier one is skipped (a phone held still)."""
    kept, centers, axes = [], [], []
    cos_min = np.cos(np.radians(opts.min_turn_deg))
    for im in images:
        c, z = im.center * unit_m, im.T_cvcam_from_map[2, :3]
        if kept:
            near = np.linalg.norm(np.array(centers) - c, axis=1) < opts.min_move_m
            same = np.array(axes) @ z > cos_min
            if (near & same).any():
                continue
        kept.append(im)
        centers.append(c)
        axes.append(z)
    if len(kept) > opts.max_frames:
        kept = [kept[i] for i in np.unique(np.linspace(0, len(kept) - 1, opts.max_frames).round().astype(int))]
    return kept


def prepare_all(lmap: LocMap, device: str = "auto", progress: Progress | None = None, opts: ModelOptions | None = None,
                disparity_of: Callable | None = None, stage: str = "depth") -> tuple[list[PreparedDepth], list[dict]]:
    """Depth of every image of the map that shows something new. ``disparity_of(image, bgr)`` replaces the network."""
    opts = opts or ModelOptions()
    progress = progress or (lambda s, f, m: None)
    unit = lmap.scale_m_per_unit or 1.0
    images = pick_model_images(lmap.images, opts, unit)
    net, out, infos = None, [], []
    try:
        if disparity_of is None:
            from .roomnets import DepthNet

            net = DepthNet(device)
        for k, im in enumerate(images):
            progress(stage, k / max(1, len(images)), f"depth of image {k + 1} of {len(images)}")
            bgr = cv2.imread(str(lmap.path / "images" / im.name), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            disp = disparity_of(im, bgr) if disparity_of is not None else net.disparity(bgr)
            p, info = prepare_depth(disp, im, lmap.camera, bgr, unit, opts)
            infos.append(info)
            if p is not None:
                out.append(p)
    finally:
        if net is not None:
            from .roomnets import release

            release(net)
    progress(stage, 1.0, f"depth of {len(out)} of {len(images)} images")
    return out, infos


def plan_views(frames: list[PreparedDepth], lmap: LocMap, long_side: int = 192) -> list[tuple]:
    """The same depth as input of the room plan: pairs of (:class:`locserver.roomplan.DepthView`, map image)."""
    from . import geometry as g
    from .roomplan import DepthView

    unit = lmap.scale_m_per_unit or 1.0
    by_name = {im.name: im for im in lmap.images}
    out = []
    for p in frames:
        h, w = p.depth.shape
        s = min(1.0, long_side / max(h, w))
        small = cv2.resize(p.depth, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_NEAREST)
        sx, sy = small.shape[1] / w, small.shape[0] / h
        T = g.invert_T(p.T_cvcam_from_map).copy()
        T[:3, 3] *= unit
        out.append((DepthView(T, p.fx * sx, p.fy * sy, p.cx * sx, p.cy * sy, small, name=p.name), by_name[p.name]))
    return out


# ----------------------------------------------------------------- fusion --- #
def _points_m(p: PreparedDepth, unit_m: float, stride: int = 6) -> np.ndarray:
    h, w = p.depth.shape
    ys, xs = np.mgrid[stride // 2:h:stride, stride // 2:w:stride]
    z = p.depth[ys, xs]
    ok = np.isfinite(z)
    cam = np.stack([(xs[ok] + 0.5 - p.cx) / p.fx * z[ok], (ys[ok] + 0.5 - p.cy) / p.fy * z[ok], z[ok]], 1)
    T = g.invert_T(p.T_cvcam_from_map)
    return cam @ T[:3, :3].T + T[:3, 3] * unit_m


def grid_for(frames: list[PreparedDepth], unit_m: float, opts: ModelOptions) -> tuple[np.ndarray, np.ndarray, float]:
    """(lowest corner in metres, number of voxels per axis, voxel edge) of a grid that holds what the images saw."""
    pts = np.concatenate([_points_m(p, unit_m) for p in frames])
    lo, hi = np.percentile(pts, 0.5, axis=0) - 0.25, np.percentile(pts, 99.5, axis=0) + 0.25
    voxel = opts.voxel_m
    while np.prod(np.ceil((hi - lo) / voxel) + 1) > opts.max_voxels:
        voxel *= 1.1
    return lo, (np.ceil((hi - lo) / voxel) + 1).astype(int), float(voxel)


def fuse(frames: list[PreparedDepth], unit_m: float = 1.0, opts: ModelOptions | None = None, device: str = "auto",
         progress: Progress | None = None, stage: str = "model") -> dict:
    """Fuse the depth images into a TSDF grid. Returns ``{tsdf, weight, color, color_weight, origin_m, voxel_m}`` (numpy)."""
    import torch

    opts = opts or ModelOptions()
    progress = progress or (lambda s, f, m: None)
    lo, dims, voxel = grid_for(frames, unit_m, opts)
    nx, ny, nz = (int(v) for v in dims)
    dev = torch.device("cuda" if (device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
    trunc = opts.trunc_voxels * voxel

    def run(dev):
        D = torch.ones((nx, ny, nz), dtype=torch.float32, device=dev)
        W = torch.zeros((nx, ny, nz), dtype=torch.float32, device=dev)
        C = torch.zeros((nx, ny, nz, 3), dtype=torch.float32, device=dev)
        CW = torch.zeros((nx, ny, nz), dtype=torch.float32, device=dev)
        ys = (torch.arange(ny, device=dev, dtype=torch.float32) * voxel + float(lo[1])).view(1, ny, 1)
        zs = (torch.arange(nz, device=dev, dtype=torch.float32) * voxel + float(lo[2])).view(1, 1, nz)
        slab = max(1, int(1_500_000 // (ny * nz)))
        for k, p in enumerate(frames):
            progress(stage, 0.8 * k / len(frames), f"fusing image {k + 1} of {len(frames)}")
            T = p.T_cvcam_from_map.astype(np.float64).copy()
            T[:3, 3] *= unit_m
            R = torch.tensor(T[:3, :3], dtype=torch.float32, device=dev)
            t = torch.tensor(T[:3, 3], dtype=torch.float32, device=dev)
            depth = torch.tensor(np.nan_to_num(p.depth, nan=0.0), device=dev)
            color = torch.tensor(p.color[:, :, ::-1].copy(), dtype=torch.float32, device=dev)  # RGB
            h, w = depth.shape
            for x0 in range(0, nx, slab):
                x1 = min(nx, x0 + slab)
                xs = (torch.arange(x0, x1, device=dev, dtype=torch.float32) * voxel + float(lo[0])).view(-1, 1, 1)
                zc = R[2, 0] * xs + R[2, 1] * ys + R[2, 2] * zs + t[2]
                front = zc > 0.15
                if not bool(front.any()):
                    continue
                zc_safe = torch.where(front, zc, torch.ones_like(zc))
                u = ((R[0, 0] * xs + R[0, 1] * ys + R[0, 2] * zs + t[0]) / zc_safe * p.fx + p.cx).floor().long()
                v = ((R[1, 0] * xs + R[1, 1] * ys + R[1, 2] * zs + t[1]) / zc_safe * p.fy + p.cy).floor().long()
                ok = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
                if not bool(ok.any()):
                    continue
                idx = ok.nonzero(as_tuple=True)
                uu, vv, zz = u[idx], v[idx], zc[idx]
                d = depth[vv, uu]
                sdf = d - zz
                keep = (d > 0) & (sdf > -trunc)
                if not bool(keep.any()):
                    continue
                gi = (idx[0][keep] + x0, idx[1][keep], idx[2][keep])
                d, sdf = d[keep], sdf[keep]
                wn = 1.0 / torch.clamp(d, min=1.0) ** 2
                tn = torch.clamp(sdf / trunc, max=1.0)
                w_old = W[gi]
                D[gi] = (D[gi] * w_old + tn * wn) / (w_old + wn)
                W[gi] = w_old + wn
                near = sdf < trunc
                if bool(near.any()):
                    gn = (gi[0][near], gi[1][near], gi[2][near])
                    cn, wc = color[vv[keep][near], uu[keep][near]], wn[near]
                    cw_old = CW[gn]
                    C[gn] = (C[gn] * cw_old[:, None] + cn * wc[:, None]) / (cw_old + wc)[:, None]
                    CW[gn] = cw_old + wc
        return {"tsdf": D.cpu().numpy(), "weight": W.cpu().numpy(), "color": C.cpu().numpy(), "color_weight": CW.cpu().numpy()}

    try:
        out = run(dev)
    except RuntimeError as e:  # a full graphics card: the same on the processor, slower
        if dev.type != "cuda" or "out of memory" not in str(e).lower():
            raise
        log.warning("fusion does not fit on the graphics card - running on the processor")
        torch.cuda.empty_cache()
        out = run(torch.device("cpu"))
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    out.update(origin_m=lo, voxel_m=voxel, device=dev.type)
    return out


# ---------------------------------------------------------------- surface --- #
_CORNERS = np.array([(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)])
_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if np.abs(_CORNERS[a] - _CORNERS[b]).sum() == 1]


def surface_nets(grid: dict, min_weight: float = 1.2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Surface of a TSDF grid: (vertices in metres, colours RGB uint8, unit normals, triangles)."""
    D, W, C, CW = grid["tsdf"], grid["weight"], grid["color"], grid["color_weight"]
    lo, voxel = np.asarray(grid["origin_m"], float), float(grid["voxel_m"])
    nx, ny, nz = D.shape
    valid, inside = W >= min_weight, D < 0

    def corner(A, c):
        i, j, k = _CORNERS[c]
        return A[i:nx - 1 + i, j:ny - 1 + j, k:nz - 1 + k]

    all_valid = np.ones((nx - 1, ny - 1, nz - 1), bool)
    n_in = np.zeros((nx - 1, ny - 1, nz - 1), np.uint8)
    for c in range(8):
        all_valid &= corner(valid, c)
        n_in += corner(inside, c)
    active = all_valid & (n_in > 0) & (n_in < 8)
    ci, cj, ck = np.nonzero(active)
    n = len(ci)
    if n == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32)
    cell = np.stack([ci, cj, ck], 1)
    dv = np.stack([D[ci + o[0], cj + o[1], ck + o[2]] for o in _CORNERS], 1)        # n x 8
    pos, cnt = np.zeros((n, 3)), np.zeros(n)
    for a, b in _EDGES:
        da, db = dv[:, a], dv[:, b]
        cross = (da < 0) != (db < 0)
        t = np.where(cross, da / np.where(cross, da - db, 1.0), 0.0)
        pos += np.where(cross[:, None], _CORNERS[a] + t[:, None] * (_CORNERS[b] - _CORNERS[a]), 0.0)
        cnt += cross
    local = pos / cnt[:, None]
    verts = lo + (cell + local) * voxel
    # colour: the corners' colours, by how well each was seen
    cw = np.stack([CW[ci + o[0], cj + o[1], ck + o[2]] for o in _CORNERS], 1)
    col = sum(C[ci + o[0], cj + o[1], ck + o[2]] * cw[:, c, None] for c, o in enumerate(_CORNERS)) / np.maximum(cw.sum(1), 1e-9)[:, None]
    seen = cw.sum(1) > 0
    col[~seen] = 128.0
    # normal: towards free space (positive distances)
    grad = np.stack([dv[:, _CORNERS[:, ax] == 1].mean(1) - dv[:, _CORNERS[:, ax] == 0].mean(1) for ax in range(3)], 1)
    normals = grad / np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-9)

    vid = np.full((nx - 1, ny - 1, nz - 1), -1, np.int32)
    vid[ci, cj, ck] = np.arange(n, dtype=np.int32)
    tris = []
    for ax in range(3):
        b, c = (ax + 1) % 3, (ax + 2) % 3          # the two axes across the edge, right-handed with `ax`
        sl = [slice(None)] * 3
        sl[ax] = slice(0, D.shape[ax] - 1)
        sl[b] = slice(1, D.shape[b] - 1)
        sl[c] = slice(1, D.shape[c] - 1)
        hi = list(sl)
        hi[ax] = slice(1, D.shape[ax])
        a0, a1 = D[tuple(sl)], D[tuple(hi)]
        crossing = ((a0 < 0) != (a1 < 0)) & valid[tuple(sl)] & valid[tuple(hi)]
        e = np.stack(np.nonzero(crossing), 1)
        if not len(e):
            continue
        e[:, b] += 1
        e[:, c] += 1                                 # back to voxel indices
        quad = []
        for db, dc in ((-1, -1), (0, -1), (0, 0), (-1, 0)):
            q = e.copy()
            q[:, b] += db
            q[:, c] += dc
            quad.append(vid[q[:, 0], q[:, 1], q[:, 2]])
        quad = np.stack(quad, 1)
        ok = (quad >= 0).all(1)
        quad, e = quad[ok], e[ok]
        flip = D[e[:, 0], e[:, 1], e[:, 2]] >= 0     # free space on the low side: the face looks the other way
        quad[flip] = quad[flip][:, ::-1]
        tris += [quad[:, [0, 1, 2]], quad[:, [0, 2, 3]]]
    tri = np.concatenate(tris).astype(np.uint32) if tris else np.zeros((0, 3), np.uint32)
    return verts.astype(np.float32), np.clip(col, 0, 255).astype(np.uint8), normals.astype(np.float32), tri


def drop_small_pieces(verts, colors, normals, tris, min_area_m2: float):
    """Remove floating pieces: connected parts of the surface smaller than ``min_area_m2`` (the largest always stays)."""
    import torch

    if not len(tris):
        return verts, colors, normals, tris
    t = torch.from_numpy(tris.astype(np.int64))
    label = torch.arange(len(verts), dtype=torch.int64)
    for _ in range(500):
        m = label[t].min(1).values
        new = label.scatter_reduce(0, t.reshape(-1), m.repeat_interleave(3), reduce="amin", include_self=True)
        new = new[new]                                # jump: a label points to its own label
        if bool((new == label).all()):
            break
        label = new
    lab = label.numpy()
    p = verts[tris.astype(np.int64)].astype(float)
    area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    per = np.bincount(lab[tris[:, 0].astype(np.int64)], weights=area, minlength=len(verts))
    keep_label = per >= min(min_area_m2, per.max())
    keep_tri = keep_label[lab[tris[:, 0].astype(np.int64)]]
    tris = tris[keep_tri]
    used = np.zeros(len(verts), bool)
    used[tris.reshape(-1).astype(np.int64)] = True
    remap = np.cumsum(used) - 1
    return verts[used], colors[used], normals[used], remap[tris.astype(np.int64)].astype(np.uint32)


# --------------------------------------------------------------------- io --- #
def write_model(path: Path, verts: np.ndarray, colors: np.ndarray, normals: np.ndarray, tris: np.ndarray, voxel: float) -> None:
    """``model.bin``: header, float32 positions, uint8 RGB, int8 normals (x127), uint32 triangles - each block 4-byte aligned."""
    n, m = len(verts), len(tris)
    lo = verts.min(0) if n else np.zeros(3)
    hi = verts.max(0) if n else np.zeros(3)

    def pad(b: bytes) -> bytes:
        return b + b"\0" * (-len(b) % 4)

    data = MAGIC + struct.pack("<IIf6f", n, m, voxel, *lo, *hi)
    data += verts.astype("<f4").tobytes()
    data += pad(colors.astype(np.uint8).tobytes())
    data += pad(np.clip(np.round(normals * 127), -127, 127).astype(np.int8).tobytes())
    data += tris.astype("<u4").tobytes()
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def read_model(path: Path) -> dict:
    raw = Path(path).read_bytes()
    if raw[:4] != MAGIC:
        raise ValueError("not a room model file")
    n, m, voxel, *box = struct.unpack_from("<IIf6f", raw, 4)
    o = 4 + struct.calcsize("<IIf6f")
    verts = np.frombuffer(raw, "<f4", 3 * n, o).reshape(n, 3)
    o += 12 * n
    colors = np.frombuffer(raw, np.uint8, 3 * n, o).reshape(n, 3)
    o += 3 * n + (-3 * n % 4)
    normals = np.frombuffer(raw, np.int8, 3 * n, o).reshape(n, 3).astype(np.float32) / 127.0
    o += 3 * n + (-3 * n % 4)
    tris = np.frombuffer(raw, "<u4", 3 * m, o).reshape(m, 3)
    return {"vertices": verts, "colors": colors, "normals": normals, "triangles": tris, "voxel": voxel,
            "bbox_min": box[:3], "bbox_max": box[3:]}


def model_ply(model: dict) -> bytes:
    """The model as a binary PLY (opens in MeshLab, Blender, CloudCompare)."""
    v, c, t = model["vertices"], model["colors"], model["triangles"]
    head = ("ply\nformat binary_little_endian 1.0\ncomment AR Portal room model, map frame\n"
            f"element vertex {len(v)}\nproperty float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            f"element face {len(t)}\nproperty list uchar int vertex_indices\nend_header\n").encode("ascii")
    vert = np.empty(len(v), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    vert["p"], vert["c"] = v, c
    face = np.empty(len(t), dtype=[("n", "u1"), ("i", "<i4", 3)])
    face["n"], face["i"] = 3, t
    return head + vert.tobytes() + face.tobytes()


def load_info(map_dir: Path) -> dict | None:
    f = Path(map_dir) / MODEL_INFO
    if not f.exists() or not (Path(map_dir) / MODEL_BIN).exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------- main --- #
def build_room_model(lmap: LocMap, frames: list[PreparedDepth], infos: list[dict] | None = None, device: str = "auto",
                     progress: Progress | None = None, opts: ModelOptions | None = None, stage: str = "model") -> dict:
    """Fuse ``frames`` and write ``model.bin`` / ``model.json`` next to the map. Returns the numbers of ``model.json``."""
    opts = opts or ModelOptions()
    progress = progress or (lambda s, f, m: None)
    unit = lmap.scale_m_per_unit or 1.0
    t0 = time.time()
    if len(frames) < 3:
        raise ValueError(f"only {len(frames)} images have usable depth - no model")
    grid = fuse(frames, unit, opts, device, progress, stage)
    progress(stage, 0.82, "extracting the surface")
    verts, colors, normals, tris = surface_nets(grid, opts.min_weight)
    n_before = len(tris)
    progress(stage, 0.92, "removing floating pieces")
    verts, colors, normals, tris = drop_small_pieces(verts, colors, normals, tris, opts.min_piece_m2)
    if not len(tris):
        raise ValueError("the fused depth has no surface")
    p = verts[tris.astype(np.int64)].astype(float)
    area = float(0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1).sum())
    write_model(lmap.path / MODEL_BIN, verts / unit, colors, normals, tris, grid["voxel_m"] / unit)
    infos = infos or [f.info for f in frames]
    used = [i for i in infos if i.get("used")]
    err = [i["median_rel_error"] for i in used if "median_rel_error" in i]
    err_c = [i["median_rel_error_corrected"] for i in used if "median_rel_error_corrected" in i]
    info = {
        "version": 1, "frame": "map", "created_utc": utc_now(), "format": "LRM1",
        "vertices": int(len(verts)), "triangles": int(len(tris)), "area_m2": round(area, 2),
        "voxel_m": round(float(grid["voxel_m"]), 4), "grid": [int(v) for v in grid["tsdf"].shape],
        "bbox_min": [round(float(v), 3) for v in verts.min(0) / unit], "bbox_max": [round(float(v), 3) for v in verts.max(0) / unit],
        "images_used": len(frames), "images_tried": len(infos), "images_rejected": len(infos) - len(used),
        "depth_median_rel_error": round(float(np.median(err)), 4) if err else None,
        "depth_median_rel_error_corrected": round(float(np.median(err_c)), 4) if err_c else None,
        "floating_triangles_removed": int(n_before - len(tris)),
        "sources": ["mono_depth", "sfm_points"], "fused_on": grid["device"], "build_time_s": round(time.time() - t0, 1),
        "bytes": int((lmap.path / MODEL_BIN).stat().st_size),
    }
    (lmap.path / MODEL_INFO).write_text(json.dumps(info, indent=1), encoding="utf-8")
    progress(stage, 1.0, f"room model: {info['triangles']} triangles, {info['area_m2']} m2 of surface")
    return info
