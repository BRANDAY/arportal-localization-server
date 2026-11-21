"""The coloured 3D model of the room: depth images fused into one surface, in the map frame."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from locserver import geometry as g
from locserver import roommodel as rm

ROOM = np.array([4.0, 2.6, 3.0])                     # x, y (up), z in metres; the room is the box [0, ROOM]
FLOOR, WALL, CEIL = (40, 90, 160), (200, 200, 190), (250, 250, 250)   # BGR
W, H, F = 128, 96, 80.0
CAM = {"fx": F, "fy": F, "cx": W / 2, "cy": H / 2}


def look(center, yaw_deg, pitch_deg=0.0):
    """A camera at ``center`` (OpenCV axes: x right, y down, z forward) turned about the vertical and tilted."""
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    fwd = np.array([np.sin(y) * np.cos(p), np.sin(p), np.cos(y) * np.cos(p)])
    right = np.cross(fwd, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R_wc = np.stack([right, down, fwd], 1)            # columns: camera axes in the world
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R_wc.T, -R_wc.T @ np.asarray(center, float)
    return SimpleNamespace(name=f"y{yaw_deg}p{pitch_deg}", T_cvcam_from_map=T, center=np.asarray(center, float))


def render(im):
    """Exact depth and colour of the box seen by ``im``."""
    ys, xs = np.mgrid[0:H, 0:W]
    d = np.stack([(xs + 0.5 - CAM["cx"]) / F, (ys + 0.5 - CAM["cy"]) / F, np.ones((H, W))], -1) @ im.T_cvcam_from_map[:3, :3]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(d > 0, (ROOM - im.center) / d, (0.0 - im.center) / d)   # distance to the three planes ahead
    axis = t.argmin(-1)
    depth = t.min(-1)
    color = np.empty((H, W, 3), np.uint8)
    color[:] = WALL
    color[(axis == 1) & (d[..., 1] < 0)] = FLOOR
    color[(axis == 1) & (d[..., 1] > 0)] = CEIL
    return depth.astype(np.float32), color


def scan(centers=((1.8, 1.4, 1.4), (2.3, 1.5, 1.7), (2.0, 1.3, 1.2)), yaws=range(0, 360, 30), pitches=(-35, 0, 35)):
    opts = rm.ModelOptions(voxel_m=0.06, depth_long_side=W, min_piece_m2=0.05)
    frames = []
    for c in centers:
        for yaw in yaws:
            for pitch in pitches:
                im = look(c, yaw, pitch)
                depth, color = render(im)
                frames.append(rm.prepared_from_metric(depth, im, CAM, color, opts))
    return frames, opts


def box_distance(v):
    """Distance of points to the nearest face of the room."""
    return np.minimum(np.abs(v), np.abs(v - ROOM)).min(1)


@pytest.fixture(scope="module")
def fused():
    frames, opts = scan()
    grid = rm.fuse(frames, 1.0, opts, device="cpu")
    return frames, opts, grid, rm.surface_nets(grid, opts.min_weight)


def test_the_surface_lies_on_the_walls_and_looks_into_the_room(fused):
    _, opts, grid, (verts, colors, normals, tris) = fused
    assert len(verts) > 2000 and len(tris) > 4000 and tris.max() < len(verts)
    dist = box_distance(verts)
    assert np.median(dist) < 0.25 * grid["voxel_m"] and np.percentile(dist, 99) < 1.0 * grid["voxel_m"]
    # every face of the box is there: the scan looked all around, up and down
    for axis in range(3):
        for side in (0.0, ROOM[axis]):
            assert (np.abs(verts[:, axis] - side) < 0.06).sum() > 150, (axis, side)
    # the normals look into the room (towards the cameras)
    inward = (ROOM / 2 - verts) / np.linalg.norm(ROOM / 2 - verts, axis=1, keepdims=True)
    assert np.mean((normals * inward).sum(1) > 0.2) > 0.9
    # and the faces are wound the same way as the normals
    p = verts[tris.astype(np.int64)].astype(float)
    face_n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    assert np.mean((face_n * normals[tris[:, 0].astype(np.int64)]).sum(1) > 0) > 0.97
    area = 0.5 * np.linalg.norm(face_n, axis=1).sum()
    true_area = 2 * (ROOM[0] * ROOM[1] + ROOM[1] * ROOM[2] + ROOM[0] * ROOM[2])
    assert 0.85 * true_area < area < 1.1 * true_area


def test_the_colours_come_from_the_images(fused):
    _, _, _, (verts, colors, _, _) = fused
    mid = (verts[:, 0] > 0.5) & (verts[:, 0] < 3.5) & (verts[:, 2] > 0.5) & (verts[:, 2] < 2.5)
    floor = colors[mid & (verts[:, 1] < 0.08)].mean(0)       # stored as RGB
    ceil = colors[mid & (verts[:, 1] > ROOM[1] - 0.08)].mean(0)
    assert np.allclose(floor, FLOOR[::-1], atol=12) and np.allclose(ceil, CEIL[::-1], atol=12)


def test_a_phone_held_still_adds_nothing_and_floating_pieces_go(fused):
    frames, opts, _, (verts, colors, normals, tris) = fused
    ims = [look((2, 1.4, 1.5), 0), look((2, 1.4, 1.5), 1), look((2.01, 1.4, 1.5), 0.5), look((2, 1.4, 1.5), 20), look((2.3, 1.4, 1.5), 0)]
    assert [im.name for im in rm.pick_model_images(ims, opts)] == ["y0p0.0", "y20p0.0", "y0p0.0"]
    # a small triangle far from everything is a floating piece
    extra_v = np.array([[9, 9, 9], [9.05, 9, 9], [9, 9.05, 9]], np.float32)
    v2 = np.concatenate([verts, extra_v])
    t2 = np.concatenate([tris, np.array([[len(verts), len(verts) + 1, len(verts) + 2]], np.uint32)])
    c2, n2 = np.concatenate([colors, colors[:3]]), np.concatenate([normals, normals[:3]])
    v3, c3, n3, t3 = rm.drop_small_pieces(v2, c2, n2, t2, opts.min_piece_m2)
    assert v3.max() < 5 and len(t3) <= len(tris) and len(t3) > 0.95 * len(tris) and t3.max() < len(v3)


def test_the_depth_of_the_network_is_bent_to_the_map_points():
    """Relative inverse depth with a distortion that scale and shift cannot take out: the correction field does."""
    im = look((2.0, 1.4, 1.5), 40, -10)
    depth, color = render(im)
    ys, xs = np.mgrid[0:H, 0:W]
    warp = 1.0 + 0.10 * (xs / W - 0.5)                         # the network sees the right side 5 % too far, the left too near
    disp = (2.0 / (depth * warp) + 0.3).astype(np.float32)     # "inverse depth up to scale and shift"
    rng = np.random.default_rng(0)
    px = np.stack([rng.uniform(2, W - 2, 400), rng.uniform(2, H - 2, 400)], 1)
    z = depth[px[:, 1].astype(int), px[:, 0].astype(int)].astype(float)
    cam = np.stack([(np.floor(px[:, 0]) + 0.5 - CAM["cx"]) / F * z, (np.floor(px[:, 1]) + 0.5 - CAM["cy"]) / F * z, z], 1)
    Twc = g.invert_T(im.T_cvcam_from_map)
    im.xyz = (cam @ Twc[:3, :3].T + Twc[:3, 3]).astype(np.float32)
    im.features = SimpleNamespace(keypoints=px.astype(np.float32))
    opts = rm.ModelOptions(depth_long_side=W)
    p, info = rm.prepare_depth(disp, im, CAM, color, 1.0, opts)
    assert info["used"] and info["median_rel_error_corrected"] < 0.4 * info["median_rel_error"]
    ok = np.isfinite(p.depth)
    assert ok.mean() > 0.7 and np.median(np.abs(p.depth[ok] - depth[ok]) / depth[ok]) < 0.012
    plain = rm.ModelOptions(depth_long_side=W, field_prior=1e9)   # the field switched off
    q, _ = rm.prepare_depth(disp, im, CAM, color, 1.0, plain)
    okq = np.isfinite(q.depth)
    assert np.median(np.abs(q.depth[okq] - depth[okq]) / depth[okq]) > 2 * np.median(np.abs(p.depth[ok] - depth[ok]) / depth[ok])
    # an image whose depth does not fit the points is refused
    bad, why = rm.prepare_depth(rng.uniform(0.1, 3, disp.shape).astype(np.float32), im, CAM, color, 1.0, opts)
    assert bad is None and not why["used"]


def test_the_model_file_round_trip_and_ply(tmp_path, fused):
    _, _, grid, (verts, colors, normals, tris) = fused
    rm.write_model(tmp_path / rm.MODEL_BIN, verts, colors, normals, tris, grid["voxel_m"])
    m = rm.read_model(tmp_path / rm.MODEL_BIN)
    assert np.allclose(m["vertices"], verts) and (m["colors"] == colors).all() and (m["triangles"] == tris).all()
    assert np.abs(m["normals"] - normals).max() < 0.01 and m["voxel"] == pytest.approx(grid["voxel_m"])
    assert np.allclose(m["bbox_min"], verts.min(0)) and np.allclose(m["bbox_max"], verts.max(0))
    ply = rm.model_ply(m)
    head = ply[:ply.index(b"end_header\n")].decode()
    assert f"element vertex {len(verts)}" in head and f"element face {len(tris)}" in head
    assert len(ply) == ply.index(b"end_header\n") + 11 + 15 * len(verts) + 13 * len(tris)
    (tmp_path / "x.bin").write_bytes(b"nope")
    with pytest.raises(ValueError):
        rm.read_model(tmp_path / "x.bin")
    assert rm.load_info(tmp_path) is None                       # no model.json: no model


def test_the_room_plan_is_made_from_the_same_depth():
    from locserver.roomplan import compute_room_plan

    frames, _ = scan()
    ims = [look(c, yaw, pitch) for c in ((1.8, 1.4, 1.4), (2.3, 1.5, 1.7), (2.0, 1.3, 1.2)) for yaw in range(0, 360, 30) for pitch in (-35, 0, 35)]
    lmap = SimpleNamespace(scale_m_per_unit=1.0, images=ims)
    pairs = rm.plan_views(frames, lmap)
    assert len(pairs) == len(frames) and all(im.name == v.name for v, im in pairs)
    room = compute_room_plan([v for v, _ in pairs], (0.0, 1.0, 0.0))
    assert room["status"] == "ok" and len(room["walls"]) == 4 and len(room["corners"]) == 4
    assert all(abs(c["angle_deg"] - 90) < 2 for c in room["corners"])
    assert abs(room["floor"]["level"]) < 0.05 and abs(room["height"] - ROOM[1]) < 0.1


def test_walls_nearly_square_to_the_room_are_squared_and_a_slanted_one_stays():
    from locserver.roomplan import PlanOptions, Wall, square_walls

    def wall(deg, d):
        n = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
        return Wall(n, d, -1.0, 1.0, 0.0, 2.5)

    rng = np.random.default_rng(0)
    main = np.radians(6.0)                                   # the room's own axes are turned 6 degrees in the map
    a = np.concatenate([main + np.pi / 2 * rng.integers(0, 4, 900) + rng.normal(0, np.radians(3), 900),
                        rng.uniform(-np.pi, np.pi, 300)])    # a quarter of the surface points anywhere (furniture)
    n2 = np.stack([np.cos(a), np.sin(a)], 1)
    walls = [wall(6 + 4, 2.0), wall(96 - 3, 1.5), wall(186 + 9, 2.0), wall(276 - 1, 1.5), wall(51, 3.0)]
    mids = [w.point(0.0, 0)[:2] for w in walls]
    info = square_walls(walls, n2, PlanOptions())
    assert info["applied"] and abs(info["main_direction_deg"] - 6.0) < 1.0 and info["share_within_10_deg"] > 0.6
    for w, deg, mid in zip(walls[:4], (6, 96, 186, 276), mids):
        assert np.degrees(np.arccos(np.clip(w.n @ [np.cos(np.radians(deg)), np.sin(np.radians(deg))], -1, 1))) < 1.2
        assert abs(mid @ w.n - w.d) < 1e-9 and w.u1 - w.u0 > 1.9      # turned about its middle, as long as before
    assert np.degrees(np.arctan2(walls[4].n[1], walls[4].n[0])) == pytest.approx(51)
    # a room that is not square at all is left alone
    loose = [wall(10, 2.0)]
    assert not square_walls(loose, np.stack([np.cos(a[900:]), np.sin(a[900:])], 1), PlanOptions())["applied"]
    assert np.degrees(np.arctan2(loose[0].n[1], loose[0].n[0])) == pytest.approx(10)
