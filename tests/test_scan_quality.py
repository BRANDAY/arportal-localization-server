"""The verdict on a scan: from the path of the phone, the walls that were seen and the frames that made it into the map."""

from types import SimpleNamespace

import numpy as np

from locserver import scanquality as sq


def path(seconds, fps, xy_of, yaw_of, pitch_of):
    """Positions (y up, phone at 1.4 m), viewing directions and times of a walk."""
    t = np.arange(0, seconds, 1.0 / fps)
    P = np.array([[*xy_of(s), 0.0] for s in t])[:, [0, 2, 1]]
    P[:, 1] = 1.4
    yaw, pitch = np.radians([yaw_of(s) for s in t]), np.radians([pitch_of(s) for s in t])
    F = np.stack([np.cos(yaw) * np.cos(pitch), np.sin(pitch), np.sin(yaw) * np.cos(pitch)], 1)
    return P, F, t


def walk_through_the_room():
    # 80 s: once around a loop of 3 x 2.4 m, turning all the time, the phone going up and down between -45 and +25 degrees
    return path(80, 8, lambda s: (1.5 * np.cos(2 * np.pi * s / 80), 1.2 * np.sin(2 * np.pi * s / 80)),
                lambda s: 360.0 * s / 16, lambda s: -10 + 35 * np.sin(2 * np.pi * s / 11))


def turn_on_the_spot():
    # like the first real scan: three turns within half a metre, the phone a little down all the time, then 35 s standing
    return path(100, 8, lambda s: (0.25 * np.cos(min(s, 65) / 9), 0.2 * np.sin(min(s, 65) / 7)),
                lambda s: 1080.0 * min(s, 65) / 65, lambda s: -14 + 3 * np.sin(min(s, 65)))


def room(coverage):
    return {"walls": [{"width": 4.0, "height": 2.6, "coverage": c} for c in coverage], "corners": [{}] * len(coverage)}


def test_a_walk_through_the_room_is_good():
    P, F, t = walk_through_the_room()
    q = sq.scan_quality(P, F, t, room=room([0.8, 0.7, 0.75, 0.6]), registered=150, selected=160)
    assert q["verdict"] == "good" and q["issues"] == [] and q["poses_from"] == "recording"
    m = q["motion"]
    assert m["footprint_m"][0] > 2.5 and m["footprint_m"][1] > 2.0 and m["floor_cells_visited"] > 15
    assert m["pitch_deg"]["lowest"] < -40 and m["pitch_deg"]["highest"] > 20
    assert m["directions_seen_share"] == 1.0 and m["repeated_share"] < 0.2 and m["standing_still_share"] == 0.0
    assert q["walls"]["walls_seen_share"] > 0.7 and q["frames"] == {"selected": 160, "in_map": 150, "share": 0.94}
    assert q["limits"] == sq.LIMITS and q["version"] == 1


def test_turning_on_the_spot_and_standing_is_a_rescan_with_reasons():
    P, F, t = turn_on_the_spot()
    q = sq.scan_quality(P, F, t, room=room([0.6, 0.5, 0.4, 0.3]), registered=153, selected=165)
    codes = [i["code"] for i in q["issues"]]
    assert q["verdict"] == "rescan" and q["summary"].startswith("Scan the room again")
    assert {"one_spot", "never_looked_up", "never_looked_down", "repeated_views"} <= set(codes)
    assert max(q["motion"]["footprint_m"]) < 1.0 and q["motion"]["standing_still_share"] > 0.3
    assert q["motion"]["turned_deg"] > 900
    for i in q["issues"]:                                   # every reason says what was measured, the limit and what to do
        assert i["text"] and i["do"] and i["limit"] == sq.LIMITS[{"one_spot": "footprint_min_m", "never_looked_up": "look_up_deg",
                                                                  "never_looked_down": "look_down_deg",
                                                                  "repeated_views": "repeated_share_max"}[i["code"]]]


def test_one_shortcoming_is_usable_with_limits_and_lost_frames_are_a_rescan():
    P, F, t = walk_through_the_room()
    q = sq.scan_quality(P, F, t, room=room([0.5, 0.2, 0.3, 0.2]), registered=150, selected=160)
    assert q["verdict"] == "limited" and [i["code"] for i in q["issues"]] == ["walls_poorly_seen"]
    assert q["walls"]["least_seen_wall_share"] == 0.2
    q = sq.scan_quality(P, F, t, room=room([0.8, 0.8]), registered=40, selected=100)
    assert q["verdict"] == "rescan" and [i["code"] for i in q["issues"]] == ["frames_lost"]
    q = sq.scan_quality(P, F, t, room=room([0.8, 0.8]), registered=65, selected=100)
    assert q["verdict"] == "limited"                         # a third lost: it works, with a warning


def test_without_poses_the_walls_and_the_frames_are_still_judged():
    q = sq.scan_quality(room=None, registered=10, selected=10)
    assert q["verdict"] == "good" and q["motion"] is None and q["poses_from"] is None and q["walls"]["walls"] == 0


def test_from_the_recording_and_from_the_cameras_of_a_map():
    P, F, t = turn_on_the_spot()

    def T(p, f):                                            # AR camera: looks along its -Z
        z = -f
        x = np.cross([0, 1, 0], z)
        x /= np.linalg.norm(x)
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = np.stack([x, np.cross(z, x), z], 1), p
        return m

    slog = SimpleNamespace(poses=[SimpleNamespace(T_arworld_from_arcam=T(p, f), t_ns=int(s * 1e9) + 5_000_000_000)
                                  for p, f, s in zip(P, F, t)])
    a = sq.from_session_log(slog, registered=153, selected=165)
    b = sq.scan_quality(P, F, t, registered=153, selected=165)
    assert a["motion"] == b["motion"] and a["verdict"] == "rescan" and a["poses_from"] == "recording"

    def cam(p, f):                                          # map image: row 2 of the rotation is the viewing direction
        R = np.eye(3)
        R[2] = f
        return SimpleNamespace(center=p / 2.0, T_cvcam_from_map=np.vstack([np.c_[R, np.zeros(3)], [0, 0, 0, 1]]))

    lmap = SimpleNamespace(scale_m_per_unit=2.0, images=[cam(p, f) for p, f in zip(P[::8], F[::8])])
    c = sq.from_map_images(lmap, (0, 1, 0), registered=100, selected=100)
    assert c["poses_from"] == "map_images" and c["motion"]["poses"] == 100
    assert max(c["motion"]["footprint_m"]) < 1.0 and "one_spot" in [i["code"] for i in c["issues"]]
