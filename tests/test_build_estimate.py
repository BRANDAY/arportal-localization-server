"""The bar and the estimate of the time left of a map build follow the amount of work, not fixed shares."""

from __future__ import annotations

import pytest

from locserver.jobs import STAGES, Job, build_costs


def running(kind="build_map"):
    job = Job("j1", kind, "room", {})
    job.state, job.started = "running", 0.0
    job.report(STAGES[kind][0][0], 0.0, "starting", now=0.0)
    return job


def test_the_shares_of_the_stages_follow_the_size_of_the_scan():
    small, big = build_costs(32, 496), build_costs(186, 2415)
    share = lambda c, s: c[s] / sum(c.values())  # noqa: E731
    # a short scan spends its time in the fixed steps, a long one in matching and the reconstruction
    assert share(small, "depth") + share(small, "room_plan") > 0.25 > share(big, "depth") + share(big, "room_plan")
    assert share(big, "matching") + share(big, "sfm") > 0.8
    assert big["sfm"] / small["sfm"] > 186 / 32 * 2          # the reconstruction grows faster than the number of frames
    assert big["sfm"] == pytest.approx(504, rel=0.1) and big["matching"] == pytest.approx(475, rel=0.1)  # as measured


def test_the_estimate_comes_from_the_pace_of_the_build_itself():
    job = running()
    assert job.eta_s(now=2.0) is None                         # nothing to judge by yet
    job.report("frames", 1.0, "186 frames selected", now=60.0)
    job.plan({"frames": 186, "pairs": 2604}, now=60.0)         # the worker's first guess of the pairs
    assert job.costs["frames"] == 60.0 and job.planned
    job.report("features", 0.0, now=60.0)
    job.report("features", 0.5, now=160.0)                     # half of the features in 100 s: this machine is twice as slow
    cost = job.costs
    slow = job.eta_s(now=160.0)
    later = cost["matching"] + cost["sfm"] + cost["depth"] + cost["room_plan"] + cost["finalize"]
    assert slow == pytest.approx(100 + later * (100 / (cost["features"] * 0.5)), rel=0.02)
    job.report("features", 1.0, now=260.0)
    job.plan({"frames": 186, "pairs": 2415}, now=260.0)        # now the pairs are known exactly
    job.report("matching", 0.0, now=260.0)
    job.report("matching", 0.5, now=490.0)
    half = job.eta_s(now=490.0)
    # the rest of matching at the pace it shows (230 s more), the later stages at the pace of features + matching so far
    pace = (200.0 + 230.0) / (job.costs["features"] + job.costs["matching"] * 0.5)
    rest = job.costs["sfm"] + job.costs["depth"] + job.costs["room_plan"] + job.costs["finalize"]
    assert half == pytest.approx(230 + rest * pace, rel=0.02)
    assert 0.25 < job.progress < 0.45 and job.public()["eta_s"] is not None
    job.state = "succeeded"
    assert job.eta_s(now=2000.0) is None


def test_the_bar_never_goes_back_and_a_job_without_a_plan_keeps_its_fixed_shares():
    job = running()
    job.report("frames", 1.0, now=5.0)
    before = job.progress                                      # 5 %: the fixed share of reading the recording
    job.plan({"frames": 186, "pairs": 2415}, now=5.0)          # 5 s of some 1100: far less than 5 %
    job.report("features", 0.1, now=10.0)
    assert job.progress == before                              # the bar waits instead of jumping back
    job.report("matching", 0.0, now=100.0)
    job.report("matching", 0.5, now=300.0)
    assert job.progress > before
    job.report("frames", 0.2, now=301.0)                       # a late line of an earlier stage changes nothing
    assert job.stage == "matching" and job.stage_progress == 0.5
    job.plan({"frames": "many"})                               # nonsense from the worker is ignored
    assert job.planned and job.costs["matching"] == pytest.approx(0.19 * 2415)

    plan_only = running("room_plan")
    plan_only.plan({"frames": 10, "pairs": 45})                # only map builds are planned
    assert not plan_only.planned and plan_only.costs == dict(STAGES["room_plan"])
    plan_only.report("depth", 0.0, now=0.0)
    plan_only.report("depth", 0.5, now=20.0)
    assert plan_only.progress == pytest.approx(0.2) and plan_only.eta_s(now=20.0) == pytest.approx(20 + 0.6 / 0.2 * 20, abs=1)
