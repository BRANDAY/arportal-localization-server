"""Background jobs: building a map from an uploaded recording, recomputing a room plan.

One job runs at a time, in its own **process** (``python -m locserver.jobworker``):
structure from motion keeps a CPU core busy for minutes inside native code, and
running it in the server process would make ``/health`` and the job polling hang
with it. The worker prints one JSON line per progress step on stdout; this
module turns them into the ``Job`` objects of the API.

A map is built in a hidden directory next to the maps and moved into place only
when the build succeeded, so a failed or cancelled build never damages the map
that is already there.
"""

from __future__ import annotations

import json
import logging
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

from .mapstore import utc_now

log = logging.getLogger(__name__)

STAGES = {
    "build_map": [("frames", 0.05), ("features", 0.05), ("matching", 0.35), ("sfm", 0.35), ("depth", 0.08),
                  ("room_plan", 0.10), ("finalize", 0.02)],
    "room_plan": [("depth", 0.40), ("room_plan", 0.50), ("finalize", 0.10)],
}
FINAL_STATES = ("succeeded", "failed", "cancelled")
BUILD_PREFIX = ".build-"


def build_costs(frames: int, pairs: int) -> dict[str, float]:
    """What the stages of a map build cost, in seconds on the reference PC (GTX 1050 Ti), from the amount of work.

    Only the proportions matter: the pace of the machine at hand is measured
    while the build runs. Measured 2026-11-21 on builds of 32 frames (a scan
    of 10 s: 496 pairs, reconstruction 20 s) and of 186 frames (85 s: 2415
    pairs, reconstruction 504 s - it grows faster than the number of frames).
    ``frames`` (reading the recording) is set from its real duration.
    """
    frames, pairs = max(1, frames), max(1, pairs)
    return {"frames": 0.0, "features": 8 + 0.5 * frames, "matching": 0.19 * pairs, "sfm": 0.3 * frames + 0.013 * frames**2,
            "depth": 25.0, "room_plan": 35.0, "finalize": 5.0}


class JobError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


class JobCancelled(Exception):
    pass


class Job:
    def __init__(self, job_id: str, kind: str, map_id: str, params: dict):
        self.job_id, self.kind, self.map_id, self.params = job_id, kind, map_id, params
        self.state = "queued"
        self.stage: str | None = None
        self.stage_progress = 0.0
        self.progress = 0.0
        self.message = "waiting for the previous job" if kind else ""
        self.created = time.time()
        self.created_utc = utc_now()
        self.started: float | None = None
        self.started_utc: str | None = None
        self.finished: float | None = None
        self.finished_utc: str | None = None
        self.error: dict | None = None
        self.result: dict | None = None
        self.cancel_requested = False
        self.proc: subprocess.Popen | None = None
        self.costs: dict[str, float] = dict(STAGES[kind]) if kind in STAGES else {}  # share of each stage; `plan` refines it
        self.planned = False
        self.stage_started: float | None = None
        self.stage_spent: dict[str, float] = {}  # seconds each finished stage took

    # ------------------------------------------------------------------ #
    def report(self, stage: str, fraction: float, message: str = "", now: float | None = None) -> None:
        names = [s for s, _ in STAGES[self.kind]]
        if stage not in names:
            return
        k = names.index(stage)
        now = time.time() if now is None else now
        if self.stage in names and k < names.index(self.stage):
            return  # a late message of an earlier stage
        if stage != self.stage:
            if self.stage in names and self.stage_started is not None:
                self.stage_spent[self.stage] = now - self.stage_started
            self.stage_started = now
        fraction = float(min(1.0, max(0.0, fraction)))
        self.stage, self.stage_progress = stage, round(fraction, 4)
        total = sum(self.costs.values())
        overall = (sum(self.costs[s] for s in names[:k]) + self.costs[stage] * fraction) / total if total > 0 else 0.0
        self.progress = round(max(self.progress, min(0.999, overall)), 4)  # 1.0 is reserved for "succeeded"
        if message:
            self.message = message

    def plan(self, counts: dict, now: float | None = None) -> None:
        """The worker says how much work the build is (frames, image pairs): the shares of the stages follow from it.

        A scan of ten seconds and one of ninety spend their time very
        differently (the fixed steps dominate the first, matching and the
        reconstruction the second), so fixed shares make the bar and the
        estimate of the time left wrong for one of them.
        """
        if self.kind != "build_map":
            return
        try:
            costs = build_costs(int(counts["frames"]), int(counts["pairs"]))
        except (KeyError, TypeError, ValueError):
            return
        now = time.time() if now is None else now
        spent = self.stage_spent.get("frames")
        if spent is None and self.stage == "frames" and self.stage_started is not None:
            spent = now - self.stage_started
        # reading the recording is over when the plan arrives: it counts with what it really took
        costs["frames"] = max(1.0, spent or 0.0)
        self.costs, self.planned = costs, True

    def eta_s(self, now: float | None = None) -> int | None:
        """Seconds left: the rest of the running stage at the pace it shows, the later stages at the pace of the build so far."""
        names = [s for s, _ in STAGES[self.kind]]
        if self.state != "running" or self.stage not in names or self.started is None or self.stage_started is None:
            return None
        now = time.time() if now is None else now
        k, f, t_cur = names.index(self.stage), self.stage_progress, max(0.0, now - self.stage_started)
        # pace: real seconds per unit of cost, from the stages that are over and the part of this one that is done
        paced = [s for s in names[:k] if s in self.stage_spent and not (self.planned and s == "frames")]
        real = sum(self.stage_spent[s] for s in paced) + t_cur
        cost = sum(self.costs[s] for s in paced) + self.costs[self.stage] * f
        if cost <= 0 or real < 3.0 or cost < 0.02 * sum(self.costs.values()):
            return None  # nothing to judge the pace by yet
        pace = real / cost
        here = t_cur * (1 - f) / f if f >= 0.1 and t_cur >= 3.0 else self.costs[self.stage] * (1 - f) * pace
        return int(round(here + sum(self.costs[s] for s in names[k + 1:]) * pace))

    def public(self, queue_position: int = 0) -> dict:
        stages = [s for s, _ in STAGES[self.kind]]
        now = time.time()
        elapsed = ((self.finished or now) - self.started) if self.started else 0.0
        eta = self.eta_s(now)
        return {
            "job_id": self.job_id, "kind": self.kind, "map_id": self.map_id, "state": self.state,
            "stage": self.stage, "stage_index": stages.index(self.stage) + 1 if self.stage in stages else None,
            "stage_count": len(stages), "stage_progress": self.stage_progress, "progress": self.progress,
            "message": self.message, "queue_position": queue_position,
            "created_utc": self.created_utc, "started_utc": self.started_utc, "finished_utc": self.finished_utc,
            "elapsed_s": round(elapsed, 1), "eta_s": eta, "error": self.error, "result": self.result,
        }


class JobManager:
    """Queue + one runner thread. ``runner(job) -> result`` can be injected (tests)."""

    def __init__(self, maps_dir: Path, work_dir: Path, on_ready: Callable[[str], None] | None = None,
                 device: str = "auto", runner: Callable[[Job], dict] | None = None, keep: int = 20):
        self.maps_dir, self.root = Path(maps_dir), Path(work_dir) / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.on_ready = on_ready or (lambda map_id: None)
        self.device, self.keep = device, keep
        self.runner = runner or self._run_worker
        self._lock = threading.RLock()
        self._io_lock = threading.Lock()
        self._save_seq = self._saved_seq = 0
        self._jobs: dict[str, Job] = {}
        self._old: list[dict] = []  # finished jobs of earlier server runs (already public dicts)
        self._queue: queue.Queue[Job] = queue.Queue()
        self._load()
        self._thread = threading.Thread(target=self._loop, name="locserver-jobs", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------ storage --- #
    @property
    def _index(self) -> Path:
        return self.root / "jobs.json"

    def _load(self) -> None:
        try:
            old = json.loads(self._index.read_text(encoding="utf-8")) if self._index.exists() else []
        except (OSError, ValueError):
            old = []
        for j in old:
            if j.get("state") not in FINAL_STATES:
                j.update(state="failed", finished_utc=utc_now(), eta_s=None,
                         error={"code": "server_restarted", "message": "the server stopped while this job was queued or running"})
        self._old = old[: self.keep]
        if self.maps_dir.exists():  # leftovers of builds that never finished
            for d in self.maps_dir.glob(BUILD_PREFIX + "*"):
                shutil.rmtree(d, ignore_errors=True)
        self._save()

    def _save(self) -> None:
        """Write the job index. The state lock is NOT held during the disk write: request handlers
        (`/health`, `/jobs/...`) wait on that lock, and a slow disk must never stall them."""
        with self._lock:
            mine = [j.public() for j in sorted(self._jobs.values(), key=lambda j: -j.created)]
            data = (mine + self._old)[: self.keep]
            self._save_seq += 1
            seq = self._save_seq
        with self._io_lock:
            if seq < self._saved_seq:
                return  # a newer snapshot is already on disk
            tmp = self._index.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
            tmp.replace(self._index)
            self._saved_seq = seq

    # ---------------------------------------------------------------- API --- #
    def new_job_dir(self) -> tuple[str, Path]:
        job_id = uuid.uuid4().hex[:12]
        d = self.root / job_id
        d.mkdir(parents=True)
        return job_id, d

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def active_for(self, map_id: str) -> Job | None:
        with self._lock:
            return next((j for j in self._jobs.values() if j.map_id == map_id and j.state not in FINAL_STATES), None)

    def submit(self, job_id: str, kind: str, map_id: str, params: dict) -> Job:
        job = Job(job_id, kind, map_id, params)
        with self._lock:
            self._jobs[job_id] = job
        self._save()
        self._queue.put(job)
        return job

    def _position(self, job: Job) -> int:
        if job.state != "queued":
            return 0
        with self._lock:
            ahead = [j for j in self._jobs.values() if j.state in ("queued", "running") and j.created < job.created]
        return len(ahead)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            return job.public(self._position(job))
        return next((j for j in self._old if j.get("job_id") == job_id), None)

    def list(self, map_id: str | None = None) -> list[dict]:
        with self._lock:
            mine = [j.public(self._position(j)) for j in sorted(self._jobs.values(), key=lambda j: -j.created)]
        return [j for j in (mine + self._old)[: self.keep] if map_id is None or j.get("map_id") == map_id]

    def counts(self) -> dict:
        with self._lock:
            states = [j.state for j in self._jobs.values()]
        return {"running": states.count("running"), "queued": states.count("queued")}

    def cancel(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return self.get(job_id)
            if job.state not in FINAL_STATES:
                job.cancel_requested = True
                if job.state == "queued":
                    self._finish(job, "cancelled", save=False)
                elif job.proc is not None and job.proc.poll() is None:
                    job.proc.terminate()
        self._save()
        return self.get(job_id)

    def wait(self, job_id: str, timeout: float = 30.0) -> dict | None:
        """Block until the job is final (tests and scripts)."""
        end = time.time() + timeout
        while time.time() < end:
            j = self.get(job_id)
            if j is None or j["state"] in FINAL_STATES:
                return j
            time.sleep(0.02)
        return self.get(job_id)

    # ------------------------------------------------------------- runner --- #
    def build_dir(self, job: Job) -> Path:
        return self.maps_dir / f"{BUILD_PREFIX}{job.job_id}"

    def _finish(self, job: Job, state: str, error: dict | None = None, result: dict | None = None, save: bool = True) -> None:
        job.state, job.error, job.result = state, error, result
        job.finished, job.finished_utc = time.time(), utc_now()
        if state == "succeeded":
            job.progress, job.stage_progress = 1.0, 1.0
            job.message = "done"
        elif state == "cancelled":
            job.message = "cancelled"
        else:
            job.message = (error or {}).get("message", "failed")
        if save:
            self._save()

    def _loop(self) -> None:
        while True:
            job = self._queue.get()
            if job.state != "queued":
                continue
            job.state, job.started, job.started_utc = "running", time.time(), utc_now()
            job.report(STAGES[job.kind][0][0], 0.0, "starting")
            self._save()
            outcome: tuple = ("failed", {"code": "internal", "message": "unknown"}, None)
            try:
                result = self.runner(job)
                if job.cancel_requested:
                    raise JobCancelled()
                job.report("finalize", 0.5, "loading the map")
                if job.kind == "build_map":
                    self._install(job)
                self.on_ready(job.map_id)
                outcome = ("succeeded", None, result)
            except JobCancelled:
                outcome = ("cancelled", None, None)
            except JobError as e:
                outcome = ("cancelled" if job.cancel_requested else "failed", {"code": e.code, "message": e.message}, None)
            except Exception as e:  # a bug in the manager must not kill the queue
                log.exception("job %s crashed", job.job_id)
                outcome = ("failed", {"code": "internal", "message": f"{type(e).__name__}: {e}"}, None)
            # tidy up first: a client that sees the final state must not race with the clean-up
            shutil.rmtree(self.build_dir(job), ignore_errors=True)
            try:
                self._cleanup_inputs(job)
            except OSError:
                log.warning("could not clean up old job folders", exc_info=True)
            self._finish(job, *outcome)

    def _install(self, job: Job) -> None:
        """Move the finished build into place (replacing the previous map of that id)."""
        new, final = self.build_dir(job), self.maps_dir / job.map_id
        if not (new / "map.json").exists():
            raise JobError("internal", "the worker reported success but wrote no map")
        old = self.maps_dir / f"{BUILD_PREFIX}old-{job.job_id}"
        try:  # what the staff typed in for this room outlives a new scan (the door does not: its frame is gone)
            before = json.loads((final / "map.json").read_text(encoding="utf-8-sig")) if (final / "map.json").exists() else {}
            keep = {k: before[k] for k in ("vps", "label", "wifi") if before.get(k) is not None}
            if (keep.get("vps") or {}).get("bridge"):  # measured against the old frame of the room, like the door
                keep["vps"] = {**keep["vps"], "bridge": None, "bridge_dropped": {
                    "reason": "room_rescanned", "utc": utc_now(), "frame_id": before.get("created_utc")}}
            if keep:
                meta = json.loads((new / "map.json").read_text(encoding="utf-8-sig"))
                meta.update({k: v for k, v in keep.items() if meta.get(k) is None})
                (new / "map.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except (OSError, ValueError):
            log.warning("could not carry the room's name over to the new map", exc_info=True)
        for attempt in range(5):
            try:
                if final.exists():
                    final.rename(old)
                new.rename(final)
                break
            except OSError:
                if attempt == 4:
                    if old.exists() and not final.exists():
                        old.rename(final)  # put the previous map back
                    raise
                time.sleep(0.5)
        shutil.rmtree(old, ignore_errors=True)

    def _cleanup_inputs(self, job: Job) -> None:
        """Keep the uploaded recording of the newest job of each map (for a rebuild or a bug report)."""
        if getattr(self, "keep_inputs", False):  # the server was asked to keep everything the phones send
            return
        with self._lock:
            older = [j for j in self._jobs.values() if j is not job and j.map_id == job.map_id and j.state in FINAL_STATES]
            known = {j.job_id for j in self._jobs.values()} | {j.get("job_id") for j in self._old}
        for j in older:
            shutil.rmtree(self.job_dir(j.job_id), ignore_errors=True)
        for d in self.root.iterdir():
            if d.is_dir() and d.name not in known:
                shutil.rmtree(d, ignore_errors=True)

    def _run_worker(self, job: Job) -> dict:
        d = self.job_dir(job.job_id)
        spec = dict(job.params, kind=job.kind, map_id=job.map_id, job_dir=str(d), device=self.device,
                    out_dir=str(self.build_dir(job)), map_dir=str(self.maps_dir / job.map_id))
        spec_file = d / "spec.json"
        spec_file.write_text(json.dumps(spec, indent=1), encoding="utf-8")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result, error = None, None
        with open(d / "worker.log", "wb") as logf:
            job.proc = subprocess.Popen([sys.executable, "-m", "locserver.jobworker", str(spec_file)], stdout=subprocess.PIPE,
                                        stderr=logf, text=True, encoding="utf-8", errors="replace", creationflags=flags)
            for line in job.proc.stdout:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if "stage" in msg:
                    job.report(msg["stage"], msg.get("p", 0.0), msg.get("msg", ""))
                elif "plan" in msg:
                    job.plan(msg["plan"])
                elif "result" in msg:
                    result = msg["result"]
                elif "error" in msg:
                    error = msg["error"]
            rc = job.proc.wait()
        if job.cancel_requested:
            raise JobCancelled()
        if error:
            raise JobError(error.get("code", "internal"), error.get("message", "failed"))
        if rc != 0 or result is None:
            raise JobError("internal", f"the worker process exited with code {rc} (log: {d / 'worker.log'})")
        return result
