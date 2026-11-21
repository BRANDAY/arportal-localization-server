# The first run with a real phone: what to start, what will most likely go wrong, how to see it

Everything in this server was verified with public test data and simulated
phones. A real room recorded with the phone has **not** been through it yet.
This page is for that first run.

## Start

Double click **`Start AR Portal Server.cmd`** in the server's folder. A window
says "STARTING ...", then "SERVER IS RUNNING" with the address for the phone;
the panel opens by itself. Closing the window stops the server. (By hand, and
why three plain commands fail in an ordinary PowerShell window: README, "Run
the server".)

* Started this way the server keeps every clip, every recording (also a
  refused one), every answer and one line per request - so that a failure can
  be studied without walking through the room again. The folder
  (`<work>\incoming`, or `LOCSERVER_KEEP_INCOMING` in `env.local.ps1`; on the
  development PC `E:\VR-FESB\incoming`) is outside the repository; recordings
  of a real room are private.
* **Nothing else on the graphics card.** A game on the same card makes every
  step two to three times slower and can make the build run out of memory.
  `nvidia-smi` must show only this server.
* **Scan first, then localise.** While a map is being built, `/localize`
  answers slowly (the build uses the same card).
* A scan of 60 - 90 s: see "Build time" in the README for how long the build
  takes on the reference PC.

## Where to look

| Place | What it shows |
|---|---|
| The panel, strip at the top | which step the phone is in (connect, scan, upload, build, room, door, localise) and what it waits for |
| The panel, "Jobs" | the build: step, percent, and at the end its **warnings** - read them |
| The panel, "Localisations" | every `/localize`: good / weak / failed, matches, inliers, milliseconds, and how far the door moved since the previous request of the same AR session |
| The panel, "Device log" | what the app itself reported (tracking state, errors) |
| `E:\VR-FESB\incoming\requests.jsonl` | one line per request: status, duration, the reason of a refusal |
| `E:\VR-FESB\incoming\localizations.csv` | the same list as a table for a report: one row per answered `/localize` |
| `E:\VR-FESB\incoming\<date>\..._localize_...` | the clip, `request.json`, `response.json` of one `/localize` |
| `E:\VR-FESB\work\jobs\<job id>\` | the recording (`video.mp4`, `session.jsonl`), `spec.json`, `worker.log` of a build |

## What will most likely go wrong

Ordered by how likely it is. "Sign" is what you see; "then" is what to do
first.

1. **The recording is refused at upload** (`422`).
   Sign: the phone shows an error right after "upload"; `requests.jsonl` has
   the reason; the header `X-Error-Code` says `recording_too_short` / `too_little_motion` / `bad_poses`;
   the recording is in `incoming\..._recording-refused-...`.
   Then: read the reason - it is written for the user. A scan needs at least a
   few seconds of walking, not turning on the spot.

2. **Frames are turned or mirrored** (the phone held upright, the sensor
   lying).
   Sign: the build finishes with very few images in the map ("only N of M
   images are in the map"), or the room plan has no floor / walls stand at odd
   angles; in the panel's 3D view the camera path lies on its side.
   Then: look at `worker.log` for the rotation that was read from the video
   and from `session.jsonl`; the upload field `display_rotation_cw` overrides
   it. This is the most likely *systematic* error, because only two
   one-second captures of the real recorder have been read so far.

3. **ARCore's poses drift, the map is built against them.**
   The build uses the recorded poses to give the map its metric scale and its
   upright direction, and as a starting guess.
   Sign: warning "the reconstruction could not be aligned to the recorded AR
   poses: the map is not metric", or a warning about a low share of cameras
   that agree with the AR poses; in the job's result `metric: false`.
   Then: the map still localises, but distances are unknown - the door cannot
   be saved in the recording's session (the server says so). Scan again:
   slower, without covering the camera, ending near where it started.

4. **White walls, little texture.**
   Sign: "SfM failed: no model could be initialised", or few images in the
   map; later `/localize` answers `failed` with "too few keypoints" or few
   inliers (under ~30) whenever the phone looks at a bare wall.
   Then: include furniture, posters, door frames, the floor edge in the scan
   and when localising; a bare wall cannot be recognised by any method that
   looks at the picture.

5. **Blurred frames** (walking fast, dim light, long exposure).
   Sign: in `worker.log` the line "selected N frames" shows a high
   `rejected_blur` and a low `median_sharpness`; `/localize` gives `weak`.
   Then: more light, slower movement. The server already drops the blurriest
   frames of every window.

6. **The camera's focal length is not what the header says** (or the
   recorder writes it for another resolution).
   Sign: the map builds, `/localize` finds many matches but few inliers, or
   the alignment's `position_rms_m` stays above ~10 cm.
   Then: compare `intrinsics` in `request.json` with the header of the
   recording's `session.jsonl`; `intrinsics_mode=map` makes `/localize` use
   the camera the map was built with.

7. **The door is saved a few centimetres to decimetres off** although
   everything "worked".
   Sign: the staff's door is right in the session that scanned, and shifted
   for a visitor.
   Reason: either ARCore moved its world between the end of the scan and the
   moment the door was saved (the door is saved in that world without a new
   look), or the visitor's alignment is off (`alignment.position_rms_m`,
   number of observations).
   Then: let the visitor look around for a few more clips (the alignment
   improves up to about six frames); place the door right after the build.

8. **The build takes too long or runs out of memory.**
   Sign: the job stays in `matching` or `sfm` for minutes; "CUDA out of
   memory" in `worker.log`; the panel shows a stall.
   Then: close everything else that uses the graphics card; see "Build time"
   in the README for what to expect.

9. **The phone cannot reach the server.**
   Sign: nothing appears in the panel at all.
   Then: same Wi-Fi, the firewall question of Windows answered with "allow"
   for private networks, the address shown at the top of the panel typed into
   the app (README, "Connect a phone").

## After a failure

Do not repeat the walk. Send the folder of the failed request (or the job
folder of the failed build) - everything needed to reproduce it on the PC is
in there:

```powershell
uv run locserver-build-map E:\VR-FESB\work\jobs\<job id> --map-id test   # build again from the same recording
```
