# Lab day: recording the handwriting robot

Goal: get real recordings that Robot CI can compare with simulation. Everything else is already built — after the lab you only upload files.

## The night before (10 minutes, at your laptop)

```powershell
cd "C:\Users\Aishwarya Kowta\Desktop\robot-ci"
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m robot_ci writing-selftest
.venv\Scripts\python.exe -m robot_ci writing-run --backend kinematic --trials 3
```

Export the plans you will draw (start with one simple target; add more if time allows):

```powershell
.venv\Scripts\python.exe -m robot_ci writing-export --target word_hello --policy writing_baseline
.venv\Scripts\python.exe -m robot_ci writing-export --target word_hello --policy writing_candidate
.venv\Scripts\python.exe -m robot_ci writing-export --target pattern_square --policy writing_baseline
.venv\Scripts\python.exe -m robot_ci writing-export --target pattern_square --policy writing_candidate
```

Each export folder (`results/writing/exports/<target>__<policy>__seed<N>__<hash>/`) contains `drawing.svg`, `waypoints.csv`, `plan.json`, `manifest.json` and `LAB_INSTRUCTIONS.txt`. Copy the folders to a USB stick if the lab PC drives the robot.

Bring: plain white A4 paper, masking tape, a dark non-glossy sheet/board to put under the paper, phone + stand/tripod, the same pen for every run.

## At the lab (find out first — 5 minutes)

1. **Robot model** and **how it is driven**: does its software import **SVG** (most handwriting robots), accept **G-code** (GRBL-type sender), or only type text in its own font?
   * SVG import → use `drawing.svg`. Geometry is comparable; timing is not (the app plans speeds).
   * G-code → fill in `config/writing_robot_profile.yaml` (coordinate mapping, pen up/down commands, `is_configured: true`, `execution_mode: gcode`) and re-export to get `drawing.gcode`. Do a dry run with the pen lifted first.
   * Only built-in text/font → you can still record it, but it is **not** the Robot CI policy: upload as `robot_phone_recording` **without** "executed plan" (cross-controller comparison).
2. Where is the robot's origin relative to the paper? Write it in `config/writing_robot_profile.yaml` (`machine_origin`, `offset_mm`). A small placement offset is fine — the analysis measures and removes it, and reports it separately.
3. Ask the lab supervisor to confirm normal safe operation of the robot (its own E-stop, speeds). Robot CI never drives the robot directly.

## Recording each run

1. Fresh sheet, taped flat on the dark board. All four paper corners visible.
2. Phone **above the page** (as straight down as you can), whole page in view, not moving. Lock focus/exposure if possible.
3. Start video → start the robot → wait until it finishes and moves away → **keep filming 2 seconds** → stop.
4. Take one **sharp photo** of the finished page from straight above (all 4 corners, no shadow over the writing).
5. Write on the back of the sheet / in your notes: target, policy, run number. Rename files, e.g. `hello_baseline_run1.mp4`, `hello_baseline_run1.jpg`.

Recommended set (≈30–40 min): `word_hello` baseline ×3, candidate ×3; then `pattern_square` baseline ×2, candidate ×2. Repeats give real **repeatability**.

Optional: a small coloured sticker (red/green) on the pen holder gives an approximate motion track from the video (`--marker-color red`).

## After the lab: upload

**Dashboard:** `python -m robot_ci dashboard` → Section D → *Upload recording* → choose the exported plan → upload video and/or photo → "Real robot drawing" → tick "The robot drew THIS exported file" only if it did → choose a regression run → **Analyse**.

**Terminal:**

```powershell
.venv\Scripts\python.exe -m robot_ci writing-analyze `
  --plan results/writing/exports/word_hello__writing_baseline__seed11__4b69978ebb27afd1/plan.json `
  --video recordings/hello_baseline_run1.mp4 --photo recordings/hello_baseline_run1.jpg `
  --source robot_phone_recording --executed-plan `
  --run results/writing/run_<id>.json
```

If the page corners are not found automatically, open the photo in Paint, read the pixel coordinates of the four corners and pass
`--corners "x1,y1;x2,y2;x3,y3;x4,y4"` (page top-left, top-right, bottom-right, bottom-left).

If the ink is coloured and the background is busy, try `--ink blue` (or black/red).

## What you can claim afterwards

* With matched recordings: "The real robot executing the exported baseline plan deviates from the simulation of the same plan by X mm (after removing a Y mm placement offset); recognition Z%; repeatability across 3 runs R mm."
* Whether the candidate's simulated regression is reproduced on paper.
* Not claimable from video: pen force, joint angles, motor stalls.
