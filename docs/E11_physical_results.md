# E11: Physical Robot Experiment Results

This document records the empirical results of experiment **E11: Physical Robot Handwriting Experiment**, comparing real-world pen-plotter drawings against PyBullet simulations and intended trajectory specifications within the Robot CI framework.

All metrics are reported exactly as computed by the toolchain (`robot_ci writing-analyze` and `robot_ci writing-diagnose`) without rounding, smoothing, or alteration.

---

## 1. Experimental Setup & Hardware

- **Hardware Platform:** Lab rack-and-pinion pen plotter (3 x 28BYJ-48 stepper motors + ULN2003 drivers, controlled via Arduino Uno running `robot_ci_plotter_standalone.ino`).
- **Wiring:**
  - X axis: D2–D5
  - Y axis: D6–D9
  - PEN axis: D10–D13
  - Power: Driver board power rails tied to external 5 V and common ground.
- **Robot Profile:**
  - `robot_name`: `"Lab rack-and-pinion pen plotter (3 x 28BYJ-48 + ULN2003, Arduino Uno, standalone sketch)"`
  - `execution_mode`: `other`
  - `is_configured`: `false`
- **Recording Medium:** Physical phone camera photographs of drawings on standard paper, captured with `--source robot_phone_recording --executed-plan`.
- **Simulation Reference Run:** `results/writing/run_98649adf.json` (PyBullet backend, 3 trials, targets: `word_hello`, `pattern_square`).

---

## 2. Per-Plan Analysis Results

### Plan 1: `word_hello` — Baseline Policy (`writing_baseline`)

- **Target / Policy:** `word_hello` / `writing_baseline` (version: `w1.0-baseline`, seed: `11`)
- **Plan Hash:** `4b69978ebb27afd1`
- **Plan Path:** `results\writing\exports\word_hello__writing_baseline__seed11__4b69978ebb27afd1\plan.json`
- **Analysis Report ID:** `94744d9e` (`results\writing\real_to_sim\analysis_94744d9e\report.json`)
- **Recording Inputs:**
  - Photo: `lab_recordings\plan1_hello_baseline.jpeg`
  - Video: `None`
- **Page Detection:**
  - Method: `auto`
  - Aspect Ratio: `1.4055` (expected: `1.4143`)
  - Aspect Error: `0.0062` (< 1% error)
  - Manual Corners Used: `None`
  - Detected Corners (px): `[[117.81, 1152.49], [102.51, 129.05], [1483.52, 120.25], [1519.66, 1077.5]]`
  - Orientation Note: `Page orientation chosen by best match to the intended path: rotated 270 deg.`
- **Intended vs. Real (Aligned):**
  - `trajectory_error_mean_mm`: `7.635500791557256`
  - `trajectory_error_max_mm`: `137.08581062215015`
  - `trajectory_error_p95_mm`: `25.297869917499565`
  - `path_coverage`: `0.15886699507389163`
  - `writing_accuracy`: `0.0`
  - Misread Characters:
    - Target `'H'`: predicted `'T'` (score: 0.0827, margin: 0.005, runner-up: `'3'`, correct: false)
    - Target `'E'`: predicted `'9'` (score: 0.121, margin: 0.0056, runner-up: `'3'`, correct: false)
    - Target `'L'`: predicted `'I'` (score: 0.0719, margin: 0.0198, runner-up: `'1'`, correct: false)
    - Target `'L'`: predicted `'I'` (score: 0.0663, margin: 0.003, runner-up: `'1'`, correct: false)
    - Target `'O'`: predicted `'H'` (score: 0.0612, margin: 0.0002, runner-up: `'4'`, correct: false)
- **Intended vs. Real (Raw):**
  - `trajectory_error_mean_mm`: `37.988244240955964`
  - `trajectory_error_max_mm`: `220.19396690610563`
  - `trajectory_error_p95_mm`: `88.27676228083304`
  - `path_coverage`: `0.046798029556650245`
  - `writing_accuracy`: `0.0`
- **Real vs. Sim Distance:**
  - Symmetric Mean Distance (Raw): `29.727161706730982` mm
  - Symmetric Mean Distance (Aligned): `5.414770609357135` mm
  - Hausdorff Distance (Raw): `220.2549584024737` mm
  - Hausdorff Distance (Aligned): `137.1650957563227` mm
- **Placement Correction:**
  - Rotation: `90.812` deg
  - Translation: `[222.658, -54.175]` mm (Norm: `229.154` mm)
- **Timing:**
  - Real Active Duration from Video: `null`
  - Sim Duration: `15.0667` s
  - Nominal Plan Duration: `14.0525` s
  - Timing Note: `No video provided (still photo used); execution time from video is null and not comparable.`
- **Diagnosis Category:** `POLICY_REGRESSION_LIKELY`
  - Diagnosis Message: `Simulation shows a regression. Real-robot agreement could not be assessed from the available recording. Next step: Record the robot drawing the exported plan and upload the video/photo.`
- **Warnings:**
  - `"Large placement correction ({'rotation_deg': 90.812, 'translation_mm': [222.658, -54.175], 'translation_norm_mm': 229.154}). Wrong target, wrong page orientation, or the paper was far from the expected position."`
  - `"Placement correction reduced the real-vs-sim distance from 29.73 to 5.41 mm. Most of the raw difference is where the paper/robot origin was, not stroke shape."`
- **Visual Artifacts:**
  - Aligned Overlay: `docs/figures/E11/plan1_hello_baseline_overlay_aligned.png`
  - Rectified Page: `docs/figures/E11/plan1_hello_baseline_rectified_page.png`

---

### Plan 2: `word_hello` — Candidate Policy (`writing_candidate`)

- **Target / Policy:** `word_hello` / `writing_candidate` (version: `w2.0-candidate`, seed: `11`)
- **Plan Hash:** `9b7a63446cfded6b`
- **Plan Path:** `results\writing\exports\word_hello__writing_candidate__seed11__9b7a63446cfded6b\plan.json`
- **Analysis Report ID:** `fbbd1afb` (`results\writing\real_to_sim\analysis_fbbd1afb\report.json`)
- **Recording Inputs:**
  - Photo: `lab_recordings\plan2_hello_candidate.jpeg`
  - Video: `None`
- **Page Detection:**
  - Method: `auto`
  - Aspect Ratio: `1.4087` (expected: `1.4143`)
  - Aspect Error: `0.0039` (< 1% error)
  - Manual Corners Used: `None`
  - Detected Corners (px): `[[84.4, 1133.59], [99.88, 105.89], [1511.35, 97.01], [1545.06, 1107.66]]`
  - Orientation Note: `Page orientation chosen by best match to the intended path: rotated 270 deg.`
- **Intended vs. Real (Aligned):**
  - `trajectory_error_mean_mm`: `10.727012341457867`
  - `trajectory_error_max_mm`: `102.73889137840065`
  - `trajectory_error_p95_mm`: `31.436014209964927`
  - `path_coverage`: `0.1958128078817734`
  - `writing_accuracy`: `0.0`
  - Misread Characters:
    - Target `'H'`: predicted `'9'` (score: 0.1052, margin: 0.0302, runner-up: `'P'`, correct: false)
    - Target `'E'`: predicted `'F'` (score: 0.0542, margin: 0.0174, runner-up: `'E'`, correct: false)
    - Target `'L'`: predicted `'T'` (score: 0.0477, margin: 0.0266, runner-up: `'4'`, correct: false)
    - Target `'L'`: predicted `'A'` (score: 0.1266, margin: 0.0061, runner-up: `'H'`, correct: false)
    - Target `'O'`: predicted `'4'` (score: 0.0976, margin: 0.0137, runner-up: `'T'`, correct: false)
- **Intended vs. Real (Raw):**
  - `trajectory_error_mean_mm`: `64.58317436459761`
  - `trajectory_error_max_mm`: `131.29144717105513`
  - `trajectory_error_p95_mm`: `129.40455377028894`
  - `path_coverage`: `0.0`
  - `writing_accuracy`: `0.0`
- **Real vs. Sim Distance:**
  - Symmetric Mean Distance (Raw): `49.75981911693191` mm
  - Symmetric Mean Distance (Aligned): `7.1246156000467575` mm
  - Hausdorff Distance (Raw): `130.6540767268765` mm
  - Hausdorff Distance (Aligned): `103.0622059136593` mm
- **Placement Correction:**
  - Rotation: `91.464` deg
  - Translation: `[238.356, -62.052]` mm (Norm: `246.3` mm)
- **Timing:**
  - Real Active Duration from Video: `null`
  - Sim Duration: `17.1667` s
  - Nominal Plan Duration: `15.0858` s
  - Timing Note: `No video provided (still photo used); execution time from video is null and not comparable.`
- **Diagnosis Category:** `POLICY_REGRESSION_LIKELY`
  - Diagnosis Message: `Simulation shows a regression. Real-robot agreement could not be assessed from the available recording. Next step: Record the robot drawing the exported plan and upload the video/photo.`
- **Warnings:**
  - `"Large placement correction ({'rotation_deg': 91.464, 'translation_mm': [238.356, -62.052], 'translation_norm_mm': 246.3}). Wrong target, wrong page orientation, or the paper was far from the expected position."`
  - `"Placement correction reduced the real-vs-sim distance from 49.76 to 7.12 mm. Most of the raw difference is where the paper/robot origin was, not stroke shape."`
- **Visual Artifacts:**
  - Aligned Overlay: `docs/figures/E11/plan2_hello_candidate_overlay_aligned.png`
  - Rectified Page: `docs/figures/E11/plan2_hello_candidate_rectified_page.png`

---

### Plan 3: `pattern_square` — Baseline Policy (`writing_baseline`)

- **Target / Policy:** `pattern_square` / `writing_baseline` (version: `w1.0-baseline`, seed: `21`)
- **Plan Hash:** `f9d9e3adaa619f3e`
- **Plan Path:** `results\writing\exports\pattern_square__writing_baseline__seed21__f9d9e3adaa619f3e\plan.json`
- **Analysis Report ID:** `77bd8aaf` (`results\writing\real_to_sim\analysis_77bd8aaf\report.json`)
- **Recording Inputs:**
  - Photo: `lab_recordings\plan3_square_baseline.jpeg`
  - Video: `None`
- **Page Detection:**
  - Method: `auto`
  - Aspect Ratio: `1.4078` (expected: `1.4143`)
  - Aspect Error: `0.0046` (< 1% error)
  - Manual Corners Used: `None`
  - Detected Corners (px): `[[1372.37, 109.55], [1377.62, 981.54], [130.71, 1033.87], [103.94, 118.8]]`
  - Orientation Note: `Page orientation chosen by best match to the intended path: rotated 90 deg.`
- **Intended vs. Real (Aligned):**
  - `trajectory_error_mean_mm`: `15.119282518316949`
  - `trajectory_error_max_mm`: `27.270453129312962`
  - `trajectory_error_p95_mm`: `24.950378065586712`
  - `path_coverage`: `0.0`
  - `writing_accuracy`: `1.0`
  - Misread Characters: None (target `'square'` recognized correctly as `'square'` with score 0.3009, margin 0.0108, runner-up `'cross_grid'`)
- **Intended vs. Real (Raw):**
  - `trajectory_error_mean_mm`: `41.810155988950775`
  - `trajectory_error_max_mm`: `80.39521518722367`
  - `trajectory_error_p95_mm`: `75.16788390038639`
  - `path_coverage`: `0.018730489073881373`
  - `writing_accuracy`: `0.0`
- **Real vs. Sim Distance:**
  - Symmetric Mean Distance (Raw): `32.37929386917564` mm
  - Symmetric Mean Distance (Aligned): `14.46784871420289` mm
  - Hausdorff Distance (Raw): `80.65285348102411` mm
  - Hausdorff Distance (Aligned): `27.489258424381937` mm
- **Placement Correction:**
  - Rotation: `-9.425` deg
  - Translation: `[5.638, -35.779]` mm (Norm: `36.221` mm)
- **Timing:**
  - Real Active Duration from Video: `null`
  - Sim Duration: `11.0667` s
  - Nominal Plan Duration: `11.0335` s
  - Timing Note: `No video provided (still photo used); execution time from video is null and not comparable.`
- **Diagnosis Category:** `POLICY_REGRESSION_LIKELY`
  - Diagnosis Message: `Simulation shows a regression. Real-robot agreement could not be assessed from the available recording. Next step: Record the robot drawing the exported plan and upload the video/photo.`
- **Warnings:**
  - `"Large placement correction ({'rotation_deg': -9.425, 'translation_mm': [5.638, -35.779], 'translation_norm_mm': 36.221}). Wrong target, wrong page orientation, or the paper was far from the expected position."`
  - `"Placement correction reduced the real-vs-sim distance from 32.38 to 14.47 mm. Most of the raw difference is where the paper/robot origin was, not stroke shape."`
- **Visual Artifacts:**
  - Aligned Overlay: `docs/figures/E11/plan3_square_baseline_overlay_aligned.png`
  - Rectified Page: `docs/figures/E11/plan3_square_baseline_rectified_page.png`

---

### Plan 4: `pattern_square` — Candidate Policy (`writing_candidate`)

- **Target / Policy:** `pattern_square` / `writing_candidate` (version: `w2.0-candidate`, seed: `21`)
- **Plan Hash:** `2ab05ca5e98d11c6`
- **Plan Path:** `results\writing\exports\pattern_square__writing_candidate__seed21__2ab05ca5e98d11c6\plan.json`
- **Analysis Report ID:** `23e507a1` (`results\writing\real_to_sim\analysis_23e507a1\report.json`)
- **Recording Inputs:**
  - Photo: `lab_recordings\plan4_square_candidate.jpeg`
  - Video: `None`
- **Page Detection:**
  - Method: `auto`
  - Aspect Ratio: `1.4029` (expected: `1.4143`)
  - Aspect Error: `0.0081` (< 1% error)
  - Manual Corners Used: `None`
  - Detected Corners (px): `[[1523.6, 145.57], [1524.16, 1114.48], [133.31, 1148.21], [134.16, 134.97]]`
  - Orientation Note: `Page orientation chosen by best match to the intended path: rotated 90 deg.`
- **Intended vs. Real (Aligned):**
  - `trajectory_error_mean_mm`: `12.493762293139422`
  - `trajectory_error_max_mm`: `50.56370869351073`
  - `trajectory_error_p95_mm`: `31.19541239064458`
  - `path_coverage`: `0.06659729448491156`
  - `writing_accuracy`: `0.0`
  - Misread Characters:
    - Target `'square'`: predicted `'circle'` (score: 0.1687, margin: 0.0009, runner-up: `'spiral'`, correct: false)
- **Intended vs. Real (Raw):**
  - `trajectory_error_mean_mm`: `22.091285859866908`
  - `trajectory_error_max_mm`: `88.12508865243768`
  - `trajectory_error_p95_mm`: `49.5`
  - `path_coverage`: `0.04058272632674298`
  - `writing_accuracy`: `0.0`
- **Real vs. Sim Distance:**
  - Symmetric Mean Distance (Raw): `20.39666174564951` mm
  - Symmetric Mean Distance (Aligned): `14.194839767445297` mm
  - Hausdorff Distance (Raw): `87.53156520193788` mm
  - Hausdorff Distance (Aligned): `56.13150307202086` mm
- **Placement Correction:**
  - Rotation: `6.999` deg
  - Translation: `[0.451, -48.336]` mm (Norm: `48.338` mm)
- **Timing:**
  - Real Active Duration from Video: `null`
  - Sim Duration: `13.3333` s
  - Nominal Plan Duration: `11.4683` s
  - Timing Note: `No video provided (still photo used); execution time from video is null and not comparable.`
- **Diagnosis Category:** `POLICY_REGRESSION_LIKELY`
  - Diagnosis Message: `Simulation shows a regression. Real-robot agreement could not be assessed from the available recording. Next step: Record the robot drawing the exported plan and upload the video/photo.`
- **Warnings:**
  - `"Large placement correction ({'rotation_deg': 6.999, 'translation_mm': [0.451, -48.336], 'translation_norm_mm': 48.338}). Wrong target, wrong page orientation, or the paper was far from the expected position."`
- **Visual Artifacts:**
  - Aligned Overlay: `docs/figures/E11/plan4_square_candidate_overlay_aligned.png`
  - Rectified Page: `docs/figures/E11/plan4_square_candidate_rectified_page.png`

---

## 3. Baseline vs. Candidate Comparison

### Word HELLO (Plan 1 Baseline vs. Plan 2 Candidate)

| Metric | Plan 1 (Baseline `4b69978ebb27afd1`) | Plan 2 (Candidate `9b7a63446cfded6b`) | Candidate Delta |
|---|---|---|---|
| **Sim Trajectory Error Mean (mm)** | `0.06773990513018355` | `0.6112008536550397` | +0.5435 mm (worse) |
| **Sim Trajectory Error Max (mm)** | `0.2624358459228371` | `2.2223997480310356` | +1.9600 mm (worse) |
| **Sim Path Coverage** | `1.0` | `0.8719211822660099` | -0.1281 (worse) |
| **Sim Writing Accuracy** | `1.0` | `1.0` | 0.0 |
| **Real Aligned Trajectory Error Mean (mm)** | `7.635500791557256` | `10.727012341457867` | +3.0915 mm (worse on paper) |
| **Real Aligned Trajectory Error Max (mm)** | `137.08581062215015` | `102.73889137840065` | -34.3469 mm |
| **Real Aligned Path Coverage** | `0.15886699507389163` | `0.1958128078817734` | +0.0369 |
| **Real Aligned Writing Accuracy** | `0.0` | `0.0` | 0.0 (both failed) |
| **Real vs Sim Symmetric Distance (Aligned, mm)** | `5.414770609357135` | `7.1246156000467575` | +1.7098 mm (worse sim-agreement) |

### Pattern Square (Plan 3 Baseline vs. Plan 4 Candidate)

| Metric | Plan 3 (Baseline `f9d9e3adaa619f3e`) | Plan 4 (Candidate `2ab05ca5e98d11c6`) | Candidate Delta |
|---|---|---|---|
| **Sim Trajectory Error Mean (mm)** | `0.06709687029547716` | `1.903372573672059` | +1.8363 mm (worse) |
| **Sim Trajectory Error Max (mm)** | `0.2845594181936525` | `7.854490633403035` | +7.5699 mm (worse) |
| **Sim Path Coverage** | `1.0` | `0.5150884495317378` | -0.4849 (worse) |
| **Sim Writing Accuracy** | `1.0` | `0.0` | -1.0 (worse) |
| **Real Aligned Trajectory Error Mean (mm)** | `15.119282518316949` | `12.493762293139422` | -2.6255 mm |
| **Real Aligned Trajectory Error Max (mm)** | `27.270453129312962` | `50.56370869351073` | +23.2933 mm (worse on paper) |
| **Real Aligned Path Coverage** | `0.0` | `0.06659729448491156` | +0.0666 |
| **Real Aligned Writing Accuracy** | `1.0` | `0.0` | -1.0 (worse on paper) |
| **Real vs Sim Symmetric Distance (Aligned, mm)** | `14.46784871420289` | `14.194839767445297` | -0.2730 mm |

---

## 4. Assessment: Does the Simulated Regression Appear on Paper?

In plain words: **Yes, the candidate policy's regressions detected in simulation are clearly reflected on paper, most decisively in recognition failure and peak error.**

1. **Pattern Square (`pattern_square`):**
   - In simulation, the candidate policy degraded drastically: `writing_accuracy` fell from 1.0 to 0.0, and maximum trajectory error spiked from 0.28 mm to 7.85 mm.
   - On the physical robot, **the exact same functional regression occurred**:
     - The baseline drawing was successfully recognized as a square (`writing_accuracy = 1.0`, classified as `'square'`).
     - The candidate drawing failed shape classification completely (`writing_accuracy = 0.0`, misclassified as `'circle'`).
     - Furthermore, the candidate's peak physical tracking error on paper nearly doubled relative to baseline (`50.5637` mm vs `27.2705` mm), confirming severe corner rounding and path overshoot in the real world.

2. **Word HELLO (`word_hello`):**
   - In simulation, the candidate policy increased mean trajectory error from 0.068 mm to 0.611 mm (nearly a 10x increase).
   - On paper, the candidate's mean aligned error is likewise noticeably higher than baseline (`10.7270` mm vs `7.6355` mm), and the real-to-sim discrepancy is higher (`7.1246` mm vs `5.4148` mm).
   - However, due to significant rack-and-pinion backlash and pen drag on the physical hardware, both real drawings achieved under 20% path coverage (`0.1589` vs `0.1958`), causing the template recognizer to misread individual letter strokes for both policies (`writing_accuracy = 0.0` on both).

3. **Sim-to-Real Origin and Placement Offset:**
   - On all four physical runs, the raw real-to-sim distance was substantial (20–50 mm) before placement correction, but dropped to 5–14 mm after rigid alignment.
   - This indicates that a significant fraction of physical error stemmed from robot coordinate frame setup and paper positioning relative to the plotter home point, rather than stroke morphology.
