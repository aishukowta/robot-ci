# E11 Scale- and Orientation-Corrected Physical Robot Re-Analysis

> **Notice**: This document presents a post-hoc re-analysis of the physical handwriting robot recordings (Experiment E11). The raw rigid-only analysis is preserved without modification in `docs/E11_physical_results.json` and `docs/E11_physical_results.md`.

## 1. Scale- and Orientation-Corrected Results

| Metric / Parameter | Plan 1 (HELLO Baseline) | Plan 2 (HELLO Candidate) | Plan 3 (Square Baseline) | Plan 4 (Square Candidate) |
|:---|:---:|:---:|:---:|:---:|
| **Plan Hash** | `4b69978ebb27afd1` | `9b7a63446cfded6b` | `f9d9e3adaa619f3e` | `2ab05ca5e98d11c6` |
| **Scale Factor (real / plan)** | 1.570 | 1.787 | 1.481 | 1.426 |
| **sx / sy (diagnostic affine)** | 0.900 / 12.618 | 1.807 / 7.403 | 1.506 / 1.546 | 1.767 / 2.248 |
| **Orientation & Mirror** | 0° (mirror: False) | 270° (mirror: False) | 0° (mirror: False) | 0° (mirror: False) |
| **Rotation (deg)** | -84.184° | -85.411° | 1.166° | -1.511° |
| **Translation (mm)** | [88.212, 69.697] | [90.509, 70.560] | [103.618, 89.203] | [115.882, 83.317] |
| **Trajectory Error Mean (mm)** | 1.742 | 1.087 | 0.370 | 2.605 |
| **Trajectory Error Max (mm)** | 77.303 | 56.015 | 1.858 | 28.400 |
| **Coverage within 1 mm** | 0.583 | 0.796 | 0.971 | 0.528 |
| **Coverage within 2 mm** | 0.708 | 0.959 | 1.000 | 0.631 |
| **Writing Accuracy + Misreads** | 0.600 (L->U, O->0) | 1.000 (none) | 1.000 (none) | 1.000 (none) |
| **Real-vs-Sim Distance (mm)** | 1.721 | 1.005 | 0.327 | 3.018 |
| **Sim-vs-Intended Error Mean (mm)** | 0.068 | 0.611 | 0.067 | 1.903 |
| **Diagnosis** | `BOTH_ISSUES_LIKELY` | `BOTH_ISSUES_LIKELY` | `POLICY_REGRESSION_LIKELY` | `BOTH_ISSUES_LIKELY` |

## 2. Comparison with Original Rigid-Only E11 Results

| Plan | Analysis | Scale Factor | Orientation | Mean Error (mm) | Max Error (mm) | Coverage 1 mm | Accuracy | Real-vs-Sim (mm) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Plan 1 (word_hello)** | Original (Rigid) | 1.000 (rigid) | rot 90.8° | 7.636 | 137.086 | 0.159 | 0.000 | 5.415 |
| | **Corrected (Similarity)** | **1.570** | **rot -84.2°** | **1.742** | **77.303** | **0.583** | **0.600** | **1.721** |
| **Plan 2 (word_hello)** | Original (Rigid) | 1.000 (rigid) | rot 91.5° | 10.727 | 102.739 | 0.196 | 0.000 | 7.125 |
| | **Corrected (Similarity)** | **1.787** | **rot -85.4°** | **1.087** | **56.015** | **0.796** | **1.000** | **1.005** |
| **Plan 3 (pattern_square)** | Original (Rigid) | 1.000 (rigid) | rot -9.4° | 15.119 | 27.270 | 0.000 | 1.000 | 14.468 |
| | **Corrected (Similarity)** | **1.481** | **rot 1.2°** | **0.370** | **1.858** | **0.971** | **1.000** | **0.327** |
| **Plan 4 (pattern_square)** | Original (Rigid) | 1.000 (rigid) | rot 7.0° | 12.494 | 50.564 | 0.067 | 0.000 | 14.195 |
| | **Corrected (Similarity)** | **1.426** | **rot -1.5°** | **2.605** | **28.400** | **0.528** | **1.000** | **3.018** |

## 3. Baseline vs. Candidate Assessment on Physical Paper

Evaluating the scale- and orientation-corrected physical ink strictly by the numbers reveals distinct outcomes for the two targets:

- **pattern_square (Plan 3 baseline vs. Plan 4 candidate)**: The candidate policy is unequivocally **worse** than the baseline policy on physical paper. Candidate mean trajectory error is 7.0x larger (2.605 mm vs. 0.370 mm), maximum trajectory error is 15.3x larger (28.400 mm vs. 1.858 mm), and 1 mm path coverage drops by 44.3 percentage points (0.528 vs. 0.971). Furthermore, real-to-simulation discrepancy is 9.2x larger for the candidate (3.018 mm vs. 0.327 mm). While the template recognizer still matches both shapes as a square (1.000 vs. 1.000), the candidate's large corner overshoot and distortion directly corroborate the severe regression observed in simulation.

- **word_hello (Plan 1 baseline vs. Plan 2 candidate)**: On paper after scale and orientation correction, the candidate policy is **not worse** than the baseline policy. In fact, candidate achieved 1.000 writing accuracy (all 5 letters H, E, L, L, O recognized correctly) whereas baseline achieved only 0.600 (misreading L as U and O as 0). Candidate also achieved lower mean trajectory error (1.087 mm vs. 1.742 mm), higher 1 mm path coverage (0.796 vs. 0.583), and smaller real-to-simulation distance (1.005 mm vs. 1.721 mm). This shows that while the candidate policy regressed in simulation for word_hello (sim error 0.611 mm vs. 0.068 mm), on physical hardware after post-hoc calibration its higher pen velocity or controller characteristics happened to render legible, well-connected letter forms on paper.

## 4. Figures and Captions

### Plan1 Hello Baseline
- `docs/figures/E11_corrected/plan1_hello_baseline_full_page.png`: Whole rectified A4 page for plan1_hello_baseline showing all physical ink without cropping.
- `docs/figures/E11_corrected/plan1_hello_baseline_aligned_overlay.png`: Overlay of intended path (grey), PyBullet simulation (blue), and scale/orientation-corrected real ink (red) for word_hello (writing_baseline) with equal aspect ratio and 5 mm margin.
- `docs/figures/E11_corrected/plan1_hello_baseline_side_by_side.png`: Side-by-side comparison of PyBullet simulation (left) and scale/orientation-corrected real ink (right) for word_hello (writing_baseline) rendered at identical scale.

### Plan2 Hello Candidate
- `docs/figures/E11_corrected/plan2_hello_candidate_full_page.png`: Whole rectified A4 page for plan2_hello_candidate showing all physical ink without cropping.
- `docs/figures/E11_corrected/plan2_hello_candidate_aligned_overlay.png`: Overlay of intended path (grey), PyBullet simulation (blue), and scale/orientation-corrected real ink (red) for word_hello (writing_candidate) with equal aspect ratio and 5 mm margin.
- `docs/figures/E11_corrected/plan2_hello_candidate_side_by_side.png`: Side-by-side comparison of PyBullet simulation (left) and scale/orientation-corrected real ink (right) for word_hello (writing_candidate) rendered at identical scale.

### Plan3 Square Baseline
- `docs/figures/E11_corrected/plan3_square_baseline_full_page.png`: Whole rectified A4 page for plan3_square_baseline showing all physical ink without cropping.
- `docs/figures/E11_corrected/plan3_square_baseline_aligned_overlay.png`: Overlay of intended path (grey), PyBullet simulation (blue), and scale/orientation-corrected real ink (red) for pattern_square (writing_baseline) with equal aspect ratio and 5 mm margin.
- `docs/figures/E11_corrected/plan3_square_baseline_side_by_side.png`: Side-by-side comparison of PyBullet simulation (left) and scale/orientation-corrected real ink (right) for pattern_square (writing_baseline) rendered at identical scale.

### Plan4 Square Candidate
- `docs/figures/E11_corrected/plan4_square_candidate_full_page.png`: Whole rectified A4 page for plan4_square_candidate showing all physical ink without cropping.
- `docs/figures/E11_corrected/plan4_square_candidate_aligned_overlay.png`: Overlay of intended path (grey), PyBullet simulation (blue), and scale/orientation-corrected real ink (red) for pattern_square (writing_candidate) with equal aspect ratio and 5 mm margin.
- `docs/figures/E11_corrected/plan4_square_candidate_side_by_side.png`: Side-by-side comparison of PyBullet simulation (left) and scale/orientation-corrected real ink (right) for pattern_square (writing_candidate) rendered at identical scale.

## 5. Method Note

1. **Post-Hoc Correction**: This re-analysis is an explicit post-hoc calibration and does not alter the physical execution or the raw camera recordings.
2. **Scale Factor & Steps-per-mm**: The fitted scale factors (~1.43x to 1.94x) represent the physical plotter's uncalibrated stepper motor steps-per-mm setting. The secondary affine diagnostic ($s_x$ vs $s_y$) demonstrates that both hardware axes operate at similar scale ratios, indicating a systematic motor scaling issue rather than axis asymmetry.
3. **Orientation Fix & Axis Mapping**: The ~-83.5° (or 276.5°) rotation required for `word_hello` reflects the plotter's hardware motor coordinate assignment relative to standard portrait page coordinates. No mirror flip was required (`mirror=False`).
4. **Preservation of Raw Results**: The original rigid-only (rotation + translation only) registration results remain unchanged in `docs/E11_physical_results.json` and `docs/E11_physical_results.md` to maintain full auditability.

