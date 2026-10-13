"""Evidence-based diagnosis for the handwriting robot.

Combines (a) a simulated baseline-vs-candidate writing regression run and
(b) a real-to-sim analysis of a recording, into the same five categories used
by the rest of Robot CI. Rules are explicit and conservative:

* Only a declared robot recording (data_source = robot_phone_recording) is
  physical evidence.
* Sim-to-real conclusions about a POLICY need the robot to have drawn the
  exported plan (--executed-plan). Otherwise it is a cross-controller check.
* Poor page detection or too little matching ink -> insufficient evidence.
* Paper-placement offset is reported separately from stroke-shape mismatch.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from robot_ci.diagnosis.models import DiagnosisCategory, DiagnosisReport, EvidenceItem, DIAGNOSIS_LIMITATIONS_DISCLAIMER

DEFAULT_SHAPE_MISMATCH_MM = 0.75     # aligned real-vs-sim symmetric distance
DEFAULT_ACCURACY_DROP = 0.20         # real recognition rate below sim by more than this
MIN_COVERAGE_FOR_EVIDENCE = 0.50     # share of the intended path with real ink within 1 mm (after alignment)
MAX_ASPECT_ERROR = 0.12


def diagnose_writing(run: Optional[Dict[str, Any]] = None, analysis: Optional[Dict[str, Any]] = None,
                     shape_mismatch_mm: float = DEFAULT_SHAPE_MISMATCH_MM,
                     accuracy_drop: float = DEFAULT_ACCURACY_DROP) -> DiagnosisReport:
    evidence: List[EvidenceItem] = []
    supporting: List[str] = []
    conflicting: List[str] = []
    missing: List[str] = []
    quality: List[str] = []
    unsupported: List[str] = [
        "Pen pressure, contact force, joint limits and motor stalls cannot be measured from a photo or video.",
        "Causation is not proven by trajectory differences alone.",
    ]

    # ---------------- simulation regression evidence
    policy_regression: Optional[bool] = None
    if not run:
        missing.append("Simulated baseline-vs-candidate writing regression run is missing.")
    else:
        regs = [f for f in run.get("findings", []) if f.get("regression_detected")]
        evidence.append(EvidenceItem(
            source_type="SIMULATION_WRITING_REGRESSION", identifier=run.get("run_id", "?"),
            provenance=f"{str(run.get('backend', 'unknown')).upper()}_SIMULATION", is_physical_hardware=False,
            is_valid=True,
            summary=f"{run.get('baseline', {}).get('name')} vs {run.get('candidate', {}).get('name')}, "
                    f"{run.get('trials')} trial(s) x {len(run.get('targets', []))} target(s)",
            details={"overall_status": run.get("overall_status")},
        ))
        policy_regression = bool(regs)
        if regs:
            for f in regs:
                supporting.append(f"Simulation: {f['label']} regressed ({f['severity']}): {f['reason']}")
        else:
            supporting.append("Simulation: no handwriting metric crossed its regression threshold.")

    # ---------------- real recording evidence
    sim_to_real: Optional[bool] = None
    if not analysis:
        missing.append("Real-to-sim analysis of a robot recording is missing.")
    else:
        src = analysis.get("data_source")
        ctype = analysis.get("comparison_type")
        m = analysis.get("metrics", {})
        det = analysis.get("page_detection", {})
        aligned = m.get("intended_vs_real_aligned", {})
        rvs = m.get("real_vs_sim_aligned", {})
        rvs_raw = m.get("real_vs_sim_raw", {})
        sim_m = m.get("intended_vs_sim", {})
        plan = analysis.get("plan", {})
        evidence.append(EvidenceItem(
            source_type="WRITING_RECORDING_ANALYSIS", identifier=analysis.get("report_id", "?"),
            provenance=str(src).upper(), is_physical_hardware=(src == "robot_phone_recording"), is_valid=True,
            summary=f"plan {plan.get('plan_hash')} ({plan.get('policy_name')}, {plan.get('target_id')}), {ctype}",
            details={"page_aspect_error": det.get("aspect_error"), "coverage_aligned": aligned.get("path_coverage"),
                     "real_vs_sim_aligned_mm": rvs.get("symmetric_mean_mm"),
                     "placement_correction": analysis.get("placement_correction")},
        ))
        if run:
            names = {run.get("baseline", {}).get("name"), run.get("candidate", {}).get("name")}
            if plan.get("policy_name") not in names:
                quality.append(f"The recording executed '{plan.get('policy_name')}', which is not part of the "
                               f"simulation run ({sorted(n for n in names if n)}).")
            if plan.get("target_id") not in run.get("targets", []):
                quality.append(f"Target '{plan.get('target_id')}' was not part of the simulation run.")
        if src != "robot_phone_recording":
            label = {"synthetic_fixture": "a synthetic test recording",
                     "handheld_marker_recording": "a hand-made recording (no robot)",
                     "controlled_sim_perturbation": "a controlled simulation perturbation"}.get(src, f"'{src}'")
            quality.append(f"The recording is {label}; it validates the analysis pipeline, not the robot.")
            unsupported.append("Physical sim-to-real agreement cannot be assessed from non-robot data.")
        elif ctype != "matched_plan_execution":
            quality.append("The robot was not declared to have drawn the exported plan (cross-controller), so "
                           "differences cannot be attributed to the Robot CI policy.")
            unsupported.append("Sim-to-real conclusions about the policy require the robot to execute the exported plan.")
        elif det.get("aspect_error", 1.0) > MAX_ASPECT_ERROR:
            quality.append(f"Page detection unreliable (aspect error {det.get('aspect_error'):.2f}); "
                           "re-take the photo from directly above or supply manual corners.")
            unsupported.append("Measurements are not trusted because the page could not be rectified reliably.")
        elif (aligned.get("path_coverage") or 0.0) < MIN_COVERAGE_FOR_EVIDENCE:
            quality.append(f"Only {100 * (aligned.get('path_coverage') or 0):.0f}% of the intended path has matching "
                           "ink; wrong target, faint ink or a poor photo.")
            unsupported.append("Too little matching ink to compare real and simulated writing.")
        else:
            d = rvs.get("symmetric_mean_mm")
            acc_real = aligned.get("writing_accuracy")
            acc_sim = sim_m.get("writing_accuracy")
            shape_bad = d is not None and d > shape_mismatch_mm
            acc_bad = (acc_real is not None and acc_sim is not None and acc_sim - acc_real > accuracy_drop)
            sim_to_real = bool(shape_bad or acc_bad)
            supporting.append(
                f"Real vs simulated ink after placement correction: {d:.2f} mm mean distance "
                f"(threshold {shape_mismatch_mm} mm); recognition real {acc_real} vs sim {acc_sim}."
            )
            pc = analysis.get("placement_correction", {})
            supporting.append(
                f"Paper placement offset removed before comparison: {pc.get('translation_norm_mm')} mm, "
                f"{pc.get('rotation_deg')} deg (raw real-vs-sim distance {rvs_raw.get('symmetric_mean_mm'):.2f} mm)."
            )
            if not sim_to_real:
                supporting.append("Real handwriting agrees with the simulation of the same plan within tolerance.")

    # ---------------- combine
    if policy_regression is not None and sim_to_real is not None:
        if policy_regression and sim_to_real:
            cat = DiagnosisCategory.BOTH_ISSUES_LIKELY
            expl = "The candidate regressed in simulation AND the real robot's writing differs from the simulation."
            nxt = "Fix the simulated regression first, then investigate the real-robot mismatch (pen holder play, speed, paper friction)."
        elif policy_regression:
            cat = DiagnosisCategory.POLICY_REGRESSION_LIKELY
            expl = "The candidate regressed in simulation, and the real robot reproduces the simulated writing within tolerance."
            nxt = "Treat the regression as a policy problem: inspect the failing metrics and the overlay images."
        elif sim_to_real:
            cat = DiagnosisCategory.SIM_TO_REAL_MISMATCH_LIKELY
            expl = "No regression in simulation, but the real writing differs from the simulation of the same plan."
            nxt = "Improve the simulator or robot setup (calibration, pen holder, speeds) before trusting simulated results."
        else:
            cat = DiagnosisCategory.NO_REGRESSION_DETECTED
            expl = "No regression in simulation and the real writing matches the simulation within tolerance."
            nxt = "Repeat with more targets and trials to strengthen the evidence."
    elif policy_regression is not None:
        cat = DiagnosisCategory.POLICY_REGRESSION_LIKELY if policy_regression else DiagnosisCategory.NO_REGRESSION_DETECTED
        expl = ("Simulation shows a regression. " if policy_regression else "Simulation shows no regression. ") + \
               "Real-robot agreement could not be assessed from the available recording."
        nxt = "Record the robot drawing the exported plan and upload the video/photo."
    elif sim_to_real is not None:
        cat = DiagnosisCategory.SIM_TO_REAL_MISMATCH_LIKELY if sim_to_real else DiagnosisCategory.NO_REGRESSION_DETECTED
        expl = ("Real writing differs from simulation. " if sim_to_real else "Real writing matches simulation. ") + \
               "No baseline-vs-candidate simulation run was provided."
        nxt = "Run: python -m robot_ci writing-run"
    else:
        cat = DiagnosisCategory.INCONCLUSIVE_INSUFFICIENT_EVIDENCE
        expl = "Not enough trustworthy evidence for a diagnosis."
        nxt = "Run the simulated regression and upload a robot recording of an exported plan."

    return DiagnosisReport(
        diagnosis_id=uuid.uuid4().hex[:8], category=cat, summary_explanation=expl, recommended_next_step=nxt,
        evidence_sources=evidence, supporting_findings=supporting, conflicting_findings=conflicting,
        missing_evidence=missing, data_quality_warnings=quality, unsupported_conclusions=unsupported,
        limitations=[DIAGNOSIS_LIMITATIONS_DISCLAIMER],
    )
