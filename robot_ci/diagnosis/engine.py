"""Evidence-based diagnosis engine for Robot CI.

Combines policy-regression benchmarks and real-to-simulation trajectory comparison
evidence into structured assessments of system failures without fabricating causality.
"""

from typing import Dict, Any, List, Optional, Union
import uuid

from robot_ci.diagnosis.models import (
    DiagnosisCategory,
    DiagnosisReport,
    EvidenceItem,
    DIAGNOSIS_LIMITATIONS_DISCLAIMER,
)
from robot_ci.results.models import TestRunResult
from robot_ci.vision.comparison import VideoSimComparisonReport


def evaluate_diagnosis(
    ci_result: Optional[Union[Dict[str, Any], TestRunResult]] = None,
    video_report: Optional[Union[Dict[str, Any], VideoSimComparisonReport]] = None,
    calibrated_pixel_threshold: float = 30.0,
    shape_normalized_threshold: float = 0.25,
    min_tracking_coverage: float = 0.50,
) -> DiagnosisReport:
    """Evaluate available simulation regression and real-to-sim evidence.
    
    Args:
        ci_result: TestRunResult object or dict from results/run_*.json.
        video_report: VideoSimComparisonReport object or dict from results/video_sim_comparisons/*.json.
        calibrated_pixel_threshold: Pixel reprojection error threshold for Mode A divergence.
        shape_normalized_threshold: Dimensionless normalized error threshold for Mode B divergence.
        min_tracking_coverage: Minimum valid visual tracking coverage ratio [0..1].
    
    Returns:
        DiagnosisReport containing category, plain-English summary, and provenance.
    """
    evidence_sources: List[EvidenceItem] = []
    supporting_findings: List[str] = []
    conflicting_findings: List[str] = []
    missing_evidence: List[str] = []
    data_quality_warnings: List[str] = []
    unsupported_conclusions: List[str] = [
        "Causation cannot be proven from trajectory differences alone.",
        "Physical collisions, contact forces, and joint torques cannot be inferred from monocular video.",
    ]

    # Standardize dictionary formats
    ci_dict: Optional[Dict[str, Any]] = None
    if isinstance(ci_result, TestRunResult):
        ci_dict = ci_result.to_dict()
    elif isinstance(ci_result, dict) and ci_result:
        ci_dict = ci_result

    v_dict: Optional[Dict[str, Any]] = None
    if isinstance(video_report, VideoSimComparisonReport):
        v_dict = video_report.to_dict()
    elif isinstance(video_report, dict) and video_report:
        v_dict = video_report

    # -----------------------------------------------------------------------
    # 1. Evaluate Simulation Policy Regression Evidence
    # -----------------------------------------------------------------------
    policy_regression_detected: Optional[bool] = None

    if ci_dict is None:
        missing_evidence.append("Baseline-vs-candidate simulation policy regression results are missing.")
    else:
        run_id = ci_dict.get("run_id", "unknown_run")
        status = str(ci_dict.get("overall_status", "UNKNOWN")).upper()
        findings = ci_dict.get("regression_findings", [])

        # Validate structure
        if not findings and status == "UNKNOWN":
            data_quality_warnings.append(f"CI run '{run_id}' report is malformed or lacks regression findings.")
        else:
            evidence_sources.append(EvidenceItem(
                source_type="SIMULATION_REGRESSION",
                identifier=run_id,
                provenance="PYBULLET_SIMULATION",
                is_physical_hardware=False,
                is_valid=True,
                summary=f"CI regression run {run_id} ({ci_dict.get('baseline_policy_version')} vs {ci_dict.get('candidate_policy_version')})",
                details={
                    "overall_status": status,
                    "simulator": ci_dict.get("simulator_name", "pybullet"),
                    "seed": ci_dict.get("seed", 42),
                },
            ))

            regressed_findings = [f for f in findings if f.get("regression_detected", False)]
            if regressed_findings or status in ("REGRESSION_DETECTED", "FAIL", "FAILED"):
                policy_regression_detected = True
                for rf in regressed_findings:
                    m_name = rf.get("metric_name", "unknown_metric")
                    sev = rf.get("severity", "none").upper()
                    thresh = rf.get("threshold", "-")
                    delta = rf.get("delta", 0.0)
                    supporting_findings.append(
                        f"Simulation policy regression in '{m_name}' (severity: {sev}, delta: {delta:+.4f}, threshold: {thresh})."
                    )
            else:
                policy_regression_detected = False
                supporting_findings.append(
                    "No simulation regressions detected: candidate policy met all metric thresholds relative to baseline."
                )

    # -----------------------------------------------------------------------
    # 2. Evaluate Video-to-Simulation Trajectory Evidence
    # -----------------------------------------------------------------------
    sim_to_real_mismatch_detected: Optional[bool] = None

    if v_dict is None:
        missing_evidence.append("Video-to-simulation trajectory comparison report is missing.")
    else:
        v_id = v_dict.get("report_id", "unknown_report")
        prov = v_dict.get("provenance", {})
        metrics = v_dict.get("metrics", {})
        mode = v_dict.get("comparison_mode", "")
        comp_type = v_dict.get("comparison_type", "")
        phys_ctrl = v_dict.get("physical_controller_id", "unknown")
        video_file = str(prov.get("video_file", ""))
        units = v_dict.get("units", "")

        is_synthetic = bool(
            prov.get("video_source_is_synthetic", False)
            or v_dict.get("is_synthetic", False)
            or "synthetic" in video_file.lower()
        )

        tracking_cov = float(metrics.get("tracking_coverage", 0.0))
        rms_dev = float(metrics.get("rms_trajectory_deviation", 0.0))
        raw_rms_dev = float(metrics.get("raw_rms_trajectory_deviation", rms_dev))

        # Check validity
        if not metrics:
            data_quality_warnings.append(f"Video comparison report '{v_id}' lacks metrics.")
        else:
            evidence_sources.append(EvidenceItem(
                source_type="VIDEO_TRAJECTORY_COMPARISON",
                identifier=v_id,
                provenance="SYNTHETIC_FIXTURE" if is_synthetic else "PHYSICAL_ROBOT_RECORDING",
                is_physical_hardware=not is_synthetic,
                is_valid=True,
                summary=f"Video comparison {v_id} ({mode}, controller: {phys_ctrl}, tracking: {tracking_cov * 100:.1f}%)",
                details={
                    "mode": mode,
                    "units": units,
                    "rms_deviation": rms_dev,
                    "raw_rms_deviation": raw_rms_dev,
                    "comparison_type": comp_type,
                },
            ))

            # Rule: Synthetic fixtures can NEVER qualify as physical robot evidence
            if is_synthetic:
                data_quality_warnings.append(
                    "Synthetic video fixture detected: synthetic test fixtures cannot qualify as physical robot evidence."
                )
                unsupported_conclusions.append(
                    "Physical sim-to-real transfer cannot be assessed because the video source is a synthetic test fixture."
                )
            # Rule: Adequate tracking coverage
            elif tracking_cov < min_tracking_coverage:
                data_quality_warnings.append(
                    f"Low visual tracking coverage ({tracking_cov * 100:.1f}% < {min_tracking_coverage * 100:.1f}%): "
                    "excessive tracking dropout prevents reliable trajectory deviation evaluation."
                )
                unsupported_conclusions.append(
                    "Sim-to-real trajectory divergence cannot be confirmed due to excessive camera tracking dropout."
                )
            # Rule: Manual teleoperation / cross-controller cannot validate candidate policy execution
            elif comp_type != "matched_policy_verification" or "teleop" in phys_ctrl.lower():
                data_quality_warnings.append(
                    f"Cross-controller / teleoperated recording ('{phys_ctrl}'): cannot validate candidate policy execution on hardware."
                )
                unsupported_conclusions.append(
                    "Hardware trajectory discrepancy cannot prove candidate policy failure because physical robot was operated "
                    "under manual teleoperation or a non-matching controller."
                )
            # Genuine physical recording of matching policy with sufficient tracking coverage!
            else:
                if "CALIBRATED" in mode.upper():
                    if rms_dev > calibrated_pixel_threshold:
                        sim_to_real_mismatch_detected = True
                        supporting_findings.append(
                            f"Calibrated reprojection error ({rms_dev:.1f} px) exceeds optical threshold ({calibrated_pixel_threshold:.1f} px), "
                            "indicating significant physical robot trajectory divergence from simulation."
                        )
                    else:
                        sim_to_real_mismatch_detected = False
                        supporting_findings.append(
                            f"Calibrated reprojection error ({rms_dev:.1f} px) is within optical threshold ({calibrated_pixel_threshold:.1f} px)."
                        )
                else:
                    if rms_dev > shape_normalized_threshold:
                        sim_to_real_mismatch_detected = True
                        supporting_findings.append(
                            f"Normalized shape deviation ({rms_dev:.4f}) exceeds geometric threshold ({shape_normalized_threshold:.4f}), "
                            "indicating substantial 2D trajectory divergence in normalized screen space."
                        )
                    else:
                        sim_to_real_mismatch_detected = False
                        supporting_findings.append(
                            f"Normalized shape deviation ({rms_dev:.4f}) is within geometric threshold ({shape_normalized_threshold:.4f})."
                        )

                # Check if alignment masked raw discrepancy
                if raw_rms_dev > 1.8 * rms_dev:
                    data_quality_warnings.append(
                        f"Alignment masking notice: Raw unaligned deviation was {raw_rms_dev:.4f}, but Procrustes alignment reduced it to {rms_dev:.4f}. "
                        "Rigid alignment may conceal significant spatial translation or scale mismatch."
                    )

    # -----------------------------------------------------------------------
    # 3. Combine Evidence into Diagnosis Category
    # -----------------------------------------------------------------------
    diag_id = str(uuid.uuid4())[:8]

    # Case A: Both evidence channels are conclusive
    if policy_regression_detected is not None and sim_to_real_mismatch_detected is not None:
        if policy_regression_detected and sim_to_real_mismatch_detected:
            category = DiagnosisCategory.BOTH_ISSUES_LIKELY
            explanation = (
                "Both a simulation policy regression and a physical sim-to-real mismatch are indicated by the available evidence. "
                "The candidate policy degraded benchmark performance relative to the baseline in simulation, and physical robot "
                "trajectory execution also diverged substantially from simulated rollouts."
            )
            recommendation = (
                "Address simulation policy regressions first. Verify reward curves, success rates, and action magnitudes in PyBullet "
                "before testing candidate revisions on physical robot hardware."
            )
        elif policy_regression_detected and not sim_to_real_mismatch_detected:
            category = DiagnosisCategory.POLICY_REGRESSION_LIKELY
            explanation = (
                "Evidence supports a simulation policy regression, while physical trajectory tracking remained within expected tolerances. "
                "The candidate policy failed one or more benchmark metric thresholds relative to baseline in simulation."
            )
            recommendation = (
                "Investigate simulated benchmark regressions (e.g. completion time or action jerk). The physical robot trajectory "
                "did not exhibit abnormal geometric deviation, suggesting an algorithmic rather than physical transfer issue."
            )
        elif not policy_regression_detected and sim_to_real_mismatch_detected:
            category = DiagnosisCategory.SIM_TO_REAL_MISMATCH_LIKELY
            explanation = (
                "Evidence supports a physical sim-to-real mismatch. The candidate policy met all benchmark thresholds in PyBullet simulation, "
                "but physical robot execution diverged significantly from the simulated trajectory."
            )
            recommendation = (
                "Audit simulator fidelity: verify robot mass/inertia, joint friction, latency/control frequency, and camera calibration parameters. "
                "The policy behaves correctly in simulation but encounters an unmodeled physical domain shift."
            )
        else:
            category = DiagnosisCategory.NO_REGRESSION_DETECTED
            explanation = (
                "No policy regression or sim-to-real mismatch is indicated. The candidate policy satisfied all benchmark thresholds in simulation, "
                "and physical robot execution closely matched the simulated rollout trajectory."
            )
            recommendation = (
                "Proceed with expanded validation across additional random seeds, obstacle configurations, and lighting conditions."
            )

    # Case B: Only Policy Regression is conclusive
    elif policy_regression_detected is not None and sim_to_real_mismatch_detected is None:
        if policy_regression_detected:
            category = DiagnosisCategory.POLICY_REGRESSION_LIKELY
            explanation = (
                "Simulation benchmark evidence indicates a policy regression between baseline and candidate. "
                "Physical sim-to-real transfer could not be assessed because physical robot video evidence is absent, synthetic, or teleoperated."
            )
            recommendation = (
                "Fix simulation regressions in PyBullet benchmarks. Collect a genuine physical robot video executing the matching candidate policy."
            )
            unsupported_conclusions.append(
                "Physical sim-to-real transfer cannot be evaluated without genuine physical robot video observations."
            )
        else:
            category = DiagnosisCategory.NO_REGRESSION_DETECTED
            explanation = (
                "No policy regressions were detected in simulation benchmarks. "
                "However, physical hardware transfer remains unverified because physical robot video evidence is absent or synthetic."
            )
            recommendation = (
                "Record a physical robot execution video of the candidate policy to evaluate real-world trajectory transfer."
            )
            unsupported_conclusions.append(
                "Physical robot performance and sim-to-real transfer remain unverified."
            )

    # Case C: Only Video Comparison is conclusive
    elif policy_regression_detected is None and sim_to_real_mismatch_detected is not None:
        if sim_to_real_mismatch_detected:
            category = DiagnosisCategory.SIM_TO_REAL_MISMATCH_LIKELY
            explanation = (
                "Physical robot trajectory diverged significantly from the simulated rollout, indicating a sim-to-real mismatch. "
                "However, algorithmic policy regression relative to a baseline was not benchmarked."
            )
            recommendation = (
                "Run the complete simulation regression pipeline (`robot_ci run`) to determine whether the policy itself degraded relative to baseline."
            )
            unsupported_conclusions.append(
                "Algorithmic policy degradation relative to baseline cannot be evaluated without CI benchmark reports."
            )
        else:
            category = DiagnosisCategory.NO_REGRESSION_DETECTED
            explanation = (
                "Physical robot trajectory aligned closely with the simulated rollout. "
                "However, baseline-versus-candidate regression benchmarks were not provided."
            )
            recommendation = (
                "Run the CI policy regression pipeline to verify candidate performance relative to the baseline policy."
            )
            unsupported_conclusions.append(
                "Algorithmic improvement or regression relative to baseline remains unverified without simulation CI results."
            )

    # Case D: Inconclusive / Insufficient Evidence
    else:
        category = DiagnosisCategory.INCONCLUSIVE_INSUFFICIENT_EVIDENCE
        explanation = (
            "Available evidence is insufficient to reach an evidence-based diagnosis. "
            "Simulation regression benchmarks and/or usable physical robot video comparisons are missing, synthetic, or incomplete."
        )
        recommendation = (
            "Run the full regression test suite (`robot_ci run`) and capture a genuine physical robot execution recording with clear marker tracking."
        )
        unsupported_conclusions.append(
            "Neither policy regression nor sim-to-real mismatch can be assessed from the supplied data."
        )

    return DiagnosisReport(
        diagnosis_id=diag_id,
        category=category,
        summary_explanation=explanation,
        recommended_next_step=recommendation,
        evidence_sources=evidence_sources,
        supporting_findings=supporting_findings,
        conflicting_findings=conflicting_findings,
        missing_evidence=missing_evidence,
        data_quality_warnings=data_quality_warnings,
        unsupported_conclusions=unsupported_conclusions,
        limitations=[DIAGNOSIS_LIMITATIONS_DISCLAIMER],
    )
