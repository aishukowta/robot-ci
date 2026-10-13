"""Tests for the handwriting robot extension (robot_ci.writing)."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from robot_ci.writing.font import GLYPHS, PATTERNS, SUPPORTED_CHARACTERS, densify, glyph, polyline_length
from robot_ci.writing.targets import WritingTarget, load_targets, get_target
from robot_ci.writing.plan import PenPlan
from robot_ci.writing.policies import WRITING_POLICY_NAMES, get_writing_policy
from robot_ci.writing.simulate import (KinematicParams, KinematicPenSimulator, PyBulletPenSimulator,
                                       build_commanded_trajectory)
from robot_ci.writing.metrics import (get_recognizer, nearest_distances, repeatability, score_execution,
                                      strokes_to_points, trajectory_error, writing_accuracy)
from robot_ci.writing.regression import WritingRunConfig, evaluate_metric, run_writing_regression
from robot_ci.writing.export import ExportSafetyError, RobotProfile, export_plan, plan_to_gcode
from robot_ci.writing.diagnosis import diagnose_writing
from robot_ci.diagnosis.models import DiagnosisCategory


# ---------------------------------------------------------------- font / targets

def test_font_covers_letters_digits_and_patterns():
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
        assert ch in GLYPHS
        w, strokes = glyph(ch)
        assert w > 0 and strokes and all(s.shape[1] == 2 for s in strokes)
    assert {"square", "circle", "triangle", "star", "spiral", "zigzag", "sine_wave"} <= set(PATTERNS)
    with pytest.raises(ValueError):
        glyph("@")


def test_all_shipped_targets_load_and_fit_on_page():
    targets = load_targets("writing_targets")
    assert len(targets) >= 10
    kinds = {t.kind for t in targets}
    assert kinds == {"text", "pattern"}
    for t in targets:
        lay = t.layout()
        assert lay.cells and lay.strokes


def test_target_outside_page_is_rejected():
    with pytest.raises(ValueError, match="does not fit"):
        WritingTarget.from_dict({"target_id": "x", "kind": "text", "content": "HELLO",
                                 "origin_mm": [180, 60], "height_mm": 20})


def test_densify_preserves_length_and_spacing():
    s = np.array([[0, 0], [10, 0], [10, 5]], dtype=float)
    d = densify(s, 0.5)
    assert abs(polyline_length(d) - 15.0) < 1e-9
    assert np.max(np.linalg.norm(np.diff(d, axis=0), axis=1)) <= 0.5 + 1e-9


# ---------------------------------------------------------------- plans / policies

def test_plan_is_deterministic_and_hash_detects_tampering(tmp_path):
    t = get_target("word_hello")
    p1 = get_writing_policy("writing_baseline").plan(t.layout(), 5)
    p2 = get_writing_policy("writing_baseline").plan(t.layout(), 5)
    assert p1.plan_hash == p2.plan_hash
    path = tmp_path / "plan.json"
    p1.save(path)
    assert PenPlan.load(path).plan_hash == p1.plan_hash
    data = json.loads(path.read_text())
    data["waypoints"][5][0] += 1.0
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hash mismatch"):
        PenPlan.load(path)


def test_baseline_plan_follows_target_strokes_exactly():
    t = get_target("pattern_square")
    plan = get_writing_policy("writing_baseline").plan(t.layout(), 1)
    ref = strokes_to_points(t.layout().strokes, 0.25)
    pts = strokes_to_points(plan.pen_down_strokes(), 0.25)
    assert nearest_distances(pts, ref).max() < 1e-6
    assert len(plan.pen_down_strokes()) == len(t.layout().strokes)


def test_candidate_policies_inject_defects():
    # Defects are target-dependent (e.g. pen dragging needs strokes < 6 mm apart), so use the alphabet
    t = get_target("letters_a_m")
    base = get_writing_policy("writing_baseline").plan(t.layout(), 3)
    for name in WRITING_POLICY_NAMES:
        if name == "writing_baseline":
            continue
        cand = get_writing_policy(name).plan(t.layout(), 3)
        assert cand.plan_hash != base.plan_hash, name
    slow = get_writing_policy("writing_candidate_slow").plan(t.layout(), 3)
    assert slow.nominal_duration_s() > base.nominal_duration_s()
    drag = get_writing_policy("writing_candidate_drag").plan(t.layout(), 3)
    assert len(drag.pen_down_strokes()) < len(base.pen_down_strokes())


def test_existing_manipulation_policies_untouched():
    from robot_ci.policies.registry import get_policy
    assert get_policy("baseline").version == "v1.0-baseline"
    assert get_policy("candidate").version == "v2.0-candidate"


# ---------------------------------------------------------------- simulators

def test_commanded_trajectory_respects_feed():
    t = get_target("pattern_square")
    plan = get_writing_policy("writing_baseline").plan(t.layout(), 1)
    cmd = build_commanded_trajectory(plan, 0.01)
    step = np.linalg.norm(np.diff(cmd.xy, axis=0), axis=1)
    assert step.max() <= 60.0 * 0.01 + 1e-6  # never faster than the travel feed
    assert cmd.pen_down.any() and (~cmd.pen_down).any()


def test_kinematic_simulator_writes_accurately():
    t = get_target("word_hello")
    layout = t.layout()
    plan = get_writing_policy("writing_baseline").plan(layout, 1)
    ex = KinematicPenSimulator().execute(plan, 1)
    m = score_execution(ex, layout, sum(polyline_length(s) for s in layout.strokes))
    assert m.values["trajectory_error_mean_mm"] < 0.2
    assert m.values["writing_accuracy"] == 1.0
    assert len(ex.ink_strokes()) == len(plan.pen_down_strokes())


def test_pybullet_simulator_writes_with_stand_in_arm():
    t = get_target("pattern_triangle")
    layout = t.layout()
    plan = get_writing_policy("writing_baseline").plan(layout, 1)
    sim = PyBulletPenSimulator()
    try:
        ex = sim.execute(plan, 1)
    finally:
        sim.close()
    assert abs(sim.control_dt - 4 / 240) < 1e-12
    m = score_execution(ex, layout, sum(polyline_length(s) for s in layout.strokes))
    assert m.values["trajectory_error_mean_mm"] < 0.3
    assert m.values["writing_accuracy"] == 1.0
    assert ex.safety["joint_limit_violations"] == 0
    assert ex.safety["unexpected_contact_events"] == 0


def test_simulator_offset_shows_up_as_trajectory_error():
    t = get_target("word_robot")
    layout = t.layout()
    plan = get_writing_policy("writing_baseline").plan(layout, 1)
    ex = KinematicPenSimulator(replace(KinematicParams(), offset_mm=(1.0, 0.0))).execute(plan, 1)
    te = trajectory_error(strokes_to_points(ex.ink_strokes()), layout.strokes)
    assert 0.5 < te["mean_mm"] < 1.2


# ---------------------------------------------------------------- metrics

def test_recogniser_reads_every_font_symbol():
    rec = get_recognizer()
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
        pts = np.vstack([densify(s, 0.03) for s in GLYPHS[ch][1]])
        assert rec.classify(pts, "character")["predicted"] == ch, ch
    for name, strokes in PATTERNS.items():
        pts = np.vstack([densify(s, 0.01) for s in strokes])
        assert rec.classify(pts, "pattern")["predicted"] == name, name


def test_empty_writing_scores_zero_accuracy():
    lay = get_target("word_hello").layout()
    acc = writing_accuracy(np.zeros((0, 2)), lay)
    assert acc["accuracy"] == 0.0


def test_repeatability_zero_for_identical_runs():
    s = [np.array([[0, 0], [10, 0], [10, 10]], dtype=float)]
    assert repeatability([s, s])["mean_pairwise_rms_mm"] < 1e-9
    assert repeatability([s]) is None


def test_paired_regression_rule_requires_consistent_change():
    cfg = {"threshold": 0.3, "direction": "increase", "comparison": "absolute",
           "severity_levels": {"minor": 0.3, "major": 1.0, "critical": 2.0}}
    consistent = evaluate_metric("trajectory_error_mean_mm", [0.1, 0.1, 0.1, 0.1], [0.6, 0.7, 0.65, 0.6], cfg)
    assert consistent["regression_detected"] and consistent["severity"] == "minor"
    one_outlier = evaluate_metric("trajectory_error_mean_mm", [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 2.0], cfg)
    assert not one_outlier["regression_detected"]
    assert "includes 0" in one_outlier["reason"]


def test_writing_regression_detects_injected_candidate(tmp_path):
    r = run_writing_regression(WritingRunConfig(targets=["word_hello", "pattern_star"], trials=3,
                                                backend="kinematic", results_dir=str(tmp_path)))
    assert r["overall_status"] == "REGRESSION_DETECTED"
    flagged = {f["metric"] for f in r["findings"] if f["regression_detected"]}
    assert "trajectory_error_mean_mm" in flagged
    assert Path(r["saved_to"]).exists()
    assert all(Path(p).exists() for p in r["images"].values())


def test_writing_regression_passes_baseline_against_itself(tmp_path):
    r = run_writing_regression(WritingRunConfig(candidate="writing_baseline", targets=["word_hello"], trials=3,
                                                backend="kinematic", results_dir=str(tmp_path), render=False))
    assert r["overall_status"] == "PASS"


# ---------------------------------------------------------------- export

def test_export_without_configured_profile_withholds_gcode(tmp_path):
    t = get_target("word_hello")
    plan = get_writing_policy("writing_baseline").plan(t.layout(), t.seed)
    man = export_plan(plan, RobotProfile.load("config/writing_robot_profile.yaml"), str(tmp_path))
    assert "gcode" not in man["files"]
    assert {"plan", "svg", "csv", "manifest", "lab_instructions"} <= set(man["files"])
    assert plan.plan_hash in Path(man["files"]["svg"]).read_text()


def test_export_refuses_points_outside_workspace(tmp_path):
    t = get_target("word_hello")
    plan = get_writing_policy("writing_baseline").plan(t.layout(), t.seed)
    prof = replace(RobotProfile(), is_configured=True, workspace_mm=[[0, 0], [50, 50]])
    with pytest.raises(ExportSafetyError):
        export_plan(plan, prof, str(tmp_path))


def test_gcode_coordinates_and_feed_clamp():
    t = get_target("pattern_square")
    plan = get_writing_policy("writing_baseline").plan(t.layout(), t.seed)
    prof = replace(RobotProfile(), is_configured=True, machine_origin="bottom_left", max_draw_feed_mm_s=20.0)
    text, clamped = plan_to_gcode(plan, prof)
    assert clamped["draw"] > 0
    first_move = next(l for l in text.splitlines() if l.startswith("G1 X"))
    assert "Y287.000" in first_move  # park (10,10) -> bottom-left origin y = 297 - 10
    assert "F1200.0" in text         # 20 mm/s clamp = 1200 mm/min


# ---------------------------------------------------------------- real-to-sim analysis + diagnosis

@pytest.fixture(scope="module")
def synthetic_analysis(tmp_path_factory):
    from robot_ci.writing.analysis import AnalysisInputs, analyze_recording
    from robot_ci.writing.selftest import make_synthetic_recording
    out = tmp_path_factory.mktemp("selftest")
    rec = make_synthetic_recording(out, "word_hello", with_video=False)
    rep = analyze_recording(AnalysisInputs(plan_path=rec["plan_path"], data_source="robot_phone_recording",
                                           executed_plan=True, photo_path=rec["photo_path"],
                                           sim_backend="kinematic"), out / "analysis")
    return rep


def test_photo_analysis_recovers_known_offset(synthetic_analysis):
    rep = synthetic_analysis
    # file name contains 'synthetic' -> forced to synthetic_fixture even though robot was declared
    assert rep["data_source"] == "synthetic_fixture"
    assert rep["comparison_type"] == "not_robot_evidence"
    t = rep["placement_correction"]["translation_mm"]
    assert abs(t[0] + 1.2) < 0.5 and abs(t[1] - 0.8) < 0.5
    assert rep["metrics"]["intended_vs_real_aligned"]["writing_accuracy"] == 1.0
    assert rep["metrics"]["intended_vs_real_aligned"]["trajectory_error_mean_mm"] < 0.6
    assert rep["page_detection"]["aspect_error"] < 0.06


def test_diagnosis_never_uses_synthetic_as_robot_evidence(synthetic_analysis):
    run = {"run_id": "r", "backend": "kinematic", "trials": 3, "targets": ["word_hello"],
           "baseline": {"name": "writing_baseline"}, "candidate": {"name": "writing_candidate"},
           "findings": [{"label": "x", "severity": "major", "reason": "r", "regression_detected": True}]}
    d = diagnose_writing(run, synthetic_analysis)
    assert d.category == DiagnosisCategory.POLICY_REGRESSION_LIKELY
    assert any("synthetic" in w for w in d.data_quality_warnings)


def test_diagnosis_matched_robot_recording_rules(synthetic_analysis):
    real = json.loads(json.dumps(synthetic_analysis))
    real["data_source"] = "robot_phone_recording"
    real["comparison_type"] = "matched_plan_execution"
    run_pass = {"run_id": "r", "backend": "kinematic", "trials": 3, "targets": ["word_hello"],
                "baseline": {"name": "writing_baseline"}, "candidate": {"name": "writing_candidate"},
                "findings": [{"label": "x", "regression_detected": False}]}
    assert diagnose_writing(run_pass, real).category == DiagnosisCategory.NO_REGRESSION_DETECTED
    real["metrics"]["real_vs_sim_aligned"]["symmetric_mean_mm"] = 2.0
    assert diagnose_writing(run_pass, real).category == DiagnosisCategory.SIM_TO_REAL_MISMATCH_LIKELY
    cross = dict(real, comparison_type="cross_controller_behavioural_comparison")
    assert diagnose_writing(None, cross).category == DiagnosisCategory.INCONCLUSIVE_INSUFFICIENT_EVIDENCE
