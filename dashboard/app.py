"""Robot CI - Streamlit Results & Interactive Telemetry Dashboard.

Two Primary Sections:
- Section A: Policy Regression (CI test runs, regression findings, interactive simulation, and offline telemetry)
- Section B: Real-to-Simulation (Video trajectory extraction, PyBullet rollout comparison, calibrated projection, shape-only alignment, and structured reports)
"""

import sys
import json
import argparse
from pathlib import Path
import html
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from robot_ci.policies.registry import get_policy
from robot_ci.scenarios.manager import ScenarioManager
from robot_ci.simulation.pybullet_sim import PyBulletSimulator
from robot_ci.replay.engine import ReplayEngine


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

COLORS = {
    "bg": "#F7F3EB",
    "bg_elevated": "#FBF8F2",
    "bg_sidebar": "#EFE8DC",
    "border": "#D9CFC0",
    "text": "#3A2A1F",
    "text_muted": "#7A6555",
    "accent": "#A66A4E",
    "baseline": "#5B7A5A",
    "candidate": "#A85A4A",
    "pass": "#5B7A5A",
    "fail": "#A85A4A",
    "warn": "#B8893A",
    "critical_bg": "#F3E4DE",
    "pass_bg": "#E6EDE5",
    "warn_bg": "#F4EBDC",
}


def inject_theme() -> None:
    """Apply warm laboratory styling. Presentation only."""
    c = COLORS
    st.markdown(
        f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {{
        font-family: "Source Sans 3", "Segoe UI", sans-serif;
        color: {c["text"]};
    }}

    .stApp {{
        background-color: {c["bg"]};
    }}

    [data-testid="stSidebar"] {{
        background-color: {c["bg_sidebar"]};
        border-right: 1px solid {c["border"]};
    }}

    [data-testid="stSidebar"] * {{
        color: {c["text"]};
    }}

    [data-testid="stHeader"] {{
        background: {c["bg"]};
    }}

    h1, h2, h3, h4 {{
        color: {c["text"]} !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em;
    }}

    .block-container {{
        padding-top: 1.25rem;
        padding-bottom: 2.5rem;
        max-width: 1180px;
    }}

    .rci-header {{
        border-bottom: 1px solid {c["border"]};
        padding-bottom: 0.85rem;
        margin-bottom: 1.25rem;
    }}

    .rci-brand {{
        font-size: 1.35rem;
        font-weight: 700;
        color: {c["text"]};
        margin: 0;
        line-height: 1.2;
    }}

    .rci-title {{
        font-size: 0.95rem;
        font-weight: 500;
        color: {c["accent"]};
        margin: 0.15rem 0 0.35rem 0;
    }}

    .rci-subtitle {{
        font-size: 0.88rem;
        color: {c["text_muted"]};
        margin: 0;
        max-width: 46rem;
    }}

    .rci-section {{
        margin: 1.35rem 0 0.65rem 0;
        padding-bottom: 0.35rem;
        border-bottom: 1px solid {c["border"]};
    }}

    .rci-section h2 {{
        font-size: 1.05rem !important;
        margin: 0 !important;
        padding: 0 !important;
    }}

    .rci-section-note {{
        font-size: 0.82rem;
        color: {c["text_muted"]};
        margin: 0.35rem 0 0.85rem 0;
    }}

    .rci-panel {{
        background: {c["bg_elevated"]};
        border: 1px solid {c["border"]};
        border-radius: 4px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.75rem;
    }}

    .rci-meta-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.65rem;
        margin-bottom: 0.75rem;
    }}

    .rci-meta-item {{
        background: {c["bg"]};
        border: 1px solid {c["border"]};
        border-radius: 3px;
        padding: 0.55rem 0.65rem;
    }}

    .rci-meta-label {{
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {c["text_muted"]};
        margin-bottom: 0.2rem;
    }}

    .rci-meta-value {{
        font-family: "IBM Plex Mono", ui-monospace, monospace;
        font-size: 0.84rem;
        color: {c["text"]};
        word-break: break-word;
    }}

    .rci-alert {{
        border-radius: 3px;
        padding: 0.7rem 0.9rem;
        border: 1px solid;
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        margin-top: 0.15rem;
    }}

    .rci-alert-fail {{
        background: {c["critical_bg"]};
        border-color: {c["fail"]};
        color: {c["fail"]};
    }}

    .rci-alert-pass {{
        background: {c["pass_bg"]};
        border-color: {c["pass"]};
        color: {c["pass"]};
    }}

    .rci-sidebar-brand {{
        font-size: 1.05rem;
        font-weight: 700;
        color: {c["text"]};
        margin: 0.2rem 0 0.85rem 0;
        padding-bottom: 0.65rem;
        border-bottom: 1px solid {c["border"]};
    }}

    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
</style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data Loading Utilities
# ---------------------------------------------------------------------------

def load_results(results_dir: str = "results") -> list:
    """Load all saved CI run JSON reports from the results directory."""
    path = Path(results_dir)
    if not path.exists():
        return []

    files = sorted(path.glob("run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    results = []
    for f in files:
        try:
            with open(f, encoding="utf-8") as fp:
                results.append(json.load(fp))
        except Exception:
            pass
    return results


def load_comparison_reports(directory: str = "results/video_sim_comparisons") -> list:
    """Load all saved video-sim comparison reports from results directory."""
    p = Path(directory)
    if not p.exists():
        return []
    files = sorted(p.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    reports = []
    for f in files:
        try:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
                data["_file_path"] = str(f)
                reports.append(data)
        except Exception:
            pass
    return reports


def _fmt_ts(ts: str) -> str:
    if not ts:
        return "-"
    return ts.replace("T", " ")[:19]


def _is_regression(status: str) -> bool:
    return status.upper() in ("REGRESSION_DETECTED", "FAIL", "FAILED")


def render_header() -> None:
    st.markdown(
        """
<div class="rci-header">
  <div class="rci-brand">ROBOT CI</div>
  <div class="rci-title">Robotic Policy Regression Testing & Simulation-to-Real Verification</div>
  <div class="rci-subtitle">Unified platform for continuous policy evaluation, automated regression detection, and empirical real-to-simulation trajectory comparisons.</div>
</div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# SECTION A: POLICY REGRESSION
# ===========================================================================

def render_section_a_policy_regression(results_dir: str = "results") -> None:
    st.markdown('<div class="rci-section"><h2>Section A: Policy Regression</h2></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="rci-section-note">'
        'Inspect actual baseline-vs-candidate simulation results, regression status, '
        'responsible thresholds, and live/offline telemetry from genuine rollouts.'
        '</div>',
        unsafe_allow_html=True,
    )

    tab_runs, tab_live, tab_offline = st.tabs([
        "📋 CI Run Reports & Regressions",
        "📊 Interactive Simulation & Telemetry",
        "📁 Offline Telemetry Analysis",
    ])

    with tab_runs:
        _render_ci_run_reports_view(results_dir)

    with tab_live:
        render_live_telemetry_panel()

    with tab_offline:
        render_offline_telemetry_panel(results_dir)


def _render_ci_run_reports_view(results_dir: str) -> None:
    results = load_results(results_dir)
    if not results:
        st.warning(
            f"No stored CI run reports found in `{results_dir}/`.\n\n"
            "Run the automated regression pipeline first:\n\n"
            "```powershell\n"
            "python -m robot_ci run --baseline baseline --candidate candidate --simulator pybullet\n"
            "```"
        )
        return

    # Run selector
    col_sel, col_stat = st.columns([3, 1])
    with col_sel:
        run_idx = st.selectbox(
            "Select Saved CI Run Report",
            options=list(range(len(results))),
            format_func=lambda i: f"Run {results[i].get('run_id')} ({_fmt_ts(results[i].get('timestamp'))}) — {results[i].get('simulator_name', 'pybullet')}",
            key="section_a_run_select",
        )
    result = results[run_idx]

    status = result.get("overall_status", "UNKNOWN")
    reg_detected = _is_regression(status)
    css_cls = "rci-alert-fail" if reg_detected else "rci-alert-pass"
    status_str = "REGRESSION DETECTED" if reg_detected else "PASS — ALL METRICS WITHIN THRESHOLDS"

    with col_stat:
        st.markdown(f'<div class="rci-alert {css_cls}" style="margin-top:1.75rem; text-align:center;">{status_str}</div>', unsafe_allow_html=True)

    # 1. Run Metadata
    st.markdown(
        f"""
<div class="rci-panel">
  <div class="rci-meta-grid">
    <div class="rci-meta-item">
      <div class="rci-meta-label">Run ID</div>
      <div class="rci-meta-value">{result.get("run_id", "-")}</div>
    </div>
    <div class="rci-meta-item">
      <div class="rci-meta-label">Timestamp</div>
      <div class="rci-meta-value">{_fmt_ts(result.get("timestamp", ""))}</div>
    </div>
    <div class="rci-meta-item">
      <div class="rci-meta-label">Baseline Policy</div>
      <div class="rci-meta-value">{result.get("baseline_policy_version", "-")}</div>
    </div>
    <div class="rci-meta-item">
      <div class="rci-meta-label">Candidate Policy</div>
      <div class="rci-meta-value">{result.get("candidate_policy_version", "-")}</div>
    </div>
    <div class="rci-meta-item">
      <div class="rci-meta-label">Simulator Backend</div>
      <div class="rci-meta-value">{result.get("simulator_name", "pybullet")}</div>
    </div>
    <div class="rci-meta-item">
      <div class="rci-meta-label">Random Seed</div>
      <div class="rci-meta-value">{result.get("seed", 42)}</div>
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Overall Evaluation Metrics Summary
    st.markdown('<div class="rci-section"><h3>Evaluation Metrics Summary (Baseline vs Candidate)</h3></div>', unsafe_allow_html=True)
    metric_results = result.get("metric_results", [])
    if metric_results:
        cols = st.columns(min(len(metric_results), 3))
        for idx, mr in enumerate(metric_results):
            col = cols[idx % len(cols)]
            name = mr.get("metric_name", "-")
            b_val = mr.get("baseline_value", 0.0)
            c_val = mr.get("candidate_value", 0.0)
            delta = mr.get("delta", 0.0)
            col.metric(
                label=name.replace("_", " ").title(),
                value=f"{c_val:.4f}",
                delta=f"Δ {delta:+.4f} (Base: {b_val:.4f})",
            )

    # 3. Regression Findings & Thresholds Responsible
    st.markdown('<div class="rci-section"><h3>Regression Status & Thresholds Responsible</h3></div>', unsafe_allow_html=True)
    findings = result.get("regression_findings", [])
    if findings:
        for f in findings:
            is_reg = f.get("regression_detected", False)
            sev = f.get("severity", "none").upper()
            metric_name = f.get("metric_name", "")
            thresh = f.get("threshold", 0.0)
            reason = f.get("reason", "")
            if is_reg:
                st.error(
                    f"**[{sev} REGRESSION] {metric_name.replace('_', ' ').title()}** — "
                    f"Configured Threshold: `{thresh}` | Responsible Finding: {reason}"
                )
            else:
                st.success(f"**[OK] {metric_name.replace('_', ' ').title()}** — {reason}")

    # 4. Scenario Breakdown
    st.markdown('<div class="rci-section"><h3>Scenario Breakdown (Success, Steps, Collisions, Reward)</h3></div>', unsafe_allow_html=True)
    sc_results = result.get("scenario_results", {})
    if sc_results:
        rows = []
        for sid, data in sc_results.items():
            b = data.get("baseline", {})
            c = data.get("candidate", {})
            rows.append({
                "Scenario ID": sid,
                "Baseline Success": "YES" if b.get("success") else "NO",
                "Candidate Success": "YES" if c.get("success") else "NO",
                "Baseline Steps": b.get("steps", "-"),
                "Candidate Steps": c.get("steps", "-"),
                "Baseline Collisions": b.get("collisions", 0),
                "Candidate Collisions": c.get("collisions", 0),
                "Baseline Reward": f"{b.get('reward', 0.0):.2f}",
                "Candidate Reward": f"{c.get('reward', 0.0):.2f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_live_telemetry_panel() -> None:
    st.markdown('<div class="rci-section"><h3>Interactive PyBullet Simulation & Real Telemetry</h3></div>', unsafe_allow_html=True)
    st.markdown('<div class="rci-section-note">Executes real PyBullet policy rollouts and displays live telemetry, trajectory charts, and baseline vs candidate comparisons.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        selected_policy = st.selectbox("Select Policy", ["baseline", "candidate"], index=0, key="live_pol")
    with c2:
        selected_scenario = st.selectbox(
            "Select Scenario",
            ["reach_simple", "reach_diagonal", "pick_and_place_easy", "pick_and_place_medium", "pick_and_place_hard"],
            index=0,
            key="live_sc",
        )
    with c3:
        speed = st.slider("Telemetry Speed", 0.1, 5.0, 1.0, 0.1, key="live_spd")

    sm = ScenarioManager()
    scenarios = sm.load_scenarios("scenarios")
    sc_dict = {s.scenario_id: s for s in scenarios}
    scenario = sc_dict.get(selected_scenario, scenarios[0])
    policy = get_policy(selected_policy)

    sim = PyBulletSimulator(gui=False)
    engine = ReplayEngine(sim)

    with st.spinner(f"Simulating rollout: {selected_policy} on {selected_scenario}..."):
        rollout = engine.run(policy, scenario, seed=42)

    df_telemetry = rollout.get_telemetry_dataframe(target_position=scenario.target_state["position"])

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Policy", selected_policy)
    m2.metric("Scenario", selected_scenario)
    succ_label = "SUCCESS" if rollout.success else "FAILED"
    m3.metric("Outcome", succ_label, delta="Pass" if rollout.success else "Fail")
    m4.metric("Steps / Max", f"{rollout.total_steps} / {scenario.max_episode_steps}")
    m5.metric("Collisions / Reward", f"{rollout.collision_count} col | {sum(rollout.rewards):.2f} rwd")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("**Reward vs Simulation Step**")
        st.line_chart(df_telemetry.set_index("Step")[["Reward"]], use_container_width=True)

        st.markdown("**Distance to Target vs Step (meters)**")
        st.line_chart(df_telemetry.set_index("Step")[["Distance_to_Target"]], use_container_width=True)

    with col_g2:
        st.markdown("**End-Effector Position (X, Y, Z) vs Step**")
        st.line_chart(df_telemetry.set_index("Step")[["EE_X", "EE_Y", "EE_Z"]], use_container_width=True)

        st.markdown("**Action Magnitude vs Step**")
        st.line_chart(df_telemetry.set_index("Step")[["Action_Magnitude"]], use_container_width=True)


def render_offline_telemetry_panel(results_dir: str = "results") -> None:
    st.markdown('<div class="rci-section"><h3>Offline Telemetry Analysis</h3></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="rci-section-note">'
        'Inspect offline robot telemetry logs, field coverage, validation warnings, and trajectory deviations. '
        'Operates strictly on recorded data and does not perform live physical actuation.'
        '</div>',
        unsafe_allow_html=True,
    )

    from robot_ci.telemetry.workflow import OfflineReportStore

    telemetry_dir = Path(results_dir) / "telemetry"
    store = OfflineReportStore(telemetry_dir)
    reports = store.load_all()

    if not reports:
        st.info(
            f"No offline telemetry reports found in `{telemetry_dir}/`.\n\n"
            "Run offline telemetry analysis first:\n\n"
            "```powershell\n"
            "python -m robot_ci analyze-telemetry --input tests/fixtures/synthetic_telemetry.json --sim-scenario reach_simple\n"
            "```"
        )
        return

    selected_idx = st.selectbox(
        "Select Offline Report",
        options=list(range(len(reports))),
        format_func=lambda i: f"{reports[i].report_id} - {reports[i].telemetry_source} ({reports[i].created_at[:19]})",
        key="offline_report_sel",
    )
    report = reports[selected_idx]

    st.warning("⚠️ **OFFLINE RECORDED TELEMETRY** — Data origin is recorded/synthetic. Not live robot execution.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Recorded Fields Present:**")
        st.write(", ".join(f"`{f}`" for f in report.available_fields) if report.available_fields else "None")
        if report.missing_fields:
            st.markdown("**Missing Fields (Not Fabricated):**")
            st.write(", ".join(f"`{f}`" for f in report.missing_fields))

    with c2:
        val = report.validation_summary
        if val.get("is_valid", True):
            st.success("✅ **Telemetry Validation: PASSED**")
        else:
            st.error("❌ **Telemetry Validation: ERRORS DETECTED**")
            for err in val.get("errors", []):
                st.write(f"- 🔴 {err}")
        for warn in val.get("warnings", []):
            st.warning(f"- ⚠️ {warn}")

    if report.trajectory_comparison:
        st.markdown(f'<div class="rci-section"><h4>Trajectory Comparison (vs {report.reference_source})</h4></div>', unsafe_allow_html=True)
        tc = report.trajectory_comparison
        tc1, tc2, tc3, tc4 = st.columns(4)
        tc1.metric("RMS Deviation", f"{tc.get('rms_euclidean_distance', 0.0):.4f} m")
        tc2.metric("Mean Deviation", f"{tc.get('mean_euclidean_distance', 0.0):.4f} m")
        tc3.metric("Max Deviation", f"{tc.get('max_euclidean_distance', 0.0):.4f} m")
        tc4.metric("DTW Avg Step Dist", f"{tc.get('dtw_average_distance', 0.0):.4f} m")
        st.caption(f"Alignment method: `{tc.get('alignment_method')}` (Resampled points: {tc.get('resampled_points_count')})")
        st.info(f"ℹ️ *{report.disclaimer}*")


# ===========================================================================
# SECTION B: REAL-TO-SIMULATION
# ===========================================================================

def render_section_b_real_to_sim(results_dir: str = "results") -> None:
    st.markdown('<div class="rci-section"><h2>Section B: Real-to-Simulation Trajectory Comparison</h2></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="rci-section-note">'
        'End-to-end workflow to extract 2D end-effector trajectories from video recordings, '
        'execute actual PyBullet simulation rollouts, align paths via calibrated projection or shape-only matching, '
        'and persist structured audit reports.'
        '</div>',
        unsafe_allow_html=True,
    )

    try:
        from robot_ci.vision import (
            TrackerConfig,
            extract_trajectory_from_video,
            VideoTrajectoryResult,
            CameraCalibration,
            extract_simulated_trajectory,
            compare_video_to_simulation,
            ComparisonMode,
            create_synthetic_camera_calibration,
            is_cv2_available,
            UNCALIBRATED_CAMERA_DISCLAIMER,
        )
    except ImportError as e:
        st.error(f"Failed to load vision module: {e}")
        return

    # Check OpenCV availability
    cv_available = is_cv2_available()
    if not cv_available:
        st.warning(
            "⚠️ OpenCV (`cv2`) is not installed or unavailable in the environment. "
            "You can still inspect pre-extracted trajectory JSON files and existing comparison reports."
        )

    # -----------------------------------------------------------------------
    # Step 1: Video Input & Tracking Settings
    # -----------------------------------------------------------------------
    st.markdown('<div class="rci-section"><h3>Step 1 & 2: Video Selection & Marker Tracking</h3></div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "Trajectory Source",
        [
            "🎥 Upload Video File (.mp4, .mov)",
            "📁 Select Existing Video Fixture",
            "📄 Load Pre-Extracted Trajectory JSON",
        ],
        index=1,
        key="r2s_input_mode",
    )

    video_path = None
    trajectory_json_path = None

    col_inp, col_cfg = st.columns([1, 1])

    with col_inp:
        if input_mode == "🎥 Upload Video File (.mp4, .mov)":
            uploaded = st.file_uploader("Upload Phone Video Recording", type=["mp4", "mov", "avi", "mkv"])
            if uploaded:
                save_dir = Path(results_dir) / "telemetry"
                save_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = save_dir / f"uploaded_{uploaded.name}"
                with open(tmp_path, "wb") as f:
                    f.write(uploaded.read())
                video_path = str(tmp_path)
                st.success(f"Video saved to `{tmp_path}`")
        elif input_mode == "📁 Select Existing Video Fixture":
            default_fixture = "tests/fixtures/synthetic_phone_recording.mp4"
            video_path = st.text_input("Video File Path", value=default_fixture)
            if not Path(video_path).exists():
                st.warning(f"File not found: {video_path}")
                video_path = None
        else:
            default_json = "results/telemetry/synthetic_phone_trajectory.json"
            trajectory_json_path = st.text_input("Pre-Extracted JSON Path", value=default_json)

    with col_cfg:
        color_preset = st.selectbox(
            "Marker Color Preset",
            ["red", "green", "blue", "yellow", "orange", "cyan", "magenta"],
            index=0,
            key="r2s_color_preset",
        )
        min_area = st.slider("Minimum Contour Area (px)", 5.0, 300.0, 10.0, 5.0, key="r2s_min_area")
        apply_morph = st.checkbox("Apply Morphology Filtering", value=True, key="r2s_morph")

    extract_btn = st.button("▶ Extract Trajectory Observations", type="primary", key="r2s_extract_btn")

    # Handle Extraction / Loading
    if extract_btn:
        if trajectory_json_path and Path(trajectory_json_path).exists():
            with st.spinner("Loading pre-extracted trajectory JSON..."):
                try:
                    res = VideoTrajectoryResult.load(trajectory_json_path)
                    st.session_state["extracted_video_traj"] = res
                    st.success(f"Loaded {res.detected_count}/{res.total_frames} detected frames from JSON.")
                except Exception as e:
                    st.error(f"Failed to load trajectory JSON: {e}")
        elif video_path and Path(video_path).exists():
            if not cv_available:
                st.error("Cannot extract from video without OpenCV installed.")
            else:
                with st.spinner("Processing video frames and extracting marker trajectory..."):
                    try:
                        cfg = TrackerConfig(
                            color_preset=color_preset,
                            min_contour_area=min_area,
                            apply_morphology=apply_morph,
                        )
                        res = extract_trajectory_from_video(video_path, config=cfg)
                        st.session_state["extracted_video_traj"] = res
                        st.success(f"Extracted {res.detected_count}/{res.total_frames} frames ({res.detection_rate * 100:.1f}% coverage).")
                    except Exception as e:
                        st.error(f"Extraction failed: {e}")
        else:
            st.error("Please provide a valid video file or trajectory JSON path.")

    # -----------------------------------------------------------------------
    # Step 3 & 4: Trajectory Preview & Missing Observations
    # -----------------------------------------------------------------------
    v_traj: Optional[VideoTrajectoryResult] = st.session_state.get("extracted_video_traj")
    if v_traj is not None:
        st.markdown('<div class="rci-section"><h3>Trajectory Observation Preview</h3></div>', unsafe_allow_html=True)

        # Provenance Banner
        if v_traj.is_synthetic or v_traj.video_metadata.is_synthetic:
            st.info("ℹ️ **DATA PROVENANCE: SYNTHETIC TEST FIXTURE** — Video is an algorithmic test fixture, not real physical hardware.")
        else:
            st.warning("📹 **DATA PROVENANCE: PHYSICAL VIDEO RECORDING** — Uncalibrated 2D monocular tracking.")

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Total Frames", v_traj.total_frames)
        p2.metric("Detected Frames", v_traj.detected_count)
        p3.metric("Lost Frames", v_traj.lost_count)
        p4.metric("Tracking Coverage", f"{v_traj.detection_rate * 100:.1f}%")

        if v_traj.lost_count > 0:
            st.warning(
                f"⚠️ **Missing Observations**: {v_traj.lost_count} frames were lost or occluded. "
                "Missing frames are strictly excluded rather than silently fabricated."
            )

        # Plot 2D Video Path
        detected_obs = [o for o in v_traj.observations if o.detection_status.value == "DETECTED"]
        if detected_obs:
            u_px = [o.image_coordinates[0] for o in detected_obs if o.image_coordinates]
            v_px = [o.image_coordinates[1] for o in detected_obs if o.image_coordinates]
            u_norm = [o.normalized_coordinates[0] for o in detected_obs if o.normalized_coordinates]
            v_norm = [o.normalized_coordinates[1] for o in detected_obs if o.normalized_coordinates]

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown("**2D Image Pixel Trajectory $[u, v]$**")
                fig1, ax1 = plt.subplots(figsize=(5, 3.8))
                ax1.plot(u_px, v_px, color="#A85A4A", lw=2, label="Observed Path")
                ax1.scatter(u_px[0], v_px[0], color="#5B7A5A", s=60, zorder=5, label="Start")
                ax1.scatter(u_px[-1], v_px[-1], color="#A66A4E", s=60, marker="x", zorder=5, label="End")
                ax1.set_xlim(0, v_traj.video_metadata.width)
                ax1.set_ylim(v_traj.video_metadata.height, 0)  # Inverted image Y
                ax1.set_xlabel("u (pixels)")
                ax1.set_ylabel("v (pixels)")
                ax1.legend(loc="upper right", fontsize=8)
                ax1.grid(True, linestyle="--", alpha=0.5)
                st.pyplot(fig1)
                plt.close(fig1)

            with col_p2:
                st.markdown("**Normalized Screen Space $[u_{norm}, v_{norm}]$**")
                fig2, ax2 = plt.subplots(figsize=(5, 3.8))
                ax2.plot(u_norm, v_norm, color="#5B7A5A", lw=2, label="Normalized Path")
                ax2.scatter(u_norm[0], v_norm[0], color="#5B7A5A", s=60, zorder=5, label="Start")
                ax2.scatter(u_norm[-1], v_norm[-1], color="#A66A4E", s=60, marker="x", zorder=5, label="End")
                ax2.set_xlim(0, 1)
                ax2.set_ylim(1, 0)  # Inverted image Y
                ax2.set_xlabel("u_norm [0..1]")
                ax2.set_ylabel("v_norm [0..1]")
                ax2.legend(loc="upper right", fontsize=8)
                ax2.grid(True, linestyle="--", alpha=0.5)
                st.pyplot(fig2)
                plt.close(fig2)

    # -----------------------------------------------------------------------
    # Step 5 & 6 & 7: Simulation Rollout Setup & Comparison Mode
    # -----------------------------------------------------------------------
    st.markdown('<div class="rci-section"><h3>Step 5–7: Simulation Rollout & Comparison Mode</h3></div>', unsafe_allow_html=True)

    col_sim1, col_sim2 = st.columns(2)

    with col_sim1:
        sim_scenario = st.selectbox(
            "Simulation Scenario",
            ["reach_simple", "reach_diagonal", "pick_and_place_easy", "pick_and_place_medium", "pick_and_place_hard"],
            index=0,
            key="r2s_sim_sc",
        )
        sim_policy = st.selectbox("Simulation Policy", ["baseline", "candidate"], index=0, key="r2s_sim_pol")
        sim_backend = st.selectbox("Simulator Backend", ["pybullet", "mock"], index=0, key="r2s_sim_backend")
        sim_seed = st.number_input("Simulation Seed", value=42, step=1, key="r2s_sim_seed")

    with col_sim2:
        phys_controller = st.text_input("Physical Controller Identifier", value="human_teleop", key="r2s_controller")
        robot_model = st.text_input("Robot Model (if known)", value="ur5e", key="r2s_robot_model")
        comp_mode_choice = st.radio(
            "Comparison Mode",
            [
                "Mode B: Shape-Only 2D Comparison (Uncalibrated Video)",
                "Mode A: Calibrated Projection Mode (Camera Calibration Required)",
            ],
            index=0,
            key="r2s_mode_choice",
        )

    # Calibration parameters if Mode A
    calibration_obj = None
    if comp_mode_choice.startswith("Mode A"):
        st.markdown("**Camera Calibration Configuration (Mode A)**")
        calib_source = st.radio(
            "Calibration Source",
            ["Use Synthetic Side-View Calibration Fixture", "Upload Custom Calibration JSON"],
            index=0,
            key="r2s_calib_src",
        )
        if calib_source == "Use Synthetic Side-View Calibration Fixture":
            st.info("ℹ️ Using standard synthetic camera calibration fixture (`tests/fixtures/synthetic_camera_calibration.json`).")
            fixture_calib_path = Path("tests/fixtures/synthetic_camera_calibration.json")
            if fixture_calib_path.exists():
                calibration_obj = CameraCalibration.load(fixture_calib_path)
            else:
                calibration_obj = create_synthetic_camera_calibration()
        else:
            calib_file = st.file_uploader("Upload Calibration JSON", type=["json"], key="r2s_calib_upload")
            if calib_file:
                try:
                    calib_dict = json.load(calib_file)
                    calibration_obj = CameraCalibration.from_dict(calib_dict)
                    st.success("Loaded valid camera calibration.")
                except Exception as e:
                    st.error(f"Invalid camera calibration: {e}")
    else:
        shape_plane = st.selectbox("Shape Projection Plane", ["xy (Horizontal)", "xz (Frontal)", "yz (Side)"], index=0, key="r2s_plane")
        shape_plane_key = shape_plane[:2]
        st.info("ℹ️ Mode B normalizes both trajectories to screen space $[0, 1]$ and performs optimal 2D Procrustes rigid alignment. Metrics are dimensionless.")

    # -----------------------------------------------------------------------
    # Step 8–10: Run Comparison & Inspect Report
    # -----------------------------------------------------------------------
    run_comp_btn = st.button("⚡ Run Simulation Rollout & Compare Trajectories", type="primary", key="r2s_run_comp")

    if run_comp_btn:
        if v_traj is None:
            st.error("Please extract or load a video trajectory first in Step 1 & 2.")
        else:
            with st.spinner(f"Executing rollout in {sim_backend} and computing trajectory comparison..."):
                try:
                    # 1. Run actual simulation rollout
                    sim_traj = extract_simulated_trajectory(
                        policy_name=sim_policy,
                        scenario_id=sim_scenario,
                        simulator_name=sim_backend,
                        seed=int(sim_seed),
                    )
                    st.session_state["sim_trajectory"] = sim_traj

                    # 2. Run comparison
                    effective_mode = ComparisonMode.CALIBRATED_PROJECTION if comp_mode_choice.startswith("Mode A") else ComparisonMode.SHAPE_ONLY_2D
                    report = compare_video_to_simulation(
                        video_trajectory=v_traj,
                        sim_trajectory=sim_traj,
                        calibration=calibration_obj,
                        mode=effective_mode,
                        physical_controller_id=phys_controller,
                        robot_model=robot_model,
                        shape_projection_plane=shape_plane_key if comp_mode_choice.startswith("Mode B") else "xy",
                    )
                    st.session_state["comparison_report"] = report
                    st.success(f"Trajectory comparison complete! (Report ID: {report.report_id})")
                except Exception as e:
                    st.error(f"Comparison failed: {e}")

    # Display comparison results if available
    comp_rep = st.session_state.get("comparison_report")
    if comp_rep is not None:
        st.markdown('<div class="rci-section"><h3>Comparison Results & Metrics</h3></div>', unsafe_allow_html=True)

        # Overview banner
        mode_str = comp_rep.comparison_mode.value if hasattr(comp_rep.comparison_mode, "value") else str(comp_rep.comparison_mode)
        st.markdown(
            f"""
<div class="rci-panel">
  <div class="rci-meta-grid">
    <div class="rci-meta-item">
      <div class="rci-meta-label">Report ID</div>
      <div class="rci-meta-value">{comp_rep.report_id}</div>
    </div>
    <div class="rci-meta-item">
      <div class="rci-meta-label">Comparison Mode</div>
      <div class="rci-meta-value">{mode_str}</div>
    </div>
    <div class="rci-meta-item">
      <div class="rci-meta-label">Comparison Type</div>
      <div class="rci-meta-value">{comp_rep.comparison_type}</div>
    </div>
    <div class="rci-meta-item">
      <div class="rci-meta-label">Coordinate System</div>
      <div class="rci-meta-value">{comp_rep.coordinate_system}</div>
    </div>
    <div class="rci-meta-item">
      <div class="rci-meta-label">Measurement Units</div>
      <div class="rci-meta-value">{comp_rep.units}</div>
    </div>
    <div class="rci-meta-item">
      <div class="rci-meta-label">Physical Controller</div>
      <div class="rci-meta-value">{comp_rep.physical_controller_id}</div>
    </div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

        # Trajectory Comparison Overlay Plot
        st.markdown("#### Trajectory Alignment Visualization")
        m_fig, m_ax = plt.subplots(figsize=(7, 4.2))

        # Check alignment details
        align_det = comp_rep.alignment_details
        unit_label = comp_rep.units

        if mode_str == "CALIBRATED_PROJECTION":
            # Plot in pixel space
            v_obs = [o for o in v_traj.observations if o.detection_status.value == "DETECTED" and o.image_coordinates]
            u_v = [o.image_coordinates[0] for o in v_obs]
            v_v = [o.image_coordinates[1] for o in v_obs]
            m_ax.plot(u_v, v_v, color="#A85A4A", lw=2, label="Video Observed Path (px)")
            m_ax.scatter(u_v[0], v_v[0], color="#A85A4A", s=50, zorder=5)

            # Simulated points projected
            sim_traj_obj = st.session_state.get("sim_trajectory")
            if sim_traj_obj is not None and calibration_obj is not None:
                sim_px, _, valid_m = calibration_obj.project_trajectory(sim_traj_obj.ee_positions)
                sim_vis_px = sim_px[valid_m]
                m_ax.plot(sim_vis_px[:, 0], sim_vis_px[:, 1], color="#5B7A5A", lw=2, linestyle="--", label="Simulated Projected Path (px)")
                m_ax.scatter(sim_vis_px[0, 0], sim_vis_px[0, 1], color="#5B7A5A", s=50, zorder=5)

            m_ax.set_xlim(0, v_traj.video_metadata.width)
            m_ax.set_ylim(v_traj.video_metadata.height, 0)
            m_ax.set_xlabel("Camera u (pixels)")
            m_ax.set_ylabel("Camera v (pixels)")
        else:
            # Shape-only mode: normalized space
            v_obs = [o for o in v_traj.observations if o.detection_status.value == "DETECTED" and o.normalized_coordinates]
            u_v = [o.normalized_coordinates[0] for o in v_obs]
            v_v = [o.normalized_coordinates[1] for o in v_obs]
            m_ax.plot(u_v, v_v, color="#A85A4A", lw=2, label="Video Path (Normalized Screen)")

            sim_traj_obj = st.session_state.get("sim_trajectory")
            if sim_traj_obj is not None:
                ee = sim_traj_obj.ee_positions
                sim_raw = ee[:, [0, 1]]
                s_min = np.min(sim_raw, axis=0)
                s_max = np.max(sim_raw, axis=0)
                span = np.where((s_max - s_min) > 1e-6, s_max - s_min, 1.0)
                sim_norm = (sim_raw - s_min) / span
                m_ax.plot(sim_norm[:, 0], sim_norm[:, 1], color="#5B7A5A", lw=2, linestyle="--", label="Sim Path (Normalized)")

            m_ax.set_xlabel("Normalized Progress X [0..1]")
            m_ax.set_ylabel("Normalized Progress Y [0..1]")

        m_ax.legend(loc="upper right", fontsize=8)
        m_ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(m_fig)
        plt.close(m_fig)

        # Quantitative Metrics Grid
        m_dict = comp_rep.metrics
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        c_m1.metric("RMS Deviation", f"{m_dict.get('rms_trajectory_deviation')} {unit_label}")
        c_m2.metric("Mean Deviation", f"{m_dict.get('mean_trajectory_deviation')} {unit_label}")
        c_m3.metric("Max Deviation", f"{m_dict.get('max_trajectory_deviation')} {unit_label}")
        c_m4.metric("DTW Step Distance", f"{m_dict.get('dtw_average_step_distance')} {unit_label}")

        c_m5, c_m6, c_m7, c_m8 = st.columns(4)
        c_m5.metric("Video Path Length", f"{m_dict.get('video_path_length')} {unit_label}")
        c_m6.metric("Sim Path Length", f"{m_dict.get('sim_path_length')} {unit_label}")
        c_m7.metric("Video Duration", f"{m_dict.get('video_duration_seconds')}s")
        c_m8.metric("Sim Duration", f"{m_dict.get('sim_duration_seconds')}s")

        # Warnings and Disclaimers
        if comp_rep.warnings:
            st.markdown("#### Warnings")
            for w in comp_rep.warnings:
                st.warning(f"⚠️ {w}")

        st.markdown("#### Disclaimers & Limitations")
        for d in comp_rep.disclaimers:
            st.info(f"ℹ️ {d}")

        with st.expander("🔬 Alignment Details & Mathematical Transformation", expanded=False):
            st.json(comp_rep.alignment_details)

        with st.expander("🚫 Unavailable Hardware Metrics (Cannot be inferred from video alone)", expanded=False):
            for u in comp_rep.unavailable_metrics:
                st.markdown(f"- 🔒 `{u}`")

        # Save Report
        col_sv1, col_sv2 = st.columns([1, 2])
        with col_sv1:
            if st.button("💾 Persist Report to Disk", key="r2s_save_btn"):
                saved_target = comp_rep.save()
                st.success(f"Report saved to `{saved_target}`")

        with col_sv2:
            st.download_button(
                "⬇️ Download Report JSON",
                data=comp_rep.to_json(),
                file_name=f"comparison_{comp_rep.report_id}.json",
                mime="application/json",
                key="r2s_download_btn",
            )

    # -----------------------------------------------------------------------
    # Previously Saved Reports History
    # -----------------------------------------------------------------------
    st.markdown('<div class="rci-section"><h3>Saved Video-Sim Comparison Reports History</h3></div>', unsafe_allow_html=True)
    past_reports = load_comparison_reports(f"{results_dir}/video_sim_comparisons")
    if past_reports:
        sel_hist_idx = st.selectbox(
            "Inspect Saved Comparison Report",
            options=list(range(len(past_reports))),
            format_func=lambda i: f"Report {past_reports[i].get('report_id')} — {past_reports[i].get('comparison_mode')} ({past_reports[i].get('scenario_id')}) [{past_reports[i].get('created_at', '')[:19]}]",
            key="r2s_hist_sel",
        )
        hist_rep = past_reports[sel_hist_idx]
        with st.expander(f"Inspect Raw Report JSON ({hist_rep.get('report_id')})", expanded=False):
            st.json(hist_rep)
    else:
        st.caption("No previously saved comparison reports in `results/video_sim_comparisons/` yet.")


# ===========================================================================
# SECTION C: EVIDENCE-BASED DIAGNOSIS
# ===========================================================================

def render_section_c_diagnosis(results_dir: str = "results") -> None:
    st.markdown('<div class="rci-section"><h2>Section C: Evidence-Based Diagnosis</h2></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="rci-section-note">'
        'Combines simulation policy regression benchmarks and real-to-simulation video trajectory evidence '
        'to evaluate whether failures stem from algorithmic policy regressions, sim-to-real domain gaps, '
        'or insufficient evidence without fabricating causality.'
        '</div>',
        unsafe_allow_html=True,
    )

    try:
        from robot_ci.diagnosis import (
            DiagnosisCategory,
            DiagnosisReport,
            evaluate_diagnosis,
            DIAGNOSIS_LIMITATIONS_DISCLAIMER,
        )
    except ImportError as e:
        st.error(f"Failed to load diagnosis module: {e}")
        return

    # 1. Select Evidence Inputs
    st.markdown('<div class="rci-section"><h3>Select Evidence Sources</h3></div>', unsafe_allow_html=True)

    ci_runs = load_results(results_dir)
    video_reports = load_comparison_reports(f"{results_dir}/video_sim_comparisons")

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        st.markdown("**1. Simulation Policy Regression Evidence**")
        ci_options = ["None / Not Available"] + [f"Run {r.get('run_id')} ({_fmt_ts(r.get('timestamp'))}) — {r.get('overall_status')}" for r in ci_runs]
        selected_ci_idx = st.selectbox("Select CI Regression Report", options=list(range(len(ci_options))), format_func=lambda i: ci_options[i], key="diag_ci_sel")
        selected_ci_data = ci_runs[selected_ci_idx - 1] if selected_ci_idx > 0 else None

    with col_e2:
        st.markdown("**2. Real-to-Simulation Video Comparison Evidence**")
        v_options = ["None / Not Available"]
        # Add session state report if available
        curr_rep = st.session_state.get("comparison_report")
        if curr_rep is not None:
            v_options.append(f"Current Session: {curr_rep.report_id} ({curr_rep.comparison_mode.value})")
        for vr in video_reports:
            v_options.append(f"Saved: {vr.get('report_id')} — {vr.get('comparison_mode')} ({vr.get('scenario_id')})")

        selected_v_idx = st.selectbox("Select Video Comparison Report", options=list(range(len(v_options))), format_func=lambda i: v_options[i], key="diag_v_sel")

        selected_v_data = None
        if selected_v_idx > 0:
            if curr_rep is not None and selected_v_idx == 1:
                selected_v_data = curr_rep
            else:
                offset = 2 if curr_rep is not None else 1
                v_file_idx = selected_v_idx - offset
                if 0 <= v_file_idx < len(video_reports):
                    selected_v_data = video_reports[v_file_idx]

    # Advanced Thresholds
    with st.expander("⚙️ Advanced Evidence Threshold Settings", expanded=False):
        c_th1, c_th2, c_th3 = st.columns(3)
        th_pixel = c_th1.number_input("Calibrated Mode Reprojection Threshold (px)", value=30.0, step=5.0)
        th_shape = c_th2.number_input("Shape-Only Mode Normalized Threshold", value=0.25, step=0.05)
        th_cov = c_th3.number_input("Minimum Tracking Coverage Ratio", value=0.50, step=0.05)

    eval_diag_btn = st.button("🔍 Evaluate Combined Evidence & Diagnose", type="primary", key="diag_eval_btn")

    if eval_diag_btn:
        with st.spinner("Analyzing multi-source evidence and applying diagnosis rules..."):
            diag_rep = evaluate_diagnosis(
                ci_result=selected_ci_data,
                video_report=selected_v_data,
                calibrated_pixel_threshold=float(th_pixel),
                shape_normalized_threshold=float(th_shape),
                min_tracking_coverage=float(th_cov),
            )
            st.session_state["active_diagnosis_report"] = diag_rep

    diag: Optional[DiagnosisReport] = st.session_state.get("active_diagnosis_report")
    if diag is not None:
        st.markdown('<div class="rci-section"><h3>Evidence-Based Diagnosis Assessment</h3></div>', unsafe_allow_html=True)

        # Category alert banner styling
        cat_str = diag.category.value if hasattr(diag.category, "value") else str(diag.category)
        if cat_str == "NO_REGRESSION_DETECTED":
            alert_cls = "rci-alert-pass"
            cat_label = "✅ NO REGRESSION DETECTED"
        elif cat_str in ("SIM_TO_REAL_MISMATCH_LIKELY", "BOTH_ISSUES_LIKELY"):
            alert_cls = "rci-alert-fail"
            cat_label = f"🚨 {cat_str.replace('_', ' ')}"
        elif cat_str == "POLICY_REGRESSION_LIKELY":
            alert_cls = "rci-alert-fail"
            cat_label = "⚠️ POLICY REGRESSION LIKELY (SIMULATION BENCHMARKS)"
        else:
            alert_cls = "rci-panel"
            cat_label = "ℹ️ INCONCLUSIVE — INSUFFICIENT OR SYNTHETIC EVIDENCE"

        st.markdown(f'<div class="rci-alert {alert_cls}" style="font-size:1.05rem; text-align:center;">{cat_label}</div>', unsafe_allow_html=True)

        # Plain-English Summary & Recommended Next Step
        col_sum, col_rec = st.columns(2)
        with col_sum:
            st.markdown("#### Assessment Explanation")
            st.info(diag.summary_explanation)

        with col_rec:
            st.markdown("#### Recommended Next Validation Step")
            st.success(diag.recommended_next_step)

        # Evidence Sources Grid
        st.markdown("#### Evidence Sources Used & Provenance")
        if diag.evidence_sources:
            cols_src = st.columns(len(diag.evidence_sources))
            for i, es in enumerate(diag.evidence_sources):
                with cols_src[i]:
                    st.markdown(
                        f"""
<div class="rci-panel">
  <div class="rci-meta-label">{es.source_type}</div>
  <div class="rci-meta-value">{es.identifier}</div>
  <div style="font-size:0.75rem; color:{COLORS['text_muted']}; margin-top:0.35rem;">Provenance: <b>{es.provenance}</b></div>
  <div style="font-size:0.75rem; color:{COLORS['text_muted']};">Hardware: <b>{'Physical Robot' if es.is_physical_hardware else 'Synthetic / Sim'}</b></div>
</div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("No evidence sources were provided.")

        # Findings & Warnings Breakdown
        col_f1, col_f2 = st.columns(2)

        with col_f1:
            st.markdown("#### Supporting Findings")
            if diag.supporting_findings:
                for sf in diag.supporting_findings:
                    st.markdown(f"- 🟢 {sf}")
            else:
                st.caption("No positive supporting findings.")

            if diag.missing_evidence:
                st.markdown("#### Missing Evidence")
                for me in diag.missing_evidence:
                    st.markdown(f"- ❓ {me}")

        with col_f2:
            st.markdown("#### Data Quality & Provenance Warnings")
            if diag.data_quality_warnings:
                for w in diag.data_quality_warnings:
                    st.warning(f"⚠️ {w}")
            else:
                st.caption("No data-quality warnings reported.")

            st.markdown("#### Unsupported Conclusions (Cannot Be Claimed)")
            for uc in diag.unsupported_conclusions:
                st.markdown(f"- 🚫 {uc}")

        # Limitations Disclaimer
        st.markdown("#### Fundamental Measurement Limitations")
        st.caption(DIAGNOSIS_LIMITATIONS_DISCLAIMER)


# ===========================================================================
# MAIN APPLICATION
# ===========================================================================

def main():
    results_dir = "results"
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        extra_args = sys.argv[idx + 1:]
        parser = argparse.ArgumentParser()
        parser.add_argument("--results-dir", default="results")
        parsed, _ = parser.parse_known_args(extra_args)
        results_dir = parsed.results_dir

    st.set_page_config(
        page_title="Robot CI - Policy Regression & Sim-to-Real Dashboard",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme()
    render_header()

    st.sidebar.markdown('<div class="rci-sidebar-brand">Navigation</div>', unsafe_allow_html=True)
    app_section = st.sidebar.radio(
        "Select Platform Section",
        [
            "📈 Section A: Policy Regression",
            "🎥 Section B: Real-to-Simulation",
            "🩺 Section C: Evidence-Based Diagnosis",
        ],
        index=0,
        key="app_main_section",
    )

    if app_section == "📈 Section A: Policy Regression":
        render_section_a_policy_regression(results_dir)
    elif app_section == "🎥 Section B: Real-to-Simulation":
        render_section_b_real_to_sim(results_dir)
    else:
        render_section_c_diagnosis(results_dir)


if __name__ == "__main__":
    main()
