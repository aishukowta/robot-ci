"""Robot CI - Streamlit Results & Interactive Telemetry Dashboard.

Displays:
- Live PyBullet Telemetry & Interactive Controls (Play/Pause, Step, Reset, Speed, Policy/Scenario selection)
- Telemetry Graphs (Reward vs Step, EE X/Y/Z vs Step, Distance to Target vs Step, Action Magnitude vs Step)
- Single Scenario Baseline vs Candidate Comparison View
- Overall 5-Scenario Comparison Matrix
- CI Test Run Reports & Regression Findings
"""

import sys
import json
import argparse
from pathlib import Path
import html

import numpy as np
import pandas as pd
import streamlit as st

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
        max-width: 42rem;
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

    .rci-metric-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 0.7rem;
        margin-bottom: 0.5rem;
    }}

    .rci-metric-card {{
        background: {c["bg_elevated"]};
        border: 1px solid {c["border"]};
        border-radius: 4px;
        padding: 0.75rem 0.85rem;
    }}

    .rci-metric-name {{
        font-size: 0.78rem;
        font-weight: 600;
        color: {c["text"]};
        margin-bottom: 0.55rem;
    }}

    .rci-metric-row {{
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        font-size: 0.78rem;
        margin-bottom: 0.2rem;
    }}

    .rci-baseline {{ color: {c["baseline"]}; font-family: "IBM Plex Mono", monospace; }}
    .rci-candidate {{ color: {c["candidate"]}; font-family: "IBM Plex Mono", monospace; }}
    .rci-delta {{
        margin-top: 0.35rem;
        font-size: 0.72rem;
        color: {c["text_muted"]};
        font-family: "IBM Plex Mono", monospace;
    }}

    .rci-badge {{
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        padding: 0.15rem 0.45rem;
        border-radius: 2px;
        border: 1px solid;
        text-transform: uppercase;
    }}

    .rci-badge-pass {{ color: {c["pass"]}; background: {c["pass_bg"]}; border-color: {c["pass"]}; }}
    .rci-badge-fail {{ color: {c["fail"]}; background: {c["critical_bg"]}; border-color: {c["fail"]}; }}
    .rci-badge-critical {{ color: {c["fail"]}; background: {c["critical_bg"]}; border-color: {c["fail"]}; }}
    .rci-badge-none {{ color: {c["text_muted"]}; background: {c["bg"]}; border-color: {c["border"]}; }}
    .rci-badge-warn {{ color: {c["warn"]}; background: {c["warn_bg"]}; border-color: {c["warn"]}; }}

    .rci-sidebar-brand {{
        font-size: 1.05rem;
        font-weight: 700;
        color: {c["text"]};
        margin: 0.2rem 0 0.85rem 0;
        padding-bottom: 0.65rem;
        border-bottom: 1px solid {c["border"]};
    }}

    /* Quiet Streamlit chrome */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
</style>
        """,
        unsafe_allow_html=True,
    )


def load_results(results_dir: str = "results") -> list:
    path = Path(results_dir)
    if not path.exists():
        return []

    files = sorted(path.glob("run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    results = []
    for f in files:
        try:
            with open(f) as fp:
                results.append(json.load(fp))
        except Exception:
            pass
    return results


def render_header() -> None:
    st.markdown(
        """
<div class="rci-header">
  <div class="rci-brand">ROBOT CI</div>
  <div class="rci-title">Robotic Policy Regression Testing & Simulation Telemetry</div>
  <div class="rci-subtitle">Interactive PyBullet simulation viewer, live telemetry monitoring, and automated CI regression reports.</div>
</div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Interactive Live PyBullet Telemetry Panel
# ---------------------------------------------------------------------------

def render_live_telemetry_panel() -> None:
    st.markdown('<div class="rci-section"><h2>Interactive PyBullet Simulation & Real Telemetry</h2></div>', unsafe_allow_html=True)
    st.markdown('<div class="rci-section-note">Executes real PyBullet policy rollouts and displays live telemetry, trajectory charts, and baseline vs candidate comparisons.</div>', unsafe_allow_html=True)

    # 1. Sidebar Controls
    st.sidebar.markdown('<div class="rci-sidebar-brand">Simulation Controls</div>', unsafe_allow_html=True)
    selected_policy = st.sidebar.radio("Select Policy", ["baseline", "candidate"], index=0)
    selected_scenario = st.sidebar.selectbox(
        "Select Scenario",
        ["reach_simple", "reach_diagonal", "pick_and_place_easy", "pick_and_place_medium", "pick_and_place_hard"],
        index=0,
    )
    speed = st.sidebar.slider("Playback / Telemetry Speed", 0.1, 5.0, 1.0, 0.1)

    c1, c2 = st.sidebar.columns(2)
    run_btn = c1.button("🚀 Run Rollout", use_container_width=True)
    reset_btn = c2.button("↺ Reset", use_container_width=True)

    sm = ScenarioManager()
    scenarios = sm.load_scenarios("scenarios")
    sc_dict = {s.scenario_id: s for s in scenarios}
    scenario = sc_dict.get(selected_scenario, scenarios[0])

    policy = get_policy(selected_policy)

    # Run PyBullet rollout to get real simulation telemetry
    sim = PyBulletSimulator(gui=False)
    engine = ReplayEngine(sim)

    with st.spinner(f"Simulating rollout: {selected_policy} on {selected_scenario}..."):
        rollout = engine.run(policy, scenario, seed=42)

    df_telemetry = rollout.get_telemetry_dataframe(target_position=scenario.target_state["position"])

    # 2. Live Metrics Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Policy", selected_policy)
    m2.metric("Scenario", selected_scenario)
    succ_label = "SUCCESS" if rollout.success else "FAILED"
    m3.metric("Outcome", succ_label, delta="Pass" if rollout.success else "Fail")
    m4.metric("Steps / Max", f"{rollout.total_steps} / {scenario.max_episode_steps}")
    m5.metric("Collisions / Reward", f"{rollout.collision_count} col | {sum(rollout.rewards):.2f} rwd")

    # 3. Telemetry Graphs (4 charts)
    st.markdown('<div class="rci-section"><h3>Simulation Telemetry Graphs</h3></div>', unsafe_allow_html=True)

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("**1. Reward vs Simulation Step**")
        st.line_chart(df_telemetry.set_index("Step")[["Reward"]], use_container_width=True)

        st.markdown("**3. Distance to Target vs Step (meters)**")
        st.line_chart(df_telemetry.set_index("Step")[["Distance_to_Target"]], use_container_width=True)

    with col_g2:
        st.markdown("**2. End-Effector Position (X, Y, Z) vs Step**")
        st.line_chart(df_telemetry.set_index("Step")[["EE_X", "EE_Y", "EE_Z"]], use_container_width=True)

        st.markdown("**4. Action Magnitude vs Step**")
        st.line_chart(df_telemetry.set_index("Step")[["Action_Magnitude"]], use_container_width=True)

    # 4. Comparison View (Baseline vs Candidate for Selected Scenario)
    st.markdown(f'<div class="rci-section"><h3>Scenario Comparison: Baseline vs Candidate ({selected_scenario})</h3></div>', unsafe_allow_html=True)

    b_pol = get_policy("baseline")
    c_pol = get_policy("candidate")
    b_rollout = engine.run(b_pol, scenario, seed=42)
    c_rollout = engine.run(c_pol, scenario, seed=42)

    b_df = b_rollout.get_telemetry_dataframe(scenario.target_state["position"])
    c_df = c_rollout.get_telemetry_dataframe(scenario.target_state["position"])

    cb1, cb2, cb3, cb4 = st.columns(4)
    cb1.metric("Baseline Outcome", "SUCCESS" if b_rollout.success else "FAILED", f"{b_rollout.total_steps} steps")
    cb2.metric("Candidate Outcome", "SUCCESS" if c_rollout.success else "FAILED", f"{c_rollout.total_steps} steps")
    cb3.metric("Collisions (Base vs Cand)", f"{b_rollout.collision_count} vs {c_rollout.collision_count}")
    cb4.metric("Reward (Base vs Cand)", f"{sum(b_rollout.rewards):.2f} vs {sum(c_rollout.rewards):.2f}")

    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        st.markdown("**Distance to Target Comparison (Baseline vs Candidate)**")
        df_comp_dist = pd.DataFrame({
            "Baseline Distance": b_df.set_index("Step")["Distance_to_Target"],
            "Candidate Distance": c_df.set_index("Step")["Distance_to_Target"],
        })
        st.line_chart(df_comp_dist, use_container_width=True)

    with comp_col2:
        st.markdown("**Step Reward Comparison (Baseline vs Candidate)**")
        df_comp_rwd = pd.DataFrame({
            "Baseline Reward": b_df.set_index("Step")["Reward"],
            "Candidate Reward": c_df.set_index("Step")["Reward"],
        })
        st.line_chart(df_comp_rwd, use_container_width=True)

    # 5. Overall Scenario Comparison (All 5 Scenarios)
    st.markdown('<div class="rci-section"><h3>Overall Scenario Comparison (All 5 Scenarios)</h3></div>', unsafe_allow_html=True)

    all_b = engine.run_batch(b_pol, scenarios, global_seed=42)
    all_c = engine.run_batch(c_pol, scenarios, global_seed=42)

    rows = []
    for sc in scenarios:
        sid = sc.scenario_id
        br = all_b[sid]
        cr = all_c[sid]
        rows.append({
            "Scenario": sid,
            "Task": sc.task_name,
            "Baseline": "SUCCESS" if br.success else "FAILED",
            "B-Steps": br.total_steps,
            "B-Collisions": br.collision_count,
            "B-Reward": f"{sum(br.rewards):.2f}",
            "Candidate": "SUCCESS" if cr.success else "FAILED",
            "C-Steps": cr.total_steps,
            "C-Collisions": cr.collision_count,
            "C-Reward": f"{sum(cr.rewards):.2f}",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# CI Test Run Reports & Regressions Section
# ---------------------------------------------------------------------------

def _fmt_ts(ts: str) -> str:
    if not ts:
        return "-"
    return ts.replace("T", " ")[:19]


def _is_regression(status: str) -> bool:
    return status.upper() in ("REGRESSION_DETECTED", "FAIL", "FAILED")


def render_latest_run(result: dict) -> None:
    status = result.get("overall_status", "UNKNOWN")
    reg_detected = _is_regression(status)
    css_cls = "rci-alert-fail" if reg_detected else "rci-alert-pass"
    status_str = "REGRESSION DETECTED" if reg_detected else "PASS — ALL METRICS WITHIN THRESHOLDS"

    st.markdown(f'<div class="rci-alert {css_cls}">OVERALL STATUS: {status_str}</div>', unsafe_allow_html=True)


def render_run_information(result: dict) -> None:
    st.markdown('<div class="rci-section"><h2>Run Information</h2></div>', unsafe_allow_html=True)
    status = result.get("overall_status", "UNKNOWN")
    status_label = "REGRESSION DETECTED" if _is_regression(status) else "PASS"

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
      <div class="rci-meta-label">Simulator</div>
      <div class="rci-meta-value">{result.get("simulator_name", "pybullet")}</div>
    </div>
    <div class="rci-meta-item">
      <div class="rci-meta-label">Overall Status</div>
      <div class="rci-meta-value">{status_label}</div>
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        page_title="Robot CI",
        page_icon="R",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme()
    render_header()

    st.sidebar.markdown('<div class="rci-sidebar-brand">Robot CI Dashboard</div>', unsafe_allow_html=True)
    mode = st.sidebar.radio(
        "View Mode",
        ["📊 Interactive Telemetry & Control", "📋 CI Run Reports & Regressions"],
        index=0,
    )

    if mode == "📊 Interactive Telemetry & Control":
        render_live_telemetry_panel()
    else:
        results = load_results(results_dir)
        if not results:
            st.warning(
                f"No stored results found in `{results_dir}/`. "
                "Run the automated pipeline first:\n\n"
                "```\npython -m robot_ci run --baseline baseline --candidate candidate --simulator pybullet\n```"
            )
            return

        selected_run = st.sidebar.selectbox(
            "Select Run Report",
            options=list(range(len(results))),
            format_func=lambda i: f"{results[i].get('run_id')} ({_fmt_ts(results[i].get('timestamp'))})",
        )
        result = results[selected_run]
        render_latest_run(result)
        render_run_information(result)


if __name__ == "__main__":
    main()
