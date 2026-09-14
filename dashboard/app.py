"""Robot CI - Streamlit Results Dashboard.

Reads actual stored results from the results/ directory and displays:
- Run metadata (policies, timestamp, overall status)
- Side-by-side metric comparisons
- Per-scenario results table with pass/fail and regression severity
- Regression details with thresholds
- Historical runs
"""

import sys
import json
import argparse
from pathlib import Path
import html

import streamlit as st


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

    .rci-bar-track {{
        height: 6px;
        background: #E8DFD2;
        border-radius: 2px;
        margin-top: 0.15rem;
        margin-bottom: 0.35rem;
        overflow: hidden;
    }}

    .rci-bar-fill-b {{ height: 100%; background: {c["baseline"]}; }}
    .rci-bar-fill-c {{ height: 100%; background: {c["candidate"]}; }}

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

    .rci-badge-pass {{
        color: {c["pass"]};
        background: {c["pass_bg"]};
        border-color: {c["pass"]};
    }}

    .rci-badge-fail {{
        color: {c["fail"]};
        background: {c["critical_bg"]};
        border-color: {c["fail"]};
    }}

    .rci-badge-critical {{
        color: {c["fail"]};
        background: {c["critical_bg"]};
        border-color: {c["fail"]};
    }}

    .rci-badge-none {{
        color: {c["text_muted"]};
        background: {c["bg"]};
        border-color: {c["border"]};
    }}

    .rci-badge-warn {{
        color: {c["warn"]};
        background: {c["warn_bg"]};
        border-color: {c["warn"]};
    }}

    .rci-sidebar-brand {{
        font-size: 1.05rem;
        font-weight: 700;
        color: {c["text"]};
        margin: 0.2rem 0 0.85rem 0;
        padding-bottom: 0.65rem;
        border-bottom: 1px solid {c["border"]};
    }}

    .rci-nav-item {{
        font-size: 0.86rem;
        padding: 0.28rem 0.45rem;
        margin-bottom: 0.12rem;
        border-radius: 3px;
        color: {c["text_muted"]};
    }}

    .rci-nav-item.active {{
        background: #E4D9C8;
        color: {c["text"]};
        font-weight: 600;
        border-left: 2px solid {c["accent"]};
    }}

    .rci-nav-group {{
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: {c["text_muted"]};
        margin: 1rem 0 0.35rem 0;
    }}

    .rci-legend {{
        display: flex;
        gap: 1rem;
        font-size: 0.75rem;
        color: {c["text_muted"]};
        margin: 0.25rem 0 0.75rem 0;
    }}

    .rci-legend span::before {{
        content: "";
        display: inline-block;
        width: 8px;
        height: 8px;
        margin-right: 0.35rem;
        border-radius: 1px;
        vertical-align: middle;
    }}

    .rci-legend .b::before {{ background: {c["baseline"]}; }}
    .rci-legend .c::before {{ background: {c["candidate"]}; }}

    .rci-table-wrap {{
        overflow-x: auto;
        border: 1px solid {c["border"]};
        border-radius: 4px;
        background: {c["bg_elevated"]};
        margin-bottom: 0.75rem;
    }}

    table.rci-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.8rem;
    }}

    table.rci-table th {{
        text-align: left;
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: {c["text_muted"]};
        background: {c["bg_sidebar"]};
        border-bottom: 1px solid {c["border"]};
        padding: 0.55rem 0.65rem;
        white-space: nowrap;
    }}

    table.rci-table td {{
        border-bottom: 1px solid {c["border"]};
        padding: 0.5rem 0.65rem;
        color: {c["text"]};
        font-family: "IBM Plex Mono", ui-monospace, monospace;
        font-size: 0.78rem;
        vertical-align: middle;
    }}

    table.rci-table tr:last-child td {{
        border-bottom: none;
    }}

    table.rci-table td.plain {{
        font-family: "Source Sans 3", "Segoe UI", sans-serif;
    }}

    .stExpander {{
        border: 1px solid {c["border"]} !important;
        border-radius: 4px !important;
        background: {c["bg_elevated"]} !important;
    }}

    .stExpander details summary p,
    .stExpander details summary span,
    [data-testid="stExpander"] summary p {{
        color: {c["text"]} !important;
        font-weight: 500 !important;
    }}

    /* Quiet Streamlit chrome */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
</style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir: str) -> list:
    """Load all run_*.json result files from the results directory."""
    results_path = Path(results_dir)
    if not results_path.exists():
        return []

    results = []
    for f in sorted(results_path.glob("run_*.json"), reverse=True):
        try:
            with open(f) as fp:
                data = json.load(fp)
                data["_filepath"] = str(f)
                results.append(data)
        except Exception:
            continue

    # Sort newest first
    results.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return results


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_ts(timestamp: str) -> str:
    if not timestamp:
        return "-"
    return timestamp[:19].replace("T", " ")


def _fmt_num(value, digits: int = 4) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "-"
    if abs(v) >= 100:
        return f"{v:.2f}"
    if abs(v) >= 10:
        return f"{v:.3f}"
    return f"{v:.{digits}f}"


def _fmt_delta(delta) -> str:
    try:
        v = float(delta)
    except (TypeError, ValueError):
        return "-"
    sign = "+" if v > 0 else ""
    return f"{sign}{_fmt_num(v)}"


def _metric_label(name: str) -> str:
    return name.replace("_", " ").title()


def _bar_pct(value: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return max(0.0, min(100.0, abs(value) / scale * 100.0))


def _is_regression(status: str) -> bool:
    return status != "PASS"


def _scenario_status(bl: dict, cd: dict) -> str:
    if bl.get("success") and cd.get("success"):
        return "PASS"
    if bl.get("success") and not cd.get("success"):
        return "REGRESSION"
    if not bl.get("success") and cd.get("success"):
        return "IMPROVED"
    return "FAIL"


def _count_regressed_scenarios(result: dict) -> int:
    summary = result.get("rollout_summary") or {}
    count = 0
    for data in summary.values():
        bl = data.get("baseline", {})
        cd = data.get("candidate", {})
        if _scenario_status(bl, cd) == "REGRESSION":
            count += 1
    return count


def _status_badge(label: str) -> str:
    mapping = {
        "PASS": "rci-badge-pass",
        "REGRESSION": "rci-badge-fail",
        "IMPROVED": "rci-badge-pass",
        "FAIL": "rci-badge-warn",
        "YES": "rci-badge-fail",
        "NO": "rci-badge-none",
        "CRITICAL": "rci-badge-critical",
        "NONE": "rci-badge-none",
        "MAJOR": "rci-badge-warn",
        "MODERATE": "rci-badge-warn",
        "MINOR": "rci-badge-warn",
    }
    cls = mapping.get(label.upper(), "rci-badge-none")
    return f'<span class="rci-badge {cls}">{html.escape(label)}</span>'


def _render_html_table(headers: list, rows: list) -> None:
    """Render a light-theme HTML table (presentation only)."""
    ths = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = []
    for row in rows:
        cells = []
        for cell in row:
            if isinstance(cell, dict):
                cells.append(f'<td class="plain">{cell["html"]}</td>')
            else:
                cells.append(f"<td>{html.escape(str(cell))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    st.markdown(
        f'<div class="rci-table-wrap"><table class="rci-table"><thead><tr>{ths}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def render_header() -> None:
    st.markdown(
        """
<div class="rci-header">
  <p class="rci-brand">Robot CI</p>
  <p class="rci-title">Regression Testing Dashboard</p>
  <p class="rci-subtitle">Automated evaluation of robotic manipulation policies in simulation.</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(results: list) -> int:
    st.sidebar.markdown('<div class="rci-sidebar-brand">Robot CI</div>', unsafe_allow_html=True)

    st.sidebar.markdown('<div class="rci-nav-group">Navigation</div>', unsafe_allow_html=True)
    nav_items = [
        ("Dashboard", True),
        ("Test Runs", False),
        ("Scenarios", False),
        ("Policies", False),
        ("Metrics", False),
        ("Reports", False),
    ]
    for label, active in nav_items:
        cls = "rci-nav-item active" if active else "rci-nav-item"
        st.sidebar.markdown(f'<div class="{cls}">{label}</div>', unsafe_allow_html=True)

    st.sidebar.markdown('<div class="rci-nav-group">Test Runs</div>', unsafe_allow_html=True)

    run_labels = []
    for r in results:
        ts = _fmt_ts(r.get("timestamp", "unknown"))
        status = r.get("overall_status", "UNKNOWN")
        mark = "REG" if _is_regression(status) else "OK"
        run_labels.append(f"[{mark}] {ts} · {r.get('run_id', '?')[:8]}")

    selected_idx = st.sidebar.selectbox(
        "Select run",
        range(len(run_labels)),
        format_func=lambda i: run_labels[i],
        label_visibility="collapsed",
    )

    return selected_idx


def render_latest_run(result: dict) -> None:
    st.markdown('<div class="rci-section"><h2>Latest Test Run</h2></div>', unsafe_allow_html=True)

    status = result.get("overall_status", "UNKNOWN")
    n_scenarios = len(result.get("scenario_ids", []))
    simulator_name = result.get("simulator_name", "mock")
    trial_count = result.get("trial_count", 1)

    if _is_regression(status):
        alert = '<div class="rci-alert rci-alert-fail">REGRESSION DETECTED</div>'
    else:
        alert = '<div class="rci-alert rci-alert-pass">PASS - NO REGRESSIONS DETECTED</div>'

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
      <div class="rci-meta-label">Scenarios</div>
      <div class="rci-meta-value">{n_scenarios} x {trial_count}</div>
    </div>
    <div class="rci-meta-item">
      <div class="rci-meta-label">Simulator</div>
      <div class="rci-meta-value">{simulator_name}</div>
    </div>
  </div>
  {alert}
</div>
        """,
        unsafe_allow_html=True,
    )


def render_key_metrics(metric_results: list) -> None:
    st.markdown('<div class="rci-section"><h2>Key Metrics</h2></div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="rci-section-note">Baseline vs candidate aggregate values from the selected run.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="rci-legend"><span class="b">Baseline</span><span class="c">Candidate</span></div>',
        unsafe_allow_html=True,
    )

    if not metric_results:
        st.info("No metric results available.")
        return

    cards = []
    for m in metric_results:
        name = _metric_label(m.get("metric_name", "metric"))
        bv = float(m.get("baseline_value", 0.0))
        cv = float(m.get("candidate_value", 0.0))
        delta = m.get("delta", cv - bv)
        scale = max(abs(bv), abs(cv), 1e-9)
        bp = _bar_pct(bv, scale)
        cp = _bar_pct(cv, scale)
        cards.append(
            f"""
<div class="rci-metric-card">
  <div class="rci-metric-name">{name}</div>
  <div class="rci-metric-row">
    <span>Baseline</span>
    <span class="rci-baseline">{_fmt_num(bv)}</span>
  </div>
  <div class="rci-bar-track"><div class="rci-bar-fill-b" style="width:{bp:.1f}%"></div></div>
  <div class="rci-metric-row">
    <span>Candidate</span>
    <span class="rci-candidate">{_fmt_num(cv)}</span>
  </div>
  <div class="rci-bar-track"><div class="rci-bar-fill-c" style="width:{cp:.1f}%"></div></div>
  <div class="rci-delta">Δ {_fmt_delta(delta)}</div>
</div>
            """
        )

    st.markdown(
        f'<div class="rci-metric-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def render_scenario_details(result: dict) -> None:
    st.markdown('<div class="rci-section"><h2>Per-Scenario Results</h2></div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="rci-section-note">Rollout outcomes for each scenario in the evaluation corpus.</p>',
        unsafe_allow_html=True,
    )

    rollout_summary = result.get("rollout_summary", {})
    if not rollout_summary:
        st.info("No per-scenario rollout summary available.")
        return

    headers = [
        "Scenario",
        "Baseline Success",
        "Candidate Success",
        "Baseline Steps",
        "Candidate Steps",
        "Baseline Reward",
        "Candidate Reward",
        "Status",
    ]
    rows = []
    for sid, data in sorted(rollout_summary.items()):
        bl = data.get("baseline", {})
        cd = data.get("candidate", {})
        status = _scenario_status(bl, cd)
        rows.append([
            sid,
            "Yes" if bl.get("success") else "No",
            "Yes" if cd.get("success") else "No",
            bl.get("steps", "-"),
            cd.get("steps", "-"),
            _fmt_num(bl.get("reward", 0), digits=2),
            _fmt_num(cd.get("reward", 0), digits=2),
            {"html": _status_badge(status)},
        ])

    _render_html_table(headers, rows)


def render_regression_table(findings: list) -> None:
    st.markdown('<div class="rci-section"><h2>Regression Analysis</h2></div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="rci-section-note">'
        "What changed between the baseline and candidate, and why did Robot CI reject the candidate?"
        "</p>",
        unsafe_allow_html=True,
    )

    if not findings:
        st.info("No regression analysis available.")
        return

    headers = [
        "Metric",
        "Baseline",
        "Candidate",
        "Delta",
        "Threshold",
        "Regression",
        "Severity",
    ]
    rows = []
    for f in findings:
        detected = bool(f.get("regression_detected"))
        severity = str(f.get("severity", "none")).upper()
        rows.append([
            _metric_label(f.get("metric_name", "")),
            _fmt_num(f.get("baseline_value")),
            _fmt_num(f.get("candidate_value")),
            _fmt_num(f.get("delta")),
            _fmt_num(f.get("threshold")),
            {"html": _status_badge("YES" if detected else "No")},
            {"html": _status_badge(severity)},
        ])

    _render_html_table(headers, rows)

    n_reg = sum(1 for f in findings if f.get("regression_detected"))
    n_crit = sum(
        1
        for f in findings
        if f.get("regression_detected") and str(f.get("severity", "")).lower() == "critical"
    )
    if n_reg:
        st.markdown(
            f'<span class="rci-badge rci-badge-fail">{n_reg} regression(s)</span>'
            + (
                f' &nbsp; <span class="rci-badge rci-badge-critical">{n_crit} CRITICAL</span>'
                if n_crit
                else ""
            ),
            unsafe_allow_html=True,
        )


def render_per_scenario_metrics(metric_results: list) -> None:
    """Render per-scenario metric breakdown."""
    st.markdown('<div class="rci-section"><h2>Detailed Metric Breakdown</h2></div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="rci-section-note">Expand a metric to inspect per-scenario baseline, candidate, and delta values.</p>',
        unsafe_allow_html=True,
    )

    if not metric_results:
        return

    for m in metric_results:
        per_scenario = m.get("per_scenario", {})
        if not per_scenario:
            continue

        with st.expander(f"{_metric_label(m['metric_name'])} - Per-Scenario Breakdown"):
            rows = []
            for sid, vals in sorted(per_scenario.items()):
                rows.append([
                    sid,
                    _fmt_num(vals.get("baseline", 0)),
                    _fmt_num(vals.get("candidate", 0)),
                    _fmt_delta(vals.get("delta", 0)),
                ])
            _render_html_table(["Scenario", "Baseline", "Candidate", "Delta"], rows)


def render_run_information(result: dict) -> None:
    st.markdown('<div class="rci-section"><h2>Run Information</h2></div>', unsafe_allow_html=True)

    status = result.get("overall_status", "UNKNOWN")
    status_label = "REGRESSION DETECTED" if _is_regression(status) else "PASS"
    n_scenarios = len(result.get("scenario_ids", []))
    n_regressed = _count_regressed_scenarios(result)
    simulator_name = result.get("simulator_name", "mock")
    trial_count = result.get("trial_count", 1)

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
      <div class="rci-meta-label">Total Scenarios</div>
      <div class="rci-meta-value">{n_scenarios}</div>
    </div>
    <div class="rci-meta-item">
      <div class="rci-meta-label">Trials / Scenario</div>
      <div class="rci-meta-value">{trial_count}</div>
    </div>
    <div class="rci-meta-item">
      <div class="rci-meta-label">Simulator</div>
      <div class="rci-meta-value">{simulator_name}</div>
    </div>
    <div class="rci-meta-item">
      <div class="rci-meta-label">Regressed Scenarios</div>
      <div class="rci-meta-value">{n_regressed}</div>
    </div>
    <div class="rci-meta-item">
      <div class="rci-meta-label">Overall Status</div>
      <div class="rci-meta-value">{status_label}</div>
    </div>
    <div class="rci-meta-item">
      <div class="rci-meta-label">Seed</div>
      <div class="rci-meta-value">{result.get("seed", "-")}</div>
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
    # Parse results-dir from Streamlit's '--' args
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

    results = load_results(results_dir)

    if not results:
        st.warning(
            f"No results found in `{results_dir}/`. "
            "Run the pipeline first:\n\n"
            "```\npython -m robot_ci run --baseline baseline --candidate candidate --scenarios scenarios/\n```"
        )
        st.sidebar.markdown('<div class="rci-sidebar-brand">Robot CI</div>', unsafe_allow_html=True)
        st.sidebar.caption("No runs available yet.")
        return

    selected_idx = render_sidebar(results)
    result = results[selected_idx]

    # Layout hierarchy
    render_latest_run(result)
    render_key_metrics(result.get("metric_results", []))
    render_scenario_details(result)
    render_regression_table(result.get("regression_findings", []))
    render_per_scenario_metrics(result.get("metric_results", []))
    render_run_information(result)


if __name__ == "__main__":
    main()
