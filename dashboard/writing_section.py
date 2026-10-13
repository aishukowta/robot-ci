"""Section D of the Robot CI dashboard: the handwriting robot.

Tabs:
  1. Simulate        - run one writing policy on one target (kinematic or PyBullet), see the page + metrics,
                       optionally record a video of the simulated arm writing.
  2. Regression      - baseline vs candidate over many targets and trials; findings with confidence intervals.
  3. Export for robot - SVG / G-code / CSV + plan.json + lab checklist for tomorrow's recording.
  4. Upload recording - video and/or photo of the real robot -> real-vs-sim comparison and diagnosis.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from robot_ci.writing.font import polyline_length
from robot_ci.writing.metrics import score_execution
from robot_ci.writing.policies import WRITING_POLICY_NAMES, get_writing_policy
from robot_ci.writing.render import BLUE, RED, render_comparison
from robot_ci.writing.simulate import make_writing_simulator
from robot_ci.writing.targets import load_targets

METRIC_FMT = {
    "trajectory_error_mean_mm": ("Trajectory error mean", "mm", 3),
    "trajectory_error_max_mm": ("Trajectory error max", "mm", 3),
    "path_coverage": ("Coverage within 1 mm", "", 3),
    "writing_accuracy": ("Writing accuracy", "", 3),
    "execution_time_s": ("Execution time", "s", 2),
    "path_efficiency_deviation": ("Path efficiency |ratio-1|", "", 3),
    "safety_events": ("Safety events", "", 0),
}


def _section(title: str) -> None:
    st.markdown(f'<div class="rci-section"><h3>{title}</h3></div>', unsafe_allow_html=True)


def _status(ok: bool, text: str) -> None:
    css = "rci-alert-pass" if ok else "rci-alert-fail"
    st.markdown(f'<div class="rci-alert {css}" style="text-align:center;">{text}</div>', unsafe_allow_html=True)


def _targets() -> List:
    try:
        return load_targets("writing_targets")
    except Exception as e:  # pragma: no cover - shown in UI
        st.error(f"Could not load writing targets: {e}")
        return []


def render_section_d_writing(results_dir: str = "results") -> None:
    st.markdown('<div class="rci-section"><h2>Section D: Handwriting Robot</h2></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="rci-section-note">The robot writes letters, digits, words and geometric patterns. '
        "Baseline and candidate writing policies are compared on the same targets and seeds in simulation; "
        "the exported plan is drawn by the real robot, and the recording is compared with the simulation of the "
        "same plan. Writing policies are rule-based planners; candidates contain deliberately injected defects.</div>",
        unsafe_allow_html=True,
    )
    wdir = str(Path(results_dir) / "writing")
    tabs = st.tabs(["🖊️ Simulate", "📊 Regression (baseline vs candidate)", "📤 Export for robot",
                    "🎥 Upload recording (real-to-sim)"])
    with tabs[0]:
        _tab_simulate(wdir)
    with tabs[1]:
        _tab_regression(wdir)
    with tabs[2]:
        _tab_export(wdir)
    with tabs[3]:
        _tab_upload(wdir)


# ---------------------------------------------------------------------------
# 1. Simulate
# ---------------------------------------------------------------------------

def _tab_simulate(wdir: str) -> None:
    targets = _targets()
    if not targets:
        return
    ids = [t.target_id for t in targets]
    c1, c2, c3 = st.columns(3)
    tid = c1.selectbox("Target", ids, index=ids.index("word_hello") if "word_hello" in ids else 0, key="wd_sim_target")
    pol = c2.selectbox("Writing policy", WRITING_POLICY_NAMES, key="wd_sim_policy")
    backend = c3.selectbox("Simulator", ["pybullet", "kinematic"], key="wd_sim_backend",
                           help="pybullet: Franka Panda stand-in arm with IK; kinematic: fast plotter model")
    record = st.checkbox("Record a video of the simulated arm writing (PyBullet, ~30-60 s)", value=False,
                         key="wd_sim_record", disabled=backend != "pybullet")
    st.caption(f"Policy: {get_writing_policy(pol).description}")
    if st.button("▶ Run simulation", type="primary", key="wd_sim_run"):
        t = next(x for x in targets if x.target_id == tid)
        layout = t.layout()
        plan = get_writing_policy(pol).plan(layout, t.seed)
        sim = make_writing_simulator(backend)
        out = Path(wdir) / "sim"
        video = str(out / f"{tid}__{pol}__sim.mp4") if (record and backend == "pybullet") else None
        with st.spinner("Simulating..."):
            try:
                ex = sim.execute(plan, t.seed, record_path=video) if video else sim.execute(plan, t.seed)
            finally:
                if hasattr(sim, "close"):
                    sim.close()
            m = score_execution(ex, layout, float(sum(polyline_length(s) for s in layout.strokes)))
            img = render_comparison(layout, [(pol, ex.ink_strokes(), BLUE)], out / f"{tid}__{pol}__{backend}.png",
                                    title=f"{tid} | {pol} | {backend}")
        st.session_state["wd_sim_result"] = {"img": img, "values": m.values, "video": video, "hash": plan.plan_hash,
                                             "safety": ex.safety, "acc": m.details["writing_accuracy"]}
    res = st.session_state.get("wd_sim_result")
    if res:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.image(res["img"], caption=f"Plan {res['hash']}: grey = intended path, blue = simulated ink")
            if res.get("video") and Path(res["video"]).exists():
                st.video(res["video"])
        with c2:
            rows = [{"Metric": METRIC_FMT[k][0], "Value": round(v, METRIC_FMT[k][2]) if v is not None else None,
                     "Unit": METRIC_FMT[k][1]} for k, v in res["values"].items() if k in METRIC_FMT]
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            mis = [f"{c['label']}→{c['predicted']}" for c in res["acc"]["cells"] if not c["correct"]]
            st.caption("Misread: " + (", ".join(mis) if mis else "none"))
            st.caption(f"Safety: {res['safety'].get('joint_limit_note', '')}")


# ---------------------------------------------------------------------------
# 2. Regression
# ---------------------------------------------------------------------------

def _tab_regression(wdir: str) -> None:
    from robot_ci.writing.regression import WritingRunConfig, load_writing_runs, run_writing_regression
    targets = _targets()
    ids = [t.target_id for t in targets]
    with st.expander("Run a new regression test", expanded=not load_writing_runs(wdir)):
        c1, c2 = st.columns(2)
        base = c1.selectbox("Baseline", WRITING_POLICY_NAMES, index=0, key="wd_reg_base")
        cand = c2.selectbox("Candidate", WRITING_POLICY_NAMES, index=1, key="wd_reg_cand")
        sel = st.multiselect("Targets", ids, default=ids, key="wd_reg_targets")
        c3, c4 = st.columns(2)
        trials = c3.slider("Trials per target", 1, 10, 3, key="wd_reg_trials",
                           help=">= 3 trials enables confidence intervals and repeatability")
        backend = c4.selectbox("Simulator", ["kinematic", "pybullet"], key="wd_reg_backend",
                               help="kinematic ~20 s for all targets x 3 trials; pybullet ~2-3 min")
        if st.button("▶ Run regression test", type="primary", key="wd_reg_run"):
            if not sel:
                st.error("Select at least one target.")
            else:
                bar = st.progress(0.0, text="Starting...")
                total = len(sel) * trials
                done = {"n": 0}

                def prog(msg: str) -> None:
                    done["n"] += 1
                    bar.progress(min(1.0, done["n"] / total), text=msg.strip())

                r = run_writing_regression(WritingRunConfig(base, cand, sel, trials, backend,
                                                            results_dir=wdir), progress=prog)
                bar.empty()
                st.session_state["wd_reg_selected"] = r["saved_to"]
                st.success(f"Run {r['run_id']} saved: {r['saved_to']}")

    runs = load_writing_runs(wdir)
    if not runs:
        st.info("No handwriting regression runs yet. Run one above, or from the terminal: "
                "`python -m robot_ci writing-run --backend kinematic`")
        return
    labels = [f"{r['run_id']} | {r['baseline']['name']} vs {r['candidate']['name']} | {r['backend']} | "
              f"{r['trials']} trial(s) | {r['overall_status']}" for r in runs]
    default = next((i for i, r in enumerate(runs) if r["saved_to"] == st.session_state.get("wd_reg_selected")), 0)
    idx = st.selectbox("Saved runs", range(len(runs)), index=default, format_func=lambda i: labels[i], key="wd_reg_pick")
    run = runs[idx]
    _status(run["overall_status"] == "PASS",
            "PASS — no handwriting metric regressed" if run["overall_status"] == "PASS" else "REGRESSION DETECTED")
    st.caption(f"Baseline: {run['baseline']['description']}")
    st.caption(f"Candidate: {run['candidate']['description']}")

    _section("Findings (paired over targets x trials)")
    rows = []
    for f in run["findings"]:
        ci = f.get("delta_ci95")
        rows.append({
            "Metric": f["label"], "Unit": f["unit"],
            "Baseline": round(f["baseline"]["mean"], 4) if f.get("baseline") else None,
            "Candidate": round(f["candidate"]["mean"], 4) if f.get("candidate") else None,
            "Change (95% CI)": f"{f['delta_mean']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]" if (ci and "delta_mean" in f)
            else (f"{f['delta_mean']:+.4f}" if "delta_mean" in f else "-"),
            "Threshold": f["threshold"], "Status": (f"REGRESSION ({f['severity']})" if f["regression_detected"] else "ok"),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    with st.expander("Why each metric was or was not flagged"):
        for f in run["findings"]:
            st.markdown(f"- **{f['label']}**: {f['reason']}")

    _section("Per-target results")
    pt = []
    for tid_, d in run["per_target"].items():
        pt.append({"Target": tid_,
                   "Err mean B": d["baseline"]["trajectory_error_mean_mm"], "Err mean C": d["candidate"]["trajectory_error_mean_mm"],
                   "Accuracy B": d["baseline"]["writing_accuracy"], "Accuracy C": d["candidate"]["writing_accuracy"],
                   "Time B (s)": d["baseline"]["execution_time_s"], "Time C (s)": d["candidate"]["execution_time_s"],
                   "Repeat B (mm)": d["baseline"]["repeatability_mm"], "Repeat C (mm)": d["candidate"]["repeatability_mm"],
                   "Misreads C": ", ".join(f"{m['label']}→{m['predicted']}" for m in d["misread_cells"]["candidate"][:4])})
    st.dataframe(pd.DataFrame(pt).round(3), hide_index=True)

    _section("Pages (trial 1): grey = intended, blue = baseline, red = candidate")
    imgs = [(k, v) for k, v in run.get("images", {}).items() if Path(v).exists()]
    for i in range(0, len(imgs), 3):
        cols = st.columns(3)
        for col, (k, v) in zip(cols, imgs[i:i + 3]):
            col.image(v, caption=k)
    with st.expander("Run notes and environment"):
        for n in run.get("notes", []):
            st.markdown(f"- {n}")
        st.json(run.get("environment", {}))


# ---------------------------------------------------------------------------
# 3. Export
# ---------------------------------------------------------------------------

def _tab_export(wdir: str) -> None:
    from robot_ci.writing.export import ExportSafetyError, RobotProfile, export_plan
    targets = _targets()
    ids = [t.target_id for t in targets]
    prof_path = "config/writing_robot_profile.yaml"
    try:
        prof = RobotProfile.load(prof_path)
    except Exception as e:
        st.error(f"Robot profile error ({prof_path}): {e}")
        return
    if not prof.is_configured:
        st.warning(f"Robot profile `{prof_path}` is not configured yet (robot: {prof.robot_name}). SVG, CSV and "
                   "plan.json are exported; G-code is withheld until you fill in the coordinate mapping and pen commands.")
    else:
        st.success(f"Robot profile: {prof.robot_name} ({prof.execution_mode})")
    c1, c2 = st.columns(2)
    tid = c1.selectbox("Target", ids, key="wd_ex_target")
    pol = c2.selectbox("Policy to draw on the robot", WRITING_POLICY_NAMES, key="wd_ex_policy")
    if st.button("📤 Export files for the robot", type="primary", key="wd_ex_run"):
        t = next(x for x in targets if x.target_id == tid)
        plan = get_writing_policy(pol).plan(t.layout(), t.seed)
        try:
            st.session_state["wd_export"] = export_plan(plan, prof, str(Path(wdir) / "exports"))
        except ExportSafetyError as e:
            st.error(str(e))
    man = st.session_state.get("wd_export")
    if man:
        st.success(f"Plan {man['plan_hash']} exported to `{man['folder']}`")
        for w in man["warnings"]:
            st.info(w)
        cols = st.columns(4)
        for col, key, mime in zip(cols, ["svg", "gcode", "csv", "plan"],
                                  ["image/svg+xml", "text/plain", "text/csv", "application/json"]):
            path = man["files"].get(key)
            if path and Path(path).exists():
                col.download_button(f"⬇ {Path(path).name}", Path(path).read_bytes(), file_name=Path(path).name,
                                    mime=mime, key=f"wd_dl_{key}")
        lab = man["files"].get("lab_instructions")
        if lab and Path(lab).exists():
            with st.expander("Lab checklist for this export", expanded=True):
                st.code(Path(lab).read_text(encoding="utf-8"), language=None)


# ---------------------------------------------------------------------------
# 4. Upload recording
# ---------------------------------------------------------------------------

def _list_exported_plans(wdir: str) -> List[Path]:
    root = Path(wdir) / "exports"
    return sorted(root.glob("*/plan.json"), key=lambda p: p.stat().st_mtime, reverse=True) if root.exists() else []


def _tab_upload(wdir: str) -> None:
    from robot_ci.writing.analysis import AnalysisInputs, analyze_recording
    from robot_ci.writing.diagnosis import diagnose_writing
    from robot_ci.writing.export import RobotProfile
    from robot_ci.writing.regression import load_writing_runs

    plans = _list_exported_plans(wdir)
    if not plans:
        st.info("Export a plan first (tab 'Export for robot'), draw it on the robot, then upload the recording here.")
        return
    st.markdown("**1. Which exported plan did the robot draw?**")
    pidx = st.selectbox("Exported plan", range(len(plans)), format_func=lambda i: plans[i].parent.name, key="wd_up_plan")
    plan_path = plans[pidx]

    st.markdown("**2. Upload the recording** (a photo of the finished page gives the most accurate measurement; "
                "a video adds execution time)")
    c1, c2 = st.columns(2)
    vid = c1.file_uploader("Video of the robot writing", type=["mp4", "mov", "avi", "mkv", "m4v"], key="wd_up_vid")
    pho = c2.file_uploader("Photo of the finished page", type=["jpg", "jpeg", "png"], key="wd_up_pho")

    st.markdown("**3. Provenance**")
    src_label = st.radio("What is this recording?", [
        "Real robot drawing (robot_phone_recording)",
        "Drawn by hand / no robot (handheld_marker_recording)",
        "Synthetic test file (synthetic_fixture)",
    ], key="wd_up_src")
    source = src_label.split("(")[-1].rstrip(")")
    executed = st.checkbox("The robot drew THIS exported file (not its own built-in font or text tool)",
                           value=False, key="wd_up_exec", disabled=not source.startswith("robot"))

    with st.expander("Advanced options"):
        a1, a2, a3 = st.columns(3)
        ink = a1.selectbox("Ink colour", ["dark", "black", "blue", "red"], key="wd_up_ink")
        sim_backend = a2.selectbox("Simulator for comparison", ["pybullet", "kinematic"], key="wd_up_sim")
        marker = a3.selectbox("Pen-holder marker colour (video)", ["none", "red", "green", "blue", "yellow", "orange"],
                              key="wd_up_marker")
        corners_txt = st.text_input("Manual page corners (only if automatic detection fails): "
                                    "x1,y1;x2,y2;x3,y3;x4,y4 = page top-left, top-right, bottom-right, bottom-left",
                                    key="wd_up_corners")
        runs = load_writing_runs(wdir)
        run_labels = ["(none)"] + [f"{r['run_id']} | {r['baseline']['name']} vs {r['candidate']['name']} | "
                                   f"{r['overall_status']}" for r in runs]
        ridx = st.selectbox("Simulation regression run to combine for diagnosis", range(len(run_labels)),
                            format_func=lambda i: run_labels[i], key="wd_up_run")

    if st.button("🔍 Analyse recording against simulation", type="primary", key="wd_up_go"):
        if not vid and not pho:
            st.error("Upload a video, a photo, or both.")
            return
        up_dir = Path(wdir) / "uploads" / time.strftime("%Y%m%d_%H%M%S")
        up_dir.mkdir(parents=True, exist_ok=True)
        vpath = ppath = None
        if vid:
            vpath = up_dir / Path(vid.name).name
            vpath.write_bytes(vid.getbuffer())
        if pho:
            ppath = up_dir / Path(pho.name).name
            ppath.write_bytes(pho.getbuffer())
        corners = None
        if corners_txt.strip():
            try:
                corners = [[float(v) for v in pair.split(",")] for pair in corners_txt.split(";")]
                assert len(corners) == 4 and all(len(c) == 2 for c in corners)
            except Exception:
                st.error("Corners must look like: 120,80;900,95;930,1250;100,1230")
                return
        try:
            mode = RobotProfile.load("config/writing_robot_profile.yaml").execution_mode
        except Exception:
            mode = "svg_import"
        with st.spinner("Detecting the page, extracting ink, simulating the same plan, comparing..."):
            try:
                rep = analyze_recording(AnalysisInputs(
                    plan_path=str(plan_path), data_source=source, executed_plan=bool(executed and source.startswith("robot")),
                    photo_path=str(ppath) if ppath else None, video_path=str(vpath) if vpath else None,
                    manual_corners=corners, ink=ink, sim_backend=sim_backend, execution_mode=mode,
                    marker_color=None if marker == "none" else marker), str(Path(wdir) / "real_to_sim"))
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.info("Tips: page on a darker surface, all 4 corners visible, camera above the page, no shadow over "
                        "the writing. If corners are not found, enter them manually under Advanced options.")
                return
        run = runs[ridx - 1] if ridx > 0 else None
        st.session_state["wd_up_report"] = rep
        st.session_state["wd_up_diag"] = diagnose_writing(run, rep)

    rep = st.session_state.get("wd_up_report")
    if not rep:
        return
    diag = st.session_state.get("wd_up_diag")
    m = rep["metrics"]
    _section("Result")
    if diag is not None:
        _status(diag.category.value in ("NO_REGRESSION_DETECTED",), f"DIAGNOSIS: {diag.category.value}")
        st.markdown(f"{diag.summary_explanation}  \n**Next step:** {diag.recommended_next_step}")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.image(rep["files"]["overlay_page"], caption="Rectified page: grey intended, blue simulation, red real ink")
        st.image(rep["files"]["overlay_aligned"], caption="After removing the paper placement offset")
    with c2:
        def fmt(v, n=3):
            return None if v is None else round(v, n)
        table = []
        for key, lab in (("trajectory_error_mean_mm", "Trajectory error mean (mm)"),
                         ("trajectory_error_max_mm", "Trajectory error max (mm)"),
                         ("path_coverage", "Coverage within 1 mm"), ("writing_accuracy", "Writing accuracy")):
            table.append({"Metric": lab, "Simulation": fmt(m["intended_vs_sim"].get(key)),
                          "Real (raw)": fmt(m["intended_vs_real_raw"].get(key)),
                          "Real (aligned)": fmt(m["intended_vs_real_aligned"].get(key))})
        st.dataframe(pd.DataFrame(table), hide_index=True)
        st.metric("Real vs simulation (aligned)", f"{m['real_vs_sim_aligned']['symmetric_mean_mm']:.2f} mm",
                  help="Mean distance between real ink and simulated ink after removing the placement offset")
        pc = rep["placement_correction"]
        st.caption(f"Placement offset removed: {pc['translation_norm_mm']} mm, {pc['rotation_deg']}°")
        t = m["timing"]
        st.caption(f"Execution time: real {t.get('real_active_duration_s')} s, simulation {t.get('sim_duration_s')} s"
                   + ("" if t.get("comparable") else " (not compared: see note)"))
        st.caption(f"Data source: {rep['data_source']} | comparison: {rep['comparison_type']}")
    for w in rep["warnings"]:
        st.warning(w)
    if diag is not None:
        with st.expander("Diagnosis details", expanded=False):
            for title, items in (("Supporting", diag.supporting_findings), ("Data quality", diag.data_quality_warnings),
                                 ("Missing", diag.missing_evidence), ("Cannot be concluded", diag.unsupported_conclusions)):
                if items:
                    st.markdown(f"**{title}**")
                    for it in items:
                        st.markdown(f"- {it}")
    with st.expander("Per-character recognition (real, aligned)"):
        st.dataframe(pd.DataFrame([{k: c.get(k) for k in ("label", "predicted", "correct", "score", "runner_up")}
                                   for c in rep["character_results_real"]]), hide_index=True)
    st.download_button("⬇ Download analysis report (JSON)", json.dumps(rep, indent=2, default=str),
                       file_name=f"writing_analysis_{rep['report_id']}.json", mime="application/json",
                       key="wd_up_dl")
    st.caption(f"Saved to {rep.get('saved_to', '-')}. Not measurable from video: "
               + "; ".join(rep.get("unmeasurable_from_video", [])))
