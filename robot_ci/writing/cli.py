"""Command-line interface for the handwriting robot workflow (wired into `python -m robot_ci`)."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def add_writing_commands(subparsers) -> None:
    from .policies import WRITING_POLICY_NAMES

    p = subparsers.add_parser("writing-targets", help="List handwriting targets")
    p.add_argument("--targets-dir", default="writing_targets")

    p = subparsers.add_parser("writing-sim", help="Simulate one writing policy on one target (optionally in the PyBullet GUI)")
    p.add_argument("--policy", default="writing_baseline", choices=WRITING_POLICY_NAMES)
    p.add_argument("--target", default="word_hello")
    p.add_argument("--backend", choices=["pybullet", "kinematic"], default="pybullet")
    p.add_argument("--gui", action="store_true", help="Open the PyBullet window and draw the ink live")
    p.add_argument("--record", default=None, help="Save an MP4 of the simulated arm writing (PyBullet backend)")
    p.add_argument("--seed", type=int, default=None, help="Plan/noise seed (default: the target's seed)")
    p.add_argument("--targets-dir", default="writing_targets")
    p.add_argument("--out", default="results/writing/sim")

    p = subparsers.add_parser("writing-run", help="Baseline vs candidate handwriting regression test")
    p.add_argument("--baseline", default="writing_baseline", choices=WRITING_POLICY_NAMES)
    p.add_argument("--candidate", default="writing_candidate", choices=WRITING_POLICY_NAMES)
    p.add_argument("--targets", default="all", help="Comma-separated target ids, or 'all'")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--backend", choices=["pybullet", "kinematic"], default="pybullet")
    p.add_argument("--targets-dir", default="writing_targets")
    p.add_argument("--thresholds", default="config/writing_thresholds.yaml")
    p.add_argument("--results-dir", default="results/writing")

    p = subparsers.add_parser("writing-export", help="Export a writing plan for the real robot (SVG / G-code / CSV)")
    p.add_argument("--policy", default="writing_baseline", choices=WRITING_POLICY_NAMES)
    p.add_argument("--target", default="word_hello")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--profile", default="config/writing_robot_profile.yaml")
    p.add_argument("--targets-dir", default="writing_targets")
    p.add_argument("--out", default="results/writing/exports")

    p = subparsers.add_parser("writing-analyze", help="Analyse a recording of the real robot against simulation")
    p.add_argument("--plan", required=True, help="plan.json from writing-export")
    p.add_argument("--photo", default=None, help="Photo of the finished page")
    p.add_argument("--video", default=None, help="Video of the robot writing")
    p.add_argument("--source", required=True,
                   choices=["robot_phone_recording", "handheld_marker_recording", "synthetic_fixture",
                            "controlled_sim_perturbation"])
    p.add_argument("--executed-plan", action="store_true",
                   help="Declare that the robot drew THIS exported plan (not its own font/text tool)")
    p.add_argument("--corners", default=None,
                   help="Manual page corners in the image: 'x1,y1;x2,y2;x3,y3;x4,y4' = page top-left, top-right, bottom-right, bottom-left")
    p.add_argument("--ink", choices=["dark", "black", "blue", "red"], default="dark")
    p.add_argument("--sim-backend", choices=["pybullet", "kinematic"], default="pybullet")
    p.add_argument("--profile", default="config/writing_robot_profile.yaml",
                   help="Robot profile (its execution_mode decides whether timing is comparable)")
    p.add_argument("--marker-color", default=None, help="Optional coloured marker on the pen holder (red, green, blue, ...)")
    p.add_argument("--run", default=None, help="Writing regression run JSON to combine into a diagnosis")
    p.add_argument("--targets-dir", default="writing_targets")
    p.add_argument("--out", default="results/writing/real_to_sim")
    p.add_argument("--notes", default="")

    p = subparsers.add_parser("writing-diagnose", help="Diagnose from a writing run and/or a recording analysis")
    p.add_argument("--run", default=None)
    p.add_argument("--analysis", default=None)

    p = subparsers.add_parser("writing-selftest", help="End-to-end check of the upload pipeline with synthetic recordings")
    p.add_argument("--out", default="results/writing/selftest")
    p.add_argument("--sim-backend", choices=["pybullet", "kinematic"], default="kinematic")


def _print_diag(rep) -> None:
    print("\n" + "=" * 72)
    print(f"  DIAGNOSIS: {rep.category.value}")
    print("=" * 72)
    print(f"  {rep.summary_explanation}")
    print(f"  Next step: {rep.recommended_next_step}")
    for title, items in (("Supporting", rep.supporting_findings), ("Data quality", rep.data_quality_warnings),
                         ("Missing", rep.missing_evidence), ("Cannot conclude", rep.unsupported_conclusions)):
        if items:
            print(f"\n  {title}:")
            for it in items:
                print(f"    - {it}")


def handle_writing_command(args) -> bool:
    cmd = args.command
    if not cmd or not cmd.startswith("writing-"):
        return False
    try:
        if cmd == "writing-targets":
            from .targets import load_targets
            for t in load_targets(args.targets_dir):
                print(f"  {t.target_id:20s} {t.kind:8s} {t.content!r:18s} height {t.height_mm:g} mm  - {t.description}")

        elif cmd == "writing-sim":
            from .metrics import score_execution
            from .font import polyline_length
            from .policies import get_writing_policy
            from .render import BLUE, render_comparison
            from .simulate import make_writing_simulator
            from .targets import get_target
            t = get_target(args.target, args.targets_dir)
            layout = t.layout()
            seed = t.seed if args.seed is None else args.seed
            plan = get_writing_policy(args.policy).plan(layout, seed)
            sim = make_writing_simulator(args.backend, gui=args.gui)
            print(f"Simulating {args.policy} on {args.target} ({args.backend}{', GUI' if args.gui else ''}) ...")
            try:
                if args.record and args.backend == "pybullet":
                    ex = sim.execute(plan, seed, record_path=args.record)
                    print(f"Simulation video: {args.record}")
                else:
                    if args.record:
                        print("NOTE: --record needs --backend pybullet; no video written.")
                    ex = sim.execute(plan, seed)
                if args.gui:
                    input("Done. Press Enter to close the PyBullet window...")
            finally:
                if hasattr(sim, "close"):
                    sim.close()
            m = score_execution(ex, layout, float(sum(polyline_length(s) for s in layout.strokes)))
            img = render_comparison(layout, [(args.policy, ex.ink_strokes(), BLUE)],
                                    Path(args.out) / f"{args.target}__{args.policy}__{args.backend}.png",
                                    title=f"{args.target} | {args.policy} | {args.backend}")
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.values.items()}, indent=2))
            print(f"Plan hash {plan.plan_hash} | image: {img}")

        elif cmd == "writing-run":
            from .regression import WritingRunConfig, run_writing_regression
            ids = None if args.targets.strip().lower() == "all" else [s.strip() for s in args.targets.split(",") if s.strip()]
            cfg = WritingRunConfig(baseline=args.baseline, candidate=args.candidate, targets=ids, trials=args.trials,
                                   backend=args.backend, targets_dir=args.targets_dir,
                                   thresholds_path=args.thresholds, results_dir=args.results_dir)
            print("=" * 72)
            print(f"  ROBOT CI - Handwriting regression: {args.baseline} vs {args.candidate} "
                  f"({args.backend}, {args.trials} trial(s))")
            print("=" * 72)
            r = run_writing_regression(cfg, progress=print)
            print("\n  Findings:")
            for f in r["findings"]:
                tag = f"[{f['severity'].upper()}]" if f.get("regression_detected") else "[OK]"
                print(f"  {tag:11s} {f['label']}: {f['reason']}")
            print(f"\n  OVERALL: {r['overall_status']}")
            print(f"  Saved: {r['saved_to']}")
            sys.exit(0 if r["overall_status"] == "PASS" else 1)

        elif cmd == "writing-export":
            from .export import RobotProfile, export_plan
            from .policies import get_writing_policy
            from .targets import get_target
            t = get_target(args.target, args.targets_dir)
            seed = t.seed if args.seed is None else args.seed
            plan = get_writing_policy(args.policy).plan(t.layout(), seed)
            prof = RobotProfile.load(args.profile)
            man = export_plan(plan, prof, args.out)
            print(f"Exported plan {man['plan_hash']} to {man['folder']}")
            for k, v in man["files"].items():
                print(f"  {k:17s} {v}")
            for w in man["warnings"]:
                print(f"  NOTE: {w}")

        elif cmd == "writing-analyze":
            from .analysis import AnalysisInputs, analyze_recording
            from .export import RobotProfile
            corners = None
            if args.corners:
                corners = [[float(v) for v in pair.split(",")] for pair in args.corners.split(";")]
                if len(corners) != 4 or any(len(c) != 2 for c in corners):
                    raise ValueError("--corners must be 'x1,y1;x2,y2;x3,y3;x4,y4'")
            mode = RobotProfile.load(args.profile).execution_mode if Path(args.profile).exists() else "svg_import"
            rep = analyze_recording(AnalysisInputs(
                plan_path=args.plan, data_source=args.source, executed_plan=args.executed_plan,
                photo_path=args.photo, video_path=args.video, manual_corners=corners, ink=args.ink,
                sim_backend=args.sim_backend, targets_dir=args.targets_dir, execution_mode=mode,
                marker_color=args.marker_color, notes=args.notes), args.out)
            m = rep["metrics"]
            print("=" * 72)
            print(f"  REAL-TO-SIM ANALYSIS {rep['report_id']}  ({rep['data_source']}, {rep['comparison_type']})")
            print("=" * 72)
            print(f"  Plan {rep['plan']['plan_hash']} | {rep['plan']['policy_name']} | {rep['plan']['target_id']}")
            print(f"  Page: {rep['page_detection']['method']} detection, aspect error "
                  f"{rep['page_detection']['aspect_error'] * 100:.1f}%  ({rep['inputs']['page_image_source']})")
            row = "  {:34s} {:>12s} {:>12s} {:>12s}"
            print(row.format("", "simulation", "real (raw)", "real (aligned)"))
            for key, lab in (("trajectory_error_mean_mm", "trajectory error mean (mm)"),
                             ("trajectory_error_max_mm", "trajectory error max (mm)"),
                             ("path_coverage", "coverage within 1 mm"), ("writing_accuracy", "writing accuracy")):
                vals = [m["intended_vs_sim"].get(key), m["intended_vs_real_raw"].get(key), m["intended_vs_real_aligned"].get(key)]
                print(row.format(lab, *[("-" if v is None else f"{v:.3f}") for v in vals]))
            print(f"  Real vs sim (aligned): {m['real_vs_sim_aligned']['symmetric_mean_mm']:.3f} mm mean, "
                  f"raw {m['real_vs_sim_raw']['symmetric_mean_mm']:.3f} mm")
            print(f"  Placement correction: {rep['placement_correction']}")
            print(f"  Timing: {m['timing']}")
            for w in rep["warnings"]:
                print(f"  WARNING: {w}")
            print(f"  Report: {rep['saved_to']}")
            print(f"  Overlay: {rep['files']['overlay_page']}")
            if args.run:
                from .diagnosis import diagnose_writing
                run = json.loads(Path(args.run).read_text(encoding="utf-8"))
                _print_diag(diagnose_writing(run, rep))

        elif cmd == "writing-diagnose":
            from .diagnosis import diagnose_writing
            run = json.loads(Path(args.run).read_text(encoding="utf-8")) if args.run else None
            ana = json.loads(Path(args.analysis).read_text(encoding="utf-8")) if args.analysis else None
            _print_diag(diagnose_writing(run, ana))

        elif cmd == "writing-selftest":
            from .selftest import run_selftest
            print("Generating synthetic photo + video with a known 1.2 / -0.8 mm offset and running the upload pipeline...")
            s = run_selftest(args.out, args.sim_backend)
            for c in s["checks"]:
                print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['check']}: {c['detail']}")
            print(f"\n  SELF-TEST {'PASSED' if s['passed'] else 'FAILED'} - summary: {s['saved_to']}")
            sys.exit(0 if s["passed"] else 1)
    except SystemExit:
        raise
    except Exception as e:  # clear message instead of a traceback for users
        print(f"\n[ERROR] {cmd} failed: {e}", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)
