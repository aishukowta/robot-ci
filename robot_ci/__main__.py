import argparse
import sys
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        prog="robot_ci",
        description="Robot CI — Regression Testing for Robotic Manipulation Policies",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # 'run' subcommand
    run_parser = subparsers.add_parser("run", help="Run the regression testing pipeline")
    run_parser.add_argument("--baseline", default="baseline", help="Baseline policy name (default: baseline)")
    run_parser.add_argument("--candidate", default="candidate", help="Candidate policy name (default: candidate)")
    run_parser.add_argument("--scenarios", default="scenarios", help="Path to scenarios directory (default: scenarios)")
    run_parser.add_argument("--seed", type=int, default=42, help="Global random seed (default: 42)")
    run_parser.add_argument("--trials", type=int, default=1, help="Trials per scenario (default: 1)")
    run_parser.add_argument(
        "--simulator",
        choices=["pybullet", "mock", "hardware"],
        default="pybullet",
        help="Simulator backend to use (default: pybullet)",
    )
    run_parser.add_argument("--results-dir", default="results", help="Results output directory (default: results)")
    run_parser.add_argument("--config-dir", default="config", help="Config directory (default: config)")
    
    # 'demo' subcommand
    demo_parser = subparsers.add_parser("demo", help="Launch a visual PyBullet rollout")
    demo_parser.add_argument("--policy", default="baseline", help="Policy name to demo (default: baseline)")
    demo_parser.add_argument("--scenario", default="reach_simple", help="Scenario id to demo (default: reach_simple)")
    demo_parser.add_argument("--scenarios", default="scenarios", help="Path to scenarios directory (default: scenarios)")
    demo_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # 'dashboard' subcommand
    dash_parser = subparsers.add_parser("dashboard", help="Launch the Streamlit results dashboard")
    dash_parser.add_argument("--results-dir", default="results", help="Results directory to visualize")

    # 'analyze-telemetry' subcommand
    analyze_parser = subparsers.add_parser("analyze-telemetry", help="Analyze offline robot telemetry log")
    analyze_parser.add_argument("--input", required=True, help="Path to input telemetry file (.json or .jsonl)")
    analyze_parser.add_argument("--output", default=None, help="Output path for analysis report JSON")
    analyze_parser.add_argument("--reference", default=None, help="Reference telemetry log file for comparison")
    analyze_parser.add_argument("--sim-scenario", default=None, help="Scenario ID to simulate for comparison baseline")
    analyze_parser.add_argument("--sim-policy", default="baseline", help="Policy for simulation baseline (default: baseline)")
    analyze_parser.add_argument("--scenarios", default="scenarios", help="Scenarios directory (default: scenarios)")
    analyze_parser.add_argument("--simulator", choices=["pybullet", "mock"], default="pybullet", help="Simulator for reference simulation (default: pybullet)")
    analyze_parser.add_argument("--notes", default=None, help="Optional descriptive notes for this analysis run")

    # 'extract-trajectory' subcommand
    extract_parser = subparsers.add_parser("extract-trajectory", help="Extract visual end-effector trajectory from video")
    extract_parser.add_argument("--video", required=True, help="Path to video file (.mp4, .mov, etc.)")
    extract_parser.add_argument("--output", default=None, help="Output JSON path for extracted trajectory")
    extract_parser.add_argument("--csv", default=None, help="Optional CSV export path")
    extract_parser.add_argument("--color", default="red", help="Marker color preset: red, green, blue, yellow, orange (default: red)")
    extract_parser.add_argument("--min-area", type=float, default=10.0, help="Minimum marker contour area in pixels (default: 10.0)")
    extract_parser.add_argument("--notes", default=None, help="Optional description notes")

    # 'compare-video-sim' subcommand
    compare_parser = subparsers.add_parser("compare-video-sim", help="Compare extracted video trajectory against simulation rollout")
    compare_parser.add_argument("--trajectory", "--video-trajectory", default=None, help="Path to extracted trajectory JSON file")
    compare_parser.add_argument("--video", default=None, help="Path to video file (will extract trajectory if --trajectory not provided)")
    compare_parser.add_argument("--scenario", default="reach_simple", help="Scenario ID to simulate (default: reach_simple)")
    compare_parser.add_argument("--policy", default="baseline", help="Simulation policy (default: baseline)")
    compare_parser.add_argument("--simulator", choices=["pybullet", "mock"], default="pybullet", help="Simulator backend (default: pybullet)")
    compare_parser.add_argument("--seed", type=int, default=42, help="Simulation seed (default: 42)")
    compare_parser.add_argument("--scenarios", default="scenarios", help="Scenarios directory (default: scenarios)")
    compare_parser.add_argument("--calibration", default=None, help="Optional camera calibration JSON path (enables Calibrated Projection mode)")
    compare_parser.add_argument("--controller", default="unknown_physical_controller", help="Physical robot controller name")
    compare_parser.add_argument("--robot-model", default="unknown_robot", help="Physical robot model")
    compare_parser.add_argument("--plane", choices=["xy", "xz", "yz"], default="xy", help="2D projection plane for shape-only mode (default: xy)")
    compare_parser.add_argument("--color", default="red", help="Marker color preset if extracting from video (default: red)")
    compare_parser.add_argument("--output", default=None, help="Optional output path for comparison report JSON")

    # 'check-hardware' subcommand (alias: 'readiness')
    hw_parser = subparsers.add_parser("check-hardware", aliases=["readiness"], help="Audit physical robot hardware integration readiness")
    hw_parser.add_argument("--config", default=None, help="Path to hardware config JSON file")
    hw_parser.add_argument("--transport", choices=["none", "fake", "ros2"], default="none", help="Transport type to inspect (default: none)")
    hw_parser.add_argument("--robot-model", default=None, help="Override robot model name")
    hw_parser.add_argument("--driver", default=None, help="Override driver name/type")
    hw_parser.add_argument("--output", default=None, help="Output path for readiness report JSON")
    hw_parser.add_argument("--strict", action="store_true", help="Exit with non-zero code if status is not READY")
    
    args = parser.parse_args()

    
    if args.command == "run":
        from robot_ci.pipeline.runner import PipelineRunner
        runner = PipelineRunner(config_dir=args.config_dir, results_dir=args.results_dir)
        result = runner.run(
            baseline_name=args.baseline,
            candidate_name=args.candidate,
            scenarios_dir=args.scenarios,
            seed=args.seed,
            simulator_name=args.simulator,
            trials=args.trials,
        )
        sys.exit(0 if result.overall_status == "PASS" else 1)
    
    elif args.command == "demo":
        from robot_ci.demo import run_demo
        success = run_demo(
            policy_name=args.policy,
            scenario_id=args.scenario,
            scenarios_dir=args.scenarios,
            seed=args.seed,
        )
        sys.exit(0 if success else 1)

    elif args.command == "dashboard":
        dashboard_path = Path(__file__).parent.parent / "dashboard" / "app.py"
        if not dashboard_path.exists():
            print(f"Dashboard not found at {dashboard_path}")
            sys.exit(1)
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path),
                       "--", "--results-dir", args.results_dir])

    elif args.command == "analyze-telemetry":
        from robot_ci.telemetry.workflow import OfflineTelemetryWorkflow, OfflineReportStore
        print("=" * 70)
        print("  ROBOT CI — Offline Telemetry Analysis")
        print("=" * 70)
        try:
            workflow = OfflineTelemetryWorkflow()
            report = workflow.analyze(
                input_path=args.input,
                reference_path=args.reference,
                sim_scenario=args.sim_scenario,
                sim_policy=args.sim_policy,
                scenarios_dir=args.scenarios,
                simulator_name=args.simulator,
                notes=args.notes,
            )
            store = OfflineReportStore()
            if args.output:
                out_path = report.save(args.output)
            else:
                out_path = store.save(report)

            print(f"\n[OK] Analysis completed successfully:")
            print(f"  Report ID: {report.report_id}")
            print(f"  Source ID: {report.telemetry_source} (Origin: {report.provenance.get('data_origin')})")
            print(f"  Samples: {report.sample_count} | Duration: {report.time_range.get('duration_seconds')}s")
            print(f"  Validation: {'VALID' if report.validation_summary.get('is_valid') else 'WARNINGS/ERRORS'}")
            if report.trajectory_comparison:
                tc = report.trajectory_comparison
                print(f"  Trajectory Comparison (vs {report.reference_source}):")
                print(f"    RMS Euclidean: {tc.get('rms_euclidean_distance')}m")
                print(f"    Max Euclidean: {tc.get('max_euclidean_distance')}m")
                print(f"    DTW Cost: {tc.get('dtw_cost')} | Avg Step Dist: {tc.get('dtw_average_distance')}m")
            print(f"  Saved report to: {out_path}")
            sys.exit(0)
        except Exception as e:
            print(f"\n[ERROR] Offline telemetry analysis failed: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "extract-trajectory":
        from robot_ci.vision import (
            TrackerConfig,
            extract_trajectory_from_video,
            UNCALIBRATED_CAMERA_DISCLAIMER,
        )
        print("=" * 70)
        print("  ROBOT CI — Video End-Effector Trajectory Extraction")
        print("=" * 70)
        try:
            cfg = TrackerConfig(
                color_preset=args.color,
                min_contour_area=args.min_area,
            )
            result = extract_trajectory_from_video(
                video_path=args.video,
                config=cfg,
                notes=args.notes,
            )

            out_json = args.output or f"results/telemetry/{Path(args.video).stem}_trajectory.json"
            result.to_json(out_json)

            if args.csv:
                result.to_csv(args.csv)

            print(f"\n[OK] Video extraction completed successfully:")
            print(f"  Video: {result.video_metadata.file_path}")
            print(f"  Resolution: {result.video_metadata.width}x{result.video_metadata.height} @ {result.video_metadata.fps:.1f} FPS")
            print(f"  Duration: {result.video_metadata.duration_seconds:.2f}s ({result.total_frames} frames)")
            print(f"  Detections: {result.detected_count}/{result.total_frames} ({result.detection_rate * 100:.1f}%)")
            print(f"  Coordinates: 2D Image Space ({result.coordinate_space})")
            print(f"  Disclaimer: {UNCALIBRATED_CAMERA_DISCLAIMER[:110]}...")
            print(f"  Saved JSON to: {out_json}")
            if args.csv:
                print(f"  Saved CSV to:  {args.csv}")
            sys.exit(0)
        except Exception as e:
            print(f"\n[ERROR] Video trajectory extraction failed: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "compare-video-sim":
        from robot_ci.vision import (
            TrackerConfig,
            extract_trajectory_from_video,
            VideoTrajectoryResult,
            CameraCalibration,
            extract_simulated_trajectory,
            compare_video_to_simulation,
            ComparisonMode,
        )
        print("=" * 70)
        print("  ROBOT CI — Real Video to Simulation Trajectory Comparison")
        print("=" * 70)
        try:
            # 1. Obtain video trajectory
            if args.trajectory:
                print(f"Loading video trajectory from: {args.trajectory}")
                video_traj = VideoTrajectoryResult.load(args.trajectory)
            elif args.video:
                print(f"Extracting trajectory from video: {args.video}")
                cfg = TrackerConfig(color_preset=args.color)
                video_traj = extract_trajectory_from_video(args.video, config=cfg)
            else:
                print("[ERROR] Must specify either --trajectory <path.json> or --video <path.mp4>", file=sys.stderr)
                sys.exit(1)

            # 2. Load camera calibration if provided
            calib = None
            if args.calibration:
                print(f"Loading camera calibration from: {args.calibration}")
                calib = CameraCalibration.load(args.calibration)

            # 3. Extract actual simulation trajectory
            print(f"Running simulation rollout:")
            print(f"  Simulator: {args.simulator} | Scenario: {args.scenario} | Policy: {args.policy} | Seed: {args.seed}")
            sim_traj = extract_simulated_trajectory(
                policy_name=args.policy,
                scenario_id=args.scenario,
                simulator_name=args.simulator,
                seed=args.seed,
                scenarios_dir=args.scenarios,
            )
            print(f"  Simulated steps: {sim_traj.total_steps} | Success: {sim_traj.success} | Collisions: {sim_traj.collision_count}")

            # 4. Perform trajectory comparison
            report = compare_video_to_simulation(
                video_trajectory=video_traj,
                sim_trajectory=sim_traj,
                calibration=calib,
                physical_controller_id=args.controller,
                robot_model=args.robot_model,
                shape_projection_plane=args.plane,
            )

            # 5. Persist comparison report
            saved_path = report.save(args.output)

            # 6. Print formatted summary
            print("\n" + "=" * 70)
            print(f"[OK] Trajectory Comparison Completed successfully")
            print("=" * 70)
            print(f"  Report ID:           {report.report_id}")
            print(f"  Comparison Mode:     {report.comparison_mode.value}")
            print(f"  Comparison Type:     {report.comparison_type}")
            print(f"  Scenario:            {report.scenario_id}")
            print(f"  Simulation Policy:   {report.sim_policy_id} ({report.provenance.get('sim_backend')})")
            print(f"  Physical Controller: {report.physical_controller_id}")
            print(f"  Coordinate System:   {report.coordinate_system}")
            print(f"  Measurement Units:   {report.units}")
            print(f"  Tracking Coverage:   {report.metrics.get('tracking_coverage') * 100:.1f}% ({report.metrics.get('video_frames_detected')}/{report.metrics.get('video_frames_total')} frames)")
            print(f"  Video Duration:      {report.metrics.get('video_duration_seconds')}s | Sim Duration: {report.metrics.get('sim_duration_seconds')}s")
            print(f"  Video Path Length:   {report.metrics.get('video_path_length')} {report.units}")
            print(f"  Sim Path Length:     {report.metrics.get('sim_path_length')} {report.units}")
            print(f"  RMS Deviation:       {report.metrics.get('rms_trajectory_deviation')} {report.units}")
            print(f"  Mean Deviation:      {report.metrics.get('mean_trajectory_deviation')} {report.units}")
            print(f"  Max Deviation:       {report.metrics.get('max_trajectory_deviation')} {report.units}")
            print(f"  DTW Average Step:    {report.metrics.get('dtw_average_step_distance')} {report.units} (Cost: {report.metrics.get('dtw_cost')})")
            if report.warnings:
                print("\n  Warnings:")
                for w in report.warnings:
                    print(f"    - {w}")
            print("\n  Disclaimers:")
            for d in report.disclaimers:
                print(f"    * {d[:90]}...")
            print(f"\n  Saved Report:        {saved_path}")
            print("=" * 70)
            sys.exit(0)
        except Exception as e:
            print(f"\n[ERROR] Trajectory comparison failed: {e}", file=sys.stderr)
            sys.exit(1)
    
    elif args.command in ("check-hardware", "readiness"):
        from robot_ci.hardware.readiness import (
            audit_hardware_readiness,
            format_readiness_report_text,
            HardwareReadinessStatus,
        )
        from robot_ci.simulation.safety import HardwareConfig
        import json

        cfg = HardwareConfig()
        if args.config:
            with open(args.config, "r", encoding="utf-8") as f:
                raw_cfg = json.load(f)
            for k, v in raw_cfg.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)

        if args.robot_model:
            cfg.robot_name = args.robot_model
        if args.driver:
            cfg.driver_type = args.driver
            cfg.driver_configured = True

        transport = None
        if args.transport == "fake":
            from robot_ci.transport.fake_transport import FakeRobotTransport
            transport = FakeRobotTransport()
            transport.connect()
        elif args.transport == "ros2":
            from robot_ci.transport.ros2_adapter import ROS2RobotTransport
            transport = ROS2RobotTransport()

        report = audit_hardware_readiness(config=cfg, transport=transport)
        formatted_text = format_readiness_report_text(report)
        print(formatted_text)

        if args.output:
            saved_path = report.save(args.output)
            print(f"\n[OK] Hardware readiness report saved to: {saved_path}")

        if args.strict and report.status != HardwareReadinessStatus.READY:
            sys.exit(1)
        elif report.status == HardwareReadinessStatus.BLOCKED:
            sys.exit(1)
        else:
            sys.exit(0)

    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
