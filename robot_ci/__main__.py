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
        choices=["pybullet", "mock"],
        default="pybullet",
        help="Simulator backend to use (default: pybullet)",
    )
    run_parser.add_argument("--results-dir", default="results", help="Results output directory (default: results)")
    run_parser.add_argument("--config-dir", default="config", help="Config directory (default: config)")
    
    # 'demo' subcommand
    demo_parser = subparsers.add_parser("demo", help="Launch a visual PyBullet rollout")
    demo_parser.add_argument("--policy", default="candidate", help="Policy name to demo (default: candidate)")
    demo_parser.add_argument("--scenario", default="pick_and_place_medium", help="Scenario id to demo")
    demo_parser.add_argument("--scenarios", default="scenarios", help="Path to scenarios directory (default: scenarios)")
    demo_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # 'dashboard' subcommand
    dash_parser = subparsers.add_parser("dashboard", help="Launch the Streamlit results dashboard")
    dash_parser.add_argument("--results-dir", default="results", help="Results directory to visualize")
    
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
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
