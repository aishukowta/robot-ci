import uuid
from datetime import datetime, timezone
from pathlib import Path

from robot_ci.policies.registry import get_policy
from robot_ci.scenarios.manager import ScenarioManager
from robot_ci.simulation.mock import MockSimulator
from robot_ci.simulation.pybullet_sim import PyBulletSimulator
from robot_ci.replay.engine import ReplayEngine
from robot_ci.evaluation.evaluator import EvaluationEngine
from robot_ci.regression.detector import RegressionDetector
from robot_ci.results.models import TestRunResult
from robot_ci.results.store import ResultStore
from robot_ci.stats import summarize, bootstrap_delta_ci


def compute_variability(metric_results) -> dict:
    """Per-metric spread across all matched rollouts (scenarios x trials).

    Detection rules are NOT changed; this is reported alongside them so a
    reader can judge how stable each difference is.
    """
    out = {}
    for mr in metric_results:
        b = [float(v["baseline"]) for v in mr.per_scenario.values()]
        c = [float(v["candidate"]) for v in mr.per_scenario.values()]
        out[mr.metric_name] = {
            "baseline": summarize(b),
            "candidate": summarize(c),
            "delta_mean_ci95": bootstrap_delta_ci(b, c),
            "note": "Spread is across scenarios and trials together; use --trials N for per-scenario repeatability.",
        }
    return out

class PipelineRunner:
    def __init__(self, config_dir: str = "config", results_dir: str = "results"):
        self.config_dir = config_dir
        self.results_dir = results_dir
    
    def run(
        self,
        baseline_name: str,
        candidate_name: str,
        scenarios_dir: str,
        seed: int = 42,
        simulator_name: str = "pybullet",
        trials: int = 1,
    ) -> TestRunResult:
        print("=" * 70)
        print("  ROBOT CI — Regression Testing Pipeline")
        print("=" * 70)
        
        # 1. Load policies
        print("\n[1/8] Loading policies...")
        baseline = get_policy(baseline_name)
        candidate = get_policy(candidate_name)
        print(f"  Baseline: {baseline.version}")
        print(f"  Candidate: {candidate.version}")
        
        # 2. Load scenarios
        print("\n[2/8] Loading scenarios...")
        manager = ScenarioManager()
        scenarios = manager.load_scenarios(scenarios_dir)
        if not scenarios:
            raise ValueError(f"No valid scenarios found in directory '{scenarios_dir}'.")
        print(f"  Loaded {len(scenarios)} scenarios: {[s.scenario_id for s in scenarios]}")
        
        # 3. Create simulator and replay engine
        print("\n[3/8] Initializing simulator...")
        simulator = self._create_simulator(simulator_name)
        engine = ReplayEngine(simulator)
        print(f"  Simulator: {simulator_name}")
        print(f"  Trials per scenario: {trials}")
        
        # 4. Run baseline rollouts
        print("\n[4/8] Running baseline rollouts...")
        baseline_rollouts = engine.run_batch(baseline, scenarios, global_seed=seed, trials=trials)
        for sid, r in baseline_rollouts.items():
            print(f"  {sid}: {'SUCCESS' if r.success else 'FAILED'} ({r.total_steps} steps, {r.collision_count} collisions)")
        
        # 5. Run candidate rollouts
        print("\n[5/8] Running candidate rollouts...")
        candidate_rollouts = engine.run_batch(candidate, scenarios, global_seed=seed, trials=trials)
        for sid, r in candidate_rollouts.items():
            print(f"  {sid}: {'SUCCESS' if r.success else 'FAILED'} ({r.total_steps} steps, {r.collision_count} collisions)")
        
        # 6. Evaluate metrics
        print("\n[6/8] Evaluating metrics...")
        eval_engine = EvaluationEngine()
        metric_results = eval_engine.evaluate(baseline_rollouts, candidate_rollouts)
        variability_preview = compute_variability(metric_results)
        for mr in metric_results:
            ci = variability_preview[mr.metric_name]["delta_mean_ci95"]
            ci_txt = f", delta 95% CI=[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ""
            print(f"  {mr.metric_name}: baseline={mr.baseline_value:.4f}, candidate={mr.candidate_value:.4f}, delta={mr.delta:.4f}{ci_txt}")
        
        # 7. Detect regressions
        print("\n[7/8] Detecting regressions...")
        config_path = str(Path(self.config_dir) / "regression_thresholds.yaml")
        detector = RegressionDetector(config_path=config_path)
        findings = detector.analyze(metric_results)
        regressions = [f for f in findings if f.regression_detected]
        for f in findings:
            status = f"{'[REGRESSION]' if f.regression_detected else '[OK]':>15}"
            print(f"  {status} | {f.metric_name}: {f.reason}")
        
        # 8. Save results
        print("\n[8/8] Saving results...")
        overall_status = "REGRESSION_DETECTED" if regressions else "PASS"
        
        # Build rollout summary
        rollout_summary = {}
        for sid in baseline_rollouts:
            br = baseline_rollouts[sid]
            cr = candidate_rollouts[sid]
            rollout_summary[sid] = {
                "baseline": {"success": br.success, "steps": br.total_steps, "collisions": br.collision_count, "reward": sum(br.rewards)},
                "candidate": {"success": cr.success, "steps": cr.total_steps, "collisions": cr.collision_count, "reward": sum(cr.rewards)},
            }
        
        result = TestRunResult(
            run_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(timezone.utc).isoformat(),
            baseline_policy_version=baseline.version,
            candidate_policy_version=candidate.version,
            seed=seed,
            scenario_ids=[s.scenario_id for s in scenarios],
            simulator_name=simulator_name,
            trial_count=trials,
            metric_results=[mr.to_dict() for mr in metric_results],
            regression_findings=[f.to_dict() for f in findings],
            overall_status=overall_status,
            rollout_summary=rollout_summary,
            variability=compute_variability(metric_results),
        )
        
        store = ResultStore(self.results_dir)
        filepath = store.save(result)
        print(f"  Results saved to: {filepath}")
        
        # Print summary
        print("\n" + "=" * 70)
        print(f"  OVERALL RESULT: {overall_status}")
        if regressions:
            print(f"  Regressions detected: {len(regressions)}")
            for r in regressions:
                print(f"    - {r.metric_name} ({r.severity})")
        print("=" * 70)
        
        if hasattr(simulator, "close"):
            simulator.close()
        return result

    def _create_simulator(self, simulator_name: str):
        if simulator_name == "pybullet":
            return PyBulletSimulator(gui=False)
        if simulator_name == "mock":
            return MockSimulator()
        if simulator_name == "hardware":
            from robot_ci.simulation.hardware import HardwareBackend
            return HardwareBackend()
        raise ValueError(f"Unknown simulator: {simulator_name}. Available: pybullet, mock, hardware")
