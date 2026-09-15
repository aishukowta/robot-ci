from robot_ci.policies.registry import get_policy
from robot_ci.replay.engine import ReplayEngine
from robot_ci.scenarios.manager import ScenarioManager
from robot_ci.simulation.pybullet_sim import PyBulletSimulator


def run_demo(policy_name: str, scenario_id: str, scenarios_dir: str, seed: int = 42) -> bool:
    """Run one visible PyBullet rollout for panel/demo use."""
    manager = ScenarioManager()
    scenarios = manager.load_scenarios(scenarios_dir)
    scenario_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    if scenario_id not in scenario_by_id:
        available = ", ".join(sorted(scenario_by_id))
        raise ValueError(f"Unknown scenario: {scenario_id}. Available: {available}")

    policy = get_policy(policy_name)
    simulator = PyBulletSimulator(gui=True, action_repeat=18, sleep_gui=True)
    engine = ReplayEngine(simulator)
    try:
        rollout = engine.run(policy, scenario_by_id[scenario_id], seed=seed)
    finally:
        simulator.close()

    print(
        f"Demo complete: policy={policy.version}, scenario={scenario_id}, "
        f"success={rollout.success}, steps={rollout.total_steps}, "
        f"collisions={rollout.collision_count}, reward={sum(rollout.rewards):.3f}"
    )
    print("This GUI rollout uses PyBullet. The CI pipeline uses the same backend with GUI disabled.")
    return rollout.success
