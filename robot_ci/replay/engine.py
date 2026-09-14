import numpy as np
from typing import Dict, List
from robot_ci.policies.base import PolicyInterface
from robot_ci.scenarios.models import Scenario
from robot_ci.simulation.backend import SimulatorBackend
from .rollout import Rollout

class ReplayEngine:
    """Executes policy rollouts against scenarios using a simulator.
    
    Deterministic matched-seed design: for each scenario, the SAME seed
    is used when running both baseline and candidate policies. This ensures
    that any behavioural difference is due to the policy, not randomness.
    """
    
    def __init__(self, simulator: SimulatorBackend):
        self.simulator = simulator
    
    def run(self, policy: PolicyInterface, scenario: Scenario, seed: int = 42) -> Rollout:
        """Run a single rollout. Seed the simulator, reset, loop act->step->record."""
        self.simulator.seed(seed)
        policy.reset()
        
        obs = self.simulator.reset(scenario)
        observations = [obs.copy()]
        actions = []
        states = [self.simulator.get_state()]
        rewards = []
        
        success = False
        for step in range(scenario.max_episode_steps):
            action = policy.act(obs)
            if not isinstance(action, np.ndarray):
                action = np.array(action)
            
            obs, reward, done, info = self.simulator.step(action)
            
            observations.append(obs.copy())
            actions.append(action.copy())
            states.append(self.simulator.get_state())
            rewards.append(float(reward))
            
            if info.get("success", False):
                success = True
            
            if done:
                break
        
        final_state = self.simulator.get_state()
        
        return Rollout(
            scenario_id=scenario.scenario_id,
            policy_version=policy.version,
            seed=seed,
            observations=observations,
            actions=actions,
            states=states,
            rewards=rewards,
            success=success,
            collision_count=final_state.get("collision_count", 0),
            total_steps=len(actions),
        )
    
    def run_batch(
        self,
        policy: PolicyInterface,
        scenarios: List[Scenario],
        global_seed: int = 42,
        trials: int = 1,
    ) -> Dict[str, Rollout]:
        """Run rollouts for all scenarios with matched deterministic seeds."""
        if trials <= 0:
            raise ValueError("trials must be >= 1")

        results = {}
        for scenario in scenarios:
            for trial_index in range(trials):
                if trials == 1:
                    rollout_key = scenario.scenario_id
                    rollout_seed = scenario.seed
                else:
                    rollout_key = f"{scenario.scenario_id}__trial_{trial_index + 1}"
                    rollout_seed = global_seed + scenario.seed + trial_index

                rollout = self.run(policy, scenario, seed=rollout_seed)
                results[rollout_key] = rollout
        return results
