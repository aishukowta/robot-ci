import time
from typing import List
import numpy as np

from robot_ci.policies.registry import get_policy
from robot_ci.scenarios.manager import ScenarioManager
from robot_ci.simulation.pybullet_sim import PyBulletSimulator


def run_demo(
    policy_name: str = "baseline",
    scenario_id: str = "reach_simple",
    scenarios_dir: str = "scenarios",
    seed: int = 42,
) -> bool:
    """Run an interactive PyBullet viewer/control panel for demonstration."""
    manager = ScenarioManager()
    scenarios = manager.load_scenarios(scenarios_dir)
    scenario_by_id = {scenario.scenario_id: scenario for scenario in scenarios}

    scenario_order = [
        "reach_simple",
        "reach_diagonal",
        "pick_and_place_easy",
        "pick_and_place_medium",
        "pick_and_place_hard",
    ]
    available_scenarios = [sid for sid in scenario_order if sid in scenario_by_id]
    if not available_scenarios:
        available_scenarios = sorted(list(scenario_by_id.keys()))

    policy_list = ["baseline", "candidate"]

    current_policy_name = policy_name if policy_name in policy_list else policy_list[0]
    current_scenario_id = scenario_id if scenario_id in available_scenarios else available_scenarios[0]

    # Initialize PyBullet simulator with GUI enabled, sleep_gui disabled for manual speed control
    simulator = PyBulletSimulator(gui=True, action_repeat=18, sleep_gui=False)
    simulator._connect()
    p = simulator._p
    client_id = simulator._client_id

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=client_id)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=client_id)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=client_id)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=client_id)

    policy = get_policy(current_policy_name)
    scenario = scenario_by_id[current_scenario_id]

    simulator.seed(seed)
    policy.reset()
    obs = simulator.reset(scenario)

    # Set up PyBullet GUI Debug Parameters
    btn_play_pause = p.addUserDebugParameter("Play / Pause", 1, 0, 0, physicsClientId=client_id)
    btn_step = p.addUserDebugParameter("Step 1 Frame", 1, 0, 0, physicsClientId=client_id)
    btn_reset = p.addUserDebugParameter("Reset Simulation", 1, 0, 0, physicsClientId=client_id)
    slider_speed = p.addUserDebugParameter("Speed (0.1x - 5.0x)", 0.1, 5.0, 1.0, physicsClientId=client_id)

    init_pol_idx = policy_list.index(current_policy_name) if current_policy_name in policy_list else 0
    slider_policy = p.addUserDebugParameter("Policy (0=Base, 1=Cand)", 0, len(policy_list) - 1, init_pol_idx, physicsClientId=client_id)

    init_sc_idx = available_scenarios.index(current_scenario_id) if current_scenario_id in available_scenarios else 0
    slider_scenario = p.addUserDebugParameter("Scenario (0..4)", 0, len(available_scenarios) - 1, init_sc_idx, physicsClientId=client_id)

    btn_load = p.addUserDebugParameter("Load Selected Policy & Scenario", 1, 0, 0, physicsClientId=client_id)

    prev_play_pause = p.readUserDebugParameter(btn_play_pause, physicsClientId=client_id)
    prev_step = p.readUserDebugParameter(btn_step, physicsClientId=client_id)
    prev_reset = p.readUserDebugParameter(btn_reset, physicsClientId=client_id)
    prev_load = p.readUserDebugParameter(btn_load, physicsClientId=client_id)

    is_paused = False
    is_completed = False
    step_count = 0
    total_reward = 0.0
    success = False
    debug_text_id = -1
    last_overall_success = False

    print("=" * 60)
    print("  ROBOT CI — Interactive PyBullet Demonstration")
    print("=" * 60)
    print(f"  Available scenarios: {available_scenarios}")
    print("  Use GUI side panel controls to Play/Pause, Step, Reset, change Speed, or Load scenarios.")
    print("=" * 60)

    try:
        while p.isConnected(client_id):
            curr_play_pause = p.readUserDebugParameter(btn_play_pause, physicsClientId=client_id)
            curr_step = p.readUserDebugParameter(btn_step, physicsClientId=client_id)
            curr_reset = p.readUserDebugParameter(btn_reset, physicsClientId=client_id)
            curr_load = p.readUserDebugParameter(btn_load, physicsClientId=client_id)
            speed = float(p.readUserDebugParameter(slider_speed, physicsClientId=client_id))

            step_requested = False

            # Play / Pause toggle
            if curr_play_pause != prev_play_pause:
                prev_play_pause = curr_play_pause
                is_paused = not is_paused

            # Step 1 Frame
            if curr_step != prev_step:
                prev_step = curr_step
                is_paused = True
                step_requested = True

            # Reset Simulation
            if curr_reset != prev_reset:
                prev_reset = curr_reset
                simulator.seed(seed)
                policy.reset()
                obs = simulator.reset(scenario)
                is_paused = False
                is_completed = False
                step_count = 0
                total_reward = 0.0
                success = False

            # Load Selected Policy & Scenario
            if curr_load != prev_load:
                prev_load = curr_load
                pol_idx = int(round(p.readUserDebugParameter(slider_policy, physicsClientId=client_id)))
                sc_idx = int(round(p.readUserDebugParameter(slider_scenario, physicsClientId=client_id)))

                pol_idx = max(0, min(len(policy_list) - 1, pol_idx))
                sc_idx = max(0, min(len(available_scenarios) - 1, sc_idx))

                current_policy_name = policy_list[pol_idx]
                current_scenario_id = available_scenarios[sc_idx]

                policy = get_policy(current_policy_name)
                scenario = scenario_by_id[current_scenario_id]

                simulator.seed(seed)
                policy.reset()
                obs = simulator.reset(scenario)

                is_paused = False
                is_completed = False
                step_count = 0
                total_reward = 0.0
                success = False
                print(f"[Demo] Switched to policy='{policy.version}', scenario='{current_scenario_id}'")

            # Simulation Step Execution
            if not is_completed and (not is_paused or step_requested):
                action = policy.act(obs)
                obs, reward, done, info = simulator.step(action)
                step_count += 1
                total_reward += reward

                if info.get("success", False):
                    success = True

                if done:
                    is_completed = True
                    last_overall_success = success
                    print(
                        f"[Demo Complete] policy={policy.version}, scenario={current_scenario_id}, "
                        f"success={success}, steps={step_count}, collisions={simulator.collision_count}, reward={total_reward:.2f}"
                    )

            # Viewport Status Text Overlay
            status_str = "COMPLETED" if is_completed else ("PAUSED" if is_paused else "RUNNING")
            succ_str = "SUCCESS" if success else ("FAILED" if is_completed else "IN PROGRESS")

            overlay = (
                f"ROBOT CI INTERACTIVE DEMO\n"
                f"-----------------------------\n"
                f"Policy   : {current_policy_name} ({policy.version})\n"
                f"Scenario : {current_scenario_id}\n"
                f"Status   : {status_str}\n"
                f"Step     : {step_count} / {scenario.max_episode_steps}\n"
                f"Success  : {succ_str}\n"
                f"Collisions: {simulator.collision_count}\n"
                f"Reward   : {total_reward:.2f}\n"
                f"Speed    : {speed:.1f}x\n"
                f"-----------------------------\n"
                f"Controls in Side Panel ->"
            )

            text_color = [0.1, 0.8, 0.2] if success else ([0.9, 0.2, 0.1] if is_completed else [0.1, 0.5, 0.9])
            if debug_text_id < 0:
                debug_text_id = p.addUserDebugText(
                    overlay,
                    [0.1, -0.65, 0.55],
                    textColorRGB=text_color,
                    textSize=1.1,
                    physicsClientId=client_id,
                )
            else:
                p.addUserDebugText(
                    overlay,
                    [0.1, -0.65, 0.55],
                    textColorRGB=text_color,
                    textSize=1.1,
                    replaceItemUniqueId=debug_text_id,
                    physicsClientId=client_id,
                )

            # Frame sleep (controlled by speed slider)
            base_delay = (1.0 / 240.0) * simulator.action_repeat
            actual_delay = max(0.001, base_delay / max(0.1, speed))
            time.sleep(actual_delay)

    except KeyboardInterrupt:
        pass
    finally:
        simulator.close()

    return last_overall_success
