# SPDX-License-Identifier: BSD-3-Clause

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import json
import csv
from datetime import datetime

import isaacgym  # noqa: F401
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.math import euler_from_quaternion

import torch


EVAL_COMMANDS = [
    {"name": "stand_still", "posture": "stand", "x": 0.0, "y": 0.0, "yaw": 0.0},
    {"name": "stand_forward", "posture": "stand", "x": 0.3, "y": 0.0, "yaw": 0.0},
    {"name": "stand_turn", "posture": "stand", "x": 0.0, "y": 0.0, "yaw": 0.5},
    {"name": "crouch_still", "posture": "crouch", "x": 0.0, "y": 0.0, "yaw": 0.0},
    {"name": "crouch_forward", "posture": "crouch", "x": 0.3, "y": 0.0, "yaw": 0.0},
    {"name": "crouch_turn", "posture": "crouch", "x": 0.0, "y": 0.0, "yaw": 0.5},
]

EPISODES_PER_COMMAND = 20
EVALUATE_RANDOM_UPPER_BODY = False


def configure_eval_env(env_cfg, episodes_per_command):
    env_cfg.env.num_envs = episodes_per_command
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.mesh_type = "plane"
    if hasattr(env_cfg.terrain, "measure_heights"):
        env_cfg.terrain.measure_heights = False

    env_cfg.noise.add_noise = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.use_random = False

    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.domain_rand.randomize_body_displacement = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_kp = False
    env_cfg.domain_rand.randomize_kd = False
    env_cfg.domain_rand.randomize_initial_joint_pos = False
    env_cfg.domain_rand.randomize_joint_injection = False
    env_cfg.domain_rand.randomize_actuation_offset = False
    env_cfg.domain_rand.delay = False

    env_cfg.asset.self_collision = 0
    env_cfg.env.upper_teleop = False
    return env_cfg


def set_command(env, command, height):
    env.commands[:, 0] = command["x"]
    env.commands[:, 1] = command["y"]
    env.commands[:, 2] = command["yaw"]
    if env.commands.shape[1] > 4:
        env.commands[:, 4] = height
    if hasattr(env, "action_curriculum_ratio"):
        env.action_curriculum_ratio = 1.0 if EVALUATE_RANDOM_UPPER_BODY else 0.0


def suppress_random_upper_body(env):
    if EVALUATE_RANDOM_UPPER_BODY:
        return
    if hasattr(env, "current_upper_actions"):
        env.current_upper_actions.zero_()
    if hasattr(env, "random_upper_actions"):
        env.random_upper_actions.zero_()
    if hasattr(env, "delta_upper_actions"):
        env.delta_upper_actions.zero_()
    if hasattr(env, "common_step_counter") and hasattr(env.cfg.domain_rand, "upper_interval"):
        upper_interval = max(int(env.cfg.domain_rand.upper_interval), 2)
        if env.common_step_counter % upper_interval == 0:
            env.common_step_counter += 1



def build_step_metrics(env, command, height, previous_dof_vel):
    x_error = torch.abs(env.base_lin_vel[:, 0] - command["x"])
    y_error = torch.abs(env.base_lin_vel[:, 1] - command["y"])
    yaw_error = torch.abs(env.base_ang_vel[:, 2] - command["yaw"])

    roll, pitch, _ = euler_from_quaternion(env.base_quat)
    roll_abs = torch.abs(roll)
    pitch_abs = torch.abs(pitch)

    dof_vel_abs = torch.abs(env.dof_vel)
    if previous_dof_vel is None:
        dof_acc_abs = torch.zeros_like(dof_vel_abs)
    else:
        dof_acc_abs = torch.abs((env.dof_vel - previous_dof_vel) / env.dt)

    foot_slip = torch.zeros(env.num_envs, device=env.device)
    relative_height_error = torch.zeros(env.num_envs, device=env.device)
    if hasattr(env, "feet_pos"):
        base_to_ankle_height = torch.max(
            env.root_states[:, 2].unsqueeze(1) - env.feet_pos[:, :, 2], dim=1
        ).values
        measured_height = base_to_ankle_height + env.cfg.asset.ankle_sole_distance
        relative_height_error = torch.abs(measured_height - height)

    if hasattr(env, "feet_indices") and hasattr(env, "contact_forces"):
        contact = env.contact_forces[:, env.feet_indices, 2] > 1.0
        if hasattr(env, "feet_vel"):
            foot_xy_speed = torch.norm(env.feet_vel[:, :, :2], dim=2)
        elif hasattr(env, "rigid_body_states"):
            foot_xy_speed = torch.norm(env.rigid_body_states[:, env.feet_indices, 7:9], dim=2)
        else:
            foot_xy_speed = torch.zeros_like(contact, dtype=torch.float)
        foot_slip = torch.sum(foot_xy_speed * contact.float(), dim=1)

    return {
        "x_vel_abs_error": x_error,
        "y_vel_abs_error": y_error,
        "yaw_vel_abs_error": yaw_error,
        "roll_abs": roll_abs,
        "pitch_abs": pitch_abs,
        "dof_vel_abs_mean": dof_vel_abs.mean(dim=1),
        "dof_acc_abs_mean": dof_acc_abs.mean(dim=1),
        "foot_slip": foot_slip,
        "relative_height_abs_error": relative_height_error,
    }


def summarize_command(command, episodes, max_episode_length):
    count = len(episodes)
    summary = {
        "command": command,
        "episodes": count,
        "success_rate": sum(ep["success"] for ep in episodes) / count,
        "mean_episode_length": sum(ep["episode_length"] for ep in episodes) / count,
        "max_episode_length": int(max_episode_length),
    }

    metric_names = [
        "x_vel_abs_error",
        "y_vel_abs_error",
        "yaw_vel_abs_error",
        "roll_abs_mean",
        "pitch_abs_mean",
        "roll_abs_max",
        "pitch_abs_max",
        "dof_vel_abs_mean",
        "dof_acc_abs_mean",
        "foot_slip_mean",
        "relative_height_abs_error_mean",
    ]
    for name in metric_names:
        summary[name] = sum(ep[name] for ep in episodes) / count
    return summary


def evaluate_command(env, policy, command, height, episodes_per_command):
    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    obs = env.get_observations()
    set_command(env, command, height)

    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    lengths = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    sums = {}
    max_roll = torch.zeros(env.num_envs, device=env.device)
    max_pitch = torch.zeros(env.num_envs, device=env.device)
    episodes = []
    previous_dof_vel = env.dof_vel.clone()

    for _ in range(int(env.max_episode_length) + 5):
        with torch.no_grad():
            set_command(env, command, height)
            suppress_random_upper_body(env)
            actions = policy(obs.detach())
            set_command(env, command, height)
            suppress_random_upper_body(env)
            obs, _, _, dones, _, _, _ = env.step(actions.detach())
            suppress_random_upper_body(env)
            set_command(env, command, height)

            step_metrics = build_step_metrics(env, command, height, previous_dof_vel)
            previous_dof_vel = env.dof_vel.clone()

            active_ids = active.nonzero(as_tuple=False).flatten()
            if active_ids.numel() > 0:
                lengths[active_ids] += 1
                for name, values in step_metrics.items():
                    if name not in sums:
                        sums[name] = torch.zeros(env.num_envs, device=env.device)
                    sums[name][active_ids] += values[active_ids]
                max_roll[active_ids] = torch.maximum(max_roll[active_ids], step_metrics["roll_abs"][active_ids])
                max_pitch[active_ids] = torch.maximum(max_pitch[active_ids], step_metrics["pitch_abs"][active_ids])

            done_ids = ((dones > 0) & active).nonzero(as_tuple=False).flatten()
            for env_id in done_ids.tolist():
                denom = max(float(lengths[env_id].item()), 1.0)
                episodes.append({
                    "env_id": env_id,
                    "success": bool(env.time_out_buf[env_id].item()),
                    "episode_length": int(lengths[env_id].item()),
                    "x_vel_abs_error": float((sums["x_vel_abs_error"][env_id] / denom).item()),
                    "y_vel_abs_error": float((sums["y_vel_abs_error"][env_id] / denom).item()),
                    "yaw_vel_abs_error": float((sums["yaw_vel_abs_error"][env_id] / denom).item()),
                    "roll_abs_mean": float((sums["roll_abs"][env_id] / denom).item()),
                    "pitch_abs_mean": float((sums["pitch_abs"][env_id] / denom).item()),
                    "roll_abs_max": float(max_roll[env_id].item()),
                    "pitch_abs_max": float(max_pitch[env_id].item()),
                    "dof_vel_abs_mean": float((sums["dof_vel_abs_mean"][env_id] / denom).item()),
                    "dof_acc_abs_mean": float((sums["dof_acc_abs_mean"][env_id] / denom).item()),
                    "foot_slip_mean": float((sums["foot_slip"][env_id] / denom).item()),
                    "relative_height_abs_error_mean": float((sums["relative_height_abs_error"][env_id] / denom).item()),
                })
                active[env_id] = False
                if len(episodes) >= episodes_per_command:
                    return episodes[:episodes_per_command]

    active_ids = active.nonzero(as_tuple=False).flatten()
    for env_id in active_ids.tolist():
        denom = max(float(lengths[env_id].item()), 1.0)
        episodes.append({
            "env_id": env_id,
            "success": True,
            "episode_length": int(lengths[env_id].item()),
            "x_vel_abs_error": float((sums["x_vel_abs_error"][env_id] / denom).item()),
            "y_vel_abs_error": float((sums["y_vel_abs_error"][env_id] / denom).item()),
            "yaw_vel_abs_error": float((sums["yaw_vel_abs_error"][env_id] / denom).item()),
            "roll_abs_mean": float((sums["roll_abs"][env_id] / denom).item()),
            "pitch_abs_mean": float((sums["pitch_abs"][env_id] / denom).item()),
            "roll_abs_max": float(max_roll[env_id].item()),
            "pitch_abs_max": float(max_pitch[env_id].item()),
            "dof_vel_abs_mean": float((sums["dof_vel_abs_mean"][env_id] / denom).item()),
            "dof_acc_abs_mean": float((sums["dof_acc_abs_mean"][env_id] / denom).item()),
            "foot_slip_mean": float((sums["foot_slip"][env_id] / denom).item()),
            "relative_height_abs_error_mean": float((sums["relative_height_abs_error"][env_id] / denom).item()),
        })
        if len(episodes) >= episodes_per_command:
            break
    return episodes[:episodes_per_command]


def write_results(train_cfg, results):
    experiment = train_cfg.runner.experiment_name or "default_experiment"
    output_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", experiment, "deterministic_eval")
    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"eval_{stamp}.json")
    csv_path = os.path.join(output_dir, f"eval_{stamp}.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    rows = []
    for item in results["commands"]:
        row = {"name": item["command"]["name"]}
        row.update(item["summary"])
        row.pop("command", None)
        rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return json_path, csv_path


def main(args):
    args.headless = True
    args.resume = True

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg = configure_eval_env(env_cfg, EPISODES_PER_COMMAND)
    stand_height = env_cfg.rewards.base_height_target
    crouch_height = stand_height + env_cfg.commands.ranges.height[0]

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None)
    policy = ppo_runner.get_inference_policy(device=env.device)

    results = {
        "task": args.task,
        "load_run": train_cfg.runner.load_run,
        "checkpoint": train_cfg.runner.checkpoint,
        "episodes_per_command": EPISODES_PER_COMMAND,
        "stand_height_command": stand_height,
        "crouch_height_command": crouch_height,
        "max_episode_length": int(env.max_episode_length),
        "evaluate_random_upper_body": EVALUATE_RANDOM_UPPER_BODY,
        "commands": [],
    }

    for command in EVAL_COMMANDS:
        height = crouch_height if command["posture"] == "crouch" else stand_height
        print(f"Evaluating {command['name']}: height={height} x={command['x']} y={command['y']} yaw={command['yaw']}")
        episodes = evaluate_command(env, policy, command, height, EPISODES_PER_COMMAND)
        summary = summarize_command(command, episodes, env.max_episode_length)
        summary["height_command"] = height
        results["commands"].append({"command": command, "summary": summary, "episodes": episodes})
        print(
            f"  success={summary['success_rate']:.2f}, "
            f"len={summary['mean_episode_length']:.1f}, "
            f"x_err={summary['x_vel_abs_error']:.3f}, "
            f"y_err={summary['y_vel_abs_error']:.3f}, "
            f"yaw_err={summary['yaw_vel_abs_error']:.3f}, "
            f"height_err={summary['relative_height_abs_error_mean']:.3f}, "
            f"slip={summary['foot_slip_mean']:.3f}"
        )

    json_path, csv_path = write_results(train_cfg, results)
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote CSV: {csv_path}")


if __name__ == "__main__":
    main(get_args())
