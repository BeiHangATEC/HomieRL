#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import sys
from datetime import datetime

REPO_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_PACKAGE_ROOT.parent
for package_root in (REPO_PACKAGE_ROOT, WORKSPACE_ROOT / "rsl_rl"):
    if package_root.exists() and str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

import isaacgym  # noqa: F401
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.math import euler_from_quaternion

import torch


DEFAULT_CASES = [
    {
        "name": "full_squat_normal",
        "start_height": 0.945,
        "end_height": 0.765,
        "settle_time": 2.0,
        "descend_time": 6.0,
        "hold_time": 4.0,
    },
    {
        "name": "full_squat_slow",
        "start_height": 0.945,
        "end_height": 0.765,
        "settle_time": 2.0,
        "descend_time": 10.0,
        "hold_time": 4.0,
    },
    {
        "name": "shallow_squat_normal",
        "start_height": 0.945,
        "end_height": 0.800,
        "settle_time": 2.0,
        "descend_time": 6.0,
        "hold_time": 4.0,
    },
    {
        "name": "fixed_low_height",
        "start_height": 0.765,
        "end_height": 0.765,
        "settle_time": 0.0,
        "descend_time": 0.0,
        "hold_time": 8.0,
    },
]


def _set_if_present(obj, name, value):
    if hasattr(obj, name):
        setattr(obj, name, value)


def _zero_random_upper_body(env):
    if hasattr(env, "action_curriculum_ratio"):
        env.action_curriculum_ratio = 0.0
    for name in ("current_upper_actions", "random_upper_actions", "delta_upper_actions"):
        if hasattr(env, name):
            getattr(env, name).zero_()


def _mean(values):
    return sum(values) / len(values) if values else None


def _max_abs(values):
    return max((abs(v) for v in values), default=None)


def _tail_mean(values, count):
    if not values:
        return None
    selected = values[-min(count, len(values)):]
    return sum(selected) / len(selected)


def _to_float(value):
    return float(value.detach().cpu().item())


def _get_case_height(case, step, env_dt):
    settle_steps = int(case["settle_time"] / env_dt)
    descend_steps = int(case["descend_time"] / env_dt)

    if step < settle_steps:
        return "settle", case["start_height"]

    squat_step = step - settle_steps
    if descend_steps <= 0 or squat_step >= descend_steps:
        return "hold", case["end_height"]

    ratio = squat_step / float(descend_steps)
    height = case["start_height"] + ratio * (case["end_height"] - case["start_height"])
    return "descend", height


def _clip_height_command(env_cfg, height):
    min_offset, max_offset = env_cfg.commands.ranges.height
    min_height = env_cfg.rewards.base_height_target + min_offset
    max_height = env_cfg.rewards.base_height_target + max_offset
    return min(max(height, min_height), max_height)


def _set_commands(env, env_cfg, x_vel, y_vel, yaw_vel, height):
    height_command = _clip_height_command(env_cfg, height)
    env.commands[:, 0] = x_vel
    env.commands[:, 1] = y_vel
    env.commands[:, 2] = yaw_vel
    if env.commands.shape[1] > 4:
        env.commands[:, 4] = height_command
    return height_command


def _measured_base_height(env):
    if hasattr(env, "feet_pos"):
        base_to_ankle = torch.max(
            env.root_states[:, 2].unsqueeze(1) - env.feet_pos[:, :, 2], dim=1
        ).values
        return base_to_ankle + getattr(env.cfg.asset, "ankle_sole_distance", 0.0)
    return env.root_states[:, 2]


def _configure_eval_env(env_cfg, num_envs):
    env_cfg.env.num_envs = num_envs
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.mesh_type = "plane"
    _set_if_present(env_cfg.terrain, "measure_heights", False)

    env_cfg.noise.add_noise = False
    env_cfg.commands.heading_command = False
    _set_if_present(env_cfg.commands, "use_random", False)
    _set_if_present(env_cfg.commands, "curriculum", False)

    for name in (
        "use_random",
        "randomize_friction",
        "push_robots",
        "disturbance",
        "randomize_payload_mass",
        "randomize_com_displacement",
        "randomize_body_displacement",
        "randomize_link_mass",
        "randomize_restitution",
        "randomize_kp",
        "randomize_kd",
        "randomize_initial_joint_pos",
        "randomize_joint_injection",
        "randomize_actuation_offset",
        "delay",
    ):
        _set_if_present(env_cfg.domain_rand, name, False)

    env_cfg.asset.self_collision = 0
    _set_if_present(env_cfg.env, "upper_teleop", False)
    return env_cfg


def _finalize_episode(case, env_id, records, done_step, time_out):
    xs = [r["x"] for r in records]
    ys = [r["y"] for r in records]
    yaws = [r["yaw"] for r in records]
    x_vels = [r["x_vel"] for r in records]
    y_vels = [r["y_vel"] for r in records]
    yaw_vels = [r["yaw_vel"] for r in records]
    height_errors = [r["height_error"] for r in records]
    roll_abs = [abs(r["roll"]) for r in records]
    pitch_abs = [abs(r["pitch"]) for r in records]

    x0 = xs[0] if xs else 0.0
    y0 = ys[0] if ys else 0.0
    yaw0 = yaws[0] if yaws else 0.0
    tail_count = max(1, int(1.0 / case["dt"]))

    return {
        "env_id": env_id,
        "done_step": done_step,
        "time_out": bool(time_out),
        "completed_case": done_step is None,
        "duration_s": len(records) * case["dt"],
        "final_x_drift": (xs[-1] - x0) if xs else None,
        "final_y_drift": (ys[-1] - y0) if ys else None,
        "final_yaw_drift": (yaws[-1] - yaw0) if yaws else None,
        "min_x_drift": (min(xs) - x0) if xs else None,
        "max_x_drift": (max(xs) - x0) if xs else None,
        "max_abs_y_drift": _max_abs([(y - y0) for y in ys]),
        "mean_x_vel": _mean(x_vels),
        "min_x_vel": min(x_vels) if x_vels else None,
        "max_abs_y_vel": _max_abs(y_vels),
        "max_abs_yaw_vel": _max_abs(yaw_vels),
        "mean_height_abs_error": _mean(height_errors),
        "tail_height_abs_error": _tail_mean(height_errors, tail_count),
        "max_roll_abs": max(roll_abs) if roll_abs else None,
        "max_pitch_abs": max(pitch_abs) if pitch_abs else None,
        "first_records": records[:5],
        "last_records": records[-5:],
    }


def _summarize_episodes(episodes):
    count = len(episodes)
    keys = [
        "final_x_drift",
        "final_y_drift",
        "final_yaw_drift",
        "min_x_drift",
        "max_x_drift",
        "max_abs_y_drift",
        "mean_x_vel",
        "min_x_vel",
        "max_abs_y_vel",
        "max_abs_yaw_vel",
        "mean_height_abs_error",
        "tail_height_abs_error",
        "max_roll_abs",
        "max_pitch_abs",
    ]
    summary = {
        "episodes": count,
        "completion_rate": sum(1 for ep in episodes if ep["completed_case"]) / count,
        "timeout_rate": sum(1 for ep in episodes if ep["time_out"]) / count,
    }
    for key in keys:
        values = [ep[key] for ep in episodes if ep[key] is not None]
        summary[f"mean_{key}"] = _mean(values)
        summary[f"worst_abs_{key}"] = _max_abs(values)
    return summary


def evaluate_case(env, env_cfg, policy, case, num_envs, x_vel, y_vel, yaw_vel):
    case = dict(case)
    case["dt"] = env.dt
    total_steps = max(
        int(case["settle_time"] / env.dt)
        + int(case["descend_time"] / env.dt)
        + int(case["hold_time"] / env.dt),
        1,
    )

    env_ids = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(env_ids)
    _zero_random_upper_body(env)
    _set_commands(env, env_cfg, x_vel, y_vel, yaw_vel, case["start_height"])
    obs = env.get_observations()

    per_env_records = [[] for _ in range(num_envs)]
    active = torch.ones(num_envs, dtype=torch.bool, device=env.device)
    episodes = []

    for step in range(total_steps):
        phase, target_height = _get_case_height(case, step, env.dt)
        height_command = _set_commands(env, env_cfg, x_vel, y_vel, yaw_vel, target_height)
        obs = env.get_observations()

        with torch.no_grad():
            _zero_random_upper_body(env)
            actions = policy(obs.detach())
            _set_commands(env, env_cfg, x_vel, y_vel, yaw_vel, target_height)
            _zero_random_upper_body(env)
            obs, _, _, dones, _, _, _ = env.step(actions.detach())

        measured_height = _measured_base_height(env)
        roll, pitch, yaw = euler_from_quaternion(env.base_quat)
        active_ids = active.nonzero(as_tuple=False).flatten()
        for env_id in active_ids.tolist():
            per_env_records[env_id].append(
                {
                    "step": step,
                    "time": step * env.dt,
                    "phase": phase,
                    "target_height": float(target_height),
                    "height_command": float(height_command),
                    "measured_height": _to_float(measured_height[env_id]),
                    "height_error": abs(_to_float(measured_height[env_id]) - float(height_command)),
                    "x": _to_float(env.root_states[env_id, 0]),
                    "y": _to_float(env.root_states[env_id, 1]),
                    "z": _to_float(env.root_states[env_id, 2]),
                    "roll": _to_float(roll[env_id]),
                    "pitch": _to_float(pitch[env_id]),
                    "yaw": _to_float(yaw[env_id]),
                    "x_vel": _to_float(env.base_lin_vel[env_id, 0]),
                    "y_vel": _to_float(env.base_lin_vel[env_id, 1]),
                    "yaw_vel": _to_float(env.base_ang_vel[env_id, 2]),
                }
            )

        done_ids = ((dones > 0) & active).nonzero(as_tuple=False).flatten()
        for env_id in done_ids.tolist():
            episodes.append(
                _finalize_episode(
                    case,
                    env_id,
                    per_env_records[env_id],
                    step,
                    env.time_out_buf[env_id].item() if hasattr(env, "time_out_buf") else False,
                )
            )
            active[env_id] = False

        if not active.any():
            break

    for env_id in active.nonzero(as_tuple=False).flatten().tolist():
        episodes.append(_finalize_episode(case, env_id, per_env_records[env_id], None, False))

    return {
        "case": {k: v for k, v in case.items() if k != "dt"},
        "total_steps": total_steps,
        "summary": _summarize_episodes(episodes),
        "episodes": episodes,
    }


def write_results(args, train_cfg, results):
    if args.output:
        output = Path(args.output)
    else:
        experiment = train_cfg.runner.experiment_name or args.task
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = Path(LEGGED_GYM_ROOT_DIR) / "logs" / experiment / "squat_eval" / f"squat_eval_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return output


def main(args):
    args.headless = True
    args.resume = True

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    train_cfg.runner.resume = True
    env_cfg = _configure_eval_env(env_cfg, args.num_envs)
    min_height = env_cfg.rewards.base_height_target + env_cfg.commands.ranges.height[0]
    max_height = env_cfg.rewards.base_height_target + env_cfg.commands.ranges.height[1]

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    results = {
        "task": args.task,
        "num_envs": args.num_envs,
        "x_vel_command": args.x_vel,
        "y_vel_command": args.y_vel,
        "yaw_vel_command": args.yaw_vel,
        "dt": env.dt,
        "base_height_target": env_cfg.rewards.base_height_target,
        "height_command_range_abs": [min_height, max_height],
        "height_command_range_offset": list(env_cfg.commands.ranges.height),
        "load_run": train_cfg.runner.load_run,
        "checkpoint": train_cfg.runner.checkpoint,
        "cases": [],
    }

    for case in DEFAULT_CASES:
        print(
            f"Evaluating {case['name']}: "
            f"{case['start_height']:.3f}->{case['end_height']:.3f}, "
            f"descend={case['descend_time']}s hold={case['hold_time']}s"
        )
        case_result = evaluate_case(
            env,
            env_cfg,
            policy,
            case,
            args.num_envs,
            args.x_vel,
            args.y_vel,
            args.yaw_vel,
        )
        results["cases"].append(case_result)
        summary = case_result["summary"]
        print(
            f"  completion={summary['completion_rate']:.2f}, "
            f"mean_x_drift={summary['mean_final_x_drift']:.3f}, "
            f"worst_x_drift={summary['worst_abs_final_x_drift']:.3f}, "
            f"mean_min_x_drift={summary['mean_min_x_drift']:.3f}, "
            f"height_err={summary['mean_mean_height_abs_error']:.3f}, "
            f"max_pitch={summary['worst_abs_max_pitch_abs']:.3f}"
        )

    output = write_results(args, train_cfg, results)
    print(f"Wrote squat evaluation JSON: {output}")


if __name__ == "__main__":
    has_task_arg = any(arg == "--task" or arg.startswith("--task=") for arg in sys.argv[1:])
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--x_vel", type=float, default=0.0)
    parser.add_argument("--y_vel", type=float, default=0.0)
    parser.add_argument("--yaw_vel", type=float, default=0.0)
    parser.add_argument("--output", default=None)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    gym_args = get_args()
    if not has_task_arg:
        gym_args.task = "elf3_dof31"
    for key, value in vars(known).items():
        setattr(gym_args, key, value)
    main(gym_args)
