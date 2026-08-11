#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import sys

REPO_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_PACKAGE_ROOT.parent
for package_root in (REPO_PACKAGE_ROOT, WORKSPACE_ROOT / "rsl_rl"):
    if package_root.exists() and str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

import isaacgym  # noqa: F401
from isaacgym import gymtorch
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


def _set_if_present(obj, name, value):
    if hasattr(obj, name):
        setattr(obj, name, value)


def _to_float(value):
    return float(value.detach().cpu().item())


def _to_list(value):
    return value.detach().cpu().tolist()


def _zero_upper_body_buffers(env):
    for name in (
        "current_upper_actions",
        "random_upper_actions",
        "delta_upper_actions",
    ):
        if hasattr(env, name):
            getattr(env, name).zero_()
    if hasattr(env, "action_curriculum_ratio"):
        env.action_curriculum_ratio = 0.0


def _find_body_index(env, body_name):
    if not body_name:
        return None
    try:
        return env.body_names.index(body_name)
    except ValueError:
        return None


def _mean_tail(values, tail):
    if not values:
        return None
    selected = values[-min(tail, len(values)):]
    return sum(selected) / len(selected)


def build_measurement_env(args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = 1
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.noise.add_noise = False
    env_cfg.commands.curriculum = False
    env_cfg.commands.heading_command = False
    env_cfg.domain_rand.use_random = False

    for name in (
        "randomize_joint_injection",
        "randomize_actuation_offset",
        "randomize_payload_mass",
        "randomize_com_displacement",
        "randomize_body_displacement",
        "randomize_link_mass",
        "randomize_friction",
        "randomize_restitution",
        "randomize_kp",
        "randomize_kd",
        "randomize_initial_joint_pos",
        "push_robots",
        "delay",
        "disturbance",
    ):
        _set_if_present(env_cfg.domain_rand, name, False)

    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    return env, env_cfg


def measure(args):
    env, env_cfg = build_measurement_env(args)
    device = env.device
    env_ids = torch.arange(env.num_envs, device=device)

    env.reset_idx(env_ids)
    env.commands.zero_()
    if env.commands.shape[1] > 4:
        env.commands[:, 4] = 0.0
    _zero_upper_body_buffers(env)

    env.root_states[:, 7:13] = 0.0
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))

    action_dim = len(env.lower_dof_indices) if hasattr(env, "lower_dof_indices") else env.num_actions
    zero_actions = torch.zeros((env.num_envs, action_dim), device=device)
    left_foot_idx = _find_body_index(env, getattr(env_cfg.asset, "left_foot_name", None))
    right_foot_idx = _find_body_index(env, getattr(env_cfg.asset, "right_foot_name", None))
    upper_body_idx = _find_body_index(env, getattr(env_cfg.asset, "upper_body_link", None))
    imu_idx = getattr(env, "imu_index", None)

    left_knee_idx = _find_body_index(env, "l_knee_y_link")
    right_knee_idx = _find_body_index(env, "r_knee_y_link")

    records = []
    for step in range(args.steps):
        _zero_upper_body_buffers(env)
        env.commands.zero_()
        if env.commands.shape[1] > 4:
            env.commands[:, 4] = 0.0
        env.step(zero_actions)

        if step < args.warmup_steps:
            continue

        root_z = _to_float(env.root_states[0, 2])
        projected_gravity_xy_norm = _to_float(torch.linalg.norm(env.projected_gravity[0, :2]))
        left_foot_z = _to_float(env.rigid_body_states[0, left_foot_idx, 2]) if left_foot_idx is not None else None
        right_foot_z = _to_float(env.rigid_body_states[0, right_foot_idx, 2]) if right_foot_idx is not None else None
        upper_body_z = _to_float(env.rigid_body_states[0, upper_body_idx, 2]) if upper_body_idx is not None else None
        imu_z = _to_float(env.rigid_body_states[0, imu_idx, 2]) if imu_idx is not None else None

        feet_contact_force_norm = None
        if hasattr(env, "feet_indices"):
            feet_contact_force_norm = _to_list(torch.linalg.norm(env.contact_forces[0, env.feet_indices], dim=-1))

        feet_lateral_distance = None
        if left_foot_idx is not None and right_foot_idx is not None:
            feet_lateral_distance = abs(
                _to_float(env.rigid_body_states[0, left_foot_idx, 1] - env.rigid_body_states[0, right_foot_idx, 1])
            )

        knee_lateral_distance = None
        if left_knee_idx is not None and right_knee_idx is not None:
            knee_lateral_distance = abs(
                _to_float(env.rigid_body_states[0, left_knee_idx, 1] - env.rigid_body_states[0, right_knee_idx, 1])
            )

        records.append(
            {
                "step": step,
                "root_z": root_z,
                "upper_body_z": upper_body_z,
                "imu_z": imu_z,
                "left_foot_z": left_foot_z,
                "right_foot_z": right_foot_z,
                "projected_gravity_xy_norm": projected_gravity_xy_norm,
                "feet_contact_force_norm": feet_contact_force_norm,
                "feet_lateral_distance": feet_lateral_distance,
                "knee_lateral_distance": knee_lateral_distance,
                "dof_pos": _to_list(env.dof_pos[0]),
                "dof_vel_norm": _to_float(torch.linalg.norm(env.dof_vel[0])),
            }
        )

    root_values = [r["root_z"] for r in records]
    upper_values = [r["upper_body_z"] for r in records if r["upper_body_z"] is not None]
    imu_values = [r["imu_z"] for r in records if r["imu_z"] is not None]
    foot_values = [v for r in records for v in (r["left_foot_z"], r["right_foot_z"]) if v is not None]
    gravity_tilt = [r["projected_gravity_xy_norm"] for r in records]
    dof_vel_norm = [r["dof_vel_norm"] for r in records]
    feet_lat = [r["feet_lateral_distance"] for r in records if r["feet_lateral_distance"] is not None]
    knee_lat = [r["knee_lateral_distance"] for r in records if r["knee_lateral_distance"] is not None]

    tail = min(args.tail_steps, len(records))
    ankle_sole_distance = getattr(env_cfg.asset, "ankle_sole_distance", 0.0)
    summary = {
        "task": args.task,
        "num_records": len(records),
        "warmup_steps": args.warmup_steps,
        "tail_steps": tail,
        "configured_init_root_z": env_cfg.init_state.pos[2],
        "configured_base_height_target": env_cfg.rewards.base_height_target,
        "configured_ankle_sole_distance": ankle_sole_distance,
        "root_z_tail_mean": _mean_tail(root_values, tail),
        "upper_body_z_tail_mean": _mean_tail(upper_values, tail),
        "imu_z_tail_mean": _mean_tail(imu_values, tail),
        "foot_z_tail_mean": _mean_tail(foot_values, tail * 2),
        "root_minus_foot_z_tail_mean": None,
        "reward_base_height_tail_mean": None,
        "projected_gravity_xy_norm_tail_mean": _mean_tail(gravity_tilt, tail),
        "dof_vel_norm_tail_mean": _mean_tail(dof_vel_norm, tail),
        "feet_lateral_distance_tail_mean": _mean_tail(feet_lat, tail),
        "knee_lateral_distance_tail_mean": _mean_tail(knee_lat, tail),
        "dof_names": list(env.dof_names),
        "body_names": list(env.body_names),
        "default_joint_angles": dict(env_cfg.init_state.default_joint_angles),
        "last_record": records[-1] if records else None,
    }
    if summary["root_z_tail_mean"] is not None and summary["foot_z_tail_mean"] is not None:
        summary["root_minus_foot_z_tail_mean"] = summary["root_z_tail_mean"] - summary["foot_z_tail_mean"]
        summary["reward_base_height_tail_mean"] = summary["root_minus_foot_z_tail_mean"] + ankle_sole_distance

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    has_task_arg = any(arg == "--task" or arg.startswith("--task=") for arg in sys.argv[1:])
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--warmup_steps", type=int, default=400)
    parser.add_argument("--tail_steps", type=int, default=200)
    parser.add_argument(
        "--output",
        default=str(Path(LEGGED_GYM_ROOT_DIR) / "logs/elf3_dof31/stand_height_measurement.json"),
    )
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    gym_args = get_args()
    if not has_task_arg:
        gym_args.task = "elf3_dof31"
    for key, value in vars(known).items():
        setattr(gym_args, key, value)
    measure(gym_args)