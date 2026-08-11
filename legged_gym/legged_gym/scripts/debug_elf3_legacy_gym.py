#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

import isaacgym  # noqa: F401
from isaacgym import gymtorch
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


def suppress_upper(env):
    env.action_curriculum_ratio = 0.0
    env.current_upper_actions.zero_()
    env.random_upper_actions.zero_()
    env.delta_upper_actions.zero_()


def tensor(value):
    return value.detach().cpu().tolist()


def main(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.noise.add_noise = False
    env_cfg.commands.curriculum = False
    env_cfg.commands.use_random = False
    env_cfg.asset.self_collision = 0
    env_cfg.domain_rand.use_random = False
    for name in (
        "randomize_friction", "push_robots", "disturbance", "randomize_payload_mass",
        "randomize_body_displacement", "randomize_link_mass", "randomize_restitution",
        "randomize_kp", "randomize_kd", "randomize_initial_joint_pos",
        "randomize_joint_injection", "randomize_actuation_offset", "delay",
    ):
        if hasattr(env_cfg.domain_rand, name):
            setattr(env_cfg.domain_rand, name, False)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint
    runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None)
    checkpoint_policy = runner.get_inference_policy(device=env.device)
    jit_policy = torch.jit.load(args.policy, map_location=env.device).eval()

    env.commands.zero_()
    env.commands[:, 4] = args.height
    suppress_upper(env)
    env.root_states[:, 7:13] = 0.0
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
    env.compute_observations()
    obs = env.get_observations()
    records = []
    max_policy_diff = 0.0
    for step in range(args.steps):
        suppress_upper(env)
        env.commands.zero_()
        env.commands[:, 4] = args.height
        with torch.inference_mode():
            checkpoint_action = checkpoint_policy(obs)
            jit_action = jit_policy(obs)
        max_policy_diff = max(max_policy_diff, float(torch.max(torch.abs(checkpoint_action - jit_action))))
        imu_state = env.rigid_body_states[:, env.imu_index]
        record = {
            "step": step,
            "root": tensor(env.root_states[0]),
            "feet_state": tensor(env.rigid_body_states[0, env.feet_indices]),
            "feet_contact_force": tensor(env.contact_forces[0, env.feet_indices]),
            "imu_quat_xyzw": tensor(imu_state[0, 3:7]),
            "imu_world_ang_vel": tensor(imu_state[0, 10:13]),
            "imu_body_ang_vel": tensor(env.base_ang_vel[0]),
            "projected_gravity": tensor(env.projected_gravity[0]),
            "dof_pos": tensor(env.dof_pos[0]),
            "dof_vel": tensor(env.dof_vel[0]),
            "observation": tensor(obs[0]),
            "checkpoint_action": tensor(checkpoint_action[0]),
            "jit_action": tensor(jit_action[0]),
        }
        obs, _, _, done, _, _, _ = env.step(checkpoint_action)
        record["mixed_control_output"] = tensor(env.torques[0])
        record["joint_pos_target"] = tensor(env.joint_pos_target[0])
        record["done"] = bool(done[0])
        records.append(record)

    result = {
        "checkpoint": str(Path(LEGGED_GYM_ROOT_DIR) / "logs" / train_cfg.runner.experiment_name / args.load_run / f"model_{args.checkpoint}.pt"),
        "policy": args.policy,
        "dof_names": list(env.dof_names),
        "body_names": list(env.body_names),
        "lower_dof_indices": tensor(env.lower_dof_indices),
        "upper_dof_indices": tensor(env.upper_dof_indices),
        "drive_modes": env.gym.get_actor_dof_properties(env.envs[0], env.actor_handles[0])["driveMode"].tolist(),
        "max_checkpoint_jit_action_diff": max_policy_diff,
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--height", type=float, default=0.82)
    parser.add_argument("--policy", default=str(Path(LEGGED_GYM_ROOT_DIR) / "logs/elf3_dof31/exported/policies/policy.pt"))
    parser.add_argument("--output", default=str(Path(LEGGED_GYM_ROOT_DIR) / "logs/elf3_dof31/legacy_gym_evidence.json"))
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    gym_args = get_args()
    for key, value in vars(known).items():
        setattr(gym_args, key, value)
    main(gym_args)
