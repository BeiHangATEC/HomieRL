import os
from pathlib import Path
import sys

REPO_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_PACKAGE_ROOT.parent
for package_root in (REPO_PACKAGE_ROOT, WORKSPACE_ROOT / "rsl_rl"):
    if package_root.exists() and str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry

import numpy as np
import torch


EXPORT_POLICY = False
RECORD_FRAMES = False
MOVE_CAMERA = False


def zero_random_upper_body(env):
    if hasattr(env, "action_curriculum_ratio"):
        env.action_curriculum_ratio = 0.0

    if hasattr(env, "current_upper_actions"):
        env.current_upper_actions.zero_()

    if hasattr(env, "random_upper_actions"):
        env.random_upper_actions.zero_()

    if hasattr(env, "delta_upper_actions"):
        env.delta_upper_actions.zero_()


def get_squat_height(step, descend_steps, start_height, end_height):
    if descend_steps <= 0 or step >= descend_steps:
        return end_height

    ratio = step / float(descend_steps)

    return start_height + ratio * (end_height - start_height)


def height_to_command(env_cfg, height):
    min_offset, max_offset = env_cfg.commands.ranges.height
    min_height = env_cfg.rewards.base_height_target + min_offset
    max_height = env_cfg.rewards.base_height_target + max_offset
    return min(max(height, min_height), max_height)


def set_motion_commands(env, env_cfg, x_vel, y_vel, yaw_vel, height):
    height_command = height_to_command(env_cfg, height)
    env.commands[:, 0] = x_vel
    env.commands[:, 1] = y_vel
    env.commands[:, 2] = yaw_vel
    env.commands[:, 4] = height_command
    return height_command


def play_squat(
    args,
    x_vel=0.0,
    y_vel=0.0,
    yaw_vel=0.0,
    start_height=0.945,
    end_height=0.765,
    descend_time=6.0,
    hold_time=4.0,
    settle_time=2.0,
):

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)


    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)

    # terrain
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 8
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_init_terrain_level = 9
    env_cfg.terrain.mesh_type = "plane"


    # disable noise/randomization
    env_cfg.noise.add_noise = False

    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.domain_rand.randomize_body_displacement = False


    env_cfg.commands.heading_command = False
    env_cfg.commands.use_random = False


    env_cfg.asset.self_collision = 0

    if hasattr(env_cfg.env, "upper_teleop"):
        env_cfg.env.upper_teleop = False


    env, _ = task_registry.make_env(
        name=args.task,
        args=args,
        env_cfg=env_cfg
    )


    set_motion_commands(
        env,
        env_cfg,
        x_vel,
        y_vel,
        yaw_vel,
        start_height,
    )


    zero_random_upper_body(env)


    obs = env.get_observations()


    train_cfg.runner.resume = True


    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root=None
    )


    # 使用确定性策略
    policy = ppo_runner.get_inference_policy(
        device=env.device,
        #deterministic=True
    )


    if EXPORT_POLICY:

        path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies"
        )

        export_policy_as_jit(
            ppo_runner.alg.actor_critic,
            path
        )

        print("Exported policy:", path)


    print(policy)


    camera_position = np.array(
        env_cfg.viewer.pos,
        dtype=np.float64
    )

    camera_vel = np.array(
        [1.,1.,0.]
    )

    camera_direction = (
        np.array(env_cfg.viewer.lookat)
        -
        np.array(env_cfg.viewer.pos)
    )


    env.reset_idx(
        torch.arange(env.num_envs).to(env.device)
    )

    zero_random_upper_body(env)
    set_motion_commands(
        env,
        env_cfg,
        x_vel,
        y_vel,
        yaw_vel,
        start_height,
    )

    obs = env.get_observations()

    settle_steps = int(settle_time / env.dt)

    descend_steps = int(descend_time / env.dt)

    hold_steps = int(hold_time / env.dt)

    total_steps = max(
        settle_steps + descend_steps + hold_steps,
        1
    )


    for step in range(total_steps):

        if step < settle_steps:
            phase = "settle"
            height = start_height
        else:
            phase = "squat"
            squat_step = step - settle_steps
            height = get_squat_height(
                squat_step,
                descend_steps,
                start_height,
                end_height
            )


        zero_random_upper_body(env)

        height_command = set_motion_commands(
            env,
            env_cfg,
            x_vel,
            y_vel,
            yaw_vel,
            height,
        )

        obs = env.get_observations()

        actions = policy(
            obs.detach()
        )

        zero_random_upper_body(env)

        obs, _, _, _, _, _, _ = env.step(
            actions.detach()
        )


        if step % max(int(0.5/env.dt),1)==0:

            print(
                f"{phase} step {step}/{total_steps}, "
                f"target height={height:.3f}, "
                f"height command={height_command:.3f}"
            )


        if MOVE_CAMERA:

            camera_position += camera_vel * env.dt

            env.set_camera(
                camera_position,
                camera_position + camera_direction
            )



if __name__ == "__main__":

    args = get_args()


    play_squat(
        args,
        x_vel=0.0,
        y_vel=0.0,
        yaw_vel=0.0,
        start_height=0.945,
        end_height=0.765,
        descend_time=6.0,
        hold_time=4.0,
        settle_time=2.0,
    )