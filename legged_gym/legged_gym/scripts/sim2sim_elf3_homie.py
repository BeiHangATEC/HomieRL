#!/usr/bin/env python3
import argparse
import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "resources/robots/elf3_dof31/mjcf/elf3_homie.xml"
DEFAULT_POLICY = ROOT / "logs/elf3_dof31/exported/policies/policy.pt"
DEFAULT_METRICS = ROOT / "logs/elf3_dof31/sim2sim_elf3_homie_metrics.json"
JOINT_ORDER = [
    "head_z_joint", "head_y_joint",
    "l_shoulder_y_joint", "l_shoulder_x_joint", "l_shoulder_z_joint", "l_elbow_y_joint",
    "l_wrist_x_joint", "l_wrist_y_joint", "l_wrist_z_joint",
    "r_shoulder_y_joint", "r_shoulder_x_joint", "r_shoulder_z_joint", "r_elbow_y_joint",
    "r_wrist_x_joint", "r_wrist_y_joint", "r_wrist_z_joint",
    "waist_y_joint", "waist_x_joint", "waist_z_joint",
    "l_hip_y_joint", "l_hip_x_joint", "l_hip_z_joint", "l_knee_y_joint", "l_ankle_y_joint", "l_ankle_x_joint",
    "r_hip_y_joint", "r_hip_x_joint", "r_hip_z_joint", "r_knee_y_joint", "r_ankle_y_joint", "r_ankle_x_joint",
]
ACTION_JOINT_ORDER = [
    "l_hip_y_joint", "l_hip_x_joint", "l_hip_z_joint", "l_knee_y_joint", "l_ankle_y_joint", "l_ankle_x_joint",
    "r_hip_y_joint", "r_hip_x_joint", "r_hip_z_joint", "r_knee_y_joint", "r_ankle_y_joint", "r_ankle_x_joint",
]
DEFAULT_ANGLES = {name: 0.0 for name in JOINT_ORDER}
for side in ("l", "r"):
    DEFAULT_ANGLES[f"{side}_hip_y_joint"] = -0.10
    DEFAULT_ANGLES[f"{side}_knee_y_joint"] = 0.30
    DEFAULT_ANGLES[f"{side}_ankle_y_joint"] = -0.20
STIFFNESS = {"hip_z": 90, "hip_x": 90, "hip_y": 100, "knee": 140, "ankle": 35,
             "waist": 180, "head": 20, "shoulder": 80, "elbow": 50, "wrist": 20}
DAMPING = {"hip_z": 2.0, "hip_x": 2.0, "hip_y": 2.5, "knee": 4.0, "ankle": 1.5,
           "waist": 4.0, "head": 0.5, "shoulder": 2.0, "elbow": 1.0, "wrist": 0.5}
PHYSICS_DT = 0.005
DECIMATION = 4
POLICY_DT = PHYSICS_DT * DECIMATION
ACTION_SCALE = 0.25
ACTION_CLIP = 100.0
OBS_CLIP = 100.0
BASE_HEIGHT_TARGET = 0.82


def named_id(model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"MuJoCo 模型缺少对象: {name}")
    return object_id


def gain(name, table):
    for key, value in table.items():
        if key in name:
            return value
    raise RuntimeError(f"关节 {name} 没有 PD 参数")


def make_mapping(model):
    joint_ids = np.asarray([named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_ORDER])
    actuator_ids = np.asarray([named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in JOINT_ORDER])
    if len(set(joint_ids.tolist())) != 31 or len(set(actuator_ids.tolist())) != 31:
        raise RuntimeError("关节或 actuator 名称映射存在重复")
    if any(model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE for joint_id in joint_ids):
        raise RuntimeError("31 个活动关节必须全部是单自由度 hinge")
    if not np.array_equal(model.actuator_trnid[actuator_ids, 0], joint_ids):
        raise RuntimeError("motor 名称与其驱动关节不一致")
    qpos = model.jnt_qposadr[joint_ids]
    qvel = model.jnt_dofadr[joint_ids]
    action_to_full = np.asarray([JOINT_ORDER.index(name) for name in ACTION_JOINT_ORDER])
    return joint_ids, qpos, qvel, actuator_ids, action_to_full


def body_velocity(model, data, body_id):
    velocity = np.empty(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 1)
    return velocity


def rotate_world_to_body(quat_wxyz, vector):
    inverse = np.asarray(quat_wxyz, dtype=np.float64).copy()
    inverse[1:] *= -1
    result = np.empty(3, dtype=np.float64)
    mujoco.mju_rotVecQuat(result, np.asarray(vector, dtype=np.float64), inverse)
    return result


def make_frame(model, data, torso_id, qpos, qvel, default, applied_action, command, height):
    angular_velocity = body_velocity(model, data, torso_id)[:3]
    projected_gravity = rotate_world_to_body(data.xquat[torso_id], [0.0, 0.0, -1.0])
    frame = np.concatenate((
        command * np.asarray([2.0, 2.0, 0.5]),
        [height], angular_velocity * 0.5, projected_gravity,
        data.qpos[qpos] - default, data.qvel[qvel] * 0.05, applied_action,
    )).astype(np.float32)
    if frame.shape != (84,) or not np.isfinite(frame).all():
        raise RuntimeError(f"单帧观测应为有限 84 维，实际 shape={frame.shape}")
    return frame


def load_policy(path):
    if not path.is_file():
        raise FileNotFoundError(f"TorchScript policy 不存在: {path}")
    try:
        policy = torch.jit.load(str(path), map_location="cpu").eval()
    except Exception as error:
        raise RuntimeError(f"无法加载 TorchScript policy {path}: {error}") from error
    dummy = torch.zeros((1, 504), dtype=torch.float32)
    try:
        with torch.inference_mode():
            output = policy(dummy)
    except Exception as error:
        raise RuntimeError(f"policy 输入契约不兼容：要求 float32 (batch, 504)，试运行失败: {error}") from error
    if not isinstance(output, torch.Tensor) or tuple(output.shape) != (1, 12):
        shape = tuple(output.shape) if isinstance(output, torch.Tensor) else type(output).__name__
        raise RuntimeError(f"policy 输出契约不兼容：要求 (1, 12)，实际 {shape}")
    if not torch.isfinite(output).all():
        raise RuntimeError("policy 对零输入产生 NaN/Inf")
    return policy


def roll_pitch(quat_wxyz):
    w, x, y, z = quat_wxyz
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
    return math.degrees(roll), math.degrees(pitch)


def simulate(args):
    if args.duration <= 0:
        raise ValueError("--duration 必须大于 0")
    model = mujoco.MjModel.from_xml_path(str(args.model))
    if model.nu != 31 or model.nsensor < 3 or not math.isclose(model.opt.timestep, PHYSICS_DT):
        raise RuntimeError(f"模型契约不符: nu={model.nu}, nsensor={model.nsensor}, timestep={model.opt.timestep}")
    _, qpos, qvel, actuator_ids, action_to_full = make_mapping(model)
    torso_id = named_id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    floor_id = named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_ids = {named_id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in ("l_ankle_x_link", "r_ankle_x_link")}
    policy = load_policy(args.policy)

    default = np.asarray([DEFAULT_ANGLES[name] for name in JOINT_ORDER])
    kp = np.asarray([gain(name, STIFFNESS) for name in JOINT_ORDER])
    kd = np.asarray([gain(name, DAMPING) for name in JOINT_ORDER])
    effort = np.max(np.abs(model.actuator_ctrlrange[actuator_ids]), axis=1)
    data = mujoco.MjData(model)
    root_joint = named_id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    root_qpos = model.jnt_qposadr[root_joint]
    data.qpos[root_qpos:root_qpos + 7] = [0.0, 0.0, args.root_height, 1.0, 0.0, 0.0, 0.0]
    data.qpos[qpos] = default
    mujoco.mj_forward(model, data)

    command = np.asarray([args.vx, args.vy, args.yaw], dtype=np.float32)
    applied_action = np.zeros(12, dtype=np.float32)
    history = np.zeros((6, 84), dtype=np.float32)
    if args.legacy_policy:
        initial_frame = make_frame(model, data, torso_id, qpos, qvel, default, applied_action, command, args.body_height)
        history[-2:] = initial_frame
    viewer_handle = None
    if not args.headless:
        from mujoco import viewer
        viewer_handle = viewer.launch_passive(model, data)
    policy_steps = int(math.ceil(args.duration / POLICY_DT))
    wall_start = time.perf_counter()
    heights, rolls, pitches, velocities, actions, torques = [], [], [], [], [], []
    nonfoot_contact_steps = completed_physics_steps = saturation_count = saturation_total = 0
    finite = True

    for _ in range(policy_steps):
        if viewer_handle is not None and not viewer_handle.is_running():
            break
        frame = make_frame(model, data, torso_id, qpos, qvel, default, applied_action, command, args.body_height)
        history[:-1] = history[1:]
        history[-1] = frame
        observation = np.clip(history.reshape(1, 504), -OBS_CLIP, OBS_CLIP)
        with torch.inference_mode():
            output = policy(torch.from_numpy(np.ascontiguousarray(observation)))
        action = output.detach().cpu().numpy()[0]
        if action.shape != (12,) or not np.isfinite(action).all():
            finite = False
            break
        action = np.clip(action, -ACTION_CLIP, ACTION_CLIP).astype(np.float32)
        applied_action = action.copy()
        actions.extend(action.tolist())
        target = default.copy()
        target[action_to_full] += ACTION_SCALE * action

        for _ in range(DECIMATION):
            raw_torque = kp * (target - data.qpos[qpos]) - kd * data.qvel[qvel]
            if not np.isfinite(raw_torque).all():
                finite = False
                break
            saturation_count += int(np.count_nonzero(np.abs(raw_torque) >= effort))
            saturation_total += raw_torque.size
            torque = np.clip(raw_torque, -effort, effort)
            torques.extend(torque.tolist())
            data.ctrl[actuator_ids] = torque
            mujoco.mj_step(model, data)
            completed_physics_steps += 1
            finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
            if not finite:
                break
            heights.append(float(data.qpos[root_qpos + 2]))
            roll, pitch = roll_pitch(data.xquat[torso_id])
            rolls.append(roll)
            pitches.append(pitch)
            velocity = body_velocity(model, data, torso_id)
            velocities.append([velocity[3], velocity[4], velocity[2]])
            floor_bodies = set()
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                if floor_id in (contact.geom1, contact.geom2):
                    other = contact.geom2 if contact.geom1 == floor_id else contact.geom1
                    floor_bodies.add(int(model.geom_bodyid[other]))
            nonfoot_contact_steps += int(any(body not in foot_ids for body in floor_bodies))
            if viewer_handle is not None:
                viewer_handle.sync()
                delay = wall_start + completed_physics_steps * PHYSICS_DT - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
        # #region debug-point B:policy-step-state
        _payload = {"sessionId": "old-policy-sim2sim", "runId": "post-fix", "hypothesisId": "B,C,D,E", "location": "sim2sim_elf3_homie.py:policy-loop", "msg": "[DEBUG] Policy-step state", "data": {"step": completed_physics_steps // DECIMATION, "root_z": float(data.qpos[root_qpos + 2]), "roll": float(rolls[-1]) if rolls else None, "pitch": float(pitches[-1]) if pitches else None, "body_ang_vel": body_velocity(model, data, torso_id)[:3].tolist(), "action_max": float(np.max(np.abs(action))), "torque_max": float(np.max(np.abs(torque))), "contacts": int(data.ncon)}, "ts": int(time.time() * 1000)}
        _request = __import__("urllib.request").request.Request("http://127.0.0.1:7777/event", data=json.dumps(_payload).encode(), headers={"Content-Type": "application/json"})
        try:
            __import__("urllib.request").request.urlopen(_request, timeout=0.05).read()
        except Exception:
            pass
        # #endregion
        if not finite:
            break

    if viewer_handle is not None:
        viewer_handle.close()
    velocity_array = np.asarray(velocities)
    tracking_rmse = np.sqrt(np.mean((velocity_array - command) ** 2, axis=0)) if len(velocity_array) else np.full(3, np.nan)
    height_array = np.asarray(heights)
    action_array = np.asarray(actions)
    torque_array = np.asarray(torques)
    completed_policy_steps = completed_physics_steps // DECIMATION
    alive = bool(finite and completed_policy_steps == policy_steps and len(heights) and heights[-1] > 0.4
                 and max(np.max(np.abs(rolls)), np.max(np.abs(pitches))) < 60.0)
    metrics = {
        "alive": alive,
        "finite": finite,
        "requested_duration_s": args.duration,
        "sim_time_s": completed_physics_steps * PHYSICS_DT,
        "completed_policy_steps": completed_policy_steps,
        "base_height_m": {"min": float(height_array.min()), "mean": float(height_array.mean()), "final": float(height_array[-1])} if len(height_array) else None,
        "body_height_command_m": args.body_height,
        "base_height_error_rmse_m": float(np.sqrt(np.mean((height_array - args.body_height) ** 2))) if len(height_array) else None,
        "roll_deg": {"max_abs": float(np.max(np.abs(rolls))), "final": float(rolls[-1])} if rolls else None,
        "pitch_deg": {"max_abs": float(np.max(np.abs(pitches))), "final": float(pitches[-1])} if pitches else None,
        "velocity_command": command.tolist(),
        "velocity_tracking_rmse": tracking_rmse.tolist(),
        "action_finite": bool(np.isfinite(action_array).all()),
        "action_abs_max": float(np.max(np.abs(action_array))) if action_array.size else None,
        "torque_finite": bool(np.isfinite(torque_array).all()),
        "torque_abs_max_nm": float(np.max(np.abs(torque_array))) if torque_array.size else None,
        "torque_saturation_fraction": saturation_count / saturation_total if saturation_total else None,
        "nonfoot_contact_fraction": nonfoot_contact_steps / completed_physics_steps if completed_physics_steps else None,
        "observation_contract": "504 = 6 frames x 84, frame-major oldest-to-newest; each frame [cmd3,height1,ang_vel3,gravity3,q31,dq31,action12]",
        "action_joint_order": ACTION_JOINT_ORDER,
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0 if finite else 2


def main():
    parser = argparse.ArgumentParser(description="ELF3 HOMIE Isaac Gym -> MuJoCo Sim2Sim")
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--body-height", type=float, default=BASE_HEIGHT_TARGET)
    parser.add_argument("--root-height", type=float, default=0.82, help="与旧 Gym init_state 一致的初始根节点高度")
    parser.add_argument("--legacy-policy", dest="legacy_policy", action="store_true", default=True,
                        help="复现旧 Gym 的 DOF 顺序与 reset 后两帧历史初始化（默认启用）")
    parser.add_argument("--no-legacy-policy", dest="legacy_policy", action="store_false")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--headless", action="store_true", help="无界面运行（默认建议用于服务器）")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    args = parser.parse_args()
    raise SystemExit(simulate(args))


if __name__ == "__main__":
    main()
