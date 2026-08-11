import argparse
import copy
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


ROOT = _repo_root()
sys.path.insert(0, str(ROOT / "rsl_rl"))

from rsl_rl.modules.him_actor_critic import HIMActorCritic  # noqa: E402


class ExportedHIMPolicy(nn.Module):
    def __init__(self, actor_critic: HIMActorCritic, one_step_obs: int):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.estimator = copy.deepcopy(actor_critic.estimator.encoder)
        self.one_step_obs = one_step_obs

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        parts = self.estimator(obs_history)
        vel = parts[..., :3]
        latent = F.normalize(parts[..., 3:], dim=-1, p=2.0)
        actor_input = torch.cat((obs_history[:, -self.one_step_obs :], vel, latent), dim=1)
        return self.actor(actor_input)


def build_actor_critic() -> HIMActorCritic:
    # This checkpoint was trained with 84 one-step observations and 6 history frames.
    return HIMActorCritic(
        num_actor_obs=504,
        num_critic_obs=87,
        num_one_step_obs=84,
        num_one_step_critic_obs=87,
        actor_history_length=6,
        critic_history_length=1,
        num_actions=12,
        actor_hidden_dims=[512, 256, 256],
        critic_hidden_dims=[512, 256, 256],
        activation="elu",
        init_noise_std=1.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export elf3_dof31 HIM checkpoint to TorchScript policy.")
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "legged_gym" / "logs" / "elf3_dof31" / "Jul27_13-05-15_" / "model_15999.pt"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "legged_gym" / "logs" / "elf3_dof31" / "exported" / "policies" / "policy.pt"),
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    actor_critic = build_actor_critic()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    actor_critic.load_state_dict(checkpoint["model_state_dict"])
    actor_critic.eval()

    policy = ExportedHIMPolicy(actor_critic, one_step_obs=84).cpu().eval()
    scripted = torch.jit.script(policy)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output_path))

    reloaded = torch.jit.load(str(output_path), map_location="cpu")
    dummy_obs = torch.zeros(1, 504)
    dummy_action = reloaded(dummy_obs)
    if tuple(dummy_action.shape) != (1, 12):
        raise RuntimeError(f"Unexpected policy output shape: {tuple(dummy_action.shape)}")

    print(f"Exported TorchScript policy: {output_path}")
    print(f"Input shape: {tuple(dummy_obs.shape)}")
    print(f"Output shape: {tuple(dummy_action.shape)}")


if __name__ == "__main__":
    main()
