"""Unitree's performance-gated velocity curriculum, with a lower ceiling.

Same gating logic as the plain-B2 curriculum, deliberately duplicated rather than
imported so the two policies share no code that could be tuned for one and break
the other.  The difference that matters is `LIMIT_RANGES`.

mjlab's stock `commands_vel` widens the command range at a fixed step count,
regardless of whether the policy can already track what it is being asked for.
On B2 that jump (max forward speed 1.0 -> 2.0 m/s at iteration 5000) wrecked
tracking: `error_vel_xy` went 0.43 -> 0.75 and never recovered, and 14000 further
iterations made it worse, not better.

Unitree gates the same widening on performance instead
(`unitree_rl_lab/tasks/locomotion/mdp/curriculums.py`): each episode boundary,
the range grows by a small step *only if* the tracking reward is already above
80% of its weight, and linear and angular are judged separately.  A policy that
cannot yet track 1.0 m/s is never asked for 1.2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Unitree ships their go2 / g1 / r1 / h1_2 velocity policies with this command
# range (deploy/robots/*/config/policy/velocity/v0/params/deploy.yaml), so it is
# what a deployed policy is actually asked for.  Start here.
DEPLOY_RANGES = {
  "lin_vel_x": (-0.5, 1.0),
  "lin_vel_y": (-0.5, 0.5),
  "ang_vel_z": (-1.0, 1.0),
}

# Ceiling the curriculum may grow to, if the policy earns it.  Lower than plain
# B2's (2.0 / 1.0 / 1.5): this robot is 105 kg with 15 kg of it 0.33 m above the
# trunk, and the payload is a 3 kg cobot rather than ballast, so headroom is
# worth more than top speed.  Nav2 asks for at most 0.9 m/s
# (`shalom/application/config/nav2_b2.yaml`), so 1.5 still leaves margin.
LIMIT_RANGES = {
  "lin_vel_x": (-0.8, 1.5),
  "lin_vel_y": (-0.8, 0.8),
  "ang_vel_z": (-1.2, 1.2),
}

_STEP = 0.1
_REWARD_FRACTION = 0.8


def _grow(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  cfg: UniformVelocityCommandCfg,
  reward_term_name: str,
  axes: tuple[str, ...],
) -> None:
  """Widen `axes` by one step if `reward_term_name` is being earned well."""
  term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
  sums = env.reward_manager._episode_sums[reward_term_name]  # noqa: SLF001
  # Normalise the episode total to a per-second rate so it is comparable with
  # the term's weight, which is what Unitree tests against.
  reward = torch.mean(sums[env_ids]) / env.max_episode_length_s

  if reward <= term_cfg.weight * _REWARD_FRACTION:
    return

  for axis in axes:
    low, high = getattr(cfg.ranges, axis)
    lo_lim, hi_lim = LIMIT_RANGES[axis]
    setattr(
      cfg.ranges,
      axis,
      (max(low - _STEP, lo_lim), min(high + _STEP, hi_lim)),
    )


def commands_vel_adaptive(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
) -> dict[str, torch.Tensor]:
  """Grow the velocity command range only as fast as the policy can follow it."""
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

  # Unitree re-evaluates once per episode length, not every reset.
  if env.common_step_counter % env.max_episode_length == 0:
    _grow(env, env_ids, cfg, "track_linear_velocity", ("lin_vel_x", "lin_vel_y"))
    _grow(env, env_ids, cfg, "track_angular_velocity", ("ang_vel_z",))

  return {
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }
