"""
Open-loop dataset generation for VBD.

Every scene is initialized from a real Waymo TFRecord scenario (its map + agents).
The diffusion model samples one novel plan for the whole rollout (no re-planning)
and Waymax executes it with guidance enabled by default:

    * collision avoidance : OverlapReward guidance pushes agents apart
    * stay on road         : OnroadReward guidance penalizes leaving the road edge

Examples:
    # Guided one-shot generation for 5 scenarios, save comparison videos:
    python script/generate.py --model_path <ckpt.ckpt> \
        --waymo_path /mnt/sdb/waymo/validation --num_scenes 5 --video

    # Generate every scenario (collision + on-road guidance are already on):
    python script/generate.py --model_path <ckpt.ckpt> \
        --waymo_path /mnt/sdb/waymo/validation \
        --num_scenes -1 --video

Agents that enter after the 10-step history are included in the initial diffusion
call through a planning-only synthetic history, but remain absent in Waymax until
their original logged entry timestep.  From then on, their time-shifted diffusion
plan controls them. If a late entrant's initial box overlaps an agent already in
the generated scene, that entrant is suppressed and never shown in the generated
rollout (the original/log comparison panel remains unchanged).

Each generated scene is written as `--out_dir/scenario_<scenario_id>.json` using
the same schema as `prepare_waymo.py`, plus optionally an `.mp4` render and a
`_sim.pkl` of the collision-cleaned Waymax sim trajectory.

To launch multiple shard processes with GNU parallel, use the lightweight launcher:

    python script/main_generate.py ... --jobs 4 -- --video
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import re
import glob
import pickle
import json
import argparse
import copy
import zlib
import numpy as np
import torch
import imageio

# Keep TF and JAX on CPU (Torch owns the GPU).
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import jax
jax.config.update('jax_platform_name', 'cpu')

from vbd.data.dataset import WaymaxTestDataset
from vbd.model.utils import set_seed
from vbd.sim_agent.sim_actor import VBDTest, sample_to_action
from vbd.sim_agent.guidance_metrics.overlap_metric import OverlapReward
from vbd.sim_agent.guidance_metrics.onroad_metric import OnroadReward
from vbd.waymax_visualization.plotting import plot_state

from waymax import dynamics
from waymax.config import EnvironmentConfig, ObjectType, DatasetConfig, DataFormat
from vbd.sim_agent.waymax_env import WaymaxEnvironment
from vbd.data.waymax_utils import create_iter


CURRENT_TIME_INDEX = 10
PEDESTRIAN_TYPE = 2  # WOMD object type ids: 1=vehicle, 2=pedestrian, 3=cyclist


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------
def resolve_waymo_path(path):
    """Turn a directory of TFRecord shards into a waymax path spec.

    Waymax does not glob '*': it reads either a single file or the sharded
    spec '<prefix>.tfrecord@<N>' (which it expands to '<prefix>-00000-of-000N',
    ...). A single file or an already-'@'-style spec is passed through as-is.
    """
    if not os.path.isdir(path):
        return path

    files = sorted(glob.glob(os.path.join(path, "*.tfrecord-*-of-*")))
    if files:
        base = os.path.basename(files[0])
        m = re.match(r"(.+)-\d+-of-(\d+)$", base)
        if m:
            prefix, n_shards = m.group(1), int(m.group(2))
            return os.path.join(path, f"{prefix}@{n_shards}")

    # Fall back to any single .tfrecord file in the directory.
    files = sorted(glob.glob(os.path.join(path, "*.tfrecord*")))
    if not files:
        raise FileNotFoundError(f"No *.tfrecord* files under {path}")
    return files[0]


def list_shard_files(path):
    """The list of TFRecord shard files behind a directory / file / spec."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.tfrecord-*-of-*")))
        return files or sorted(glob.glob(os.path.join(path, "*.tfrecord*")))
    if "@" in os.path.basename(path):
        prefix, n = path.rsplit("@", 1)
        n = int(n)
        return [f"{prefix}-{i:05d}-of-{n:05d}" for i in range(n)]
    return [path]


def count_scenarios(path):
    """Total number of scenarios = total TFRecord entries across all shards.

    Reads every record's bytes (no parsing), so it is I/O bound -- a couple of
    minutes for the full 150-shard validation set.
    """
    files = list_shard_files(path)
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    return int(ds.reduce(np.int64(0), lambda x, _: x + 1).numpy())


class OnroadRewardWeighted(OnroadReward):
    """OnroadReward with a fixed weight so it composes with the other rewards.

    OnroadReward needs the scenario's `roadgraph_points`, which are not in the
    encoder outputs `c`; they are threaded through as a keyword argument from
    `sample_denoiser` -> `guidance` -> here.
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.fixed_weight = weight

    def forward(self, traj_pred, c, roadgraph_points=None, **kwargs):
        if roadgraph_points is None:
            raise ValueError("OnroadReward requires roadgraph_points to be passed to sample_denoiser")
        return super().forward(traj_pred, c, roadgraph_points, weight=self.fixed_weight)


class TimeShiftedOverlapReward(torch.nn.Module):
    """Overlap guidance on the actual timeline of delayed-entry agents.

    The model predicts every agent from its own virtual start state.  For an agent
    entering ``offset`` frames later, prediction frame 0 therefore belongs at
    global rollout frame ``offset``.  Standard overlap guidance compares equal
    tensor indices and would incorrectly treat that agent as present immediately.
    This wrapper shifts every trajectory onto the shared rollout timeline first.
    """

    def __init__(self, clip=5.0, weight=1.0, exact=False):
        super().__init__()
        self.clip = clip
        self.weight = weight
        self.exact = exact
        self.exact_reward = OverlapReward(clip=clip, weight=weight) if exact else None

    @staticmethod
    def _align_to_rollout(traj_pred, entry_offsets):
        B, A, T, D = traj_pred.shape
        offsets = torch.as_tensor(entry_offsets, device=traj_pred.device, dtype=torch.long)
        if offsets.ndim == 1:
            offsets = offsets.unsqueeze(0).expand(B, -1)
        if offsets.shape != (B, A):
            raise ValueError(
                f"entry_offsets must have shape [A] or [B, A], got {tuple(offsets.shape)}")

        rollout_t = torch.arange(T, device=traj_pred.device).view(1, 1, T)
        source_t = rollout_t - offsets.unsqueeze(-1)
        active = source_t >= 0
        source_t = source_t.clamp(0, T - 1)
        gather_idx = source_t.unsqueeze(-1).expand(B, A, T, D)
        aligned = torch.gather(traj_pred, dim=2, index=gather_idx)
        return aligned, active

    def forward(self, traj_pred, c, entry_offsets=None, aoi=None, **kwargs):
        if entry_offsets is None:
            raise ValueError("TimeShiftedOverlapReward requires entry_offsets")

        aligned, active = self._align_to_rollout(traj_pred, entry_offsets)
        static_valid = ~c['agents_mask']

        if self.exact:
            # The exact reward only supports a static agent mask.  Put not-yet-
            # active boxes at unique, remote positions; torch.where also makes
            # their gradient zero until their actual entry frame.
            B, A, T, D = aligned.shape
            sentinel = torch.zeros_like(aligned)
            agent_slot = torch.arange(A, device=aligned.device, dtype=aligned.dtype)
            sentinel[..., 0] = 1e6 + agent_slot.view(1, A, 1) * 100.0
            sentinel[..., 1] = 1e6
            aligned = torch.where(active.unsqueeze(-1), aligned, sentinel)
            return self.exact_reward(aligned, c, aoi=aoi, **kwargs)

        if aoi is not None:
            aligned = aligned[:, aoi]
            active = active[:, aoi]
            static_valid = static_valid[:, aoi]

        B, A, T, _ = aligned.shape
        valid = static_valid.unsqueeze(-1) & active              # [B, A, T]
        xy = aligned[..., :2]
        xy_i = xy.unsqueeze(3)                                   # [B, A, T, 1, 2]
        xy_j = xy.permute(0, 2, 1, 3).unsqueeze(1)               # [B, 1, T, A, 2]
        distance = torch.norm(xy_i - xy_j, dim=-1)               # [B, A, T, A]

        valid_i = valid.unsqueeze(3)
        valid_j = valid.permute(0, 2, 1).unsqueeze(1)
        pair_valid = valid_i & valid_j
        # Count each unordered pair once. Both trajectories still receive the
        # gradient, matching the strength of the original detached-pair reward.
        upper = torch.triu(
            torch.ones(A, A, device=aligned.device, dtype=torch.bool), diagonal=1)
        pair_valid = pair_valid & upper.view(1, A, 1, A)

        distance = torch.where(pair_valid, distance, self.clip)
        return distance * (distance < self.clip) * self.weight


def build_env(n_agents, allow_new_objects=True):
    """Waymax environment where the sim agent controls every valid object.

    `allow_new_objects` controls whether agents that only become valid *after*
    the warmup timestep (i.e. enter the scene later) are introduced into the
    simulation. With it False they are frozen out for the whole rollout; with it
    True they appear per their log and are then driven by their time-shifted
    one-shot diffusion plan.
    """
    env_config = EnvironmentConfig(
        controlled_object=ObjectType.VALID,
        max_num_objects=n_agents,
        allow_new_objects_after_warmup=allow_new_objects,
    )
    return WaymaxEnvironment(
        dynamics_model=dynamics.StateDynamics(),
        config=env_config,
    )


def configure_guidance(vbd, args):
    """Wire up diffusion guidance from the enabled control knobs.

    Leaves guidance disabled (plain denoising) when no knob is requested.
    """
    reward_funcs = []
    if args.avoid_collisions:
        if args.exact_overlap:
            # Exact box geometry: backward Jacobian is O(A^3) in memory and OOMs
            # for large scenes (e.g. ~160 GB at 128 agents). Only for small scenes.
            print("WARNING: --exact_overlap uses O(A^3) memory; expect OOM for many agents.")
            exact = True
        else:
            exact = False
        reward_funcs.append(TimeShiftedOverlapReward(
            clip=args.overlap_clip,
            weight=args.collision_weight,
            exact=exact,
        ))
    if args.stay_onroad:
        reward_funcs.append(OnroadRewardWeighted(weight=args.onroad_weight))

    if not reward_funcs:
        print("Guidance: disabled (plain diffusion sampling)")
        return False

    vbd.reward_func = reward_funcs
    vbd.guidance_func = vbd.guidance
    vbd.guidance_iter = args.guidance_iter
    vbd.guidance_start = 99
    vbd.guidance_end = 1
    vbd.gradient_scale = args.gradient_scale
    vbd.scale_grad_by_std = True
    knobs = [type(r).__name__ for r in reward_funcs]
    print(f"Guidance: enabled {knobs} "
          f"(scale={args.gradient_scale}, iter={args.guidance_iter})")
    return True


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------
def _scenario_seed(base_seed, scenario_id):
    """Stable per-scenario seed, independent of worker scheduling."""
    return int(base_seed) ^ zlib.crc32(str(scenario_id).encode("utf-8"))


def _set_metadata_values(metadata, field, indices, values):
    """Update a Waymax ObjectMetadata field if that version exposes it."""
    if not hasattr(metadata, field):
        return
    array = np.asarray(getattr(metadata, field)).copy()
    array[indices] = values
    setattr(metadata, field, array)


def inject_synthetic_agents(
        scenario, ratio, rng, current_index=CURRENT_TIME_INDEX,
        clearance=0.5, max_attempts=200):
    """Add collision-free synthetic agents at the first simulation timestep.

    Synthetic agents occupy padding slots only, so real agents that enter later
    are never replaced. Positions are sampled from valid freeway/surface-street
    lane-center points. Their history is constant-velocity and their future log is
    only a fallback: the diffusion policy controls them from ``current_index``.

    Returns ``(scenario, synthetic_mask)``. The requested count is
    ``floor(num_present * ratio)`` and is capped by padding/placement capacity.
    """
    traj = scenario.log_trajectory
    metadata = scenario.object_metadata
    valid = np.asarray(traj.valid).copy()
    A, T = valid.shape
    present = valid[:, current_index]
    requested = int(np.floor(int(present.sum()) * ratio))
    synthetic_mask = np.zeros(A, dtype=bool)
    if requested <= 0:
        return scenario, synthetic_mask

    metadata_valid = np.asarray(metadata.is_valid, dtype=bool)
    padding_slots = np.flatnonzero(~metadata_valid)
    target = min(requested, len(padding_slots))
    if target == 0:
        print(f"  synthetic agents: requested {requested}, added 0 (no padding slots)")
        return scenario, synthetic_mask

    roadgraph = scenario.roadgraph_points
    rg_xy = np.asarray(roadgraph.xy)
    rg_dir = np.asarray(roadgraph.dir_xy)
    rg_type = np.asarray(roadgraph.types)
    rg_valid = np.asarray(roadgraph.valid, dtype=bool)
    dir_norm = np.linalg.norm(rg_dir, axis=-1)
    lane_mask = (
        rg_valid & np.isin(rg_type, (1, 2, 3)) &
        np.isfinite(rg_xy).all(axis=-1) & np.isfinite(rg_dir).all(axis=-1) &
        (dir_norm > 1e-3)
    )
    lane_indices = np.flatnonzero(lane_mask)
    road_lane_indices = np.flatnonzero(lane_mask & np.isin(rg_type, (1, 2)))
    if len(lane_indices) == 0:
        print(f"  synthetic agents: requested {requested}, added 0 (no lane centers)")
        return scenario, synthetic_mask

    x = np.asarray(traj.x).copy()
    y = np.asarray(traj.y).copy()
    yaw = np.asarray(traj.yaw).copy()
    vel_x = np.asarray(traj.vel_x).copy()
    vel_y = np.asarray(traj.vel_y).copy()
    length = np.asarray(traj.length).copy()
    width = np.asarray(traj.width).copy()
    height = np.asarray(traj.height).copy()

    existing = []
    for i in np.flatnonzero(present):
        box = (x[i, current_index], y[i, current_index], yaw[i, current_index],
               length[i, current_index], width[i, current_index])
        if np.isfinite(box).all() and box[3] > 0 and box[4] > 0:
            existing.append(box)

    types = []
    accepted_slots = []
    for slot in padding_slots:
        if len(accepted_slots) >= target:
            break
        # WOMD types: 1=vehicle, 2=pedestrian, 3=cyclist. Sampling is uniform.
        object_type = int(rng.integers(1, 4))
        type_lane_indices = (lane_indices if object_type == 3
                             else road_lane_indices)
        if len(type_lane_indices) == 0:
            continue
        accepted = None
        for _ in range(max_attempts):
            road_idx = int(rng.choice(type_lane_indices))
            direction = rg_dir[road_idx] / dir_norm[road_idx]
            position = rg_xy[road_idx] + direction * rng.uniform(-1.0, 1.0)

            if object_type == 1:
                agent_length = rng.uniform(3.8, 5.2)
                agent_width = rng.uniform(1.7, 2.2)
                agent_height = rng.uniform(1.4, 2.0)
                speed = rng.uniform(2.0, 13.0)
                heading = np.arctan2(direction[1], direction[0])
            elif object_type == 2:
                agent_length = rng.uniform(0.4, 0.9)
                agent_width = rng.uniform(0.4, 0.9)
                agent_height = rng.uniform(1.4, 2.0)
                speed = rng.uniform(0.3, 1.8)
                if rng.random() < 0.5:
                    direction = -direction
                heading = np.arctan2(direction[1], direction[0])
            else:  # cyclist
                agent_length = rng.uniform(1.5, 2.2)
                agent_width = rng.uniform(0.5, 0.9)
                agent_height = rng.uniform(1.4, 1.9)
                speed = rng.uniform(2.0, 8.0)
                heading = np.arctan2(direction[1], direction[0])

            candidate = (position[0], position[1], heading,
                         agent_length, agent_width)
            overlaps = any(
                _oriented_boxes_overlap(
                    candidate[:2], candidate[2],
                    candidate[3] + clearance, candidate[4] + clearance,
                    other[:2], other[2],
                    other[3] + clearance, other[4] + clearance)
                for other in existing
            )
            if not overlaps:
                accepted = (candidate, object_type, agent_height, speed, direction)
                break

        if accepted is None:
            continue

        candidate, object_type, agent_height, speed, direction = accepted
        px, py, heading, agent_length, agent_width = candidate
        times = (np.arange(T) - current_index) * 0.1
        x[slot] = px + speed * direction[0] * times
        y[slot] = py + speed * direction[1] * times
        yaw[slot] = heading
        vel_x[slot] = speed * direction[0]
        vel_y[slot] = speed * direction[1]
        length[slot] = agent_length
        width[slot] = agent_width
        height[slot] = agent_height
        valid[slot] = True
        synthetic_mask[slot] = True
        accepted_slots.append(slot)
        types.append(object_type)
        existing.append(candidate)

    accepted_slots = np.asarray(accepted_slots, dtype=int)
    if len(accepted_slots):
        traj.x, traj.y, traj.yaw = x, y, yaw
        traj.vel_x, traj.vel_y = vel_x, vel_y
        traj.length, traj.width, traj.height = length, width, height
        traj.valid = valid

        _set_metadata_values(metadata, 'is_valid', accepted_slots, True)
        _set_metadata_values(metadata, 'object_types', accepted_slots, np.asarray(types))
        _set_metadata_values(metadata, 'is_sdc', accepted_slots, False)
        _set_metadata_values(metadata, 'is_modeled', accepted_slots, False)
        _set_metadata_values(metadata, 'objects_of_interest', accepted_slots, False)
        _set_metadata_values(metadata, 'is_controlled', accepted_slots, False)
    print(f"  synthetic agents: requested {requested}, added {len(accepted_slots)}")
    return scenario, synthetic_mask


def build_planning_scenario(scenario, current_index=CURRENT_TIME_INDEX, dt=0.1):
    """Build one-shot diffusion conditioning for current and future agents.

    Waymax must keep the original scenario untouched so future agents enter at
    their logged first-valid timestep.  The diffusion model, however, masks an
    agent that is invalid at ``current_index``.  We therefore create a separate
    planning-only copy and back-fill a synthetic history for every future entrant.

    The synthetic history is a constant-velocity line ending at the agent's
    *next* valid state.  This makes the agent visible to the single diffusion call
    without making it appear early in the actual simulation.

    Returns ``(planning_scenario, entry_steps)``. ``entry_steps[a]`` is the
    timestep at which diffusion may start controlling agent ``a``; ``-1`` means
    that the agent is neither present now nor appears later.
    """
    planning_scenario = copy.deepcopy(scenario)
    traj = planning_scenario.log_trajectory
    valid = np.asarray(traj.valid)                    # [A, T]
    A, T = valid.shape

    x = np.asarray(traj.x).copy()
    y = np.asarray(traj.y).copy()
    yaw = np.asarray(traj.yaw).copy()
    vel_x = np.asarray(traj.vel_x).copy()
    vel_y = np.asarray(traj.vel_y).copy()
    length = np.asarray(traj.length).copy()
    width = np.asarray(traj.width).copy()
    height = np.asarray(traj.height).copy()
    new_valid = valid.copy()
    history_times = (np.arange(current_index + 1) - current_index) * dt
    entry_steps = np.full(A, -1, dtype=np.int32)

    for a in range(A):
        if valid[a, current_index]:
            entry_steps[a] = current_index
            continue

        # Use the next appearance, not the first appearance in the whole record;
        # otherwise an agent that already left would be incorrectly resurrected.
        vidx = np.where(valid[a, current_index + 1:])[0]
        if len(vidx) == 0:
            continue

        t0 = int(vidx[0] + current_index + 1)
        entry_steps[a] = t0
        hist = slice(0, current_index + 1)
        x[a, hist] = x[a, t0] + vel_x[a, t0] * history_times
        y[a, hist] = y[a, t0] + vel_y[a, t0] * history_times
        yaw[a, hist] = yaw[a, t0]
        vel_x[a, hist] = vel_x[a, t0]
        vel_y[a, hist] = vel_y[a, t0]
        length[a, hist] = length[a, t0]
        width[a, hist] = width[a, t0]
        height[a, hist] = height[a, t0]
        new_valid[a, hist] = True

    traj.x, traj.y, traj.yaw = x, y, yaw
    traj.vel_x, traj.vel_y = vel_x, vel_y
    traj.length, traj.width, traj.height = length, width, height
    traj.valid = new_valid
    return planning_scenario, entry_steps


def rollout_scenario(vbd, env, dataset, scenario, args, n_agents):
    """Generate one diffusion plan, then execute it without re-planning.

    Future agents remain absent in the real Waymax state until their logged entry
    time.  At entry, log replay introduces them; from that state onward their
    time-shifted one-shot diffusion trajectory controls them.

    Returns ``(log_states, diffusion_masks, planned_agents, entry_steps,
    suppressed_agents)``.
    ``diffusion_masks[k]`` identifies agents whose diffusion action produced
    ``log_states[k]`` and is therefore also the exact mask used for collision
    attribution.
    """
    initial_state = current_state = env.reset(scenario)
    planning_scenario, entry_steps = build_planning_scenario(scenario)
    planning_state = env.reset(planning_scenario)
    planned_agents = entry_steps >= CURRENT_TIME_INDEX
    delayed_agents = entry_steps > CURRENT_TIME_INDEX

    # One and only diffusion call for this scene.  The planning state contains
    # synthetic histories for future entrants, while the simulated state above
    # retains the unmodified log and is used by the comparison visualization.
    with torch.no_grad():
        sample = dataset.process_scenario(
            planning_state, planning_state.timestep, use_log=False)
        batch = dataset.__collate_fn__([sample])
        pred = vbd.sample_denoiser(
            batch, use_tqdm=not args.quiet,
            roadgraph_points=planning_state.roadgraph_points,
            entry_offsets=entry_steps - CURRENT_TIME_INDEX)
        pred_traj = pred['denoised_trajs'].cpu().numpy()[0]

    print(f"  one-shot plan: {int(planned_agents.sum())} agents "
          f"({int(delayed_agents.sum())} enter later)")
    log_states = [initial_state]
    diffusion_masks = [np.zeros(n_agents, dtype=bool)]
    suppressed_agents = np.zeros(n_agents, dtype=bool)

    for _ in range(int(initial_state.remaining_timesteps)):
        current_step = int(np.asarray(current_state.timestep))
        present = np.asarray(
            current_state.sim_trajectory.valid[:, current_step], dtype=bool)
        active = planned_agents & present & (entry_steps <= current_step)

        # Each delayed agent starts at prediction index zero when it enters,
        # rather than skipping ahead by its entry delay.
        action_sample = np.zeros_like(pred_traj[:, 0, :])
        agent_idx = np.flatnonzero(active)
        if len(agent_idx):
            plan_idx = current_step - entry_steps[agent_idx]
            within_horizon = plan_idx < pred_traj.shape[1]
            if not np.all(within_horizon):
                active[agent_idx[~within_horizon]] = False
                agent_idx = agent_idx[within_horizon]
                plan_idx = plan_idx[within_horizon]
            action_sample[agent_idx] = pred_traj[agent_idx, plan_idx, :]

        action = sample_to_action(action_sample, active, None, n_agents)
        current_state = env.step_sim_agent(current_state, [action])

        next_step = int(np.asarray(current_state.timestep))

        # A previously suppressed object must stay absent even if Waymax's
        # new-object logic attempts to replay it from the log again.
        if suppressed_agents.any():
            _suppress_agents_in_generated_state(
                current_state, suppressed_agents, from_step=next_step)

        # Future agents are introduced by log replay for exactly their entry
        # transition. If their entry box overlaps anything already accepted in
        # the generated scene, remove the newcomer before recording this state.
        newcomers = (
            delayed_agents & ~suppressed_agents &
            (entry_steps == next_step)
        )
        newly_suppressed = _colliding_newcomers(current_state, newcomers)
        if newly_suppressed.any():
            suppressed_agents |= newly_suppressed
            planned_agents[newly_suppressed] = False
            _suppress_agents_in_generated_state(
                current_state, suppressed_agents, from_step=next_step)
            dropped_ids = np.flatnonzero(newly_suppressed).tolist()
            print(f"  suppressed {len(dropped_ids)} colliding late-entry "
                  f"agent(s) at t={next_step}: {dropped_ids}")

        log_states.append(current_state)
        diffusion_masks.append(active.copy())

    return (log_states, diffusion_masks, planned_agents, entry_steps,
            suppressed_agents)


def _oriented_boxes_overlap(center_i, yaw_i, length_i, width_i,
                            center_j, yaw_j, length_j, width_j,
                            min_penetration=1e-3):
    """Exact 2-D oriented-box overlap using the separating-axis theorem.

    Merely touching edges is not treated as a collision; the boxes must overlap
    by at least ``min_penetration`` on every separating axis.  This avoids
    floating-point false positives from grazing boxes.
    """
    ui = np.array([np.cos(yaw_i), np.sin(yaw_i)])
    vi = np.array([-np.sin(yaw_i), np.cos(yaw_i)])
    uj = np.array([np.cos(yaw_j), np.sin(yaw_j)])
    vj = np.array([-np.sin(yaw_j), np.cos(yaw_j)])
    delta = np.asarray(center_j) - np.asarray(center_i)
    half_li, half_wi = length_i / 2.0, width_i / 2.0
    half_lj, half_wj = length_j / 2.0, width_j / 2.0

    for axis in (ui, vi, uj, vj):
        distance = abs(np.dot(delta, axis))
        radius_i = (half_li * abs(np.dot(ui, axis)) +
                    half_wi * abs(np.dot(vi, axis)))
        radius_j = (half_lj * abs(np.dot(uj, axis)) +
                    half_wj * abs(np.dot(vj, axis)))
        if distance >= radius_i + radius_j - min_penetration:
            return False
    return True


def _colliding_newcomers(state, newcomer_mask):
    """Select late-entry agents whose entry box overlaps an accepted object.

    Existing objects always win. Newcomers at the same timestep are processed in
    stable object-index order, so the first non-conflicting newcomer is retained
    and later newcomers that overlap it are suppressed.
    """
    newcomer_mask = np.asarray(newcomer_mask, dtype=bool)
    drop = np.zeros_like(newcomer_mask)
    if not newcomer_mask.any():
        return drop

    traj = state.sim_trajectory
    step = int(np.asarray(state.timestep))
    valid = np.asarray(traj.valid[:, step], dtype=bool)
    x = np.asarray(traj.x[:, step])
    y = np.asarray(traj.y[:, step])
    yaw = np.asarray(traj.yaw[:, step])
    length = np.asarray(traj.length[:, step])
    width = np.asarray(traj.width[:, step])
    finite_box = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(yaw) &
        np.isfinite(length) & np.isfinite(width) &
        (length > 0) & (width > 0)
    )
    present = valid & finite_box
    candidates = np.flatnonzero(present & newcomer_mask)
    accepted = list(np.flatnonzero(present & ~newcomer_mask))

    for i in candidates:
        overlaps = any(
            _oriented_boxes_overlap(
                (x[i], y[i]), yaw[i], length[i], width[i],
                (x[j], y[j]), yaw[j], length[j], width[j])
            for j in accepted
        )
        if overlaps:
            drop[i] = True
        else:
            accepted.append(i)
    return drop


def _suppress_agents_in_generated_state(state, suppress_mask, from_step):
    """Permanently hide agents from generated state while preserving log truth."""
    suppress_mask = np.asarray(suppress_mask, dtype=bool)
    if not suppress_mask.any():
        return

    # Only sim_trajectory is changed. log_trajectory stays intact, so the left
    # side of the comparison video continues to show the original newcomer.
    sim_valid = np.asarray(state.sim_trajectory.valid).copy()
    sim_valid[suppress_mask, from_step:] = False
    state.sim_trajectory.valid = sim_valid

    metadata_valid = np.asarray(state.object_metadata.is_valid).copy()
    metadata_valid[suppress_mask] = False
    state.object_metadata.is_valid = metadata_valid

    metadata_controlled = np.asarray(state.object_metadata.is_controlled).copy()
    metadata_controlled[suppress_mask] = False
    state.object_metadata.is_controlled = metadata_controlled


def find_collided_agents(traj, start_index=CURRENT_TIME_INDEX):
    """Return every agent involved in any valid box collision in a trajectory."""
    valid = np.asarray(traj.valid, dtype=bool)
    x = np.asarray(traj.x)
    y = np.asarray(traj.y)
    yaw = np.asarray(traj.yaw)
    length = np.asarray(traj.length)
    width = np.asarray(traj.width)
    collided = np.zeros(valid.shape[0], dtype=bool)

    for step in range(start_index, valid.shape[1]):
        finite_box = (
            np.isfinite(x[:, step]) & np.isfinite(y[:, step]) &
            np.isfinite(yaw[:, step]) & np.isfinite(length[:, step]) &
            np.isfinite(width[:, step]) &
            (length[:, step] > 0) & (width[:, step] > 0)
        )
        indices = np.flatnonzero(valid[:, step] & finite_box)
        for p, i in enumerate(indices):
            for j in indices[p + 1:]:
                if _oriented_boxes_overlap(
                        (x[i, step], y[i, step]), yaw[i, step],
                        length[i, step], width[i, step],
                        (x[j, step], y[j, step]), yaw[j, step],
                        length[j, step], width[j, step]):
                    collided[i] = True
                    collided[j] = True
    return collided


def collision_removal_mask_preserving_ego(
        traj, metadata, start_index=CURRENT_TIME_INDEX):
    """Remove collision participants except the SDC used as the ego reference."""
    removal_mask = find_collided_agents(traj, start_index=start_index)
    ego_idx = _ego_index(metadata)
    ego_collided = bool(removal_mask[ego_idx])
    removal_mask[ego_idx] = False
    return removal_mask, ego_idx, ego_collided


def _json_agent_ids(metadata, synthetic_agents, num_agents):
    """Return JSON-safe real IDs plus stable negative IDs for synthetic agents."""
    if hasattr(metadata, 'ids'):
        raw_ids = np.asarray(metadata.ids).reshape(-1).tolist()
    else:
        raw_ids = list(range(num_agents))

    result = []
    for i, value in enumerate(raw_ids):
        if synthetic_agents[i]:
            result.append(-100000 - i)
        elif isinstance(value, bytes):
            result.append(value.decode('utf-8'))
        elif isinstance(value, np.generic):
            result.append(value.item())
        else:
            result.append(value)
    return result


def _ego_index(metadata, scenario_id=None):
    is_sdc = np.asarray(metadata.is_sdc).reshape(-1)
    ego_indices = np.flatnonzero(is_sdc)
    if len(ego_indices) == 0:
        prefix = f"Scenario {scenario_id} " if scenario_id is not None else "Scenario "
        raise ValueError(f"{prefix}has no SDC/ego agent")
    return int(ego_indices[0])


def save_scene(scenario_id, log_states, original_log_trajectory,
               synthetic_agents, removed_collision_agents, out_dir, args):
    """Save the cleaned rollout using prepare_waymo.py's JSON schema."""
    traj_final = log_states[-1].sim_trajectory  # waymax Trajectory, [A, T]
    x = np.asarray(traj_final.x)
    z = (np.asarray(traj_final.z) if hasattr(traj_final, 'z')
         else np.zeros_like(x))
    full_states = np.stack([
        x,
        np.asarray(traj_final.y),
        z,
        np.asarray(traj_final.length),
        np.asarray(traj_final.width),
        np.asarray(traj_final.height),
        np.asarray(traj_final.yaw),
    ], axis=-1)
    # VBD generates the 80 future frames after the 10 past + 1 current
    # conditioning frames. Save only those generated frames.
    all_states = full_states[:, CURRENT_TIME_INDEX + 1:].copy()  # [A, 80, 7]
    # np.asarray(JAX array) can be read-only; np.array(..., copy=True) is required
    # before invalidating collision agents.
    full_valid = np.array(traj_final.valid, dtype=bool, copy=True)
    valid = full_valid[:, CURRENT_TIME_INDEX + 1:].copy()
    removed_collision_agents = np.array(
        removed_collision_agents, dtype=bool, copy=True)

    metadata = log_states[-1].object_metadata
    ego_idx = _ego_index(metadata, scenario_id)
    if removed_collision_agents[ego_idx]:
        raise ValueError(
            f"Scenario {scenario_id}: ego must not be in collision-removal mask")
    if not valid[ego_idx].all():
        valid_count = int(valid[ego_idx].sum())
        raise ValueError(
            f"Scenario {scenario_id}: ego is valid for only "
            f"{valid_count}/{valid.shape[1]} generated frames")

    # Delete collided agents from the entire stored history.
    all_states[removed_collision_agents] = 0.0
    valid[removed_collision_agents] = False
    full_valid[removed_collision_agents] = False

    json_data = {
        "all_states": all_states.tolist(),
        "all_states_mask": valid.tolist(),
        "agent_ids": _json_agent_ids(metadata, synthetic_agents, all_states.shape[0]),
        "agent_types": np.asarray(metadata.object_types).reshape(-1).tolist(),
        "ego_idx": ego_idx,
        "source_file": args.waymo_path,
    }
    json_path = os.path.join(out_dir, "json", f"scenario_{scenario_id}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)

    if args.save_sim:
        clean_traj = copy.deepcopy(traj_final)
        for field in ('x', 'y', 'z', 'yaw', 'vel_x', 'vel_y',
                      'length', 'width', 'height'):
            if not hasattr(clean_traj, field):
                continue
            values = np.asarray(getattr(clean_traj, field)).copy()
            values[removed_collision_agents] = 0.0
            setattr(clean_traj, field, values)
        clean_traj.valid = full_valid
        with open(os.path.join(out_dir, f"{scenario_id}_sim.pkl"), 'wb') as f:
            pickle.dump(clean_traj, f)

    saved = [json_path]

    if args.video:
        # original/log | raw diffusion | collision-cleaned diffusion
        frames = []
        for state in log_states:
            original = plot_state(
                state,
                traj_override=original_log_trajectory,
                panel_title="Original")
            diffusion = plot_state(
                state, log_traj=False,
                highlight_mask=synthetic_agents,
                panel_title="Diffusion")
            cleaned = plot_state(
                state, log_traj=False,
                highlight_mask=synthetic_agents,
                hide_mask=removed_collision_agents,
                panel_title="Cleaned")
            frames.append(np.concatenate([original, diffusion, cleaned], axis=1))
        video_path = os.path.join(out_dir, "debug", f"{scenario_id}.mp4")
        imageio.mimwrite(video_path, frames, fps=args.fps, macro_block_size=None)
        saved.append(video_path)

    print("  saved " + ", ".join(saved))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# Waymax parses WOMD tf_examples with at most this many objects (the default in
# `womd_utils.get_features_description`, which `create_iter` relies on). The env
# and the loaded scenario must agree on the object count, so we cannot exceed it.
WOMD_MAX_OBJECTS = 128


def _create_runtime(args, n_agents, device=None):
    """Load one model/runtime instance for a serial process or worker."""
    device = device or args.device
    vbd = VBDTest.load_from_checkpoint(args.model_path, device)
    vbd.reset_agent_length(n_agents)
    configure_guidance(vbd, args)
    vbd.eval()
    env = build_env(n_agents, allow_new_objects=True)
    dataset = WaymaxTestDataset(
        data_dir=None, anchor_path=args.anchor_path, max_object=n_agents)
    return vbd, env, dataset


def _process_scenario(runtime, scenario_id, scenario, args, n_agents):
    """Generate, clean, visualize, and save one scenario."""
    vbd, env, dataset = runtime
    scenario_seed = _scenario_seed(args.seed, scenario_id)
    set_seed(scenario_seed)
    # Preserve the untouched WOMD log for the left-most video panel. Synthetic
    # injection mutates scenario.log_trajectory for model conditioning.
    original_log_trajectory = copy.deepcopy(scenario.log_trajectory)
    scenario, synthetic_agents = inject_synthetic_agents(
        scenario,
        ratio=args.synthetic_ratio,
        rng=np.random.default_rng(scenario_seed),
        clearance=args.synthetic_clearance,
    )
    (log_states, diffusion_masks, planned_agents, entry_steps,
     suppressed_agents) = rollout_scenario(
        vbd, env, dataset, scenario, args, n_agents)

    metadata = log_states[-1].object_metadata
    removed_collision_agents, ego_idx, ego_collided = (
        collision_removal_mask_preserving_ego(
            log_states[-1].sim_trajectory,
            metadata,
            start_index=CURRENT_TIME_INDEX + 1,
        )
    )

    ego_valid = np.asarray(log_states[-1].sim_trajectory.valid)[
        ego_idx, CURRENT_TIME_INDEX + 1:]
    if not np.asarray(ego_valid, dtype=bool).all():
        valid_count = int(np.asarray(ego_valid, dtype=bool).sum())
        print(f"  skip save: ego is valid for only {valid_count}/80 generated frames")
        return {
            'scenario_id': scenario_id,
            'saved': False,
            'reason': 'invalid_ego',
            'synthetic': int(synthetic_agents.sum()),
            'collision_removed': 0,
            'late_entry_suppressed': int(suppressed_agents.sum()),
        }

    print(f"  removed {int(removed_collision_agents.sum())} agent(s) "
          f"involved in a collision"
          f"{' (ego preserved)' if ego_collided else ''}")
    save_scene(
        scenario_id, log_states, original_log_trajectory, synthetic_agents,
        removed_collision_agents, args.out_dir, args)
    return {
        'scenario_id': scenario_id,
        'saved': True,
        'synthetic': int(synthetic_agents.sum()),
        'collision_removed': int(removed_collision_agents.sum()),
        'late_entry_suppressed': int(suppressed_agents.sum()),
    }


def _scenario_stats(scenario):
    present = np.asarray(
        scenario.log_trajectory.valid[:, CURRENT_TIME_INDEX], dtype=bool)
    types = np.asarray(scenario.object_metadata.object_types)
    return int(present.sum()), int((present & (types == PEDESTRIAN_TYPE)).sum())


def generate(args):
    if args.num_scenes == 0:
        print("Done. Requested 0 scenes; nothing to generate.")
        return
    if args.scenario_index < 0:
        raise ValueError("--scenario_index must be >= 0")
    if args.synthetic_ratio < 0:
        raise ValueError("--synthetic_ratio must be >= 0")
    if args.synthetic_clearance < 0:
        raise ValueError("--synthetic_clearance must be >= 0")

    n_agents = args.max_agents
    if n_agents > WOMD_MAX_OBJECTS:
        print(f"WARNING: --max_agents {n_agents} exceeds the WOMD scenario cap of "
              f"{WOMD_MAX_OBJECTS}; clamping to {WOMD_MAX_OBJECTS}.")
        n_agents = WOMD_MAX_OBJECTS

    path = resolve_waymo_path(args.waymo_path)
    print(f"Loading scenarios from {path}")

    # Counting is optional. Generating all scenes can simply consume the iterator
    # to exhaustion, avoiding a full extra I/O pass over every TFRecord.
    total_available = None
    if args.count:
        print("Counting scenarios across TFRecord shards ...")
        total_available = count_scenarios(path)
        print(f"Total scenarios available: {total_available}")

    # How many we intend to generate from scenario_index onward.
    if args.num_scenes < 0:
        target = (total_available - args.scenario_index
                  if total_available is not None else None)
    else:
        target = args.num_scenes
    target_label = target if target is not None else "all"

    config = DatasetConfig(
        path=path,
        max_num_objects=n_agents,
        max_num_rg_points=30000,
        repeat=1,
        data_format=DataFormat.TFRECORD,
    )
    data_iter = create_iter(config)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "json"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "debug"), exist_ok=True)
    runtime = _create_runtime(args, n_agents, device=args.device)
    generated = 0
    for k, (scenario_id, scenario) in enumerate(data_iter):
        if k < args.scenario_index:
            continue
        n_present, n_peds = _scenario_stats(scenario)
        if not (n_present > args.min_agents or n_peds >= args.min_peds):
            print(f"skip {scenario_id}: {n_present} agents, {n_peds} pedestrians")
            continue
        print(f"[{generated + 1}/{target_label}] Generating scenario {scenario_id} "
              f"({n_present} agents, {n_peds} pedestrians) ...")
        result = _process_scenario(runtime, scenario_id, scenario, args, n_agents)
        if not result['saved']:
            continue
        generated += 1
        if target is not None and generated >= target:
            break

    print(f"Done. Generated {generated} scene(s) -> {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model / data
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to a trained VBD checkpoint (.ckpt)")
    parser.add_argument("--waymo_path", type=str, required=True,
                        help="Waymo TFRecord file, glob, or directory of shards to initialize scenes from")
    parser.add_argument("--anchor_path", type=str,
                        default="./vbd/data/cluster_64_center_dict.pkl")
    parser.add_argument("--out_dir", type=str, default="generated_scenes5")

    # How many / which scenes
    parser.add_argument("--num_scenes", type=int, default=1,
                        help="Number of scenarios to generate (-1 = all available)")
    parser.add_argument("--scenario_index", type=int, default=0,
                        help="Index of the first scenario to use within the stream")
    parser.add_argument("--count", action="store_true",
                        help="Count and print the total scenarios across the TFRecord shards before generating")
    parser.add_argument("--max_agents", type=int, default=128,
                        help="Maximum real + synthetic object slots per scene")
    parser.add_argument("--min_agents", type=int, default=50,
                        help="Keep scenes with MORE than this many agents at the generation timestep")
    parser.add_argument("--min_peds", type=int, default=5,
                        help="Also keep scenes with at least this many pedestrians at the generation timestep")

    # Synthetic agents (only inserted at the first simulation timestep).
    parser.add_argument("--synthetic_ratio", type=float, default=0.25,
                        help="Synthetic agents to add as a fraction of initially present agents")
    parser.add_argument("--synthetic_clearance", type=float, default=0.5,
                        help="Minimum extra placement clearance around synthetic boxes in meters")

    # Guidance: collision avoidance
    parser.set_defaults(avoid_collisions=True, stay_onroad=True)
    parser.add_argument("--avoid_collisions", dest="avoid_collisions", action="store_true",
                        help="Enable overlap guidance (default: enabled)")
    parser.add_argument("--no_avoid_collisions", dest="avoid_collisions", action="store_false",
                        help="Disable overlap guidance")
    parser.add_argument("--collision_weight", type=float, default=1.0,
                        help="Weight of the overlap reward")
    parser.add_argument("--overlap_clip", type=float, default=5.0,
                        help="Distance (m) beyond which overlap reward is ignored")
    parser.add_argument("--exact_overlap", action="store_true",
                        help="Use exact box-geometry overlap reward (accurate but O(A^3) memory; OOMs for large scenes). Default is the fast center-distance reward.")

    # Guidance: stay on road
    parser.add_argument("--stay_onroad", dest="stay_onroad", action="store_true",
                        help="Enable onroad guidance (default: enabled)")
    parser.add_argument("--no_stay_onroad", dest="stay_onroad", action="store_false",
                        help="Disable onroad guidance")
    parser.add_argument("--onroad_weight", type=float, default=0.1,
                        help="Weight of the onroad reward")

    # Shared guidance params
    parser.add_argument("--gradient_scale", type=float, default=0.1,
                        help="Guidance gradient scale; higher = stronger steering")
    parser.add_argument("--guidance_iter", type=int, default=5,
                        help="Guidance optimization iterations per diffusion step")

    # Output
    parser.add_argument("--video", action="store_true",
                        help="Render: original | raw diffusion | collision-cleaned diffusion")
    parser.add_argument("--fps", type=int, default=10, help="Video frames per second")
    parser.add_argument("--save_sim", action="store_true",
                        help="Also pickle the collision-cleaned Waymax sim trajectory")

    # Misc
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true",
                        help="Disable the diffusion progress bar")

    generate(parser.parse_args())
