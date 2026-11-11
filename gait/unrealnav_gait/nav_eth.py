"""ETH's navigation policy, reproduced for the A2 in mjlab: Lee, Bjelonic, Reske, Wellhausen, Miki, Hutter, "Learning
Robust Autonomous Navigation and Locomotion for Wheeled-Legged Robots", Science Robotics 9(89) adi9641, 2024 (arXiv
2405.01792). The design, what the paper gives and what it leaves out, and how ours differs: docs/NOTES-ETH-NAVIGATION.md. Where this file chooses something the paper does not give, the comment says "ours".

A high-level navigation policy (HLC) at 10 Hz gives velocity commands to a frozen low-level gait (LLC) at 50 Hz, in
physics, so a collision is the body touching something and what the gait cannot climb it does not climb. The HLC
observes (paper p.13):
    a height scan round the robot from the terrain: 3 m ahead, 1.5 m to the sides and behind; with the two scans
        0.1 s and 0.2 s before it                                                          group "height" (3, 46, 31)
    the gait's own state: the LLC's hidden layer (the paper's LLC is a GRU and hands over its hidden state; ours is an
        MLP, whose last hidden layer is the nearest thing: ours)                            in group "actor"
    20 visited positions, recorded 0.5 m apart, with the steps spent at each, in the robot's frame   group "memory"
    two waypoints, the two given before, and its three last actions                         in group "actor"
and is trained (paper p.12-13, p.18, Table S3) on worlds made by Wave Function Collapse from three tiles (Stair,
Floor 0, Floor 1) with a navigation graph, a path between two nodes by Dijkstra each episode, waypoints on it, rewards
for reaching the first waypoint, a dense reward toward it early in training, an exploration bonus from the visited
positions, standing still near it, staying alive, and the gait's own rewards (weighted in), PPO with Table S3's settings.

Not yet here (NOTES M2-M3): dynamic obstacles, height noise.

    .venv-gait/bin/python gait/train.py UnrealNav-Nav-ETH-Unitree-A2 --env.scene.num-envs 4096 --log-root data/gait/logs
"""
from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeneratorCfg, TerrainGeometry, TerrainOutput

from .velocity import a2_rough_env_cfg

ROOT = Path(__file__).resolve().parents[2]
GAIT = ROOT / "gait" / "policies" / "a2_rough.npz"  # the gait the closed loop walks with (src/unrealnav/policies/a2.py POLICY)

# The world (paper p.12, Fig. 8). Ours: the tile size, the grid, the heights. A2's gait gets over risers of 0.10 m
# reliably (policies/a2.py MAX_STEP_M), so a stair of eight steps rises 0.4-0.8 m: Floor 1 is that high, which the
# gait cannot climb onto from Floor 0 except by a stair, so Floor 1 is the walls, rooms and corridors.
TILE = 2.0
TILES = 8
PATCH = TILE * TILES  # m, one sub-terrain
RES = 0.05  # m, the height raster of a patch (the scan is sampled from it: a perfect elevation map; noise is M3)
STAIR_STEPS = 8
FLOOR1_HEIGHT = (0.4, 0.8)
TILE_WEIGHTS = (0.50, 0.30, 0.05, 0.05, 0.05, 0.05)  # F0, F1, stairs rising +x, -x, +y, -y (ours)
PATHS_PER_PATCH = 64
PATH_LENGTH = (5.0, 20.0)  # m along the graph (paper: Table S1's paths 5-20 m)
MAX_NODES = 16

# The HLC (paper p.13, p.21; Table S3).
NAV_DT = 0.1
EPISODE_S = 15.0
LOOKAHEAD = (5.0, 20.0)  # m, drawn each episode (paper p.12)
REACH = 0.75  # m, the first waypoint reached (paper eq. 9)
V_THRES = 0.5  # m/s (paper eq. 10)
MEMORY = 20
MEMORY_SPACING = 0.5
SCAN_X = (-1.5, 3.0)
SCAN_Y = (-1.5, 1.5)
SCAN_RES = 0.1  # ours: the paper gives no resolution
# The paper's bounds are vx [-1, 2], vy +-0.75, wz +-1.25 (p.13); the gait was trained on vx [-1, 1.5], vy +-0.6,
# wz +-1 (its meta), so its bounds (ours: clipped to what the LLC knows).
COMMAND_LOW = (-1.0, -0.6, -1.0)
COMMAND_HIGH = (1.5, 0.6, 1.0)
DENSE_DECAY_ITERATIONS = 1500  # ours: the paper says the dense reward "gradually diminished", not over how long
STEPS_PER_ITERATION = 74  # 2048 x 74 = 151,552 transitions an update: Table S3's batch of 150,000 (4096 robots ran out of GPU memory)


# ---------------------------------------------------------------------------------------------------- the world

F0, F1, SXP, SXN, SYP, SYN = range(6)
# The side of a tile facing +x, -x, +y, -y: a floor's level (0 or 1), a stair's end ("e0" low, "e1" high), or None
# (a stair's flank, which nothing walks across).
SIDES = {F0: (0, 0, 0, 0), F1: (1, 1, 1, 1),
         SXP: ("e1", "e0", None, None), SXN: ("e0", "e1", None, None),
         SYP: (None, None, "e1", "e0"), SYN: (None, None, "e0", "e1")}
OPPOSITE = (1, 0, 3, 2)
STEP = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _fits(mine, theirs) -> bool:
    """Can a tile side ``mine`` touch a neighbour's side ``theirs``: a stair's end only a floor of its own level."""
    for a, b in ((mine, theirs), (theirs, mine)):
        if isinstance(a, str):
            return isinstance(b, int) and b == int(a[1])
    return True


def wfc(rng: np.random.Generator, n: int = TILES, tries: int = 50) -> np.ndarray:
    """An n x n tile map by Wave Function Collapse (Gumin's algorithm, the paper's [30]): each cell starts with every
    tile, the least uncertain is collapsed at random by the tiles' weights, and its neighbours lose what no longer
    fits; a contradiction starts again. A stair's end must not face the edge of the map."""
    weights = np.array(TILE_WEIGHTS)
    for _ in range(tries):
        domain = np.ones((n, n, 6), bool)
        for i in range(n):
            for j in range(n):
                for t in range(6):
                    for side, (di, dj) in enumerate(STEP):
                        if not (0 <= i + di < n and 0 <= j + dj < n) and isinstance(SIDES[t][side], str):
                            domain[i, j, t] = False
        ok = True
        while ok:
            counts = domain.sum(-1)
            open_ = counts > 1
            if not open_.any():
                break
            c = np.where(open_, counts + rng.random((n, n)) * 0.1, np.inf)
            i, j = np.unravel_index(int(np.argmin(c)), (n, n))
            p = weights * domain[i, j]
            t = rng.choice(6, p=p / p.sum())
            domain[i, j] = False
            domain[i, j, t] = True
            queue = [(i, j)]
            while queue and ok:
                ci, cj = queue.pop()
                for side, (di, dj) in enumerate(STEP):
                    ni, nj = ci + di, cj + dj
                    if not (0 <= ni < n and 0 <= nj < n):
                        continue
                    keep = np.array([domain[ni, nj, u] and any(
                        domain[ci, cj, v] and _fits(SIDES[v][side], SIDES[u][OPPOSITE[side]]) for v in range(6)) for u in range(6)])
                    if not keep.any():
                        ok = False
                        break
                    if (keep != domain[ni, nj]).any():
                        domain[ni, nj] = keep
                        queue.append((ni, nj))
        if ok and (domain.sum(-1) == 1).all():
            return domain.argmax(-1)
    raise RuntimeError("wave function collapse found no tile map")


def graph(tiles: np.ndarray, h1: float):
    """The navigation graph: a node a tile (its centre, at its height), an edge where the robot can walk from one tile
    to the next: floor to floor of the same level, a stair's end to the floor of its level."""
    n = tiles.shape[0]
    z = {F0: 0.0, F1: h1}
    nodes = np.array([[(i + 0.5) * TILE, (j + 0.5) * TILE, z.get(int(tiles[i, j]), h1 / 2)] for i in range(n) for j in range(n)])
    edges = {k: [] for k in range(n * n)}
    for i in range(n):
        for j in range(n):
            for side, (di, dj) in enumerate(STEP[::2]):  # +x, +y: each pair once
                side = side * 2
                ni, nj = i + di, j + dj
                if ni >= n or nj >= n:
                    continue
                a, b = SIDES[int(tiles[i, j])][side], SIDES[int(tiles[ni, nj])][OPPOSITE[side]]
                linked = (isinstance(a, int) and isinstance(b, int) and a == b) or \
                         (isinstance(a, str) and isinstance(b, int) and b == int(a[1])) or \
                         (isinstance(b, str) and isinstance(a, int) and a == int(b[1]))
                if linked:
                    edges[i * n + j].append(ni * n + nj)
                    edges[ni * n + nj].append(i * n + j)
    return nodes, edges


def dijkstra(nodes, edges, start):
    dist = {start: 0.0}
    prev = {}
    heap = [(0.0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v in edges[u]:
            nd = d + float(np.linalg.norm(nodes[v, :2] - nodes[u, :2]))
            if nd < dist.get(v, np.inf):
                dist[v], prev[v] = nd, u
                heapq.heappush(heap, (nd, v))
    return dist, prev


def raster(tiles: np.ndarray, h1: float) -> np.ndarray:
    """The patch's heights on a RES grid: what the boxes make (a stair's step k of its eight is (k+1)/8 of h1 high)."""
    per = int(round(TILE / RES))
    out = np.zeros((tiles.shape[0] * per, tiles.shape[1] * per), np.float32)
    ramp = (np.floor(np.arange(per) / (per / STAIR_STEPS)) + 1) * h1 / STAIR_STEPS  # rising along the index
    for i in range(tiles.shape[0]):
        for j in range(tiles.shape[1]):
            t = int(tiles[i, j])
            block = out[i * per:(i + 1) * per, j * per:(j + 1) * per]
            if t == F1:
                block[:] = h1
            elif t == SXP:
                block[:] = ramp[:, None]
            elif t == SXN:
                block[:] = ramp[::-1, None]
            elif t == SYP:
                block[:] = ramp[None, :]
            elif t == SYN:
                block[:] = ramp[None, ::-1]
    return out


_PATCHES: list[dict] = []  # what each generated patch is, in the generator's order (row-major in random mode)


@dataclass(kw_only=True)
class WfcNavTerrainCfg(SubTerrainCfg):
    """A navigation world (paper Fig. 8): WFC's tiles as boxes, its graph, and paths on it."""

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        del difficulty
        body = spec.body("terrain")
        h1 = float(rng.uniform(*FLOOR1_HEIGHT))
        tiles = wfc(rng)
        geoms = []

        def box(centre, half, colour):
            g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=half, pos=centre)
            geoms.append(TerrainGeometry(geom=g, color=colour))

        box((PATCH / 2, PATCH / 2, -0.25), (PATCH / 2, PATCH / 2, 0.25), (0.45, 0.45, 0.45, 1.0))
        for i in range(TILES):  # Floor 1 as runs along x, one box a run
            j = 0
            while j < TILES:
                if tiles[i, j] != F1:
                    j += 1
                    continue
                k = j
                while k < TILES and tiles[i, k] == F1:
                    k += 1
                box(((i + 0.5) * TILE, (j + k) / 2 * TILE, h1 / 2), (TILE / 2, (k - j) * TILE / 2, h1 / 2), (0.6, 0.35, 0.3, 1.0))
                j = k
        depth = TILE / STAIR_STEPS
        for i in range(TILES):
            for j in range(TILES):
                t = int(tiles[i, j])
                if t < SXP:
                    continue
                for s in range(STAIR_STEPS):
                    top = (s + 1) * h1 / STAIR_STEPS
                    along = (s + 0.5) * depth if t in (SXP, SYP) else TILE - (s + 0.5) * depth
                    if t in (SXP, SXN):
                        centre, half = (i * TILE + along, (j + 0.5) * TILE, top / 2), (depth / 2, TILE / 2, top / 2)
                    else:
                        centre, half = ((i + 0.5) * TILE, j * TILE + along, top / 2), (TILE / 2, depth / 2, top / 2)
                    box(centre, half, (0.3, 0.5, 0.7, 1.0))

        nodes, edges = graph(tiles, h1)
        paths = np.zeros((PATHS_PER_PATCH, MAX_NODES, 3), np.float32)
        count = np.zeros(PATHS_PER_PATCH, np.int64)
        made, guard = 0, 0
        while made < PATHS_PER_PATCH and guard < 50 * PATHS_PER_PATCH:
            guard += 1
            start = int(rng.integers(len(nodes)))
            dist, prev = dijkstra(nodes, edges, start)
            ends = [v for v, d in dist.items() if PATH_LENGTH[0] <= d <= PATH_LENGTH[1]]
            if not ends:
                continue
            end = ends[int(rng.integers(len(ends)))]
            route = [end]
            while route[-1] != start:
                route.append(prev[route[-1]])
            route = route[::-1][:MAX_NODES]
            paths[made, :len(route)] = nodes[route]
            paths[made, len(route):] = nodes[route[-1]]
            count[made] = len(route)
            made += 1
        if made == 0:  # a world without one path of 5-20 m: a straight line on the floor (rare)
            paths[:] = np.array([[1.0, 1.0, 0.0]] * MAX_NODES)
            count[:] = 1
        elif made < PATHS_PER_PATCH:
            paths[made:], count[made:] = paths[:PATHS_PER_PATCH - made], count[:PATHS_PER_PATCH - made]
        _PATCHES.append({"height": raster(tiles, h1), "paths": paths, "count": count, "tiles": tiles, "h1": h1})
        return TerrainOutput(origin=np.array((PATCH / 2, PATCH / 2, 0.0)), geometries=geoms)


# ---------------------------------------------------------------------------------------------------- the LLC

@dataclass(kw_only=True)
class FrozenGaitActionCfg(ActionTermCfg):
    """The HLC's action, a velocity command in [0, 1]^3 (the Beta policy's range), walked by the frozen gait at 50 Hz."""
    policy: str = str(GAIT)
    gait_decimation: int = 4  # physics steps of 5 ms per gait step: 50 Hz

    def build(self, env) -> "FrozenGaitAction":
        return FrozenGaitAction(self, env)


class FrozenGaitAction(ActionTerm):
    """Runs gait/export_policy.py's arrays in torch, observing what src/unrealnav/policies/a2.py gives it: the IMU's
    velocity and turn rate, gravity in the body, joints minus the default pose, joint speeds, its last action, the
    command. Keeps its last hidden layer (128 numbers) for the HLC."""

    def __init__(self, cfg: FrozenGaitActionCfg, env):
        super().__init__(cfg, env)
        with np.load(cfg.policy, allow_pickle=False) as f:
            arrays = {k: f[k] for k in f.files}
        meta = json.loads(str(arrays.pop("meta")))
        if int(meta.get("history_length", 1)) != 1 or len(meta["observation_names"]) != 7:
            raise ValueError(f"{cfg.policy}: a one-frame velocity gait is expected")
        dev = env.device
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)  # noqa: E731
        self._mean, self._scale = t(arrays["mean"]), t(arrays["scale"])
        n_layers = sum(k.startswith("weight") for k in arrays)
        self._layers = [(t(arrays[f"weight{k}"]), t(arrays[f"bias{k}"])) for k in range(n_layers)]
        self._joint_ids, names = self._entity.find_joints(list(meta["joint_names"]), preserve_order=True)
        assert names == list(meta["joint_names"]), names
        self._default = t(meta["default_joint_pos"])
        self._action_scale = t(meta["action_scale"])
        n = env.num_envs
        self._low, self._high = t(COMMAND_LOW), t(COMMAND_HIGH)
        self._raw = torch.zeros(n, 3, device=dev)
        self.command = torch.zeros(n, 3, device=dev)
        self.history = torch.zeros(n, 3, 3, device=dev)  # the HLC's last three actions, newest first
        self.gait_actions = torch.zeros(n, len(self._joint_ids), device=dev)
        self.prev_gait_actions = torch.zeros_like(self.gait_actions)
        self.hidden = torch.zeros(n, self._layers[-1][0].shape[1], device=dev)
        self._sub = 0
        self._lin_vel = env.scene["robot/imu_lin_vel"]
        self._ang_vel = env.scene["robot/imu_ang_vel"]

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw[:] = actions
        self.history = torch.cat([actions[:, None], self.history[:, :2]], dim=1)
        self.command[:] = self._low + actions.clamp(0.0, 1.0) * (self._high - self._low)
        self._sub = 0

    def apply_actions(self) -> None:
        if self._sub % self.cfg.gait_decimation == 0:
            d = self._entity.data
            x = torch.cat([self._lin_vel.data, self._ang_vel.data, d.projected_gravity_b,
                           d.joint_pos[:, self._joint_ids] - self._default, d.joint_vel[:, self._joint_ids],
                           self.gait_actions, self.command], dim=1)
            x = (x - self._mean) / self._scale
            for k, (w, b) in enumerate(self._layers):
                x = x @ w.T + b
                if k < len(self._layers) - 1:
                    x = torch.nn.functional.elu(x)
                    if k == len(self._layers) - 2:
                        self.hidden = x
            self.prev_gait_actions = self.gait_actions
            self.gait_actions = x
            self._entity.set_joint_position_target(self._default + self._action_scale * x, joint_ids=self._joint_ids)
        self._sub += 1

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for buf in (self._raw, self.command, self.history, self.gait_actions, self.prev_gait_actions, self.hidden):
            buf[ids] = 0.0
        return {}


# ---------------------------------------------------------------------------------------------------- the HLC's state

def _yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class NavState:
    """Everything the HLC's observations and rewards share, one per env (attached to it): the patches' height rasters
    and paths, and per robot its path, lookahead, waypoints, visited positions and scans."""

    def __init__(self, env):
        self.env = env
        dev = env.device
        terrain = env.scene.terrain
        rows, cols = terrain.terrain_origins.shape[:2]
        patches = _PATCHES[-rows * cols:]
        assert len(patches) == rows * cols, "the navigation world was not generated in this process"
        self.cols = cols
        self.height = torch.as_tensor(np.stack([p["height"] for p in patches]), device=dev)  # (P, H, W)
        self.paths = torch.as_tensor(np.stack([p["paths"] for p in patches]), device=dev)  # (P, K, M, 3) local
        self.count = torch.as_tensor(np.stack([p["count"] for p in patches]), device=dev)  # (P, K)
        corner = terrain.terrain_origins - torch.tensor([PATCH / 2, PATCH / 2, 0.0], device=dev)
        self.corner = corner.reshape(-1, 3)  # (P, 3), row-major as generated
        n = env.num_envs
        self.patch = torch.zeros(n, dtype=torch.long, device=dev)
        self.path = torch.zeros(n, MAX_NODES, 3, device=dev)  # world
        self.npts = torch.ones(n, dtype=torch.long, device=dev)
        self.lookahead = torch.full((n,), LOOKAHEAD[0], device=dev)
        self.wp = torch.zeros(n, 2, 3, device=dev)  # WP1, WP2, world
        self.wp_hist = torch.zeros(n, 2, 2, 3, device=dev)  # the two before, world
        self.mem = torch.zeros(n, MEMORY, 2, device=dev)  # world xy
        self.mem_steps = torch.zeros(n, MEMORY, device=dev)
        self.mem_len = torch.zeros(n, dtype=torch.long, device=dev)
        gx = torch.arange(SCAN_X[0], SCAN_X[1] + SCAN_RES / 2, SCAN_RES, device=dev)
        gy = torch.arange(SCAN_Y[0], SCAN_Y[1] + SCAN_RES / 2, SCAN_RES, device=dev)
        self.scan_shape = (len(gx), len(gy))
        self.grid = torch.stack(torch.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)  # (G, 2) body yaw frame
        self.scans = torch.zeros(n, 3, *self.scan_shape, device=dev)
        self.reached = torch.zeros(n, dtype=torch.bool, device=dev)
        self.stamp = -1

    # where the robot is
    def pose(self):
        d = self.env.scene["robot"].data
        return d.root_link_pos_w, _yaw(d.root_link_quat_w)

    def heights(self, xy: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        """Terrain height at world xy (n, G, 2) on each robot's patch (nearest cell, clamped to the patch)."""
        local = xy - self.corner[patch][:, None, :2]
        idx = (local / RES).long()
        h, w = self.height.shape[1:]
        i, j = idx[..., 0].clamp(0, h - 1), idx[..., 1].clamp(0, w - 1)
        return self.height[patch[:, None], i, j] + self.corner[patch][:, None, 2:3].squeeze(-1)

    def _point_at(self, s: torch.Tensor) -> torch.Tensor:
        """The point at arc length s (n,) along each robot's path."""
        seg = torch.linalg.norm(torch.diff(self.path[..., :2], dim=1), dim=-1)  # (n, M-1)
        arc = torch.cat([torch.zeros_like(seg[:, :1]), seg.cumsum(1)], 1)
        s = torch.minimum(s, arc.gather(1, (self.npts - 1)[:, None]).squeeze(1))
        k = torch.searchsorted(arc.contiguous(), s[:, None].contiguous()).clamp(1, MAX_NODES - 1)
        a0, a1 = arc.gather(1, k - 1), arc.gather(1, k)
        f = ((s[:, None] - a0) / (a1 - a0).clamp(min=1e-6)).clamp(0, 1)
        p0 = self.path.gather(1, (k - 1)[..., None].expand(-1, 1, 3)).squeeze(1)
        p1 = self.path.gather(1, k[..., None].expand(-1, 1, 3)).squeeze(1)
        return p0 + f * (p1 - p0)

    def _projection(self, xy: torch.Tensor) -> torch.Tensor:
        """Arc length of the point of each robot's path nearest to it."""
        a, b = self.path[:, :-1, :2], self.path[:, 1:, :2]
        ab = b - a
        t = (((xy[:, None] - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-9)).clamp(0, 1)
        d = torch.linalg.norm(a + t[..., None] * ab - xy[:, None], dim=-1)
        valid = torch.arange(MAX_NODES - 1, device=xy.device)[None] < (self.npts - 1)[:, None]
        d = torch.where(valid, d, torch.full_like(d, 1e9))
        k = d.argmin(1)
        seg = torch.linalg.norm(ab, dim=-1)
        arc = torch.cat([torch.zeros_like(seg[:, :1]), seg.cumsum(1)], 1)
        return arc.gather(1, k[:, None]).squeeze(1) + t.gather(1, k[:, None]).squeeze(1) * seg.gather(1, k[:, None]).squeeze(1)

    def waypoints(self, xy: torch.Tensor) -> torch.Tensor:
        """WP1 and WP2 (ours, reading p.12): the path's points one and two lookaheads beyond the robot's projection
        onto it; at the path's end, its last node twice (paper p.12)."""
        s = self._projection(xy)
        return torch.stack([self._point_at(s + self.lookahead), self._point_at(s + 2 * self.lookahead)], 1)

    def start(self, ids: torch.Tensor) -> None:
        """A new path for these robots on a random patch (paper p.4: new paths every episode)."""
        terrain = self.env.scene.terrain
        terrain.randomize_env_origins(ids)
        patch = terrain.terrain_levels[ids] * self.cols + terrain.terrain_types[ids]
        k = torch.randint(0, self.paths.shape[1], (len(ids),), device=ids.device)
        self.patch[ids] = patch
        self.path[ids] = self.paths[patch, k] + self.corner[patch][:, None]
        self.npts[ids] = self.count[patch, k].clamp(min=1)
        self.lookahead[ids] = torch.empty(len(ids), device=ids.device).uniform_(*LOOKAHEAD)
        self.mem_len[ids] = 0
        self.mem_steps[ids] = 0.0
        self.reached[ids] = False
        self.stamp = -1  # observations are made afresh for everyone at the next read

    def update(self) -> None:
        """Once per env step, before the observations are read: waypoints, the visited positions, the scans."""
        step = self.env.common_step_counter
        if self.stamp == step:
            return
        self.stamp = step
        pos, yaw = self.pose()
        xy = pos[:, :2]
        fresh = self.env.episode_length_buf == 0
        wp = self.waypoints(xy)
        self.wp_hist = torch.where(fresh[:, None, None, None], wp[:, None].expand(-1, 2, -1, -1),
                                   torch.stack([self.wp, self.wp_hist[:, 0]], 1))
        self.wp = wp
        # visited positions: the nearest within MEMORY_SPACING gets the step, or a new one is recorded (paper p.13)
        n = xy.shape[0]
        slots = torch.arange(MEMORY, device=xy.device)[None]
        used = slots < self.mem_len[:, None]
        d = torch.linalg.norm(self.mem - xy[:, None], dim=-1)
        d = torch.where(used, d, torch.full_like(d, 1e9))
        near, at = d.min(1)
        hit = near < MEMORY_SPACING
        rows = torch.arange(n, device=xy.device)
        self.mem_steps[rows[hit], at[hit]] += 1.0
        new = ~hit
        if new.any():
            full = self.mem_len >= MEMORY
            roll = new & full
            if roll.any():  # the oldest is dropped
                self.mem[roll] = torch.roll(self.mem[roll], -1, 1)
                self.mem_steps[roll] = torch.roll(self.mem_steps[roll], -1, 1)
                self.mem_len[roll] = MEMORY - 1
            slot = self.mem_len[new]
            self.mem[rows[new], slot] = xy[new]
            self.mem_steps[rows[new], slot] = 1.0
            self.mem_len[new] += 1
        # the scan: heights of the grid round the robot, relative to its base (ours: the paper gives no reference)
        c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
        gx, gy = self.grid[None, :, 0], self.grid[None, :, 1]
        world = torch.stack([xy[:, 0:1] + c * gx - s * gy, xy[:, 1:2] + s * gx + c * gy], -1)
        scan = (self.heights(world, self.patch) - pos[:, 2:3]).clamp(-2.0, 1.5).view(n, *self.scan_shape)
        self.scans = torch.where(fresh[:, None, None, None], scan[:, None].expand(-1, 3, -1, -1),
                                 torch.stack([scan, self.scans[:, 0], self.scans[:, 1]], 1))
        self.reached |= torch.linalg.norm(xy - self.path.gather(1, (self.npts - 1)[:, None, None].expand(-1, 1, 3)).squeeze(1)[:, :2], dim=-1) < 0.5


def nav(env) -> NavState:
    if not hasattr(env, "_eth_nav"):
        env._eth_nav = NavState(env)
    return env._eth_nav


def gait(env) -> FrozenGaitAction:
    return env.action_manager.get_term("nav")


# ---------------------------------------------------------------------------------------------------- MDP terms

def reset_on_path(env, env_ids, yaw_spread: float = math.pi / 2) -> None:
    """A new path, the robot at its start, facing along it within yaw_spread (ours: the paper does not say)."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    st = nav(env)
    st.start(env_ids)
    p0, p1 = st.path[env_ids, 0], st.path[env_ids, 1]
    heading = torch.atan2(p1[:, 1] - p0[:, 1], p1[:, 0] - p0[:, 0])
    yaw = heading + torch.empty_like(heading).uniform_(-yaw_spread, yaw_spread)
    robot = env.scene["robot"]
    root = robot.data.default_root_state[env_ids].clone()
    root[:, 0:2] = p0[:, :2]
    # standing on the highest ground under its footprint (a node's height is the tile's middle; on a stair the step
    # under the body is higher than that)
    off = torch.stack(torch.meshgrid(torch.linspace(-0.45, 0.45, 7, device=env.device),
                                     torch.linspace(-0.3, 0.3, 5, device=env.device), indexing="ij"), -1).reshape(-1, 2)
    ground = st.heights(p0[:, None, :2] + off[None], st.patch[env_ids]).max(1).values
    root[:, 2] = ground + root[:, 2]
    root[:, 3:7] = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], -1)
    robot.write_root_link_pose_to_sim(root[:, 0:7], env_ids=env_ids)
    robot.write_root_link_velocity_to_sim(torch.zeros_like(root[:, 7:13]), env_ids=env_ids)


def _robot_frame(env, world: torch.Tensor) -> torch.Tensor:
    """World points (n, ..., 3) in each robot's yaw frame, relative to it."""
    pos, yaw = nav(env).pose()
    d = world - pos.view(-1, *([1] * (world.dim() - 2)), 3)
    c, s = torch.cos(yaw).view(-1, *([1] * (world.dim() - 2))), torch.sin(yaw).view(-1, *([1] * (world.dim() - 2)))
    return torch.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1], d[..., 2]], -1)


def obs_waypoints(env) -> torch.Tensor:
    st = nav(env)
    st.update()
    return torch.cat([_robot_frame(env, st.wp).flatten(1), _robot_frame(env, st.wp_hist.flatten(1, 2)).flatten(1)], 1)


def obs_last_actions(env) -> torch.Tensor:
    return gait(env).history.flatten(1)


def obs_gait_state(env) -> torch.Tensor:
    return gait(env).hidden


def obs_height(env) -> torch.Tensor:
    st = nav(env)
    st.update()
    return st.scans


def obs_memory(env) -> torch.Tensor:
    """(n, 3, MEMORY, 1): x, y in the robot's frame, seconds spent there; unused slots zero."""
    st = nav(env)
    st.update()
    rel = _robot_frame(env, torch.cat([st.mem, torch.zeros_like(st.mem[..., :1])], -1))[..., :2]
    used = (torch.arange(MEMORY, device=env.device)[None] < st.mem_len[:, None]).float()
    out = torch.cat([rel * used[..., None], (st.mem_steps * NAV_DT * used)[..., None]], -1)  # (n, M, 3)
    return out.permute(0, 2, 1)[..., None]


def _to_wp1(env):
    st = nav(env)
    pos, _ = st.pose()
    e = st.wp[:, 0, :2] - pos[:, :2]
    return st, pos, e, torch.linalg.norm(e, dim=-1)


def rew_goal(env) -> torch.Tensor:
    """Eq. 9: 1 within REACH of the first waypoint."""
    _, _, _, dist = _to_wp1(env)
    return (dist < REACH).float()


def rew_dense(env) -> torch.Tensor:
    """Eq. 10: 1 within REACH, else the speed toward the first waypoint, capped at V_THRES, over V_THRES."""
    _, _, e, dist = _to_wp1(env)
    v = env.scene["robot"].data.root_link_lin_vel_w[:, :2]
    toward = (v * e / dist.clamp(min=1e-6)[:, None]).sum(-1).clamp(0.0, V_THRES) / V_THRES
    return torch.where(dist < REACH, torch.ones_like(toward), toward)


def rew_exploration(env) -> torch.Tensor:
    """Eq. 11-12: away from the first waypoint, minus the steps spent at each remembered position within 1 m."""
    st, pos, _, dist = _to_wp1(env)
    used = torch.arange(MEMORY, device=env.device)[None] < st.mem_len[:, None]
    near = (torch.linalg.norm(st.mem - pos[:, None, :2], dim=-1) < 1.0) & used
    cost = -(st.mem_steps * near).sum(1)
    return torch.where(dist < REACH, torch.zeros_like(cost), cost)


def rew_stability(env) -> torch.Tensor:
    """Eq. 13: exp(-2 |v|^2) within REACH."""
    _, _, _, dist = _to_wp1(env)
    v = env.scene["robot"].data.root_link_lin_vel_w[:, :2]
    return torch.where(dist < REACH, torch.exp(-2.0 * (v * v).sum(-1)), torch.zeros_like(dist))


def rew_track_lin(env) -> torch.Tensor:
    """Eq. 14 (the LLC's, on the HLC's command)."""
    g = gait(env)
    v = g._lin_vel.data[:, :2]
    want = g.command[:, :2]
    still = torch.linalg.norm(want, dim=-1) < 0.05
    return torch.where(still, 2.0 * torch.exp(-2.0 * (v * v).sum(-1)),
                       torch.exp(-2.0 * ((v - want) ** 2).sum(-1)) + (want * v).sum(-1))


def rew_track_ang(env) -> torch.Tensor:
    """Eq. 15."""
    g = gait(env)
    return torch.exp(-2.0 * (g._ang_vel.data[:, 2] - g.command[:, 2]) ** 2)


def rew_body_motion(env) -> torch.Tensor:
    """Eq. 16."""
    g = gait(env)
    return -1.25 * g._lin_vel.data[:, 2] ** 2 - 0.4 * g._ang_vel.data[:, 0].abs() - 0.4 * g._ang_vel.data[:, 1].abs()


def rew_orientation(env) -> torch.Tensor:
    """Eq. 17 (as a cost): the angle between the world's and the body's z axes, squared."""
    gz = env.scene["robot"].data.projected_gravity_b[:, 2]
    return torch.arccos((-gz).clamp(-1.0, 1.0)) ** 2


def rew_gait_smoothness(env) -> torch.Tensor:
    g = gait(env)
    return ((g.gait_actions - g.prev_gait_actions) ** 2).sum(-1)


def dense_decay(env, env_ids, term: str = "dense", iterations: int = DENSE_DECAY_ITERATIONS, weight: float = 1.0):
    """Paper p.12: the dense reward 'gradually diminished', and the policy trained on the sparse one at the end."""
    left = max(0.0, 1.0 - env.common_step_counter / (iterations * STEPS_PER_ITERATION))
    env.reward_manager.get_term_cfg(term).weight = weight * left
    return {"dense_weight": weight * left}


def metric_reached(env) -> torch.Tensor:
    """Logged only: the path's end reached within 0.5 m this episode (Table S1's success)."""
    return nav(env).reached.float()


# ---------------------------------------------------------------------------------------------------- the task

# r_h + w_l (r_l + r_r) (paper p.18). The paper gives no weights: ours, set so that the two halves are of a size
# over an episode (its rule), to be read off the first training's log and adjusted.
W_HLC = {"goal": 10.0, "dense": 2.0, "exploration": 0.02, "stability": 2.0, "alive": 1.0}  # exploration: eq. 11-12 are already negative
W_LLC = 0.5
LLC_TERMS = {"track_lin": 1.0, "track_ang": 0.5, "body_motion": 1.0, "orientation": -2.0, "torques": -1e-5,
             "joint_vel": -1e-4, "gait_smoothness": -0.01, "shank_contact": -1.0, "trunk_contact": -1.0}


def nav_eth_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = a2_rough_env_cfg(deployment_baseline=True, payload=True)  # the robot, sim and contact sensors of the gait's training
    cfg.decimation = int(round(NAV_DT / cfg.sim.mujoco.timestep))  # 20 physics steps a nav step
    cfg.sim.nconmax = 100  # the gait's 60 overflowed (68) among a stair's step boxes; 160 x 4096 robots asked for 19.7 GB at once
    cfg.episode_length_s = EPISODE_S
    cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name not in ("terrain_scan", "foot_height_scan"))
    for s in cfg.scene.sensors:  # the contact history covers a nav step
        if getattr(s, "history_length", 0):
            s.history_length = cfg.decimation
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        size=(PATCH, PATCH), num_rows=8, num_cols=8, curriculum=False, border_width=4.0, border_height=1.0,
        sub_terrains={"wfc": WfcNavTerrainCfg()}, color_scheme="height")
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.commands = {}
    cfg.actions = {"nav": FrozenGaitActionCfg(entity_name="robot")}
    cfg.events = {k: v for k, v in cfg.events.items() if v.mode == "startup" and k != "encoder_bias"}
    cfg.events["reset_robot_joints"] = EventTermCfg(func=mdp.reset_joints_by_offset, mode="reset",
                                                    params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0),
                                                            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))})
    cfg.events["reset_on_path"] = EventTermCfg(func=reset_on_path, mode="reset", params={})

    one = lambda f: ObservationGroupCfg(terms={"x": ObservationTermCfg(func=f)}, concatenate_terms=True, enable_corruption=False)  # noqa: E731
    cfg.observations = {
        "actor": ObservationGroupCfg(terms={"waypoints": ObservationTermCfg(func=obs_waypoints),
                                            "last_actions": ObservationTermCfg(func=obs_last_actions),
                                            "gait_state": ObservationTermCfg(func=obs_gait_state)},
                                     concatenate_terms=True, enable_corruption=False),
        "height": one(obs_height),
        "memory": one(obs_memory),
    }

    rewards = {
        "goal": RewardTermCfg(func=rew_goal, weight=W_HLC["goal"]),
        "dense": RewardTermCfg(func=rew_dense, weight=W_HLC["dense"]),
        "exploration": RewardTermCfg(func=rew_exploration, weight=W_HLC["exploration"]),
        "stability": RewardTermCfg(func=rew_stability, weight=W_HLC["stability"]),
        "alive": RewardTermCfg(func=envs_mdp.is_alive, weight=W_HLC["alive"]),
        "track_lin": RewardTermCfg(func=rew_track_lin, weight=W_LLC * LLC_TERMS["track_lin"]),
        "track_ang": RewardTermCfg(func=rew_track_ang, weight=W_LLC * LLC_TERMS["track_ang"]),
        "body_motion": RewardTermCfg(func=rew_body_motion, weight=W_LLC * LLC_TERMS["body_motion"]),
        "orientation": RewardTermCfg(func=rew_orientation, weight=W_LLC * LLC_TERMS["orientation"]),
        "torques": RewardTermCfg(func=envs_mdp.joint_torques_l2, weight=W_LLC * LLC_TERMS["torques"]),
        "joint_vel": RewardTermCfg(func=envs_mdp.joint_vel_l2, weight=W_LLC * LLC_TERMS["joint_vel"]),
        "gait_smoothness": RewardTermCfg(func=rew_gait_smoothness, weight=W_LLC * LLC_TERMS["gait_smoothness"]),
        "shank_contact": RewardTermCfg(func=mdp.self_collision_cost, weight=W_LLC * LLC_TERMS["shank_contact"],
                                       params={"sensor_name": "shank_ground_touch"}),
        "trunk_contact": RewardTermCfg(func=mdp.self_collision_cost, weight=W_LLC * LLC_TERMS["trunk_contact"],
                                       params={"sensor_name": "trunk_ground_touch"}),
        "reached": RewardTermCfg(func=metric_reached, weight=0.0),
    }
    cfg.rewards = rewards
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "thigh_contact": TerminationTermCfg(func=mdp.illegal_contact, params={"sensor_name": "thigh_ground_touch"}),
        "fell_over": TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}),
        "out_of_terrain_bounds": TerminationTermCfg(func=mdp.out_of_terrain_bounds, time_out=True),
    }
    cfg.curriculum = {} if play else {"dense_decay": CurriculumTermCfg(func=dense_decay, params={"weight": W_HLC["dense"]})}
    cfg.metrics = {}
    if play:
        cfg.episode_length_s = 60.0  # Table S1: 60 s to arrive
    return cfg


def nav_eth_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Table S3: discount 0.991, KL target 0.01, clip 0.2, entropy 0.001, a batch of 150,000 in 10 minibatches, 5
    epochs, adaptive learning rate. Networks (p.14): the height scan through a 3-layer 2D CNN, the visited positions
    through a 1 x 1 1D CNN and a max pool (PointNet-like), the rest an MLP; a Beta policy (sizes: ours)."""
    groups = ("actor", "height", "memory")
    cnn = {"height": {"output_channels": [16, 32, 16], "kernel_size": [5, 3, 3], "stride": [2, 2, 2], "activation": "elu"},
           "memory": {"output_channels": [32, 32], "kernel_size": [1, 1], "stride": [1, 1], "activation": "elu", "global_pool": "max"}}
    return RslRlOnPolicyRunnerCfg(
        obs_groups={"actor": groups, "critic": groups},
        actor=RslRlModelCfg(class_name="CNNModel", hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
                            cnn_cfg=cnn, distribution_cfg={"class_name": "BetaDistribution", "action_range": (0.0, 1.0)}),
        critic=RslRlModelCfg(class_name="CNNModel", hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True, cnn_cfg=cnn),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.001, num_learning_epochs=5,
            num_mini_batches=10, learning_rate=1.0e-3, schedule="adaptive", gamma=0.991, lam=0.95, desired_kl=0.01, max_grad_norm=1.0),
        experiment_name="a2_nav_eth", save_interval=50, num_steps_per_env=STEPS_PER_ITERATION, max_iterations=4000,
        logger="tensorboard")
