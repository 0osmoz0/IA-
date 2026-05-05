"""Environnement Gymnasium : labyrinthe avec curriculum et mécaniques."""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Deque, Dict, Optional, Set, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.curriculum import CurriculumManager, level_layout, MAX_LEVEL
from src.maze_gen import farthest_cell_from, generate_maze
from src.mechanics import MechanicPlacements

LOCAL = 9
LEVEL_DIM = MAX_LEVEL + 1


class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        seed: Optional[int] = None,
        curriculum: Optional[CurriculumManager] = None,
        *,
        sticky_maze: bool = True,
    ):
        super().__init__()
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.curriculum = curriculum or CurriculumManager()
        self.level = 0
        self.sticky_maze = sticky_maze
        # Si sticky : nouvelle grille seulement après succès (sortie). Sinon : à chaque reset d'épisode.
        self._full_maze_reset_next = True

        self.action_space = spaces.Discrete(4)
        # patch + norm pos + has_key + sin/cos phase + level one-hot + vec vers sortie
        obs_dim = LOCAL * LOCAL + 2 + 1 + 2 + LEVEL_DIM + 2
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self._grid: np.ndarray = np.zeros((9, 9), dtype=np.int8)
        self._exit: Tuple[int, int] = (1, 1)
        self._start: Tuple[int, int] = (1, 1)
        self._agent: Tuple[int, int] = (1, 1)
        self._mech = MechanicPlacements()
        self._has_key = False
        self._key_taken = False
        self._step_count = 0
        self._episode_steps = 0
        self._max_steps = 200
        self._last_info: Dict[str, Any] = {}
        self._last_action: int = -1
        self._last_moved: bool = False
        self._last_reward_step: float = 0.0
        self._agent_trail: Deque[Tuple[int, int]] = deque(maxlen=40)

    def _seed_trail_at_agent(self) -> None:
        self._agent_trail.clear()
        self._agent_trail.append(self._agent)

    def _layout(self) -> Dict[str, Any]:
        return level_layout(self.level)

    def _pick_corridors(self, exclude: Set[Tuple[int, int]]) -> list[Tuple[int, int]]:
        ys, xs = np.where(self._grid == 0)
        return [(int(y), int(x)) for y, x in zip(ys, xs) if (int(y), int(x)) not in exclude]

    def _reset_maze(self) -> None:
        lay = self._layout()
        h, w = int(lay["maze_h"]), int(lay["maze_w"])
        self._grid = generate_maze(h, w, self.rng)
        self._start = (1, 1)
        if self._grid[self._start] != 0:
            walk = self._pick_corridors(set())
            self._start = walk[0]
        self._exit = farthest_cell_from(self._grid, self._start[0], self._start[1])
        excl = {self._start, self._exit}
        self._mech = MechanicPlacements(
            mobile_period=6 + self.level * 2,
        )

        if lay["use_mobile_wall"]:
            cands = self._pick_corridors(excl)
            if cands:
                self.rng.shuffle(cands)
                self._mech.mobile_wall = cands[0]

        if lay["use_key_door"]:
            cands = [p for p in self._pick_corridors(excl) if p != self._mech.mobile_wall]
            if len(cands) >= 2:
                self.rng.shuffle(cands)
                self._mech.key_pos = cands[0]
                self._mech.door_pos = cands[1]
                excl.update({self._mech.key_pos, self._mech.door_pos})

        if lay["use_traps"]:
            n_traps = 2 + self.level
            cands = [p for p in self._pick_corridors(excl) if p != self._mech.mobile_wall]
            self.rng.shuffle(cands)
            for t in cands[:n_traps]:
                self._mech.trap_cells.add(t)
                excl.add(t)

        if lay["use_portal"]:
            cands = [p for p in self._pick_corridors(excl) if p != self._mech.mobile_wall]
            if len(cands) >= 2:
                self.rng.shuffle(cands)
                self._mech.portal_a = cands[0]
                self._mech.portal_b = cands[1]

        self._has_key = False
        self._key_taken = False
        self._agent = self._start
        self._max_steps = max(150, self._grid.size // 2 + 50 * (self.level + 1))
        self._step_count = 0
        self._last_action = -1
        self._last_moved = False
        self._last_reward_step = 0.0
        self._seed_trail_at_agent()

    def _soft_reset_same_maze(self) -> None:
        """Même labyrinthe : repositionne l'agent et l'état d'épisode (échec piège ou timeout)."""
        self._episode_steps = 0
        self._step_count = 0
        self._has_key = False
        self._key_taken = False
        self._agent = self._start
        self._last_action = -1
        self._last_moved = False
        self._last_reward_step = 0.0
        self._seed_trail_at_agent()

    def _mobile_blocking(self) -> bool:
        if self._mech.mobile_wall is None:
            return False
        half = max(1, self._mech.mobile_period // 2)
        return (self._step_count // half) % 2 == 1

    def _blocked(self, y: int, x: int) -> bool:
        h, w = self._grid.shape
        if y < 0 or x < 0 or y >= h or x >= w:
            return True
        if self._grid[y, x] == 1:
            return True
        if self._mech.mobile_wall == (y, x) and self._mobile_blocking():
            return True
        if self._mech.door_pos == (y, x) and not self._has_key:
            return True
        return False

    def _cell_feature(self, y: int, x: int) -> float:
        """Valeur dans [-1,1] pour une cellule visible localement."""
        h, w = self._grid.shape
        if y < 0 or x < 0 or y >= h or x >= w:
            return 1.0
        if self._grid[y, x] == 1:
            return 1.0
        if self._mech.mobile_wall == (y, x) and self._mobile_blocking():
            return 1.0
        if self._mech.door_pos == (y, x) and not self._has_key:
            return 0.45
        if (y, x) == self._exit:
            return 0.85
        if self._mech.key_pos == (y, x) and not self._key_taken:
            return 0.55
        if (y, x) in self._mech.trap_cells:
            return 0.25
        if self._mech.portal_a == (y, x) or self._mech.portal_b == (y, x):
            return 0.65
        if self._mech.mobile_wall == (y, x):
            return -0.1
        return 0.0

    def _observe(self) -> np.ndarray:
        ay, ax = self._agent
        half = LOCAL // 2
        patch = []
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                patch.append(self._cell_feature(ay + dy, ax + dx))
        patch_arr = np.clip(np.array(patch, dtype=np.float32), -1.0, 1.0)

        h, w = self._grid.shape
        norm_y = (ay / max(1, h - 1)) * 2 - 1
        norm_x = (ax / max(1, w - 1)) * 2 - 1

        phase = (self._step_count % max(1, self._mech.mobile_period)) / max(
            1, self._mech.mobile_period
        )
        sin_p = np.sin(phase * 2 * np.pi).astype(np.float32)
        cos_p = np.cos(phase * 2 * np.pi).astype(np.float32)

        level_oh = np.zeros(LEVEL_DIM, dtype=np.float32)
        level_oh[min(self.level, MAX_LEVEL)] = 1.0

        gy, gx = self._exit
        gdx = np.float32((gx - ax) / max(1.0, float(w - 1)))
        gdy = np.float32((gy - ay) / max(1.0, float(h - 1)))

        return np.concatenate(
            [
                patch_arr,
                np.array([norm_y, norm_x, float(self._has_key), sin_p, cos_p], dtype=np.float32),
                level_oh,
                np.array([gdx, gdy], dtype=np.float32),
            ]
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
            self.np_rng = np.random.default_rng(seed)
        self.level = self.curriculum.level
        self._episode_steps = 0
        if not self.sticky_maze or self._full_maze_reset_next:
            self._reset_maze()
        else:
            self._soft_reset_same_maze()
        obs = self._observe()
        self._last_info = {
            "level": self.level,
            "exit": self._exit,
            "agent": self._agent,
            "success": False,
        }
        return obs, self._last_info

    def step(self, action: int):
        prev: Tuple[int, int] = self._agent
        self._step_count += 1
        self._episode_steps += 1

        deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dy, dx = deltas[int(action)]
        ny, nx = self._agent[0] + dy, self._agent[1] + dx

        reward = np.float32(-0.02)
        terminated = False
        truncated = False

        if not self._blocked(ny, nx):
            self._agent = (ny, nx)

        y, x = self._agent

        if (y, x) in self._mech.trap_cells:
            reward = np.float32(-5.0)
            terminated = True

        if not terminated and not self._key_taken and self._mech.key_pos == (y, x):
            self._has_key = True
            self._key_taken = True
            reward += np.float32(0.5)

        if not terminated and self._mech.portal_a is not None and self._mech.portal_b is not None:
            if (y, x) == self._mech.portal_a:
                self._agent = self._mech.portal_b
            elif (y, x) == self._mech.portal_b:
                self._agent = self._mech.portal_a
            y, x = self._agent
            if (y, x) in self._mech.trap_cells:
                reward = np.float32(-5.0)
                terminated = True

        success = False
        if not terminated and (y, x) == self._exit:
            reward += np.float32(10.0)
            terminated = True
            success = True

        if self._episode_steps >= self._max_steps:
            truncated = True

        self._last_action = int(action)
        self._last_moved = self._agent != prev
        self._last_reward_step = float(reward)
        self._agent_trail.append(self._agent)

        obs = self._observe()
        self._last_info = {
            "level": self.level,
            "exit": self._exit,
            "agent": self._agent,
            "success": success,
            "reward": float(reward),
        }
        if self.sticky_maze and (terminated or truncated):
            self._full_maze_reset_next = success
        return obs, float(reward), terminated, truncated, self._last_info

    def snapshot_for_render(self) -> Dict[str, Any]:
        """État sérialisable pour synchroniser le rendu Ursina."""
        return {
            "grid": self._grid.copy(),
            "agent": self._agent,
            "exit": self._exit,
            "level": self.level,
            "has_key": self._has_key,
            "key_pos": self._mech.key_pos,
            "key_taken": self._key_taken,
            "door_pos": self._mech.door_pos,
            "traps": set(self._mech.trap_cells),
            "portal_a": self._mech.portal_a,
            "portal_b": self._mech.portal_b,
            "mobile_wall": self._mech.mobile_wall,
            "mobile_blocking": self._mobile_blocking(),
            "step_count": self._step_count,
            "episode_steps": self._episode_steps,
            "last_action": self._last_action,
            "last_moved": self._last_moved,
            "last_reward": self._last_reward_step,
            "trail": list(self._agent_trail),
        }
