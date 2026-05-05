"""Génération procédurale de labyrinthes (recursive backtracker)."""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np


def generate_maze(height: int, width: int, rng: random.Random | None = None) -> np.ndarray:
    """Retourne une grille: 1 = mur, 0 = couloir. height et width doivent être impairs >= 7."""
    rng = rng or random.Random()
    if height % 2 == 0:
        height += 1
    if width % 2 == 0:
        width += 1
    height = max(7, height)
    width = max(7, width)

    grid = np.ones((height, width), dtype=np.int8)

    stack: list[Tuple[int, int]] = []
    sy, sx = 1, 1
    grid[sy, sx] = 0
    stack.append((sy, sx))
    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]

    while stack:
        cy, cx = stack[-1]
        rng.shuffle(directions)
        moved = False
        for dy, dx in directions:
            ny, nx = cy + dy, cx + dx
            if 0 < ny < height - 1 and 0 < nx < width - 1 and grid[ny, nx] == 1:
                grid[cy + dy // 2, cx + dx // 2] = 0
                grid[ny, nx] = 0
                stack.append((ny, nx))
                moved = True
                break
        if not moved:
            stack.pop()

    return grid


def farthest_cell_from(grid: np.ndarray, start_y: int, start_x: int) -> Tuple[int, int]:
    """BFS pour placer la sortie au plus loin du départ (dans les couloirs)."""
    h, w = grid.shape
    from collections import deque

    q = deque([(start_y, start_x)])
    dist = { (start_y, start_x): 0 }
    best = (start_y, start_x)
    best_d = 0
    while q:
        y, x = q.popleft()
        d = dist[(y, x)]
        if d > best_d and grid[y, x] == 0:
            best_d = d
            best = (y, x)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] == 0 and (ny, nx) not in dist:
                dist[(ny, nx)] = d + 1
                q.append((ny, nx))
    return best
