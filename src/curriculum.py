"""Progression de difficulté (curriculum)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

MAX_LEVEL = 4  # A=0 .. E=4


@dataclass
class CurriculumConfig:
    successes_to_advance: int = 3
    eval_episodes_window: int = 20

    def threshold_for_level(self, level: int) -> float:
        """Récompense moyenne minimale (approx.) pour considérer le niveau maîtrisé; optionnel."""
        return -0.2 * (level + 1)


def level_layout(level: int) -> Dict[str, Any]:
    """Paramètres de génération par niveau (A–E)."""
    base: Dict[str, Any] = {
        "maze_h": 9,
        "maze_w": 9,
        "use_key_door": False,
        "use_traps": False,
        "use_portal": False,
        "use_mobile_wall": False,
    }
    if level >= 1:
        base["maze_h"] = 13
        base["maze_w"] = 13
        base["use_mobile_wall"] = True
    if level >= 2:
        base["maze_h"] = 15
        base["maze_w"] = 15
        base["use_key_door"] = True
    if level >= 3:
        base["maze_h"] = 17
        base["maze_w"] = 17
        base["use_traps"] = True
    if level >= 4:
        base["maze_h"] = 19
        base["maze_w"] = 19
        base["use_portal"] = True
    return base


class CurriculumManager:
    def __init__(self, cfg: CurriculumConfig | None = None):
        self.cfg = cfg or CurriculumConfig()
        self.level = 0
        self.consecutive_successes = 0
        self.last_episode_success = False

    def register_episode(self, success: bool, mean_reward_recent: float | None = None) -> None:
        self.last_episode_success = success
        if success:
            self.consecutive_successes += 1
        else:
            self.consecutive_successes = 0

        thr = self.cfg.threshold_for_level(self.level)
        if mean_reward_recent is not None and mean_reward_recent < thr - 1.0:
            self.consecutive_successes = 0

    def maybe_advance(self) -> bool:
        if self.level >= MAX_LEVEL:
            return False
        if self.consecutive_successes >= self.cfg.successes_to_advance:
            self.level += 1
            self.consecutive_successes = 0
            return True
        return False

    def reset_progression(self) -> None:
        self.level = 0
        self.consecutive_successes = 0
