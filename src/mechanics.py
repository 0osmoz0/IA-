"""Mécaniques de jeu sur grille (clé, porte, pièges, portail, mur mobile)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Set, Tuple


@dataclass
class MechanicPlacements:
    key_pos: Optional[Tuple[int, int]] = None
    door_pos: Optional[Tuple[int, int]] = None
    trap_cells: Set[Tuple[int, int]] = field(default_factory=set)
    portal_a: Optional[Tuple[int, int]] = None
    portal_b: Optional[Tuple[int, int]] = None
    mobile_wall: Optional[Tuple[int, int]] = None
    mobile_period: int = 8
