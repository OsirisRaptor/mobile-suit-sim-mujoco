from dataclasses import dataclass
from typing import Sequence

@dataclass
class ThrusterConfig:
    name: str
    body_name: str
    pos: Sequence[float]      # [x, y, z] in parent body frame
    dir_world: Sequence[float]  # force direction in world frame (unit-ish)
    max_force: float = 1000.0  # N
