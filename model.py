"""Simulation entities, immutable scenarios, and local UAV auction state."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


Coordinate = Tuple[int, int]
TASK_TYPES = ("I", "M", "S", "N")


@dataclass(frozen=True)
class Task:
    task_id: int
    coordinate: Coordinate
    task_type: str
    original_coordinate: Optional[Tuple[float, float]] = None
    source_record_id: Optional[str] = None


@dataclass(frozen=True)
class Scenario:
    seed: int
    tasks: tuple[Task, ...]
    uav_start_positions: tuple[Coordinate, ...]


@dataclass
class UAVState:
    uav_id: int
    coordinate: Coordinate
    remaining_energy: float


@dataclass
class LocalAuctionState:
    """Winner and bid records maintained independently by one UAV."""

    uav_id: int
    bid_table: Dict[int, float]
    winner_table: Dict[int, Optional[int]]
