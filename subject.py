"""Data contract (spec section 4)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Metric:
    name: str                   # "resting_hr"
    unit: str                   # "bpm"
    works_direction: str        # "up" | "down": the change that means the compound works
    harm_direction: str         # "up" | "down": the change that means harm
    onset_days: int             # days for the effect to start/stop; sets washout

    def __post_init__(self):
        for d in (self.works_direction, self.harm_direction):
            if d not in ("up", "down"):
                raise ValueError(f"direction must be 'up' or 'down', got {d!r}")


@dataclass(frozen=True)
class Subject:
    subject_id: str
    series: pd.Series           # float, complete daily DatetimeIndex, NaN = missing
    start: pd.Timestamp         # first day on compound; earlier days = before
    simulated: bool = False

    def __post_init__(self):
        idx = self.series.index
        if not isinstance(idx, pd.DatetimeIndex):
            raise TypeError("series must have a DatetimeIndex")
        if len(idx) > 1:
            full = pd.date_range(idx[0], idx[-1], freq="D")
            if len(full) != len(idx) or not (full == idx).all():
                raise ValueError("series index must be a complete daily range (missing days are NaN rows)")


@dataclass(frozen=True)
class Schedule:
    start: pd.Timestamp
    block_days: int             # 7
    washout_days: int           # = metric.onset_days, excluded at the start of every block
    cycles: list[tuple[str, str]]   # e.g. [("on","off"), ("off","on"), ...]


# Demo metric: a recovery-type compound. Working = resting HR falls; harm = it rises.
DEMO_METRIC = Metric("resting_hr", "bpm", works_direction="down", harm_direction="up", onset_days=2)

# Second metric: sleep. Harm = less sleep than this person's normal; working = more.
SLEEP_METRIC = Metric("minutes_asleep", "min", works_direction="up", harm_direction="down", onset_days=2)

# Sleep efficiency: % of time in bed spent asleep (Fitbit stages nights). Harm = falls.
EFFICIENCY_METRIC = Metric("sleep_efficiency", "%", works_direction="up", harm_direction="down", onset_days=2)
