"""PMData loader (spec section 2). Reads pXX/fitbit/resting_heart_rate.json (resting HR)
and pXX/fitbit/sleep.json (sleep efficiency; minutes asleep). Values are read as recorded; nothing is filled in."""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd

from engine import MIN_BEFORE
from subject import DEMO_METRIC, EFFICIENCY_METRIC, SLEEP_METRIC, Subject

RHR_PATH = os.path.join("fitbit", "resting_heart_rate.json")
SLEEP_PATH = os.path.join("fitbit", "sleep.json")


class SubjectUnusable(ValueError):
    """The subject's data can't support the requested start date or schedule."""


def list_pmdata(root: str) -> list[str]:
    """All participant folders (pXX) under root, sorted. A folder without a
    resting-HR file is still listed; load_pmdata raises the reason."""
    return sorted(d for d in os.listdir(root)
                  if re.fullmatch(r"p\d\d", d) and os.path.isdir(os.path.join(root, d)))


def read_raw(root: str, pid: str) -> list[dict]:
    path = os.path.join(root, pid, RHR_PATH)
    if not os.path.exists(path):
        raise SubjectUnusable(f"{pid}: no {RHR_PATH}")
    with open(path) as f:
        return json.load(f)


def raw_to_series(entries: list[dict]) -> pd.Series:
    """Missing days (value 0.0, error 0.0, date null) -> NaN; reindex to a complete
    daily range so absent dates are NaN too. No interpolation."""
    dates = pd.to_datetime([e["dateTime"] for e in entries]).normalize()
    vals = np.array([e["value"]["value"] for e in entries], dtype=float)
    vals[vals == 0.0] = np.nan
    s = pd.Series(vals, index=dates).sort_index()
    if s.index.has_duplicates:
        raise ValueError("duplicate dates in resting_heart_rate.json")
    full = pd.date_range(s.index[0], s.index[-1], freq="D")
    return s.reindex(full).astype(float)


def read_sleep(root: str, pid: str) -> pd.Series:
    """Minutes asleep per night from fitbit/sleep.json: main-sleep records only
    (naps excluded), summed by dateOfSleep. A night with no record is NaN; a
    record of 0 minutes is treated as not worn (NaN). No interpolation."""
    path = os.path.join(root, pid, SLEEP_PATH)
    if not os.path.exists(path):
        raise SubjectUnusable(f"{pid}: no {SLEEP_PATH}")
    with open(path) as f:
        entries = json.load(f)
    nights: dict[str, float] = {}
    for e in entries:
        if not e.get("mainSleep", True):
            continue
        d = e["dateOfSleep"]
        nights[d] = nights.get(d, 0.0) + float(e["minutesAsleep"])
    if not nights:
        raise SubjectUnusable(f"{pid}: no main-sleep records in {SLEEP_PATH}")
    s = pd.Series(nights)
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s[s == 0.0] = np.nan
    full = pd.date_range(s.index[0], s.index[-1], freq="D")
    return s.reindex(full).astype(float)


def read_efficiency(root: str, pid: str) -> pd.Series:
    """Sleep efficiency (%) per night from fitbit/sleep.json: main-sleep records of
    type "stages" only (Fitbit computes efficiency differently for "classic"
    records, so the two are not mixed). If a date has more than one, the longest
    record is used. A night with no such record is NaN. No interpolation."""
    path = os.path.join(root, pid, SLEEP_PATH)
    if not os.path.exists(path):
        raise SubjectUnusable(f"{pid}: no {SLEEP_PATH}")
    with open(path) as f:
        entries = json.load(f)
    best: dict[str, tuple[float, float]] = {}
    for e in entries:
        if not e.get("mainSleep", True) or e.get("type") != "stages":
            continue
        d = e["dateOfSleep"]
        if d not in best or e["timeInBed"] > best[d][0]:
            best[d] = (float(e["timeInBed"]), float(e["efficiency"]))
    if not best:
        raise SubjectUnusable(f"{pid}: no main-sleep stages records in {SLEEP_PATH}")
    s = pd.Series({d: v[1] for d, v in best.items()})
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    full = pd.date_range(s.index[0], s.index[-1], freq="D")
    return s.reindex(full).astype(float)


def read_rhr(root: str, pid: str) -> pd.Series:
    return raw_to_series(read_raw(root, pid))


# key -> (Metric, display label, reader). The engine is the same for every metric.
METRICS = {
    "resting_hr": (DEMO_METRIC, "Resting heart rate", read_rhr),
    "sleep_efficiency": (EFFICIENCY_METRIC, "Sleep efficiency", read_efficiency),
    # "minutes_asleep": (SLEEP_METRIC, "Minutes asleep", read_sleep),  # replaced: 2 SD of sleep duration is ~3.5 h
}


def load_pmdata(root: str, pid: str, start_day: int = 60, metric: str = "resting_hr") -> Subject:
    """start_day is an offset (days) from the first date in the file."""
    series = METRICS[metric][2](root, pid)
    series.name = pid
    n_days = len(series)
    if start_day >= n_days:
        raise SubjectUnusable(f"{pid}: series is {n_days} days; start day {start_day} leaves no days after start")
    n_before = int(series.iloc[:start_day].notna().sum())
    if n_before < MIN_BEFORE:
        raise SubjectUnusable(f"{pid}: {n_before} days with data before start day {start_day}; need at least {MIN_BEFORE}")
    start = series.index[0] + pd.Timedelta(days=start_day)
    return Subject(subject_id=pid, series=series, start=start)


def days_available(root: str, pid: str) -> int:
    """Calendar length of the subject's complete daily series."""
    return len(raw_to_series(read_raw(root, pid)))


def demo_subject_id(root: str, start_day: int = 60, trial_days: int = 84) -> str:
    """The participant with the most non-NaN days whose series supports
    start_day + trial_days. Ties broken by lowest participant ID."""
    best = None
    for pid in list_pmdata(root):
        try:
            s = raw_to_series(read_raw(root, pid))
        except SubjectUnusable:
            continue
        if len(s) < start_day + trial_days:
            continue
        key = (-int(s.notna().sum()), pid)
        if best is None or key < best:
            best = key
    if best is None:
        raise SubjectUnusable(f"no subject supports start day {start_day} + {trial_days} days")
    return best[1]
