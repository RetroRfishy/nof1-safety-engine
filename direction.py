"""Which way a signal has to move to count as harm: 'up', 'down' or 'either'.

'up'/'down' use engine.harm_monitor unchanged (one-sided CUSUM). 'either' runs both one-sided
CUSUMs on the same residuals and alarms when the larger of the two crosses ONE line, set from
other people's compound-free windows so that max(upper peak, lower peak) exceeds it FP of the
time. Measured rates for all three settings: python twosided.py -> twosided_results.json."""
from __future__ import annotations

import json
import os
from dataclasses import replace

import numpy as np
import pandas as pd

import engine as E

DIRECTIONS = ("up", "down", "either")
LABEL = {"up": "Rises", "down": "Falls", "either": "Either way"}
HERE = os.path.dirname(os.path.abspath(__file__))


def with_direction(metric, d: str):
    return replace(metric, harm_direction=d) if d in ("up", "down") else metric


def two_sided_limit(subject_id: str, lib, n_valid: int, k: float, fp: float = E.FP) -> float:
    used = [w.z[:n_valid] for w in lib.windows if w.subject_id != subject_id and len(w.z) >= n_valid]
    if not used or n_valid < 1:
        return np.inf
    Z = np.stack(used)
    m = np.maximum(E._cusum_max_batch(Z, k, 1), E._cusum_max_batch(Z, k, -1))
    return float(np.percentile(m, 100 * (1 - fp)))


def monitor(subject, base, lib, metric, direction: str, method: str = "reference"):
    """Returns (HarmResult, side) where side is 'up'/'down' for an alarm, else None."""
    if direction in ("up", "down"):
        r = E.harm_monitor(subject, base, lib, with_direction(metric, direction), method=method)
        return r, (direction if r.alarm_date is not None else None)
    z_all = E.standardised_residuals(subject, base)
    after = z_all[z_all.index >= subject.start]
    win = after.iloc[:min(E.HORIZON, len(after))]
    n_valid = int(win.notna().sum())
    k = E.cusum_k(base.phi)
    h = two_sided_limit(subject.subject_id, lib, n_valid, k)
    Su, Sd = E.cusum(win.to_numpy(), k, 1), E.cusum(win.to_numpy(), k, -1)
    S = pd.Series(np.maximum(Su, Sd), index=win.index)
    over = S[S > h]
    alarm = over.index[0] if len(over) else None
    side, record = None, None
    if alarm is not None:
        i = win.index.get_loc(alarm)
        side = "up" if Su[i] > h else "down"
        record = {"subject": subject.subject_id, "date": alarm.strftime("%Y-%m-%d"), "metric": metric.name,
                  "h": round(h, 4), "n_valid": n_valid, "FP": E.FP, "simulated": bool(subject.simulated),
                  "direction": "either", "moved": side}
    return E.HarmResult(S, win, h, k, "reference", n_valid, len(win), alarm, record), side


def measured(metric_key: str) -> dict | None:
    """{'alert_rate': {dir: rate}, 'catch_2sd': {dir: rate}, 'median_day': {dir: day}, 'n': windows}"""
    try:
        with open(os.path.join(HERE, "twosided_results.json")) as f:
            return json.load(f).get(metric_key)
    except (OSError, ValueError):
        return None
