"""Combined harm score across signals, for a shift in a known direction.

Each signal is scored exactly as in engine.py (AR(1) residual against the person's own
60-day normal, standardised). On each day the harm-direction scores are added and
divided by their standard deviation, so the sum is again one 'surprise' per day:

    u_t = (s_1 z_1,t + s_2 z_2,t) / sqrt(2 + 2 rho)      (both present)
    u_t = s_i z_i,t                                       (only one present)

where s_i = +1 if harm is 'up', -1 if 'down', and rho is the correlation of the two
harm-direction scores in the person's own baseline. A one-sided CUSUM runs on u.
For a shift in a known direction this linear combination is the standard detector
(Healey 1987, Technometrics 29:409-412). The alarm line is set exactly as for one
signal: the 95th percentile of CUSUM maxima over other people's real compound-free
windows (each with its own refitted baselines), leaving the monitored person out.

Run: python combined.py [--root PATH]  -> prints the validation and writes combined_results.json
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

import engine as E
import loaders as L
from checks import BASELINE_DAYS, DEFAULT_ROOT, MW_OFFSETS, wilson
from subject import Subject

KEYS = ("resting_hr", "sleep_efficiency")
TARGET_SD = 1.5          # k is tuned for a joint shift of 1.5 own-SDs in every signal
HERE = os.path.dirname(os.path.abspath(__file__))


def _sign(metric) -> int:
    return 1 if metric.harm_direction == "up" else -1


@dataclass
class Fitted:
    z: dict            # key -> pd.Series of standardised residuals over the whole series
    base: dict         # key -> engine.Baseline
    rho: float
    start: pd.Timestamp


def fit_person(series: dict, start: pd.Timestamp) -> Fitted:
    """series: key -> full daily series. Baselines from the BASELINE_DAYS before start."""
    z, base = {}, {}
    for k, s in series.items():
        sub = E.rebase(Subject("x", s, start), start, BASELINE_DAYS)
        b = E.fit_baseline(sub)                      # raises ValueError if unusable
        base[k] = b
        z[k] = E.standardised_residuals(sub, b)
    before = pd.concat({k: _sign(L.METRICS[k][0]) * z[k][z[k].index < start] for k in z}, axis=1).dropna()
    rho = float(np.corrcoef(before.to_numpy().T)[0, 1]) if len(before) > 5 else 0.0
    rho = float(np.clip(np.nan_to_num(rho), -0.9, 0.9))
    return Fitted(z, base, rho, start)


WEIGHTING = os.environ.get("COMBINED_WEIGHTING", "optimal")   # "equal" or "optimal"


def weights(f: Fitted) -> np.ndarray:
    """Healey (1987): w proportional to Sigma^-1 m, where m is the expected daily residual shift
    in each signal for the same shift in own-SDs. 'equal' gives each signal the same weight."""
    keys = list(f.z)
    if WEIGHTING == "equal":
        return np.ones(len(keys))
    m = np.array([f.base[k].sd_x * (1 - f.base[k].phi) / f.base[k].sd_e for k in keys])
    Sigma = np.array([[1.0, f.rho], [f.rho, 1.0]])
    return np.linalg.solve(Sigma, m)


def combined_u(f: Fitted, horizon: int = E.HORIZON) -> pd.Series:
    """Daily combined harm score over the monitoring course (NaN when no signal is present)."""
    idx = pd.date_range(f.start, periods=horizon, freq="D")
    a = pd.concat({k: _sign(L.METRICS[k][0]) * f.z[k].reindex(idx) for k in f.z}, axis=1)
    w = weights(f)
    Sigma = np.array([[1.0, f.rho], [f.rho, 1.0]])
    both = a.notna().all(axis=1)
    u = a.sum(axis=1, min_count=1)                   # one present -> that one alone; none -> NaN
    u[both] = (a[both].to_numpy() @ w) / np.sqrt(w @ Sigma @ w)
    return u


def k_for(f: Fitted, target_sd: float = TARGET_SD) -> float:
    """Half the expected daily u under a joint shift of target_sd own-SDs in every signal."""
    m = np.array([target_sd * b.sd_x * (1 - b.phi) / b.sd_e for b in f.base.values()])
    w = weights(f)
    Sigma = np.array([[1.0, f.rho], [f.rho, 1.0]])
    return 0.5 * float(w @ m) / float(np.sqrt(w @ Sigma @ w))


def load_series(root: str, pid: str) -> dict:
    return {k: L.METRICS[k][2](root, pid) for k in KEYS}


def anchor_start(root: str, pid: str, offset: int) -> pd.Timestamp:
    s = L.METRICS["resting_hr"][2](root, pid)
    return s.index[0] + pd.Timedelta(days=offset)


def library(root: str, pids: list, first_offset: int = 60, horizon: int = E.HORIZON):
    """Other people's compound-free windows: one window per valid day after first_offset,
    baselines refitted for every window, as in engine.reference_library."""
    out = []
    for pid in pids:
        ser = load_series(root, pid)
        s0 = ser["resting_hr"].index[0]
        last = min(s.index[-1] for s in ser.values())
        d = s0 + pd.Timedelta(days=first_offset)
        while d + pd.Timedelta(days=horizon - 1) <= last:
            try:
                f = fit_person(ser, d)
                u = combined_u(f, horizon).dropna().to_numpy()
                if len(u):
                    out.append((pid, u, f))
            except ValueError:
                pass
            d += pd.Timedelta(days=1)
    return out


def cusum_max(u: np.ndarray, k: float) -> float:
    S = M = 0.0
    for x in u:
        S = max(0.0, S + x - k)
        M = max(M, S)
    return M


def limit(lib, pid: str, n_valid: int, k: float, fp: float = E.FP) -> float:
    maxes = [cusum_max(u[:n_valid], k) for q, u, _ in lib if q != pid and len(u) >= n_valid]
    return float(np.percentile(maxes, 100 * (1 - fp))) if maxes else np.inf


def monitor(f: Fitted, lib, pid: str):
    u = combined_u(f)
    k = k_for(f)
    n_valid = int(u.notna().sum())
    h = limit(lib, pid, n_valid, k)
    S, out = 0.0, []
    for x in u:
        if not np.isnan(x):
            S = max(0.0, S + x - k)
        out.append(S)
    S = pd.Series(out, index=u.index)
    over = S[S > h]
    return {"S": S, "h": h, "k": k, "alarm": over.index[0] if len(over) else None, "u": u}


def plant(series: dict, start: pd.Timestamp, shift_sd: dict, offset: int = 5, ramp: int = 3) -> dict:
    """Sustained shift in each signal's harm direction from start+offset, size in that person's before-SD."""
    out = {}
    for k, s in series.items():
        d = shift_sd.get(k, 0.0)
        if not d:
            out[k] = s
            continue
        before = s[(s.index < start) & (s.index >= start - pd.Timedelta(days=BASELINE_DAYS))].dropna()
        sd = float(np.std(before, ddof=1))
        days_in = (s.index - (start + pd.Timedelta(days=offset))).days.to_numpy()
        frac = np.clip((days_in + 1) / ramp, 0, 1)
        frac[days_in < 0] = 0
        out[k] = s + _sign(L.METRICS[k][0]) * d * sd * frac
    return out


def single_alarm(series: dict, start: pd.Timestamp, key: str, libs: dict, pid: str) -> bool:
    met = L.METRICS[key][0]
    sub = E.rebase(Subject(pid, series[key], start), start, BASELINE_DAYS)
    b = E.fit_baseline(sub)
    return E.harm_monitor(sub, b, libs[key], met, method="reference").alarm_date is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    root = ap.parse_args().root
    pids = []
    for pid in L.list_pmdata(root):
        try:
            load_series(root, pid)
            L.load_pmdata(root, pid, 60, "resting_hr")
            pids.append(pid)
        except (L.SubjectUnusable, ValueError):
            pass
    print(f"people with both signals: {len(pids)} {pids}")
    lib = library(root, pids)
    print(f"combined library: {len(lib)} windows")
    libs = {}
    for k in KEYS:
        subs = []
        for pid in L.list_pmdata(root):
            try:
                subs.append(L.load_pmdata(root, pid, 60, k))
            except (L.SubjectUnusable, ValueError):
                pass
        libs[k] = E.reference_library(subs)

    windows = []
    for pid in pids:
        ser = load_series(root, pid)
        last = min(s.index[-1] for s in ser.values())
        for off in MW_OFFSETS:
            st = anchor_start(root, pid, off)
            if st + pd.Timedelta(days=E.HORIZON - 1) > last:
                continue
            try:
                fit_person(ser, st)
                for k in KEYS:
                    E.fit_baseline(E.rebase(Subject(pid, ser[k], st), st, BASELINE_DAYS))
            except ValueError:
                continue
            windows.append((pid, off, st, ser))
    print(f"test windows: {len(windows)}")

    def evaluate(shift):
        res = {"combined": [], "rhr": [], "eff": [], "or": [], "and": []}
        for pid, off, st, ser in windows:
            s2 = plant(ser, st, shift) if shift else ser
            c = monitor(fit_person(s2, st), lib, pid)["alarm"] is not None
            r = single_alarm(s2, st, "resting_hr", libs, pid)
            e = single_alarm(s2, st, "sleep_efficiency", libs, pid)
            for name, v in (("combined", c), ("rhr", r), ("eff", e), ("or", r or e), ("and", r and e)):
                res[name].append(v)
        return {n: float(np.mean(v)) for n, v in res.items()}, {n: int(sum(v)) for n, v in res.items()}

    out = {"n_windows": len(windows), "people": pids, "target_sd": TARGET_SD, "weighting": WEIGHTING}
    rates, counts = evaluate(None)
    out["false_alarms"] = {n: {"rate": rates[n], "alarmed": counts[n], "ci": wilson(counts[n], len(windows))}
                           for n in rates}
    print("\nFALSE ALARMS (nothing planted):")
    for n in rates:
        lo, hi = wilson(counts[n], len(windows))
        print(f"  {n:9s} {counts[n]}/{len(windows)} = {rates[n]:.1%}  (95% {lo:.0%}-{hi:.0%})")
    out["detection"] = {}
    for label, shift in (("both 0.5 SD", {"resting_hr": 0.5, "sleep_efficiency": 0.5}),
                         ("both 1 SD", {"resting_hr": 1.0, "sleep_efficiency": 1.0}),
                         ("both 1.5 SD", {"resting_hr": 1.5, "sleep_efficiency": 1.5}),
                         ("both 2 SD", {"resting_hr": 2.0, "sleep_efficiency": 2.0}),
                         ("sleep only 2 SD", {"sleep_efficiency": 2.0}),
                         ("HR only 2 SD", {"resting_hr": 2.0})):
        rates, counts = evaluate(shift)
        out["detection"][label] = rates
        print(f"\nCAUGHT within 30 days, {label}: " +
              "  ".join(f"{n} {rates[n]:.0%}" for n in rates))
    ws = [weights(fit_person(ser, st)) for pid, off, st, ser in windows]
    ws = np.array([w / np.abs(w).sum() for w in ws])
    out["median_weight_share"] = {k: float(np.median(ws[:, i])) for i, k in enumerate(KEYS)}
    print("median weight share:", out["median_weight_share"])
    with open(os.path.join(HERE, f"combined_results_{WEIGHTING}.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\nwrote combined_results.json")


if __name__ == "__main__":
    main()
