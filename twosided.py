"""Two-sided harm monitor (alert if the signal moves too far EITHER way) vs the one-sided default.
Same engine pieces: two one-sided CUSUMs on the person's own residuals; one alarm line set from other
people's compound-free windows so that max(upper peak, lower peak) exceeds it 5% of the time.
Run: python twosided.py [--root PATH]"""
from __future__ import annotations
import argparse, json, os
import numpy as np, pandas as pd
import engine as E, loaders as L
from checks import DEFAULT_ROOT, wilson
from validate_metrics import usable, null_windows

def peaks(z, k):
    z = z[~np.isnan(z)]
    up = E._cusum_max_batch(z[None, :], k, 1)[0] if len(z) else 0.0
    dn = E._cusum_max_batch(z[None, :], k, -1)[0] if len(z) else 0.0
    return up, dn

def two_limit(lib, pid, n_valid, k):
    m = [max(peaks(w.z[:n_valid], k)) for w in lib.windows if w.subject_id != pid and len(w.z) >= n_valid]
    return float(np.percentile(m, 95)) if m else np.inf

def monitor_two(sub, base, lib):
    z_all = E.standardised_residuals(sub, base)
    win = z_all[z_all.index >= sub.start].iloc[:E.HORIZON]
    n_valid = int(win.notna().sum()); k = E.cusum_k(base.phi)
    h = two_limit(lib, sub.subject_id, n_valid, k)
    Su = pd.Series(E.cusum(win.to_numpy(), k, 1), index=win.index)
    Sd = pd.Series(E.cusum(win.to_numpy(), k, -1), index=win.index)
    over = win.index[(Su > h) | (Sd > h)]
    return over[0] if len(over) else None

def flip(metric):
    return E.replace(metric, harm_direction="down" if metric.harm_direction == "up" else "up") if hasattr(E, "replace") else None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=DEFAULT_ROOT); root = ap.parse_args().root
    from dataclasses import replace
    out = {}
    for key in L.METRICS:
        met = L.METRICS[key][0]; opp = replace(met, harm_direction="down" if met.harm_direction == "up" else "up")
        subs, _ = usable(root, key); lib = E.reference_library(subs); wins = null_windows(subs)
        res = {"one_null": [], "two_null": [], "one_harm": [], "two_harm": [], "one_opp": [], "two_opp": [],
               "one_day": [], "two_day": [], "flip_null": [], "flip_catch": [], "flip_day": []}
        for off, sub, base in wins:
            r1 = E.harm_monitor(sub, base, lib, met); res["one_null"].append(r1.alarm_date is not None)
            res["two_null"].append(monitor_two(sub, base, lib) is not None)
            p = E.plant_harm(sub, met, delta_sd=2.0)
            a1 = E.harm_monitor(p, base, lib, met).alarm_date; a2 = monitor_two(p, base, lib)
            res["one_harm"].append(a1 is not None); res["two_harm"].append(a2 is not None)
            if a1 is not None: res["one_day"].append((a1 - sub.start).days)
            if a2 is not None: res["two_day"].append((a2 - sub.start).days)
            q = E.plant_harm(sub, opp, delta_sd=2.0)          # same size shift, the OTHER direction
            res["one_opp"].append(E.harm_monitor(q, base, lib, met).alarm_date is not None)
            res["two_opp"].append(monitor_two(q, base, lib) is not None)
            res["flip_null"].append(E.harm_monitor(sub, base, lib, opp).alarm_date is not None)   # watching the other side only
            fa = E.harm_monitor(q, base, lib, opp).alarm_date
            res["flip_catch"].append(fa is not None)
            if fa is not None: res["flip_day"].append((fa - sub.start).days)
        n = len(wins)
        row = {k: float(np.mean(v)) for k, v in res.items() if not k.endswith("_day")}
        row.update({"one_median_day": float(np.median(res["one_day"])), "two_median_day": float(np.median(res["two_day"])),
                    "flip_median_day": float(np.median(res["flip_day"])) if res["flip_day"] else None,
                    "n": n, "one_null_k": int(sum(res["one_null"])), "two_null_k": int(sum(res["two_null"]))})
        rate = {met.harm_direction: row["one_null"], opp.harm_direction: row["flip_null"], "either": row["two_null"]}
        catch = {met.harm_direction: row["one_harm"], opp.harm_direction: row["flip_catch"], "either": row["two_harm"]}
        row["alert_rate"] = rate; row["catch_2sd"] = catch
        row["median_day"] = {met.harm_direction: row["one_median_day"], opp.harm_direction: row["flip_median_day"],
                             "either": row["two_median_day"]}
        out[key] = row
        print(f"{L.METRICS[key][1]} ({n} windows)")
        print(f"  alerts, nothing given : one-sided {row['one_null_k']}/{n} = {row['one_null']:.1%}   two-sided {row['two_null_k']}/{n} = {row['two_null']:.1%}")
        print(f"  2 SD in harm direction: one-sided {row['one_harm']:.0%} (median day {row['one_median_day']:.0f})   two-sided {row['two_harm']:.0%} (median day {row['two_median_day']:.0f})")
        print(f"  2 SD the OTHER way    : one-sided {row['one_opp']:.0%}   two-sided {row['two_opp']:.0%}")
        print(f"  watching the other side only: alerts {row['flip_null']:.1%}, catches a 2 SD shift that way {row['flip_catch']:.0%}")
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "twosided_results.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
