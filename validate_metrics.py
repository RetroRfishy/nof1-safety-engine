"""Run the false-alarm check (check 6) and the 2 SD detection check (check 7)
for every metric in loaders.METRICS, on real PMData, and write
metric_results.json for the app's Results page.

Same procedure as checks.py: window starts at days 60/74/88/102, a fixed 60-day
baseline before each start, 30 days monitored, reference limit built leaving
the monitored person out. Nothing is planted in the false-alarm run.
Run: python validate_metrics.py [--root PATH]"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import engine as E
import loaders as L
from checks import BASELINE_DAYS, DEFAULT_ROOT, MW_OFFSETS, START_DAY, wilson

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "metric_results.json")
METHODS = ("reference", "own_baseline")
SHIFTS = (0.5, 1.0, 1.5, 2.0, 3.0)   # planted shift sizes, in before-period SDs


def usable(root, key):
    out, skipped = [], []
    for pid in L.list_pmdata(root):
        try:
            out.append(L.load_pmdata(root, pid, START_DAY, key))
        except (L.SubjectUnusable, ValueError) as err:
            skipped.append(str(err))
    return out, skipped


def null_windows(subjects):
    out = []
    for subj in subjects:
        s0 = subj.series.index[0]
        for off in MW_OFFSETS:
            if off + E.HORIZON > len(subj.series):
                continue
            sub = E.rebase(subj, s0 + pd.Timedelta(days=off), BASELINE_DAYS)
            try:
                base = E.fit_baseline(sub)
            except ValueError:
                continue
            out.append((off, sub, base))
    return out


def run(root, key):
    metric, label, _ = L.METRICS[key]
    subjects, skipped = usable(root, key)
    lib = E.reference_library(subjects)
    wins = null_windows(subjects)
    res = {"label": label, "unit": metric.unit, "harm_direction": metric.harm_direction,
           "n_subjects": len({w[1].subject_id for w in wins}), "n_windows": len(wins),
           "library_windows": len(lib.windows), "skipped": skipped,
           "median_sd": float(np.median([b.sd_x for _, _, b in wins])),
           "median_phi": float(np.median([b.phi for _, _, b in wins])),
           "tail_3sd": float(np.mean(np.concatenate([np.abs(b.z) > 3 for _, _, b in wins]))),
           "false_alarms": {}, "detection": {}, "windows": [], "unplanted_alarms": []}
    for method in METHODS:
        alarms, days = [], []
        for off, sub, base in wins:
            r = E.harm_monitor(sub, base, lib, metric, method=method, seed=0)
            a = r.alarm_date is not None
            alarms.append(a)
            if method == "reference":
                res["windows"].append({"subject": sub.subject_id, "offset": off, "start": str(sub.start.date()),
                                       "h": r.h, "peak": float(r.S.max()), "alarm": a})
                if a:
                    res["unplanted_alarms"].append({"subject": sub.subject_id, "start": str(sub.start.date()),
                                                    "alarm": str(r.alarm_date.date()), "offset": off})
            p = E.harm_monitor(E.plant_harm(sub, metric, delta_sd=2.0), base, lib, metric, method=method, seed=0)
            days.append((p.alarm_date - sub.start).days if p.alarm_date is not None else None)
        n, k = len(alarms), int(sum(alarms))
        lo, hi = wilson(k, n)
        res["false_alarms"][method] = {"alarmed": k, "windows": n, "rate": k / n, "ci": [lo, hi],
                                       "subjects_alarmed": sorted({w[1].subject_id for w, a in zip(wins, alarms) if a})}
        hit = [d for d in days if d is not None]
        q = np.percentile(hit, [25, 50, 75]) if hit else [np.nan] * 3
        res["detection"][method] = {"caught": len(hit), "windows": len(days), "rate": len(hit) / len(days),
                                    "median_day": float(q[1]), "iqr": [float(q[0]), float(q[2])]}
        fa = res["false_alarms"][method]; de = res["detection"][method]
        print(f"  {label:20s} {method:12s} false alarms {k}/{n} = {k/n:.1%} (Wilson {lo:.0%}-{hi:.0%}); "
              f"2 SD caught {de['caught']}/{de['windows']} = {de['rate']:.0%}, median day {de['median_day']:.0f}")
    # detection by planted shift size, reference limit (the default)
    res["by_shift"] = []
    for d_sd in SHIFTS:
        days = []
        for off, sub, base in wins:
            p = E.harm_monitor(E.plant_harm(sub, metric, delta_sd=d_sd), base, lib, metric, method="reference")
            days.append((p.alarm_date - sub.start).days if p.alarm_date is not None else None)
        hit = [d for d in days if d is not None]
        res["by_shift"].append({"shift_sd": d_sd, "shift_units": d_sd * res["median_sd"],
                                "rate": len(hit) / len(days),
                                "median_day": float(np.median(hit)) if hit else None})
    print("  by shift:", [(b["shift_sd"], f"{b['rate']:.0%}") for b in res["by_shift"]])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    a = ap.parse_args()
    out = {"procedure": {"starts": MW_OFFSETS, "baseline_days": BASELINE_DAYS, "horizon": E.HORIZON,
                         "target_fp": E.FP, "planted_shift_sd": 2.0}}
    for key in L.METRICS:
        out[key] = run(a.root, key)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
