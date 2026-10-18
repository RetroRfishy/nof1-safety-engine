"""Replay one person being monitored live, against a reference cohort on disk.

This is the deployment architecture in miniature. The reference library is built
once from a cohort that took nothing (PMData on disk). The person being watched
arrives a day at a time from the WHOOP simulator, and every morning the whole
monitoring window is rescored from scratch.

    # terminal 1
    python whoop_sim.py --root data/pmdata --subject p08 --port 8787
    # terminal 2
    python live_monitor.py --root data/pmdata --subject p08 --dose-day 60 --sleep 0.3

Rescoring from scratch every day is not laziness. The CUSUM is path-dependent,
so when the feed restates a value that was already scored, the only correct
response is to recompute from that day forward. This script makes the
consequence visible: an alarm raised on one day can be withdrawn on the next,
because the evidence that raised it was revised. Nothing in the retrospective
build can show that, and a real deployment has to have a policy for it.

THE THRESHOLD IS FIXED AT ARMING, and this is the second thing the live path
gets wrong if you are not careful. The obvious implementation recalibrates h
each morning for the number of steps taken so far, which is valid at each
instant and wrong over the course: thirty daily tests at 5% is not a 5% course,
it is engine.py's peeking argument reintroduced by the loop. So h is computed
once, when monitoring is armed, for the number of steps the course is EXPECTED
to supply -- horizon times this person's coverage squared, since a residual
needs two consecutive present days -- and then held constant. It is the same
discipline as calibrating before exposure: the rule is fixed before the data
that it judges arrives.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from engine import (FP, HORIZON, MIN_BEFORE, _sign, cusum, cusum_k, fit_baseline,
                    reference_library, reference_limit, standardised_residuals)
from loaders import SubjectUnusable, list_pmdata, load_pmdata
from subject import DEMO_METRIC
from whoop_client import clock, fetch

# Measured on PMData; see README. Shown in the header so the screen states what
# the detector can and cannot do before it says anything about this person.
MEASURED_FP = "7% per 30-day course"
MEASURED_DETECT = "a 2 SD sustained shift about half the time, typically by day 13"

BAR_W = 22
SPARK = "▁▂▃▄▅▆▇█"
COLOR = sys.stdout.isatty()


def c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if COLOR else text


def rule(width: int = 78) -> str:
    return c("─" * width, "2")


def sparkline(vals: np.ndarray, lo: float, hi: float) -> str:
    """Fixed scale, not auto-scaled: anchored to the person's own normal band so
    a quiet week looks quiet instead of being stretched to fill the glyphs."""
    out = []
    for v in vals:
        if np.isnan(v):
            out.append(c("·", "2"))
            continue
        f = (v - lo) / (hi - lo) if hi > lo else 0.5
        out.append(SPARK[int(np.clip(f, 0, 0.999) * len(SPARK))])
    return "".join(out)


def evidence_bar(S: float, h: float) -> str:
    """How far the accumulated evidence has climbed toward the alarm limit."""
    if not np.isfinite(h) or h <= 0:
        return c("·" * BAR_W, "2")
    frac = min(1.0, max(0.0, S / h))
    filled = int(round(frac * BAR_W))
    code = "31" if frac >= 0.85 else "33" if frac >= 0.5 else "32"
    return c("█" * filled, code) + c("░" * (BAR_W - filled), "2")


def header(subject: str, compound: str, metric, n_sub: int, n_win: int,
           h: float, n_expected: int, coverage: float, mu: float, sd: float):
    print()
    print(rule())
    print(f"  {c('COMPOUND SAFETY MONITOR', '1')}      {compound} · "
          f"{metric.name.replace('_', ' ')} ({metric.unit})")
    print(rule())
    print(f"  {'subject':<18}{subject}   harm = {metric.harm_direction}ward movement")
    print(f"  {'their normal':<18}{mu:.1f} ± {sd:.1f} {metric.unit}   "
          f"coverage {coverage:.0%}")
    print(f"  {'reference cohort':<18}{n_sub} people, {n_win} windows, none exposed")
    print(f"  {'alarm limit':<18}h = {h:.2f}   fixed at arming for ~{n_expected} "
          f"scorable days")
    print(f"  {'false alarms':<18}{MEASURED_FP} (measured, not nominal)")
    print(f"  {'detects':<18}{MEASURED_DETECT}")
    print(rule())
    print()


def build_reference(root: str, exclude: str, start_day: int):
    """The cohort that took nothing. Built once, before anyone is monitored."""
    others = []
    for pid in list_pmdata(root):
        if pid == exclude:
            continue
        try:
            others.append(load_pmdata(root, pid, start_day))
        except (SubjectUnusable, ValueError):
            continue
    return reference_library(others)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="PMData root, for the reference cohort")
    ap.add_argument("--subject", required=True, help="the person being monitored")
    ap.add_argument("--base", default="http://127.0.0.1:8787")
    ap.add_argument("--compound", default="Compound A")
    ap.add_argument("--dose-day", type=int, default=60, help="days from series start")
    ap.add_argument("--days", type=int, default=HORIZON + 5, help="days to replay")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between days")
    ap.add_argument("--json", default=None,
                    help="also write the replay to this file, for the HTML view")
    args = ap.parse_args()
    capture = {"days": [], "revisions": []} if args.json else None

    metric = DEMO_METRIC
    sign = _sign(metric.harm_direction)

    print(c(f"\n  building reference library from {args.root}, excluding "
            f"{args.subject} ...", "2"))
    lib = build_reference(args.root, args.subject, args.dose_day)
    print(c(f"  {len(lib.windows)} windows from {len(lib.subject_ids)} subjects", "2"))

    state = clock(args.base, day=args.dose_day - 1)
    first = pd.Timestamp(state["now"]) - pd.Timedelta(days=args.dose_day - 1)
    dose_date = first + pd.Timedelta(days=args.dose_day)

    prev_snap = prev_alarm = prev_last = None
    armed = None
    alarms_withdrawn = 0
    printed_header = False

    for i in range(args.days):
        state = clock(args.base, advance=1)
        today = pd.Timestamp(state["now"])
        snap = fetch(args.base)

        revisions = snap.revisions_since(prev_snap)
        last = snap.values.index[-1] if len(snap.values) else None
        prev_last = last

        try:
            subj = snap.to_subject(dose_date, subject_id=args.subject)
            base = fit_baseline(subj)
        except Exception:                                   # noqa: BLE001
            n_have = int((snap.values.index < dose_date).sum())
            print(c(f"  {str(today.date())}  accumulating baseline "
                    f"({n_have}/{MIN_BEFORE} days)", "2"))
            prev_snap = snap
            continue

        k = cusum_k(base.phi)

        # Arm once: fix h for the number of steps the course is EXPECTED to
        # supply, then never recompute it. Recalibrating daily would be a fresh
        # test every morning and inflate the course false-alarm rate.
        if armed is None:
            n_expected = max(1, int(round(HORIZON * base.coverage ** 2)))
            ref0 = reference_limit(subj, lib, n_expected, k, metric)
            armed = (ref0.h, n_expected)
            header(args.subject, args.compound, metric, ref0.n_subjects,
                   ref0.n_windows, ref0.h, n_expected, base.coverage,
                   base.mu, base.sd_x)
            if capture is not None:
                capture["meta"] = {
                    "subject": args.subject, "compound": args.compound,
                    "metric": metric.name, "unit": metric.unit,
                    "harm_direction": metric.harm_direction,
                    "mu": round(base.mu, 2), "sd": round(base.sd_x, 2),
                    "phi": round(base.phi, 3), "coverage": round(base.coverage, 3),
                    "h": round(ref0.h, 3), "k": round(k, 3),
                    "n_expected": n_expected, "horizon": HORIZON,
                    "ref_subjects": ref0.n_subjects, "ref_windows": ref0.n_windows,
                    "fp": FP, "dose_date": str(dose_date.date()),
                    "measured_fp": MEASURED_FP, "measured_detect": MEASURED_DETECT,
                }
            cols = (f"  {'day':>3}  {'reading':<11} {metric.unit:>6} {'z':>6}  "
                    f"{'last 8 days':<9}  {'evidence vs limit':<{BAR_W}}  "
                    f"{'S / h':>11}  status")
            print(c(cols, "1"))
            print(rule(len(cols) + 2))
            printed_header = True
        h, n_expected = armed

        # Rescore the whole window from scratch, every day, against that fixed h.
        z = standardised_residuals(subj, base)
        after = z[z.index >= dose_date]
        win = after.iloc[:min(HORIZON, len(after))]
        n_valid = int(win.notna().sum())
        S = pd.Series(cusum(win.to_numpy(), k, sign), index=win.index)
        over = S[S > h]
        alarm = over.index[0] if len(over) else None
        S_now = float(S.iloc[-1]) if len(S) else 0.0

        for d, old, new in revisions:
            tag = c("  → already scored, rescoring from here", "33") \
                if d >= dose_date else ""
            print(c(f"       ↻ {d.date()} restated {old:.1f} → {new:.1f} "
                    f"{metric.unit}", "2") + tag)

        withdrawn = alarm is None and prev_alarm is not None
        if alarm is not None and prev_alarm is None:
            status = c(f"ALARM  {alarm.date()}", "1;31")
        elif withdrawn:
            status = c("ALARM WITHDRAWN", "1;33")
            alarms_withdrawn += 1
        elif alarm is not None:
            status = c(f"alarm  {alarm.date()}", "31")
        else:
            status = c("watching", "32")

        today_val = snap.values.get(last) if last is not None else np.nan
        z_now = win.iloc[-1] if len(win) and not np.isnan(win.iloc[-1]) else np.nan
        recent = snap.values.reindex(
            pd.date_range(end=last, periods=8, freq="D")).to_numpy() \
            if last is not None else np.full(8, np.nan)

        val_txt = f"{today_val:6.1f}" if np.isfinite(today_val) else "     ·"
        z_txt = f"{z_now:+6.2f}" if np.isfinite(z_now) else "     ·"
        if np.isfinite(z_now) and sign * z_now > 1.0:
            z_txt = c(z_txt, "33")

        # The date shown is the latest reading the feed has delivered, not the
        # simulated "today": with a delivery lag they differ, and the row
        # describes the reading being scored.
        shown = str(last.date()) if last is not None else str(today.date())
        print(f"  {args.dose_day + i:>3}  {shown:<11} {val_txt} {z_txt}  "
              f"{sparkline(recent, base.mu - 2.5 * base.sd_x, base.mu + 2.5 * base.sd_x):<9}  "
              f"{evidence_bar(S_now, h)}  {S_now:5.2f} / {h:4.2f}  {status}")

        if withdrawn:
            print(c("       !! the alarm no longer holds: a restated value changed "
                    "the path the CUSUM took.", "1;33"))
            print(c("          A deployment needs a written policy for this.", "33"))

        if capture is not None:
            capture["days"].append({
                "day": args.dose_day + i,
                "date": shown,
                "value": None if not np.isfinite(today_val) else round(float(today_val), 2),
                "z": None if not np.isfinite(z_now) else round(float(z_now), 3),
                "S": round(S_now, 3),
                "n_valid": n_valid,
                "alarm": str(alarm.date()) if alarm is not None else None,
                "withdrawn": bool(withdrawn),
                "revisions": [{"date": str(d.date()), "old": round(old, 2),
                               "new": round(new, 2), "scored": bool(d >= dose_date)}
                              for d, old, new in revisions],
            })

        prev_snap, prev_alarm = snap, alarm
        if args.sleep:
            time.sleep(args.sleep)

    if printed_header:
        print()
        print(rule())
        verdict = (c("no alarm", "32") if prev_alarm is None
                   else c(f"ALARM on {prev_alarm.date()}", "1;31"))
        print(f"  after {args.days} days: {verdict}    "
              f"alarms withdrawn after a revision: {alarms_withdrawn}")
        print(c(f"  the limit never moved: h = {armed[0]:.2f}, fixed before the first "
                f"monitored day", "2"))
        print(rule())
        print()

    if capture is not None and capture["days"]:
        import json
        capture["summary"] = {"alarms_withdrawn": alarms_withdrawn,
                              "final_alarm": str(prev_alarm.date()) if prev_alarm else None,
                              "days_replayed": args.days}
        with open(args.json, "w") as f:
            json.dump(capture, f, indent=1)
        print(c(f"  replay written to {args.json}", "2"))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
