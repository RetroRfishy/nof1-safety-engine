"""Acceptance checks (spec section 7). Run: python checks.py [--root PATH] [N ...]

Fixed seeds. Prints PASS / FAIL with the numbers. Checks 6 and 10 are
measurements on real null data and print MEASURED; they have no threshold.
Measured rates are written to check_results.json for the app to read."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

import engine as E
import loaders as L
from subject import DEMO_METRIC, Subject

DEFAULT_ROOT = os.path.expanduser("~/Downloads/pmdata")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_results.json")
START_DAY = 60
TRIAL_DAYS = 84
M = DEMO_METRIC


def wilson(a: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    zc = stats.norm.ppf(1 - (1 - conf) / 2)
    p = a / n
    den = 1 + zc ** 2 / n
    centre = (p + zc ** 2 / (2 * n)) / den
    half = zc * np.sqrt(p * (1 - p) / n + zc ** 2 / (4 * n ** 2)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def usable_subjects(root: str, start_day: int = START_DAY, min_days: int = 0, quiet: bool = False):
    out = []
    for pid in L.list_pmdata(root):
        try:
            subj = L.load_pmdata(root, pid, start_day)
            if len(subj.series) < min_days:
                raise L.SubjectUnusable(f"{pid}: series is {len(subj.series)} days; need {min_days}")
            out.append(subj)
        except L.SubjectUnusable as err:
            if not quiet:
                print(f"    skip {err}")
    return out


def demo_subject(root: str) -> Subject:
    return L.load_pmdata(root, L.demo_subject_id(root, START_DAY, TRIAL_DAYS), START_DAY)


def save_result(key: str, value) -> None:
    data = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            data = json.load(f)
    data[key] = value
    with open(RESULTS_PATH, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------- checks
def check1(root):
    print("1. Loader")
    ok_all = True
    for pid in L.list_pmdata(root):
        try:
            raw = L.read_raw(root, pid)
        except L.SubjectUnusable as err:
            print(f"    skip {err}")
            continue
        s = L.raw_to_series(raw)
        full = pd.date_range(s.index[0], s.index[-1], freq="D")
        complete = len(s.index) == len(full) and (s.index == full).all()
        no_zeros = not (s == 0).any()
        zero_entries = sum(e["value"]["value"] == 0.0 for e in raw)
        absent = len(full) - len(raw)
        nan_ok = int(s.isna().sum()) == zero_entries + absent
        ok = complete and no_zeros and nan_ok
        ok_all &= ok
        print(f"    {pid}: days={len(s)} NaN={int(s.isna().sum())} (zero entries {zero_entries} + absent {absent}) "
              f"complete={complete} no_zeros={no_zeros} -> {verdict(ok)}")
    print(f"  {verdict(ok_all)}")
    return ok_all


def check2(root):
    print("2. Before/after on a known +5 shift, SD 2")
    rng = np.random.default_rng(2)
    n = 500
    idx = pd.date_range("2020-01-01", periods=2 * n, freq="D")
    x = rng.normal(60, 2, 2 * n)
    x[n:] += 5
    subj = Subject("synthetic", pd.Series(x, index=idx), idx[n])
    ba = E.before_after(subj)
    ok = abs(ba.change - 5) < 0.5 and abs(ba.std_change - 2.5) < 0.25
    print(f"    change={ba.change:.3f} (expect ~5), standardised={ba.std_change:.3f} (expect ~2.5), "
          f"before SD={ba.before_sd:.3f}")
    print(f"  {verdict(ok)}")
    demo = demo_subject(root)
    d = E.before_after(demo)
    print(f"    demo subject {demo.subject_id} (start {demo.start.date()}): "
          f"before {d.before_mean:.2f} ± {d.before_sd:.2f} {M.unit} (N={d.before_n}); "
          f"after {d.after_mean:.2f} ± {d.after_sd:.2f} (N={d.after_n}); "
          f"change {d.change:+.2f} {M.unit} = {d.std_change:+.2f} SD")
    return ok


def check3(root):
    print("3. Baseline model on a 500-day AR(1), phi = 0.8")
    rng = np.random.default_rng(3)
    n, phi_true = 500, 0.8
    x = np.empty(n + 1)
    x[0] = rng.normal(0, 1 / np.sqrt(1 - phi_true ** 2))
    for t in range(1, n + 1):
        x[t] = phi_true * x[t - 1] + rng.normal()
    x += 60
    idx = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    subj = Subject("ar1", pd.Series(x, index=idx), idx[n])     # first n days are "before"
    base = E.fit_baseline(subj)
    e = E.residuals(x[:n], base.mu, base.phi)[1:]
    r1 = np.corrcoef(e[1:], e[:-1])[0, 1]
    ok = abs(base.phi - phi_true) <= 0.08 and abs(r1) <= 0.1
    print(f"    fitted phi={base.phi:.3f} (true 0.8, tol ±0.08); residual lag-1 autocorr={r1:+.3f} (tol ±0.1)")
    print(f"  {verdict(ok)}")
    return ok


def check4(root):
    print("4. CUSUM at rest")
    S = E.cusum(np.zeros(100), E.cusum_k(0.7), 1)
    S2 = E.cusum(np.zeros(100), E.cusum_k(0.7), -1)
    ok = bool(np.all(S == 0) and np.all(S2 == 0))
    print(f"    max S (up)={S.max()}, max S (down)={S2.max()}")
    print(f"  {verdict(ok)}")
    return ok


def check5(root):
    print("5. Limit self-check on the demo subject (checks the code, not real-world validity)")
    demo = demo_subject(root)
    base = E.fit_baseline(demo)
    n_valid = E.harm_monitor(demo, base, None, M, method="own_baseline", seed=0).n_valid
    h = E.harm_limit(base, n_valid, seed=1, metric=M)
    rng = np.random.default_rng(5)
    Z = rng.choice(base.z, size=(1000, n_valid), replace=True)
    maxes = E._cusum_max_batch(Z, E.cusum_k(base.phi), 1 if M.harm_direction == "up" else -1)
    rate = float(np.mean(maxes > h))
    ok = 0.035 <= rate <= 0.065
    print(f"    subject {demo.subject_id}, n_valid={n_valid}, h={h:.3f}, k={E.cusum_k(base.phi):.3f}: "
          f"{int(np.sum(maxes > h))}/1000 fresh windows alarmed = {rate:.1%} (need 3.5–6.5%)")
    print(f"  {verdict(ok)}")
    return ok


MW_OFFSETS = [60, 74, 88, 102]   # check 6 window starts (14-day stride)
BASELINE_DAYS = 60               # fixed before period for every window


def null_windows(root):
    """(subject, rebased subject, baseline) for every check-6 window: fixed 60-day
    baseline ending the day before the start, HORIZON days after it."""
    out = []
    for subj in usable_subjects(root):
        s0 = subj.series.index[0]
        for off in MW_OFFSETS:
            if off + E.HORIZON > len(subj.series):
                continue
            sub = E.rebase(subj, s0 + pd.Timedelta(days=off), BASELINE_DAYS)
            try:
                base = E.fit_baseline(sub)
            except ValueError as err:
                print(f"    skip window: {err}")
                continue
            out.append((off, sub, base))
    return out


def report_null_rate(rows, label, key):
    """rows: list of (offset, subject_id, alarmed)."""
    a = sum(r[2] for r in rows); n = len(rows)
    subs = sorted({r[1] for r in rows})
    subs_alarmed = sorted({r[1] for r in rows if r[2]})
    lo, hi = wilson(a, n)
    lo_s, hi_s = wilson(len(subs_alarmed), len(subs))
    d60 = [r for r in rows if r[0] == 60]
    a60 = sum(r[2] for r in d60)
    print(f"    {label}: {a} of {n} windows alarmed ({a / n:.1%}), Wilson 95% CI {lo:.1%}–{hi:.1%} "
          f"(windows overlap within a person, so this CI is too narrow)")
    print(f"    subjects: {len(subs)}; with at least one alarm: {len(subs_alarmed)} {subs_alarmed} "
          f"({len(subs_alarmed) / len(subs):.0%}, Wilson {lo_s:.0%}–{hi_s:.0%})")
    print(f"    day-60 windows only: {a60} of {len(d60)} ({a60 / len(d60):.1%})")
    save_result(key, {"alarmed": a, "windows": n, "rate": a / n, "ci_windows": [lo, hi],
                      "subjects": len(subs), "subjects_alarmed": subs_alarmed,
                      "day60": {"alarmed": a60, "n": len(d60)}})


METHODS = ("reference", "own_baseline")


def reference_library(root):
    """Library from every usable subject's real post-start data (start day 60).
    reference_limit leaves the monitored subject out by ID."""
    return E.reference_library(usable_subjects(root, quiet=True))


def library_summary(lib):
    per = {}
    for w in lib.windows:
        per[w.subject_id] = per.get(w.subject_id, 0) + 1
    return f"{len(lib.windows)} windows from {len(per)} subjects ({min(per.values())}–{max(per.values())} per subject)"


def check6(root):
    print(f"6. Harm false alarms on real data. Starts {MW_OFFSETS}, {BASELINE_DAYS}-day baseline, "
          f"horizon {E.HORIZON}. Nobody took anything.")
    wins = null_windows(root)
    lib = reference_library(root)
    print(f"    reference library: {library_summary(lib)}")
    for method in METHODS:
        rows, nw, ns = [], [], []
        for off, sub, base in wins:
            r = E.harm_monitor(sub, base, lib, M, method=method, seed=0)
            rows.append((off, sub.subject_id, r.alarm_date is not None))
            if method == "reference":
                nw.append(r.ref_windows); ns.append(r.ref_subjects)
        if method == "reference":
            print(f"    reference limit built from {min(nw)}–{max(nw)} windows, {min(ns)}–{max(ns)} subjects per limit")
        report_null_rate(rows, f"{method} limit", f"check6_{method}")
    print("  MEASURED (no threshold; reported as measured, not tuned). Spec: reference 7.0% (3/43), own baseline 14.0% (6/43).")
    reference_stability(wins, lib)
    return True


def reference_stability(wins, lib):
    """Reference h for each subject's day-60 window, recomputed with each other
    library subject left out in turn. Measurement of how much h depends on
    library composition; nothing is changed."""
    print("    Reference-limit stability: h recomputed with each library subject left out in turn (day-60 windows)")
    out = {}
    for off, sub, base in wins:
        if off != 60:
            continue
        n_valid = E.harm_monitor(sub, base, lib, M).n_valid
        k = E.cusum_k(base.phi)
        full = E.reference_limit(sub, lib, n_valid, k, M).h
        loo = {}
        for j in sorted(lib.subject_ids - {sub.subject_id}):
            reduced = E.RefLibrary(tuple(w for w in lib.windows if w.subject_id != j))
            loo[j] = E.reference_limit(sub, reduced, n_valid, k, M).h
        lo_id, hi_id = min(loo, key=loo.get), max(loo, key=loo.get)
        lo, hi = loo[lo_id], loo[hi_id]
        print(f"      {sub.subject_id}: full h={full:5.2f}; leave-one-out range {lo:5.2f} (without {lo_id}) – "
              f"{hi:5.2f} (without {hi_id}); spread {hi - lo:4.2f} = {(hi - lo) / full:.0%} of h")
        out[sub.subject_id] = {"full": full, "min": lo, "min_without": lo_id, "max": hi,
                               "max_without": hi_id, "loo": loo}
    spreads = [(v["max"] - v["min"]) / v["full"] for v in out.values()]
    print(f"    spread across subjects: {min(spreads):.0%}–{max(spreads):.0%} of h (median {np.median(spreads):.0%})")
    save_result("check6_stability", out)

    print("    False-alarm rate with each library subject left out in turn (all 43 windows re-run):")
    rates = {}
    for j in sorted(lib.subject_ids):
        reduced = E.RefLibrary(tuple(w for w in lib.windows if w.subject_id != j))
        alarmed = [E.harm_monitor(sub, base, reduced, M).alarm_date is not None for off, sub, base in wins]
        rates[j] = (sum(alarmed), len(alarmed))
        print(f"      without {j}: {sum(alarmed)} of {len(alarmed)} ({sum(alarmed) / len(alarmed):.1%})")
    lo_j = min(rates, key=lambda j: rates[j][0])
    hi_j = max(rates, key=lambda j: rates[j][0])
    lo, hi = rates[lo_j][0] / rates[lo_j][1], rates[hi_j][0] / rates[hi_j][1]
    counts = {}
    for j, (a, n) in rates.items():
        counts.setdefault(a, []).append(j)
    summary = "; ".join(f"{a} of {n_w} ({a / n_w:.1%}) with {len(js)} of {len(rates)} removals"
                        + (f" ({', '.join(js)})" if len(js) <= 3 else "")
                        for a, js in sorted(counts.items()) for n_w in [rates[js[0]][1]])
    print(f"    false-alarm rate range: {lo:.1%}–{hi:.1%}. {summary}")
    save_result("check6_fp_range", {"min": lo, "min_without": lo_j, "max": hi, "max_without": hi_j,
                                    "by_left_out": {j: {"alarmed": a, "windows": n} for j, (a, n) in rates.items()}})


def check7(root):
    print("7. Harm detection: plant_harm(delta_sd=2) in each check-6 window")
    wins = null_windows(root)
    lib = reference_library(root)
    out = {}
    for method in METHODS:
        days = []
        for off, sub, base in wins:
            r = E.harm_monitor(E.plant_harm(sub, M, delta_sd=2.0), base, lib, M, method=method, seed=0)
            days.append((r.alarm_date - sub.start).days if r.alarm_date is not None else None)
        hit = [d for d in days if d is not None]
        q = np.percentile(hit, [25, 50, 75]) if hit else [np.nan] * 3
        print(f"    {method:12s}: caught {len(hit)} of {len(days)} within {E.HORIZON} days "
              f"({len(hit) / len(days):.0%}); days to alarm median {q[1]:.0f}, IQR {q[0]:.0f}–{q[2]:.0f} "
              f"(shift begins day 5, full by day 7)")
        out[method] = {"caught": len(hit), "windows": len(days), "rate": len(hit) / len(days),
                       "median_day": float(q[1]), "iqr": [float(q[0]), float(q[2])], "days": days}
    print("    Spec: reference 53% (23/43), median 16 (IQR 10–24); own baseline 79% (34/43), median 11.")
    save_result("check7", {m: {k: v for k, v in d.items() if k != "days"} for m, d in out.items()})
    demo = demo_subject(root)
    base = E.fit_baseline(demo)
    planted = E.plant_harm(demo, M, delta_sd=2.0)
    r = E.harm_monitor(planted, base, lib, M, method="reference")
    ok = r.alarm_date is not None
    if ok:
        print(f"    demo {demo.subject_id}, reference limit: alarm on {r.alarm_date.date()} "
              f"(day {(r.alarm_date - demo.start).days} after start), h={r.h:.2f} from {r.ref_windows} windows/"
              f"{r.ref_subjects} subjects, simulated={planted.simulated}")
        print(f"    alarm record: {json.dumps(r.record)}")
    else:
        print(f"    demo {demo.subject_id}, reference limit: NO alarm within {E.HORIZON} days; "
              f"max S={r.S.max():.2f}, h={r.h:.2f}")
    print(f"  {verdict(ok)}")
    return ok


CHECK_SEED = 8
TRIAL_CYCLES = 6          # PMData can't fit DEFAULT_CYCLES (8) after a day-60 start, so the demo uses 6


def record_end(subj):
    return subj.series.last_valid_index()


def fits(subj, n_cycles=TRIAL_CYCLES):
    try:
        E.make_schedule(subj.start, n_cycles, M, seed=0, record_end=record_end(subj))
        return True, ""
    except E.ScheduleDoesNotFit as err:
        return False, str(err)


def check8(root):
    print("8. Efficacy gates")
    want = {3: 0.25, 5: 0.0625}
    demo = demo_subject(root)
    base = E.fit_baseline(demo)
    ok = True
    for n, v in want.items():
        sc = E.schedule_check(base, n, M, seed=CHECK_SEED)
        good = abs(sc.p_floor - v) < 1e-12 and not sc.allowed
        ok &= good
        print(f"    {n} cycles: p_floor={sc.p_floor:.4g} (want {v}); refused={not sc.allowed} -> {verdict(good)}")
    sc6 = E.schedule_check(base, 6, M, seed=CHECK_SEED)
    good = sc6.p_floor < E.ALPHA and sc6.allowed
    ok &= good
    print(f"    6 cycles: p_floor={sc6.p_floor:.4g} (< {E.ALPHA}); allowed={sc6.allowed} -> {verdict(good)}")

    subs = usable_subjects(root, quiet=True)
    print(f"    Schedule-length precondition ({TRIAL_CYCLES} cycles from day {START_DAY}, against the last day with data):")
    fit = {}
    for subj in subs:
        f, why = fits(subj)
        fit[subj.subject_id] = f
        if not f:
            print(f"      {subj.subject_id}: rejected. {why}")
    print(f"      fit: {sum(fit.values())} of {len(fit)} subjects ({', '.join(k for k, v in fit.items() if v)})")
    good = fit.get("p10") is False
    ok &= good
    print(f"    p10 rejected by the length precondition -> {verdict(good)}")

    print(f"    Per subject at {TRIAL_CYCLES} cycles. p_complete resamples the before-period missingness in "
          f"blocks of {E.MISS_BLOCK_LENGTHS} days and takes the minimum; binomial shown for comparison.")
    print("    Observed = share of 20 random real schedules that reached a verdict (subjects whose schedule fits).")
    print("      subj  phi   cov   p_compl(block)  binomial  ceiling  min_detectable  observed")
    table = {}
    for subj in subs:
        b = E.fit_baseline(subj)
        sc = E.schedule_check(b, TRIAL_CYCLES, M, seed=CHECK_SEED)
        binom_pc = E.p_complete(b.coverage, TRIAL_CYCLES, E.BLOCK_DAYS, M.onset_days)
        md = f"{sc.min_detectable:.1f} SD" if sc.min_detectable is not None else "not reachable"
        obs = "-"
        if fit[subj.subject_id]:
            done = sum(E.efficacy(subj, E.make_schedule(subj.start, TRIAL_CYCLES, M, seed=k), M).verdict
                       != "not_enough" for k in range(20))
            obs = f"{done}/20"
        print(f"      {subj.subject_id}  {b.phi:4.2f}  {b.coverage:4.0%}      {sc.p_complete:4.2f}         {binom_pc:4.2f}"
              f"     {sc.power.get('ceiling', float('nan')):4.2f}   {md:>13s}    {obs}")
        table[subj.subject_id] = {"phi": b.phi, "coverage_before": b.coverage, "p_complete": sc.p_complete,
                                  "p_complete_method": sc.p_complete_method, "p_complete_binomial": binom_pc,
                                  "p_complete_by_block": sc.p_complete_by_block,
                                  "min_detectable": sc.min_detectable, "fits": fit[subj.subject_id], "observed": obs}
    good = table["p03"]["p_complete"] < 0.8
    ok &= good
    print(f"    p03 block-bootstrap p_complete = {table['p03']['p_complete']:.2f} (< 0.8 expected) -> {verdict(good)}")

    print(f"    Block-length sensitivity: p_complete at block lengths {E.MISS_BLOCK_LENGTHS} "
          f"(reported value is the minimum), subjects where they differ:")
    for pid, row in table.items():
        vals = row["p_complete_by_block"]
        if vals and max(vals.values()) - min(vals.values()) >= 0.005:
            print(f"      {pid}: " + ", ".join(f"L={L}: {v:.2f}" for L, v in vals.items()) +
                  f" -> {row['p_complete']:.2f}")
    save_result("check8", table)
    print(f"  {verdict(ok)}")
    return ok


def check9(root):
    print("9. Efficacy wiring: plant_effect(delta_sd=1.0) on the demo subject, 6-cycle weekly schedule")
    demo = demo_subject(root)
    sched = E.make_schedule(demo.start, TRIAL_CYCLES, M, seed=CHECK_SEED, record_end=record_end(demo))
    planted = E.plant_effect(demo, sched, M, delta_sd=1.0)
    r0, r1 = E.efficacy(demo, sched, M), E.efficacy(planted, sched, M)
    sd = float(np.std(E.before_values(demo).to_numpy(), ddof=1))
    shift = (r1.cycles["d"] - r0.cycles["d"]).to_numpy()
    expect = (1 if M.works_direction == "up" else -1) * 1.0 * sd
    right_size = bool(np.allclose(shift[r1.cycles["usable"]], expect))
    in_dir = np.sign(r1.mean_d) == (1 if M.works_direction == "up" else -1)
    print(f"    schedule (seed {CHECK_SEED}): {[ '/'.join(c) for c in sched.cycles]}; before SD={sd:.2f} {M.unit}")
    for _, row in r1.cycles.iterrows():
        print(f"      cycle {row.cycle}: on {row.on_mean:.2f}  off {row.off_mean:.2f}  d={row.d:+.2f} {M.unit}")
    print(f"    planted shift per cycle = {np.round(shift, 3).tolist()} (expected {expect:+.3f} each) -> right size: {right_size}")
    print(f"    mean d = {r1.mean_d:+.2f} {M.unit} = {r1.mean_d_sd:+.2f} SD (works direction is {M.works_direction}: {in_dir});"
          f" cycles agreeing: {r1.n_agree} of {r1.n_usable}; randomisation p = {r1.p:.4f}")
    print(f"    verdict: {r1.message}" + ("" if r1.p < E.ALPHA else
          " — expected outcome: 1 SD is below min_detectable for this schedule, not a bug."))
    print(f"    (same schedule, nothing planted: mean d {r0.mean_d_sd:+.2f} SD, p = {r0.p:.4f})")
    ok = bool(right_size and in_dir)
    save_result("check9", {"p": r1.p, "d": r1.cycles["d"].tolist(), "mean_d_sd": r1.mean_d_sd,
                           "n_agree": r1.n_agree, "verdict": r1.verdict})
    print(f"  {verdict(ok)}")
    return ok


def check10(root):
    print(f"10. Efficacy false positives on real data: 20 random 6-cycle schedules per subject, nothing planted")
    subs = [s for s in usable_subjects(root, quiet=True) if fits(s)[0]]
    print(f"    subjects whose {TRIAL_CYCLES}-cycle schedule fits the record: {len(subs)}")
    runs, fp, not_enough, by_subj = 0, 0, 0, {}
    for subj in subs:
        hits = 0
        for seed in range(20):
            r = E.efficacy(subj, E.make_schedule(subj.start, TRIAL_CYCLES, M, seed=seed), M)
            runs += 1
            hits += r.verdict in ("working", "opposite")
            not_enough += r.verdict == "not_enough"
        fp += hits
        by_subj[subj.subject_id] = hits
        print(f"    {subj.subject_id}: {hits} of 20 'Consistent effect'")
    lo, hi = wilson(fp, runs)
    n_subj_fp = sum(v > 0 for v in by_subj.values())
    lo_s, hi_s = wilson(n_subj_fp, len(subs))
    print(f"    {fp} of {runs} runs said 'Consistent effect' ({fp / runs:.1%}), Wilson 95% CI {lo:.1%}–{hi:.1%} "
          f"(runs share data within a subject, so this CI is too narrow)")
    print(f"    subjects: {len(subs)}; with at least one: {n_subj_fp} (Wilson {lo_s:.0%}–{hi_s:.0%}); "
          f"runs with too few usable cycles: {not_enough}")
    print(f"    expected about {2 / 64:.1%} (p < 0.05 at 6 cycles means p = 2/64)")
    print("  MEASURED (no threshold; reported as measured, not tuned)")
    save_result("check10", {"fp": fp, "runs": runs, "rate": fp / runs, "ci": [lo, hi], "subjects": len(subs),
                            "subjects_with_fp": n_subj_fp, "by_subject": by_subj, "not_enough": not_enough})
    return True


def check11(root):
    print("11. Baseline-model diagnostic on real monitoring days: standardised residuals in each check-6 window")
    print("    Under the fitted AR(1), z on monitoring days has mean 0 and SD 1. Mean away from 0 = location error")
    print("    (mu moved); SD away from 1 = scale error (phi or sd_e mis-estimated). Report only.")
    rows = []
    for off, sub, base in null_windows(root):
        z = E.standardised_residuals(sub, base)
        zw = z[(z.index >= sub.start) & (z.index < sub.start + pd.Timedelta(days=E.HORIZON))].to_numpy()
        pair = ~np.isnan(zw[1:]) & ~np.isnan(zw[:-1])
        r1 = float(np.corrcoef(zw[1:][pair], zw[:-1][pair])[0, 1]) if pair.sum() > 2 else np.nan
        win = zw[~np.isnan(zw)]
        n = len(win)
        m, sd = float(win.mean()), float(win.std(ddof=1))
        big = abs(m) > 2 / np.sqrt(n)
        rows.append({"subject": sub.subject_id, "start_day": off, "n_valid": n, "mean_z": m, "sd_z": sd,
                     "mean_beyond_2se": bool(big), "lag1": r1})
        print(f"    {sub.subject_id} day {off:3d}: n_valid={n:2d}  mean z={m:+.2f}{' *' if big else '  '}  SD z={sd:.2f}  lag-1 r={r1:+.2f}")
    means = np.array([r["mean_z"] for r in rows]); sds = np.array([r["sd_z"] for r in rows])
    q = lambda a: f"median {np.median(a):+.2f}, IQR {np.percentile(a, 25):+.2f} to {np.percentile(a, 75):+.2f}, range {a.min():+.2f} to {a.max():+.2f}"
    r1s = np.array([r["lag1"] for r in rows])
    print(f"    lag-1 autocorrelation of z within windows: {q(r1s)} (about 0 if the whitening holds; the 2/sqrt(n) "
          f"rule below assumes it)")
    n_big = sum(r["mean_beyond_2se"] for r in rows)
    lo, hi = wilson(n_big, len(rows))
    subs_big = sorted({r["subject"] for r in rows if r["mean_beyond_2se"]})
    print(f"    mean z across {len(rows)} windows: {q(means)}")
    print(f"    SD z across windows:   {q(sds)}; RMS of window SDs {np.sqrt(np.mean(sds ** 2)):.2f}")
    print(f"    |mean z| > 2/sqrt(n_valid) (* above): {n_big} of {len(rows)} windows ({n_big / len(rows):.0%}, "
          f"Wilson {lo:.0%}–{hi:.0%}); about 5% expected if the model held. Subjects: {subs_big}")
    print("  MEASURED (no threshold; report only)")
    save_result("check11", {"windows": rows, "mean_z_median": float(np.median(means)),
                            "sd_z_median": float(np.median(sds)), "lag1_median": float(np.median(r1s)), "mean_beyond_2se": n_big,
                            "n_windows": len(rows), "ci": [lo, hi]})
    return True


CHECKS = {"1": check1, "2": check2, "3": check3, "4": check4, "5": check5,
          "6": check6, "7": check7, "8": check8, "9": check9, "10": check10, "11": check11}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("which", nargs="*")
    args = ap.parse_args(argv)
    which = args.which or list(CHECKS)
    results = {}
    for n in which:
        results[n] = CHECKS[n](args.root)
        print()
    failed = [n for n, ok in results.items() if not ok]
    print("All checks passed." if not failed else f"FAILED: {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
