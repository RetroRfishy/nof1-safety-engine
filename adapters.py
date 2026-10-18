"""Dataset adapters: turn raw per-person daily data into Subjects.

engine.py needs only a Subject (a complete daily series plus a start date) and a
Metric. Everything dataset-specific stops at this file, so adding a dataset means
writing a reader that ends in to_subject() -- not touching the engine.

The hazard this file exists to remove: a hand-written loader that drops missing
days instead of marking them NaN still runs, but the index is no longer a run of
consecutive calendar days, so phi is fitted across gaps and comes out wrong with
nothing to signal it. to_subject() reindexes and validates so that cannot happen.

Nothing here changes a measured number. describe() reports what the engine is
about to assume about a dataset; measured_on() refuses to let the app quote a
false-alarm rate from a dataset other than the one it was measured on.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine import (HORIZON, MIN_BEFORE, REF_BASELINE_DAYS, BLOCK_DAYS,
                    MIN_CYCLES, fit_baseline, standardised_residuals)
from subject import Metric, Subject


class LoaderError(ValueError):
    """The raw data cannot be turned into a Subject."""


# ----------------------------------------------------------------------
# The one constructor every loader should end in
# ----------------------------------------------------------------------
def to_subject(subject_id: str,
               dates,
               values,
               start,
               *,
               simulated: bool = False,
               missing_values: tuple = (),
               tz_naive: bool = True) -> Subject:
    """Build a Subject from raw (dates, values).

    Does the three things a loader must get right and is easy to get wrong:
      1. maps sentinel values (e.g. PMData's 0.0) to NaN;
      2. reindexes to a COMPLETE daily range, so absent dates become NaN rows;
      3. rejects duplicate dates rather than silently keeping one.

    `start` is the first day on the compound: a date, or an int offset in days
    from the first date in the series.
    """
    d = pd.to_datetime(pd.Index(dates))
    if tz_naive and d.tz is not None:
        d = d.tz_convert(None)
    d = d.normalize()
    v = np.asarray(values, dtype=float)
    if len(d) != len(v):
        raise LoaderError(f"{subject_id}: {len(d)} dates but {len(v)} values")
    if len(d) == 0:
        raise LoaderError(f"{subject_id}: no rows")

    for sentinel in missing_values:
        v = np.where(v == sentinel, np.nan, v)

    s = pd.Series(v, index=d).sort_index()
    if s.index.has_duplicates:
        dupes = s.index[s.index.duplicated()].unique()[:3]
        raise LoaderError(f"{subject_id}: duplicate dates, e.g. "
                          f"{', '.join(str(x.date()) for x in dupes)}")
    if not s.notna().any():
        raise LoaderError(f"{subject_id}: every value is missing")

    full = pd.date_range(s.index[0], s.index[-1], freq="D")
    s = s.reindex(full).astype(float)
    s.name = subject_id

    if isinstance(start, (int, np.integer)):
        if start >= len(s):
            raise LoaderError(f"{subject_id}: series is {len(s)} days; "
                              f"start day {start} leaves no days after start")
        start_ts = s.index[0] + pd.Timedelta(days=int(start))
    else:
        start_ts = pd.Timestamp(start).normalize()
        if start_ts <= s.index[0] or start_ts > s.index[-1]:
            raise LoaderError(f"{subject_id}: start {start_ts.date()} is outside the "
                              f"record {s.index[0].date()} to {s.index[-1].date()}")

    n_before = int(s[s.index < start_ts].notna().sum())
    if n_before < MIN_BEFORE:
        raise LoaderError(f"{subject_id}: {n_before} days with data before "
                          f"{start_ts.date()}; need at least {MIN_BEFORE}")

    return Subject(subject_id=subject_id, series=s, start=start_ts, simulated=simulated)


# ----------------------------------------------------------------------
# A working CSV loader, not a stub
# ----------------------------------------------------------------------
def load_csv(path: str,
             start,
             *,
             id_col: str = "subject_id",
             date_col: str = "date",
             value_col: str = "value",
             missing_values: tuple = (),
             subject_ids: list[str] | None = None) -> tuple[list[Subject], list[str]]:
    """Long-format CSV: one row per (subject, day).

        subject_id,date,value
        p01,2019-11-20,52.3
        p01,2019-11-21,

    `start` may be a single date applied to everyone, an int offset applied to
    everyone, or a dict {subject_id: date-or-offset}.

    Returns (subjects, skipped) where skipped holds one reason per subject that
    could not be loaded. Nothing is dropped silently.
    """
    df = pd.read_csv(path)
    for col in (id_col, date_col, value_col):
        if col not in df.columns:
            raise LoaderError(f"{path}: no column {col!r}; found {list(df.columns)}")

    out, skipped = [], []
    ids = subject_ids if subject_ids is not None else sorted(df[id_col].astype(str).unique())
    for sid in ids:
        rows = df[df[id_col].astype(str) == sid]
        if rows.empty:
            skipped.append(f"{sid}: no rows in {os.path.basename(path)}")
            continue
        st = start[sid] if isinstance(start, dict) else start
        try:
            out.append(to_subject(sid, rows[date_col], rows[value_col], st,
                                  missing_values=missing_values))
        except (LoaderError, ValueError) as exc:
            skipped.append(str(exc))
    if not out:
        raise LoaderError(f"{path}: no usable subjects. " + "; ".join(skipped))
    return out, skipped


def load_wide_csv(path: str, start, *, date_col: str = "date",
                  missing_values: tuple = ()) -> tuple[list[Subject], list[str]]:
    """Wide-format CSV: a date column, then one column per subject."""
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise LoaderError(f"{path}: no column {date_col!r}; found {list(df.columns)}")
    out, skipped = [], []
    for sid in [c for c in df.columns if c != date_col]:
        st = start[sid] if isinstance(start, dict) else start
        try:
            out.append(to_subject(str(sid), df[date_col], df[sid], st,
                                  missing_values=missing_values))
        except (LoaderError, ValueError) as exc:
            skipped.append(str(exc))
    if not out:
        raise LoaderError(f"{path}: no usable subjects. " + "; ".join(skipped))
    return out, skipped


# ----------------------------------------------------------------------
# What the engine is about to assume about this dataset
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Description:
    subject_id: str
    n_days: int          # calendar length
    n_present: int
    coverage: float
    longest_gap: int     # consecutive absent days
    n_before: int        # present days before start
    n_after: int         # calendar days from start to the end of the record
    n_valid: int         # scorable residuals in the first HORIZON days after start
    phi: float
    n_eff: float         # effective sample size of the before period
    sd_x: float
    warnings: tuple[str, ...]


def _longest_gap(s: pd.Series) -> int:
    miss = s.isna().to_numpy()
    best = run = 0
    for m in miss:
        run = run + 1 if m else 0
        best = max(best, run)
    return best


def describe(subject: Subject) -> Description:
    """Report, per subject, the quantities that decide whether this project's
    constants still make sense on a new dataset. Never raises: a subject that
    cannot be fitted comes back with the reason in `warnings`."""
    s = subject.series
    before = s[s.index < subject.start]
    after = s[s.index >= subject.start]
    n_present = int(s.notna().sum())
    warn: list[str] = []

    try:
        base = fit_baseline(subject)
        phi, sd_x = base.phi, base.sd_x
        n_eff = base.n_before * (1 - phi) / (1 + phi)
        z = standardised_residuals(subject, base)
        win = z[z.index >= subject.start].iloc[:min(HORIZON, len(after))]
        n_valid = int(win.notna().sum())
    except ValueError as exc:
        phi = sd_x = n_eff = float("nan")
        n_valid = 0
        warn.append(str(exc))

    n_before = int(before.notna().sum())
    if n_before < REF_BASELINE_DAYS:
        warn.append(f"before period has {n_before} days with data; the reference library "
                    f"re-fits on {REF_BASELINE_DAYS}-day windows")
    if len(after) < HORIZON:
        warn.append(f"only {len(after)} days after start; HORIZON is {HORIZON}")
    if n_valid and n_valid < 0.7 * min(HORIZON, len(after)):
        warn.append(f"{n_valid} scorable residuals in a {min(HORIZON, len(after))}-day "
                    f"window: gaps cost more than they look, since a residual needs "
                    f"two consecutive present days")
    if np.isfinite(phi) and abs(phi) >= 0.94:
        warn.append(f"phi = {phi:.2f} is at the clip; the baseline may be too short or "
                    f"the series may be trending")
    if np.isfinite(n_eff) and n_eff < 3:
        warn.append(f"effective sample size {n_eff:.1f}: this baseline carries very "
                    f"little information")
    if len(after) < 2 * MIN_CYCLES * BLOCK_DAYS:
        warn.append(f"{len(after)} days after start; a {MIN_CYCLES}-cycle trial needs "
                    f"{2 * MIN_CYCLES * BLOCK_DAYS}")

    return Description(subject.subject_id, len(s), n_present, n_present / len(s),
                       _longest_gap(s), n_before, len(after), n_valid,
                       phi, n_eff, sd_x, tuple(warn))


def describe_table(subjects: list[Subject]) -> pd.DataFrame:
    rows = []
    for d in (describe(s) for s in subjects):
        rows.append({"subject": d.subject_id, "days": d.n_days,
                     "coverage": round(d.coverage, 3), "longest_gap": d.longest_gap,
                     "before": d.n_before, "after": d.n_after, "n_valid": d.n_valid,
                     "phi": round(d.phi, 3) if np.isfinite(d.phi) else None,
                     "n_eff": round(d.n_eff, 1) if np.isfinite(d.n_eff) else None,
                     "sd_x": round(d.sd_x, 3) if np.isfinite(d.sd_x) else None,
                     "warnings": len(d.warnings)})
    return pd.DataFrame(rows)


def constants_report(subjects: list[Subject]) -> list[str]:
    """Whether this project's constants are plausible for this cohort. These were
    set on PMData (13 subjects, ~150 days, phi 0.58-0.92) and do not transfer
    automatically."""
    ds = [describe(s) for s in subjects]
    phis = [d.phi for d in ds if np.isfinite(d.phi)]
    out = []
    n = len(subjects)
    out.append(f"{n} subjects loaded.")
    if n < 2:
        out.append("ONLY ONE SUBJECT: the reference-group limit cannot be built. The "
                   "own-baseline limit is the only option, and it measured 14% against "
                   "a nominal 5% on PMData.")
    elif n < 30:
        out.append(f"{n} subjects is a small reference group. With 13, the threshold was "
                   f"effectively set by one person's drift episode and moved 10-46% when "
                   f"any one subject was dropped. 50-100 is what makes it stable.")
    if phis:
        lo, hi = min(phis), max(phis)
        out.append(f"phi ranges {lo:.2f} to {hi:.2f} (PMData: 0.58 to 0.92).")
        if hi < 0.3:
            out.append("phi is low: whitening costs little here, so detection should be "
                       "better than PMData's. Re-measure rather than assume.")
    lens = [d.n_days for d in ds]
    out.append(f"series length {min(lens)} to {max(lens)} days (PMData: ~150).")
    if max(lens) > 300:
        out.append(f"Records are long. HORIZON={HORIZON} and REF_BASELINE_DAYS="
                   f"{REF_BASELINE_DAYS} were chosen for ~150-day records; longer "
                   f"baselines would estimate phi better.")
    covs = [d.coverage for d in ds]
    out.append(f"coverage {min(covs):.0%} to {max(covs):.0%}.")
    gaps = max(d.longest_gap for d in ds)
    if gaps >= 7:
        out.append(f"longest gap is {gaps} days. Gaps in runs destroy whole blocks, so "
                   f"use the block-bootstrap p_complete, not the binomial.")
    out.append("")
    out.append("MEASURED NUMBERS DO NOT TRANSFER. Re-run checks 6, 7, 10 and 11 on this "
               "dataset to get its own false-alarm rate, detection rate, efficacy false "
               "positives and baseline-model diagnostic.")
    return out


# ----------------------------------------------------------------------
# The guard: never quote a rate measured on a different dataset
# ----------------------------------------------------------------------
def fingerprint(subjects: list[Subject]) -> str:
    """A stable id for the loaded cohort: subject ids, record spans and value
    counts. Changes if the data changes; does not change between runs."""
    parts = []
    for s in sorted(subjects, key=lambda x: x.subject_id):
        idx = s.series.index
        parts.append(f"{s.subject_id}|{idx[0].date()}|{idx[-1].date()}|"
                     f"{len(idx)}|{int(s.series.notna().sum())}|{s.start.date()}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def measured_on(subjects: list[Subject], results_path: str = "check_results.json") -> tuple[bool, str]:
    """(ok, message). True when check_results.json was produced from exactly this
    cohort, so its false-alarm rate may be displayed. False otherwise, and the
    app must say the rate has not been measured on the data in front of it
    rather than quoting PMData's."""
    fp = fingerprint(subjects)
    if not os.path.exists(results_path):
        return False, ("No check_results.json: the false-alarm rate has not been "
                       "measured on this dataset. Run checks.py first.")
    try:
        with open(results_path) as f:
            saved = json.load(f).get("fingerprint")
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"check_results.json unreadable ({exc}); rate not verified."
    if saved is None:
        return False, ("check_results.json has no fingerprint, so it cannot be matched "
                       "to this dataset. Re-run checks.py.")
    if saved != fp:
        return False, ("The measured false-alarm rate belongs to a different dataset "
                       f"(fingerprint {saved} vs {fp}). Re-run checks.py on this one; "
                       "until then no rate is shown.")
    return True, "Rate measured on this dataset."


def write_fingerprint(subjects: list[Subject], results_path: str = "check_results.json") -> None:
    """Called by checks.py after a run, so the app can verify the match."""
    data = {}
    if os.path.exists(results_path):
        try:
            with open(results_path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            data = {}
    data["fingerprint"] = fingerprint(subjects)
    with open(results_path, "w") as f:
        json.dump(data, f, indent=1)
