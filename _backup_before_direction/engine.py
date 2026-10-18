"""Pure engine (spec section 5). No plotting, no Streamlit, no file I/O.
Every random draw takes a seed. Nothing here refers to a specific metric:
it sees a series and a Metric."""
from __future__ import annotations

from dataclasses import dataclass, replace

import itertools

import numpy as np
import pandas as pd
from scipy.stats import binom

from subject import Metric, Schedule, Subject

# 5.1 Constants
FP = 0.05             # harm false-alarm rate per monitoring course
HORIZON = 30          # choice: days in one harm-monitoring course
N_SIMS = 1000
DELTA = 2.0           # choice: shift the harm detector is tuned for, in baseline SDs
MIN_BEFORE = 14       # choice: refuse below this many before-days with data
ALPHA = 0.05          # efficacy significance level
BLOCK_DAYS = 7        # choice: weekly blocks (fast-acting compounds only)
MIN_CYCLES = 6        # follows from ALPHA: smallest n with 2/2**n < ALPHA
MIN_USABLE_DAYS = 2   # set by measurement (5.5): a block with fewer usable days is dropped
REF_BASELINE_DAYS = 60  # baseline length re-fitted for every reference-library window

BEFORE_AFTER_CAVEAT = ("A before/after difference alone can't show the compound caused it; "
                       "season, training or illness could. The on/off trial is what "
                       "attributes change to the compound.")


def _sign(direction: str) -> int:
    return 1 if direction == "up" else -1


def before_values(subject: Subject) -> pd.Series:
    s = subject.series
    return s[s.index < subject.start].dropna()


def after_values(subject: Subject, window_days: int | None = None) -> pd.Series:
    s = subject.series
    a = s[s.index >= subject.start]
    if window_days is not None:
        a = a[a.index < subject.start + pd.Timedelta(days=window_days)]
    return a.dropna()


# 5.2 Before / after
@dataclass(frozen=True)
class BeforeAfter:
    before_mean: float
    before_sd: float
    before_n: int
    after_mean: float
    after_sd: float
    after_n: int
    change: float
    std_change: float
    caveat: str = BEFORE_AFTER_CAVEAT


def before_after(subject: Subject, window_days: int | None = None) -> BeforeAfter:
    b = before_values(subject).to_numpy()
    a = after_values(subject, window_days).to_numpy()
    bm, bsd = float(np.mean(b)), float(np.std(b, ddof=1))
    am = float(np.mean(a)) if len(a) else np.nan
    asd = float(np.std(a, ddof=1)) if len(a) > 1 else np.nan
    change = am - bm
    return BeforeAfter(bm, bsd, len(b), am, asd, len(a), change, change / bsd)


# 5.3 Baseline model: AR(1) fitted on the before period
@dataclass(frozen=True)
class Baseline:
    mu: float
    sd_x: float
    phi: float
    sd_e: float
    z: np.ndarray        # standardised before-period residuals, non-NaN only
    n_before: int
    coverage: float = 1.0  # share of before-period days with data
    missing: np.ndarray | None = None   # before-period missingness indicator, one bool per calendar day


def _lag1_autocorr(x: np.ndarray, mu: float) -> float:
    """Lag-1 autocorrelation using only pairs of consecutive present days:
    mean of (x_{t-1}-mu)(x_t-mu) over those pairs, divided by the variance
    of all present days about mu."""
    d = x - mu
    pair = ~np.isnan(d[1:]) & ~np.isnan(d[:-1])
    if pair.sum() < 2:
        return 0.0
    num = np.mean(d[1:][pair] * d[:-1][pair])
    den = np.nanmean(d ** 2)
    return float(num / den)


def residuals(values, mu: float, phi: float) -> np.ndarray:
    """e_t = (x_t - mu) - phi (x_{t-1} - mu); NaN if either day is missing.
    The first element has no predecessor and is NaN."""
    x = np.asarray(values, dtype=float)
    e = np.full(len(x), np.nan)
    if len(x) > 1:
        e[1:] = (x[1:] - mu) - phi * (x[:-1] - mu)
    return e


def fit_baseline(subject: Subject) -> Baseline:
    s = subject.series
    before = s[s.index < subject.start].to_numpy(dtype=float)   # keeps NaN rows: needed for adjacency
    present = before[~np.isnan(before)]
    if len(present) < MIN_BEFORE:
        raise ValueError(f"{subject.subject_id}: {len(present)} before-days with data; need at least {MIN_BEFORE}")
    mu = float(np.mean(present))
    sd_x = float(np.std(present, ddof=1))
    phi = float(np.clip(_lag1_autocorr(before, mu), -0.95, 0.95))
    e = residuals(before, mu, phi)          # raw residuals first
    e = e[~np.isnan(e)]
    sd_e = float(np.std(e, ddof=1))         # then sd_e
    z = e / sd_e                            # then standardise
    return Baseline(mu, sd_x, phi, sd_e, z, len(present), len(present) / len(before), np.isnan(before))


def standardised_residuals(subject: Subject, base: Baseline) -> pd.Series:
    """z_t over the whole series using the baseline fit (so the first after-day
    uses the last before-day as its predecessor)."""
    e = residuals(subject.series.to_numpy(dtype=float), base.mu, base.phi)
    return pd.Series(e / base.sd_e, index=subject.series.index)


# 5.4 Harm
def cusum_k(phi: float) -> float:
    return 0.5 * DELTA * np.sqrt((1 - phi) / (1 + phi))


def cusum(z, k: float, s: int = 1) -> np.ndarray:
    """One-sided CUSUM: S_t = max(0, S_{t-1} + s z_t - k), S_0 = 0.
    z_t NaN -> S_t = S_{t-1} (no step)."""
    z = np.asarray(z, dtype=float)
    out = np.empty(len(z))
    S = 0.0
    for i, zi in enumerate(z):
        if not np.isnan(zi):
            S = max(0.0, S + s * zi - k)
        out[i] = S
    return out


def _cusum_max_batch(Z: np.ndarray, k: float, s: int) -> np.ndarray:
    """Max of the CUSUM for each row of Z (no NaN), vectorised over rows."""
    S = np.zeros(Z.shape[0])
    M = np.zeros(Z.shape[0])
    for j in range(Z.shape[1]):
        S = np.maximum(0.0, S + s * Z[:, j] - k)
        np.maximum(M, S, out=M)
    return M


def harm_limit(base: Baseline, n_valid: int, seed: int, metric: Metric) -> float:
    """Bootstrap calibration from the person's own baseline residuals:
    draw n_valid residuals with replacement, run the CUSUM, record the max;
    repeat N_SIMS times; h = 95th percentile."""
    if n_valid < 1:
        return np.inf
    rng = np.random.default_rng(seed)
    Z = rng.choice(base.z, size=(N_SIMS, n_valid), replace=True)
    maxes = _cusum_max_batch(Z, cusum_k(base.phi), _sign(metric.harm_direction))
    return float(np.percentile(maxes, 100 * (1 - FP)))


def rebase(subject: Subject, start: pd.Timestamp, baseline_days: int) -> Subject:
    """Same subject with a new start and a fixed before period of baseline_days
    (the days start - baseline_days ... start - 1)."""
    s = subject.series
    s = s[s.index >= start - pd.Timedelta(days=baseline_days)]
    return replace(subject, series=s, start=start)


# Reference-group limit: leave-one-subject-out, built from other people's real
# post-start data monitored with no exposure. This is the default limit.
@dataclass(frozen=True)
class RefWindow:
    subject_id: str
    start: pd.Timestamp  # date of the window's first valid residual
    z: np.ndarray        # the next valid residuals from start (up to HORIZON), against a
                         # baseline re-fitted on the REF_BASELINE_DAYS ending start - 1


@dataclass(frozen=True)
class RefLibrary:
    windows: tuple[RefWindow, ...]

    @property
    def subject_ids(self) -> set[str]:
        return {w.subject_id for w in self.windows}


def reference_library(others: list[Subject], horizon: int = HORIZON,
                      baseline_days: int = REF_BASELINE_DAYS) -> RefLibrary:
    """Slide through each subject's post-start series in valid-step space: one
    window per valid post-start residual, advancing one valid step at a time.
    The baseline is re-fitted for every window from the baseline_days ending the
    day before that window's own start, the way a real monitoring course runs."""
    out = []
    for subj in others:
        x = subj.series.to_numpy(dtype=float)
        idx = subj.series.index
        valid = np.zeros(len(x), dtype=bool)
        valid[1:] = ~np.isnan(x[1:]) & ~np.isnan(x[:-1])
        for i in np.flatnonzero(valid & (idx >= subj.start)):
            if i < baseline_days:
                continue
            sub = rebase(subj, idx[i], baseline_days)
            try:
                base = fit_baseline(sub)
            except ValueError:
                continue
            seg = x[i - 1:]                                 # include the predecessor of day i
            e = residuals(seg, base.mu, base.phi)[1:]
            z = (e / base.sd_e)[~np.isnan(e)][:horizon]
            out.append(RefWindow(subj.subject_id, idx[i], z))
    return RefLibrary(tuple(out))


@dataclass(frozen=True)
class RefLimit:
    h: float
    n_windows: int
    n_subjects: int


def reference_limit(subject: Subject, others, n_valid: int, k: float,
                    metric: Metric, seed: int = 0) -> RefLimit:
    """h = the (1 - FP) percentile of CUSUM maxima over every library window of
    every other subject, each scored on its first n_valid valid residuals with
    this subject's k. `others` is a list of Subjects or a prebuilt RefLibrary
    (same thing, built once). No random draws; seed is kept for the interface.
    Windows overlap heavily, so the effective n is n_subjects."""
    lib = others if isinstance(others, RefLibrary) else reference_library(others)
    if n_valid < 1:
        return RefLimit(np.inf, 0, 0)
    used = [w for w in lib.windows if w.subject_id != subject.subject_id and len(w.z) >= n_valid]
    if not used:
        return RefLimit(np.inf, 0, 0)
    Z = np.stack([w.z[:n_valid] for w in used])
    maxes = _cusum_max_batch(Z, k, _sign(metric.harm_direction))
    return RefLimit(float(np.percentile(maxes, 100 * (1 - FP))), len(used),
                    len({w.subject_id for w in used}))


@dataclass(frozen=True)
class HarmResult:
    S: pd.Series
    z: pd.Series
    h: float
    k: float
    method: str              # "reference" | "own_baseline"
    n_valid: int
    calendar_len: int
    alarm_date: pd.Timestamp | None
    record: dict | None      # the alarm as a JSON-able line for the app to write; None if no alarm
    ref_windows: int = 0     # reference method only
    ref_subjects: int = 0


def harm_monitor(subject: Subject, base: Baseline, others, metric: Metric,
                 horizon: int = HORIZON, method: str = "reference", seed: int = 0) -> HarmResult:
    """method="reference" (default): reference_limit over `others`.
    method="own_baseline": the bootstrap comparator, harm_limit; `others` unused."""
    z_all = standardised_residuals(subject, base)
    after = z_all[z_all.index >= subject.start]
    win = after.iloc[:min(horizon, len(after))]
    n_valid = int(win.notna().sum())
    k = cusum_k(base.phi)
    n_w = n_s = 0
    if method == "reference":
        ref = reference_limit(subject, others, n_valid, k, metric, seed)
        h, n_w, n_s = ref.h, ref.n_windows, ref.n_subjects
    elif method == "own_baseline":
        h = harm_limit(base, n_valid, seed, metric)
    else:
        raise ValueError(f"unknown method {method!r}")
    S = pd.Series(cusum(win.to_numpy(), k, _sign(metric.harm_direction)), index=win.index)
    over = S[S > h]
    alarm_date = over.index[0] if len(over) else None
    record = None
    if alarm_date is not None:
        record = {"subject": subject.subject_id, "date": alarm_date.strftime("%Y-%m-%d"),
                  "metric": metric.name, "h": round(h, 4), "n_valid": n_valid,
                  "FP": FP, "simulated": bool(subject.simulated)}
    return HarmResult(S, win, h, k, method, n_valid, len(win), alarm_date, record, n_w, n_s)


# 5.5 Efficacy: n-of-1 crossover trial
POWER_TARGET = 0.80
EFFECT_GRID = np.round(np.arange(0.1, 5.0001, 0.1), 2)   # effect sizes (baseline SDs) for min_detectable
DEFAULT_CYCLES = MIN_CYCLES + 2     # deployment default: two cycles of slack, so one bad week doesn't end the trial
MISS_BLOCK_LENGTHS = (3, 5, 7, 14)  # choice: block lengths tried when resampling the missingness indicator;
                                    # p_complete is the minimum across them (the conservative end)
MIN_RESAMPLE_DAYS = 28              # choice: shorter before periods fall back to the binomial
N_SIMS_COMPLETE = 10 * N_SIMS       # p_complete is only counting: 10x replicates, Monte Carlo noise about ±0.01


class ScheduleDoesNotFit(ValueError):
    """The schedule runs past the end of the available record."""


def schedule_end(start: pd.Timestamp, n_cycles: int, block_days: int = BLOCK_DAYS) -> pd.Timestamp:
    return start + pd.Timedelta(days=2 * n_cycles * block_days - 1)


def max_cycles(start: pd.Timestamp, record_end: pd.Timestamp, block_days: int = BLOCK_DAYS) -> int:
    return max(0, ((record_end - start).days + 1) // (2 * block_days))


def make_schedule(start: pd.Timestamp, n_cycles: int, metric: Metric, seed: int,
                  record_end: pd.Timestamp | None = None) -> Schedule:
    """Paired on/off weekly blocks; the order within each cycle is a coin flip.
    With record_end (the last day the record has data), refuse a schedule that
    runs past it: a length precondition, like load_pmdata's."""
    if record_end is not None and schedule_end(start, n_cycles) > record_end:
        raise ScheduleDoesNotFit(
            f"This schedule doesn't fit in the available record: {n_cycles} cycles run to "
            f"{schedule_end(start, n_cycles).date()}, and data ends {record_end.date()} "
            f"(at most {max_cycles(start, record_end)} cycles fit).")
    flips = np.random.default_rng(seed).integers(0, 2, size=n_cycles)
    cycles = [("on", "off") if f == 0 else ("off", "on") for f in flips]
    return Schedule(start=start, block_days=BLOCK_DAYS, washout_days=metric.onset_days, cycles=cycles)


def p_floor(n_cycles: int) -> float:
    return 2.0 / 2 ** n_cycles


def p_complete(coverage: float, n_cycles: int, block_days: int, washout_days: int) -> float:
    """Binomial fallback: P(at least MIN_CYCLES of n_cycles survive) if missing days
    were independent. Optimistic when missing days come in runs."""
    usable = block_days - washout_days
    p_block = float(binom.sf(MIN_USABLE_DAYS - 1, usable, coverage))
    return float(binom.sf(MIN_CYCLES - 1, n_cycles, p_block ** 2))


def can_resample(base: Baseline) -> bool:
    return base.missing is not None and len(base.missing) >= MIN_RESAMPLE_DAYS


def _missing_draws(base: Baseline, T: int, n_sims: int, rng, L: int | None = None) -> tuple[np.ndarray, str]:
    """(n_sims, T) missingness for simulated trials. Moving-block bootstrap (block
    length L) of the person's own before-period missingness indicator, so runs of
    missing days are kept; binomial (independent days) when the before period is too short."""
    m = base.missing
    if L is not None and can_resample(base):
        nb = -(-T // L)
        starts = rng.integers(0, len(m) - L + 1, size=(n_sims, nb))
        idx = (starts[..., None] + np.arange(L)).reshape(n_sims, nb * L)[:, :T]
        return m[idx], "block bootstrap"
    return rng.random((n_sims, T)) >= base.coverage, "binomial (optimistic: assumes missing days are independent)"


def _sign_patterns(n: int) -> np.ndarray:
    return np.array(list(itertools.product((1.0, -1.0), repeat=n)))      # (2**n, n)


def _randomisation_p(D: np.ndarray, use: np.ndarray) -> np.ndarray:
    """Exact two-sided randomisation test on paired differences, one row per trial.
    D: (N, n) cycle differences; use: (N, n) usable-cycle mask. Unusable cycles get
    d = 0, which repeats every usable-cycle pattern equally often, so the
    proportion over all 2**n patterns equals the proportion over the 2**k usable ones."""
    Dz = np.where(use, D, 0.0)
    P = _sign_patterns(D.shape[1])
    stats = Dz @ P.T                                   # (N, 2**n): sum of signed d
    obs = np.abs(Dz.sum(axis=1, keepdims=True))
    return np.mean(np.abs(stats) >= obs - 1e-9, axis=1)


def _on_ramp(block_days: int, onset_days: int) -> np.ndarray:
    """Fraction of the effect on each day of an on-block: ramps in over onset_days."""
    i = np.arange(block_days)
    return np.minimum(1.0, (i + 1) / max(onset_days, 1))


def _block_stats(X: np.ndarray, washout: int):
    """X: (..., n_cycles, 2, block_days). Means and counts over the post-washout days."""
    U = X[..., washout:]
    counts = np.sum(~np.isnan(U), axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.nansum(U, axis=-1) / counts
    return means, counts


@dataclass(frozen=True)
class ScheduleCheck:
    n_cycles: int
    allowed: bool
    reason: str
    p_floor: float
    p_complete: float
    p_complete_method: str             # "block bootstrap (minimum …)" | "binomial (optimistic …)"
    min_detectable: float | None      # baseline SDs; None = not reachable (or refused)
    coverage: float
    power: dict                        # effect size -> simulated power, up to the first that reaches 80%
    p_complete_by_block: dict = None   # block length -> p_complete (empty for the binomial fallback)


def _survival(miss: np.ndarray, n: int, B: int, w: int) -> np.ndarray:
    """Per simulated trial: did at least MIN_CYCLES cycles keep both blocks usable?"""
    _, counts = _block_stats(np.where(miss, np.nan, 0.0).reshape(len(miss), n, 2, B), w)
    return np.all(counts >= MIN_USABLE_DAYS, axis=2).sum(axis=1) >= MIN_CYCLES


def schedule_check(base: Baseline, n_cycles: int, metric: Metric, seed: int,
                   n_sims: int = N_SIMS) -> ScheduleCheck:
    """Run before the schedule. p_floor: can a result exist. p_complete: will the
    trial finish, resampling this person's own before-period gap pattern at each block
    length in MISS_BLOCK_LENGTHS and taking the minimum (conditional on the device
    continuing to record as it has been). min_detectable: smallest effect (baseline SDs)
    reaching 80% power, simulated from the fitted AR(1) with missingness drawn at the
    block length that gave the minimum, through the full block-mean + randomisation-test path."""
    pf = p_floor(n_cycles)
    rng = np.random.default_rng(seed)
    B, w, n = BLOCK_DAYS, metric.onset_days, n_cycles
    T = 2 * n * B
    by_L = {}
    if can_resample(base):
        draws = {}
        for L in MISS_BLOCK_LENGTHS:
            draws[L], how = _missing_draws(base, T, N_SIMS_COMPLETE, rng, L)
            by_L[L] = float(_survival(draws[L], n, B, w).mean())
        L_min = min(by_L, key=by_L.get)
        pc = by_L[L_min]
        miss = draws[L_min][:n_sims]                      # the power simulation uses N_SIMS of these
        how = f"block bootstrap (minimum over block lengths {', '.join(map(str, MISS_BLOCK_LENGTHS))} days)"
    else:
        miss, how = _missing_draws(base, T, n_sims, rng)
        pc = p_complete(base.coverage, n, B, w)
    if n_cycles < MIN_CYCLES:
        reason = (f"With {n_cycles} cycles the smallest possible p is {pf:.3g}, which can never be below "
                  f"{ALPHA}. At least {MIN_CYCLES} cycles are needed.")
        return ScheduleCheck(n_cycles, False, reason, pf, pc, how, None, base.coverage, {}, by_L)
    # Power can't exceed the share of simulated trials that complete. When that ceiling
    # is within two standard errors of the target, power plateaus at it and any crossing
    # is a draw from a flat region: report not reachable.
    ceiling = pc
    if ceiling < POWER_TARGET + 2 * np.sqrt(POWER_TARGET * (1 - POWER_TARGET) / n_sims):
        return ScheduleCheck(n_cycles, True, "", pf, pc, how, None, base.coverage, {"ceiling": ceiling}, by_L)
    phi = base.phi
    x = np.empty((n_sims, T))
    x[:, 0] = rng.standard_normal(n_sims)
    eps = rng.standard_normal((n_sims, T)) * np.sqrt(1 - phi ** 2)
    for t in range(1, T):
        x[:, t] = phi * x[:, t - 1] + eps[:, t]          # stationary AR(1), unit marginal SD
    x[miss] = np.nan
    on_first = rng.integers(0, 2, size=(n_sims, n)) == 0
    X = x.reshape(n_sims, n, 2, B)
    on_idx = np.where(on_first, 0, 1)                    # which block of each cycle is "on"
    on_mask = np.zeros((n_sims, n, 2, B))
    np.put_along_axis(on_mask, on_idx[..., None, None], _on_ramp(B, w)[None, None, None, :], axis=2)
    ws = _sign(metric.works_direction)
    power = {"ceiling": ceiling}
    md = None
    for delta in EFFECT_GRID:
        means, counts = _block_stats(X + ws * delta * on_mask, w)
        use = np.all(counts >= MIN_USABLE_DAYS, axis=2)
        on_m = np.take_along_axis(means, on_idx[..., None], axis=2)[..., 0]
        off_m = np.take_along_axis(means, (1 - on_idx)[..., None], axis=2)[..., 0]
        D = on_m - off_m
        p = _randomisation_p(D, use)
        mean_d = np.where(use, D, 0.0).sum(axis=1)
        hit = (use.sum(axis=1) >= MIN_CYCLES) & (p < ALPHA) & (np.sign(mean_d) == ws)
        power[float(delta)] = float(hit.mean())
        if hit.mean() >= POWER_TARGET:
            md = float(delta)
            break
    return ScheduleCheck(n_cycles, True, "", pf, pc, how, md, base.coverage, power, by_L)


@dataclass(frozen=True)
class EfficacyResult:
    cycles: pd.DataFrame        # cycle, order, on_mean, off_mean, d, on_days, off_days, usable
    n_usable: int
    mean_d: float               # units
    mean_d_sd: float            # before-SD units
    n_agree: int                # usable cycles whose d has the sign of the mean
    p_floor: float              # 2 / 2**n_usable
    p: float | None             # randomisation p; None if too few usable cycles
    verdict: str                # "not_enough" | "working" | "opposite" | "none"
    message: str


def schedule_blocks(schedule: Schedule) -> list[tuple[int, str, pd.Timestamp, pd.Timestamp]]:
    """(cycle, arm, first day, first usable day) for every block."""
    out = []
    for c, pair in enumerate(schedule.cycles):
        for b, arm in enumerate(pair):
            first = schedule.start + pd.Timedelta(days=(2 * c + b) * schedule.block_days)
            out.append((c + 1, arm, first, first + pd.Timedelta(days=schedule.washout_days)))
    return out


def efficacy(subject: Subject, schedule: Schedule, metric: Metric) -> EfficacyResult:
    s = subject.series
    rows = []
    for c, pair in enumerate(schedule.cycles):
        stats = {}
        for b, arm in enumerate(pair):
            first = schedule.start + pd.Timedelta(days=(2 * c + b) * schedule.block_days)
            usable_days = s[(s.index >= first + pd.Timedelta(days=schedule.washout_days)) &
                            (s.index < first + pd.Timedelta(days=schedule.block_days))].dropna()
            ok = len(usable_days) >= MIN_USABLE_DAYS
            stats[arm] = (float(usable_days.mean()) if ok else np.nan, len(usable_days))
        usable = not (np.isnan(stats["on"][0]) or np.isnan(stats["off"][0]))
        rows.append({"cycle": c + 1, "order": "/".join(pair), "on_mean": stats["on"][0],
                     "off_mean": stats["off"][0],
                     "d": stats["on"][0] - stats["off"][0] if usable else np.nan,
                     "on_days": stats["on"][1], "off_days": stats["off"][1], "usable": usable})
    df = pd.DataFrame(rows)
    d = df.loc[df["usable"], "d"].to_numpy()
    k = len(d)
    sd = float(np.std(before_values(subject).to_numpy(), ddof=1))
    mean_d = float(d.mean()) if k else np.nan
    n_agree = int(np.sum(np.sign(d) == np.sign(mean_d))) if k else 0
    pf = p_floor(k) if k else 1.0
    if k < MIN_CYCLES:
        return EfficacyResult(df, k, mean_d, mean_d / sd if k else np.nan, n_agree, pf, None, "not_enough",
                              f"Not enough usable cycles. With {k} cycles the smallest possible p is {pf:.3g}.")
    p = float(_randomisation_p(d[None, :], np.ones((1, k), dtype=bool))[0])
    works = np.sign(mean_d) == _sign(metric.works_direction)
    if p < ALPHA and works:
        verdict, msg = "working", "Consistent effect: working."
    elif p < ALPHA:
        verdict, msg = "opposite", "Consistent effect: opposite to intended."
    else:
        verdict, msg = "none", "No consistent effect detected."
    return EfficacyResult(df, k, mean_d, mean_d / sd, n_agree, pf, p, verdict, msg)


def no_effect_note(min_detectable: float | None) -> str:
    size = f"about {min_detectable:.1f} SDs or larger" if min_detectable is not None else \
        "no size reliably (80% power is not reachable at this person's coverage)"
    return f"This is not the same as no effect. This schedule reliably detects effects of {size}."


# 5.6 Planting
def plant_harm(subject: Subject, metric: Metric, delta_sd: float = 2.0,
               start_offset: int = 5, ramp_days: int = 3) -> Subject:
    """Sustained shift of delta_sd before-SDs in harm_direction from
    start + start_offset, ramping in linearly over ramp_days."""
    sd = float(np.std(before_values(subject).to_numpy(), ddof=1))
    s = subject.series
    t0 = subject.start + pd.Timedelta(days=start_offset)
    days_in = (s.index - t0).days.to_numpy()
    frac = np.clip((days_in + 1) / ramp_days, 0.0, 1.0)
    frac[days_in < 0] = 0.0
    shifted = s + _sign(metric.harm_direction) * delta_sd * sd * frac   # NaN stays NaN
    return replace(subject, series=shifted, simulated=True)


def plant_effect(subject: Subject, schedule: Schedule, metric: Metric, delta_sd: float = 1.0) -> Subject:
    """Shift of delta_sd before-SDs in works_direction during on-blocks only,
    ramping in over onset_days at the start of each on-block."""
    sd = float(np.std(before_values(subject).to_numpy(), ddof=1))
    s = subject.series
    shift = np.zeros(len(s))
    ramp = _on_ramp(schedule.block_days, metric.onset_days)
    for c, pair in enumerate(schedule.cycles):
        b = pair.index("on")
        first = schedule.start + pd.Timedelta(days=(2 * c + b) * schedule.block_days)
        day = (s.index - first).days.to_numpy()
        inside = (day >= 0) & (day < schedule.block_days)
        shift[inside] = ramp[day[inside]]
    shifted = s + _sign(metric.works_direction) * delta_sd * sd * shift    # NaN stays NaN
    return replace(subject, series=shifted, simulated=True)
