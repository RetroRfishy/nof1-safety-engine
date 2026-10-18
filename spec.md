# Build spec: personal compound monitor

**What it is:** a tool for a medical officer. For one person taking a compound, it tracks a daily wearable metric before and after the compound starts and answers three questions:

1. **What changed?** Their numbers before versus after.
2. **Is it harming them?** A sustained move away from their own normal, flagged automatically.
3. **Is it working for them?** In an on/off schedule, were their on-weeks consistently different from their off-weeks?

A roster shows everyone on the compound at once.

**Any daily metric works.** The method fits each person's normal from their own data, so any daily metric a wearable records can be used: resting heart rate, HRV, sleep duration, and so on. The demo uses resting heart rate because that is the public data available.

**Rules:**
- Every method here is standard; the appendix names the source of each.
- Do not substitute "improvements."
- Do not change a method, constant or check threshold to make a result look better.
- If something seems wrong, stop and ask.

Read the whole file. Follow the build order in section 8 and report at each checkpoint.

---

## 1. How it is used

1. **Set up the compound.** The officer names it and chooses the metric it should affect, the direction that counts as working, and the direction that counts as harm. These are fixed before any data is looked at.
2. **Start a person.** Pick the person and set their start date.
3. **Optionally, run an on/off trial.** The app proposes a randomized weekly schedule, refuses any schedule that cannot produce a result, and states the smallest effect the schedule could reliably detect (5.5).
4. **Data comes from the watch the person already wears.** Nothing is asked of them.
5. **During the course,** the roster shows status, and the Person page shows before/after and any harm alarm.
6. **At the end of a trial,** the Person page says whether the compound did anything for that person.

---

## 2. Data

PMData (Thambawita et al., Simula; CC BY 4.0; osf.io/vx4bk). Ask the user for the root path. File: `<root>/pXX/fitbit/resting_heart_rate.json`, with XX from 01 to 16.

**Reading it**
- Inspect first with `ls -lh` and `head -c 1500`.
- Never open `heart_rate.json` or `steps.json`. They are about 120 MB each and not needed.
- Entry format: `{"dateTime": "2020-01-02 00:00:00", "value": {"date": "01/02/20", "value": 45.67, "error": 12.49}}`. The value is `entry["value"]["value"]`.
- Missing days: a missing day is stored as `value: 0.0, error: 0.0, date: null`. Convert it to NaN.
- Complete index: reindex to a complete daily range, so absent dates become NaN too.
- Never interpolate.

**Facts**
- p12 and p13 have no file. p04 has 35 days. Usable cohort is about 13.
- p03 is 34% missing and p10 is 26% missing.
- Series run to about 150 days. A day-60 start leaves about 90.
- The demo trial (6 weekly cycles = 84 days) needs at least 144 days. Choose the demo subject accordingly (section 8). Print how many subjects clear 144 days; it will be a small number, and it caps check 10. *(Measured: 8 by series length — p01, p03, p06, p07, p09, p10, p15, p16; p08 has 143 days. Against the last day with data, which is what make_schedule checks, 7: p10's data ends 2020-02-28.)*
- Skip any subject that can't support the requested start date or schedule, and print the reason.

**Labelling**
- Nobody in PMData took anything. The start date and schedule are demo choices, and any effect is planted.
- Persistent banner: *"Demo: real heart-rate data from volunteers who took nothing. Start date, schedule and any effect shown are simulated."*
- Every planted effect shows "Simulated effect" beside every status it affects.
- Because nobody took anything, real post-start data are true nulls. That is what makes checks 6 and 10 measurements.

---

## 3. Files

```
subject.py    Subject, Metric, Schedule
loaders.py    list_pmdata(), load_pmdata()
engine.py     before_after, fit_baseline, residuals, cusum,
              harm_limit, reference_library, reference_limit, harm_monitor,
              make_schedule, schedule_check, efficacy,
              plant_harm, plant_effect
checks.py     section 7, runnable as a script
app.py        Streamlit UI
requirements.txt
README.md
spec.md       this file; the single record of the spec and every decision made during the build
```

engine.py is pure functions: no plotting, no Streamlit, no file I/O. Every random draw takes a seed. Nothing in engine.py refers to heart rate. It sees a series and a Metric.

---

## 4. Data contract

```python
@dataclass(frozen=True)
class Metric:
    name: str                   # "resting_hr"
    unit: str                   # "bpm"
    works_direction: str        # "up" | "down": the change that means the compound works
    harm_direction: str         # "up" | "down": the change that means harm
    onset_days: int             # days for the effect to start/stop; sets washout

@dataclass(frozen=True)
class Subject:
    subject_id: str
    series: pd.Series           # float, complete daily DatetimeIndex, NaN = missing
    start: pd.Timestamp         # first day on compound; earlier days = before
    simulated: bool = False

@dataclass(frozen=True)
class Schedule:
    start: pd.Timestamp
    block_days: int             # 7
    washout_days: int           # = metric.onset_days, excluded at the start of every block
    cycles: list[tuple[str, str]]   # e.g. [("on","off"), ("off","on"), ...]
```

**Rules:**
- Complete daily index. Missing days are NaN rows, never absent rows.
- `start` is always passed in, never inferred.
- `simulated` must survive every transformation.

**Demo metric:** `Metric("resting_hr", "bpm", works_direction="down", harm_direction="up", onset_days=2)`. A recovery-type compound: working means resting HR falls, harm means it rises.

**Loader functions:**
```python
def list_pmdata(root: str) -> list[str]
def load_pmdata(root: str, pid: str, start_day: int = 60) -> Subject
```
`start_day` is an offset from the first date. Raise a clear error if the series is too short. The roster catches the error, skips that subject and lists the reason. `list_pmdata` lists every pXX folder; a folder with no resting-HR file makes `load_pmdata` raise that as the reason.

---

## 5. Engine

### 5.1 Constants

| Name | Value | Note |
|---|---|---|
| FP | 0.05 | Harm false-alarm rate per monitoring course |
| HORIZON | 30 days | Choice. Length of one harm-monitoring course |
| N_SIMS | 1000 | |
| DELTA | 2.0 | Choice. Shift the harm detector is tuned for, in baseline SDs |
| MIN_BEFORE | 14 days | Choice. Refuse below this |
| ALPHA | 0.05 | Efficacy significance level |
| BLOCK_DAYS | 7 | Choice. Weekly blocks. Valid only for compounds whose effect starts and stops within onset_days |
| MIN_CYCLES | 6 | Not a choice. Follows from ALPHA (5.5) |
| MIN_USABLE_DAYS | 2 | Measured, not a free choice. At 3, trials on high-missingness subjects fail to complete (5.5) |
| DEFAULT_CYCLES | 8 | Choice. MIN_CYCLES + 2: two cycles of slack (5.5). The demo uses 6 because PMData can't fit 8 |
| MISS_BLOCK_LENGTHS | 3, 5, 7, 14 | Choice. Block lengths tried when resampling the missingness indicator; p_complete is the minimum across them (5.5) |
| N_SIMS_COMPLETE | 10000 | 10 × N_SIMS for p_complete: Monte Carlo noise about ±0.01 near the 0.8 warning |
| MIN_RESAMPLE_DAYS | 28 | Choice. Shorter before periods fall back to the binomial p_complete, flagged optimistic |
| REF_BASELINE_DAYS | 60 | The baseline re-fitted for every reference-library window (5.4). Same length as the check 6 baselines |

### 5.2 before_after(subject) -> BeforeAfter

**Inputs:** before = non-NaN values before start; after = non-NaN values from start onward, or within the selected window.

**Returns:** before mean, SD and N; after mean, SD and N; change = after mean − before mean; standardised change = change ÷ before SD.

**Caveat,** shown in one line beneath the result: *"A before/after difference alone can't show the compound caused it; season, training or illness could. The on/off trial is what attributes change to the compound."*

### 5.3 fit_baseline(subject) and residuals(values, mu, phi)

Computed from the before values only:
- `mu` and `sd_x`.
- `phi`: lag-1 autocorrelation over consecutive present days, clipped to [−0.95, 0.95]. Exactly: the mean of (x_{t−1} − mu)(x_t − mu) over pairs of consecutive days that both have data, divided by the mean of (x − mu)² over all present before-days. Using only consecutive present pairs is the compromise for gappy series.
- Raw residuals: `residuals` returns e_t = (x_t − mu) − phi·(x_{t−1} − mu), NaN if either day is missing.
- `sd_e` = SD of the raw residuals (ddof=1).
- Standardised residuals: z_t = e_t / sd_e.

Order matters: raw residuals first, then sd_e, then standardise.

**Why:** the metric today predicts tomorrow. Monitoring methods assume independent observations, and monitoring the residuals of a fitted AR(1) model is the standard remedy (the residual control chart).

### 5.4 Harm: cusum, harm_limit, reference_limit, harm_monitor

**The detector:**
```
s   = +1 if metric.harm_direction == "up" else −1
k   = 0.5 · DELTA · sqrt((1 − phi)/(1 + phi))      # per person
S_0 = 0
S_t = max(0, S_{t−1} + s·z_t − k)                  # z_t NaN → S_t = S_{t−1}, step not counted
alarm when S_t > h
```

**Scored steps.** n_valid is the number of non-NaN residuals in the window being monitored. Missing days produce no step, so the limit is calibrated on n_valid, not on calendar length.

**`harm_limit(base, n_valid, seed, metric)`. The comparator, not the default.** It takes the Metric because the CUSUM is one-sided, so the distribution of its maximum depends on which tail is monitored.
1. Draw n_valid standardised baseline residuals with replacement.
2. Run the CUSUM in metric.harm_direction and record its maximum.
3. Repeat N_SIMS times. h = the 95th percentile.

Bootstrap calibration from the person's own baseline. Needs no reference group, which is its only advantage. Measured: 14.0% false alarms (6 of 43 real no-exposure windows) against a nominal 5%, because the baseline-period null contains none of the drift that appears after the start.

**`reference_limit(subject, others, n_valid, k, metric, seed)`. This is the default and the number the screen states.**

For every usable subject except this one, build a library of null CUSUM maxima:
1. Slide a window through that subject's post-start series in valid-step space: each library window is the next n_valid valid residuals, advancing one valid step at a time. A window starts only on a day whose own residual is valid.
2. Re-fit that subject's baseline before every window, from the REF_BASELINE_DAYS (60) days ending the day before that window's own start. Skip the window if that stretch has fewer than MIN_BEFORE present days. A library window must be built the way a real monitoring course runs.
3. Score it with this subject's k and record the maximum.

h = the 95th percentile across the library. Return h, the window count and the subject count. Report both: windows overlap heavily, so the effective n is the subject count.

`others` is either a list of Subjects or the library already built from them (`reference_library(others)`). The library doesn't depend on which subject is being monitored, so it is built once and the monitored subject is left out by ID. reference_limit makes no random draws; `seed` is kept for the interface.

Do not fix the baseline at the subject's original before period. Sliding one step at a time against a single fixed baseline scores late windows against a baseline up to 90 days stale. And because most PMData subjects share a start date, the earliest windows all cover the same calendar stretch, so that shared drift is replicated into hundreds of overlapping windows until the 95th percentile lands on it. Measured: h for p01 rises from 7.38 to 16.9 and detection falls to 21%. Re-fitting per window removes it.

Library size is not what sets h. Going from ~60 windows to ~500 moves p01's h from 8.9 to 7.38. The larger library is worth building because it raises the subject count behind each limit from 8 to 11 or 12, so the subjects with the cleanest data no longer get the thinnest library. It does not buy detection, and nothing else will: do not keep looking for a configuration that lowers h.

**h is not stable at this cohort size.** It is a step function of which subjects are in the library. Of the 30 largest library maxima behind p01's limit, 22 are p08's, and the 95th percentile sits in a gap between 8.52 and 7.41. A 27-window change to the library moved p01's h by 15%. Measured (check 6, each library subject left out in turn, day-60 windows): h moves by 10–46% of its value across subjects (median 37%). p01's h ranges from 5.75 (without p08) to 8.68 (without p06). Leaving out p08 alone lowers h for 10 of 13 subjects; p14 goes from 13.39 to 8.84. The mechanism is self-consistent: p08 trips its own reference limit, which excludes p08, and inflates everyone else's, because p08's drift sits in all their libraries. That is drift behaving as drift should, not noise. The false-alarm rate moves much less than h: re-running the 43 check 6 windows with each library subject left out gives 7.0% (3/43) for 12 of 13 removals and 11.6% (5/43) without p08. The screens keep "about 7%"; the range goes in "How this works" and the README. Don't fix it; it is the measured reason a deployment needs a reference cohort of 50–100 people, not 13.

**`harm_monitor(subject, base, others, metric, horizon, method="reference") -> HarmResult`:**
- Take the window from start, of length min(horizon, available days).
- Count n_valid, calibrate h on it by method (`"reference"` or `"own_baseline"`), and score the window.
- Returns: the S series, h, method, n_valid, the calendar length, alarm_date (first date with S > h, or None), and `record`, the JSON line for the app to write. engine.py does no file I/O.

**Why no drift correction.** Check 11 measures no common drift on PMData: mean z has a median of −0.09 across 43 windows (IQR −0.43 to +0.10, range −0.85 to +0.69). A population drift template subtracts a shared mean and would change nothing here; the reference-group limit is used because it absorbs spread.

**Detection is heterogeneous.** Splitting check 7's planted-harm windows by their check 11 mean z: drift toward harm caught 14 of 15 (median day 11), drift away 9 of 28 (median day 21). State this beside the 53% in the README. The split uses the same days the detector scores, so it describes, it doesn't predict. Read the 53% as a blend of about 93% and 32%, and which applies to a given person can't be known in advance. **Do not turn mean z into a per-person confidence qualifier** ("this window drifted away from the alarm direction, so harm would be harder to see"). A window drifting toward harm and a window that contains harm are the same observation: mean z > 0 cannot tell "this person's baseline drifted up" from "this person is being harmed". The qualifier collapses into the detector's own output and carries no extra information.

**Measured next step, not acted on:** check 11's leftover lag-1 of z (median +0.12) corroborates phi underestimated from 60-day baselines. A bias-corrected phi is the next step; it would mean re-running every check, and the screens already state the measured 7%.

**Measured capability,** to state on the Person page and in the README. Measured on 43 real no-exposure windows, with a 2 SD harm planted in each for the detection rows:

| | reference (default) | own baseline (comparator) |
|---|---|---|
| False alarms | 7.0% (3/43) | 14.0% (6/43) |
| 2 SD harm caught within 30 days | 53% (23/43) | 79% (34/43) |
| Days to alarm, median (IQR) | 13 (10–22) | 11 (8–17) |

The own-baseline limit catches more because its h is lower, which is the same fact as its 14% false-alarm rate, not a separate benefit. 53% of 2 SD harms within 30 days, median day 13, is the stated capability of this arm — the harm-side counterpart of min_detectable in 5.5. A monitor that says what it misses is usable; one that claims 5% and delivers 14% is not.

**README must state:**
- With high phi, whitening also weakens a real shift. That is the price of a correct false-alarm rate.
- The reference limit needs a group monitored with no exposure, so a new deployment has to build one first.
- Every PMData subject was measured over the same Nov–Mar window, so seasonal drift shared across the cohort sits inside the null rather than being caught by it. The measured 7% is optimistic for a cohort spread across different months. This cannot be fixed with this dataset. State it next to the deployment-cost line above.
- The stated capability above, with the 2 SD figure and the median day.
- The leave-one-out instability of h above (10–46% of h, median 37%; p01 5.75–8.68; p08 alone sets most of the tail). This is the concrete, measured reason a real deployment needs a reference cohort of 50 to 100 people, not 13.

### 5.5 Efficacy: make_schedule, schedule_check, efficacy

**Design: the n-of-1 crossover trial.**
- Paired on/off blocks of BLOCK_DAYS.
- The order within each cycle is randomised by coin flip.
- The first washout_days of every block are excluded, to limit carry-over from the previous block.

**`make_schedule(start, n_cycles, metric, seed, record_end) -> Schedule`:** builds that schedule. It first checks a length precondition, the same check load_pmdata makes one level up: if start + n_cycles × 2 × BLOCK_DAYS runs past `record_end` (the last day the record has data), it refuses with "This schedule doesn't fit in the available record" and says how many cycles do fit. This is known with certainty, not probability, so it is never folded into p_complete. For PMData at a day-60 start, 6 cycles fit for 7 of 13 subjects. p10 is rejected because its device records nothing after 2020-02-28, so at most 4 cycles fit.

**Cycles.** The deployment default is DEFAULT_CYCLES = MIN_CYCLES + 2 = 8. With exactly MIN_CYCLES there is no slack: completion requires every cycle, and one bad week ends a six-week trial. p03 shows this on real data: 5 of 6 cycles usable on every schedule, so no verdict, where 8 scheduled would have left 7. PMData can't fit 8 cycles after a day-60 start, so the demo uses 6, and p03 honestly fails to complete. Showing the tool decline to give a verdict is part of the demo.

**`schedule_check(base, n_cycles, metric, seed) -> ScheduleCheck`.** Run before the schedule, not after. Returns three numbers, all shown in the sidebar:

**1. p_floor** = 2 / 2**n_cycles. Randomising the order within each cycle means each cycle's sign can flip under the null. With n cycles there are 2^n equally likely sign patterns, so no result of any size can reach a p below 2/2^n. Values: 3 cycles → 0.25; 5 → 0.0625; 6 → 0.031. Below MIN_CYCLES the app refuses the schedule and says why, before it runs.

**2. p_complete** = the chance the trial produces a verdict at all, given this person's missingness. A block with fewer than MIN_USABLE_DAYS valid days is dropped and a cycle needs both its blocks, and:

```
p_complete = P(at least MIN_CYCLES of n_cycles survive)
```

Not the chance all n_cycles survive. The verdict needs MIN_CYCLES usable cycles, not every scheduled one: schedule 10, lose 3, and the randomisation test runs on 7 with p_floor = 2/2^7. The all-must-survive version falls as cycles are added, which would make "consider more cycles" actively wrong advice.

**How it is computed: resample, don't model.** Missing wearable days come in runs (a week without charging, a device left in a drawer). The binomial over independent days can't see runs, and on PMData it badly overstates completion: p03's binomial value is 0.06–0.90 depending on which coverage it uses, yet p03 completed 0 of 20 real schedules. So take the person's own before-period missingness indicator, draw contiguous blocks from it (moving-block bootstrap), lay them over the planned schedule, and count how often at least MIN_CYCLES survive. Do this at each block length in MISS_BLOCK_LENGTHS and report the minimum; "How this works" shows all of them. The block length is a free choice that can move the gate by 30 points, and a warning should fire if any defensible assumption says it should. Taking the conservative end of a sensitivity analysis already run is the opposite of tuning. Runs are kept because runs are what is resampled, and each person's own gap pattern is used rather than an assumed one. This resampling is new to this spec: harm_limit's bootstrap resamples single residuals, not blocks. See the appendix.

The binomial over the before-period coverage stays as the fallback when the before period is shorter than MIN_RESAMPLE_DAYS, flagged on screen as optimistic.

p_complete is conditional on the device continuing to record as it has been. A device that stops is a length problem (make_schedule), not a probability.

Warn below 0.8, wording: *"This person's data coverage gives the trial an X% chance of producing a verdict at all. Consider more cycles."*

**One condition on dropping cycles:** the randomisation null requires that which cycles survive is independent of the on/off assignment. Missing wearable days are normally about charging and not wearing, so this holds; but if the compound itself caused people to stop wearing the device, it would not. State it in the README.

**3. min_detectable** = the smallest effect, in baseline SDs, this schedule detects at least 80% of the time. Simulate from the fitted AR(1) at a grid of effect sizes (0.1 SD steps), with missingness drawn at the block length that gave the minimum p_complete, run the full block-mean and randomisation-test path, and report the smallest size reaching 80%. Power can never exceed the share of simulated trials that complete. When that ceiling is within two standard errors of 80% (below 0.825 at N_SIMS = 1000), power plateaus at the ceiling and any crossing of 80% is a draw from a flat region, so report "not reachable" rather than a number.

p_floor says whether a result is possible, p_complete whether the trial will finish, min_detectable whether a real effect would be found.

**Measured per subject** (6 cycles, 7-day blocks, 2-day washout, MIN_USABLE_DAYS = 2, start day 60, seed fixed; check 8):

| subject | phi | before coverage | p_complete (min over block lengths) | binomial | min_detectable | real schedules with a verdict |
|---|---|---|---|---|---|---|
| p01 | 0.63 | 100% | 1.00 | 1.00 | 1.8 SD | 20/20 |
| p02 | 0.62 | 97% | 1.00 | 1.00 | 1.8 | doesn't fit |
| p03 | 0.67 | 48% | 0.00 | 0.06 | not reachable | 0/20 |
| p05 | 0.92 | 88% | 0.64 | 0.99 | not reachable | doesn't fit |
| p06 | 0.74 | 98% | 1.00 | 1.00 | 1.8 | 20/20 |
| p07 | 0.80 | 97% | 1.00 | 1.00 | 1.9 | 20/20 |
| p08 | 0.58 | 93% | 0.97 | 1.00 | 1.8 | doesn't fit (5 cycles fit) |
| p09 | 0.85 | 98% | 1.00 | 1.00 | 1.8 | 20/20 |
| p10 | 0.73 | 93% | 0.98 | 1.00 | 1.9 | doesn't fit (data ends 2020-02-28) |
| p11 | 0.82 | 100% | 1.00 | 1.00 | 1.8 | doesn't fit |
| p14 | 0.90 | 88% | 0.93 | 0.99 | 1.7 | doesn't fit (5 cycles fit) |
| p15 | 0.81 | 98% | 1.00 | 1.00 | 1.8 | 20/20 |
| p16 | 0.78 | 100% | 1.00 | 1.00 | 1.9 | 20/20 |

With complete data, min_detectable is 1.7–1.9 SD. phi matters mildly, in the helpful direction: the lowest value, 1.7, is p14 (phi 0.90), and p05 (phi 0.92) would be lower still if its completion ceiling allowed. min_detectable carries about ±0.1 SD of Monte Carlo noise (0.1 SD grid, N_SIMS trials). Higher autocorrelation smooths within a block and also correlates adjacent blocks, and the two partly cancel in a paired difference. Missingness, through runs of missing days, is what limits this arm: it caps power at the completion ceiling. p03 cannot produce a verdict at all, and p05's before-period gap pattern puts it below the 80% ceiling.

p_complete by block length (check 8) for p05 is 0.90 / 0.79 / 0.64 / 0.77 at 3 / 5 / 7 / 14 days, so p05 reports 0.64 and warns. The non-monotonicity is a property of the data, not noise in the estimate: p05's gaps have structure at a particular scale rather than being random runs, and 7-day blocks happen to reproduce it. p03 is 0.00–0.01 at every block length. p_complete uses N_SIMS_COMPLETE = 10 × N_SIMS replicates (it is only counting usable cycles), so its Monte Carlo noise is about ±0.01 and a value near the 0.8 warning line no longer flips with the seed.

MIN_USABLE_DAYS = 2 rather than 3 is what makes this workable. Under the binomial, p_complete at 3 is 0.49 / 0.24 / 0.05 at 20 / 26 / 34% missing, so most trials never return a verdict. Dropping to 2 costs nothing on complete data (identical power) and leaves the null rate unchanged at about 3%.

Report these numbers; do not tune the design to improve them. They are a real limit of a 6-week trial on a daily metric, and the README states them alongside the whitening note.

**Calendar.** 6 cycles of 7 days on and 7 days off take 84 days; the deployment default of 8 takes 112. With washout_days = 2, each block keeps 5 usable days.
- 6 cycles fit PMData after a day-60 start for 7 subjects; 8 cycles fit none.
- The block length is valid only for fast-acting compounds. Slower compounds need longer blocks and longer trials. The README states this.

**`efficacy(subject, schedule, metric) -> EfficacyResult`:**
1. Average each block over its non-NaN days after the washout. A block with fewer than MIN_USABLE_DAYS is unusable.
2. Difference per cycle, for each cycle with two usable blocks: d = on mean − off mean.
3. Exact randomisation test, two-sided. Enumerate all 2^n sign patterns of the d values, compute the mean under each, and report the proportion at least as extreme in absolute value as observed.
4. Return: each cycle's d; the mean d, in units and in before-SD units; how many cycles agree in direction; the p floor; the randomisation p.
5. Verdict:
   - Fewer than MIN_CYCLES usable cycles → "Not enough usable cycles", with the reason: "With N cycles the smallest possible p is X."
   - p < ALPHA and the mean d is in metric.works_direction → "Consistent effect: working."
   - p < ALPHA and the mean d is in the opposite direction → "Consistent effect: opposite to intended." Show this alongside the harm status.
   - Otherwise → "No consistent effect detected." Add: "This is not the same as no effect. This schedule reliably detects effects of about min_detectable SDs or larger."

**Why block means:** individual days are autocorrelated. Treating them as independent inflates false positives, so n-of-1 analyses compare period averages.

**Note on the p grid.** Because the test enumerates 2^n patterns, the attainable p values are 2/2^n, 4/2^n, …. At 6 cycles, p < ALPHA means p = 0.031 exactly. The test is therefore conservative: its true false-positive rate is 3.1%, not 5%. Check 10 measures this and should land near 3%, not near ALPHA.

The harm detector keeps running during a trial, on on-blocks and off-blocks alike.

### 5.6 Planting

- `plant_harm(subject, metric, delta_sd=2.0, start_offset=5, ramp_days=3)`: a sustained shift in harm_direction from start + start_offset, ramping in linearly. Sets simulated=True.
- `plant_effect(subject, schedule, metric, delta_sd=1.0)`: shifts values in works_direction only during on-blocks, ramping in over onset_days at the start of each on-block. Sets simulated=True.

These are the only ways an effect enters PMData. The two demo controls are mutually exclusive in the UI: works_direction and harm_direction are opposite, so planting both would make them partially cancel during on-blocks and produce a confusing demo.

---

## 6. App

**Persistent banner:** the demo statement from section 2.

**Sidebar:**
- PMData root.
- Compound name (default "Compound A").
- Metric panel. Shows the metric's name, unit, works-direction, harm-direction and onset. Demo values are fixed; the panel is shown so the "any metric" design is visible.
- Person. Default p01; p03 is the second subject the presenter clicks into.
- Start day (default 60).
- Harm limit: reference group is the default. The own-baseline limit sits under a settings expander, labelled as the comparator with its measured 14% beside it.
- Mode: Monitor, or Monitor + on/off trial. In trial mode:
  - number of cycles (default DEFAULT_CYCLES, or the most that fit the record if fewer), with p_floor, p_complete and min_detectable shown live, a refusal below 6 cycles, a refusal when the schedule doesn't fit the record, a warning when p_complete is below 0.8, and the binomial fallback flagged as optimistic;
  - a "randomise schedule" button.
- Demo controls: plant harm, plant effect (mutually exclusive).

**Page 1: Roster** (opens first)

| Column | Meaning |
|---|---|
| Person | |
| Day | Days since start |
| Change | Standardised change, before to after |
| Harm status | Watching, or Alarm with date |
| Efficacy | Verdict, if a trial is set |
| Coverage | % of days with data |
| Note | "Simulated effect" where planted |

- Sort: Alarm first.
- Summary line, counting only unplanted people: the alarm count, the count expected with nothing wrong (people × the measured false-alarm rate for the limit in use, or FP if that is higher), and the binomial chance of seeing that many or more. Example: "2 alarms out of 13 people. With nothing wrong you'd expect about 1, and 2 or more happens about 1 time in 4. This is not more than chance." When that chance falls below 5%, the closing sentence becomes "That is more than chance would usually produce; review these alarms together." Alarms are never labelled true or false on the roster: a real deployment can't know which are false, and the banner covers the demo.

**Page 2: Person**

*What changed.* Three tiles: Before (mean ± SD, N days), After (mean ± SD, N days), Change (in units and in SD units). The 5.2 caveat sits in one line beneath them.

*Chart.*
- The metric over time.
- A shaded band at before mean ± 2 SD.
- A vertical line at the start date.
- In trial mode, on-blocks shaded and washout days hatched.
- The harm alarm date marked, if any.

*Is it harming them?*
- One sentence: "Watching, no alarm" or "Alarm on <date>: <metric> has stayed <above/below> this person's normal for several days. Review recommended."
- Below it, two lines: the measured false-alarm rate for the limit in use (7% for the reference limit), and the stated capability: *"Catches a 2 SD sustained shift about half the time within 30 days, typically by day 13. Smaller or slower shifts may not be caught."*
- The CUSUM chart goes in a collapsed expander.

*Is it working for them?* (trial mode only)
- A table of cycle, on mean, off mean, and difference.
- The verdict sentence.
- p_floor, p_complete and min_detectable, always shown.

*"How this works" expander:* mu, SD, phi, k, h, n_valid, the appendix methods, and how firm the reference limit is (the leave-one-out spread of h and the false-alarm range). Statistics like these stay off the main screens.

**Out of scope:** login, upload, styling beyond defaults.

**Output.** The app writes each alarm as a JSON line (subject, date, metric, h, n_valid, FP, simulated), taken from the `record` that harm_monitor returns. In deployment, alarms would go into an existing clinical record. No integration is built.

---

## 7. Acceptance checks (`python checks.py`)

Fixed seeds. Print PASS or FAIL with the numbers.

1. **Loader.** Complete daily index, no zeros, and NaN count = zero entries + absent dates.
2. **Before/after.** On a series with a known +5 shift at start and SD 2: change ≈ 5, standardised change ≈ 2.5.
3. **Baseline model.** On a generated 500-day AR(1) with phi = 0.8: fitted phi within ±0.08, residual lag-1 autocorrelation within ±0.1.
4. **CUSUM at rest.** All-zero z gives all-zero S.
5. **Limit self-check.** On the demo subject, 1000 fresh bootstrap windows alarm at 3.5–6.5%. This checks the code, not real-world validity.
6. **Harm false alarms, real data.** Windows at start day 60, 74, 88 and 102, each with its own 60-day baseline ending the day before, each leaving HORIZON days after. 43 windows from 13 subjects. Run both limits. Report "A of B windows (X%), Wilson 95% CI" plus the subject count, and state that windows within a person overlap so the interval is optimistic. Nobody took anything, so these are all false alarms. Report as measured. Do not tune.
   - Measured: own baseline 14.0% (6/43), reference 7.0% (3/43). A build that reproduces these is correct.
   - Day-60 windows alone give 4 of 13 under the own-baseline limit. That is a single noisy draw, not a separate result.
   - Also report reference-limit stability: for each subject's day-60 window, recompute h with each other library subject left out in turn, and report the range and which subject's removal gives each end. Measured: spread 10–46% of h (median 37%); p01 5.75–8.68. Then re-run all 43 windows with each library subject left out and report the false-alarm rate range. Measured: 7.0% for 12 of 13 removals, 11.6% without p08. Report as measured.
7. **Harm detection.** Plant a 2 SD harm in each of the 43 windows and report, for both limits, the fraction alarming within HORIZON and the median and IQR of days to alarm.
   - Measured: reference 53% (23/43), median day 13 (IQR 10–22); own baseline 79% (34/43), median day 11.
   - plant_harm(delta_sd=2) on the demo subject should alarm under the reference limit. If it doesn't, report it with the subject's position in the distribution above. Do not raise DELTA and do not switch to the own-baseline limit to make it pass.
8. **Efficacy gates.** schedule_check returns p_floor of 0.25 at 3 cycles, 0.0625 at 5, and below ALPHA at 6. Then two assertions, both deterministic given the data: the schedule-length precondition rejects p10 at 6 cycles from day 60, and the block-bootstrap p_complete for p03 comes out below 0.8. Also print, for every usable subject at 6 cycles, p_complete (block and binomial), min_detectable, and the share of 20 real schedules that reached a verdict, plus p_complete at each block length for subjects where they differ. Measured: see the 5.5 table.
9. **Efficacy wiring, and what it measures.** On the demo subject, a 6-cycle weekly schedule with plant_effect(delta_sd=1.0). Report the randomisation p, every cycle difference, how many cycles agree in direction, and the sign of the mean difference.
   - Passes if the mean difference is in works_direction and the cycle differences are the right size: the machinery is wired correctly.
   - Reaching p < ALPHA on one seeded run is about a 35% proposition at delta_sd = 1.0 for the demo subject (simulated power 0.35; less with missingness), because 1 SD sits below its min_detectable of 1.8 SD. A run that does not reach significance is an expected outcome, not a bug, and is reported as such. Do not raise delta_sd to produce a significant demo; the honest statement is that a 1 SD effect is below what a 6-week weekly trial resolves.
10. **Efficacy false positives, real data.** For every subject whose 6-cycle schedule from day 60 fits the record (7 subjects), run 20 random 6-cycle schedules (different seeds) with nothing planted. Report how many of all runs say "Consistent effect." Nobody took anything, so these are false positives.
    - Expect about 3%, not ALPHA. At 6 cycles the attainable p grid makes p < 0.05 mean p = 2/64 = 0.031, so the test is conservative by construction (5.5).
    - The 20 schedules per subject run on the same data and are not independent, so the effective n is the subject count, not the run count. Report both. Few subjects clear 144 days, so this interval will be wide.
    - Report as measured. Do not tune.
    - Measured: 5 of 140 runs (3.6%), Wilson 95% CI 1.5%–8.1%, from 3 of 7 subjects. p03's 20 runs return "Not enough usable cycles".
11. **Baseline-model diagnostic, real data.** On each of the 43 check 6 windows, take the standardised residuals over the monitoring days and report the distribution across windows of their mean and SD, the share of windows where |mean z| exceeds 2/sqrt(n_valid), and the within-window lag-1 autocorrelation of z. Under the fitted AR(1), z has mean 0, SD 1 and no autocorrelation. Mean away from 0 is a location error (mu moved); SD away from 1 is a scale error (phi or sd_e mis-estimated). This is the only measurement in the build that separates the two. No detector, no simulation, no planting; nothing depends on it. Report as measured.
    - Measured: SD z median 0.97 (IQR 0.79–1.10), RMS 1.02, so the scale holds. Mean z median −0.09 (IQR −0.43 to +0.10, range −0.85 to +0.69), and |mean z| > 2/sqrt(n_valid) in 15 of 43 windows (35%, Wilson 22–50%, 9 of 13 subjects) against about 5% if the model held. Within-window lag-1 of z has median +0.12 (IQR −0.08 to +0.27): a small leftover autocorrelation, consistent with phi underestimated from 60 days. Widening the 2-SE bound for that autocorrelation still leaves 13 of 43 (30%). The baseline fails on location (mu drifts after the start), not on scale, which matches the drift term being the largest in the false-alarm decomposition.

---

## 8. Build order

1. **subject.py, loaders.py.** Checkpoint: one subject's length, NaN count and mean; plot saved to demo_series.png; the count of subjects clearing 144 days.
2. **before_after.** Checkpoint: check 2, plus the demo subject's before/after.
3. **fit_baseline, residuals, cusum, harm_limit, reference_limit, harm_monitor, plant_harm.** Checkpoint: checks 3–7. Report both check 6 rates and the check 7 table before continuing.
4. **App: Roster and Person pages, harm only.** Checkpoint: `streamlit run app.py` works end to end, and planting harm turns the person red.
5. **make_schedule, schedule_check, efficacy, plant_effect.** Checkpoint: checks 8–10. Report the per-subject p_complete and min_detectable table, and check 10.
6. **App: trial mode.** On-block shading, efficacy section, live p_floor and min_detectable.
7. **README:**
   - how to run, including that on macOS a process launched by another app may be refused access to ~/Downloads (the Claude preview launcher was), so start Streamlit from a terminal;
   - data source and licence;
   - the appendix table;
   - the demo statement;
   - the "any daily metric" table below;
   - known limitations: power and min_detectable are simulated under the fitted AR(1), so they are a best case, with check 11's numbers: the scale holds on real monitoring days, the location does not; p_complete is conditional on the device continuing to record as it has been, and its block-length sensitivity (with the non-monotonicity line from 5.5); the before-period limit, weekly blocks being for fast-acting compounds, one pre-specified metric per compound, and the 5.5 per-subject table of p_complete and min_detectable, with the statement that missingness, through runs of missing days, rather than autocorrelation is what limits the efficacy arm.

**Demo subject:** the participant with the most non-NaN days whose series supports start day 60 plus 84 days. Ties break to the lowest subject ID. For PMData this is p01 (152 days, no missing days; it ties with p16).

**Second subject on screen: p03.** p01 has zero missing days, so a demo built only on p01 never exercises the coverage column, the n_valid calibration or the p_complete warning, and those features look like dead code. p03 at 34% missing is where all three fire.

**README table: applying the method to other metrics.** The engine is unchanged; only the Metric differs.

| Compound type | Metric | Works | Harm |
|---|---|---|---|
| Recovery agent (demo) | Resting HR | down | up |
| Recovery agent | HRV | up | down |
| Sleep aid | Sleep duration | up | down (daytime activity) |
| Stimulant | Sleep duration / resting HR | (not wearable) | sleep down / RHR up |

Limits to state alongside the table:
- The metric must be measured reliably by the device.
- Outcomes like cognition or lab values aren't on a wearable and need another data source.
- One primary metric per compound, chosen in advance. Testing several and reporting whichever moved inflates false positives.

---

## 9. Constraints

- Python 3.10+, with numpy, pandas, scipy, matplotlib and streamlit.
- No ML models, no LLM, no network at runtime.
- No interpolation.
- No planted effect without its label.
- Do not change methods, constants or check thresholds to make a result look better.
- If data or results don't match this spec, stop and report.

---

## Appendix: where each method comes from

This isn't needed to implement anything. It's here so a reviewer can check the work, and so nobody later "improves" a standard method without knowing it was one.

| Component | Method | Source |
|---|---|---|
| Overall structure | Phase I / Phase II statistical process control | Montgomery, *Introduction to Statistical Quality Control* |
| Day-to-day correlation | AR(1) model, monitor residuals (residual chart) | Montgomery; Alwan & Roberts 1988 |
| Harm detector | One-sided CUSUM, k = half the target shift | Page 1954, *Biometrika* 41:100–115 |
| Harm threshold | Bootstrap calibration from in-control data | Chatterjee & Qiu 2009, *Ann. Appl. Stat.* 3(1), doi:10.1214/08-AOAS197 |
| Resampling | Residual bootstrap | Efron & Tibshirani 1993 |
| Reference-group threshold | Leave-one-subject-out cross-validation | Stone 1974 |
| Interval on rates | Wilson score interval | Wilson 1927, *JASA* |
| On/off trial design | n-of-1 crossover: randomised paired blocks, washout, block means | Guyatt et al. 1986, *NEJM*; Lillie et al. 2011, *Per. Med.*, PMC3118090 |
| Efficacy test | Exact randomisation test on paired differences | Edgington; Edgington & Onghena, *Randomization Tests* |
| Trial completion (p_complete) | Moving-block bootstrap of the missingness indicator | Künsch 1989, *Ann. Stat.* 17:1217–1241 |

**Choices, not standards:** HORIZON, DELTA, MIN_BEFORE, BLOCK_DAYS, REF_BASELINE_DAYS, DEFAULT_CYCLES, MISS_BLOCK_LENGTHS, MIN_RESAMPLE_DAYS, the 2-SE plateau rule for min_detectable, start day 60, plant sizes, and the demo metric's directions.

**Follows from the method, not a choice:** MIN_CYCLES = 6, from p_floor = 2/2^n and ALPHA.

**Set by measurement, not preference:** MIN_USABLE_DAYS = 2, from the p_complete simulation in 5.5.

**Verified in this project:** Chatterjee & Qiu (read directly) and Lillie et al. (supplied). The rest are standard references cited from memory; check the details before putting any on a slide.
