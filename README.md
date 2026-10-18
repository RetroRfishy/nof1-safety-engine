# Personal compound monitor

A tool for a medical officer. For one person taking a compound, it follows a daily wearable metric before and after the start and answers three questions: **what changed**, **is it harming them**, and **is it working for them**. A roster shows everyone on the compound at once. The demo uses resting heart rate from PMData, but any daily metric a wearable records can be used.

## What it catches and what it misses

All of these were measured on real wearable data from people who took nothing, so every alarm and every "effect" in that data is false.

**Harm monitor**
- **False alarms: 7%.** That is 3 of 43 real 30-day no-exposure windows (Wilson 95% CI 2.4–18.6%).
- **Detection: about half.** It catches about half of 2 SD sustained shifts within 30 days (23 of 43 planted), typically by day 13 (IQR 10–22). That half is an average over people whose own course differs a lot: windows whose baseline drifted toward harm caught 14 of 15, and those drifting away caught 9 of 28.
- **What it misses:** smaller or slower shifts may not be caught at all.

**On/off trial**
- **False positives: 3.6%.** That is 5 of 140 random 6-cycle schedules (Wilson 1.5–8.1%), close to the 3.1% the test allows by construction.
- **Smallest effect reliably found:** 1.7–1.9 baseline SDs with complete data. A 1 SD effect is usually missed.
- **What it misses:** with runs of missing days, a trial may not finish at all. p03 never produces a verdict, and the tool says so instead of giving one.

> **Demo:** real heart-rate data from volunteers who took nothing. Start date, schedule and any effect shown are simulated.

## How it works

1. **What changed?** Before versus after: mean ± SD, N days, and the change in units and in before-period SDs. A before/after difference alone can't show the compound caused it; season, training or illness could. The on/off trial is what attributes change to the compound.
2. **Is it harming them?** The person's normal is fitted as an AR(1) model on the before period, because today's value predicts tomorrow's. A one-sided CUSUM watches the standardised residuals in the harm direction, tuned for a 2 SD shift. The alarm limit is the 95th percentile of the same statistic over other people's real post-start data with nothing taken (the reference group), leaving this person out.
3. **Is it working for them?** An n-of-1 crossover trial: paired weekly on/off blocks in random order, the first 2 days of each block excluded as washout, block means compared, and an exact randomisation test over every possible on/off order. Before the schedule runs, the app states the following:
   - the smallest p the schedule can reach;
   - the chance it produces a verdict at all, from this person's own gap pattern;
   - the smallest effect it reliably detects.

   It refuses schedules that can't produce a result or don't fit the record.

## How to run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python checks.py --root ~/Downloads/pmdata      # acceptance checks (about 5 s)
.venv/bin/streamlit run app.py                            # opens the Roster page
```

Set the PMData root in the sidebar if it isn't `~/Downloads/pmdata`. Python 3.10+, numpy, pandas, scipy, matplotlib, streamlit. There are no ML models, no LLM and no network access at runtime.

**macOS:** a process launched by another app may be refused access to `~/Downloads` (the Claude desktop preview launcher was: "Operation not permitted"). Start Streamlit from a terminal and open http://localhost:8501. Streamlit reruns `app.py` on change but not the modules it imports, so restart it after editing `engine.py`.

Alarms are appended to `alarms.jsonl` as one JSON line each: subject, date, metric, h, n_valid, FP, simulated, and the limit used. In deployment they would go into an existing clinical record; no integration is built.

## Data

PMData: Thambawita et al., Simula. Licence CC BY 4.0, at osf.io/vx4bk. Only `pXX/fitbit/resting_heart_rate.json` is read. Missing days (stored as value 0.0) become NaN, the series is reindexed to complete days, and nothing is interpolated.

| | |
|---|---|
| Subjects with a file | 14 (p12 and p13 have none) |
| Usable at a day-60 start | 13 (p04 has 35 days) |
| Demo subject | p01: 152 days, no missing days (ties with p16; ties go to the lowest ID) |
| Second subject on screen | p03: 34% missing, where coverage, n_valid calibration and the completion warning all fire |

## Applying the method to other metrics

The engine is unchanged; only the Metric differs.

| Compound type | Metric | Works | Harm |
|---|---|---|---|
| Recovery agent (demo) | Resting HR | down | up |
| Recovery agent | HRV | up | down |
| Sleep aid | Sleep duration | up | down (daytime activity) |
| Stimulant | Sleep duration / resting HR | (not wearable) | sleep down / RHR up |

- The metric must be measured reliably by the device.
- Outcomes like cognition or lab values aren't on a wearable and need another data source.
- One primary metric per compound, chosen in advance. Testing several and reporting whichever moved inflates false positives.

## Known limitations

**Most of this cohort can't run the efficacy arm.** A 6-cycle weekly trial needs 84 days after a baseline, and the deployment default of 8 cycles needs 112. At a day-60 start, only 7 of 13 subjects fit 6 cycles. p08 and p14 fit 5. p10's device records nothing after 2020-02-28, so 4 fit. p02, p05 and p11 fit 2 or fewer. None fit 8.

**The reference limit is not firm with 13 people.** Leaving any one subject out of the library moves h by 10–46% of its value (median 37%); p01's h ranges from 5.75 to 8.68. Most of the tail of the null comes from one subject, p08. p08 trips its own limit, which excludes p08, and inflates everyone else's, because p08's drift sits in all their libraries. The false-alarm rate moves much less: 7.0% for 12 of 13 removals, and 11.6% without p08. Both numbers matter. The first is the measured reason a real deployment needs a reference cohort of 50–100 people, not 13.

**A deployment has to build a reference group first.** The reference limit needs a group monitored with no exposure. The comparator, a bootstrap of the person's own before period, needs none, but it false-alarms 14% of the time (6 of 43) against a nominal 5%: the before period contains none of the drift that appears after the start. That limit catches 79% of 2 SD shifts, by day 11. This isn't a separate benefit: its limit is lower, which is the same fact as its higher false-alarm rate.

**Why not just correct for the drift?** Because there is no common drift to subtract. Check 11's mean z across the 43 windows has a median of −0.09; the problem is entirely spread (IQR −0.43 to +0.10, range −0.85 to +0.69), per-person drift going both ways. A population drift correction removes a shared mean and would change nothing here. The reference-group limit is the tool that works on this dataset because it absorbs spread.

**Detection varies by person far more than 53% suggests.** The harm detector is one-sided, and each person's own drift either helps or hides a real shift. Splitting check 7's planted-harm windows by their check 11 mean z: drift toward harm caught 14 of 15 (median day 11), drift away caught 9 of 28 (32%, median day 21). The split uses the same monitoring days the detector scores, so it shows how much the outcome depends on the person's course, not a way to predict it in advance.

**Seasonal drift is inside the null.** Every PMData subject was recorded November to March, so drift shared across the cohort sits inside the null instead of being caught by it. The 7% is optimistic for a cohort spread across different months, and this can't be fixed with this dataset.

**Whitening weakens real shifts.** With high day-to-day autocorrelation, removing it (whitening) also weakens a real shift. That is the price of a correct false-alarm rate.

**Power is a best case, and check 11 says where the model breaks.** min_detectable and power are simulated under each person's fitted AR(1) model. Check 11 measures that model on the 43 real monitoring windows:
- **Scale holds.** The SD of the standardised residuals has a median of 0.97 (RMS 1.02).
- **Location doesn't.** The mean is more than 2 SE from zero in 15 of 43 windows (35%) against about 5% expected, in both directions, across 9 of 13 subjects. Allowing for the small leftover autocorrelation (median lag-1 +0.12, consistent with phi underestimated from 60 days) still leaves 30%.
- **What that means:** the baseline mean drifts after the start, per person and with no common sign; the spread is estimated correctly. That is the same drift that pushes the harm false-alarm rate above 5%. The trial simulation generates from the fitted model with no drift at all, while real monitoring days carry per-person mean shifts between −0.85 and +0.69 SD. Slow drift mostly cancels inside each two-week on/off pair, but by an amount no check measures, so stated power is a best case.
- **Measured next step, not acted on:** the leftover lag-1 of +0.12 matches phi being underestimated from 60 days. That was the parameter-estimation term in the original false-alarm decomposition (8.8% → 17% on simulated data). A bias-corrected phi is the next thing to try. The reference limit already states its measured 7%, so nothing false is claimed in the meantime.

**Missingness, not autocorrelation, limits the efficacy arm.** Runs of missing days cap power at the chance of finishing. The chance of a verdict is estimated by resampling blocks of the person's own before-period gap pattern at block lengths of 3, 5, 7 and 14 days, and the lowest is reported. The block length matters: for p05 the chance is 0.90 / 0.79 / 0.64 / 0.77 at those lengths (10,000 simulated trials each, so about ±0.01). The non-monotonicity is real, not noise: p05's gaps have structure at a particular scale rather than being random runs, and 7-day blocks happen to reproduce it. That estimate also assumes the device keeps recording as it has been. A device that stops is caught by the length check, not the probability.

**Dropping cycles assumes the gaps are unrelated to the compound.** The randomisation test stays valid when cycles drop out only if which cycles survive is independent of the on/off order. Charging and not wearing are fine. A compound that made people stop wearing the device would break it.

| subject | phi | before coverage | chance of a verdict | min_detectable | real schedules with a verdict |
|---|---|---|---|---|---|
| p01 | 0.63 | 100% | 1.00 | 1.8 SD | 20/20 |
| p03 | 0.67 | 48% | 0.00 | not reachable | 0/20 |
| p05 | 0.92 | 88% | 0.64 | not reachable | doesn't fit |
| p06 | 0.74 | 98% | 1.00 | 1.8 | 20/20 |
| p07 | 0.80 | 97% | 1.00 | 1.9 | 20/20 |
| p09 | 0.85 | 98% | 1.00 | 1.8 | 20/20 |
| p14 | 0.90 | 88% | 0.93 | 1.7 | doesn't fit |
| p15 | 0.81 | 98% | 1.00 | 1.8 | 20/20 |
| p16 | 0.78 | 100% | 1.00 | 1.9 | 20/20 |

(The full table for all 13 subjects is in spec.md 5.5.) phi matters mildly, in the helpful direction: the lowest value, 1.7, is p14 at phi 0.90. min_detectable carries about ±0.1 SD of Monte Carlo noise. When power can't clear 80% because too few simulated trials finish, the app reports "not reachable" rather than a number.

**Weekly blocks are for fast-acting compounds only.** The effect must start and stop within the 2-day washout. Slower compounds need longer blocks and longer trials.

**One pre-specified metric per compound,** with its works and harm directions fixed before any data is looked at.

**Nobody in PMData took anything.** The start date, the schedule and any effect shown are simulated, and every planted effect is labelled "Simulated effect" beside each status it affects.

## Acceptance checks

`python checks.py` runs these with fixed seeds. Checks 6, 10 and 11 are measurements on real data with no threshold.

| # | Check | Result |
|---|---|---|
| 1 | Loader: complete daily index, no zeros, NaN count = zero entries + absent dates | PASS (14 subjects) |
| 2 | Before/after on a known +5 shift, SD 2 | PASS: change 5.12, standardised 2.53 |
| 3 | AR(1) fit on 500 synthetic days, phi = 0.8 | PASS: phi 0.803, residual lag-1 −0.02 |
| 4 | CUSUM at rest | PASS |
| 5 | Bootstrap limit self-check | PASS: 6.0% (band 3.5–6.5%) |
| 6 | Harm false alarms, 43 real windows | reference 7.0%, own baseline 14.0%; leave-one-out h spread 10–46%, rate 7.0–11.6% |
| 7 | Planted 2 SD harm in each window | reference 53%, median day 13; own baseline 79%, day 11; PASS on p01 (day 19) |
| 8 | Efficacy gates | PASS: p_floor 0.25 / 0.0625 / 0.031; p10 rejected by length; p03 chance of a verdict 0.00 |
| 9 | Planted 1 SD effect on p01 | PASS: every cycle shifts by exactly −1 SD; p = 0.34, the expected outcome below min_detectable |
| 10 | Trial false positives, 140 real runs | 3.6% (expected about 3.1%) |
| 11 | Baseline-model diagnostic on 43 real windows | SD z 0.97 (scale holds); mean z beyond 2 SE in 35% of windows (location drifts) |

## Methods and sources

| Component | Method | Source |
|---|---|---|
| Overall structure | Phase I / Phase II statistical process control | Montgomery, *Introduction to Statistical Quality Control* |
| Day-to-day correlation | AR(1) model, monitor residuals (residual chart) | Montgomery; Alwan & Roberts 1988 |
| Harm detector | One-sided CUSUM, k = half the target shift | Page 1954, *Biometrika* 41:100–115 |
| Harm threshold (comparator) | Bootstrap calibration from in-control data | Chatterjee & Qiu 2009, *Ann. Appl. Stat.* 3(1) |
| Resampling | Residual bootstrap | Efron & Tibshirani 1993 |
| Reference-group threshold (default) | Leave-one-subject-out cross-validation | Stone 1974 |
| Interval on rates | Wilson score interval | Wilson 1927, *JASA* |
| On/off trial design | n-of-1 crossover: randomised paired blocks, washout, block means | Guyatt et al. 1986, *NEJM*; Lillie et al. 2011, *Per. Med.* |
| Efficacy test | Exact randomisation test on paired differences | Edgington & Onghena, *Randomization Tests* |
| Trial completion | Moving-block bootstrap of the missingness indicator | Künsch 1989, *Ann. Stat.* 17:1217–1241 |

**Choices, not standards:** HORIZON (30 days), DELTA (2 SD), MIN_BEFORE (14 days), BLOCK_DAYS (7), REF_BASELINE_DAYS (60), DEFAULT_CYCLES (8), MISS_BLOCK_LENGTHS (3, 5, 7, 14), MIN_RESAMPLE_DAYS (28), the 2-SE plateau rule for min_detectable, start day 60, plant sizes, and the demo metric's directions.

**Follows from the method:** MIN_CYCLES = 6, from p_floor = 2/2^n and ALPHA = 0.05.

**Set by measurement:** MIN_USABLE_DAYS = 2. At 3, most trials on high-missingness subjects never return a verdict.

Chatterjee & Qiu and Lillie et al. were checked directly in this project. The rest are standard references cited from memory; check the details before putting any on a slide.

## Files

| File | Contents |
|---|---|
| `subject.py` | Subject, Metric, Schedule |
| `loaders.py` | `list_pmdata`, `load_pmdata` |
| `engine.py` | All methods; pure functions, seeded, no I/O, no metric-specific code |
| `checks.py` | Acceptance checks; writes `check_results.json`, which the app reads |
| `app.py` | Streamlit UI: Roster and Person pages |
| `spec.md` | The build spec, with every decision made during the build |
