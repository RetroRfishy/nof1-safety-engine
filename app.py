"""Streamlit UI (spec section 6). Run: streamlit run app.py"""
from __future__ import annotations

import json
import os
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import binom

import engine as E
import loaders as L

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "check_results.json")
METRIC_RESULTS_PATH = os.path.join(HERE, "metric_results.json")
ALARM_LOG = os.path.join(HERE, "alarms.jsonl")
DEFAULT_ROOT = os.path.expanduser("~/Downloads/pmdata")
BANNER = ("Real Fitbit data from volunteers who took nothing. "
          "Start date, schedule and any effect shown are simulated.")
SIMULATED = "Simulated effect"
CLUSTER_ALPHA = 0.05   # the roster line calls a cluster when P(this many or more) falls below this
METHOD_LABELS = {"reference": "Other people's compound-free data", "own_baseline": "This person's 60 days only"}

st.set_page_config(page_title="Compound monitor", layout="wide")

# Population reference ranges, shown for contrast with the person's own normal.
POP_RANGE = {
    "resting_hr": (60.0, 100.0, "population normal 60–100 bpm",
                   "Adult normal resting heart rate 60–100 bpm (Mayo Clinic)."),
    "sleep_efficiency": (85.0, 100.0, "population 'good sleep' ≥ 85%",
                         "Sleep efficiency of at least 85% indicates good sleep quality "
                         "(National Sleep Foundation, Sleep Health 2017)."),
}

# One-click demo presets: set the sidebar widgets, then rerun.
PRESETS = {
    "Planted harm · resting HR · p01": dict(page="Person", metric_key="resting_hr", person="p01", demo="Plant harm"),
    "Planted harm · sleep efficiency · p01": dict(page="Person", metric_key="sleep_efficiency", person="p01",
                                                 demo="Plant harm"),
    "Real, nothing planted · resting HR · p08": dict(page="Person", metric_key="resting_hr", person="p08",
                                                    demo="None"),
}


def apply_preset(name: str) -> None:
    for k, v in PRESETS[name].items():
        st.session_state[k] = v
    st.session_state["trial_on"] = False
    st.session_state["start_day"] = 60
    st.session_state[f"today_{PRESETS[name]['person']}_{PRESETS[name]['metric_key']}_60"] = 0


# ---------------------------------------------------------------- measured numbers
def metric_results() -> dict:
    try:
        with open(METRIC_RESULTS_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def measured(metric_key: str = "resting_hr") -> dict:
    """False-alarm rate (check 6) and 2 SD detection (check 7) per limit. Resting HR:
    check_results.json, falling back to spec.md 5.4. Other metrics:
    metric_results.json (python validate_metrics.py)."""
    if metric_key != "resting_hr":
        r = metric_results().get(metric_key)
        if r is None:
            return None
        m = {meth: {"alarmed": r["false_alarms"][meth]["alarmed"], "windows": r["false_alarms"][meth]["windows"],
                    "fp": r["false_alarms"][meth]["rate"], "caught": r["detection"][meth]["rate"],
                    "median_day": r["detection"][meth]["median_day"]} for meth in METHOD_LABELS}
        m["stability"] = None
        return m
    m = {"reference": {"alarmed": 3, "windows": 43, "fp": 0.070, "caught": 0.53, "median_day": 13.0},
         "own_baseline": {"alarmed": 6, "windows": 43, "fp": 0.140, "caught": 0.79, "median_day": 11.0}}
    try:
        with open(RESULTS_PATH) as f:
            r = json.load(f)
        for method in m:
            c6, c7 = r[f"check6_{method}"], r["check7"][method]
            m[method] = {"alarmed": c6["alarmed"], "windows": c6["windows"], "fp": c6["rate"],
                         "caught": c7["rate"], "median_day": c7["median_day"]}
    except (OSError, KeyError, ValueError):
        pass
    m["stability"] = {"h_spread": (0.10, 0.46, 0.37), "fp_range": (0.070, 0.116), "fp_jump_without": "p08"}
    try:
        with open(RESULTS_PATH) as f:
            r = json.load(f)
        spreads = [(v["max"] - v["min"]) / v["full"] for v in r["check6_stability"].values()]
        fr = r["check6_fp_range"]
        m["stability"] = {"h_spread": (min(spreads), max(spreads), float(np.median(spreads))),
                          "fp_range": (fr["min"], fr["max"]), "fp_jump_without": fr["max_without"]}
    except (OSError, KeyError, ValueError):
        pass
    return m


def cluster_line(n_alarm: int, n_people: int, rate: float) -> str:
    """Roster summary: alarms seen, alarms expected with nothing wrong, and the
    binomial chance of seeing that many or more."""
    expected = n_people * rate
    exp_words = "less than 1" if expected < 0.5 else f"about {expected:.0f}"
    head = f"**{n_alarm} alarm{'s' if n_alarm != 1 else ''} out of {n_people} people.**"
    if n_alarm == 0:
        return f"{head} With nothing wrong you'd expect {exp_words}."
    tail = float(binom.sf(n_alarm - 1, n_people, rate))
    if tail >= 0.5:
        often = "more often than not"
    else:
        often = f"about 1 time in {max(2, round(1 / tail))}"
    close = ("This is not more than chance." if tail >= CLUSTER_ALPHA else
             "That is more than chance would usually produce; review these alarms together.")
    return (f"{head} With nothing wrong you'd expect {exp_words}, and {n_alarm} or more happens "
            f"{often}. {close}")


# ---------------------------------------------------------------- data
@st.cache_resource(show_spinner="Loading cohort and building the reference library…")
def cohort(root: str, start_day: int, metric_key: str = "resting_hr"):
    subjects, skipped = [], []
    for pid in L.list_pmdata(root):
        try:
            subjects.append(L.load_pmdata(root, pid, start_day, metric_key))
        except (L.SubjectUnusable, ValueError) as err:
            skipped.append(str(err))
    return subjects, skipped, E.reference_library(subjects)


def assess(subj, lib, metric, method):
    base = E.fit_baseline(subj)
    return {"subject": subj, "base": base, "ba": E.before_after(subj),
            "harm": E.harm_monitor(subj, base, lib, metric, method=method, seed=0),
            "coverage": float(subj.series.notna().mean()),
            "day": int((subj.series.index[-1] - subj.start).days) + 1}


@st.cache_data(show_spinner="Checking the schedule…")
def schedule_check_cached(root: str, start_day: int, pid: str, n_cycles: int, metric_key: str, seed: int = 0):
    subjects, _, _ = cohort(root, start_day, metric_key)
    subj = next(s for s in subjects if s.subject_id == pid)
    return E.schedule_check(E.fit_baseline(subj), n_cycles, L.METRICS[metric_key][0], seed)


def log_alarm(record: dict) -> None:
    """Append each alarm once as a JSON line. In deployment this would go into
    an existing clinical record; no integration is built."""
    line = json.dumps(record, sort_keys=True)
    try:
        with open(ALARM_LOG) as f:
            if line in {l.strip() for l in f}:
                return
    except OSError:
        pass
    with open(ALARM_LOG, "a") as f:
        f.write(line + "\n")


def direction_word(metric) -> str:
    return "above" if metric.harm_direction == "up" else "below"


# ---------------------------------------------------------------- sidebar
st.caption(BANNER)

sb = st.sidebar
with sb.container(border=True):
    st.markdown("**Demo presets**")
    for _name in PRESETS:
        st.button(_name, on_click=apply_preset, args=(_name,))
main = sb.container()
settings = sb.expander("Settings")

page = main.radio("Page", ["Roster", "Person", "Results"], key="page", horizontal=True)
metric_key = main.selectbox("Metric", list(L.METRICS), format_func=lambda k: L.METRICS[k][1], key="metric_key")
metric = L.METRICS[metric_key][0]

root = DEFAULT_ROOT
if not os.path.isdir(root):
    root = main.text_input("Where is the PMData folder?", DEFAULT_ROOT, key="root")
    if not os.path.isdir(root):
        st.error(f"PMData folder not found: {root}")
        st.stop()
compound = settings.text_input("Compound name", "Compound A")
st.session_state.setdefault("start_day", 60)
start_day = int(settings.number_input("Compound start (days after first date)", min_value=E.MIN_BEFORE, step=1,
                                      key="start_day"))
subjects, skipped, lib = cohort(root, start_day, metric_key)
if not subjects:
    st.error("No subject supports this start day.")
    st.stop()
by_id = {s.subject_id: s for s in subjects}
ids = list(by_id)
if st.session_state.get("person") not in ids:
    st.session_state["person"] = "p01" if "p01" in ids else ids[0]
person = main.selectbox("Person", ids, key="person")

M_ = measured(metric_key)
if M_ is None:
    st.error(f"No validation results for {L.METRICS[metric_key][1]}. Run: python validate_metrics.py --root {root}")
    st.stop()
method = settings.radio(
    "Alarm line set from", list(METHOD_LABELS),
    format_func=lambda m: f"{METHOD_LABELS[m]} · {M_[m]['fp']:.0%} false alarms",
    help="Either way, every day is scored against this person's own normal. This only sets how high the "
         "running tally must climb before it alarms.")

trial_on = main.toggle("On/off trial: is it working?", key="trial_on")
trial = None
if trial_on:
    subj0 = by_id[person]
    rec_end = subj0.series.last_valid_index()
    fit_max = E.max_cycles(subj0.start, rec_end)
    default_n = E.DEFAULT_CYCLES if fit_max >= E.DEFAULT_CYCLES else max(E.MIN_CYCLES, fit_max)
    n_cycles = int(settings.number_input("On/off cycles (1 week on + 1 week off)", min_value=1, max_value=12,
                                         value=default_n, step=1, key=f"ncyc_{person}_{start_day}_{metric_key}"))
    if settings.button("Re-randomise schedule"):
        st.session_state["sched_seed"] = st.session_state.get("sched_seed", 0) + 1
    seed = st.session_state.get("sched_seed", 0)
    sc = schedule_check_cached(root, start_day, person, n_cycles, metric_key)
    schedule, refusal = None, None
    if not sc.allowed:
        refusal = sc.reason
    else:
        try:
            schedule = E.make_schedule(subj0.start, n_cycles, metric, seed, record_end=rec_end)
        except E.ScheduleDoesNotFit as err:
            refusal = str(err)
    if refusal:
        main.error(refusal)
    else:
        md_txt = f"{sc.min_detectable:.1f} SD" if sc.min_detectable is not None else "no size reliably"
        main.caption(f"{n_cycles} weekly on/off cycles, randomised · detects ≥ {md_txt} · "
                     f"{sc.p_complete:.0%} chance of a verdict")
    trial = {"n": n_cycles, "check": sc, "schedule": schedule, "refusal": refusal}

demo_options = ["None", "Plant harm"] + (["Plant effect"] if trial and trial["schedule"] else [])
sd_units = float(np.std(E.before_values(by_id[person]).to_numpy(), ddof=1))
if st.session_state.get("demo") not in demo_options:
    st.session_state["demo"] = "None"
demo = main.radio("Test it: add a simulated change", demo_options, key="demo", horizontal=True,
                  help="Adds a labelled shift to this person's real data. Plant effect needs the on/off trial.")


def size_slider(label: str, key: str) -> float:
    """A planted size in the metric's own units (0.5–4 SD for this person); returns SDs."""
    step = max(0.1, round(sd_units / 10, 1))
    lo, hi, default = round(0.5 * sd_units, 1), round(4 * sd_units, 1), round(2 * sd_units, 1)
    v = main.slider(f"{label} ({metric.unit})", min_value=lo, max_value=hi, value=default, step=step, key=key,
                    help=f"1 SD = {sd_units:.1f} {metric.unit} for {person}")
    return v / sd_units


harm_sd, harm_from, effect_sd = 2.0, 5, 2.0
if demo == "Plant harm":
    harm_sd = size_slider("Size", f"hsize_{person}_{metric_key}")
    harm_from = int(main.slider("Starts on day", 1, 20, 5, key=f"hfrom_{person}_{metric_key}"))
elif demo == "Plant effect":
    effect_sd = size_slider("Size", f"esize_{person}_{metric_key}")

# ---------------------------------------------------------------- assess everyone
planted_id = person if demo != "None" else None
results = {}
for subj in subjects:
    s = subj
    if subj.subject_id == planted_id and demo == "Plant harm":
        s = E.plant_harm(subj, metric, delta_sd=harm_sd, start_offset=harm_from)
    elif subj.subject_id == planted_id and demo == "Plant effect":
        s = E.plant_effect(subj, trial["schedule"], metric, delta_sd=effect_sd)
    results[subj.subject_id] = assess(s, lib, metric, method)
if trial and trial["schedule"] is not None:
    results[person]["eff"] = E.efficacy(results[person]["subject"], trial["schedule"], metric)
for r in results.values():
    if r["harm"].record is not None:
        log_alarm({**r["harm"].record, "method": method})

mm = M_[method]


# ---------------------------------------------------------------- roster
def roster_page():
    st.title(f"{compound}: roster")
    rows = []
    for pid, r in results.items():
        h, sim = r["harm"], r["subject"].simulated
        status = f"Alarm {h.alarm_date.date()}" if h.alarm_date is not None else "Watching"
        if sim:
            status += f" · {SIMULATED}"
        eff = "—"
        if "eff" in r:
            eff = r["eff"].message.rstrip(".") + (f" · {SIMULATED}" if sim else "")
        elif trial and pid == person and trial["refusal"]:
            eff = "Schedule refused"
        rows.append({"Person": pid, "Day": r["day"], "Change": f"{r['ba'].std_change:+.2f} SD",
                     "Harm status": status, "Efficacy": eff, "Coverage": f"{r['coverage']:.0%}",
                     "Note": SIMULATED if sim else "", "_alarm": h.alarm_date is not None})
    df = pd.DataFrame(rows).sort_values(["_alarm", "Person"], ascending=[False, True])
    alarm_mask = df["_alarm"].to_numpy()
    df = df.drop(columns="_alarm").reset_index(drop=True)
    styled = df.style.apply(
        lambda row: ["background-color: #f8d7da; color: #842029"] * len(row) if alarm_mask[row.name]
        else [""] * len(row), axis=1)
    st.dataframe(styled, hide_index=True, width="stretch")

    real = [r for r in results.values() if not r["subject"].simulated]
    n_alarm = sum(r["harm"].alarm_date is not None for r in real)
    st.markdown(cluster_line(n_alarm, len(real), max(mm["fp"], E.FP)))
    st.caption(f"Expected count uses the measured false-alarm rate for this limit ({mm['fp']:.0%}, "
               f"{mm['alarmed']} of {mm['windows']} real no-exposure windows) and the binomial distribution.")
    if planted_id:
        st.caption(f"{planted_id} carries a planted harm ({SIMULATED.lower()}) and is not counted above.")
    st.caption(f"Harm limit: {METHOD_LABELS[method]}. Day = days since start, to the last day of data. "
               f"Change = before-to-after shift in before-period SDs.")
    if skipped:
        with st.expander(f"Skipped subjects ({len(skipped)})"):
            for reason in skipped:
                st.write(f"- {reason}")


# ---------------------------------------------------------------- person
def series_chart(r, upto=None, show_pop=True):
    subj, ba, h = r["subject"], r["ba"], r["harm"]
    s = subj.series if upto is None else subj.series[subj.series.index <= upto]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    if upto is not None:
        ax.set_xlim(subj.series.index[0] - pd.Timedelta(days=1), subj.series.index[-1] + pd.Timedelta(days=1))
    pop = POP_RANGE.get(metric_key) if show_pop else None
    if pop is not None:
        vals = subj.series.dropna()
        lo_v, hi_v = vals.min() - 5 * ba.before_sd, vals.max() + 5 * ba.before_sd
        first = True
        for bound in pop[:2]:
            if lo_v <= bound <= hi_v and bound < 100:
                ax.axhline(bound, color="#8a8985", ls=(0, (4, 3)), lw=1.3,
                           label=pop[2] if first else None)
                first = False
    ax.axhspan(ba.before_mean - 2 * ba.before_sd, ba.before_mean + 2 * ba.before_sd,
               color="tab:blue", alpha=0.12, label="before mean ± 2 SD")
    ax.plot(s.index, s.to_numpy(), "-o", ms=2.5, lw=1, color="0.25", label=L.METRICS[metric_key][1])
    ax.axvline(subj.start, color="k", ls="--", lw=1, label="start")
    sched = trial["schedule"] if trial and subj.subject_id == person else None
    if sched is not None:
        day = pd.Timedelta(days=1)
        on_label = wash_label = True
        for cyc, arm, first, usable_from in E.schedule_blocks(sched):
            end = first + pd.Timedelta(days=sched.block_days)
            if arm == "on":
                ax.axvspan(first - day / 2, end - day / 2, color="tab:green", alpha=0.13, lw=0,
                           label="on-block" if on_label else None)
                on_label = False
            ax.axvspan(first - day / 2, usable_from - day / 2, facecolor="none", edgecolor="0.55",
                       hatch="////", lw=0, label="washout (excluded)" if wash_label else None)
            wash_label = False
    if h.alarm_date is not None and (upto is None or h.alarm_date <= upto):
        ax.axvline(h.alarm_date, color="tab:red", lw=2, label=f"harm alarm {h.alarm_date.date()}")
    if upto is not None:
        ax.axvline(upto, color="#2a78d6", lw=1, alpha=0.6)
    ax.set_ylabel(f"{L.METRICS[metric_key][1]} ({metric.unit})")
    lo_y, hi_y = ax.get_ylim()
    ax.set_ylim(lo_y - 0.04 * (hi_y - lo_y), hi_y + 0.06 * (hi_y - lo_y))
    title = f"{subj.subject_id}: {L.METRICS[metric_key][1]}"
    if subj.simulated:
        title += f"  ({SIMULATED})"
    ax.set_title(title, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8, ncol=6, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


def cusum_chart(h, upto=None):
    fig, ax = plt.subplots(figsize=(11, 2.4))
    S = h.S if upto is None else h.S[h.S.index <= upto]
    ax.set_xlim(h.S.index[0] - pd.Timedelta(days=1), h.S.index[-1] + pd.Timedelta(days=1))
    ax.set_ylim(0, max(h.h * 1.4, float(h.S.max()) * 1.1, 1.0))
    ax.step(S.index, S.to_numpy(), where="post", color="#2a78d6", lw=2, label="tally (CUSUM)")
    ax.axhline(h.h, color="tab:red", ls="--", label=f"alarm line = {h.h:.2f}")
    if h.alarm_date is not None and (upto is None or h.alarm_date <= upto):
        ax.axvline(h.alarm_date, color="tab:red", lw=2)
    ax.set_ylabel("tally")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def efficacy_section(r):
    st.subheader("Is it working for them?")
    sc, sched = trial["check"], trial["schedule"]
    md_txt = f"{sc.min_detectable:.1f} SD" if sc.min_detectable is not None else "not reachable"
    sim = r["subject"].simulated
    if sched is None:
        st.info(trial["refusal"])
    else:
        eff = r["eff"]
        tbl = eff.cycles.rename(columns={"cycle": "Cycle", "order": "Order", "on_mean": f"On mean ({metric.unit})",
                                         "off_mean": f"Off mean ({metric.unit})", "d": f"Difference ({metric.unit})",
                                         "on_days": "On days used", "off_days": "Off days used"})
        tbl["Usable"] = np.where(tbl["usable"], "yes", f"no (< {E.MIN_USABLE_DAYS} days)")
        st.dataframe(tbl.drop(columns="usable").style.format(precision=2, na_rep="—"), hide_index=True,
                     width="stretch")
        label = f" — {SIMULATED}" if sim else ""
        if eff.verdict == "working":
            st.success(eff.message + label)
        elif eff.verdict == "opposite":
            st.error(eff.message + label)
        elif eff.verdict == "none":
            st.info(f"{eff.message} {E.no_effect_note(sc.min_detectable)}{label}")
        else:
            st.warning(eff.message + label)
        if eff.p is not None:
            st.caption(f"Mean difference {eff.mean_d:+.2f} {metric.unit} ({eff.mean_d_sd:+.2f} SD); "
                       f"{eff.n_agree} of {eff.n_usable} cycles agree in direction; randomisation p = {eff.p:.3g}.")
    st.caption(f"Smallest possible p {sc.p_floor:.3g} · chance of a verdict {sc.p_complete:.0%} · "
               f"smallest effect reliably detected {md_txt}")


def person_page():
    r = results[person]
    subj, ba, base, h = r["subject"], r["ba"], r["base"], r["harm"]
    st.title(f"{person} · {L.METRICS[metric_key][1].lower()}")
    st.caption(f"{compound} started {subj.start.date()}" + (f" · {SIMULATED}" if subj.simulated else ""))

    n_after = int((subj.series.index[-1] - subj.start).days)
    today_key = f"today_{person}_{metric_key}_{start_day}"
    st.session_state.setdefault(today_key, n_after)
    if st.session_state[today_key] > n_after:
        st.session_state[today_key] = n_after
    today = st.slider("Replay: days since the compound started", 0, n_after, key=today_key,
                      help="Hides the days after 'today'. Nothing is recomputed, so the alarm appears on the "
                           "same day it does in the full run.")
    upto = subj.start + pd.Timedelta(days=today)
    alarm_now = h.alarm_date is not None and h.alarm_date <= upto
    try:
        ba_now = E.before_after(replace(subj, series=subj.series[subj.series.index <= upto]))
    except Exception:
        ba_now = ba
    if today >= 2:
        ba = ba_now

    st.subheader("What changed?")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Before ({metric.unit})", f"{ba.before_mean:.1f} ± {ba.before_sd:.1f}")
    c2.metric(f"After, to day {today} ({metric.unit})",
              f"{ba.after_mean:.1f} ± {ba.after_sd:.1f}" if ba.after_n > 1 else "—")
    c3.metric(f"Change ({metric.unit})", f"{ba.change:+.2f}")
    c3.caption(f"{ba.std_change:+.2f} SD")
    show_pop = st.toggle("Show population 'normal' range", value=True, key="show_pop")
    st.pyplot(series_chart(r, upto=upto, show_pop=show_pop))

    st.subheader("Is it harming them?")
    sim_tag = f" — {SIMULATED}" if subj.simulated else ""
    if alarm_now:
        day_n = (h.alarm_date - subj.start).days
        msg = (f"Alarm, day {day_n}: {L.METRICS[metric_key][1].lower()} has moved {direction_word(metric)} "
               f"{person}'s own normal by more than day-to-day noise explains. Check on them.")
        st.error(msg + sim_tag)
    elif today < h.calendar_len:
        st.success(f"Watching · day {today} of {h.calendar_len}, no alarm" + sim_tag)
    else:
        st.success(f"No alarm in the {h.calendar_len}-day monitoring course" + sim_tag)
    st.pyplot(cusum_chart(h, upto=upto))
    pop = POP_RANGE.get(metric_key)
    if pop is not None and show_pop:
        x = subj.series
        before7 = x[(x.index < subj.start) & (x.index >= subj.start - pd.Timedelta(days=7))].dropna()
        recent = x[(x.index <= upto) & (x.index > upto - pd.Timedelta(days=7))].dropna()
        if today >= 3 and len(before7) and len(recent):
            vb, vr = float(before7.mean()), float(recent.mean())
            word = lambda v: "inside" if pop[0] <= v <= pop[1] else "outside"
            if word(vb) == word(vr):
                pop_txt = f"Population range: {word(vb)} before ({vb:.1f}) and now ({vr:.1f}), so no change seen."
            else:
                pop_txt = f"Population range: {word(vb)} before ({vb:.1f}), {word(vr)} now ({vr:.1f})."
            own = "**alarm**" if alarm_now else "no alarm"
            st.info(f"{pop_txt} Own normal: {own}.")
            st.caption(pop[3])
    st.caption(f"Measured on real people who took nothing: {mm['fp']:.0%} false alarms per 30 days. "
               f"Catches a 2 SD change {mm['caught']:.0%} of the time, median day {mm['median_day']:.0f}.")
    eff = r.get("eff")
    if eff is not None and eff.verdict == "opposite":
        st.error(f"Efficacy trial: {eff.message} {metric.name} moved {direction_word(metric)} this person's off-week "
                 f"level during on-weeks." + (f" — {SIMULATED}" if subj.simulated else ""))

    if trial:
        efficacy_section(r)

    with st.expander("How this works"):
        params = {"mu (before mean)": f"{base.mu:.2f} {metric.unit}",
                  "SD (before)": f"{base.sd_x:.2f} {metric.unit}",
                  "phi (lag-1 autocorrelation)": f"{base.phi:.3f}",
                  "k (CUSUM reference value)": f"{h.k:.3f}",
                  "h (alarm limit)": f"{h.h:.2f}",
                  "n_valid (scored steps)": f"{h.n_valid} of {h.calendar_len} days",
                  "limit": METHOD_LABELS[method]}
        if method == "reference":
            params["reference library"] = f"{h.ref_windows} windows from {h.ref_subjects} other subjects"
        st.table(pd.DataFrame({"value": params}))
        st.markdown(
            "- **Baseline.** An AR(1) model is fitted to the before period, because today's value predicts "
            "tomorrow's. The detector watches the standardised residuals (Montgomery; Alwan & Roberts 1988).\n"
            f"- **Detector.** One-sided CUSUM in the harm direction, tuned for a {E.DELTA:.0f} SD shift; "
            "k is half that shift in residual units (Page 1954).\n"
            "- **Limit (default).** Reference group: the 95th percentile of CUSUM maxima over other "
            "people's real post-start data, each window with its own re-fitted 60-day baseline, leaving this "
            "person out (Stone 1974).\n"
            "- **Limit (comparator).** Bootstrap of this person's own baseline residuals "
            "(Chatterjee & Qiu 2009; Efron & Tibshirani 1993).\n"
            "- **Rates.** Wilson score intervals (Wilson 1927).\n"
            "- **On/off trial.** n-of-1 crossover: randomised paired weekly blocks, the first "
            f"{metric.onset_days} days of every block excluded as washout, block means compared "
            "(Guyatt et al. 1986; Lillie et al. 2011).\n"
            "- **Efficacy test.** Exact randomisation test on the cycle differences, enumerating every "
            "on/off order (Edgington & Onghena).\n"
            "- **Chance of a verdict.** Blocks of this person's own before-period gap pattern laid over the "
            "schedule, so runs of missing days are kept (moving-block bootstrap, Künsch 1989).")
        if trial:
            sc = trial["check"]
            if sc.p_complete_by_block:
                rng_txt = ", ".join(f"{L}-day blocks {v:.0%}" for L, v in sc.p_complete_by_block.items())
                st.markdown(f"**Chance of a verdict, by resampling block length.** {rng_txt}. The figure shown "
                            f"is the lowest, so the warning fires if any of these assumptions says it should.")
            else:
                st.markdown("**Chance of a verdict.** The before period is too short to resample, so this uses "
                            "the binomial over independent days, which is optimistic when missing days come in runs.")
        stab = M_["stability"]
        if stab is None:
            return
        st.markdown(
            f"**How firm the reference limit is.** The library has 13 people. Leaving any one of them out moves "
            f"h by {stab['h_spread'][0]:.0%}–{stab['h_spread'][1]:.0%} of its value (median "
            f"{stab['h_spread'][2]:.0%}). The false-alarm rate moves less: it stays at {stab['fp_range'][0]:.0%} "
            f"for every left-out person but one, and rises to {stab['fp_range'][1]:.0%} without "
            f"{stab['fp_jump_without']}, whose drift sets most of the tail. A deployment needs a reference "
            f"cohort of 50–100 people for a firm limit.")


# ---------------------------------------------------------------- results: both metrics
C_REF, C_OWN, C_ALARM, C_MUTED, C_INK = "#2a78d6", "#eb6834", "#d03b3b", "#9a9893", "#52514e"
RESULTS_START = 60          # validation windows and library use start day 60 (validate_metrics.py)


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c3")
    ax.tick_params(colors=C_INK, labelsize=8)
    ax.yaxis.label.set_color(C_INK)
    ax.xaxis.label.set_color(C_INK)
    ax.grid(axis="y", color="#ecebe7", lw=0.8)
    ax.set_axisbelow(True)


def _units(x, unit):
    if unit == "min":
        return f"{x:.0f} {unit} ({x / 60:.1f} h)"
    if unit == "%":
        return f"{x:.1f} percentage points"
    return f"{x:.1f} {unit}"


@st.cache_data(show_spinner="Scoring the example window…")
def window_trace(root: str, key: str, pid: str, start: str, plant_sd: float = 0.0):
    """A validation window, re-run through the engine: 60-day baseline before
    `start`, 30 days monitored, reference limit leaving pid out."""
    subjects, _, lib = cohort(root, RESULTS_START, key)
    met = L.METRICS[key][0]
    subj = next(s for s in subjects if s.subject_id == pid)
    sub = E.rebase(subj, pd.Timestamp(start), 60)
    base = E.fit_baseline(sub)
    scored = E.plant_harm(sub, met, delta_sd=plant_sd) if plant_sd else sub
    r = E.harm_monitor(scored, base, lib, met, method="reference")
    end = sub.start + pd.Timedelta(days=E.HORIZON)
    keep = scored.series.index < end
    return {"x": scored.series[keep], "x0": sub.series[keep], "start": sub.start, "mu": base.mu, "sd": base.sd_x,
            "S": r.S, "h": r.h, "alarm": r.alarm_date}


def trigger_note(t):
    """How the alarm built up: one extreme day, or several days of drift."""
    if t["alarm"] is None:
        return "No alarm."
    S = t["S"].dropna()
    i = S.index.get_loc(t["alarm"])
    prev = S.iloc[i - 1] if i > 0 else 0.0
    if S.iloc[i] - prev > 0.7 * t["h"]:
        return "Triggered by a single extreme day."
    zeros = S.iloc[:i][S.iloc[:i] == 0]
    run = (t["alarm"] - zeros.index[-1]).days if len(zeros) else i + 1
    return f"Built up over {run} days of drift."


def strip_chart(r):
    """Every no-compound window: highest tally reached, as a fraction of its own alarm line."""
    w = sorted(r["windows"], key=lambda d: (d["subject"], d["offset"]))
    ratio = np.array([d["peak"] / d["h"] for d in w])
    alarm = np.array([d["alarm"] for d in w])
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    xs = np.arange(len(w))
    ax.scatter(xs[~alarm], ratio[~alarm], s=22, color=C_MUTED, zorder=3, label="stayed below")
    ax.scatter(xs[alarm], ratio[alarm], s=40, color=C_ALARM, zorder=4, edgecolor="white", lw=1.2,
               label="crossed: alarm")
    for x, y, d in zip(xs[alarm], ratio[alarm], np.array(w)[alarm]):
        ax.annotate(d["subject"], (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8, color=C_INK)
    ax.axhline(1.0, color=C_ALARM, ls="--", lw=1.2)
    ax.text(len(w) - 0.5, 1.03, "alarm line", ha="right", va="bottom", fontsize=8, color=C_ALARM)
    ax.set_ylim(0, max(1.6, ratio.max() * 1.1))
    ax.set_xticks([])
    ax.set_xlabel(f"{len(w)} thirty-day windows, {r['n_subjects']} people (nothing taken)", fontsize=8)
    ax.set_ylabel("peak tally / alarm line", fontsize=8)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def fa_chart(r):
    """False-alarm rate for the two limits, with Wilson 95% intervals and the 5% target."""
    fa = r["false_alarms"]
    meths = list(METHOD_LABELS)
    rates = [fa[m]["rate"] * 100 for m in meths]
    lo = [rates[i] - fa[m]["ci"][0] * 100 for i, m in enumerate(meths)]
    hi = [fa[m]["ci"][1] * 100 - rates[i] for i, m in enumerate(meths)]
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    y = np.arange(len(meths))
    ax.barh(y, rates, height=0.5, color=[C_REF, C_OWN], zorder=3)
    ax.errorbar(rates, y, xerr=[lo, hi], fmt="none", ecolor=C_INK, elinewidth=1, capsize=3, zorder=4)
    for i, m in enumerate(meths):
        ax.text(fa[m]["ci"][1] * 100 + 1, i, f"{rates[i]:.1f}%  ({fa[m]['alarmed']}/{fa[m]['windows']})",
                va="center", fontsize=8, color=C_INK)
    ax.axvline(E.FP * 100, color=C_INK, ls=":", lw=1)
    ax.text(E.FP * 100, -0.42, " 5% target", fontsize=8, color=C_INK, va="bottom")
    ax.set_yticks(y, ["Reference group\n(default)", "Own baseline\n(comparator)"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 55)
    ax.set_xlabel("false alarms per 30-day course, % (95% interval)", fontsize=8)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#ecebe7", lw=0.8)
    fig.tight_layout()
    return fig


def shift_chart(all_res, keys):
    """Detection within 30 days by planted shift size, reference limit, both metrics."""
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    for key, col in zip(keys, (C_REF, C_OWN)):
        b = all_res[key]["by_shift"]
        xs = [d["shift_sd"] for d in b]
        ys = [d["rate"] * 100 for d in b]
        ax.plot(xs, ys, "-o", color=col, lw=2, ms=6, label=all_res[key]["label"], zorder=3)
        ax.annotate(all_res[key]["label"], (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    fontsize=8, color=C_INK, va="center")
    ax.set_xlim(0.3, 3.9)
    ax.set_ylim(0, 105)
    ax.set_xlabel("planted sustained shift, in that person's own SDs", fontsize=8)
    ax.set_ylabel("caught within 30 days, %", fontsize=8)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def trace_chart(t, met, label, title):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.2, 3.8), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    a1.axhspan(t["mu"] - 2 * t["sd"], t["mu"] + 2 * t["sd"], color=C_REF, alpha=0.10, lw=0,
               label="own normal ± 2 SD")
    planted = not np.allclose(np.nan_to_num(t["x"].to_numpy()), np.nan_to_num(t["x0"].to_numpy()))
    if planted:
        a1.plot(t["x0"].index, t["x0"].to_numpy(), lw=1, color=C_MUTED, label="real data")
        a1.plot(t["x"].index, t["x"].to_numpy(), lw=1.6, color=C_INK, label="with planted harm")
    else:
        a1.plot(t["x"].index, t["x"].to_numpy(), lw=1.4, color=C_INK, label="real data, nothing planted")
    a1.axvline(t["start"], color=C_INK, ls="--", lw=1)
    a1.set_ylabel(met.unit, fontsize=8)
    a1.set_title(title, loc="left", fontsize=9, color=C_INK)
    a1.legend(loc="upper left", fontsize=7, ncol=3, frameon=True, facecolor="white", edgecolor="none", framealpha=0.85)
    a2.step(t["S"].index, t["S"].to_numpy(), where="post", color=C_REF, lw=1.8, label="tally")
    a2.axhline(t["h"], color=C_ALARM, ls="--", lw=1.2, label=f"alarm line {t['h']:.1f}")
    for a in (a1, a2):
        if t["alarm"] is not None:
            a.axvline(t["alarm"], color=C_ALARM, lw=1.6)
        _style(a)
    a2.set_ylabel("tally", fontsize=8)
    a2.legend(loc="upper left", fontsize=7, ncol=2, frameon=True, facecolor="white", edgecolor="none", framealpha=0.85)
    a2.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    return fig


def results_page():
    st.title("Results: same engine, two metrics")
    allr = metric_results()
    keys = [k for k in L.METRICS if k in allr]
    if len(keys) < 2:
        st.error(f"Run `python validate_metrics.py --root {root}` first.")
        return
    st.caption("Real PMData, nobody took anything. Each person is scored against their own 60 days before "
               "a pretend start, then watched for 30 days.")

    st.subheader("False alarms and detection")
    cols = st.columns(len(keys))
    for c, k in zip(cols, keys):
        r = allr[k]
        fa, de = r["false_alarms"]["reference"], r["detection"]["reference"]
        with c:
            st.markdown(f"#### {r['label']}")
            t1, t2, t3 = st.columns(3)
            t1.metric("False alarms", f"{fa['rate']:.1%}")
            t1.caption(f"{fa['alarmed']} of {fa['windows']} windows · {r['n_subjects']} people · "
                       f"95% {fa['ci'][0]:.0%}–{fa['ci'][1]:.0%}")
            t2.metric("2 SD harm caught", f"{de['rate']:.0%}")
            t2.caption(f"within 30 days · median day {de['median_day']:.0f}")
            t3.metric("2 SD is about", _units(2 * r["median_sd"], r["unit"]).split(" (")[0]
                      .replace(" percentage points", " pts"))
            t3.caption(f"for the median person ({'down' if r['harm_direction'] == 'down' else 'up'})")

    st.subheader("Every compound-free month")
    st.caption("One dot per real 30-day window. Above the line = a false alarm.")
    cols = st.columns(len(keys))
    for c, k in zip(cols, keys):
        with c:
            st.markdown(f"**{allr[k]['label']}**")
            st.pyplot(strip_chart(allr[k]))

    exp3 = st.expander("Why the alarm line is set from other people's data")
    exp3.markdown("Every day is still scored against **this person's own normal**. Other people's "
                  "compound-free data only answers one question: how high does the running tally climb by chance? "
                  "Sixty days of one person rarely contain their rare extreme days, so a line set from them alone "
                  "sits too low and false-alarms more.")
    cols = exp3.columns(len(keys))
    for c, k in zip(cols, keys):
        with c:
            st.markdown(f"**{allr[k]['label']}**")
            st.pyplot(fa_chart(allr[k]))

    st.subheader("How big a change it can see")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.pyplot(shift_chart(allr, keys))
    with c2:
        for k in keys:
            r = allr[k]
            one = next(b for b in r["by_shift"] if b["shift_sd"] == 1.0)
            st.markdown(f"**{r['label']}:** 1 SD ≈ {_units(r['median_sd'], r['unit'])}; "
                        f"caught {one['rate']:.0%} of the time. Day-to-day stickiness "
                        f"φ ≈ {r['median_phi']:.2f}.")
        st.caption("Resting HR is sticky day to day, so each day adds less new evidence and the same "
                   "shift takes longer to catch.")

    st.subheader("Alarms nobody planted")
    st.caption("No compound was taken. Some are real sustained changes, some are one extreme day.")
    cols = st.columns(len(keys))
    for c, k in zip(cols, keys):
        r = allr[k]
        with c:
            st.markdown(f"**{r['label']}**")
            for a in r["unplanted_alarms"]:
                t = window_trace(root, k, a["subject"], a["start"])
                st.pyplot(trace_chart(t, L.METRICS[k][0], r["label"],
                                      f"{a['subject']} · window from {a['start']} · alarm {a['alarm']}"))
                st.caption(trigger_note(t))


if page == "Roster":
    roster_page()
elif page == "Person":
    person_page()
else:
    results_page()
