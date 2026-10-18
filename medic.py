"""Medic view: one unit on a newly fielded compound, day by day.
Run: streamlit run medic.py

Same engine as app.py (own-baseline AR(1) residuals, one-sided CUSUM, alarm line
from other people's compound-free data). Real PMData; the compound and the
reactions in three people are simulated and labelled."""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import binom, wilcoxon

import engine as E
import direction as D
import loaders as L
from subject import Subject

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.expanduser("~/Downloads/pmdata")
if not os.path.isdir(DEFAULT_ROOT):                      # hosted copy: bundled data subset
    DEFAULT_ROOT = os.path.join(HERE, "demo_data", "pmdata")
START_DAY = 60
BASELINE_DAYS = 60
COURSE = E.HORIZON                      # 30-day monitoring course
ALL_KEYS = ("resting_hr", "sleep_efficiency")
# Signals available from the wearable; each alarms in its harm direction.
SIGNAL_LABELS = {"resting_hr": "Resting heart rate", "sleep_efficiency": "Sleep efficiency"}
# Simulated reactions: sustained drop in sleep efficiency from day 2, in that person's own SDs.
REACTIONS = {"p01": 2.5, "p06": 2.0, "p15": 1.5}
REACTION_START = 2
ACTIONS = ("Checked in: fine", "Escalate to physician", "Pause compound")

C_ALERT, C_WARN, C_INK, C_MUTED, C_BAND = "#d03b3b", "#eda100", "#52514e", "#9a9893", "#2a78d6"

st.set_page_config(page_title="Medic view", layout="wide")


# ---------------------------------------------------------------- data + engine
def false_alarm_rates(dirs: dict) -> dict:
    """Measured alerts-with-nothing-given rate for each signal under its chosen direction."""
    out = {}
    for k in ALL_KEYS:
        m = D.measured(k)
        if m and dirs.get(k) in m.get("alert_rate", {}):
            out[k] = m["alert_rate"][dirs[k]]
    if len(out) == len(ALL_KEYS):
        return out
    try:
        with open(os.path.join(HERE, "metric_results.json")) as f:
            r = json.load(f)
        return {k: r[k]["false_alarms"]["reference"]["rate"] for k in ALL_KEYS}
    except (OSError, KeyError, ValueError):
        return {"resting_hr": 0.07, "sleep_efficiency": 0.054}


def plant_reaction(subj: Subject, metric, delta_sd: float) -> Subject:
    return E.plant_harm(subj, metric, delta_sd=delta_sd, start_offset=REACTION_START, ramp_days=2)


@st.cache_resource(show_spinner="Loading the unit…")
def build_unit(root: str, reactions_on: bool, dirs: tuple):
    libs, metric_subjects = {}, {}
    for key in ALL_KEYS:
        subs = []
        for pid in L.list_pmdata(root):
            try:
                subs.append(L.load_pmdata(root, pid, START_DAY, key))
            except (L.SubjectUnusable, ValueError):
                pass
        libs[key] = E.reference_library(subs)
    unit = {}
    for pid in L.list_pmdata(root):
        try:
            anchor = L.load_pmdata(root, pid, START_DAY, "resting_hr")   # one start date per person
        except (L.SubjectUnusable, ValueError):
            continue
        start = anchor.start
        person = {"start": start, "metrics": {}}
        for key in ALL_KEYS:
            met = L.METRICS[key][0]
            try:
                series = L.METRICS[key][2](root, pid)
                subj = E.rebase(Subject(pid, series, start), start, BASELINE_DAYS)
                base = E.fit_baseline(subj)
            except (L.SubjectUnusable, ValueError):
                person["metrics"][key] = None
                continue
            simulated = reactions_on and key == "sleep_efficiency" and pid in REACTIONS
            scored = plant_reaction(subj, met, REACTIONS[pid]) if simulated else subj
            r, side = D.monitor(scored, base, libs[key], met, dict(dirs)[key])
            person["metrics"][key] = {"series": scored.series, "S": r.S, "h": r.h, "alarm": r.alarm_date,
                                      "mu": base.mu, "sd": base.sd_x, "simulated": simulated,
                                      "side": side, "direction": dict(dirs)[key]}
        unit[pid] = person
    return unit


def status_at(m, day: int):
    """Normal / Drifting / Alert on a given course day (0 = start)."""
    if m is None:
        return "No data", None
    today = m["S"].index[0] + pd.Timedelta(days=day) if len(m["S"]) else None
    if m["alarm"] is not None and today is not None and m["alarm"] <= today:
        return "Alert", (m["alarm"] - m["S"].index[0]).days
    S = m["S"][m["S"].index <= today] if today is not None else m["S"]
    if len(S) and S.iloc[-1] > 0.5 * m["h"]:
        return "Drifting", None
    return "Normal", None


# ---------------------------------------------------------------- state
root = DEFAULT_ROOT
if not os.path.isdir(root):
    root = st.sidebar.text_input("Where is the PMData folder?", DEFAULT_ROOT)
    if not os.path.isdir(root):
        st.error(f"PMData folder not found: {root}")
        st.stop()

st.session_state.setdefault("day", 0)
st.session_state.setdefault("actions", {})      # (pid, metric) -> (action, day)


def step(n: int) -> None:
    st.session_state["day"] = int(min(COURSE - 1, max(0, st.session_state["day"] + n)))


def act(pid: str, key: str, action: str) -> None:
    st.session_state["actions"][(pid, key)] = (action, st.session_state["day"])


with st.sidebar:
    view = st.radio("View", ["Unit monitor", "On/off trial", "Real substance: alcohol"], key="view")
    st.markdown("**Compound setup**")
    compound_name = st.text_input("Compound", "Compound A")
    st.markdown("Signals to watch")
    watched, dirs = [], []
    for k in ALL_KEYS:
        default = L.METRICS[k][0].harm_direction
        on = st.checkbox(SIGNAL_LABELS[k], value=True, key=f"watch_{k}")
        d = st.radio(f"Alert when {SIGNAL_LABELS[k].lower()}", list(D.DIRECTIONS),
                     index=list(D.DIRECTIONS).index(default), format_func=D.LABEL.get, horizontal=True,
                     key=f"dir_{k}", disabled=not on, label_visibility="collapsed")
        dirs.append((k, d))
        if on:
            watched.append(k)
    dirs = tuple(dirs)
    st.caption("Each signal is scored against the person's own normal, separately. 'Either way' watches both "
               "directions, for a compound whose effects are unknown.")
    reactions_on = st.toggle("Simulate a sleep reaction in 3 people", value=True,
                             help="Sleep efficiency drops in p01, p06 and p15 from day 2 (2.5, 2.0 and 1.5 "
                                  "of their own SDs). Everyone else is their real, unchanged data.")
    if st.button("Reset day and actions"):
        st.session_state["day"] = 0
        st.session_state["actions"] = {}

METRIC_KEYS = tuple(watched)


# ---------------------------------------------------------------- real substance: alcohol
@st.cache_data(show_spinner="Reading the alcohol reports…")
def alcohol_effects(root: str):
    """Per person: their surprise score (AR(1) residual against their own whole-record normal) on the day
    after a self-reported drinking day, minus on other days. Real data only."""
    rows, series = [], {}
    for pid in L.list_pmdata(root):
        f = os.path.join(root, pid, "googledocs", "reporting.csv")
        if not os.path.exists(f):
            continue
        x = pd.read_csv(f)
        x["d"] = pd.to_datetime(x["date"], format="%d/%m/%Y", errors="coerce")
        x = x[x["alcohol_consumed"].isin(["Yes", "No"])].dropna(subset=["d"])
        a = x.groupby("d")["alcohol_consumed"].agg(lambda v: "Yes" if (v == "Yes").any() else "No")
        row = {"Person": pid, "Drinking days": int((a == "Yes").sum()), "Other days": int((a == "No").sum())}
        for key in ALL_KEYS:
            try:
                sr = L.METRICS[key][2](root, pid)
            except (L.SubjectUnusable, ValueError):
                continue
            v = sr.to_numpy(dtype=float)
            mu = float(np.nanmean(v))
            phi = float(np.clip(E._lag1_autocorr(v, mu), -0.95, 0.95))
            e = E.residuals(v, mu, phi)
            z = pd.Series(e / np.nanstd(e), index=sr.index)
            nxt = a.copy()
            nxt.index = nxt.index + pd.Timedelta(days=1)       # drinking on day D -> that night / next day
            j = pd.concat([z.rename("z"), nxt.rename("a")], axis=1).dropna()
            yes, no = j[j["a"] == "Yes"]["z"], j[j["a"] == "No"]["z"]
            if len(yes) >= 5 and len(no) >= 5:
                row[key] = float(yes.mean() - no.mean())
            series[(pid, key)] = (sr, nxt[nxt == "Yes"].index, mu, float(np.nanstd(v)))
        rows.append(row)
    return pd.DataFrame(rows), series


def alcohol_page():
    st.caption("Real data only: PMData participants' own daily reports of whether they drank, and their Fitbit "
               "data. Nothing simulated.")
    st.title("Real substance: alcohol")
    df, series = alcohol_effects(root)
    cols = st.columns(2)
    for c, key in zip(cols, ALL_KEYS):
        met = L.METRICS[key][0]
        d = df[key].dropna() if key in df else pd.Series(dtype=float)
        harm = d > 0 if met.harm_direction == "up" else d < 0
        word = "higher" if met.harm_direction == "up" else "lower"
        with c:
            st.markdown(f"#### {L.METRICS[key][1]}")
            if len(d) < 5:
                st.info("Too few people with enough drinking and non-drinking days.")
                continue
            p = float(wilcoxon(d).pvalue)
            st.metric(f"{word} after drinking", f"{int(harm.sum())} of {len(d)} people")
            st.caption(f"Median difference {np.median(d):+.2f} of their own SDs · Wilcoxon p = {p:.3f}"
                       + ("" if p < 0.05 else " · no consistent effect"))
            fig, ax = plt.subplots(figsize=(6.2, 1.9))
            dd = d.sort_values()
            ax.axvline(0, color=C_MUTED, lw=1)
            ax.scatter(dd.to_numpy(), np.zeros(len(dd)), s=46,
                       color=[C_ALERT if (v > 0) == (met.harm_direction == "up") else C_MUTED for v in dd],
                       edgecolor="white", lw=1, zorder=3)
            ax.set_yticks([])
            ax.set_xlabel(f"after drinking minus other days, in each person's own SDs ({word} = red)", fontsize=8)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.tick_params(labelsize=8, colors=C_INK)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    st.subheader("One person")
    people = [pid for pid in df["Person"] if (pid, "resting_hr") in series and df.set_index("Person").loc[pid,
              "Drinking days"] >= 5]
    pid = st.selectbox("Person", people, index=people.index("p07") if "p07" in people else 0,
                       label_visibility="collapsed")
    cols = st.columns(2)
    for c, key in zip(cols, ALL_KEYS):
        if (pid, key) not in series:
            continue
        sr, drink_days, mu, sd = series[(pid, key)]
        met = L.METRICS[key][0]
        with c:
            fig, ax = plt.subplots(figsize=(6.2, 2.4))
            ax.axhspan(mu - 2 * sd, mu + 2 * sd, color=C_BAND, alpha=0.10, lw=0, label="their normal ± 2 SD")
            ax.plot(sr.index, sr.to_numpy(), lw=1, color=C_INK, marker="o", ms=1.8)
            dd = [d for d in drink_days if d in sr.index and not np.isnan(sr[d])]
            ax.scatter(dd, [sr[d] for d in dd], s=26, color=C_ALERT, zorder=3, label="day after drinking")
            ax.set_title(f"{pid} · {L.METRICS[key][1]} ({met.unit})", loc="left", fontsize=9, color=C_INK)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(labelsize=8, colors=C_INK)
            ax.legend(loc="upper left", fontsize=7, frameon=False, ncol=2)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    st.caption("Limits: drinking is self-reported yes/no with no amount, and people chose when to drink, so "
               "other things that come with drinking (late nights, social events) can contribute. Comparing "
               "only within the same weekday gives the same resting-HR result. 'Maybe' and 'prefer not to say' "
               "are left out; p01 never reported drinking; p15 has no reports.")


# ---------------------------------------------------------------- on/off trial for one person
@st.cache_data(show_spinner="Checking who has enough record for a trial…")
def trial_candidates(root: str, key: str):
    out = []
    for pid in L.list_pmdata(root):
        try:
            subj = L.load_pmdata(root, pid, START_DAY, key)
        except (L.SubjectUnusable, ValueError):
            continue
        n = E.max_cycles(subj.start, subj.series.last_valid_index())
        if n >= E.MIN_CYCLES:
            out.append((pid, n))
    return out


def trial_page():
    st.title("Is it working for this person?")
    c1, c2, _ = st.columns([1.2, 1, 2])
    key = c1.selectbox("Signal", ["sleep_efficiency", "resting_hr"], format_func=lambda k: L.METRICS[k][1],
                       key="trial_key")
    cands = trial_candidates(root, key)
    if not cands:
        st.error("No one has a long enough record for a 6-cycle trial on this signal.")
        return
    ids = [c[0] for c in cands]
    pid = c2.selectbox("Person", ids, index=ids.index("p01") if "p01" in ids else 0, key="trial_pid")
    n_cycles = E.MIN_CYCLES
    met = L.METRICS[key][0]
    subj = L.load_pmdata(root, pid, START_DAY, key)
    base = E.fit_baseline(subj)
    seed = 0
    sim = st.toggle("Simulated effect: compound works (2 SD better in ON weeks)", value=True, key="trial_sim")
    sched = E.make_schedule(subj.start, n_cycles, met, seed, record_end=subj.series.last_valid_index())
    scored = E.plant_effect(subj, sched, met, delta_sd=2.0) if sim else subj
    blocks = E.schedule_blocks(sched)
    end = blocks[-1][2] + pd.Timedelta(days=sched.block_days - 1)
    total_days = (end - subj.start).days + 1

    if key == "resting_hr":
        st.warning("Resting heart rate needs a large effect (about 3 SD) to show in a 6-cycle trial.")

    tday = st.slider("Trial day", 0, total_days, value=0, key=f"tday_{pid}_{key}_{n_cycles}_{seed}",
                     help=f"Random order, seed {seed} (recorded so it can be audited). "
                          f"{n_cycles} cycles; 6 is the minimum that can ever reach p < 0.05.")
    today = subj.start + pd.Timedelta(days=tday)
    rows = []
    for i, (cyc, arm, first, usable_from) in enumerate(blocks):
        last = first + pd.Timedelta(days=sched.block_days - 1)
        state = "done" if last < today else ("this week" if first <= today <= last else "")
        rows.append({"Week": i + 1, "Dates": f"{first:%b %d} – {last:%b %d}",
                     "Compound": "ON" if arm == "on" else "OFF", "Status": state, "_cycle": cyc})
    dt = pd.DataFrame(rows)

    def shade(row):
        if row["Status"] == "this week":
            return ["background-color: rgba(42,120,214,0.22); font-weight: 600"] * len(row)
        if row["Compound"] == "ON":
            return ["background-color: rgba(42,120,214,0.08)"] * len(row)
        return [""] * len(row)

    done_cycles = sum(1 for c in range(1, n_cycles + 1)
                      if all(r["Status"] == "done" for r in rows if r["_cycle"] == c))
    t1, t2 = st.columns([2, 3])
    with t1:
        st.markdown("**Dosing schedule**")
        st.dataframe(dt.drop(columns="_cycle").style.apply(lambda r: shade(r), axis=1), hide_index=True,
                     width="stretch", height=min(460, 38 + 35 * len(dt)))
        st.download_button("Download schedule", dt.drop(columns=["Status", "_cycle"]).to_csv(index=False).encode(),
                           file_name=f"{pid}_{key}_dosing_schedule.csv", mime="text/csv")
    with t2:
        this = dt[dt["Status"] == "this week"]
        m1, m2 = st.columns(2)
        m1.metric("This week", "—" if not len(this) else this["Compound"].iloc[0])
        m2.metric("Cycles done", f"{done_cycles} / {n_cycles}")

    # chart up to today
    x = scored.series[(scored.series.index >= subj.start - pd.Timedelta(days=21)) & (scored.series.index <= today)]
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    dd = pd.Timedelta(days=1)
    first_on = True
    for cyc, arm, first, usable_from in blocks:
        if arm == "on":
            ax.axvspan(first - dd / 2, first + pd.Timedelta(days=sched.block_days) - dd / 2, color=C_BAND, alpha=0.10,
                       lw=0, label="ON week" if first_on else None)
            first_on = False
        ax.axvspan(first - dd / 2, usable_from - dd / 2, facecolor="none", edgecolor=C_MUTED, hatch="////", lw=0)
    ax.plot(x.index, x.to_numpy(), lw=1.2, color=C_INK, marker="o", ms=2)
    ax.axvline(subj.start, color=C_INK, ls="--", lw=1)
    ax.axvline(today, color=C_BAND, lw=1, alpha=0.7)
    ax.set_xlim(subj.start - pd.Timedelta(days=22), end + pd.Timedelta(days=1))
    ax.set_ylabel(met.unit, fontsize=8)
    ax.set_title(f"{pid} · {L.METRICS[key][1]}" + (" · simulated effect" if sim else ""), loc="left", fontsize=9,
                 color=C_INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8, colors=C_INK)
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    fig.tight_layout()
    with t2:
        st.pyplot(fig)
    plt.close(fig)

    if done_cycles < n_cycles:
        with t2:
            st.info(f"Verdict after cycle {n_cycles}.")
        return
    eff = E.efficacy(scored, sched, met)
    tag = " (simulated effect)" if sim else ""
    with t2:
        if eff.verdict == "working":
            st.success(f"Working for {pid}: better in ON weeks in {eff.n_agree} of {eff.n_usable} cycles, "
                       f"p = {eff.p:.3f}.{tag}")
        elif eff.verdict == "opposite":
            st.error(f"Worse in ON weeks for {pid}: {eff.n_agree} of {eff.n_usable} cycles, p = {eff.p:.3f}.{tag}")
        elif eff.verdict == "none":
            st.info(f"No consistent ON/OFF difference for {pid} (p = {eff.p:.2f}). Not proof of no effect.{tag}")
        else:
            st.warning(eff.message + tag)


if view == "On/off trial":
    trial_page()
    st.stop()
if view == "Real substance: alcohol":
    alcohol_page()
    st.stop()
if not METRIC_KEYS:
    st.info("Choose at least one signal to watch in the sidebar.")
    st.stop()

unit = build_unit(root, reactions_on, dirs)
fp = false_alarm_rates(dict(dirs))
day = st.session_state["day"]
labels = {k: L.METRICS[k][1] for k in METRIC_KEYS}

# ---------------------------------------------------------------- header
st.caption("Simulated unit: real Fitbit data from civilian volunteers (PMData). The compound"
           + (" and the sleep reactions in three people are" if reactions_on else " is") + " simulated.")
h1, h2 = st.columns([3, 2])
h1.title(f"Unit roster · {len(unit)} people")
h1.caption(f"All started {compound_name} on day 0 and take it daily. "
           "Each person is watched against their own normal.")
with h2:
    st.markdown(f"### Day {day} of {COURSE - 1}")
    b1, b2, _ = st.columns([1, 1, 2])
    b1.button("Previous day", on_click=step, args=(-1,), disabled=day == 0)
    b2.button("Next day", on_click=step, args=(1,), disabled=day >= COURSE - 1, type="primary")
st.slider("Jump to day", 0, COURSE - 1, key="day", label_visibility="collapsed")

# ---------------------------------------------------------------- briefing
alerts, open_alerts = [], []
for pid, p in unit.items():
    for key in METRIC_KEYS:
        status, aday = status_at(p["metrics"][key], day)
        if status == "Alert":
            alerts.append((pid, key, aday))
            if (pid, key) not in st.session_state["actions"]:
                open_alerts.append((pid, key, aday))

st.subheader("Today's briefing")
if not open_alerts:
    st.success("No open alerts. Nobody has moved away from their own normal by more than day-to-day noise.")
for pid, key, aday in sorted(open_alerts, key=lambda a: a[2]):
    m = unit[pid]["metrics"][key]
    met = L.METRICS[key][0]
    today = m["S"].index[0] + pd.Timedelta(days=day)
    recent = m["series"][(m["series"].index <= today) & (m["series"].index > today - pd.Timedelta(days=7))].dropna()
    word = "above" if m["side"] == "up" else "below"
    with st.container(border=True):
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(f"**{pid} · {labels[key]} {word} their own normal** (since day {aday})")
            st.write(f"Last 7 days: {recent.mean():.1f} {met.unit}. Their normal: {m['mu']:.1f} ± {m['sd']:.1f} "
                     f"{met.unit}. Recommended: check in today.")
            if m["simulated"]:
                st.caption("Simulated reaction.")
        with c2:
            for a in ACTIONS:
                st.button(a, key=f"{pid}_{key}_{a}", on_click=act, args=(pid, key, a))

# ---------------------------------------------------------------- unit line
pairs = [(pid, k) for pid, p in unit.items() for k in METRIC_KEYS if p["metrics"][k] is not None]
frac = (day + 1) / COURSE
p_each = np.mean([fp[k] for _, k in pairs]) * frac
expected = len(pairs) * p_each
k_alerts = len(alerts)
tail = float(binom.sf(k_alerts - 1, len(pairs), p_each)) if k_alerts else 1.0
line = (f"**{k_alerts} alert{'s' if k_alerts != 1 else ''} so far.** With nothing given you'd expect about "
        f"{expected:.1f} by day {day} across {len(unit)} people and {len(METRIC_KEYS)} "
        f"signal{'s' if len(METRIC_KEYS) != 1 else ''}.")
if k_alerts and tail < 0.05:
    st.error(line + " That is more than chance: review the compound, not just individuals.")
else:
    st.info(line + " Not more than chance so far.")

# ---------------------------------------------------------------- roster
st.subheader("Roster")
rows = []
for pid, p in unit.items():
    row = {"Person": pid}
    for key in METRIC_KEYS:
        status, aday = status_at(p["metrics"][key], day)
        row[labels[key]] = status + (f" (day {aday})" if aday is not None else "")
    acts = [f"{labels[k]}: {a} (day {d})" for (q, k), (a, d) in st.session_state["actions"].items() if q == pid]
    row["Action taken"] = "; ".join(acts)
    rows.append(row)
df = pd.DataFrame(rows)
order = {"Alert": 0, "Drifting": 1, "Normal": 2, "No data": 3}
df["_o"] = df[[labels[k] for k in METRIC_KEYS]].apply(lambda r: min(order[v.split(" ")[0]] for v in r), axis=1)
df = df.sort_values(["_o", "Person"]).drop(columns="_o").reset_index(drop=True)


def colour(v):
    v = str(v)
    if v.startswith("Alert"):
        return f"background-color: #f8d7da; color: #842029"
    if v.startswith("Drifting"):
        return f"background-color: #fff3cd; color: #664d03"
    return ""


sty = df.style
sty = (sty.map if hasattr(sty, "map") else sty.applymap)(colour, subset=[labels[k] for k in METRIC_KEYS])
st.dataframe(sty, hide_index=True, width="stretch")
st.caption("Drifting = the running tally is past half of the alarm line. Alert = it crossed the line.")

# ---------------------------------------------------------------- person detail
st.subheader("Person")
first_alert = sorted(alerts, key=lambda a: a[2])[0][0] if alerts else df["Person"].iloc[0]
pid = st.selectbox("Person", list(df["Person"]), index=list(df["Person"]).index(first_alert),
                   label_visibility="collapsed")
cols = st.columns(len(METRIC_KEYS))
for c, key in zip(cols, METRIC_KEYS):
    m = unit[pid]["metrics"][key]
    met = L.METRICS[key][0]
    with c:
        if m is None:
            st.info(f"No {labels[key].lower()} data for {pid}.")
            continue
        start = m["S"].index[0]
        today = start + pd.Timedelta(days=day)
        x = m["series"]
        x = x[(x.index >= start - pd.Timedelta(days=BASELINE_DAYS)) & (x.index <= today)]
        fig, ax = plt.subplots(figsize=(6.2, 2.6))
        ax.axhspan(m["mu"] - 2 * m["sd"], m["mu"] + 2 * m["sd"], color=C_BAND, alpha=0.10, lw=0,
                   label="their normal ± 2 SD")
        ax.plot(x.index, x.to_numpy(), lw=1.3, color=C_INK, marker="o", ms=2)
        ax.axvline(start, color=C_INK, ls="--", lw=1, label="started compound")
        if m["alarm"] is not None and m["alarm"] <= today:
            ax.axvline(m["alarm"], color=C_ALERT, lw=1.8, label="alert")
        ax.set_xlim(start - pd.Timedelta(days=BASELINE_DAYS + 1), start + pd.Timedelta(days=COURSE))
        ax.set_title(f"{labels[key]} ({met.unit})" + (" · simulated reaction" if m["simulated"] else ""),
                     loc="left", fontsize=9, color=C_INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=8, colors=C_INK)
        ax.legend(loc="upper left", fontsize=7, frameon=False, ncol=3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        status, aday = status_at(m, day)
        st.caption(f"{status}" + (f" since day {aday}" if aday is not None else ""))

with st.expander("How alerts are decided"):
    st.markdown(
        "- **Their own normal.** The 60 days before the compound give each person's average and day-to-day "
        "wobble. Nobody is compared with a population range.\n"
        "- **A running tally.** Each day adds how far they are from what their own yesterday predicts; normal "
        "noise drains it back to zero, a sustained change builds it up.\n"
        f"- **The alarm line.** Set so that on real people who took nothing, {fp['resting_hr']:.0%} (resting HR) "
        f"and {fp['sleep_efficiency']:.1%} (sleep efficiency) of 30-day courses produce an alert. Those are "
        "alerts with nothing given, not necessarily errors: illness, alcohol or training move these signals "
        "too. The evidence is in `app.py`, Results page.")
