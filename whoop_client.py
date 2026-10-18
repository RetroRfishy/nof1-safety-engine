"""WHOOP client: recovery records -> a Subject the engine can monitor.

This is the real loader, not a mock. It is written against the documented
response shape and points at whoop_sim.py by default; switching to the live
service is a base URL and a bearer token.

    from whoop_client import fetch
    snap = fetch(base_url="http://127.0.0.1:8787", token=None)
    subject = snap.to_subject(start="2020-01-01")

THE ONE MODELLING DECISION, stated rather than buried. A recovery record has no
date on it. It has a `cycle_id`, and a cycle runs from waking one day to waking
the next. So this client joins recovery -> cycle and uses the cycle's `end`
date: the morning the recovery is scored for. That is the conventional mapping
and the one a clinician would expect, but it is a choice, and it decides what
"consecutive days" means for the AR(1) fit in engine.fit_baseline. A different
mapping (cycle start, or local midnight) shifts every reading by a day and
changes phi. Write down whichever you use.

Records with score_state != "SCORED" are dropped, not coerced: a PENDING_SCORE
is an absent reading, which is exactly a NaN day, and an UNSCORABLE one is the
device telling you it could not measure.

CONFIRMED from the WHOOP docs: base path /developer/v1, the
{"records": [...], "next_token": ...} envelope, the `limit` parameter,
newest-first ordering, score_state values, and the cycle fields
id/start/end/timezone_offset. ASSUMED and flagged in-line: the nextToken
parameter spelling and the date-range parameter names. Verify both against the
API reference before pointing this at api.prod.whoop.com.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import requests

from adapters import to_subject

BASE_PATH = "/developer/v1"
LIVE_BASE = "https://api.prod.whoop.com"
PAGE = 25

# WHOOP reports resting heart rate in bpm, which is what the demo Metric expects.
SCORE_FIELD = "resting_heart_rate"


@dataclass(frozen=True)
class Snapshot:
    """What the API showed at one moment. `revisions` is why this is a snapshot
    and not just a series: a value already served can change later, and the
    CUSUM is path-dependent, so the caller has to know when that happened."""
    values: pd.Series                 # date -> value, days with no record absent
    updated_at: pd.Series             # date -> last-updated timestamp
    n_records: int
    dropped_unscored: int

    def to_subject(self, start, subject_id: str = "whoop", **kw):
        return to_subject(subject_id, self.values.index, self.values.to_numpy(),
                          start, **kw)

    def revisions_since(self, earlier: "Snapshot | None") -> list[tuple[pd.Timestamp, float, float]]:
        """(date, old value, new value) for every day whose value changed since
        `earlier`. The engine must rescore from the earliest of these forward."""
        if earlier is None:
            return []
        out = []
        for d, v in self.values.items():
            if d in earlier.values.index and earlier.values[d] != v:
                out.append((d, float(earlier.values[d]), float(v)))
        return out


def _paginate(session, url: str, headers: dict, limit: int = PAGE):
    """Walk a WHOOP collection endpoint. `nextToken` spelling is ASSUMED."""
    token, seen = None, 0
    while True:
        params = {"limit": limit}
        if token:
            params["nextToken"] = token
        r = session.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        body = r.json()
        for rec in body.get("records", []):
            yield rec
            seen += 1
        token = body.get("next_token")
        if not token:
            return
        if seen > 20000:                       # a runaway page loop, not a long record
            raise RuntimeError("pagination did not terminate")


def fetch(base_url: str = "http://127.0.0.1:8787", token: str | None = None,
          *, score_field: str = SCORE_FIELD) -> Snapshot:
    """Pull every visible recovery and return a Snapshot. One call per
    collection: cycles for the dates, recoveries for the values."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    s = requests.Session()

    # 1. cycles: cycle_id -> the calendar day the cycle ended on
    day_of: dict[int, pd.Timestamp] = {}
    for c in _paginate(s, f"{base_url}{BASE_PATH}/cycle", headers):
        if c.get("end") is None:               # the cycle in progress has no end yet
            continue
        day_of[c["id"]] = pd.Timestamp(c["end"]).tz_localize(None).normalize()

    # 2. recoveries: join to a day, keep the scored ones
    vals: dict[pd.Timestamp, float] = {}
    upd: dict[pd.Timestamp, pd.Timestamp] = {}
    n, dropped = 0, 0
    for rec in _paginate(s, f"{base_url}{BASE_PATH}/recovery", headers):
        n += 1
        if rec.get("score_state") != "SCORED":
            dropped += 1
            continue
        d = day_of.get(rec.get("cycle_id"))
        if d is None:
            dropped += 1                       # recovery whose cycle we cannot date
            continue
        v = (rec.get("score") or {}).get(score_field)
        if v is None:
            dropped += 1
            continue
        vals[d] = float(v)
        upd[d] = pd.Timestamp(rec["updated_at"]).tz_localize(None)

    if not vals:
        raise RuntimeError(f"no scored recoveries with a dateable cycle at {base_url}")
    idx = pd.Index(sorted(vals))
    return Snapshot(pd.Series([vals[d] for d in idx], index=idx),
                    pd.Series([upd[d] for d in idx], index=idx), n, dropped)


def clock(base_url: str = "http://127.0.0.1:8787", advance: int | None = None,
          day: int | None = None) -> dict:
    """Simulator only. The live API has no clock; this is how the replay moves."""
    if advance is None and day is None:
        return requests.get(f"{base_url}/sim/clock", timeout=10).json()
    p = {"day": day} if day is not None else {"advance": advance}
    return requests.post(f"{base_url}/sim/clock", params=p, timeout=10).json()
