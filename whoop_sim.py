"""A fake WHOOP API, backed by PMData, for exercising the live path.

Why this exists. Everything in the build is retrospective: the whole series is
on disk before anything is computed. A live feed differs in three ways that no
amount of reasoning settles, so this server produces them on purpose:

  1. LATE DELIVERY. A day's record does not appear the moment the day ends.
  2. REVISION. A value is restated days after it was first served (Fitbit and
     WHOOP both do this). The CUSUM is path-dependent, so a revised day forces
     a rescore from that day forward, and an alarm already raised may un-raise.
  3. GAPS. A day the device was not worn produces no record at all, not a zero.

Run it, point whoop_client at it, and the failure modes become something you can
watch instead of something you argue about.

    python whoop_sim.py --root data/pmdata --port 8787

Endpoints mirror the shape of the real API (confirmed from the WHOOP docs):
  GET /developer/v1/recovery?limit=&nextToken=   -> {"records": [...], "next_token": ...}
  GET /developer/v1/user/profile/basic
Plus two that the real API does not have, for driving the simulation:
  GET /sim/clock, POST /sim/clock?day=N | ?advance=1

CONFIRMED from the docs: the base path, the {"records": [...], "next_token": ...}
envelope, `limit`, newest-first ordering, score_state in
{SCORED, PENDING_SCORE, UNSCORABLE}, and the created_at / updated_at /
cycle_id / user_id / score fields.
ASSUMED, and marked so nobody mistakes a guess for a fact: the exact
`start`/`end` query parameter names and the nextToken parameter spelling. Check
them against the API reference before pointing the client at the real service.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

BASE = "/developer/v1"


@dataclass
class SimConfig:
    delivery_lag_days: int = 1      # a day's record appears this many days later
    revise_after_days: int = 2      # a served value may be restated this long after
    revise_fraction: float = 0.15   # share of days that get restated
    revise_sd: float = 0.4          # size of the restatement, in baseline SDs
    seed: int = 0


@dataclass
class SimState:
    series: pd.Series               # one subject, complete daily index, NaN = not worn
    user_id: int
    cfg: SimConfig
    day: int = 0                    # simulated "now" = series.index[0] + day
    _revised: dict = field(default_factory=dict)   # date -> (new_value, revised_on_day)

    def __post_init__(self):
        rng = np.random.default_rng(self.cfg.seed)
        sd = float(np.nanstd(self.series.to_numpy(), ddof=1))
        for i, (d, v) in enumerate(self.series.items()):
            if np.isnan(v) or rng.random() >= self.cfg.revise_fraction:
                continue
            delta = rng.standard_normal() * self.cfg.revise_sd * sd
            self._revised[d] = (float(v + delta), i + self.cfg.delivery_lag_days
                                + self.cfg.revise_after_days)

    @property
    def now(self) -> pd.Timestamp:
        return self.series.index[0] + pd.Timedelta(days=self.day)

    def visible(self) -> list[tuple[pd.Timestamp, float, int]]:
        """(date, value, version) for every record a client can see right now.
        version 1 = as first served, 2 = restated."""
        out = []
        for i, (d, v) in enumerate(self.series.items()):
            if np.isnan(v):
                continue                                   # not worn: no record at all
            if i + self.cfg.delivery_lag_days > self.day:
                continue                                   # not delivered yet
            version = 1
            if d in self._revised:
                new_v, on_day = self._revised[d]
                if self.day >= on_day:
                    v, version = new_v, 2
            out.append((d, float(v), version))
        return out


STATE: SimState | None = None


def _record(d: pd.Timestamp, value: float, version: int, user_id: int) -> dict:
    """A recovery record in the documented envelope. The cycle ends on the
    morning of `d`, which is the mapping the client must undo."""
    cycle_end = d.normalize() + pd.Timedelta(hours=7)
    created = cycle_end + pd.Timedelta(days=STATE.cfg.delivery_lag_days)
    updated = created if version == 1 else created + pd.Timedelta(
        days=STATE.cfg.revise_after_days)
    return {
        "cycle_id": int(d.toordinal()),
        "sleep_id": f"{d.date()}-sleep",
        "user_id": user_id,
        "created_at": created.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "updated_at": updated.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "score_state": "SCORED",
        "score": {
            "user_calibrating": False,
            "recovery_score": None,
            "resting_heart_rate": round(value, 2),
            "hrv_rmssd_milli": None,
            "spo2_percentage": None,
            "skin_temp_celsius": None,
        },
        # not part of the real schema; lets the demo show revisions plainly
        "_sim_version": version,
        "_sim_date": str(d.date()),
    }


async def recovery(request):
    limit = int(request.query_params.get("limit", 10))
    token = request.query_params.get("nextToken")
    rows = sorted(STATE.visible(), key=lambda r: r[0], reverse=True)   # newest first
    offset = int(token) if token else 0
    page = rows[offset:offset + limit]
    records = [_record(d, v, ver, STATE.user_id) for d, v, ver in page]
    nxt = str(offset + limit) if offset + limit < len(rows) else None
    return JSONResponse({"records": records, "next_token": nxt})


def _cycle(d: pd.Timestamp, user_id: int) -> dict:
    """The cycle a recovery belongs to. A cycle runs from waking one day to
    waking the next, so its `end` is the morning the recovery is scored for.
    The client joins recovery.cycle_id to this to get a calendar date; there is
    no date field on a recovery record itself."""
    end = d.normalize() + pd.Timedelta(hours=7)
    return {
        "id": int(d.toordinal()),
        "user_id": user_id,
        "created_at": (end + pd.Timedelta(days=STATE.cfg.delivery_lag_days)
                       ).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "updated_at": (end + pd.Timedelta(days=STATE.cfg.delivery_lag_days)
                       ).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "start": (end - pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "timezone_offset": "+00:00",
        "score_state": "SCORED",
        "score": {"strain": None, "kilojoule": None,
                  "average_heart_rate": None, "max_heart_rate": None},
    }


async def cycle(request):
    limit = int(request.query_params.get("limit", 10))
    token = request.query_params.get("nextToken")
    rows = sorted(STATE.visible(), key=lambda r: r[0], reverse=True)
    offset = int(token) if token else 0
    page = rows[offset:offset + limit]
    records = [_cycle(d, STATE.user_id) for d, _, _ in page]
    nxt = str(offset + limit) if offset + limit < len(rows) else None
    return JSONResponse({"records": records, "next_token": nxt})


async def profile(request):
    return JSONResponse({"user_id": STATE.user_id, "email": "subject@example.test",
                         "first_name": "Demo", "last_name": "Subject"})


async def clock_get(request):
    rows = STATE.visible()
    return JSONResponse({"day": STATE.day, "now": str(STATE.now.date()),
                         "records_visible": len(rows),
                         "revisions_served": sum(1 for _, _, v in rows if v == 2)})


async def clock_set(request):
    if "day" in request.query_params:
        STATE.day = int(request.query_params["day"])
    else:
        STATE.day += int(request.query_params.get("advance", 1))
    return await clock_get(request)


app = Starlette(routes=[
    Route(f"{BASE}/recovery", recovery),
    Route(f"{BASE}/cycle", cycle),
    Route(f"{BASE}/user/profile/basic", profile),
    Route("/sim/clock", clock_get),
    Route("/sim/clock", clock_set, methods=["POST"]),
])


def build_state(root: str, pid: str | None, cfg: SimConfig) -> SimState:
    from loaders import demo_subject_id, read_rhr
    pid = pid or demo_subject_id(root)
    return SimState(series=read_rhr(root, pid), user_id=abs(hash(pid)) % 10**7, cfg=cfg)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="PMData root")
    ap.add_argument("--subject", default=None, help="participant id (default: demo subject)")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--day", type=int, default=0, help="initial simulated day")
    ap.add_argument("--lag", type=int, default=1, help="delivery lag, days")
    ap.add_argument("--revise-fraction", type=float, default=0.15)
    args = ap.parse_args()

    global STATE
    cfg = SimConfig(delivery_lag_days=args.lag, revise_fraction=args.revise_fraction)
    STATE = build_state(args.root, args.subject, cfg)
    STATE.day = args.day
    n = len(STATE.series)
    print(f"fake WHOOP: {n} days, {int(STATE.series.notna().sum())} worn, "
          f"{len(STATE._revised)} will be restated")
    print(f"  serving http://127.0.0.1:{args.port}{BASE}/recovery")
    print(f"  clock    http://127.0.0.1:{args.port}/sim/clock")

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
