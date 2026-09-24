#!/usr/bin/env python3
"""Re-check the recorded daily maxima in the bot's history against the METAR archive, and correct the stale ones.

bot/backfill.py rebuilds city-days that are MISSING from data/evening_history.parquet, but nothing ever re-checks a
row that was written with a wrong value.  A row can be wrong when the live METAR feed was incomplete at the moment
score_pending() finalised it -- the same partial-coverage failure that made bot/positions.py report Los Angeles
2026-09-22 as 74F (obs from 14:53 only) when the day's max was 76F at 11:53.  A wrong `actual` is worse in the
history than on screen: it feeds `err`, and through it the EWMA level, the ridge fit and every lag feature.

A row is corrected only when the archive holds a demonstrably COMPLETE day (>= 20 distinct hours, an observation at
22:00 local or later, and >= 6 of the hours 11-18) and its max differs from what the history stored.  Correcting a
row updates `actual` and `err`, refreshes `yday_max`/`yday_min` on that city's next row, and recomputes the lag
features (yday_err, e2, r5, r14) for that city from the earliest correction onward -- r14 means a single bad day
propagates up to a fortnight forward.

  python3 bot/revalidate.py             report what would change (default: no writes)
  python3 bot/revalidate.py --apply     write the corrections, after backing the history up
  python3 bot/revalidate.py --from 2026-09-01
"""
import os, sys, datetime as dt
import numpy as np, pandas as pd
from decimal import Decimal, ROUND_HALF_UP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bot import evening_config as C
from bot.backfill import add_lag_features

rh = lambda x: int(Decimal(str(x)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
HIST = C.HISTORY if os.path.isabs(C.HISTORY) else f"{ROOT}/{C.HISTORY}"
METAR = f"{ROOT}/data/metar.parquet"


def archive_days():
    """{(city, day): (max_f, min_f)} for every day the archive covers completely."""
    m = pd.read_parquet(METAR, columns=["city", "local_time", "temp_c"])
    m["tf"] = (m.temp_c * 9 / 5 + 32).map(rh); m["d"] = m.local_time.dt.date; m["h"] = m.local_time.dt.hour
    g = m.groupby(["city", "d"]).agg(mx=("tf", "max"), mn=("tf", "min"), hours=("h", "nunique"),
                                     last=("h", "max"), aft=("h", lambda s: s[(s >= 11) & (s <= 18)].nunique()))
    ok = g[(g.hours >= 20) & (g.last >= 22) & (g.aft >= 6)]
    return {k: (int(v.mx), int(v.mn)) for k, v in ok.iterrows()}


def revalidate(d_from=None, apply=False):
    H = pd.read_parquet(HIST); H["mday"] = pd.to_datetime(H.mday)
    arch = archive_days()
    day = H.mday.dt.date
    cand = H.actual.notna() & (day >= d_from if d_from else True)
    bad = []
    for i, r in H[cand].iterrows():
        a = arch.get((r.city, r.mday.date()))
        if a and int(r.actual) != a[0]: bad.append((i, r.city, r.mday.date(), int(r.actual), a[0]))
    if not bad:
        print(f"history clean: {int(cand.sum())} scored rows checked against complete archive days, 0 stale"); return 0
    print(f"{int(cand.sum())} scored rows checked against complete archive days; {len(bad)} stale:\n")
    print(f"  {'city':<14}{'day':<12}{'stored':>7}{'archive':>9}{'delta':>7}{'err':>9} -> {'err':>7}")
    for i, c, d, old, new in bad:
        e0 = H.at[i, "err"]; print(f"  {c:<14}{str(d):<12}{old:>7}{new:>9}{new-old:>+7}{e0:>9.2f} -> {new - H.at[i,'hrrr']:>7.2f}")

    first = {}
    for i, c, d, old, new in bad:
        H.at[i, "actual"] = new; H.at[i, "err"] = float(new - H.at[i, "hrrr"])
        first[c] = min(first.get(c, d), d)
        nxt = H[(H.city == c) & (H.mday.dt.date == d + dt.timedelta(days=1))]      # that day is the NEXT row's "yesterday"
        for j in nxt.index:
            H.at[j, "yday_max"] = float(new); H.at[j, "yday_min"] = float(arch[(c, d)][1])

    H2 = add_lag_features(H.copy())
    mask = [(c in first and dd.date() >= first[c]) for c, dd in zip(H2.city, H2.mday)]   # only from the first correction on
    lag = ["yday_err", "e2", "r5", "r14"]
    changed = 0
    H2i = H2.set_index(["city", "mday"]); Hi = H.set_index(["city", "mday"])
    for (c, d), row in H2i[mask].iterrows():
        for col in lag:
            if not (pd.isna(row[col]) and pd.isna(Hi.at[(c, d), col])) and row[col] != Hi.at[(c, d), col]:
                Hi.at[(c, d), col] = row[col]; changed += 1
    H = Hi.reset_index()
    print(f"\n{len(bad)} row(s) corrected, {changed} lag-feature value(s) recomputed from: "
          + ", ".join(f"{c} {d}+" for c, d in sorted(first.items())))
    if not apply:
        print("\ndry run: nothing written. Re-run with --apply to write the corrections."); return len(bad)
    bak = f"{HIST}.bak-{dt.datetime.now().strftime('%Y%m%d%H%M%S')}"
    pd.read_parquet(HIST).to_parquet(bak); print(f"backed up -> {bak}")
    H.sort_values(["city", "mday"]).to_parquet(HIST)
    print(f"written: {HIST} ({len(H)} rows through {H.mday.max().date()})")
    return len(bad)


if __name__ == "__main__":
    a = sys.argv[1:]
    d_from = dt.date.fromisoformat(a[a.index("--from") + 1]) if "--from" in a else None
    revalidate(d_from=d_from, apply="--apply" in a)
