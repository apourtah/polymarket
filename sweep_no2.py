"""NO-side band sweep down to 91c, 17:00 local, +-0.5c ask, on RANDOM samples of
signal markets (tapes cached from the YES runs are biased toward NO losers and
are not used unless the market is in a random sample).
  python3 sweep_no2.py signals | fetch N | summarize
"""
import json, os, sys, random, bisect, datetime as dt
from zoneinfo import ZoneInfo
import pandas as pd
from sweep_local import TZ, CITY_RE, EXCL, STAKE, FEE_RATE, FILL_WINDOW
from sweep_band2 import entries, BANDS, universe, LOCAL_H, CAP, LOWER
from slippage import tape as fetch_tape
SIDE = "No"

def signals():
    rows = []
    for i, m in enumerate(universe()):
        city = CITY_RE.match(m["question"]).group(1)
        try: d = json.load(open(f"data/prices/{m['market_id']}.json"))
        except FileNotFoundError: continue
        if not d["history"]: continue
        losing = d["losing_outcome"]; won = losing != SIDE
        h = sorted((x["t"], x["p"]) for x in d["history"])
        ser = h if losing == SIDE else [(t, 1 - p) for t, p in h]
        day = dt.datetime.fromisoformat(m["end_date"].replace("Z", "+00:00")).date()
        cutoff = dt.datetime(day.year, day.month, day.day, LOCAL_H, tzinfo=ZoneInfo(TZ[city])).timestamp()
        for (lo, hi), (t, p) in entries(ser, cutoff).items():
            rows.append(dict(market_id=m["market_id"], condition_id=m["condition_id"], city=city, lo=lo, hi=hi, signal_t=t, mid=p, won=won))
        if i % 20000 == 0: print(i, flush=True)
    df = pd.DataFrame(rows); df.to_parquet("out/no2_signals.parquet")
    print(f"{df.market_id.nunique()} markets with a NO signal; {len(df)} rows")

def fetch(n):
    """Random sample of markets with a signal in any sub-96 band (the >=96 bands are
    already covered by out/no_sample_ids.json); union saved to out/no2_sample_ids.json."""
    df = pd.read_parquet("out/no2_signals.parquet")
    prev = set(json.load(open("out/no_sample_ids.json")))
    low = sorted(set(df[df.lo < 0.96].market_id) - prev); random.seed(7); random.shuffle(low)
    sample = low[:n]; json.dump(sorted(prev | set(sample)), open("out/no2_sample_ids.json", "w"))
    cid = dict(zip(df.market_id, df.condition_id)); t0 = dict(df.groupby("market_id").signal_t.min())
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(4) as ex:
        for i, _ in enumerate(ex.map(lambda mid: fetch_tape(cid[mid], int(t0[mid]) - 3 * 86400), sample), 1):
            if i % 500 == 0: print(i, flush=True)
    print(f"done: {len(sample)} new + {len(prev)} previous")

def summarize():
    df = pd.read_parquet("out/no2_signals.parquet")
    prev = set(json.load(open("out/no_sample_ids.json"))); allsamp = set(json.load(open("out/no2_sample_ids.json")))
    new = allsamp - prev
    mk = {m["market_id"]: m for m in json.load(open("data/markets.json"))}
    s = df[df.market_id.isin(allsamp)].copy(); tapes = {}; fills = []
    for r in s.itertuples():
        if r.market_id not in tapes:
            m = mk[r.market_id]; tok = str(m["tokens"][m["outcomes"].index(SIDE)])
            try: tp = json.load(open(f"data/tapes/{m['condition_id']}.json"))
            except FileNotFoundError: tapes[r.market_id] = None; fills.append(None); continue
            buys = sorted((x["timestamp"], float(x["price"])) for x in tp if str(x["asset"]) == tok and x["side"] == "BUY")
            tapes[r.market_id] = (buys, [b[0] for b in buys])
        if tapes[r.market_id] is None: fills.append(None); continue
        buys, bts = tapes[r.market_id]; j = bisect.bisect_left(bts, r.signal_t)
        fills.append(buys[j][1] if (j < len(buys) and buys[j][0] <= r.signal_t + FILL_WINDOW) else None)
    s["fill"] = fills; s["slip_c"] = (s.fill - s.mid) * 100
    s.to_parquet("out/no2_fills.parquet")
    # per band: the unbiased sample is `prev` for lo>=0.96 bands (drawn from >=0.96 signal markets)
    # and `new` for lo<0.96 bands (drawn from sub-96 signal markets not in prev). Scale by population/sample.
    out = []
    for lo, hi in BANDS:
        pop = df[(df.lo == lo) & (df.hi == hi)]; samp_ids = prev if lo >= 0.96 else new
        sm = s[(s.lo == lo) & (s.hi == hi) & s.market_id.isin(samp_ids)]
        if len(sm) == 0: continue
        scale = len(pop) / len(sm)
        f = sm[sm.fill.notna() & (sm.slip_c >= LOWER - 1e-9) & (sm.slip_c <= CAP + 1e-9)].copy()
        if len(f) == 0: continue
        f["ret"] = ((1 / f.fill - 1) - FEE_RATE * (1 - f.fill)).where(f.won, -1.0)
        pl = pop.won.eq(False).mean()   # population loss rate for this band (all signals)
        gain = f[f.won].ret.mean() if f.won.any() else 0
        out.append(dict(band=f"{lo*100:g}-{hi*100:g}", lo=lo, hi=hi, pop_signals=len(pop), sample_signals=len(sm), accepted=len(f),
                        accept_rate=len(f) / len(sm), lost_in_sample=int((~f.won).sum()), pop_loss_rate=pl,
                        avg_fill=f.fill.mean(), gain_per_win=gain, roi_observed=f.ret.mean(),
                        roi_expected=(1 - pl) * gain - pl, trades_scaled=len(f) * scale))
    g = pd.DataFrame(out); g["pnl_1usd_expected"] = g.trades_scaled * g.roi_expected
    g.to_csv("out/sweep_no2_summary.csv", index=False)
    pd.set_option("display.width", 250)
    for col, name, sc in [("pnl_1usd_expected", "expected P&L at $1/trade (scaled to population)", 1), ("roi_expected", "expected ROI (%)", 100),
                          ("trades_scaled", "trades (scaled)", 1), ("pop_loss_rate", "population loss rate (%)", 100), ("accept_rate", "accept rate (%)", 100)]:
        print(f"\n{name}: rows = min, cols = max")
        print((g.pivot(index="lo", columns="hi", values=col) * sc).round(2).to_string())

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "signals": signals()
    elif cmd == "fetch": fetch(int(sys.argv[2]))
    else: summarize()
