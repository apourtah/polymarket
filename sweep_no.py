"""NO-side version of the band sweep (17:00 local, 0.5c slip cap), on a random
sample of signal markets because losing buckets have no cached tape.
  python3 sweep_no.py signals   -> out/no_signals.parquet (all markets, no tape needed)
  python3 sweep_no.py fetch N   -> pull tapes for a random N-market sample of signal markets
  python3 sweep_no.py summarize -> fills + P&L on the sampled markets, scaled to the population
"""
import json, os, sys, random, bisect, datetime as dt
from zoneinfo import ZoneInfo
import pandas as pd
from sweep_local import TZ, CITY_RE, EXCL, STAKE, FEE_RATE, FILL_WINDOW
import sweep_band as sb
from slippage import tape as fetch_tape   # cached, throttled data-api pull

SIDE = "No"; LOCAL_H = 17; CAP = 0.5; LOWER = -0.5

def signals():
    markets = [m for m in json.load(open("data/markets.json"))
               if CITY_RE.match(m["question"] or "") and not EXCL.search(m["question"])]
    rows = []
    for i, m in enumerate(markets):
        city = CITY_RE.match(m["question"]).group(1)
        try: d = json.load(open(f"data/prices/{m['market_id']}.json"))
        except FileNotFoundError: continue
        if not d["history"]: continue
        losing = d["losing_outcome"]; won = losing != SIDE
        h = sorted((x["t"], x["p"]) for x in d["history"])
        ser = h if losing == SIDE else [(t, 1 - p) for t, p in h]
        day = dt.datetime.fromisoformat(m["end_date"].replace("Z", "+00:00")).date()
        cutoff = dt.datetime(day.year, day.month, day.day, LOCAL_H, tzinfo=ZoneInfo(TZ[city])).timestamp()
        for (lo, hi), (t, p) in sb.entries(ser, cutoff).items():
            rows.append(dict(market_id=m["market_id"], condition_id=m["condition_id"], city=city,
                             lo=lo, hi=hi, signal_t=t, mid=p, won=won))
        if i % 20000 == 0: print(i, flush=True)
    df = pd.DataFrame(rows); df.to_parquet("out/no_signals.parquet")
    print(f"{df.market_id.nunique()} markets with a NO signal; {len(df)} (market, band) rows")

def fetch(n):
    os.makedirs("data/tapes", exist_ok=True)
    df = pd.read_parquet("out/no_signals.parquet")
    ids = sorted(df.market_id.unique()); random.seed(42); random.shuffle(ids)
    sample = ids[:n]; json.dump(sample, open("out/no_sample_ids.json", "w"))
    cid = dict(zip(df.market_id, df.condition_id)); t0 = dict(df.groupby("market_id").signal_t.min())
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(4) as ex:
        for i, _ in enumerate(ex.map(lambda mid: fetch_tape(cid[mid], int(t0[mid]) - 3 * 86400), sample), 1):
            if i % 500 == 0: print(i, flush=True)
    print("done")

def summarize():
    df = pd.read_parquet("out/no_signals.parquet")
    sample = set(json.load(open("out/no_sample_ids.json")))
    mk = {m["market_id"]: m for m in json.load(open("data/markets.json"))}
    s = df[df.market_id.isin(sample)].copy()
    fills = []
    tapes = {}
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
    scale = df.market_id.nunique() / len(sample)
    f = s[s.fill.notna() & (s.slip_c >= LOWER - 1e-9) & (s.slip_c <= CAP + 1e-9)].copy()
    f["fee"] = STAKE / f.fill * FEE_RATE * f.fill * (1 - f.fill)
    f["pnl"] = (STAKE / f.fill - STAKE).where(f.won, -STAKE) - f.fee
    g = f.groupby(["lo", "hi"]).agg(trades=("pnl", "size"), lost=("won", lambda x: int((~x).sum())),
                                    avg_fill=("fill", "mean"), pnl=("pnl", "sum")).reset_index()
    g["roi"] = g.pnl / (g.trades * STAKE); g["loss_rate"] = g.lost / g.trades
    g["trades_scaled"] = (g.trades * scale).round(); g["pnl_scaled"] = (g.pnl * scale).round()
    g.to_csv("out/sweep_band_no_summary.csv", index=False)
    pd.set_option("display.width", 250)
    nb = s[(s.lo == 0.96) & (s.hi == 1.0)]
    print(f"sample: {len(sample)} of {df.market_id.nunique()} signal markets (scale x{scale:.1f})")
    print(f"no-band signals in sample: {len(nb)}; no print within 60 min: {int(nb.fill.isna().sum())}; "
          f"slip-rejected: {int(((nb.slip_c > CAP) | (nb.slip_c < LOWER)).sum())}; accepted: {int((nb.slip_c.abs() <= CAP).sum())}")
    for col, name, sc in [("pnl_scaled", "P&L ($, scaled to full population)", 1), ("trades_scaled", "trades (scaled)", 1),
                          ("lost", "losses (in sample)", 1), ("roi", "ROI per trade (%)", 100), ("loss_rate", "loss rate (%)", 100)]:
        print(f"\n{name}: rows = min entry, cols = max entry")
        print((g.pivot(index="lo", columns="hi", values=col) * sc).round(2 if sc == 100 else 0).to_string())

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "signals": signals()
    elif cmd == "fetch": fetch(int(sys.argv[2]))
    else: summarize()
