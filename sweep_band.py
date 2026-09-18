"""Entry-price band sweep (min x max) at a fixed 17:00-local cutoff and 0.5c
slippage cap. Same universe/rules as sweep_local.py otherwise.
"""
import json, sys, datetime as dt, bisect, itertools
from zoneinfo import ZoneInfo
import pandas as pd
from sweep_local import TZ, CITY_RE, EXCL, THRESH, STEP, NEED, STAKE, FEE_RATE, FILL_WINDOW

LOCAL_H = 17; CAP = 0.5; LOWER = -0.5
import os
SIDE = os.environ.get("SIDE", "Yes")          # which token we buy
OUT = "out/sweep_band" + ("" if SIDE == "Yes" else "_no")
MINS = [0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995]
MAXS = [0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 1.0]
BANDS = [(lo, hi) for lo in MINS for hi in MAXS if lo < hi]

def entries(series, cutoff):
    out = {}; run = 0; prev = None
    for t, p in series:
        ok = p >= THRESH
        run = run + 1 if (ok and prev is not None and t - prev <= 2 * STEP * 60 + 30) else (1 if ok else 0)
        prev = t if ok else None
        if run >= NEED and t >= cutoff:
            for b in BANDS:
                if b not in out and b[0] <= p <= b[1]:
                    out[b] = (t, p)
            if len(out) == len(BANDS): break
    return out

def main():
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
        yes = h if losing == SIDE else [(t, 1 - p) for t, p in h]   # price series of the token we buy
        day = dt.datetime.fromisoformat(m["end_date"].replace("Z", "+00:00")).date()
        cutoff = dt.datetime(day.year, day.month, day.day, LOCAL_H, tzinfo=ZoneInfo(TZ[city])).timestamp()
        ent = entries(yes, cutoff)
        if not ent: continue
        try: tape = json.load(open(f"data/tapes/{m['condition_id']}.json"))
        except FileNotFoundError: continue
        yes_tok = str(m["tokens"][m["outcomes"].index(SIDE)])
        buys = sorted((r["timestamp"], float(r["price"])) for r in tape if str(r["asset"]) == yes_tok and r["side"] == "BUY")
        bts = [b[0] for b in buys]
        for (lo, hi), (t, p) in ent.items():
            j = bisect.bisect_left(bts, t)
            fill = buys[j][1] if (j < len(buys) and buys[j][0] <= t + FILL_WINDOW) else None
            rows.append(dict(market_id=m["market_id"], city=city, lo=lo, hi=hi, signal_t=t, mid=p, fill=fill,
                             slip_c=None if fill is None else (fill - p) * 100, won=won))
        if i % 20000 == 0: print(i, flush=True)
    df = pd.DataFrame(rows); df.to_parquet(OUT + "_raw.parquet")
    summarize(df)

def summarize(df):
    f = df[df.fill.notna() & (df.slip_c >= LOWER - 1e-9) & (df.slip_c <= CAP + 1e-9)].copy()
    f["fee"] = STAKE / f.fill * FEE_RATE * f.fill * (1 - f.fill)
    f["pnl"] = (STAKE / f.fill - STAKE).where(f.won, -STAKE) - f.fee
    g = f.groupby(["lo", "hi"]).agg(trades=("pnl", "size"), lost=("won", lambda s: int((~s).sum())),
                                    avg_fill=("fill", "mean"), pnl=("pnl", "sum")).reset_index()
    g["loss_rate"] = g.lost / g.trades; g["roi"] = g.pnl / (g.trades * STAKE)
    g.to_csv(OUT + "_summary.csv", index=False)
    pd.set_option("display.width", 250)
    for col, name, scale in [("pnl", "P&L ($)", 1), ("trades", "trades", 1), ("lost", "losses", 1), ("roi", "ROI per trade (%)", 100)]:
        print(f"\n{name}: rows = min entry, cols = max entry")
        print((g.pivot(index="lo", columns="hi", values=col) * scale).round(2 if col == "roi" else 0).to_string())

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summarize": summarize(pd.read_parquet(OUT + "_raw.parquet"))
    else: main()
