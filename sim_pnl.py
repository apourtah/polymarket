"""Usage: python3 sim_pnl.py [MAX_ENTRY] [MIN_ENTRY] [CUTOFF_HRS]

Simulate: buy $100 of an outcome the first time it has been >=96c for 30
contiguous minutes AND we are within 6h of the market's end date (or later).
Hold to resolution. Uses the 5-min midpoint series in data/prices/ for every
market (the winning side's series is 1 - the losing side's series).

Rules mirror analyze.py's flag_strong, minus the trade-tape volume rule (the
tape was only pulled for candidates, not for winners).
"""
import json, glob, sys, re, datetime as dt
import pandas as pd

THRESH = 0.96; RUN_MIN = 30; STAKE = 100.0; STEP = 5  # 5-min samples
CUTOFF_HRS = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0   # enter no earlier than end_date - CUTOFF_HRS
MAX_ENTRY = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0   # skip entries priced above this
MIN_ENTRY = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0   # ...or below this
FEE_RATE = 0.05          # weather taker fee: shares * 0.05 * p * (1-p)
EXCLUDE = re.compile(r"or higher|or above", re.I)   # upper-tail buckets: the one market type with negative ROI
NEED = RUN_MIN // STEP                # consecutive samples for a qualifying run

def parse_ts(s):
    s = s.replace(" ", "T").replace("+00", "+00:00")
    if s.endswith("Z"): s = s[:-1] + "+00:00"
    return dt.datetime.fromisoformat(s).timestamp()

def entry(series, not_before):
    """First (t, p) where the last NEED samples are all >= THRESH, contiguous
    (<= 2 steps apart), and t >= not_before. None if never."""
    run = 0; prev = None
    for t, p in series:
        run = run + 1 if (p >= THRESH and prev is not None and t - prev <= 2 * STEP * 60 + 30) else (1 if p >= THRESH else 0)
        prev = t if p >= THRESH else None
        if run >= NEED and t >= not_before and MIN_ENTRY <= p <= MAX_ENTRY:
            return t, p
    return None

def main():
    markets = {m["market_id"]: m for m in json.load(open("data/markets.json"))}
    rows = []
    for f in glob.glob("data/prices/*.json"):
        d = json.load(open(f)); m = markets.get(d["market_id"])
        if not m or not d["history"] or not m.get("end_date"): continue
        if EXCLUDE.search(m["question"] or ""): continue
        h = sorted(((x["t"], x["p"]) for x in d["history"]))
        not_before = parse_ts(m["end_date"]) - CUTOFF_HRS * 3600
        losing = d["losing_outcome"]
        for side in ("Yes", "No"):
            ser = h if losing == side else [(t, 1 - p) for t, p in h]
            e = entry(ser, not_before)
            if e is None: continue
            won = losing != side
            fee = STAKE / e[1] * FEE_RATE * e[1] * (1 - e[1])
            pnl = (STAKE / e[1] - STAKE if won else -STAKE) - fee
            rows.append(dict(market_id=m["market_id"], question=m["question"], side=side,
                             entry_time=dt.datetime.fromtimestamp(e[0], dt.timezone.utc),
                             entry_price=e[1], won=won, pnl=round(pnl, 2),
                             url="https://polymarket.com/event/" + m["event_slug"]))
    df = pd.DataFrame(rows)
    tag = f"max{MAX_ENTRY}" + (f"_min{MIN_ENTRY}" if MIN_ENTRY else "") + (f"_cut{CUTOFF_HRS:g}" if CUTOFF_HRS != 6 else "")
    df.to_csv(f"out/sim_positions_{tag}.csv", index=False)
    days = (df.entry_time.max() - df.entry_time.min()).days + 1
    for label, sub in [("YES side only", df[df.side == "Yes"]), ("either side", df)]:
        n = len(sub); w = int(sub.won.sum())
        print(f"\n== {label}: {n} positions, {w} won / {n-w} lost (win rate {w/n:.2%})")
        print(f"   avg entry {sub.entry_price.mean():.4f}   staked ${n*STAKE:,.0f}")
        print(f"   gross from winners +${sub[sub.won].pnl.sum():,.0f}   losers -${-sub[~sub.won].pnl.sum():,.0f}")
        print(f"   NET P&L after fees ${sub.pnl.sum():,.0f}   ROI {sub.pnl.sum()/(n*STAKE):+.2%}   breakeven win rate {1/(1+ (1/sub.entry_price.mean()-1)):.2%}   entries/day {n/days:.1f}")

if __name__ == "__main__":
    main()
