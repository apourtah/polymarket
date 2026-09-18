"""YES-side entry-price band sweep, extended down to 91c.
Rule: 17:00 local cutoff; YES has been >= band-min for 30 contiguous minutes;
midpoint within [min, max]; fill = first taker BUY of YES within 60 min, within
[mid-0.5c, mid+0.5c]; $100 stake; fees 0.05*p*(1-p).
  python3 sweep_band2.py signals | fetch | summarize
"""
import json, os, sys, bisect, datetime as dt
from zoneinfo import ZoneInfo
import pandas as pd
from sweep_local import TZ, CITY_RE, EXCL, STEP, NEED, STAKE, FEE_RATE, FILL_WINDOW
from slippage import tape as fetch_tape

LOCAL_H = 17; CAP = 0.5; LOWER = -0.5
MINS = [0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
MAXS = [0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 1.0]
BANDS = [(lo, hi) for lo in MINS for hi in MAXS if lo < hi]

def entries(series, cutoff):
    """Per band: first (t,p) after cutoff where p has been >= lo for NEED samples and lo<=p<=hi."""
    out = {}; run = {lo: 0 for lo in MINS}; prev = {lo: None for lo in MINS}
    for t, p in series:
        for lo in MINS:
            ok = p >= lo
            run[lo] = run[lo] + 1 if (ok and prev[lo] is not None and t - prev[lo] <= 2 * STEP * 60 + 30) else (1 if ok else 0)
            prev[lo] = t if ok else None
        if t >= cutoff:
            for lo, hi in BANDS:
                if (lo, hi) not in out and run[lo] >= NEED and lo <= p <= hi:
                    out[(lo, hi)] = (t, p)
            if len(out) == len(BANDS): break
    return out

def universe():
    return [m for m in json.load(open("data/markets.json"))
            if CITY_RE.match(m["question"] or "") and not EXCL.search(m["question"])]

def signals():
    rows = []
    for i, m in enumerate(universe()):
        city = CITY_RE.match(m["question"]).group(1)
        try: d = json.load(open(f"data/prices/{m['market_id']}.json"))
        except FileNotFoundError: continue
        if not d["history"]: continue
        losing = d["losing_outcome"]; won = losing == "No"
        h = sorted((x["t"], x["p"]) for x in d["history"])
        yes = h if losing == "Yes" else [(t, 1 - p) for t, p in h]
        day = dt.datetime.fromisoformat(m["end_date"].replace("Z", "+00:00")).date()
        cutoff = dt.datetime(day.year, day.month, day.day, LOCAL_H, tzinfo=ZoneInfo(TZ[city])).timestamp()
        for (lo, hi), (t, p) in entries(yes, cutoff).items():
            rows.append(dict(market_id=m["market_id"], condition_id=m["condition_id"], city=city,
                             lo=lo, hi=hi, signal_t=t, mid=p, won=won))
        if i % 20000 == 0: print(i, flush=True)
    df = pd.DataFrame(rows); df.to_parquet("out/yes2_signals.parquet")
    print(f"{df.market_id.nunique()} markets with a signal; {len(df)} rows")

def fetch():
    df = pd.read_parquet("out/yes2_signals.parquet")
    need = df.drop_duplicates("market_id")
    need = need[~need.condition_id.map(lambda c: os.path.exists(f"data/tapes/{c}.json"))]
    print(f"{len(need)} markets missing a tape", flush=True)
    t0 = dict(df.groupby("market_id").signal_t.min())
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(4) as ex:
        for i, _ in enumerate(ex.map(lambda r: fetch_tape(r.condition_id, int(t0[r.market_id]) - 3 * 86400), need.itertuples()), 1):
            if i % 500 == 0: print(i, flush=True)
    print("done")

def summarize():
    df = pd.read_parquet("out/yes2_signals.parquet")
    mk = {m["market_id"]: m for m in json.load(open("data/markets.json"))}
    tapes = {}; fills = []
    for r in df.itertuples():
        if r.market_id not in tapes:
            m = mk[r.market_id]; tok = str(m["tokens"][m["outcomes"].index("Yes")])
            try: tp = json.load(open(f"data/tapes/{m['condition_id']}.json"))
            except FileNotFoundError: tapes[r.market_id] = None; fills.append(None); continue
            buys = sorted((x["timestamp"], float(x["price"])) for x in tp if str(x["asset"]) == tok and x["side"] == "BUY")
            tapes[r.market_id] = (buys, [b[0] for b in buys])
        if tapes[r.market_id] is None: fills.append(None); continue
        buys, bts = tapes[r.market_id]; j = bisect.bisect_left(bts, r.signal_t)
        fills.append(buys[j][1] if (j < len(buys) and buys[j][0] <= r.signal_t + FILL_WINDOW) else None)
    df["fill"] = fills; df["slip_c"] = (df.fill - df.mid) * 100
    df.to_parquet("out/yes2_fills.parquet")
    f = df[df.fill.notna() & (df.slip_c >= LOWER - 1e-9) & (df.slip_c <= CAP + 1e-9)].copy()
    f["fee"] = STAKE / f.fill * FEE_RATE * f.fill * (1 - f.fill)
    f["pnl"] = (STAKE / f.fill - STAKE).where(f.won, -STAKE) - f.fee
    g = f.groupby(["lo", "hi"]).agg(trades=("pnl", "size"), lost=("won", lambda s: int((~s).sum())),
                                    avg_fill=("fill", "mean"), pnl=("pnl", "sum")).reset_index()
    g["roi"] = g.pnl / (g.trades * STAKE); g["loss_rate"] = g.lost / g.trades
    g.to_csv("out/sweep_band2_summary.csv", index=False)
    pd.set_option("display.width", 250)
    for col, name, sc in [("roi", "ROI per trade (%)", 100), ("pnl", "P&L ($)", 1), ("trades", "trades", 1),
                          ("lost", "losses", 1), ("loss_rate", "loss rate (%)", 100)]:
        print(f"\n{name}: rows = min entry, cols = max entry")
        print((g.pivot(index="lo", columns="hi", values=col) * sc).round(2 if sc == 100 else 0).to_string())

if __name__ == "__main__":
    {"signals": signals, "fetch": fetch, "summarize": summarize}[sys.argv[1]]()
