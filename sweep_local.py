"""Per-city LOCAL-time cutoff sweep x slippage-cap sweep, single pass.

Universe: "Highest temperature in <city> ..." markets (excluding "or higher"
buckets). Rule: buy $100 YES the first time YES has been >=96c for 30
contiguous minutes, midpoint >= FLOOR, and local time in the city is at or
after HH:00 on the market's day. Fill = first taker BUY of YES on the tape
within 60 min of the signal; the fill must lie within [mid-0.5c, mid+X] where X
is the slippage cap (a book monitor). Fees 0.05*p*(1-p) on the fill.
"""
import json, glob, re, sys, datetime as dt, bisect
from zoneinfo import ZoneInfo
import pandas as pd

FLOOR = 0.99; THRESH = 0.96; RUN_MIN = 30; STEP = 5; NEED = RUN_MIN // STEP
STAKE = 100.0; FEE_RATE = 0.05; FILL_WINDOW = 3600
HOURS = [0, 6, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22]        # local-time cutoffs
CAPS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5]                                  # ask <= mid + cap (cents)
LOWER = -0.5                                                             # never accept a print > 0.5c below mid
TZ = {
 'New York City':'America/New_York','London':'Europe/London','Miami':'America/New_York','Paris':'Europe/Paris',
 'Tokyo':'Asia/Tokyo','Shanghai':'Asia/Shanghai','Hong Kong':'Asia/Hong_Kong','Seoul':'Asia/Seoul',
 'Seoul (Incheon)':'Asia/Seoul','Toronto':'America/Toronto','Dallas':'America/Chicago','Atlanta':'America/New_York',
 'Buenos Aires':'America/Argentina/Buenos_Aires','Seattle':'America/Los_Angeles','Ankara':'Europe/Istanbul',
 'Wellington':'Pacific/Auckland','Chicago':'America/Chicago','Sao Paulo':'America/Sao_Paulo','Lucknow':'Asia/Kolkata',
 'Munich':'Europe/Berlin','Tel Aviv':'Asia/Jerusalem','Singapore':'Asia/Singapore','Milan':'Europe/Rome',
 'Madrid':'Europe/Madrid','Warsaw':'Europe/Warsaw','Taipei':'Asia/Taipei','Chongqing':'Asia/Shanghai',
 'Beijing':'Asia/Shanghai','Wuhan':'Asia/Shanghai','Shenzhen':'Asia/Shanghai','Chengdu':'Asia/Shanghai',
 'Houston':'America/Chicago','Austin':'America/Chicago','Denver':'America/Denver','Los Angeles':'America/Los_Angeles',
 'San Francisco':'America/Los_Angeles','Istanbul':'Europe/Istanbul','Moscow':'Europe/Moscow',
 'Mexico City':'America/Mexico_City','Busan':'Asia/Seoul','Amsterdam':'Europe/Amsterdam','Helsinki':'Europe/Helsinki',
 'Kuala Lumpur':'Asia/Kuala_Lumpur','Panama City':'America/Panama','Jeddah':'Asia/Riyadh','Cape Town':'Africa/Johannesburg',
 'Karachi':'Asia/Karachi','Guangzhou':'Asia/Shanghai','Manila':'Asia/Manila','Qingdao':'Asia/Shanghai',
 'Jinan':'Asia/Shanghai','Zhengzhou':'Asia/Shanghai','Jakarta':'Asia/Jakarta','Lagos':'Africa/Lagos'}
CITY_RE = re.compile(r'^Will the highest temperature in (.+?) be ')
EXCL = re.compile(r'or higher|or above', re.I)

def entries(series, cutoffs):
    """For each cutoff ts (sorted), first (t,p) with a qualifying run, p>=FLOOR, t>=cutoff."""
    out = {}; run = 0; prev = None
    for t, p in series:
        ok = p >= THRESH
        run = run + 1 if (ok and prev is not None and t - prev <= 2 * STEP * 60 + 30) else (1 if ok else 0)
        prev = t if ok else None
        if run >= NEED and p >= FLOOR:
            for h, c in cutoffs.items():
                if h not in out and t >= c:
                    out[h] = (t, p)
            if len(out) == len(cutoffs): break
    return out

def main():
    markets = [m for m in json.load(open("data/markets.json"))
               if CITY_RE.match(m["question"] or "") and not EXCL.search(m["question"])]
    print(f"{len(markets)} highest-temperature markets", flush=True)
    rows = []; n_no_tape = 0
    for i, m in enumerate(markets):
        city = CITY_RE.match(m["question"]).group(1)
        try: d = json.load(open(f"data/prices/{m['market_id']}.json"))
        except FileNotFoundError: continue
        if not d["history"]: continue
        losing = d["losing_outcome"]; won = losing == "No"
        h = sorted((x["t"], x["p"]) for x in d["history"])
        yes = h if losing == "Yes" else [(t, 1 - p) for t, p in h]
        day = dt.datetime.fromisoformat(m["end_date"].replace("Z", "+00:00")).date()
        tz = ZoneInfo(TZ[city])
        cutoffs = {hh: dt.datetime(day.year, day.month, day.day, hh, tzinfo=tz).timestamp() for hh in HOURS}
        ent = entries(yes, cutoffs)
        if not ent: continue
        try: tape = json.load(open(f"data/tapes/{m['condition_id']}.json"))
        except FileNotFoundError: n_no_tape += 1; continue
        yes_tok = str(m["tokens"][m["outcomes"].index("Yes")])
        buys = sorted((r["timestamp"], float(r["price"])) for r in tape if str(r["asset"]) == yes_tok and r["side"] == "BUY")
        bts = [b[0] for b in buys]
        for hh, (t, p) in ent.items():
            j = bisect.bisect_left(bts, t)
            fill = None
            if j < len(buys) and buys[j][0] <= t + FILL_WINDOW: fill = buys[j][1]
            rows.append(dict(market_id=m["market_id"], city=city, day=day, cutoff_h=hh, signal_t=t, mid=p,
                             fill=fill, slip_c=None if fill is None else (fill - p) * 100, won=won))
        if i % 20000 == 0: print(i, flush=True)
    df = pd.DataFrame(rows)
    print(f"tape missing for {n_no_tape} markets with a signal")
    df.to_parquet("out/sweep_local_raw.parquet")
    summarize(df)

def summarize(df):
    f = df[df.fill.notna()].copy()
    f["fee"] = STAKE / f.fill * FEE_RATE * f.fill * (1 - f.fill)
    f["pnl"] = (STAKE / f.fill - STAKE).where(f.won, -STAKE) - f.fee
    out = []
    for hh in HOURS:
        for cap in CAPS:
            s = f[(f.cutoff_h == hh) & (f.slip_c >= LOWER - 1e-9) & (f.slip_c <= cap + 1e-9)]
            n = len(s)
            if n == 0: continue
            out.append(dict(local_cutoff=f"{hh:02d}:00", slip_cap=cap, trades=n, lost=int((~s.won).sum()),
                            loss_rate=(~s.won).mean(), avg_fill=s.fill.mean(), pnl=s.pnl.sum(), roi=s.pnl.sum() / (n * STAKE)))
    res = pd.DataFrame(out); res.to_csv("out/sweep_local_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print("\nP&L ($) by local cutoff (rows) x slippage cap (cols):")
    print(res.pivot(index="local_cutoff", columns="slip_cap", values="pnl").round(0).to_string())
    print("\nROI per trade (%):")
    print((res.pivot(index="local_cutoff", columns="slip_cap", values="roi") * 100).round(2).to_string())
    print("\ntrades:")
    print(res.pivot(index="local_cutoff", columns="slip_cap", values="trades").to_string())
    print("\nloss rate (%):")
    print((res.pivot(index="local_cutoff", columns="slip_cap", values="loss_rate") * 100).round(2).to_string())

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summarize":
        summarize(pd.read_parquet("out/sweep_local_raw.parquet"))
    else:
        main()
