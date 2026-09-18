"""Event study: when an hourly METAR print rules a bucket OUT (rounded temp
exceeds the bucket's upper bound for the first time that day), what does that
bucket's NO midpoint do in the minutes after?

  python3 event_study.py events            -> out/events.parquet
  python3 event_study.py fetch [N]         -> 1-min NO midpoints, -15..+90 min around N random events
  python3 event_study.py summarize
"""
import json, os, re, sys, random, time, threading, datetime as dt
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, numpy as np, requests
from sweep_local import TZ, CITY_RE, EXCL

OFFSETS = [-5, -1, 0, 1, 2, 3, 4, 5, 10, 15, 30, 60]
BUCKET_RE = re.compile(r"be (?:between (-?\d+)-(-?\d+)|(-?\d+)°[FC] or (higher|below)|(-?\d+)°[FC])")
def bucket(q):
    m = BUCKET_RE.search(q)
    if not m: return None
    if m.group(1): return (int(m.group(1)), int(m.group(2)))
    if m.group(3): return (int(m.group(3)), 999) if m.group(4) == "higher" else (-999, int(m.group(3)))
    return (int(m.group(5)), int(m.group(5)))

def events():
    met = pd.read_parquet("data/metar.parquet")
    met["day"] = met.local_time.dt.date
    ms = [m for m in json.load(open("data/markets.json")) if CITY_RE.match(m["question"] or "") and not EXCL.search(m["question"])]
    by_day = {}
    for m in ms:
        city = CITY_RE.match(m["question"]).group(1); day = dt.date.fromisoformat(m["end_date"][:10])
        b = bucket(m["question"])
        if b and b[1] != 999 and len(m["tokens"]) == 2: by_day.setdefault((city, day), []).append((m, b))
    rows = []
    for (city, day), mk in by_day.items():
        o = met[(met.city == city) & (met.day == day)].sort_values("local_time")
        if o.empty: continue
        unit = o.unit.iloc[0]; tz = ZoneInfo(TZ[city])
        vals = (o.temp_c * 9 / 5 + 32 if unit == "F" else o.temp_c).round().astype(int).tolist()
        times = o.local_time.tolist()
        running = -999
        for t, v in zip(times, vals):
            if v <= running: continue
            for m, b in mk:
                if running <= b[1] < v:      # this print is the first to exceed the bucket's upper bound
                    ts = int(t.replace(tzinfo=tz).timestamp())
                    won_no = [float(x) for x in m["outcome_prices"]] == [0.0, 1.0]
                    rows.append(dict(market_id=m["market_id"], city=city, day=str(day), question=m["question"],
                                     lo=b[0], hi=b[1], print_local=t, print_ts=ts, temp=v, prev_max=running,
                                     margin=v - b[1], no_token=m["tokens"][1], no_won=won_no, end_date=m["end_date"]))
            running = v
    df = pd.DataFrame(rows); df.to_parquet("out/events.parquet")
    print(f"{len(df)} rule-out events across {df.groupby(['city','day']).ngroups} city-days; NO won in {df.no_won.mean():.2%}")

_local = threading.local(); _gate = threading.Lock(); _slot = [0.0]
def throttle(rate=20.0):
    with _gate:
        now = time.monotonic(); s = max(_slot[0], now); _slot[0] = s + 1 / rate
    time.sleep(max(0.0, s - now))
def fetch_one(r):
    out = f"data/event_prices/{r.market_id}_{r.print_ts}.json"
    if os.path.exists(out): return
    if not hasattr(_local, "s"): _local.s = requests.Session()
    for attempt in range(20):
        throttle()
        try:
            resp = _local.s.get("https://clob.polymarket.com/prices-history", params={"market": r.no_token, "startTs": r.print_ts - 900, "endTs": r.print_ts + 5400, "fidelity": 1}, timeout=60)
            if resp.status_code == 200:
                json.dump(resp.json().get("history", []), open(out, "w")); return
            time.sleep(min(30, 3 * (attempt + 1)))
        except requests.RequestException: time.sleep(3)

def fetch(n):
    os.makedirs("data/event_prices", exist_ok=True)
    df = pd.read_parquet("out/events.parquet"); random.seed(3)
    idx = list(df.index); random.shuffle(idx); sample = df.loc[idx[:n]]
    sample.to_parquet("out/events_sample.parquet")
    with ThreadPoolExecutor(12) as ex:
        for i, _ in enumerate(ex.map(fetch_one, sample.itertuples()), 1):
            if i % 1000 == 0: print(i, flush=True)
    print("done")

def summarize():
    df = pd.read_parquet("out/events_sample.parquet"); rows = []
    for r in df.itertuples():
        p = f"data/event_prices/{r.market_id}_{r.print_ts}.json"
        if not os.path.exists(p): continue
        h = json.load(open(p))
        if not h: continue
        ts = np.array([x["t"] for x in h]); ps = np.array([x["p"] for x in h])
        rec = dict(market_id=r.market_id, city=r.city, margin=r.margin, no_won=r.no_won, prev_max=r.prev_max, hi=r.hi, print_local=r.print_local)
        for k in OFFSETS:
            t = r.print_ts + 60 * k; i = np.searchsorted(ts, t, side="right") - 1   # last sample at or before t
            rec[f"m{k}"] = ps[i] if i >= 0 and ts[i] >= t - 180 else np.nan        # require a sample within 3 min
        rows.append(rec)
    e = pd.DataFrame(rows); e.to_csv("out/event_study.csv", index=False)
    cols = [f"m{k}" for k in OFFSETS]
    print(f"{len(e)} events with price data (NO won {e.no_won.mean():.2%})\n")
    print("NO midpoint, mean / median by minutes after the print:")
    print(pd.DataFrame({"mean": e[cols].mean(), "median": e[cols].median(), "p10": e[cols].quantile(.1), "p25": e[cols].quantile(.25)}).T.round(3).to_string())
    print("\nshare of events where NO is still <= X at each offset:")
    print(pd.DataFrame({f"<= {x:.2f}": (e[cols] <= x).mean() for x in [0.90, 0.95, 0.97, 0.98, 0.99]}).T.round(3).to_string())
    print("\nby margin (how far the print exceeded the bucket), NO median at +1/+5/+10 and share <=0.97 at +5:")
    g = e.groupby(e.margin.clip(upper=4)).agg(n=("m5", "size"), no_m1=("m1", "median"), no_m5=("m5", "median"), no_m10=("m10", "median"), le97_m5=("m5", lambda s: (s <= .97).mean()), no_won=("no_won", "mean"))
    print(g.round(3).to_string())
    print("\nby pre-print price (NO at -1 min): where the market already knew vs. didn't")
    e["pre"] = pd.cut(e["m-1"], [0, .5, .8, .9, .95, .99, 1.01])
    print(e.groupby("pre", observed=True).agg(n=("m5", "size"), no_m1=("m1", "median"), no_m5=("m5", "median"), no_m10=("m10", "median"), no_m30=("m30", "median"), no_won=("no_won", "mean")).round(3).to_string())

if __name__ == "__main__":
    c = sys.argv[1]
    if c == "events": events()
    elif c == "fetch": fetch(int(sys.argv[2]) if len(sys.argv) > 2 else 8000)
    else: summarize()
