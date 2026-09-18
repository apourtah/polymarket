"""Backtest of the confirmed signal: a bucket is 'disqualified' at the EARLIER of
   (a) a 5-minute reading rounded to >= bucket_hi + 2 °F, or
   (b) an hourly/special METAR reading rounded to > bucket_hi.
Then: per-minute NO midpoint after the signal timestamp, and time to reach 0.99.
  python3 signal_study.py events | fetch N | summarize
"""
import json, os, sys, random, bisect, time, threading, datetime as dt
from decimal import Decimal, ROUND_HALF_UP
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, numpy as np, requests
from sweep_local import TZ, CITY_RE, EXCL
from event_study import bucket, OFFSETS as _O
OFFSETS = [-5, -1, 0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 60]
MARGIN5 = int(os.environ.get("MARGIN5", "2"))
rh = lambda x: int(Decimal(str(x)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))

def events():
    m5 = pd.read_parquet("data/metar5.parquet"); m1 = pd.read_parquet("data/metar.parquet")
    stations = set(m5.station.unique()); m1 = m1[m1.station.isin(stations)]
    for d in (m5, m1): d["day"] = d.local_time.dt.date; d["tf"] = (d.temp_c * 9 / 5 + 32).map(rh)
    g5 = {k: v.sort_values("local_time") for k, v in m5.groupby(["station", "day"])}
    g1 = {k: v.sort_values("local_time") for k, v in m1.groupby(["station", "day"])}
    site = json.load(open("out/stations.json")); city2st = {c: s for c, (s, u) in site.items() if u == "F"}
    ms = [m for m in json.load(open("data/markets.json")) if CITY_RE.match(m["question"] or "") and not EXCL.search(m["question"]) and len(m["tokens"]) == 2]
    rows = []
    for m in ms:
        city = CITY_RE.match(m["question"]).group(1); st = city2st.get(city)
        if not st: continue
        b = bucket(m["question"]);
        if not b or b[1] == 999: continue
        day = dt.date.fromisoformat(m["end_date"][:10]); tz = ZoneInfo(TZ[city])
        o5 = g5.get((st, day)); o1 = g1.get((st, day))
        if o1 is None or o5 is None or len(o5) < 200: continue
        t5 = o5[o5.tf >= b[1] + MARGIN5].local_time.min(); th = o1[o1.tf > b[1]].local_time.min()
        cands = [(t5, "5min"), (th, "hourly")]; cands = [(t, k) for t, k in cands if pd.notna(t)]
        if not cands: continue
        t, kind = min(cands)
        hourly_max = int(o1.tf.max())
        rows.append(dict(market_id=m["market_id"], city=city, station=st, day=str(day), question=m["question"], hi=b[1],
                         signal_local=t, signal_ts=int(t.replace(tzinfo=tz).timestamp()), trigger=kind,
                         t5=t5, th=th, lead_min=((th - t5).total_seconds() / 60 if pd.notna(t5) and pd.notna(th) else None),
                         no_token=m["tokens"][1], no_won=[float(x) for x in m["outcome_prices"]] == [0.0, 1.0],
                         hourly_confirms=hourly_max > b[1]))
    df = pd.DataFrame(rows); df.to_parquet(f"out/signal_events_m{MARGIN5}.parquet")
    print(f"{len(df)} disqualification signals | trigger: {df.trigger.value_counts().to_dict()} | NO won {df.no_won.mean():.2%} | hourly later confirms the 5-min trigger: {df[df.trigger=='5min'].hourly_confirms.mean():.2%}")
    print("lead of 5-min signal over hourly print (min):", df[df.trigger=='5min'].lead_min.describe()[['25%','50%','75%']].round(0).to_dict())

_local = threading.local(); _gate = threading.Lock(); _slot = [0.0]
def throttle(rate=22.0):
    with _gate:
        now = time.monotonic(); s = max(_slot[0], now); _slot[0] = s + 1 / rate
    time.sleep(max(0.0, s - now))
def fetch_one(r):
    out = f"data/signal_prices/{r.market_id}_{r.signal_ts}.json"
    if os.path.exists(out): return
    if not hasattr(_local, "s"): _local.s = requests.Session()
    for attempt in range(20):
        throttle()
        try:
            resp = _local.s.get("https://clob.polymarket.com/prices-history", params={"market": r.no_token, "startTs": r.signal_ts - 900, "endTs": r.signal_ts + 5400, "fidelity": 1}, timeout=60)
            if resp.status_code == 200: json.dump(resp.json().get("history", []), open(out, "w")); return
            time.sleep(min(30, 3 * (attempt + 1)))
        except requests.RequestException: time.sleep(3)
def fetch(n):
    os.makedirs("data/signal_prices", exist_ok=True)
    df = pd.read_parquet(f"out/signal_events_m{MARGIN5}.parquet"); random.seed(5); idx = list(df.index); random.shuffle(idx)
    s = df.loc[idx[:n]]; s.to_parquet(f"out/signal_sample_m{MARGIN5}.parquet")
    with ThreadPoolExecutor(12) as ex:
        for i, _ in enumerate(ex.map(fetch_one, s.itertuples()), 1):
            if i % 1000 == 0: print(i, flush=True)
    print("done")

def summarize():
    df = pd.read_parquet(f"out/signal_sample_m{MARGIN5}.parquet"); rows = []
    for r in df.itertuples():
        p = f"data/signal_prices/{r.market_id}_{r.signal_ts}.json"
        if not os.path.exists(p): continue
        h = json.load(open(p))
        if not h: continue
        ts = np.array([x["t"] for x in h]); ps = np.array([x["p"] for x in h])
        rec = dict(market_id=r.market_id, trigger=r.trigger, no_won=r.no_won, hourly_confirms=r.hourly_confirms, hi=r.hi)
        for k in OFFSETS:
            t = r.signal_ts + 60 * k; i = np.searchsorted(ts, t, side="right") - 1
            rec[f"m{k}"] = ps[i] if i >= 0 and ts[i] >= t - 180 else np.nan
        after = ps[ts >= r.signal_ts]; ta = ts[ts >= r.signal_ts]
        hit = np.where(after >= 0.99)[0]; rec["min_to_99"] = (ta[hit[0]] - r.signal_ts) / 60 if len(hit) else np.nan
        hit97 = np.where(after >= 0.97)[0]; rec["min_to_97"] = (ta[hit97[0]] - r.signal_ts) / 60 if len(hit97) else np.nan
        rows.append(rec)
    e = pd.DataFrame(rows); e.to_csv(f"out/signal_study_m{MARGIN5}.csv", index=False)
    cols = [f"m{k}" for k in OFFSETS]; pd.set_option("display.width", 250)
    print(f"{len(e)} events with price data | trigger split {e.trigger.value_counts().to_dict()} | NO won {e.no_won.mean():.2%}")
    for name, s in [("ALL", e), ("trigger = 5-min +2F", e[e.trigger == "5min"]), ("trigger = hourly print", e[e.trigger == "hourly"])]:
        print(f"\n=== {name}: {len(s)} events")
        print("NO midpoint by minutes after signal:")
        print(pd.DataFrame({"mean": s[cols].mean(), "median": s[cols].median(), "p25": s[cols].quantile(.25), "p10": s[cols].quantile(.1)}).T.round(3).to_string())
        print("share of events with NO still <= X:")
        print(pd.DataFrame({f"<={x:.2f}": (s[cols] <= x).mean() for x in [0.90, 0.95, 0.97, 0.99]}).T.round(3).to_string())
        pre = s[s["m-1"] <= 0.95]
        print(f"events with NO <= 0.95 one minute before the signal: {len(pre)} ({len(pre)/len(s):.1%}); of those, minutes to reach 0.99: "
              f"median {pre.min_to_99.median():.1f}, p75 {pre.min_to_99.quantile(.75):.1f}, never within 90 min {pre.min_to_99.isna().mean():.1%}; "
              f"to 0.97: median {pre.min_to_97.median():.1f}")
        print("  NO path for those (median):", pre[cols].median().round(3).to_dict())

if __name__ == "__main__":
    c = sys.argv[1]
    if c == "events": events()
    elif c == "fetch": fetch(int(sys.argv[2]) if len(sys.argv) > 2 else 6000)
    else: summarize()
