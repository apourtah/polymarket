"""Flag resolved weather markets where the LOSING outcome was priced at >=96c.

Pipeline
  1. data/prices/*.json      5-min midpoint series of the losing token, whole universe
     -> candidates = markets whose losing side ever printed >= THRESH
  2. data/prices_1m/*.json   1-min series, candidates only (fetch_prices.py --fine)
  3. data-api trade tape     $ actually traded at >= THRESH, candidates only

Per candidate we compute (from the 1-min series when available, else 5-min):
  max_price          highest midpoint of the losing outcome
  minutes_ge96       total minutes with midpoint >= THRESH
  longest_run_ge96   longest contiguous run (minutes) with midpoint >= THRESH
  first/last_ge96    when the losing side first/last sat at >= THRESH
  hrs_before_end     hours from last >=96c sample to the market's end date
  vol_ge96_usd       $ notional of trades in the losing outcome at >= THRESH
  n_trades_ge96      number of such trades

Flag rules (all require max_price >= THRESH):
  flag        longest_run_ge96 >= MIN_RUN_MIN  OR  vol_ge96_usd >= MIN_VOL_USD
              i.e. the market *sat* there, or real money changed hands there;
              a single thin print that reverted in a minute is excluded.
  flag_strong flag AND last >=96c sample within CUTOFF_HRS of the end date
              (was still "certain" close to resolution, not an early mis-price
              that corrected long before the outcome was known).
"""
import json, os, glob, sys, time, datetime as dt, threading
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import requests

THRESH = 0.96
MIN_RUN_MIN = 30        # contiguous minutes at >=96c
MIN_VOL_USD = 500       # $ traded at >=96c in the losing outcome
CUTOFF_HRS = 6          # last >=96c sample within N hours of end_date -> "late"
U = dt.timezone.utc
_local = threading.local()
_gate = threading.Lock(); _next_slot = [0.0]
TAPE_RATE = 8.0   # data-api requests/sec across all threads

def throttle():
    with _gate:
        now = time.monotonic(); slot = max(_next_slot[0], now)
        _next_slot[0] = slot + 1.0 / TAPE_RATE
    time.sleep(max(0.0, slot - now))

def get_json(url, params, tries=10):
    """GET with global throttle + retry; returns None if it never succeeds."""
    if not hasattr(_local, "s"): _local.s = requests.Session()
    for attempt in range(tries):
        throttle()
        try:
            r = _local.s.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            time.sleep(min(30, (5 if r.status_code == 429 else 2) * (attempt + 1)))
        except Exception:
            time.sleep(min(30, 2 + attempt))
    return None

def parse_ts(s):
    if not s: return None
    s = s.replace(" ", "T").replace("+00", "+00:00")
    if s.endswith("Z"): s = s[:-1] + "+00:00"
    return dt.datetime.fromisoformat(s)

def series_stats(hist, step_min):
    """hist: list of {t,p}; step_min: nominal sample spacing in minutes."""
    if not hist:
        return dict(n_samples=0, max_price=None, minutes_ge96=0, longest_run_ge96=0,
                    first_ge96=None, last_ge96=None)
    hist.sort(key=lambda x: x["t"])
    mx = max(h["p"] for h in hist)
    tot = run = best = 0
    first = last = prev_t = None
    gap = step_min * 60 * 2 + 30   # allow one missing sample inside a run
    for h in hist:
        if h["p"] >= THRESH:
            tot += 1
            run = run + 1 if (prev_t is not None and h["t"] - prev_t <= gap) else 1
            best = max(best, run)
            first = first or h["t"]; last = h["t"]; prev_t = h["t"]
        else:
            run = 0; prev_t = None
    return dict(n_samples=len(hist), max_price=mx, minutes_ge96=tot * step_min,
                longest_run_ge96=best * step_min, first_ge96=first, last_ge96=last)

def load_hist(path):
    return json.load(open(path))["history"]

def trade_volume_ge96(condition_id, losing_token):
    """$ notional traded in the losing outcome at >= THRESH, from the data-api tape.
    A fill is counted if recorded on the losing token at price >= THRESH, or on the
    winning token at price <= 1-THRESH (the same economic event seen from the other side)."""
    total = 0.0; n = 0; offset = 0
    while True:
        rows = get_json("https://data-api.polymarket.com/trades",
                        {"market": condition_id, "limit": 1000, "offset": offset})
        if rows is None:
            return None, None   # tape unavailable -> unknown, not zero
        if not rows: break
        for t in rows:
            p = float(t.get("price", 0)); sz = float(t.get("size", 0))
            if str(t.get("asset")) == str(losing_token) and p >= THRESH:
                total += p * sz; n += 1
            elif str(t.get("asset")) != str(losing_token) and p <= 1 - THRESH:
                total += (1 - p) * sz; n += 1
        if len(rows) < 1000 or offset >= 20000: break
        offset += 1000
    return total, n

def main():
    os.makedirs("out", exist_ok=True)
    markets = {m["market_id"]: m for m in json.load(open("data/markets.json"))}

    # ---- stage 1: coarse sweep -------------------------------------------------
    rows = []
    for f in glob.glob("data/prices/*.json"):
        d = json.load(open(f))
        m = markets.get(d["market_id"])
        if not m: continue
        s = series_stats(d["history"], 5)
        rows.append(dict(market_id=m["market_id"], losing_outcome=d["losing_outcome"],
                         token=d["token"], **s))
    df = pd.DataFrame(rows)
    df.to_csv("out/all_markets_stats_5m.csv", index=False)
    print(f"{len(df)} markets swept; {int(df.n_samples.eq(0).sum())} had empty history")
    cand_ids = df[df.max_price >= THRESH].market_id.tolist()
    json.dump(cand_ids, open("out/candidate_ids.json", "w"))
    print(f"{len(cand_ids)} candidates with losing-side midpoint >= {THRESH} (5-min samples)")

    # ---- stage 2: 1-min detail for candidates ------------------------------------
    missing = [i for i in cand_ids if not os.path.exists(f"data/prices_1m/{i}.json")]
    if missing:
        print(f"!! {len(missing)} candidates lack 1-min history; run:\n"
              f"   python3 fetch_prices.py --fine out/candidate_ids.json\n"
              f"   (falling back to 5-min stats for those)")
    rows = []
    for mid in cand_ids:
        m = markets[mid]
        fine = f"data/prices_1m/{mid}.json"
        d = json.load(open(fine if os.path.exists(fine) else f"data/prices/{mid}.json"))
        s = series_stats(d["history"], 1 if os.path.exists(fine) else 5)
        end = parse_ts(m["end_date"]); closed = parse_ts(m["closed_time"])
        last = dt.datetime.fromtimestamp(s["last_ge96"], U) if s["last_ge96"] else None
        first = dt.datetime.fromtimestamp(s["first_ge96"], U) if s["first_ge96"] else None
        rows.append(dict(
            market_id=mid, condition_id=m["condition_id"], event_slug=m["event_slug"],
            question=m["question"], losing_outcome=d["losing_outcome"], token=d["token"],
            end_date=end, closed_time=closed, market_volume_usd=m["volume"],
            resolution=("1m" if os.path.exists(fine) else "5m"),
            n_samples=s["n_samples"], max_price=s["max_price"],
            minutes_ge96=s["minutes_ge96"], longest_run_ge96=s["longest_run_ge96"],
            first_ge96=first, last_ge96=last,
            hrs_before_end=(end - last).total_seconds()/3600 if (last and end) else None,
            hrs_before_close=(closed - last).total_seconds()/3600 if (last and closed) else None,
            url="https://polymarket.com/event/" + m["event_slug"],
        ))
    cand = pd.DataFrame(rows)

    # ---- stage 3: trade tape ---------------------------------------------------
    print("fetching trade tape for candidates...")
    with ThreadPoolExecutor(4) as ex:
        vols = list(ex.map(lambda r: trade_volume_ge96(r.condition_id, r.token), cand.itertuples()))
    cand["vol_ge96_usd"] = [round(v[0], 2) if v[0] is not None else float("nan") for v in vols]
    cand["n_trades_ge96"] = [v[1] for v in vols]
    print(f"  trade tape unavailable for {int(cand.vol_ge96_usd.isna().sum())} candidates")

    cand["pass_run"] = cand.longest_run_ge96 >= MIN_RUN_MIN
    cand["pass_vol"] = cand.vol_ge96_usd.fillna(0) >= MIN_VOL_USD
    cand["late"] = cand.hrs_before_end <= CUTOFF_HRS
    cand["flag"] = cand.pass_run | cand.pass_vol
    cand["flag_strong"] = cand.flag & cand.late
    cand = cand.sort_values(["flag_strong", "flag", "vol_ge96_usd", "longest_run_ge96"],
                            ascending=False)
    cand.to_csv("out/candidates_ge96.csv", index=False)
    cand[cand.flag].to_csv("out/flagged.csv", index=False)

    print(f"\nflagged (run>={MIN_RUN_MIN}m at >=96c OR >=${MIN_VOL_USD} traded at >=96c): {int(cand.flag.sum())}")
    print(f"  of which still >=96c within {CUTOFF_HRS}h of end date (flag_strong): {int(cand.flag_strong.sum())}")
    print(f"  rejected as single-print / thin: {int((~cand.flag).sum())}")
    cols = ["question","losing_outcome","max_price","longest_run_ge96","minutes_ge96",
            "vol_ge96_usd","n_trades_ge96","hrs_before_end","market_volume_usd","flag_strong"]
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.max_rows", 500)
    print(cand[cand.flag][cols].to_string())

if __name__ == "__main__":
    main()
