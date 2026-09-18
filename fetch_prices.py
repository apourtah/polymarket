"""Pull CLOB price history for the LOSING token of every cleanly resolved binary
weather market in data/markets.json.

Only the losing side matters for "did an outcome trade >=96c and still lose",
and in a binary market the two tokens are complements, so one token per market
is enough.

Pass 1 (default): 5-minute midpoint samples for the whole universe, stored as
the raw API response in data/prices/<market_id>.json. Re-runnable; cached files
are skipped.

Pass 2 (--fine <ids file>): 1-minute samples for a subset (the candidates), into
data/prices_1m/.
"""
import json, os, sys, time, datetime as dt, threading
from concurrent.futures import ThreadPoolExecutor
import requests

CLOB = "https://clob.polymarket.com/prices-history"
WORKERS = 12
RATE = 25.0               # global requests/sec; bursts above ~25/s get 429-throttled hard
MAX_WINDOW = 14 * 86400   # API rejects longer windows at fine fidelity
_local = threading.local()
_gate = threading.Lock()
_next_slot = [0.0]

def throttle():
    """Simple global token bucket: at most RATE requests/sec across all threads."""
    with _gate:
        now = time.monotonic()
        slot = max(_next_slot[0], now)
        _next_slot[0] = slot + 1.0 / RATE
    time.sleep(max(0.0, slot - now))

def session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s

def parse_ts(s):
    if not s:
        return None
    s = s.replace(" ", "T").replace("+00", "+00:00")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return dt.datetime.fromisoformat(s).timestamp()

def losing_token(m):
    """(token_id, outcome_name) for the losing outcome, or None if the market is
    not cleanly resolved to a single 0/1 winner."""
    prices = [float(p) for p in m["outcome_prices"]]
    if len(prices) != 2 or len(m["tokens"]) != 2 or sorted(prices) != [0.0, 1.0]:
        return None
    i = prices.index(0.0)
    return m["tokens"][i], m["outcomes"][i]

def fetch_one(m, out_dir, fidelity):
    out = f"{out_dir}/{m['market_id']}.json"
    if os.path.exists(out):
        return "cached"
    lt = losing_token(m)
    if lt is None:
        return "skip"
    token, name = lt
    start = parse_ts(m.get("created_at")) or parse_ts(m.get("start_date"))
    end = parse_ts(m.get("closed_time")) or parse_ts(m.get("end_date"))
    if start is None or end is None:
        return "skip"
    start = int(start) - 3600; end = int(end) + 3 * 86400
    chunks = []
    t0 = int(start)
    while t0 < end:
        t1 = min(int(end), t0 + MAX_WINDOW)
        for attempt in range(1000):   # never give up on a market: 429s are transient
            throttle()
            try:
                r = session().get(CLOB, params={"market": token, "startTs": t0, "endTs": t1,
                                                "fidelity": fidelity}, timeout=60)
                if r.status_code == 200:
                    chunks.append(r.text); break
                if r.status_code == 400:
                    return "bad_request"
                time.sleep(min(30, (5 if r.status_code == 429 else 1) * (attempt + 1)))
            except requests.RequestException:
                time.sleep(min(30, 2 + attempt))
        else:
            return "error"
        t0 = t1
    # Splice the raw {"history":[...]} chunks without parsing them
    inner = [c[c.index("[") + 1:c.rindex("]")] for c in chunks if "[" in c]
    hist = ",".join(x for x in inner if x)
    tmp = out + ".tmp"   # atomic write so a killed run never leaves a truncated file
    with open(tmp, "w") as f:
        f.write('{"market_id":"%s","token":"%s","losing_outcome":%s,"history":[%s]}'
                % (m["market_id"], token, json.dumps(name), hist))
    os.replace(tmp, out)
    return "ok"

def slim(m):
    return {k: m[k] for k in ("market_id", "tokens", "outcomes", "outcome_prices",
                              "created_at", "start_date", "closed_time", "end_date")}

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--fine":
        ids = set(json.load(open(sys.argv[2])))
        out_dir, fidelity = "data/prices_1m", 1
    else:
        ids, out_dir, fidelity = None, "data/prices", 5
    os.makedirs(out_dir, exist_ok=True)
    markets = [slim(m) for m in json.load(open("data/markets.json"))
               if ids is None or m["market_id"] in ids]
    markets = [m for m in markets if not os.path.exists(f"{out_dir}/{m['market_id']}.json")]
    print(f"{len(markets)} markets to fetch at fidelity={fidelity} -> {out_dir}", flush=True)
    counts = {}
    with ThreadPoolExecutor(WORKERS) as ex:
        for i, res in enumerate(ex.map(lambda m: fetch_one(m, out_dir, fidelity), markets), 1):
            counts[res] = counts.get(res, 0) + 1
            if i % 1000 == 0:
                print(i, counts, flush=True)
    print("done", counts)
