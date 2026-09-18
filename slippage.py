"""Replace midpoint entries with real taker fills.

For every YES position in out/sim_positions_max<cap>.csv, pull the market's
trade tape (data-api) and take the FIRST taker BUY of the YES token printed at
or after the entry signal (within FILL_WINDOW). That trade's price is what a
market order would actually have paid. Positions with no such print within the
window are marked unfilled. Writes out/sim_fills_max<cap>.csv.
"""
import json, os, sys, time, threading, datetime as dt
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, requests

CAP = sys.argv[1] if len(sys.argv) > 1 else "0.975"
TAG = f"max{CAP}" + (f"_min{sys.argv[2]}" if len(sys.argv) > 2 else "") + (f"_cut{float(sys.argv[3]):g}" if len(sys.argv) > 3 and float(sys.argv[3]) != 6 else "")
FILL_WINDOW = 60 * 60          # seconds after the signal to look for a fill
FEE_RATE = 0.05; STAKE = 100.0
RATE = 8.0
_local = threading.local(); _gate = threading.Lock(); _slot = [0.0]

def throttle():
    with _gate:
        now = time.monotonic(); s = max(_slot[0], now); _slot[0] = s + 1 / RATE
    time.sleep(max(0.0, s - now))

def get_json(url, params):
    if not hasattr(_local, "s"): _local.s = requests.Session()
    for attempt in range(10):
        throttle()
        try:
            r = _local.s.get(url, params=params, timeout=60)
            if r.status_code == 200: return r.json()
            time.sleep(min(30, (5 if r.status_code == 429 else 2) * (attempt + 1)))
        except Exception:
            time.sleep(min(30, 2 + attempt))
    return None

def tape(condition_id, oldest_needed):
    """Newest-first pages until we are past oldest_needed. Cached per market in data/tapes/."""
    path = f"data/tapes/{condition_id}.json"
    if os.path.exists(path):
        return json.load(open(path))
    rows = []; offset = 0
    while True:
        page = get_json("https://data-api.polymarket.com/trades",
                        {"market": condition_id, "limit": 1000, "offset": offset})
        if page is None: return None
        rows += page
        if len(page) < 1000 or page[-1]["timestamp"] < oldest_needed or offset >= 20000: break
        offset += 1000
    rows = [{k: r[k] for k in ("asset", "side", "price", "size", "timestamp")} for r in rows]
    json.dump(rows, open(path + ".tmp", "w")); os.replace(path + ".tmp", path)
    return rows

def fill_for(pos, m):
    yes_tok = str(m["tokens"][m["outcomes"].index("Yes")])
    t0 = int(pd.Timestamp(pos.entry_time).timestamp())
    rows = tape(m["condition_id"], t0 - 3 * 86400)   # cover any earlier signal under other caps
    if rows is None: return dict(fill_price=None, fill_delay_s=None, fill_status="tape_unavailable")
    buys = sorted((r for r in rows if str(r["asset"]) == yes_tok and r["side"] == "BUY"
                   and t0 <= r["timestamp"] <= t0 + FILL_WINDOW), key=lambda r: r["timestamp"])
    if not buys: return dict(fill_price=None, fill_delay_s=None, fill_status="no_print")
    f = buys[0]
    return dict(fill_price=float(f["price"]), fill_delay_s=f["timestamp"] - t0, fill_status="filled")

def main():
    os.makedirs("data/tapes", exist_ok=True)
    mk = {m["market_id"]: m for m in json.load(open("data/markets.json"))}
    df = pd.read_csv(f"out/sim_positions_{TAG}.csv")
    df = df[df.side == "Yes"].reset_index(drop=True)
    print(f"{len(df)} YES positions -> fetching tapes", flush=True)
    with ThreadPoolExecutor(4) as ex:
        fills = list(ex.map(lambda p: fill_for(p, mk[str(p.market_id)]), df.itertuples()))
    df = pd.concat([df, pd.DataFrame(fills)], axis=1)
    df["slip_c"] = (df.fill_price - df.entry_price) * 100
    f = df[df.fill_status == "filled"].copy()
    f["fee_real"] = STAKE / f.fill_price * FEE_RATE * f.fill_price * (1 - f.fill_price)
    f["pnl_real"] = (STAKE / f.fill_price - STAKE).where(f.won, -STAKE) - f.fee_real
    df = df.merge(f[["market_id", "fee_real", "pnl_real"]], on="market_id", how="left")
    df.to_csv(f"out/sim_fills_{TAG}.csv", index=False)

    print("\nfill status:", df.fill_status.value_counts().to_dict())
    print(f"slippage (fill - midpoint), cents: mean {f.slip_c.mean():+.2f}  median {f.slip_c.median():+.2f}  "
          f"p10 {f.slip_c.quantile(.1):+.2f}  p90 {f.slip_c.quantile(.9):+.2f}")
    print(f"fill delay: median {f.fill_delay_s.median()/60:.1f} min, p90 {f.fill_delay_s.quantile(.9)/60:.1f} min")
    n = len(f); w = int(f.won.sum())
    print(f"\n== market-order fills: {n} positions, {w} won / {n-w} lost (win rate {w/n:.2%})")
    print(f"   avg midpoint at signal {f.entry_price.mean():.4f}  avg real fill {f.fill_price.mean():.4f}")
    print(f"   P&L at midpoint (fees incl.) ${f.pnl.sum():,.0f} ({f.pnl.sum()/(n*STAKE):+.2%})")
    print(f"   P&L at real fill (fees incl.) ${f.pnl_real.sum():,.0f} ({f.pnl_real.sum()/(n*STAKE):+.2%})")
    lim = f[f.fill_price <= float(CAP)]
    n2 = len(lim); w2 = int(lim.won.sum())
    print(f"\n== limit-at-cap fills (only take if ask <= {CAP}): {n2} positions, {w2} won / {n2-w2} lost")
    print(f"   P&L ${lim.pnl_real.sum():,.0f} ({lim.pnl_real.sum()/(n2*STAKE):+.2%})   avg fill {lim.fill_price.mean():.4f}")

if __name__ == "__main__":
    main()
