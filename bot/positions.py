#!/usr/bin/env python3
"""Live status of the evening bot's positions.

  python3 bot/positions.py            one-shot table
  python3 bot/positions.py --watch 60 refresh every 60 s
  python3 bot/positions.py --all      include resolved positions from the trade log
  python3 bot/positions.py --paper    include paper (dry-run) fills; default shows live fills only
  POLYMARKET_PROXY=0x... python3 bot/positions.py --wallet   also show the exchange's view of the wallet's positions

Columns: city/day, bucket, shares, avg cost, current YES bid (what you could sell at) and mid, value, unrealized
P&L, the station's observed high so far today (METAR), and the model context recorded when the trade was placed.
"""
import argparse, csv, json, os, sys, time, datetime as dt
from zoneinfo import ZoneInfo
from decimal import Decimal, ROUND_HALF_UP
import requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bot import evening_config as C
from sweep_local import TZ
rh = lambda x: int(Decimal(str(x)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
SITE = json.load(open("out/stations.json"))
G = "\033[32m"; R = "\033[31m"; Y = "\033[33m"; D = "\033[2m"; B = "\033[1m"; N = "\033[0m"

def get(url, params=None):
    try:
        r = requests.get(url, params=params, timeout=15); return r.json() if r.status_code == 200 else None
    except Exception: return None

def obs_max(city, day):
    tz = ZoneInfo(TZ[city]); o = get(C.METAR, {"ids": SITE[city][0], "format": "json", "hours": 36}) or []
    v = [(dt.datetime.fromtimestamp(x["obsTime"], tz), rh(x["temp"] * 9 / 5 + 32)) for x in o if x.get("temp") is not None and x.get("obsTime") and dt.datetime.fromtimestamp(x["obsTime"], tz).date() == day]
    if not v: return None, None
    return max(t for _, t in v), max(v)[0].strftime("%H:%M")

def market_info(cid):
    m = get(f"{C.GAMMA}/markets", {"condition_ids": cid})
    if m and isinstance(m, list) and m: return m[0]
    return None

def book(token):
    b = get(f"{C.CLOB}/book", {"token_id": token})
    if not b: return None, None
    bids = sorted((float(x["price"]) for x in b.get("bids", [])), reverse=True); asks = sorted(float(x["price"]) for x in b.get("asks", []))
    bid = bids[0] if bids else None; ask = asks[0] if asks else None
    mid = (bid + ask) / 2 if bid is not None and ask is not None else (bid if bid is not None else ask)
    return bid, mid

def load_positions(include_resolved, include_paper=False):
    rows = []
    if not os.path.exists(C.TRADE_LOG): return rows
    fills = [f for f in csv.DictReader(open(C.TRADE_LOG)) if include_paper or f["dry_run"] != "True"]
    state = json.load(open(C.STATE_FILE)) if os.path.exists(C.STATE_FILE) else {"positions": {}}
    # group fills by question
    by_q = {}
    for f in fills: by_q.setdefault(f["question"], []).append(f)
    for q, fs in by_q.items():
        shares = sum(float(f["shares"]) for f in fs); usd = sum(float(f["usd"]) for f in fs)
        if shares <= 0: continue
        city = fs[0]["city"]; day = dt.date.fromisoformat(fs[0]["target_day"])
        cid = next((k for k, v in state.get("positions", {}).items() if v.get("question") == q), None)
        rows.append(dict(question=q, city=city, day=day, shares=shares, usd=usd, avg=usd / shares, why=fs[0]["why"], p_ewma=float(fs[0]["p_ewma"]), p_ridge=float(fs[0]["p_ridge"]), cid=cid, dry=fs[0]["dry_run"] == "True", n=len(fs)))
    return rows

def render(rows, wallet=None, include_resolved=False):
    now = dt.datetime.now(dt.timezone.utc); out = []; tot_cost = tot_val = tot_real = 0.0
    out.append(f"{B}Evening bot positions{N}  {D}{now.strftime('%Y-%m-%d %H:%M UTC')}{N}")
    out.append(f"{D}{'city / day':<22}{'bucket':<12}{'shares':>8}{'cost':>8}{'avg':>7}{'bid':>7}{'mid':>7}{'value':>8}{'P&L':>9}  {'obs high':<12}{'models (E/R)':<14}{'why':<9}{'status'}{N}")
    for r in sorted(rows, key=lambda x: (x["day"], x["city"])):
        m = market_info(r["cid"]) if r["cid"] else None
        bucket = r["question"].split(" be ", 1)[1].split(" on")[0].replace("between ", "")
        tz = ZoneInfo(TZ[r["city"]]); local = now.astimezone(tz)
        resolved = bool(m and m.get("closed") and m.get("outcomePrices"))
        if resolved:
            yes = float(json.loads(m["outcomePrices"])[0]); val = r["shares"] * yes; pnl = val - r["usd"]; tot_real += pnl; status = f"{G}WON{N}" if yes == 1 else f"{R}LOST{N}"; bid = mid = yes
            if not include_resolved: continue
        else:
            tok = json.loads(m["clobTokenIds"])[0] if m and m.get("clobTokenIds") else None
            bid, mid = book(tok) if tok else (None, None); mark = bid if bid is not None else (mid or 0)
            val = r["shares"] * mark; pnl = val - r["usd"]; tot_cost += r["usd"]; tot_val += val
            status = f"{Y}open{N} ({local.strftime('%H:%M')} local)" if local.date() <= r["day"] else f"{Y}awaiting resolution{N}"
        omx, ot = obs_max(r["city"], r["day"]) if local.date() >= r["day"] else (None, None)
        obs = f"{omx}°F @{ot}" if omx is not None else "—"
        col = G if pnl >= 0 else R
        out.append(f"{r['city'][:14]:<14}{r['day'].strftime('%m-%d'):<8}{bucket:<12}{r['shares']:>8.1f}{r['usd']:>8.0f}{r['avg']:>7.2f}{(bid if bid is not None else float('nan')):>7.2f}{(mid if mid is not None else float('nan')):>7.2f}{val:>8.0f}{col}{pnl:>+9.0f}{N}  {obs:<12}{r['p_ewma']:.2f}/{r['p_ridge']:.2f}{'':<5}{r['why']:<9}{status}{'  [paper]' if r['dry'] else ''}")
    out.append(f"{D}{'-'*130}{N}")
    out.append(f"open: cost ${tot_cost:,.0f}  value ${tot_val:,.0f}  unrealized {G if tot_val-tot_cost>=0 else R}{tot_val-tot_cost:+,.0f}{N}" + (f"   realized (shown) {G if tot_real>=0 else R}{tot_real:+,.0f}{N}" if include_resolved else ""))
    if wallet:
        pos = get("https://data-api.polymarket.com/positions", {"user": wallet, "limit": 100}) or []
        out.append(f"\n{B}Exchange view of {wallet[:10]}…{N}  {len(pos)} positions")
        for p in pos:
            out.append(f"  {p.get('title','')[:70]:<70} {p.get('outcome'):<4} {float(p.get('size',0)):>8.1f} sh  avg {float(p.get('avgPrice',0)):.2f}  now {float(p.get('curPrice',0)):.2f}  P&L {float(p.get('cashPnl',0)):+.0f}")
    return "\n".join(out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--watch", type=int, default=0); ap.add_argument("--all", action="store_true"); ap.add_argument("--wallet", action="store_true"); ap.add_argument("--paper", action="store_true", help="include dry-run fills")
    ap.add_argument("--redeem", action="store_true", help="redeem resolved winners for every configured wallet now (needs OWNER_PRIVATE_KEY + POLYMARKET_PROXY + POLYMARKET_LOGIN, or POLYMARKET_WALLETS; dry unless --live)"); ap.add_argument("--live", action="store_true"); a = ap.parse_args()
    if a.redeem:
        import logging; logging.basicConfig(level=logging.INFO, format="%(message)s"); import bot.redeem as R
        for i, w in enumerate(C.WALLETS):
            if not w.get("key"): print(f"wallet #{i}: no key configured"); continue
            n, usd = R.redeem_all(w, dry_run=not a.live); print(f"wallet #{i}: {n} position(s), ${usd:.2f}{' (dry run: pass --live to send)' if not a.live else ''}")
        sys.exit(0)
    wallet = (os.environ.get("POLYMARKET_PROXY") or os.environ.get("POLY_ADDRESS") or C.FUNDER) if a.wallet else None
    while True:
        rows = load_positions(a.all, a.paper)
        n_paper = sum(1 for f in csv.DictReader(open(C.TRADE_LOG)) if f["dry_run"] == "True") if os.path.exists(C.TRADE_LOG) else 0
        txt = render(rows, wallet, a.all) if rows else f"{D}no live positions{' (' + str(n_paper) + ' paper fills in the log — add --paper to see them)' if n_paper and not a.paper else ''}{N}"
        if a.watch: os.system("clear")
        print(txt)
        if not a.watch: break
        time.sleep(a.watch)
