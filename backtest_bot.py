"""Backtest of the bot's rule set:
  US (°F) stations : primary trigger = 5-min reading >= hi+3 -> buy NO at +8 min (feed lag), up to $25 (50% tranche)
                     confirmation   = hourly print > hi     -> at hourly obs time +4 min, top up to $50
                     hourly-first   = hourly print > hi     -> buy NO at +4 min, up to $50
  °C stations      : half-hourly METAR > hi                  -> buy NO at +5 min, up to $50
  Only if NO midpoint <= 0.95 at that moment; fill = midpoint + 0.5c (capped 0.95); fee 0.05*p*(1-p).
Sample events are random draws, so totals are scaled to the full population.
"""
import json, bisect, numpy as np, pandas as pd
MAX_ASK = 0.95; MAX_POS = 50.0; TRANCHE = 0.5; SLIP = 0.005; FEE = 0.05
LAG_5MIN = 8; LAG_HOURLY_US = 4; LAG_HOURLY_C = 5

def buy(price_mid, usd):
    p = min(price_mid + SLIP, MAX_ASK); sh = usd / p; fee = sh * FEE * p * (1 - p); return p, sh, fee

def price_at(market_id, ts):
    """NO midpoint from the 5-min series at/just before ts (fallback for times outside the 1-min window)."""
    try: h = json.load(open(f"data/prices/{market_id}.json"))
    except FileNotFoundError: return None
    if not h["history"]: return None
    hs = sorted((x["t"], x["p"]) for x in h["history"]); i = bisect.bisect_right([t for t, _ in hs], ts) - 1
    if i < 0: return None
    p = hs[i][1]; return p if h["losing_outcome"] == "No" else 1 - p     # series is the losing token; NO price wanted

# ---- US: +3F study sample ------------------------------------------------------------
from zoneinfo import ZoneInfo
from sweep_local import TZ
ev = pd.read_parquet("out/signal_events_m3.parquet"); smp = pd.read_csv("out/signal_study_m3.csv", dtype={"market_id": str})
smp = smp.merge(ev[["market_id", "signal_ts", "th", "city"]], on="market_id"); scale_us = len(ev) / len(smp)
out = []
for r in smp.itertuples():
    usd = sh = fee = 0.0; legs = []; tz = ZoneInfo(TZ[r.city])
    if r.trigger == "hourly":
        if pd.notna(r.m4) and r.m4 <= MAX_ASK:
            p, sh, fee = buy(r.m4, MAX_POS); usd = MAX_POS; legs = ["hourly"]
    else:
        if pd.notna(r.m8) and r.m8 <= MAX_ASK:
            p, sh, fee = buy(r.m8, MAX_POS * TRANCHE); usd = MAX_POS * TRANCHE; legs = ["5min"]
            if r.hourly_confirms and pd.notna(r.th):
                th_ts = int(pd.Timestamp(r.th).tz_localize(tz).timestamp()); pc = price_at(r.market_id, th_ts + 60 * LAG_HOURLY_US)
                if pc is not None and pc <= MAX_ASK:
                    p2, s2, f2 = buy(pc, MAX_POS - usd); usd = MAX_POS; sh += s2; fee += f2; legs.append("confirm")
    out.append(dict(kind="US-" + r.trigger, city=r.city, usd=usd, shares=sh, fee=fee, won=r.no_won, scale=scale_us, legs="+".join(legs)))
us = pd.DataFrame(out)
# ---- °C stations: hourly-print events sample ---------------------------------------------
evc = pd.read_parquet("out/events.parquet"); sc = pd.read_csv("out/event_study.csv")
site = json.load(open("out/stations.json")); c_cities = {c for c, (s, u) in site.items() if u == "C"}
evc = evc[evc.city.isin(c_cities)]; sc = sc[sc.city.isin(c_cities)]
scale_c = len(evc) / len(sc); rows = []
for r in sc.itertuples():
    p0 = r.m5; usd = sh = fee = 0
    if pd.notna(p0) and p0 <= MAX_ASK:
        p, sh, fee = buy(p0, MAX_POS); usd = MAX_POS
    rows.append(dict(kind="C-halfhourly", city=r.city, usd=usd, shares=sh, fee=fee, won=r.no_won, scale=scale_c, legs="hourly" if usd else ""))
cc = pd.DataFrame(rows)
allb = pd.concat([us, cc]); allb["payout"] = allb.shares * allb.won; allb["pnl"] = allb.payout - allb.usd - allb.fee
t = allb[allb.usd > 0]
def summ(g):
    return pd.Series(dict(signals=int(g.scale.sum().round()), trades=int((g.usd > 0).mul(g.scale).sum().round()), deployed=(g.usd * g.scale).sum(), pnl=(g.pnl * g.scale).sum(),
                          roi=(g.pnl * g.scale).sum() / max((g.usd * g.scale).sum(), 1), win=(g[g.usd > 0].won.mean() if (g.usd > 0).any() else np.nan), avg_usd=g[g.usd > 0].usd.mean() if (g.usd > 0).any() else 0))
pd.set_option("display.width", 200)
res = allb.groupby("kind").apply(summ); res.loc["TOTAL"] = summ(allb); print(res.round(3).to_string())
print("\nlegs (US 5-min triggers that traded):", us[(us.usd > 0) & us.kind.eq("US-5min")].legs.value_counts().to_dict())
print(f"\nSCALED TO JAN-SEP 16 (259 days): trades {res.loc['TOTAL','trades']:.0f} ({res.loc['TOTAL','trades']/259:.1f}/day), deployed ${res.loc['TOTAL','deployed']:,.0f}, P&L ${res.loc['TOTAL','pnl']:,.0f}, ROI {res.loc['TOTAL','roi']:+.1%}, win {res.loc['TOTAL','win']:.1%}")
byc = allb[allb.usd > 0].groupby("city").apply(lambda g: pd.Series(dict(trades=int((g.scale).sum().round()), pnl=(g.pnl * g.scale).sum(), win=g.won.mean()))).sort_values("pnl", ascending=False)
print("\nby city (scaled):"); print(byc.round(2).head(12).to_string()); print("..."); print(byc.round(2).tail(5).to_string())
allb.to_csv("out/backtest_bot.csv", index=False)
