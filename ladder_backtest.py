"""Limit-order ladder instead of rejecting a city-day whose signal bucket is priced above our band.

Production rule today (bot/evening_bot.py + backtest_evening.py): at ~21:35 local the evening before, the signal
bucket (the agreed bucket on agreement nights, the ridge/EWMA bucket on disagreement nights) is bought only if its
YES ask sits inside the band -- 0.10-0.53 for "agree", 0.05-0.45 for the model legs.  A bucket priced above the top
of the band is dropped for the night and never looked at again.

This script keeps every baseline trade untouched and asks what an above-band bucket would have been worth if,
instead of dropping it, we rested a buy limit that walks up towards the market:

    bid_k = min(band_max, (1 - t_k) * p_k)      t_k = T0 - TSTEP*k      p_k = mid at step k (re-pegged every 5 min)

The bid is one-directional -- it only ratchets UP.  Re-pegging onto a market that has fallen never pulls an existing
bid down; the older, higher bid stays in the book and is the one that gets hit.

t starts at 0.45 and the discount shrinks by 0.025 every 5 minutes, so the bid walks up from 0.55*p to 1.10*p over
110 minutes (22 steps) -- the last rungs are marketable, i.e. "buy 10% above the last trade just before the cutoff"
(see NOTE_ON_T below; the ladder is clamped by band_max either way).  The ladder stops at the cutoff, so nothing is
posted once the next model run lands.

Prices are the 5-minute CLOB midpoint series in data/prices/ (same source the baseline entry price comes from).
Fill model: a resting bid at L fills when the ask touches it, i.e. when mid + HALF_SPREAD <= L, and fills AT L; a
rung that is already marketable (L >= mid + HALF_SPREAD) pays the ask, mid + HALF_SPREAD.  HALF_SPREAD is the
conservatism knob and is swept.  Sizing, fee and hold-to-resolution are the baseline's ($10/bucket, taker fee
0.05*p*(1-p) per share, no exit).

NOTE_ON_T: the request says "t starting from 0.45 and increasing by 0.025 every 5 minutes", but also that the last
rung buys 10% above the market.  (1-t)*p reaches 1.10*p only if t *decreases* to -0.10, which is exactly 22 steps of
0.025 = 110 min ~ the 2-hour cutoff.  The walking-up reading is the default (TDIR=-1); TDIR=+1 runs the literal
reading (the bid walks away from the market) for comparison.

Usage:  python3 ladder_backtest.py              # main table + variants + sweeps
        VARIANTS=0 python3 ladder_backtest.py   # main table only
Requires out/buckets_full_prod.parquet (a full-range run of backtest_evening.py with the production MODES).
"""
import pandas as pd, numpy as np, json, re, os, bisect, datetime as dt, warnings
from zoneinfo import ZoneInfo
from sweep_local import TZ
from bot import evening_config as C

warnings.filterwarnings("ignore"); pd.set_option("display.width", 240)

BUCKETS   = os.environ.get("BUCKETS", "out/buckets_full_prod.parquet")
ENTRY_MIN = int(os.environ.get("ENTRY_MIN", "35"))      # ladder starts at 21:ENTRY_MIN local (bot start jitter median)
MODEL_MAX = float(os.environ.get("MODEL_MAX", str(C.MODEL_MAX_PRICE)))
STAKE     = C.STAKE
MODES     = C.MODES

# --- ladder parameters (defaults = the request) ---
T0          = float(os.environ.get("T0", "0.45"))
TSTEP       = float(os.environ.get("TSTEP", "0.025"))
TDIR        = int(os.environ.get("TDIR", "-1"))         # -1: discount shrinks (bid walks up); +1: literal reading
TMIN        = float(os.environ.get("TMIN", "-0.10"))    # last rung = 1.10 * mid
STEP_MIN    = float(os.environ.get("STEP_MIN", "5"))
CUTOFF_MIN  = float(os.environ.get("CUTOFF_MIN", "115"))# ladder ends here (next report); 22 rungs = 110 min
HALF_SPREAD = float(os.environ.get("HALF_SPREAD", "0.01"))
TICK        = 0.01
MAXAGE      = 40 * 60                                   # ignore a quote older than this when re-pegging


# ---------------------------------------------------------------- markets & 5-min price paths
def bucket(q):
    mm = re.search(r"between (-?\d+)-(-?\d+)|(-?\d+)°[FC] or below|(-?\d+)°[FC] or (?:above|higher)|(-?\d+)°[FC] on", q)
    if mm.group(1): return (int(mm.group(1)), int(mm.group(2)))
    if mm.group(3): return (-999, int(mm.group(3)))
    if mm.group(4): return (int(mm.group(4)), 999)
    return (int(mm.group(5)), int(mm.group(5)))

B = pd.read_parquet(BUCKETS)
B["mday"] = pd.to_datetime(B.mday).dt.date
CITIES = sorted(B.city.unique())
MK = {}
for m in json.load(open("data/markets.json")):
    if not re.search(rf'highest temperature in ({"|".join(CITIES)}) be ', m["question"] or "") or not m.get("closed"): continue
    city = re.search(r"temperature in (.+?) be ", m["question"]).group(1); d = dt.date.fromisoformat(m["end_date"][:10])
    MK.setdefault((city, d), {})[bucket(m["question"])] = m

_paths = {}
def path(mid):
    """(timestamps, YES midpoints) for a market, 5-minute fidelity."""
    if mid not in _paths:
        try: h = json.load(open(f"data/prices/{mid}.json"))
        except FileNotFoundError: _paths[mid] = ([], []); return _paths[mid]
        hs = h["history"]; flip = h["losing_outcome"] != "Yes"
        ts = [x["t"] for x in hs]; ps = [(1 - x["p"]) if flip else x["p"] for x in hs]
        _paths[mid] = (ts, ps)
    return _paths[mid]

def px_at(ts, ps, t, maxage=MAXAGE):
    i = bisect.bisect_right(ts, t) - 1
    if i < 0 or t - ts[i] > maxage: return None
    return ps[i]


# ---------------------------------------------------------------- the ladder
def run_ladder(ts, ps, t_start, band_max, t0=T0, tstep=TSTEP, tdir=TDIR, tmin=TMIN,
               step_min=STEP_MIN, cutoff_min=CUTOFF_MIN, half=HALF_SPREAD, flat=False, clamp=True,
               cap_over=0.0, ratchet=True):
    """Walk the ladder from t_start for cutoff_min minutes. Returns (fill_price, minutes_in, how) or None.

    The bid is one-directional: it only ever ratchets UP (ratchet=True), so a re-peg onto a falling market never
    pulls an existing bid down -- the old, higher bid stays in the book and is the one that gets hit.
    cap    = band_max + cap_over, the closest to market we ever allow the ladder to get (clamp=False -> no cap).
    flat   = control: no t schedule, just rest at the cap the whole window."""
    cap = (band_max + cap_over) if clamp else 1.0
    n_steps = int(cutoff_min // step_min); bid = 0.0
    for k in range(n_steps + 1):
        tk = t_start + k * step_min * 60
        p = px_at(ts, ps, tk)
        if flat:
            want = cap
        else:
            t = t0 + tdir * tstep * k
            if tdir < 0 and t < tmin - 1e-9: t = tmin           # ladder exhausted -> hold the last rung
            if p is None: continue
            want = min(cap, (1 - t) * p)
        want = np.floor(want / TICK + 1e-9) * TICK
        bid = max(bid, want) if ratchet else want               # one-directional: never re-peg downwards
        if bid < TICK: continue
        if p is not None and bid >= p + half - 1e-9:                     # marketable rung: pay the ask
            return min(bid, p + half), k * step_min, "cross"
        lo, hi = bisect.bisect_right(ts, tk), bisect.bisect_right(ts, min(tk + step_min * 60, t_start + cutoff_min * 60))
        for j in range(lo, hi):
            if ps[j] + half <= bid + 1e-9:                               # ask came down to our resting bid
                return bid, (ts[j] - t_start) / 60.0, ("rest_cap" if abs(bid - np.floor(cap / TICK + 1e-9) * TICK) < 1e-9 else "rest")
    return None


def candidates():
    """One row per night per city: the signal bucket the rule looks at, its band, its entry price and outcome."""
    rows = []
    for (city, d), g in B.groupby(["city", "mday"], sort=True):
        modes = MODES.get(city, set()); agree = bool(g.agree.iloc[0])
        be, br = int(g.be.iloc[0]), int(g.br.iloc[0])
        if agree and "agree" in modes:   lo, why, band = be, "agree", (C.AGREE_MIN_PRICE, C.AGREE_MAX_PRICE)
        elif not agree and "ridge" in modes: lo, why, band = br, "dis_ridge", (C.MODEL_MIN_PRICE, MODEL_MAX)
        elif not agree and "ewma" in modes:  lo, why, band = be, "dis_ewma", (C.MODEL_MIN_PRICE, MODEL_MAX)
        else: continue
        r = g[g.lo == lo]
        if r.empty: continue
        r = r.iloc[0]
        rows.append(dict(city=city, mday=d, why=why, lo=lo, hi=int(r.hi), price=float(r.price), band_lo=band[0],
                         band_hi=band[1], won=bool(r.won), pe=float(r.pe), pr=float(r.pr)))
    return pd.DataFrame(rows)


def simulate(cand, which="above", **kw):
    """Ladder the candidates: which="above" (the rejections) or "in" (the trades the baseline already takes,
    i.e. the ladder used as an execution rule instead of crossing at 21:35). Returns the fills."""
    out = []
    for r in cand.itertuples():
        if which == "above" and r.price <= r.band_hi: continue      # baseline already trades it
        if which == "in" and not (r.band_lo <= r.price <= r.band_hi): continue
        m = MK.get((r.city, r.mday), {}).get((r.lo, r.hi))
        if m is None: continue
        ts, ps = path(m["market_id"])
        if not ts: continue
        tz = ZoneInfo(TZ[r.city]); t0 = int(dt.datetime(r.mday.year, r.mday.month, r.mday.day, 21, tzinfo=tz).timestamp()) - 86400 + ENTRY_MIN * 60
        res = run_ladder(ts, ps, t0, r.band_hi, **kw)
        if res is None: continue
        px, mins, how = res
        sh = STAKE / px; fee = 0.05 * px * (1 - px) * sh
        out.append(dict(city=r.city, mday=r.mday, why=r.why, bucket=(r.lo, r.hi), entry_px=r.price, price=px,
                        px_drop=r.price - px, mins=mins, how=how, stake=STAKE, won=r.won,
                        pnl=(sh - STAKE if r.won else -STAKE) - fee))
    return pd.DataFrame(out)


def summ(t):
    if not len(t): return pd.Series(dict(n=0, win=np.nan, avg_px=np.nan, stake=0.0, pnl=0.0, roi=np.nan))
    return pd.Series(dict(n=len(t), win=t.won.mean(), avg_px=t.price.mean(), stake=t.stake.sum(),
                          pnl=t.pnl.sum(), roi=t.pnl.sum() / t.stake.sum()))


if __name__ == "__main__":
    cand = candidates()
    base = cand[(cand.price >= cand.band_lo) & (cand.price <= cand.band_hi)].copy()
    base["sh"] = STAKE / base.price
    base["pnl"] = np.where(base.won, base.sh - STAKE, -STAKE) - 0.05 * base.price * (1 - base.price) * base.sh
    base["stake"] = STAKE
    above = cand[cand.price > cand.band_hi]; below = cand[cand.price < cand.band_lo]
    print(f"nights with a signal bucket: {len(cand)}   in band (baseline trades): {len(base)}   "
          f"above band (ladder candidates): {len(above)}   below band (skipped, not laddered): {len(below)}")
    print("\nabove-band candidates by signal and city:")
    print(above.groupby(["why", "city"]).agg(n=("price", "size"), entry_px=("price", "mean"), won=("won", "mean")).round(3).to_string())

    L = simulate(cand)
    print(f"\n=== LADDER  t {T0:+.3f} -> {TMIN:+.3f} by {TSTEP} every {STEP_MIN:.0f} min (dir {TDIR:+d}), cutoff {CUTOFF_MIN:.0f} min, "
          f"half-spread {HALF_SPREAD:.3f}, entry 21:{ENTRY_MIN:02d} local")
    print(f"fills {len(L)} of {len(above)} candidates ({len(L)/max(len(above),1):.0%})")
    if len(L):
        print("\nby signal:");            print(L.groupby("why").apply(summ).round(3).to_string())
        print("\nby city:");              print(L.groupby("city").apply(summ).round(3).to_string())
        print("\nby fill mechanism:");    print(L.groupby("how").apply(summ).round(3).to_string())
        L["month"] = pd.to_datetime(L.mday).dt.to_period("M")
        print("\nby month:");             print(L.groupby("month").apply(summ).round(2).to_string())
        L["bin"] = pd.cut(L.price, [0, .1, .2, .3, .4, .45, .5, .53])
        print("\nby fill price:");        print(L.groupby("bin", observed=True).apply(summ).round(3).to_string())
        print(f"\nfill: mean {L.mins.mean():.0f} min into the ladder (median {L.mins.median():.0f}), "
              f"mean entry {L.entry_px.mean():.3f} -> fill {L.price.mean():.3f} (drop {L.px_drop.mean():.3f})")
        print("\nLADDER ALL:", summ(L).round(3).to_dict())
    print("\nBASELINE ALL:", summ(base).round(3).to_dict())
    both = pd.concat([base.assign(how="baseline"), L], ignore_index=True)
    print("BASELINE+LADDER:", summ(both).round(3).to_dict())
    d = both.groupby("mday").pnl.sum().sort_index().cumsum(); db = base.groupby("mday").pnl.sum().sort_index().cumsum()
    print(f"max drawdown: baseline ${(db-db.cummax()).min():.0f} -> with ladder ${(d-d.cummax()).min():.0f}")
    L.drop(columns=[c for c in ("bin", "month") if c in L], errors="ignore").to_parquet("out/ladder_trades.parquet")
    cand.to_parquet("out/ladder_candidates.parquet")

    # ---------------- why the ladder rarely fills: these markets barely move overnight ----------------
    rows = []
    for r in above.itertuples():
        m = MK.get((r.city, r.mday), {}).get((r.lo, r.hi))
        if m is None: continue
        ts, ps = path(m["market_id"])
        if not ts: continue
        tz = ZoneInfo(TZ[r.city]); t0 = int(dt.datetime(r.mday.year, r.mday.month, r.mday.day, 21, tzinfo=tz).timestamp()) - 86400 + ENTRY_MIN * 60
        def lowest(h):
            i, j = bisect.bisect_right(ts, t0), bisect.bisect_right(ts, t0 + h * 3600)
            return min(ps[i:j]) if j > i else np.nan
        rows.append(dict(city=r.city, mday=r.mday, why=r.why, entry=r.price, band=r.band_hi, won=r.won,
                         over=r.price - r.band_hi, **{f"m{h}": lowest(h) for h in (1, 2, 4, 8, 14)}))
    D = pd.DataFrame(rows)
    print("\n\n=== diagnostics on the 178 above-band candidates ===")
    print("distance above the cap:"); print(pd.cut(D.over, [0, .02, .05, .1, .2, .5]).value_counts().sort_index().to_string())
    print("\nhow often the mid ever comes back into the band, by horizon from 21:35 local:")
    for h in (1, 2, 4, 8, 14):
        print(f"  within {h:>2}h: touches band_max-1c on {(D[f'm{h}'] <= D.band - 0.01).mean():5.1%} of nights; "
              f"median deepest mid {D[f'm{h}'].median():.3f} vs entry {D.entry.median():.3f}")
    print(f"\n2h max drawdown of the mid: median {(D.entry - D.m2).median():.3f}, mean {(D.entry - D.m2).mean():.3f}, p90 {(D.entry - D.m2).quantile(.9):.3f}")
    ev = lambda g: pd.Series(dict(n=len(g), px=g.price.mean(), win=g.won.mean(), stake=STAKE * len(g),
                                  pnl=(np.where(g.won, STAKE / g.price - STAKE, -STAKE) - 0.05 * g.price * (1 - g.price) * STAKE / g.price).sum()))
    e = above.groupby("why").apply(ev); e["roi"] = e.pnl / e.stake
    print("\nupper bound on what the band is costing us -- buying every above-band candidate at the 21:35 mid:")
    print(e.round(3).to_string())
    print("\nsame, split by distance above the cap:")
    e = above.groupby([above.why, pd.cut(above.price - above.band_hi, [0, .02, .05, .1, .2, .5])], observed=True).apply(ev); e["roi"] = e.pnl / e.stake
    print(e.round(3).to_string())
    e = below.groupby("why").apply(ev); e["roi"] = e.pnl / e.stake
    print("\nbelow-band candidates (the other rejection; the ladder formula does not apply to them):")
    print(e.round(3).to_string())

    if os.environ.get("VARIANTS", "1") == "1":
        print("\n\n=== variants (n / win / avg fill px / staked / pnl / roi) ===")
        rows = {}
        rows["ladder (default)"] = summ(L)
        rows["literal t (+0.025/5min)"] = summ(simulate(cand, tdir=+1))
        rows["flat bid at band_max"] = summ(simulate(cand, flat=True))
        rows["unclamped (pay above band)"] = summ(simulate(cand, clamp=False))
        rows["unclamped, cutoff 240"] = summ(simulate(cand, clamp=False, cutoff_min=240))
        for hs in (0.0, 0.005, 0.02):
            rows[f"half-spread {hs}"] = summ(simulate(cand, half=hs))
        for cm in (30, 60, 110, 180, 240):
            rows[f"cutoff {cm} min"] = summ(simulate(cand, cutoff_min=cm))
        for t0 in (0.0, 0.15, 0.30, 0.45, 0.60):
            rows[f"T0 {t0}"] = summ(simulate(cand, t0=t0))
        for st in (0.0125, 0.025, 0.05):
            rows[f"step {st}"] = summ(simulate(cand, tstep=st))
        for tm in (0.0, -0.05, -0.10, -0.20):
            rows[f"last rung t {tm}"] = summ(simulate(cand, tmin=tm))
        rows["no ratchet (re-peg down too)"] = summ(simulate(cand, ratchet=False))
        print(pd.DataFrame(rows).T.round(3).to_string())

        # ---- how close to the market may the ladder get?  cap = band_max + cap_over ----
        print("\n=== how close to market the ladder is allowed to get (cap = band_max + x) ===")
        rows = {}
        for x in (0.0, 0.02, 0.05, 0.10, 0.20, 0.30):
            r = summ(simulate(cand, cap_over=x)); r["fill_rate"] = r["n"] / len(above); rows[f"band_max +{x:.2f}"] = r
        r = summ(simulate(cand, clamp=False)); r["fill_rate"] = r["n"] / len(above); rows["no cap (walks to 1.10x)"] = r
        print(pd.DataFrame(rows).T.round(3).to_string())

        # ---- the same ladder as an EXECUTION rule on the trades we already take ----
        print("\n=== ladder as execution on the in-band trades (vs crossing at 21:35) ===")
        rows = {}
        rows["baseline: cross at 21:35 (mid)"] = summ(base)
        bs = base.copy(); bs["price"] = bs.price + HALF_SPREAD; bs["sh"] = STAKE / bs.price      # same spread convention as the ladder
        bs["pnl"] = np.where(bs.won, bs.sh - STAKE, -STAKE) - 0.05 * bs.price * (1 - bs.price) * bs.sh
        rows["baseline: cross at 21:35 (+1c)"] = summ(bs)
        for x, name, kw in ((0.0, "ladder, cap band_max", {}), (None, "ladder, no cap", {}), (None, "ladder, no cap, mid fills", {"half": 0.0})):
            I = simulate(cand, which="in", **({"clamp": False} if x is None else {"cap_over": x}), **kw)
            r = summ(I); r["fill_rate"] = len(I) / len(base)
            r["missed_pnl"] = base.pnl.sum() - base[[k in set(zip(I.city, I.mday)) for k in zip(base.city, base.mday)]].pnl.sum() if len(I) else base.pnl.sum()
            rows[name] = r
        print(pd.DataFrame(rows).T.round(3).to_string())
        print("(missed_pnl = baseline P&L of the in-band nights the ladder never filled -- P&L given up by waiting)")
