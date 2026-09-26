"""Backtest the bot's execution cycle against real price paths.

Real: the 5-minute midpoint series for each market, so whether the market came down to a resting bid during a
cycle is taken from the actual path, not assumed. Simulated: the touch size and the bid, drawn from the measured
distributions (10089 exact depth observations recovered from cleared sweep levels; 2c median spread from 81 live
books), because no historical order book exists. Queue position and our own impact are not modelled, so read
this as a comparison between execution variants rather than an absolute P&L.

  python3 exec_backtest.py
"""
import bisect, datetime as dt, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from zoneinfo import ZoneInfo
import search as S, model_variants_backtest as MV, ladder_backtest as LB
from sweep_local import TZ
from bot import evening_config as C

SPREAD = 0.02
TOUCH = pd.read_csv("out/exact_depth.csv").query("dist < 0.005 and 0.05 <= price <= 0.55").shares.values


def orders():
    """The trades the live rule wants, with the bucket kept so we can read its price path."""
    P = S.augment(MV.P); F = S.walk(P, dict()); out = []
    for r in F.itertuples():
        pr = MV.prices_for(r.city, r.mday)
        if len(pr) < 3: continue
        bk = list(pr); Pe = MV.bucket_probs(r.mu_e, r.sd_e, bk); Pr = MV.bucket_probs(r.mu_r, r.sd_r, bk)
        be, br = max(Pe, key=Pe.get), max(Pr, key=Pr.get); agree = be == br
        mo = C.MODES.get(r.city, set()); spent = 0.0
        for b in bk:
            ask, won = pr[b]; why = None; pmod = None
            if agree and "agree" in mo and b == be and C.AGREE_MIN_PRICE <= ask <= C.AGREE_MAX_PRICE: why, pmod = "agree", (Pe[b] + Pr[b]) / 2
            elif not agree and "ridge" in mo and b == br and C.MODEL_MIN_PRICE <= ask <= C.MODEL_MAX_PRICE: why, pmod = "dis_ridge", Pr[b]
            elif not agree and "ewma" in mo and b == be and C.MODEL_MIN_PRICE <= ask <= C.MODEL_MAX_PRICE: why, pmod = "dis_ewma", Pe[b]
            if not why: continue
            want = C.STAKE * max(0.0, 1 + C.EDGE_MULT * (pmod - ask))
            st = min(want, C.MAX_PER_MARKET_USD, C.MAX_PER_CITY_DAY_USD - spent)
            if st < C.MIN_ORDER_SHARES * ask: continue
            spent += st
            out.append(dict(city=r.city, mday=r.mday, bucket=b, mid0=ask, usd=st, won=won, why=why))
    return pd.DataFrame(out)


def simulate(O, cycle=(6, 9), timeout=25, cross_ticks=0, maker_ticks=1, jitter=(0, 3), seed=0, ndraw=60):
    """Walk each order through the cycle against its real 5-minute path."""
    rng = np.random.default_rng(seed); tot = []; fills = []; prices = []
    for _ in range(ndraw):
        pnl = 0.0; got_usd = 0.0; want_usd = 0.0; pxw = 0.0; pxs = 0.0
        for r in O.itertuples():
            m = LB.MK.get((r.city, r.mday), {}).get(r.bucket)
            if not m: continue
            ts, ps = LB.path(m["market_id"])
            if not ts: continue
            tz = ZoneInfo(TZ[r.city])
            t0 = int(dt.datetime(r.mday.year, r.mday.month, r.mday.day, 21, tzinfo=tz).timestamp()) - 86400 + 35 * 60
            t = t0 + rng.uniform(*jitter) * 60; deadline = t + timeout * 60
            rem = r.usd; want_usd += r.usd
            lo, hi = (C.AGREE_MIN_PRICE, C.AGREE_MAX_PRICE) if r.why == "agree" else (C.MODEL_MIN_PRICE, C.MODEL_MAX_PRICE)
            n = 0
            while rem > 0.5 and t < deadline:
                mid = LB.px_at(ts, ps, t)
                if mid is None: break
                ask = mid + SPREAD / 2; bid = mid - SPREAD / 2
                if n > 0 and not (lo <= ask <= hi): break               # market left the band
                # take the offer: whatever the touch holds, out to cross_ticks
                lim = ask + cross_ticks * 0.01
                cap = rng.choice(TOUCH) * (1 + cross_ticks)             # deeper reach ~ more size
                take = min(rem / lim, cap)
                if take >= C.MIN_ORDER_SHARES:
                    cost = take * lim; pnl += (take - cost) if r.won else -cost
                    pnl -= 0.05 * lim * (1 - lim) * take
                    rem -= cost; got_usd += cost; pxw += lim * take; pxs += take
                if rem <= 0.5: break
                # rest the remainder at bid + maker_ticks
                rp = min(bid + maker_ticks * 0.01, ask - 0.01)
                dur = rng.uniform(*cycle) * 60; t_end = min(t + dur, deadline)
                i, j = bisect.bisect_right(ts, t), bisect.bisect_right(ts, t_end)
                hit = any(ps[k] + SPREAD / 2 <= rp + 1e-9 for k in range(i, j))   # ask came down to our bid
                if hit:
                    sh = rem / rp; cost = rem
                    pnl += (sh - cost) if r.won else -cost
                    pnl -= 0.05 * rp * (1 - rp) * sh
                    got_usd += cost; pxw += rp * sh; pxs += sh; rem = 0.0
                t = t_end; n += 1
        tot.append(pnl); fills.append(got_usd / want_usd); prices.append(pxw / max(pxs, 1e-9))
    return np.mean(tot), np.std(tot), np.mean(fills), np.mean(prices)


if __name__ == "__main__":
    O = orders()
    print(f"{len(O)} orders, ${O.usd.sum():.0f} intended, mean touch {np.mean(TOUCH):.0f} sh (median {np.median(TOUCH):.0f})\n")
    print(f"{'variant':<44}{'pnl':>8}{'sd':>6}{'filled':>9}{'avg px':>9}")
    base = dict(cycle=(6, 9), timeout=25, cross_ticks=0, maker_ticks=1)
    runs = [("LIVE: cycle 6-9, touch only, bid+1c, 25min", base),
            ("cross +1c (reach one level deeper)", dict(base, cross_ticks=1)),
            ("rest at bid+2c", dict(base, maker_ticks=2)),
            ("rest at the bid (bid+0c)", dict(base, maker_ticks=0)),
            ("cycle 3-5 min", dict(base, cycle=(3, 5))),
            ("cycle 10-15 min", dict(base, cycle=(10, 15))),
            ("timeout 15 min", dict(base, timeout=15)),
            ("timeout 45 min", dict(base, timeout=45)),
            ("timeout 60 min", dict(base, timeout=60)),
            ("cross +1c and timeout 45", dict(base, cross_ticks=1, timeout=45))]
    for lab, kw in runs:
        m, s, f, p = simulate(O, **kw)
        print(f"{lab:<44}{m:>8.0f}{s:>6.0f}{f:>8.1%}{p:>9.3f}")
