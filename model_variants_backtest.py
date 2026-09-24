"""Four candidate fixes for the lagging EWMA correction, backtested on the production rule.

Diagnosis they come from (Sep 2026 losing stretch): the per-city EWMA gain is picked on the whole history and
comes out at 0.05 for Los Angeles and Miami (14-day half-life), so when the HRRR bias shifts warm the correction
trails it by 1-2 F -- one whole bucket -- and the agree leg, which needs the EWMA to concur, loses systematically.

Levers, each swept on its own and then in combination:
  gain_floor    discard EWMA gains below this, forcing a faster correction         (baseline: no floor, gains 0.05..0.7)
  gain_window   pick the gain on the SSE of the last N one-step errors only        (baseline: all history)
  sd_window     residual window behind the bucket probabilities' sd                (baseline: 60)
  bias_gate     skip the city-day when |mean residual of the last K nights| > thr  (baseline: off)
  bias_shift    add s * (mean residual of the last K nights) to mu before the      (baseline: off)
                bucket probabilities

Unlike backtest_evening.py this runs on the bot's OWN history file (data/evening_history.parquet, 11 cities), which
is what the live Models actually fits, and it therefore also trades Chicago and Dallas -- see NOTE below.  Rule,
sizing, entry time, fee and hold-to-resolution are otherwise identical to backtest_evening.py.

NOTE: backtest_evening.py builds its panel from POOL (6 cities) but trades C.CITIES (7), so Chicago and Dallas have
no panel rows and are silently never traded there, although the live bot evaluates them every night. That is why
this script's baseline shows more trades than backtest_evening.py's 448.

Usage:  python3 model_variants_backtest.py            # all variants
        QUICK=1 python3 model_variants_backtest.py    # the headline variants only
"""
import pandas as pd, numpy as np, json, re, os, bisect, datetime as dt, warnings, itertools
from zoneinfo import ZoneInfo
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sweep_local import TZ
from bot import evening_config as C

warnings.filterwarnings("ignore"); pd.set_option("display.width", 260)

HIST      = os.environ.get("HIST", "data/evening_history.parquet")
START     = dt.date.fromisoformat(os.environ.get("START", "2026-01-18"))
END       = dt.date.fromisoformat(os.environ.get("END", "2026-09-22"))
RECENT    = dt.date.fromisoformat(os.environ.get("RECENT", "2026-09-16"))   # the losing stretch the fixes are meant to address
MID       = dt.date.fromisoformat(os.environ.get("MID", "2026-08-01"))      # a wider recent window for context
ENTRY_MIN = int(os.environ.get("ENTRY_MIN", "35"))
MODEL_MAX = float(os.environ.get("MODEL_MAX", str(C.MODEL_MAX_PRICE)))
TRADE     = list(C.MODES)


# ---------------------------------------------------------------- panel
P = pd.read_parquet(HIST)
P["mday"] = pd.to_datetime(P.mday).dt.date
P = P.dropna(subset=["err"]).sort_values(["city", "mday"]).reset_index(drop=True)
print(f"history: {len(P)} city-days, {P.mday.min()}..{P.mday.max()}, {P.city.nunique()} cities "
      f"(pool = all of them, as in the live bot)")


# ---------------------------------------------------------------- markets & prices
def bucket(q):
    mm = re.search(r"between (-?\d+)-(-?\d+)|(-?\d+)°[FC] or below|(-?\d+)°[FC] or (?:above|higher)|(-?\d+)°[FC] on", q)
    if mm.group(1): return (int(mm.group(1)), int(mm.group(2)))
    if mm.group(3): return (-999, int(mm.group(3)))
    if mm.group(4): return (int(mm.group(4)), 999)
    return (int(mm.group(5)), int(mm.group(5)))

MK = {}
for m in json.load(open("data/markets.json")):
    if not re.search(rf'highest temperature in ({"|".join(TRADE)}) be ', m["question"] or "") or not m.get("closed"): continue
    city = re.search(r"temperature in (.+?) be ", m["question"]).group(1); d = dt.date.fromisoformat(m["end_date"][:10])
    MK.setdefault((city, d), {})[bucket(m["question"])] = m

def yes_price(m, ts):
    try: h = json.load(open(f"data/prices/{m['market_id']}.json"))
    except FileNotFoundError: return None
    hs = h["history"]
    if not hs: return None
    i = bisect.bisect_right([x["t"] for x in hs], ts) - 1
    if i < 0 or ts - hs[i]["t"] > 2 * 3600: return None
    p = hs[i]["p"]; return p if h["losing_outcome"] == "Yes" else 1 - p

_px = {}
def prices_for(city, d):
    """{bucket: (yes price at 21:35 local the evening before, won)} for a city-day."""
    if (city, d) not in _px:
        out = {}
        if (city, d) in MK:
            tz = ZoneInfo(TZ[city]); ts = int(dt.datetime(d.year, d.month, d.day, 21, tzinfo=tz).timestamp()) - 86400 + ENTRY_MIN * 60
            for b, m in MK[(city, d)].items():
                v = yes_price(m, ts)
                if v is not None: out[b] = (v, [float(x) for x in m["outcome_prices"]] == [1.0, 0.0])
        _px[(city, d)] = out
    return _px[(city, d)]


# ---------------------------------------------------------------- the model, parameterised
class Models:
    def __init__(self, hist, gain_floor=0.0, gain_window=None, sd_window=60, floor_cities=None):
        self.hist = hist.dropna(subset=["err"]).copy(); self.cities = sorted(self.hist.city.unique())
        self.ewma = {}; self.ewma_sd = {}; self.ridge_sd = {}
        for c in self.cities:
            fl = gain_floor if (floor_cities is None or c in floor_cities) else 0.0   # per-city floor
            gains = [k for k in C.EWMA_GAINS if k >= fl] or [max(C.EWMA_GAINS)]
            e = self.hist[self.hist.city == c].sort_values("mday").err.values
            best = None
            for k in gains:
                b = 0.0; se = []
                for x in e: se.append(x - b); b += k * (x - b)
                w = se if gain_window is None else se[-gain_window:]          # pick the gain on a trailing window
                s = float(np.sum(np.square(w)))
                if best is None or s < best[0]: best = (s, k, b, np.std(se[-sd_window:]))
            self.ewma[c] = (best[1], best[2]); self.ewma_sd[c] = max(best[3], 1.0)
        X = self.hist[C.FEATS].copy(); self.med = X.median().fillna(0); X = X.fillna(self.med)
        self.mu = X.mean(); self.sd = X.std().replace(0, 1)
        Z = np.c_[((X - self.mu) / self.sd).values, pd.get_dummies(self.hist.city).reindex(columns=self.cities, fill_value=0).values]
        self.ridge = Ridge(alpha=C.RIDGE_ALPHA).fit(Z, self.hist.err.values)
        res = self.hist.err.values - self.ridge.predict(Z)
        for c in self.cities: self.ridge_sd[c] = max(float(np.std(res[(self.hist.city == c).values][-sd_window:])), 1.0)
    def predict(self, city, feats):
        x = pd.Series({f: feats.get(f, np.nan) for f in C.FEATS}).fillna(self.med)
        z = np.r_[((x - self.mu) / self.sd).values, [1.0 if c == city else 0.0 for c in self.cities]]
        return dict(ewma=self.ewma[city][1], ridge=float(self.ridge.predict(z[None, :])[0]),
                    ewma_sd=self.ewma_sd[city], ridge_sd=self.ridge_sd[city])

def bucket_probs(mu, sd, buckets):
    return {b: norm.cdf((min(b[1], 200) + 0.5 - mu) / sd) - norm.cdf((max(b[0], -200) - 0.5 - mu) / sd) for b in buckets}


# ---------------------------------------------------------------- stage 1: walk-forward forecasts
_fc = {}
def forecasts(gain_floor=0.0, gain_window=None, sd_window=60, floor_cities=None):
    """Walk-forward mu/sd per traded city-day for one model configuration (cached)."""
    key = (gain_floor, gain_window, sd_window, tuple(sorted(floor_cities)) if floor_cities else None)
    if key in _fc: return _fc[key]
    rows = []
    for d in sorted(x for x in P.mday.unique() if START <= x <= END):
        H = P[P.mday < d]
        if len(H) < C.MIN_HISTORY_DAYS: continue
        M = Models(H, gain_floor, gain_window, sd_window, floor_cities)
        for c in TRADE:
            r = P[(P.city == c) & (P.mday == d)]
            if r.empty: continue
            r = r.iloc[0]; p = M.predict(c, r.to_dict())
            rows.append(dict(city=c, mday=d, hrrr=r.hrrr, actual=r.actual,
                             mu_e=r.hrrr + p["ewma"], mu_r=r.hrrr + p["ridge"], sd_e=p["ewma_sd"], sd_r=p["ridge_sd"]))
    F = pd.DataFrame(rows)
    F["res_e"] = F.actual - F.mu_e; F["res_r"] = F.actual - F.mu_r        # realised only after the day is over
    _fc[key] = F
    return F


def trailing_bias(F, k):
    """Mean residual of the k nights BEFORE each row, per city (causal: shift(1) then rolling)."""
    F = F.sort_values(["city", "mday"]).copy()
    for col in ("e", "r"):
        F[f"bias_{col}"] = F.groupby("city")[f"res_{col}"].transform(lambda s: s.shift(1).rolling(k, min_periods=max(3, k // 2)).mean())
    return F


# ---------------------------------------------------------------- stage 2: the production rule
def run(F, bias_k=0, bias_gate=None, bias_shift=0.0):
    F = trailing_bias(F, bias_k) if bias_k else F.assign(bias_e=np.nan, bias_r=np.nan)
    trades = []
    for r in F.itertuples():
        pr = prices_for(r.city, r.mday)
        if len(pr) < 3: continue
        be_shift = (bias_shift * r.bias_e) if (bias_shift and pd.notna(r.bias_e)) else 0.0
        br_shift = (bias_shift * r.bias_r) if (bias_shift and pd.notna(r.bias_r)) else 0.0
        if bias_gate is not None and pd.notna(r.bias_e) and pd.notna(r.bias_r):
            if max(abs(r.bias_e), abs(r.bias_r)) > bias_gate: continue      # model is off-bias right now: stand down
        buckets = list(pr)
        Pe = bucket_probs(r.mu_e + be_shift, r.sd_e, buckets); Pr = bucket_probs(r.mu_r + br_shift, r.sd_r, buckets)
        be, br = max(Pe, key=Pe.get), max(Pr, key=Pr.get); agree = be == br
        modes = C.MODES.get(r.city, {"agree", "edge"}); spent = 0.0
        for b in buckets:
            ask, won = pr[b]; why = None
            if agree and "agree" in modes and b == be and C.AGREE_MIN_PRICE <= ask <= C.AGREE_MAX_PRICE: why = "agree"
            elif not agree and "ridge" in modes and b == br and C.MODEL_MIN_PRICE <= ask <= MODEL_MAX: why = "dis_ridge"
            elif not agree and "ewma" in modes and b == be and C.MODEL_MIN_PRICE <= ask <= MODEL_MAX: why = "dis_ewma"
            if not why: continue
            stake = min(C.STAKE, C.MAX_PER_MARKET_USD, C.MAX_PER_CITY_DAY_USD - spent)
            if stake < C.MIN_ORDER_SHARES * ask: continue
            spent += stake; sh = stake / ask; fee = 0.05 * ask * (1 - ask) * sh
            trades.append(dict(city=r.city, mday=r.mday, why=why, price=ask, stake=stake, won=won,
                               pnl=(sh - stake if won else -stake) - fee))
    return pd.DataFrame(trades)


def summ(t, tag=""):
    cols = ["n", "win", "pnl", "roi", "n_m", "pnl_m", "roi_m", "n_r", "win_r", "pnl_r"]
    if not len(t): return pd.Series(dict.fromkeys(cols, np.nan) | {"n": 0, "pnl": 0.0})
    m = t[t.mday >= MID]; r = t[t.mday >= RECENT]
    return pd.Series(dict(n=len(t), win=t.won.mean(), pnl=t.pnl.sum(), roi=t.pnl.sum() / t.stake.sum(),
                          n_m=len(m), pnl_m=m.pnl.sum(), roi_m=(m.pnl.sum() / m.stake.sum()) if len(m) else np.nan,
                          n_r=len(r), win_r=r.won.mean() if len(r) else np.nan, pnl_r=r.pnl.sum()))


def sd_study():
    """sd only enters the rule through argmax(bucket_probs), so it can only change a pick near a bucket edge or in
    the open-ended tail buckets. This measures how often it changes anything at all, down to very short windows."""
    base = forecasts(); rows = {}
    for w in (5, 7, 10, 14, 20, 30, 45, 60, 90, 120):
        F = forecasts(sd_window=w)
        j = base.merge(F, on=["city", "mday"], suffixes=("_b", "_w"))
        chg_e = (j.sd_e_b.round(6) != j.sd_e_w.round(6)).mean()
        T = run(F); r = summ(T)
        r["mean_sd_e"] = F.sd_e.mean(); r["mean_sd_r"] = F.sd_r.mean()
        r["sd_at_floor"] = ((F.sd_e <= 1.0001) | (F.sd_r <= 1.0001)).mean()
        r["sd_differs"] = chg_e
        rows[f"sd_window {w}" + (" (live)" if w == 60 else "")] = r
    R = pd.DataFrame(rows).T
    R.columns = ["n", "win", "pnl", "roi", f"n>={MID}", "pnl_aug", "roi_aug", f"n>={RECENT}", "win_bad", "pnl_bad",
                 "mean_sd_e", "mean_sd_r", "sd_at_floor", "sd_differs"]
    print(f"\n=== sd_window, short windows included ({START}..{END}) ===")
    print(R.round(3).to_string())
    # how often does the shorter sd actually change the PICK?
    print("\nnights where the chosen bucket changes vs the live 60-day sd (the only way sd can matter):")
    for w in (5, 10, 20, 120):
        F = forecasts(sd_window=w); n_e = n_r = n = 0
        for a, b in zip(base.itertuples(), F.itertuples()):
            pr = prices_for(a.city, a.mday)
            if len(pr) < 3: continue
            bk = list(pr); n += 1
            Pa, Pb = bucket_probs(a.mu_e, a.sd_e, bk), bucket_probs(b.mu_e, b.sd_e, bk)
            Qa, Qb = bucket_probs(a.mu_r, a.sd_r, bk), bucket_probs(b.mu_r, b.sd_r, bk)
            n_e += max(Pa, key=Pa.get) != max(Pb, key=Pb.get); n_r += max(Qa, key=Qa.get) != max(Qb, key=Qb.get)
        print(f"  sd_window {w:>3}: EWMA pick changes on {n_e}/{n} nights ({n_e/n:.1%}), ridge pick on {n_r}/{n} ({n_r/n:.1%})")
    R.to_parquet("out/sd_window_study.parquet")


if __name__ == "__main__":
    if os.environ.get("SD_ONLY") == "1":
        sd_study(); raise SystemExit
    quick = os.environ.get("QUICK") == "1"
    out = {}
    base_F = forecasts()
    out["baseline (live config)"] = summ(run(base_F))
    print(f"\nbaseline forecast accuracy: EWMA bias {base_F.res_e.mean():+.2f} MAE {base_F.res_e.abs().mean():.2f} | "
          f"ridge bias {base_F.res_r.mean():+.2f} MAE {base_F.res_r.abs().mean():.2f}")

    # --- lever 1: floor the EWMA gain
    for g in ([0.10, 0.20] if quick else [0.10, 0.15, 0.20, 0.30]):
        out[f"gain_floor {g}"] = summ(run(forecasts(gain_floor=g)))
    # --- lever 2: pick the gain on a trailing window
    for w in ([90, 180] if quick else [60, 90, 120, 180]):
        out[f"gain_window {w}d"] = summ(run(forecasts(gain_window=w)))
    # --- lever 3: shorter sd window
    for s in ([20, 30] if quick else [5, 7, 10, 14, 20, 30, 45, 120]):
        out[f"sd_window {s}"] = summ(run(forecasts(sd_window=s)))
    # --- lever 4a: stand down when the model is off-bias
    for k, thr in ([(7, 1.5), (14, 1.5)] if quick else [(7, 1.0), (7, 1.5), (7, 2.0), (14, 1.0), (14, 1.5), (14, 2.0)]):
        out[f"bias_gate k={k} >{thr}F"] = summ(run(base_F, bias_k=k, bias_gate=thr))
    # --- lever 4b: shift mu by the recent bias
    for k, s in ([(7, 1.0), (14, 0.5)] if quick else [(5, 0.5), (5, 1.0), (7, 0.5), (7, 1.0), (14, 0.5), (14, 1.0)]):
        out[f"bias_shift k={k} x{s}"] = summ(run(base_F, bias_k=k, bias_shift=s))

    R = pd.DataFrame(out).T
    R.columns = ["n", "win", "pnl", "roi", f"n>={MID}", "pnl_aug", "roi_aug", f"n>={RECENT}", "win_bad", "pnl_bad"]
    print(f"\n=== single levers, {START}..{END} (recent window = {RECENT}+) ===")
    print(R.round(3).to_string())

    if not quick:
        # --- combinations of whatever helped
        combo = {}
        for g, w in itertools.product([0.10, 0.20], [90, 180]):
            combo[f"floor {g} + window {w}d"] = summ(run(forecasts(gain_floor=g, gain_window=w)))
        for g in (0.10, 0.20):
            for k, s in ((7, 1.0), (14, 0.5)):
                combo[f"floor {g} + shift k={k} x{s}"] = summ(run(forecasts(gain_floor=g), bias_k=k, bias_shift=s))
            combo[f"floor {g} + sd 30"] = summ(run(forecasts(gain_floor=g, sd_window=30)))
        Rc = pd.DataFrame(combo).T; Rc.columns = R.columns
        print(f"\n=== combinations ===")
        print(Rc.round(3).to_string())
        R = pd.concat([R, Rc])
    # --- targeted: the lag is confined to the two slow-gain cities, so floor only those
    tgt = {}
    for g in (0.10, 0.20, 0.30):
        tgt[f"floor {g} on LA+Miami only"] = summ(run(forecasts(gain_floor=g, floor_cities={"Los Angeles", "Miami"})))
    for g in (0.10, 0.20):
        tgt[f"floor {g} LA+Miami + window 120d"] = summ(run(forecasts(gain_floor=g, gain_window=120, floor_cities={"Los Angeles", "Miami"})))
    Rt = pd.DataFrame(tgt).T; Rt.columns = R.columns
    print("\n=== targeted per-city floor (LA and Miami are the two 0.05-gain cities) ===")
    print(Rt.round(3).to_string())
    R = pd.concat([R, Rt])
    R.to_parquet("out/model_variants.parquet")
