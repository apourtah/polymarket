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

ALL_CITIES = sorted(pd.read_parquet(HIST).city.unique())
MK = {}
for m in json.load(open("data/markets.json")):
    if not re.search(rf'highest temperature in ({"|".join(ALL_CITIES)}) be ', m["question"] or "") or not m.get("closed"): continue
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


# ---------------------------------------------------------------- external seasonal prior
def seasonal_prior(exclude_year=2026, k=60.0, path="out/seasonality_pairs.parquet"):
    """{(city, month): F} -- the day-ahead bias SHAPE by month, from seasonality.py's four-year series.

    Only the shape is used: each city's own annual mean is removed, because the LEVEL differs between the proxy
    (Open-Meteo best_match) and the bot's HRRR, and the level is what the EWMA already tracks. What the models
    cannot know in advance is the TURN -- how the bias moves from this month to the next -- and that is what this
    supplies. Sparse city-months are shrunk toward the pooled profile by n/(n+k). Years >= exclude_year are left
    out so nothing leaks from the period being scored."""
    if not os.path.exists(path): return {}
    D = pd.read_parquet(path)
    D = D[D.year < exclude_year]
    if not len(D): return {}
    city_mean = D.groupby("city").err.mean()
    D = D.assign(anom=D.err - D.city.map(city_mean))
    pooled = D.groupby("month").anom.mean()
    g = D.groupby(["city", "month"]).anom.agg(["mean", "size"])
    out = {}
    for (c, m), r in g.iterrows():
        lam = r["size"] / (r["size"] + k)
        out[(c, m)] = lam * r["mean"] + (1 - lam) * pooled.get(m, 0.0)
    out["_pooled"] = pooled.to_dict()
    return out

def prior_at(prior, city, month):
    if not prior: return 0.0
    if (city, month) in prior: return float(prior[(city, month)])
    return float(prior.get("_pooled", {}).get(month, 0.0))


# ---------------------------------------------------------------- the model, parameterised
class Models:
    def __init__(self, hist, gain_floor=0.0, gain_window=None, sd_window=60, floor_cities=None,
                 ridge_window=None, ridge_halflife=None, ridge_alpha=None, seasonal=False,
                 per_city=False, ridge_bias_gain=None, ewma_window=None, ewma_gain=None, flat_window=None,
                 prior=None, prior_mode=None):
        self.hist = hist.dropna(subset=["err"]).copy(); self.cities = sorted(self.hist.city.unique())
        self.seasonal = seasonal; self.per_city = per_city
        self.prior = prior or {}; self.prior_mode = prior_mode
        self.hist["_pri"] = [prior_at(self.prior, c, m.month) for c, m in zip(self.hist.city, pd.to_datetime(self.hist.mday))]
        if prior_mode in ("offset", "offset_ridge", "offset_ewma"):
            self.hist = self.hist.assign(err=self.hist.err - self.hist._pri)   # model the anomaly, add the prior back later
        self.ewma = {}; self.ewma_sd = {}; self.ridge_sd = {}
        for c in self.cities:
            fl = gain_floor if (floor_cities is None or c in floor_cities) else 0.0   # per-city floor
            gains = [ewma_gain] if ewma_gain else ([k for k in C.EWMA_GAINS if k >= fl] or [max(C.EWMA_GAINS)])
            ce = self.hist[self.hist.city == c].sort_values("mday")
            if ewma_window:                                                   # only the last N days of error history
                ce = ce[ce.mday > ce.mday.max() - dt.timedelta(days=ewma_window)]
            e = (ce.err + ce._pri).values if prior_mode == "offset_ridge" else ce.err.values
            if len(e) == 0: self.ewma[c] = (gains[0], 0.0); self.ewma_sd[c] = 1.0; continue
            if flat_window:                                                   # equal weight inside a window, no decay
                se = [x - (np.mean(e[max(0, i - flat_window):i]) if i else 0.0) for i, x in enumerate(e)]
                b = float(np.mean(e[-flat_window:]))
                self.ewma[c] = (0.0, b); self.ewma_sd[c] = max(float(np.std(se[-sd_window:])), 1.0); continue
            best = None
            for k in gains:
                b = 0.0; se = []
                for x in e: se.append(x - b); b += k * (x - b)
                w = se if gain_window is None else se[-gain_window:]          # pick the gain on a trailing window
                s = float(np.sum(np.square(w)))
                if best is None or s < best[0]: best = (s, k, b, np.std(se[-sd_window:]))
            self.ewma[c] = (best[1], best[2]); self.ewma_sd[c] = max(best[3], 1.0)
        R = self.hist
        if ridge_window:                                                     # fit on a trailing window only
            R = R[R.mday > R.mday.max() - dt.timedelta(days=ridge_window)]
            if len(R) < 200: R = self.hist
        self.rhist = R
        self.win_pri = float(R._pri.mean()) if len(R) else 0.0
        ytar = (R.err + R._pri).values if prior_mode == "offset_ewma" else R.err.values   # only the EWMA saw the anomaly
        X = self.feats(R); self.med = X.median().fillna(0); X = X.fillna(self.med)
        self.mu = X.mean(); self.sd = X.std().replace(0, 1)
        alpha = C.RIDGE_ALPHA if ridge_alpha is None else ridge_alpha
        w = None
        if ridge_halflife:                                                   # exponential recency weights
            age = np.array([(R.mday.max() - d).days for d in R.mday], dtype=float)
            w = np.power(0.5, age / ridge_halflife)
        Xn = ((X - self.mu) / self.sd).values
        if per_city:                                                         # a separate fit per city, no pooling
            self.ridge = {}; pred = np.zeros(len(R)); cv = R.city.values
            for c in self.cities:
                m = cv == c
                fit = m if m.sum() >= 60 else np.ones(len(R), bool)           # too few rows: fall back to the pooled sample
                self.ridge[c] = Ridge(alpha=alpha).fit(Xn[fit], ytar[fit], sample_weight=None if w is None else w[fit])
                if m.sum(): pred[m] = self.ridge[c].predict(Xn[m])            # batched, not row by row
        else:
            Z = np.c_[Xn, pd.get_dummies(R.city).reindex(columns=self.cities, fill_value=0).values]
            self.ridge = Ridge(alpha=alpha).fit(Z, ytar, sample_weight=w)
            pred = self.ridge.predict(Z)
        res = ytar - pred
        self.ridge_bias = {}
        for c in self.cities:
            m = (R.city == c).values
            self.ridge_sd[c] = max(float(np.std(res[m][-sd_window:])) if m.sum() else 1.0, 1.0)
            b = 0.0
            if ridge_bias_gain and m.sum():                                  # EWMA of the ridge's own recent residuals
                for x in res[m]: b += ridge_bias_gain * (x - b)
            self.ridge_bias[c] = b
    def feats(self, df):
        X = df[C.FEATS].copy()
        if self.prior_mode == "feature": X["prior"] = df["_pri"].values
        if self.seasonal:                                                    # doy is linear in FEATS; give the fit a real season
            X["doy_s"] = np.sin(2 * np.pi * df.doy / 365.25); X["doy_c"] = np.cos(2 * np.pi * df.doy / 365.25)
        return X

    def predict(self, city, feats):
        cols = list(self.med.index)
        md = feats.get("mday")
        month = pd.Timestamp(md).month if md is not None else max(1, min(12, int(feats.get("doy", 1)) // 31 + 1))
        pri = prior_at(self.prior, city, month)
        x = pd.Series({f: feats.get(f, np.nan) for f in cols})
        if self.prior_mode == "feature": x["prior"] = pri
        if self.seasonal:
            d = feats.get("doy", 1); x["doy_s"] = np.sin(2 * np.pi * d / 365.25); x["doy_c"] = np.cos(2 * np.pi * d / 365.25)
        x = x.reindex(cols).fillna(self.med); xn = ((x - self.mu) / self.sd).values
        if self.per_city: r = float(self.ridge[city].predict(xn[None, :])[0])
        else: r = float(self.ridge.predict(np.r_[xn, [1.0 if c == city else 0.0 for c in self.cities]][None, :])[0])
        e = self.ewma[city][1]; r = r + self.ridge_bias.get(city, 0.0)
        if self.prior_mode == "offset": e, r = e + pri, r + pri              # both models fitted on the anomaly
        elif self.prior_mode == "offset_ridge": r = r + pri                  # ridge only
        elif self.prior_mode == "offset_ewma": e = e + pri
        elif self.prior_mode == "delta": r = r + (pri - self.win_pri)        # season has moved since the fit window
        elif self.prior_mode == "delta_both": e, r = e + (pri - self.win_pri), r + (pri - self.win_pri)
        return dict(ewma=e, ridge=r, ewma_sd=self.ewma_sd[city], ridge_sd=self.ridge_sd[city])

def bucket_probs(mu, sd, buckets):
    return {b: norm.cdf((min(b[1], 200) + 0.5 - mu) / sd) - norm.cdf((max(b[0], -200) - 0.5 - mu) / sd) for b in buckets}


# ---------------------------------------------------------------- stage 1: walk-forward forecasts
_fc = {}
def forecasts(gain_floor=0.0, gain_window=None, sd_window=60, floor_cities=None, **mk):
    """Walk-forward mu/sd per traded city-day for one model configuration (cached)."""
    all_cities = mk.pop("all_cities", False)
    def _k(v):                                                   # the prior is a dict: summarise it for the cache key
        if isinstance(v, dict): return ("prior", len(v), round(sum(x for x in v.values() if isinstance(x, (int, float))), 6))
        if isinstance(v, (set, frozenset)): return tuple(sorted(v))
        return v
    key = (gain_floor, gain_window, sd_window, tuple(sorted(floor_cities)) if floor_cities else None,
           tuple(sorted((k, _k(v)) for k, v in mk.items())), all_cities)
    if key in _fc: return _fc[key]
    rows = []
    for d in sorted(x for x in P.mday.unique() if START <= x <= END):
        H = P[P.mday < d]
        if len(H) < C.MIN_HISTORY_DAYS: continue
        M = Models(H, gain_floor, gain_window, sd_window, floor_cities, **mk)
        for c in (ALL_CITIES if all_cities else TRADE):
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
def run(F, bias_k=0, bias_gate=None, bias_shift=0.0, modes_all=None, band=None, single=None):
    """modes_all: apply this mode set to every city instead of C.MODES (drops the per-city leg assignment).
    band: one (lo, hi) price band for every leg instead of the separately tuned agree / model bands.
    single: 'ewma' | 'ridge' | 'avg' | 'fav' -- ignore the agreement structure and buy that one bucket."""
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
        modes = modes_all if modes_all is not None else C.MODES.get(r.city, {"agree", "edge"}); spent = 0.0
        if single:                                                     # one bucket, no agreement structure at all
            if single == "ewma": pick = be
            elif single == "ridge": pick = br
            elif single == "fav": pick = max(buckets, key=lambda b: pr[b][0])
            else:
                Pa = bucket_probs((r.mu_e + r.mu_r) / 2, (r.sd_e + r.sd_r) / 2, buckets); pick = max(Pa, key=Pa.get)
            ask, won = pr[pick]
            lo, hi = band if band else (C.MODEL_MIN_PRICE, C.AGREE_MAX_PRICE)
            if not (lo <= ask <= hi): continue
            sh = C.STAKE / ask; fee = 0.05 * ask * (1 - ask) * sh
            trades.append(dict(city=r.city, mday=r.mday, why=single, price=ask, stake=C.STAKE, won=won,
                               pnl=(sh - C.STAKE if won else -C.STAKE) - fee)); continue
        alo, ahi = band if band else (C.AGREE_MIN_PRICE, C.AGREE_MAX_PRICE)
        mlo, mhi = band if band else (C.MODEL_MIN_PRICE, MODEL_MAX)
        for b in buckets:
            ask, won = pr[b]; why = None
            if agree and "agree" in modes and b == be and alo <= ask <= ahi: why = "agree"
            elif not agree and "ridge" in modes and b == br and mlo <= ask <= mhi: why = "dis_ridge"
            elif not agree and "ewma" in modes and b == be and mlo <= ask <= mhi: why = "dis_ewma"
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


def gain_study():
    """gain_window changes which EWMA gain is selected, which moves mu directly -- a much stronger lever than
    sd_window. Reports forecast accuracy as well as P&L: a window that does not improve the EWMA's own MAE has
    no reason to improve P&L except by chance."""
    base = forecasts(); rows = {}; picks = {}
    for w in (30, 45, 60, 90, 120, 150, 180, 240, None):
        F = forecasts(gain_window=w); T = run(F); r = summ(T)
        r["ewma_bias"] = F.res_e.mean(); r["ewma_mae"] = F.res_e.abs().mean()
        r["mu_shift"] = (F.mu_e - base.mu_e).abs().mean()          # how far the correction moves vs the live config
        rows[f"gain_window {w if w else 'all history (live)'}"] = r
        # which gain does each city end up on, at the end of the period?
        H = P[P.mday < END]; M = Models(H, gain_window=w)
        picks[w if w else "all"] = {c: M.ewma[c][0] for c in TRADE}
    R = pd.DataFrame(rows).T
    R.columns = ["n", "win", "pnl", "roi", f"n>={MID}", "pnl_aug", "roi_aug", f"n>={RECENT}", "win_bad", "pnl_bad",
                 "ewma_bias", "ewma_mae", "mu_shift"]
    print(f"\n=== gain_window sweep ({START}..{END}); baseline EWMA MAE {base.res_e.abs().mean():.3f} ===")
    print(R.round(3).to_string())
    print("\nEWMA gain selected per city (fit on history to {}):".format(END))
    print(pd.DataFrame(picks).T.to_string())
    R.to_parquet("out/gain_window_study.parquet")


def drift_study():
    """Is the fit drifting because it never forgets? The EWMA adapts, but the ridge is refit nightly on the WHOLE
    history with equal weight on January and September, and the bucket sd comes from a 60-residual window. This
    sweeps the ridge's memory (trailing window, exponential recency weights, regularisation), its structure
    (pooled-with-dummies vs per-city, linear doy vs real seasonal terms) and an EWMA bias correction on the ridge's
    own residuals. Forecast accuracy is reported alongside P&L: a lever that does not improve MAE has no business
    improving P&L except by luck."""
    base = forecasts(); rows = {}
    def add(name, **mk):
        F = forecasts(**mk); T = run(F); r = summ(T)
        r["ewma_mae"] = F.res_e.abs().mean(); r["ridge_bias"] = F.res_r.mean(); r["ridge_mae"] = F.res_r.abs().mean()
        rows[name] = r
    add("baseline (live)")
    for w in (90, 120, 180, 240, 365):   add(f"ridge_window {w}d", ridge_window=w)
    for hl in (30, 60, 90, 120, 180):    add(f"ridge_halflife {hl}d", ridge_halflife=hl)
    for a in (5, 10, 50, 100):           add(f"ridge_alpha {a}", ridge_alpha=a)
    add("seasonal doy sin/cos", seasonal=True)
    add("per-city ridge", per_city=True)
    for g in (0.05, 0.1, 0.2):           add(f"ridge_bias EWMA g={g}", ridge_bias_gain=g)
    # combinations of the memory levers
    add("halflife 90 + seasonal", ridge_halflife=90, seasonal=True)
    add("window 180 + alpha 50", ridge_window=180, ridge_alpha=50)
    add("halflife 120 + bias g=0.1", ridge_halflife=120, ridge_bias_gain=0.1)
    add("halflife 90 + sd 30", ridge_halflife=90, sd_window=30)
    R = pd.DataFrame(rows).T
    R.columns = ["n", "win", "pnl", "roi", f"n>={MID}", "pnl_aug", "roi_aug", f"n>={RECENT}", "win_bad", "pnl_bad",
                 "ewma_mae", "ridge_bias", "ridge_mae"]
    print(f"\n=== ridge memory / structure sweep ({START}..{END}) ===")
    print(R.round(3).to_string())
    R.to_parquet("out/drift_study.parquet")
    # --- is there drift to find in the first place? ---
    F = base.copy(); F["m"] = pd.to_datetime(F.mday).dt.to_period("M")
    print("\n=== baseline forecast error by month (all 7 traded cities) ===")
    print(F.groupby("m").apply(lambda g: pd.Series(dict(n=len(g), ewma_bias=g.res_e.mean(), ewma_mae=g.res_e.abs().mean(),
          ridge_bias=g.res_r.mean(), ridge_mae=g.res_r.abs().mean()))).round(3).to_string())
    print("\n=== ridge bias by city, first half vs last 6 weeks ===")
    a = F[F.mday < dt.date(2026, 7, 1)].groupby("city").res_r.mean(); b = F[F.mday >= dt.date(2026, 8, 15)].groupby("city").res_r.mean()
    print(pd.DataFrame({"to Jun 30": a, "Aug 15+": b, "drift": b - a}).round(2).to_string())


def weight_study():
    """A hard 180-day cutoff helped; does weighting recent days MORE inside that window help further? ridge_window
    truncates the fit sample, then ridge_halflife applies exponential recency weights within it. Also sweeps the
    window finely, to check the 180-240d plateau is real and not a two-point accident. ess = effective sample size
    (sum(w)^2 / sum(w^2)) at the end of the period -- how many city-days the fit is really using."""
    rows = {}
    def add(name, **mk):
        F = forecasts(**mk); T = run(F); r = summ(T)
        r["ridge_bias"] = F.res_r.mean(); r["ridge_mae"] = F.res_r.abs().mean()
        H = P[P.mday < END]; M = Models(H, **mk)
        R = M.rhist; w = np.ones(len(R))
        if mk.get("ridge_halflife"):
            age = np.array([(R.mday.max() - d).days for d in R.mday], dtype=float)
            w = np.power(0.5, age / mk["ridge_halflife"])
        r["rows"] = len(R); r["ess"] = w.sum() ** 2 / (w ** 2).sum()
        rows[name] = r
    add("baseline: all history, equal weight")
    print("fine window sweep...", flush=True)
    for w in (150, 165, 180, 195, 210, 240, 270):
        add(f"window {w}d", ridge_window=w)
    print("window x halflife grid...", flush=True)
    for win in (150, 180, 210, 240):
        for hl in (45, 60, 90, 120, 180):
            add(f"window {win}d + halflife {hl}d", ridge_window=win, ridge_halflife=hl)
    R = pd.DataFrame(rows).T
    R.columns = ["n", "win", "pnl", "roi", f"n>={MID}", "pnl_aug", "roi_aug", f"n>={RECENT}", "win_bad", "pnl_bad",
                 "ridge_bias", "ridge_mae", "fit_rows", "ess"]
    print(f"\n=== recency weighting inside the ridge window ({START}..{END}) ===")
    print(R.round(3).to_string())
    R.to_parquet("out/weight_study.parquet")
    b = rows["baseline: all history, equal weight"]
    print(f"\nbaseline P&L ${b['pnl']:.0f}; bootstrap sd is ~$556, so a variant needs ~+$912 to clear noise.")
    top = R.sort_values("pnl", ascending=False).head(5)
    print("\ntop 5 by P&L:"); print(top[["n", "win", "pnl", "roi", "pnl_aug", "pnl_bad", "ridge_mae", "ess"]].round(3).to_string())


def ewma_study():
    """The EWMA side, mirroring what was done to the ridge.
      A  ewma_window  how many days of error history the recursion and the gain selection see at all
      B  ewma_gain    forced gain for every city -- literally "how much more do recent days count":
                      the newest error gets weight k, and the level's half-life is ln(.5)/ln(1-k) days
      C  flat_window  the ridge's winning shape applied here: equal weight inside a hard window, no decay
    Live behaviour = unlimited history, gain chosen per city by SSE (0.05 for LA and Miami, 0.10 elsewhere)."""
    hl = lambda k: np.log(0.5) / np.log(1 - k)
    rows = {}
    def add(name, **mk):
        F = forecasts(**mk); T = run(F); r = summ(T)
        r["ewma_bias"] = F.res_e.mean(); r["ewma_mae"] = F.res_e.abs().mean(); rows[name] = r
    add("baseline: all history, gain chosen per city")
    print("A: ewma_window...", flush=True)
    for w in (45, 60, 90, 120, 180, 365): add(f"ewma_window {w}d", ewma_window=w)
    print("B: forced gain...", flush=True)
    for k in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50): add(f"gain {k:.2f} (half-life {hl(k):.0f}d)", ewma_gain=k)
    print("C: flat window...", flush=True)
    for w in (5, 10, 14, 21, 30, 60): add(f"flat mean of last {w} errors", flat_window=w)
    print("D: cross...", flush=True)
    for w, k in ((90, 0.05), (90, 0.10), (180, 0.05), (180, 0.10), (180, 0.20)):
        add(f"window {w}d + gain {k:.2f}", ewma_window=w, ewma_gain=k)
    R = pd.DataFrame(rows).T
    R.columns = ["n", "win", "pnl", "roi", f"n>={MID}", "pnl_aug", "roi_aug", f"n>={RECENT}", "win_bad", "pnl_bad",
                 "ewma_bias", "ewma_mae"]
    print(f"\n=== EWMA history length and recency weight ({START}..{END}) ===")
    print(R.round(3).to_string())
    print(f"\nbaseline ${R.iloc[0]['pnl']:.0f}; bootstrap sd ~$556, so ~+$912 is needed to clear noise.")
    print("\ntop 5 by P&L:")
    print(R.sort_values("pnl", ascending=False).head(5)[["n", "win", "pnl", "roi", "pnl_aug", "pnl_bad", "ewma_mae"]].round(3).to_string())
    R.to_parquet("out/ewma_study.parquet")


def oos_study():
    """Out-of-sample check on every lever swept so far.

    The walk-forward is already causal (each night fits only on earlier data), so what is NOT yet honest is the
    HYPERPARAMETER choice: every table so far picked a setting by looking at the same period it was scored on.
    Here each configuration is run once over the whole period and its trades are split: Jan 18 - Jun 30 selects,
    Jul 1 - Sep 22 scores. The question that matters is not which config wins in the selection half, but whether
    winning there predicts anything in the test half -- reported as the rank correlation across all configs."""
    SEL_END = dt.date(2026, 6, 30)
    cfgs = {"baseline (live)": {}}
    for w in (90, 120, 150, 165, 180, 195, 210, 240, 270): cfgs[f"ridge_window {w}d"] = dict(ridge_window=w)
    for h in (30, 60, 90, 120, 180):                        cfgs[f"ridge_halflife {h}d"] = dict(ridge_halflife=h)
    for a in (5, 10, 50, 100):                              cfgs[f"ridge_alpha {a}"] = dict(ridge_alpha=a)
    cfgs["seasonal doy"] = dict(seasonal=True); cfgs["per-city ridge"] = dict(per_city=True)
    for g in (0.05, 0.1, 0.2):                              cfgs[f"ridge_bias EWMA {g}"] = dict(ridge_bias_gain=g)
    for k in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30):          cfgs[f"ewma gain {k}"] = dict(ewma_gain=k)
    for w in (60, 120, 180):                                cfgs[f"ewma_window {w}d"] = dict(ewma_window=w)
    for w in (21, 30):                                      cfgs[f"flat mean {w}"] = dict(flat_window=w)
    for g in (0.10, 0.20):                                  cfgs[f"gain_floor {g}"] = dict(gain_floor=g)
    for w in (90, 120, 180):                                cfgs[f"gain_window {w}d"] = dict(gain_window=w)
    for w in (20, 30, 120):                                 cfgs[f"sd_window {w}"] = dict(sd_window=w)
    cfgs["window 180 + alpha 50"] = dict(ridge_window=180, ridge_alpha=50)
    cfgs["window 180 + gain 0.05"] = dict(ridge_window=180, ewma_gain=0.05)
    rows = {}
    for i, (name, mk) in enumerate(cfgs.items(), 1):
        print(f"[{i}/{len(cfgs)}] {name}", flush=True)
        T = run(forecasts(**mk))
        sel = T[T.mday <= SEL_END]; tst = T[T.mday > SEL_END]
        rows[name] = pd.Series(dict(n_sel=len(sel), pnl_sel=sel.pnl.sum(), roi_sel=sel.pnl.sum() / sel.stake.sum(),
                                    n_test=len(tst), pnl_test=tst.pnl.sum(), roi_test=tst.pnl.sum() / tst.stake.sum(),
                                    win_test=tst.won.mean()))
    R = pd.DataFrame(rows).T
    b = R.loc["baseline (live)"]
    R["sel_vs_base"] = R.pnl_sel - b.pnl_sel; R["test_vs_base"] = R.pnl_test - b.pnl_test
    print(f"\n=== out of sample: choose on Jan 18 - {SEL_END}, score on Jul 1 - {END} ===")
    print(R.sort_values("pnl_sel", ascending=False).round(3).to_string())
    rho = R.pnl_sel.corr(R.pnl_test, method="spearman"); pear = R.pnl_sel.corr(R.pnl_test)
    print(f"\nDoes winning in the selection half predict the test half?")
    print(f"  Spearman rank correlation across {len(R)} configs: {rho:+.3f}   (Pearson {pear:+.3f})")
    top = R.sort_values("pnl_sel", ascending=False).head(5)
    print(f"\nthe 5 configs you would have PICKED on Jan-Jun, and what they then did Jul-Sep:")
    print(top[["pnl_sel", "sel_vs_base", "pnl_test", "test_vs_base", "roi_test"]].round(2).to_string())
    print(f"\nbaseline: selection ${b.pnl_sel:.0f}, test ${b.pnl_test:.0f} on {int(b.n_test)} trades")
    best_sel = R.pnl_sel.idxmax()
    print(f"pick by selection half -> '{best_sel}': test ${R.loc[best_sel,'pnl_test']:.0f} "
          f"({R.loc[best_sel,'test_vs_base']:+.0f} vs baseline)")
    print(f"best possible in the test half (not knowable in advance) -> '{R.pnl_test.idxmax()}': ${R.pnl_test.max():.0f}")
    R.to_parquet("out/oos_study.parquet")


def simple_study():
    """Strip out the choices that were themselves fitted on the backtest, and see what survives out of sample.

    Fitted on the same 8 months the rule was scored on: the per-city EWMA gain (SSE selection), the per-city MODES
    table, which 7 of 11 cities to trade, and the two price bands. Each version below removes one more of those.
    Selection half Jan 18 - Jun 30 is shown only for reference -- these versions have nothing left to select, so
    the Jul 1 - Sep 22 column is the result. Trade counts vary hugely, so ROI is the comparable number."""
    SEL_END = dt.date(2026, 6, 30)
    ALL3 = {"agree", "ridge", "ewma"}
    rows = {}
    def add(name, fk=None, **rk):
        T = run(forecasts(**(fk or {})), **rk)
        sel = T[T.mday <= SEL_END]; tst = T[T.mday > SEL_END]
        rows[name] = pd.Series(dict(
            n_sel=len(sel), roi_sel=sel.pnl.sum() / sel.stake.sum() if len(sel) else np.nan,
            n_test=len(tst), stake_test=tst.stake.sum(), pnl_test=tst.pnl.sum(),
            roi_test=tst.pnl.sum() / tst.stake.sum() if len(tst) else np.nan,
            win_test=tst.won.mean() if len(tst) else np.nan))
    print("A: removing the fitted choices one at a time", flush=True)
    add("A0 live: per-city gain + per-city modes + 7 cities + tuned bands")
    add("A1  + one fixed gain 0.05 everywhere", fk=dict(ewma_gain=0.05))
    add("A2  + one fixed gain 0.10 everywhere", fk=dict(ewma_gain=0.10))
    add("A3  A2, no per-city modes (every city trades all 3 legs)", fk=dict(ewma_gain=0.10), modes_all=ALL3)
    add("A4  A3, every city with markets (11, not the chosen 7)", fk=dict(ewma_gain=0.10, all_cities=True), modes_all=ALL3)
    add("A5  A4, one band 0.05-0.55 for every leg", fk=dict(ewma_gain=0.10, all_cities=True), modes_all=ALL3, band=(0.05, 0.55))
    add("A6  A5 + ridge_window 180", fk=dict(ewma_gain=0.10, all_cities=True, ridge_window=180), modes_all=ALL3, band=(0.05, 0.55))
    print("B: no agreement structure -- one model, one bucket", flush=True)
    for sg in ("ewma", "ridge", "avg"):
        add(f"B  {sg}-only best bucket, all cities, band 0.05-0.55",
            fk=dict(ewma_gain=0.10, all_cities=True), band=(0.05, 0.55), single=sg)
    add("B  avg best bucket + ridge_window 180",
        fk=dict(ewma_gain=0.10, all_cities=True, ridge_window=180), band=(0.05, 0.55), single="avg")
    print("C: null model", flush=True)
    add("C  market favourite, no model at all", fk=dict(all_cities=True), band=(0.05, 0.55), single="fav")
    R = pd.DataFrame(rows).T
    R.columns = ["n_sel", "roi_sel", "n_test", "stake_test", "pnl_test", "roi_test", "win_test"]
    R["per_$100"] = 100 * R.pnl_test / R.stake_test
    print(f"\n=== simplified models, scored out of sample (Jul 1 - {END}) ===")
    print(R.round(3).to_string())
    print("\nroi_test is the number to compare; pnl_test scales with how many trades a version takes.")
    R.to_parquet("out/simple_study.parquet")


def fusion_study():
    """Combine the adaptive models with the external seasonal profile.

    A trailing window can only LEARN a shift once it has begun, so at every season turn it is behind by
    construction -- and the pooled profile says the turn is real: the bias moves -0.70 (Sep) -> +0.23 (Oct) ->
    +0.86 (Nov). The four-year prior knows that in advance. Fusion modes:
      feature       the month's prior handed to the ridge as another column
      offset        both models fitted on the anomaly err - prior, prior added back at prediction
      offset_ridge  only the ridge fitted on the anomaly (the EWMA keeps chasing the raw level)
      offset_ewma   only the EWMA
      delta         shift the ridge by prior(target month) - mean prior over its own fit window: an explicit
                    correction for the season having moved on since the data it was fitted to
      delta_both    the same shift applied to both models
    The prior is built only from 2022-2025, so nothing leaks into the Jul-Sep 2026 scoring window."""
    SEL = dt.date(2026, 6, 30)
    pri = seasonal_prior(exclude_year=2026)
    if not pri: print("no seasonality_pairs.parquet -- run seasonality.py first"); return
    rows = {}
    def add(name, **mk):
        T = run(forecasts(**mk)); s_, t_ = T[T.mday <= SEL], T[T.mday > SEL]
        F = forecasts(**mk)
        rows[name] = pd.Series(dict(n_sel=len(s_), roi_sel=s_.pnl.sum() / s_.stake.sum(),
                                    n_test=len(t_), pnl_test=t_.pnl.sum(), roi_test=t_.pnl.sum() / t_.stake.sum(),
                                    win_test=t_.won.mean(),
                                    ridge_mae=F[F.mday > SEL].res_r.abs().mean(),
                                    ewma_mae=F[F.mday > SEL].res_e.abs().mean()))
        print(f"  {name}", flush=True)
    add("baseline (live, no prior)")
    add("ridge_window 180 (current live)", ridge_window=180)
    for m in ("feature", "offset", "offset_ridge", "offset_ewma", "delta", "delta_both"):
        add(f"prior {m}", prior=pri, prior_mode=m)
    for m in ("feature", "offset", "offset_ridge", "delta", "delta_both"):
        add(f"prior {m} + window 180", prior=pri, prior_mode=m, ridge_window=180)
    R = pd.DataFrame(rows).T
    b = R.loc["ridge_window 180 (current live)"]
    R["vs_live"] = R.pnl_test - b.pnl_test
    print(f"\n=== seasonal prior x adaptive model, scored out of sample (Jul 1 - {END}) ===")
    print(R.sort_values("pnl_test", ascending=False).round(3).to_string())
    print(f"\nvs_live is against ridge_window 180, which is what the bot runs now (${b.pnl_test:.0f}).")
    R.to_parquet("out/fusion_study.parquet")


def grid_study():
    """Does the seasonal prior earn its keep at SHORTER ridge windows?

    The shorter the fit window, the less of the annual cycle the ridge has ever seen, so the more blind it should
    be to a turn and the more an external month-of-year profile should add. A long window sees full cycles and
    should need it least. This crosses the window length with the prior, scored out of sample (Jul 1 - Sep 22),
    and reports the GAIN from adding the prior at each length -- that interaction is the whole question."""
    SEL = dt.date(2026, 6, 30)
    pri = seasonal_prior(exclude_year=2026)
    WINS = [None, 90, 120, 150, 180, 210, 240, 365]
    rows = {}
    def cell(w, mode):
        mk = {} if w is None else dict(ridge_window=w)
        if mode: mk = dict(mk, prior=pri, prior_mode=mode)
        F = forecasts(**mk); T = run(F); t = T[T.mday > SEL]; f = F[F.mday > SEL]
        print(f"  window {w} / prior {mode}", flush=True)
        return dict(n=len(t), pnl=t.pnl.sum(), roi=t.pnl.sum() / t.stake.sum(), win=t.won.mean(),
                    mae=f.res_r.abs().mean(), bias=f.res_r.mean(),
                    sep_bias=f[pd.to_datetime(f.mday).dt.month == 9].res_r.mean())
    base, feat = {}, {}
    for w in WINS:
        base[w] = cell(w, None); feat[w] = cell(w, "feature")
    A = pd.DataFrame(base).T; B = pd.DataFrame(feat).T
    A.index = [f"{w if w else 'all'}d" for w in WINS]; B.index = A.index
    out = pd.DataFrame({
        "pnl_no_prior": A.pnl, "pnl_with_prior": B.pnl, "prior_gain": B.pnl - A.pnl,
        "roi_no_prior": A.roi, "roi_with_prior": B.roi,
        "mae_no_prior": A.mae, "mae_with_prior": B.mae, "mae_gain": A.mae - B.mae,
        "sepbias_no_prior": A.sep_bias, "sepbias_with_prior": B.sep_bias})
    print("\n=== ridge window x seasonal prior, out of sample (Jul 1 - %s) ===" % END)
    print(out.round(3).to_string())
    print("\nprior_gain = P&L added by the prior at that window length; mae_gain = ridge |error| removed.")
    extra = {}
    for w in (90, 120, 180):
        extra[f"delta @ {w}d"] = cell(w, "delta")
    extra["offset_ridge @ 90d"] = cell(90, "offset_ridge")
    E = pd.DataFrame(extra).T
    print("\n=== the stronger injections at short windows (where they should stand the best chance) ===")
    print(E.round(3).to_string())
    out.to_parquet("out/grid_study.parquet")


if __name__ == "__main__":
    if os.environ.get("GRID_ONLY") == "1":
        grid_study(); raise SystemExit
    if os.environ.get("FUSION_ONLY") == "1":
        fusion_study(); raise SystemExit
    if os.environ.get("SIMPLE_ONLY") == "1":
        simple_study(); raise SystemExit
    if os.environ.get("OOS_ONLY") == "1":
        oos_study(); raise SystemExit
    if os.environ.get("EWMA_ONLY") == "1":
        ewma_study(); raise SystemExit
    if os.environ.get("WEIGHT_ONLY") == "1":
        weight_study(); raise SystemExit
    if os.environ.get("DRIFT_ONLY") == "1":
        drift_study(); raise SystemExit
    if os.environ.get("GAIN_ONLY") == "1":
        gain_study(); raise SystemExit
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
