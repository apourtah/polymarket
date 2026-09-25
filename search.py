"""Batched search over inputs, predictors and hyperparameters, with a multi-fold holdout protocol.

Why folds rather than one holdout: model_variants_backtest.py OOS_ONLY=1 established that picking a setting by
backtest P&L does not transfer (rank correlation +0.28 between halves; the in-sample winner then lost $353).  A
long search against a single Jul-Sep window would reproduce exactly that mistake, and that window has already been
inspected many times.  So every config is scored on three DISJOINT periods and judged on how many of them it wins,
not on the total.  The walk-forward itself is causal everywhere (each night fits only on earlier data), so the
folds differ only in which trades are counted.

  F1  2026-03-01..2026-05-31
  F2  2026-06-01..2026-07-31
  F3  2026-08-01..2026-09-22     (the stretch that prompted all this)

A config is promoted only if it beats the live configuration on >= 2 folds AND on the pooled total.  Results are
appended to out/search_leaderboard.csv so batches accumulate without re-running anything.

  python3 search.py BATCH_NAME    # runs the batch defined in BATCHES[BATCH_NAME]
  python3 search.py --top         # print the leaderboard
"""
import os, sys, json, warnings, datetime as dt
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
warnings.filterwarnings("ignore"); pd.set_option("display.width", 260)

import model_variants_backtest as MV          # reuse the panel, prices, trading rule and bucket maths
from bot import evening_config as C

LB = "out/search_leaderboard.csv"
FOLDS = [("F1", dt.date(2026, 3, 1), dt.date(2026, 5, 31)),
         ("F2", dt.date(2026, 6, 1), dt.date(2026, 7, 31)),
         ("F3", dt.date(2026, 8, 1), dt.date(2026, 9, 22))]
BASE_FEATS = list(C.FEATS)


# ---------------------------------------------------------------- candidate inputs
def augment(P):
    """Every derived feature the search may draw on. Computed once; configs pick subsets by name."""
    P = P.sort_values(["city", "mday"]).copy()
    g = P.groupby("city")
    P["hrrr_frac"]   = P.hrrr - np.floor(P.hrrr / 2) * 2          # where in the 2F bucket grid the forecast sits
    P["hrrr_delta"]  = P.hrrr - P.yday_max                         # forecast vs yesterday's actual
    P["diurnal"]     = P.yday_max - P.yday_min
    P["mos_mean_mh"] = (P.nbm_max + P.gfsmos_max) / 2 - P.hrrr
    P["mos_spread"]  = (P.nbm_max - P.gfsmos_max).abs()            # model disagreement as an uncertainty proxy
    P["r30"]         = g.err.transform(lambda s: s.shift(1).rolling(30, min_periods=10).mean())
    P["esd5"]        = g.err.transform(lambda s: s.shift(1).rolling(5, min_periods=3).std())
    P["esd14"]       = g.err.transform(lambda s: s.shift(1).rolling(14, min_periods=5).std())
    P["e3"]          = g.err.shift(3)
    P["rad_cloud"]   = P.rad / (1 + P.cloud)
    P["dpd_morn"]    = P.t_morn - P.dew
    P["precip_flag"] = (P.precip > 0).astype(float)
    P["wind_dpd"]    = P.wind * P.dpd
    P["doy_s"]       = np.sin(2 * np.pi * P.doy / 365.25)
    P["doy_c"]       = np.cos(2 * np.pi * P.doy / 365.25)
    P["hrrr_anom"]   = P.hrrr - g.hrrr.transform(lambda s: s.shift(1).rolling(14, min_periods=5).mean())
    P["nbm_minus_gfs"] = P.nbm_max - P.gfsmos_max                  # signed model disagreement
    P["nbm_lag"]     = g.nbm_minus_hrrr.shift(1)                   # was the MOS-HRRR gap persistent?
    P["gfs_lag"]     = g.gfs_minus_hrrr.shift(1)
    P["mos_mh_r5"]   = g.nbm_minus_hrrr.transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    P["mos_err_x"]   = P.nbm_minus_hrrr * P.gfs_minus_hrrr         # do the two MOS models agree in direction?
    return P

EXTRA = ["hrrr_frac", "hrrr_delta", "diurnal", "mos_mean_mh", "mos_spread", "r30", "esd5", "esd14", "e3",
         "rad_cloud", "dpd_morn", "precip_flag", "wind_dpd", "doy_s", "doy_c", "hrrr_anom"]

FEATSETS = {
    "live":     BASE_FEATS,
    "minimal":  ["hrrr", "yday_err", "r5", "nbm_minus_hrrr", "gfs_minus_hrrr", "doy"],
    "mos_only": ["hrrr", "nbm_minus_hrrr", "gfs_minus_hrrr", "mos_mean_mh", "mos_spread"],
    "lags_only": ["hrrr", "yday_err", "e2", "e3", "r5", "r14", "r30", "yday_max", "yday_min"],
    "wx_only":  ["hrrr", "dew", "rhum", "wind", "cloud", "rad", "pres", "precip", "dpd", "t_morn", "cloud_morn"],
    "no_wx":    [f for f in BASE_FEATS if f not in ("dew", "rhum", "wind", "cloud", "rad", "pres", "precip",
                                                    "wdir_s", "wdir_c", "dpd", "t_morn", "cloud_morn")],
    "no_lags":  [f for f in BASE_FEATS if f not in ("yday_err", "e2", "r5", "r14")],
    "no_mos":   [f for f in BASE_FEATS if f not in ("nbm_minus_hrrr", "gfs_minus_hrrr")],
    "no_doy":   [f for f in BASE_FEATS if f != "doy"],
}


# ---------------------------------------------------------------- the model
class Mdl:
    def __init__(self, hist, feats, est="ridge", alpha=None, ridge_window=180, ewma_gain=None,
                 sd_window=60, sd_scale=1.0, est_kw=None):
        self.feats = feats; self.est = est
        h = hist.dropna(subset=["err"]); self.cities = sorted(h.city.unique())
        self.ewma = {}; self.ewma_sd = {}; self.ridge_sd = {}
        for c in self.cities:
            e = h[h.city == c].sort_values("mday").err.values
            gains = [ewma_gain] if ewma_gain else C.EWMA_GAINS
            best = None
            for k in gains:
                b = 0.0; se = []
                for x in e: se.append(x - b); b += k * (x - b)
                s = float(np.sum(np.square(se)))
                if best is None or s < best[0]: best = (s, k, b, np.std(se[-sd_window:]))
            self.ewma[c] = best[2]; self.ewma_sd[c] = max(best[3], 1.0) * sd_scale
        R = h
        if ridge_window:
            W = h[h.mday > max(h.mday) - dt.timedelta(days=ridge_window)]
            if len(W) >= 200: R = W
        X = R[feats].copy(); self.med = X.median().fillna(0); X = X.fillna(self.med)
        self.mu = X.mean(); self.sd = X.std().replace(0, 1)
        Xn = ((X - self.mu) / self.sd).values
        D = pd.get_dummies(R.city).reindex(columns=self.cities, fill_value=0).values.astype(float)
        Z = np.c_[Xn, D]
        if est == "ridge":   m = Ridge(alpha=C.RIDGE_ALPHA if alpha is None else alpha)
        elif est == "huber": m = HuberRegressor(alpha=0.001 if alpha is None else alpha, max_iter=500)
        elif est == "gbm":   m = HistGradientBoostingRegressor(**(est_kw or dict(max_depth=3, max_iter=150, learning_rate=0.06, l2_regularization=1.0)))
        elif est == "rf":    m = RandomForestRegressor(**(est_kw or dict(n_estimators=120, min_samples_leaf=20, n_jobs=2, random_state=0)))
        else: raise ValueError(est)
        self.m = m.fit(Z, R.err.values)
        res = R.err.values - self.m.predict(Z)
        for c in self.cities:
            k = (R.city == c).values
            self.ridge_sd[c] = max(float(np.std(res[k][-sd_window:])) if k.any() else 1.0, 1.0) * sd_scale

    def predict(self, city, row):
        x = pd.Series({f: row.get(f, np.nan) for f in self.feats}).fillna(self.med)
        z = np.r_[((x - self.mu) / self.sd).values, [1.0 if c == city else 0.0 for c in self.cities]]
        return self.ewma[city], float(self.m.predict(z[None, :])[0]), self.ewma_sd[city], self.ridge_sd[city]


def walk(P, cfg):
    """Nightly refit walk-forward; returns the mu/sd frame run() expects."""
    feats = FEATSETS[cfg.get("featset", "live")] + list(cfg.get("add", []))
    feats = [f for f in feats if f in P.columns]
    mk = {k: v for k, v in cfg.items() if k in ("est", "alpha", "ridge_window", "ewma_gain", "sd_window", "sd_scale", "est_kw")}
    rows = []
    days = sorted(x for x in P.mday.unique() if dt.date(2026, 1, 18) <= x <= dt.date(2026, 9, 22))
    for d in days:
        H = P[P.mday < d]
        if len(H) < C.MIN_HISTORY_DAYS: continue
        M = Mdl(H, feats, **mk)
        for c in MV.TRADE:
            r = P[(P.city == c) & (P.mday == d)]
            if r.empty: continue
            r = r.iloc[0]; e, rg, es, rs = M.predict(c, r.to_dict())
            rows.append(dict(city=c, mday=d, actual=r.actual, mu_e=r.hrrr + e, mu_r=r.hrrr + rg, sd_e=es, sd_r=rs))
    F = pd.DataFrame(rows); F["res_e"] = F.actual - F.mu_e; F["res_r"] = F.actual - F.mu_r
    return F


def run2(F, agree_tol=None, third=None, blend=None, sd_spread=0.0, blend_sd="avg",
         sd_pick=None, sd_pick_dis=None, P=None):
    """The trading rule, with the agreement STRUCTURE opened up.

    Live: buy when the two models' argmax bucket is identical ("agree"), else follow the configured model.
      agree_tol   call it agreement when |mu_e - mu_r| <= tol degrees instead of requiring the same bucket
      third       'nbm': a third forecaster (raw NBM). 'all3' = every model must land on the same bucket;
                  '2of3' = any two of the three, and that shared bucket is what is bought
      blend       buy the bucket of w*mu_e + (1-w)*mu_r on agreement nights instead of the shared bucket
      sd_spread   widen sd by this * the NBM/GFS disagreement for the night (uncertainty we can actually observe)
    """
    trades = []
    for r in F.itertuples():
        pr = MV.prices_for(r.city, r.mday)
        if len(pr) < 3: continue
        bk = list(pr)
        se, sr = r.sd_e, r.sd_r
        if sd_spread and not pd.isna(getattr(r, "mos_spread", np.nan)):
            f = 1.0 + sd_spread * float(r.mos_spread); se, sr = se * f, sr * f
        # sd_pick: the sd used to CHOOSE the bucket, as a weight on the EWMA's own sd.
        #   1.0 = live (each model uses its own), 0.5 = the average, 0.0 = both pick with the ridge's sd.
        #   sd_e runs ~2.5 and sd_r ~1.6, so the live rule picks from a much wider distribution for the EWMA.
        pe_s, pr_s = se, sr
        if sd_pick is not None: pe_s = pr_s = sd_pick * se + (1 - sd_pick) * sr
        Pe = MV.bucket_probs(r.mu_e, pe_s, bk); Pr = MV.bucket_probs(r.mu_r, pr_s, bk)
        be, br = max(Pe, key=Pe.get), max(Pr, key=Pr.get)
        if sd_pick_dis is not None:
            d = sd_pick_dis * se + (1 - sd_pick_dis) * sr
            be = max(MV.bucket_probs(r.mu_e, d, bk).items(), key=lambda kv: kv[1])[0]
            br = max(MV.bucket_probs(r.mu_r, d, bk).items(), key=lambda kv: kv[1])[0]
        agree = (abs(r.mu_e - r.mu_r) <= agree_tol) if agree_tol else (be == br)
        pick_agree = be
        if third and not pd.isna(getattr(r, "mu_n", np.nan)):
            Pn = MV.bucket_probs(r.mu_n, (se + sr) / 2, bk); bn = max(Pn, key=Pn.get)
            if third == "all3": agree = agree and (bn == be)
            elif third == "2of3":
                votes = {}
                for b in (be, br, bn): votes[b] = votes.get(b, 0) + 1
                top, nv = max(votes.items(), key=lambda kv: kv[1])
                agree = nv >= 2; pick_agree = top
        if blend is not None and agree:
            # the blend moves TWO things: the mu used for the pick, and the sd. blend_sd separates them:
            #   "avg"  (se+sr)/2      "mix"  blend*se+(1-blend)*sr      "e"  se (isolates the mu mix alone)
            if isinstance(blend_sd, (int, float)):                       # explicit weight on the EWMA's own sd
                sb = blend_sd * se + (1 - blend_sd) * sr                   #   1.0 -> live behaviour, 0.5 -> the average
            else:
                sb = (se + sr) / 2 if blend_sd == "avg" else (blend * se + (1 - blend) * sr if blend_sd == "mix" else se)
            Pb = MV.bucket_probs(blend * r.mu_e + (1 - blend) * r.mu_r, sb, bk)
            pick_agree = max(Pb, key=Pb.get)
        modes = C.MODES.get(r.city, set()); spent = 0.0
        for b in bk:
            ask, won = pr[b]; why = None
            if agree and "agree" in modes and b == pick_agree and C.AGREE_MIN_PRICE <= ask <= C.AGREE_MAX_PRICE: why = "agree"
            elif not agree and "ridge" in modes and b == br and C.MODEL_MIN_PRICE <= ask <= C.MODEL_MAX_PRICE: why = "dis_ridge"
            elif not agree and "ewma" in modes and b == be and C.MODEL_MIN_PRICE <= ask <= C.MODEL_MAX_PRICE: why = "dis_ewma"
            if not why: continue
            stake = min(C.STAKE, C.MAX_PER_MARKET_USD, C.MAX_PER_CITY_DAY_USD - spent)
            if stake < C.MIN_ORDER_SHARES * ask: continue
            spent += stake; sh = stake / ask; fee = 0.05 * ask * (1 - ask) * sh
            trades.append(dict(city=r.city, mday=r.mday, why=why, price=ask, stake=stake, won=won,
                               pnl=(sh - stake if won else -stake) - fee))
    return pd.DataFrame(trades)


def score(name, cfg, P):
    rk = {k: v for k, v in cfg.items() if k in ("agree_tol", "third", "blend", "sd_spread", "blend_sd", "sd_pick", "sd_pick_dis")}
    F = walk(P, cfg)
    if rk:
        ex = P[["city", "mday", "nbm_max", "mos_spread"]].rename(columns={"nbm_max": "mu_n"})
        F = F.merge(ex, on=["city", "mday"], how="left")
        T = run2(F, **rk)
    else:
        T = MV.run(F)
    T["mday"] = pd.to_datetime(T.mday).dt.date
    rec = dict(name=name, cfg=json.dumps(cfg, default=str), n=len(T), pnl=T.pnl.sum(),
               roi=T.pnl.sum() / T.stake.sum() if len(T) else np.nan,
               mae_r=F.res_r.abs().mean(), mae_e=F.res_e.abs().mean())
    for fn, a, b in FOLDS:
        g = T[(T.mday >= a) & (T.mday <= b)]
        rec[f"{fn}_n"] = len(g); rec[f"{fn}_pnl"] = g.pnl.sum()
        rec[f"{fn}_roi"] = g.pnl.sum() / g.stake.sum() if len(g) else np.nan
    return rec


BATCHES = {
    # batch 7: isolate ONLY the sd used to pick the bucket on agreement nights (the agreement test itself is
    # untouched). blend=1.0 keeps mu_e as the pick, so blend_sd=1.0 must reproduce LIVE exactly.
    "b7": {
        "LIVE":                                  dict(),
        "pick_sd w=1.00 (must equal LIVE)":      dict(blend=1.0, blend_sd=1.00),
        "pick_sd w=0.75":                        dict(blend=1.0, blend_sd=0.75),
        "pick_sd w=0.50 (average)":              dict(blend=1.0, blend_sd=0.50),
        "pick_sd w=0.25":                        dict(blend=1.0, blend_sd=0.25),
        "pick_sd w=0.00 (ridge sd)":             dict(blend=1.0, blend_sd=0.00),
        "pick_sd 0.50 + w195":                   dict(blend=1.0, blend_sd=0.50, ridge_window=195),
        "pick_sd 0.25 + w195":                   dict(blend=1.0, blend_sd=0.25, ridge_window=195),
        "pick_sd 0.00 + w195":                   dict(blend=1.0, blend_sd=0.00, ridge_window=195),
        "pick_sd 0.50 + w210":                   dict(blend=1.0, blend_sd=0.50, ridge_window=210),
        "pick_sd 0.25 + w210":                   dict(blend=1.0, blend_sd=0.25, ridge_window=210),
        "b0.8 pick_sd 0.25 + w195":              dict(blend=0.8, blend_sd=0.25, ridge_window=195),
        "b0.8 pick_sd 0.00 + w195":              dict(blend=0.8, blend_sd=0.00, ridge_window=195),
        "b0.8 pick_sd 0.50 + w195 (=leader)":    dict(blend=0.8, blend_sd=0.50, ridge_window=195),
        "leader + mos_mean_mh":                  dict(blend=0.8, blend_sd=0.50, ridge_window=195, add=["mos_mean_mh"]),
        "leader w200":                           dict(blend=0.8, blend_sd=0.50, ridge_window=200),
        "leader w190":                           dict(blend=0.8, blend_sd=0.50, ridge_window=190),
        "leader w215":                           dict(blend=0.8, blend_sd=0.50, ridge_window=215),
        "leader + sd_window 90":                 dict(blend=0.8, blend_sd=0.50, ridge_window=195, sd_window=90),
        "leader + alpha 40":                     dict(blend=0.8, blend_sd=0.50, ridge_window=195, alpha=40),
    },
    # batch 6: the effect is the sd used to PICK the bucket. Map that weight directly (1.0 = live, 0.5 = average),
    # test it on the disagree legs too, and probe around the two configs that won all three folds.
    "b6": {
        "LIVE":                                  dict(),
        "sd_pick 1.00 (= LIVE check)":           dict(sd_pick=1.00),
        "sd_pick 0.75":                          dict(sd_pick=0.75),
        "sd_pick 0.50 (average)":                dict(sd_pick=0.50),
        "sd_pick 0.25":                          dict(sd_pick=0.25),
        "sd_pick 0.00 (ridge sd for both)":      dict(sd_pick=0.00),
        "sd_pick 0.50 + w195":                   dict(sd_pick=0.50, ridge_window=195),
        "sd_pick 0.50 + w210":                   dict(sd_pick=0.50, ridge_window=210),
        "sd_pick 0.25 + w195":                   dict(sd_pick=0.25, ridge_window=195),
        "sd_pick 0.00 + w195":                   dict(sd_pick=0.00, ridge_window=195),
        "sd_pick_dis 0.50 (disagree legs too)":  dict(sd_pick_dis=0.50),
        "sd_pick 0.5 + sd_pick_dis 0.5":         dict(sd_pick=0.50, sd_pick_dis=0.50),
        "b0.8+w195+mos_mean_mh (3-fold leader)": dict(blend=0.8, ridge_window=195, add=["mos_mean_mh"]),
        "b0.8+w210 (3-fold leader)":             dict(blend=0.8, ridge_window=210),
        "b0.8+w210+mos_mean_mh":                 dict(blend=0.8, ridge_window=210, add=["mos_mean_mh"]),
        "b0.8+w195+mos_mean_mh+gain0.05":        dict(blend=0.8, ridge_window=195, add=["mos_mean_mh"], ewma_gain=0.05),
        "b0.8+w180+mos_mean_mh":                 dict(blend=0.8, ridge_window=180, add=["mos_mean_mh"]),
        "b0.85+w210":                            dict(blend=0.85, ridge_window=210),
        "b0.8+w205":                             dict(blend=0.8, ridge_window=205),
        "b0.8+w220":                             dict(blend=0.8, ridge_window=220),
    },
    # batch 5: the blend changes the pick's mu AND its sd. Separate them, then probe around the best cell.
    "b5": {
        "LIVE":                                  dict(),
        "blend 1.0 sd=e (should equal LIVE)":    dict(blend=1.0, blend_sd="e"),
        "blend 0.8 sd=e (mu mix only)":          dict(blend=0.8, blend_sd="e"),
        "blend 0.5 sd=e (mu mix only)":          dict(blend=0.5, blend_sd="e"),
        "blend 1.0 sd=avg (sd effect only)":     dict(blend=1.0, blend_sd="avg"),
        "blend 0.8 sd=mix":                      dict(blend=0.8, blend_sd="mix"),
        "blend 0.8 sd=avg (batch-4 leader)":     dict(blend=0.8, blend_sd="avg"),
        "blend 0.8 + w180":                      dict(blend=0.8, ridge_window=180),
        "blend 0.8 + w195":                      dict(blend=0.8, ridge_window=195),
        "blend 0.8 + w210":                      dict(blend=0.8, ridge_window=210),
        "blend 0.8 + sd_window 90":              dict(blend=0.8, sd_window=90),
        "blend 0.8 + w195 + sd_window 90":       dict(blend=0.8, ridge_window=195, sd_window=90),
        "blend 0.85 + w195":                     dict(blend=0.85, ridge_window=195),
        "blend 0.75 + w195":                     dict(blend=0.75, ridge_window=195),
        "blend 0.9 + w195":                      dict(blend=0.9, ridge_window=195),
        "blend 0.8 + w195 + gain 0.05":          dict(blend=0.8, ridge_window=195, ewma_gain=0.05),
        "blend 0.8 + w195 + alpha 5":            dict(blend=0.8, ridge_window=195, alpha=5),
        "blend 0.8 + w195 + add mos_mean_mh":    dict(blend=0.8, ridge_window=195, add=["mos_mean_mh"]),
        "blend 0.8 + w195 + add hrrr_anom":      dict(blend=0.8, ridge_window=195, add=["hrrr_anom"]),
        "sd_window 90 + w195":                   dict(sd_window=90, ridge_window=195),
    },
    # batch 4: are the two leaders real? A real effect has a smooth response; a spike at one setting is noise.
    "b4": {
        "LIVE":                                  dict(),
        "blend 0.60":                            dict(blend=0.60),
        "blend 0.65":                            dict(blend=0.65),
        "blend 0.70":                            dict(blend=0.70),
        "blend 0.75":                            dict(blend=0.75),
        "blend 0.80":                            dict(blend=0.80),
        "blend 0.90":                            dict(blend=0.90),
        "blend 1.00 (pure ewma pick)":           dict(blend=1.00),
        "w180 + gain 0.05":                      dict(ridge_window=180, ewma_gain=0.05),
        "w195 + gain 0.05":                      dict(ridge_window=195, ewma_gain=0.05),
        "w210 + gain 0.05":                      dict(ridge_window=210, ewma_gain=0.05),
        "w165 + gain 0.05":                      dict(ridge_window=165, ewma_gain=0.05),
        "w195 + gain 0.075":                     dict(ridge_window=195, ewma_gain=0.075),
        "w195 + gain 0.10":                      dict(ridge_window=195, ewma_gain=0.10),
        "blend 0.7 + w195":                      dict(blend=0.7, ridge_window=195),
        "blend 0.7 + gain 0.05":                 dict(blend=0.7, ewma_gain=0.05),
        "blend 0.7 + w195 + gain 0.05":          dict(blend=0.7, ridge_window=195, ewma_gain=0.05),
        "sd_window 30":                          dict(sd_window=30),
        "sd_window 45":                          dict(sd_window=45),
        "sd_window 90":                          dict(sd_window=90),
    },
    # batch 3: the knobs are exhausted; open up the agreement STRUCTURE itself
    "b3": {
        "LIVE":                             dict(),
        "agree if |mu_e-mu_r| <= 0.5":      dict(agree_tol=0.5),
        "agree if |mu_e-mu_r| <= 1.0":      dict(agree_tol=1.0),
        "agree if |mu_e-mu_r| <= 1.5":      dict(agree_tol=1.5),
        "agree if |mu_e-mu_r| <= 2.0":      dict(agree_tol=2.0),
        "third NBM: all 3 must agree":      dict(third="all3"),
        "third NBM: 2 of 3 agree":          dict(third="2of3"),
        "2of3 + tol 1.0":                   dict(third="2of3", agree_tol=1.0),
        "blend 0.5 on agree nights":        dict(blend=0.5),
        "blend 0.3 (ridge-weighted)":       dict(blend=0.3),
        "blend 0.7 (ewma-weighted)":        dict(blend=0.7),
        "sd widened by mos_spread x0.10":   dict(sd_spread=0.10),
        "sd widened by mos_spread x0.25":   dict(sd_spread=0.25),
        "tol 1.0 + sd_spread 0.10":         dict(agree_tol=1.0, sd_spread=0.10),
        "window 195 + tol 1.0":             dict(ridge_window=195, agree_tol=1.0),
        "window 195 + 2of3":                dict(ridge_window=195, third="2of3"),
        "window 195 + alpha 5":             dict(ridge_window=195, alpha=5),
        "alpha 5 + tol 1.0":                dict(alpha=5, agree_tol=1.0),
        "alpha 5 + ewma_gain 0.05":         dict(alpha=5, ewma_gain=0.05),
        "window 195 + ewma_gain 0.05":      dict(ridge_window=195, ewma_gain=0.05),
    },
    # batch 2: MOS is the dominant block and additions dilute -> push on MOS, and sweep the two hyperparameters
    # that were never tested against this feature count (alpha) or at all (sd_scale).
    "b2": {
        "LIVE":                                     dict(),
        "alpha 2":                                  dict(alpha=2),
        "alpha 5":                                  dict(alpha=5),
        "alpha 10":                                 dict(alpha=10),
        "alpha 40":                                 dict(alpha=40),
        "alpha 80":                                 dict(alpha=80),
        "sd_scale 0.80":                            dict(sd_scale=0.80),
        "sd_scale 0.90":                            dict(sd_scale=0.90),
        "sd_scale 1.10":                            dict(sd_scale=1.10),
        "sd_scale 1.25":                            dict(sd_scale=1.25),
        "add nbm_minus_gfs":                        dict(add=["nbm_minus_gfs"]),
        "add nbm_lag+gfs_lag":                      dict(add=["nbm_lag", "gfs_lag"]),
        "add mos_mh_r5":                            dict(add=["mos_mh_r5"]),
        "add mos_err_x":                            dict(add=["mos_err_x"]),
        "add nbm_max+gfsmos_max (raw levels)":      dict(add=["nbm_max", "gfsmos_max"]),
        "add mos_mean_mh + hrrr_anom":              dict(add=["mos_mean_mh", "hrrr_anom"]),
        "add hrrr_frac + hrrr_anom":                dict(add=["hrrr_frac", "hrrr_anom"]),
        "window 165":                               dict(ridge_window=165),
        "window 195":                               dict(ridge_window=195),
        "ewma_gain 0.05":                           dict(ewma_gain=0.05),
    },
    # batch 1: what do the existing inputs actually contribute, and do the obvious new ones help?
    "b1": {
        "LIVE  (ridge_window 180, live feats)":      dict(),
        "feats minimal":                            dict(featset="minimal"),
        "feats mos_only":                           dict(featset="mos_only"),
        "feats lags_only":                          dict(featset="lags_only"),
        "feats wx_only":                            dict(featset="wx_only"),
        "feats no_wx":                              dict(featset="no_wx"),
        "feats no_lags":                            dict(featset="no_lags"),
        "feats no_mos":                             dict(featset="no_mos"),
        "feats no_doy":                             dict(featset="no_doy"),
        "add hrrr_frac":                            dict(add=["hrrr_frac"]),
        "add mos_spread":                           dict(add=["mos_spread"]),
        "add mos_mean_mh":                          dict(add=["mos_mean_mh"]),
        "add hrrr_delta":                           dict(add=["hrrr_delta"]),
        "add r30":                                  dict(add=["r30"]),
        "add esd5+esd14":                           dict(add=["esd5", "esd14"]),
        "add hrrr_anom":                            dict(add=["hrrr_anom"]),
        "add rad_cloud+dpd_morn":                   dict(add=["rad_cloud", "dpd_morn"]),
        "add all extras":                           dict(add=EXTRA),
        "est huber":                                dict(est="huber"),
        "est gbm":                                  dict(est="gbm"),
    },
}


if __name__ == "__main__":
    if "--top" in sys.argv:
        L = pd.read_csv(LB)
        live = L[L.name.str.startswith("LIVE")].iloc[-1]
        L["folds_won"] = sum((L[f"{f}_pnl"] > live[f"{f}_pnl"]).astype(int) for f, _, _ in FOLDS)
        L["vs_live"] = L.pnl - live.pnl
        cols = ["name", "n", "pnl", "roi", "vs_live", "folds_won", "F1_pnl", "F2_pnl", "F3_pnl", "mae_r"]
        print(L.sort_values("pnl", ascending=False)[cols].head(25).round(3).to_string(index=False))
        sys.exit()
    batch = sys.argv[1]
    P = augment(MV.P)
    recs = []
    for i, (name, cfg) in enumerate(BATCHES[batch].items(), 1):
        t0 = dt.datetime.now()
        try:
            r = score(name, cfg, P); r["batch"] = batch
            recs.append(r)
            print(f"[{i}/{len(BATCHES[batch])}] {name:<40} pnl {r['pnl']:8.0f} roi {r['roi']:6.3f} "
                  f"F1 {r['F1_pnl']:7.0f} F2 {r['F2_pnl']:7.0f} F3 {r['F3_pnl']:7.0f}  "
                  f"({(dt.datetime.now()-t0).seconds}s)", flush=True)
        except Exception as ex:
            print(f"[{i}] {name}: FAILED {ex.__class__.__name__}: {ex}", flush=True)
    D = pd.DataFrame(recs)
    if os.path.exists(LB): D = pd.concat([pd.read_csv(LB), D], ignore_index=True)
    D.to_csv(LB, index=False)
    print(f"\nleaderboard -> {LB} ({len(D)} rows)")
