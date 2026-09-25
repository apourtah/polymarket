"""Which city-leg combinations are actually carrying the strategy, and would dropping the weak ones have helped?

Everything here is a PAIRED comparison: one walk-forward, then subsets of the same trades. For a pure removal the
counterfactual is exactly "these trades do not happen", so the paired bootstrap is the correct test and its band is
legitimately tight -- unlike adding or swapping legs, where the new trades bring their own variance and every
effect in search.py's batches 12-14 was swamped by it.

Four views:
  1  ROI per city-leg cell
  2  paired bootstrap of removing each cell, and the best combinations
  3  expanding-window validation: pick the cells to drop on data before a cutoff, score after it
  4  slippage: the backtest prices entry at the mid, the live bot pays the ask

  python3 leg_audit.py
"""
import itertools, datetime as dt
import numpy as np, pandas as pd
import search as S, model_variants_backtest as MV
from bot import evening_config as C

FOLDS = [("F1", dt.date(2026, 3, 1), dt.date(2026, 5, 31)),
         ("F2", dt.date(2026, 6, 1), dt.date(2026, 7, 31)),
         ("F3", dt.date(2026, 8, 1), dt.date(2026, 9, 22))]
WEAK = {"Los Angeles/agree", "Austin/agree", "Seattle/agree"}


def load():
    P = S.augment(MV.P); T = MV.run(S.walk(P, dict()))
    T["mday"] = pd.to_datetime(T.mday).dt.date; T["cell"] = T.city + "/" + T.why
    return T


def paired(T, mask, rng, n=4000):
    """Bootstrap the ROI difference from removing the masked trades, resampling days."""
    days = sorted(T.mday.unique()); K = T[~mask]
    A = T.groupby("mday").agg(p=("pnl", "sum"), s=("stake", "sum")).reindex(days).fillna(0)
    B = K.groupby("mday").agg(p=("pnl", "sum"), s=("stake", "sum")).reindex(days).fillna(0)
    d = np.array([B.p.values[i].sum() / max(B.s.values[i].sum(), 1) - A.p.values[i].sum() / max(A.s.values[i].sum(), 1)
                  for i in (rng.integers(0, len(days), len(days)) for _ in range(n))])
    folds = sum(1 for f, a, b in FOLDS
                if (K[(K.mday >= a) & (K.mday <= b)].pnl.sum() / max(K[(K.mday >= a) & (K.mday <= b)].stake.sum(), 1))
                > (T[(T.mday >= a) & (T.mday <= b)].pnl.sum() / max(T[(T.mday >= a) & (T.mday <= b)].stake.sum(), 1)))
    return dict(n=len(K), removed=int(mask.sum()), pnl=round(K.pnl.sum()), dpnl=round(K.pnl.sum() - T.pnl.sum()),
                roi=round(K.pnl.sum() / K.stake.sum(), 3), P=round((d > 0).mean(), 3),
                lo=round(np.percentile(d, 5), 3), roi_folds=folds)


def reprice(T, slip):
    px = (T.price + slip).clip(upper=0.99); sh = T.stake / px
    return T.assign(pnl=np.where(T.won, sh - T.stake, -T.stake) - 0.05 * px * (1 - px) * sh)


if __name__ == "__main__":
    T = load(); rng = np.random.default_rng(23)
    print(f"LIVE: n {len(T)}  pnl ${T.pnl.sum():.0f}  roi {T.pnl.sum()/T.stake.sum():.3f}\n")

    print("=== 1. ROI per city-leg cell ===")
    print(T.groupby("cell").apply(lambda g: pd.Series(dict(n=len(g), win=g.won.mean(), avg_px=g.price.mean(),
          pnl=g.pnl.sum(), roi=g.pnl.sum() / g.stake.sum()))).round(3).sort_values("roi").to_string())

    print("\n=== 2. removing one cell (paired bootstrap on ROI) ===")
    cells = [c for c, g in T.groupby("cell") if len(g) >= 20]
    print(pd.DataFrame({c: paired(T, T.cell == c, rng) for c in cells}).T.sort_values("P", ascending=False).to_string())
    print("\nremoving the three stably-weak agree cells together:")
    print(paired(T, T.cell.isin(WEAK), rng))

    print("\n=== 3. expanding-window validation ===")
    print("choose cells on data BEFORE the cutoff (ROI < 0.25 with >= 25 trades), score AFTER it")
    for cut in (dt.date(2026, 4, 30), dt.date(2026, 5, 31), dt.date(2026, 6, 30), dt.date(2026, 7, 31)):
        tr, te = T[T.mday <= cut], T[T.mday > cut]
        if len(te) < 40: continue
        roi = tr.groupby("cell").apply(lambda x: x.pnl.sum() / x.stake.sum()); nn = tr.groupby("cell").size()
        weak = sorted(roi[(roi < 0.25) & (nn >= 25)].index); K = te[~te.cell.isin(weak)]
        if not len(K): continue
        a, b = te.pnl.sum() / te.stake.sum(), K.pnl.sum() / K.stake.sum()
        print(f"  {cut}: drop {weak}\n      after -> live roi {a:.3f} (${te.pnl.sum():.0f})  filtered roi {b:.3f} "
              f"(${K.pnl.sum():.0f})  {'BETTER' if b > a else 'worse'}")
    print("\n  per-cell ROI as data accumulates (stable cells are the ones worth acting on):")
    acc = {}
    for cut in (dt.date(2026, 4, 30), dt.date(2026, 5, 31), dt.date(2026, 6, 30), dt.date(2026, 7, 31), dt.date(2026, 9, 22)):
        tr = T[T.mday <= cut]
        acc[str(cut)] = {c: round(g.pnl.sum() / g.stake.sum(), 2) for c, g in tr.groupby("cell") if len(g) >= 20}
    print(pd.DataFrame(acc).to_string())

    print("\n=== 4. slippage (the backtest uses the mid; the bot pays the ask) ===")
    out = {}
    for slip in (0.0, 0.005, 0.01, 0.015, 0.02):
        R = reprice(T, slip)
        out[f"+{slip:.3f}"] = {c: round(g.pnl.sum() / g.stake.sum(), 3) for c, g in R.groupby("cell")}
        out[f"+{slip:.3f}"]["ALL"] = round(R.pnl.sum() / R.stake.sum(), 3)
    print(pd.DataFrame(out).to_string())
    print("\n  dropping the three weak agree cells, at each slippage:")
    for slip in (0.0, 0.005, 0.01, 0.015, 0.02):
        R = reprice(T, slip); K = R[~R.cell.isin(WEAK)]
        print(f"    +{slip:.3f}: live roi {R.pnl.sum()/R.stake.sum():.3f} (${R.pnl.sum():.0f})  ->  "
              f"filtered roi {K.pnl.sum()/K.stake.sum():.3f} (${K.pnl.sum():.0f})  dpnl {K.pnl.sum()-R.pnl.sum():+.0f}")
