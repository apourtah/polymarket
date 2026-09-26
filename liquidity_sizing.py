"""Sizing against MEASURED liquidity, not assumed liquidity.

Every earlier sizing number came from a backtest that fills any size at the midpoint. This prices the two things
that actually constrain us, both measured rather than guessed:

  spread  81 live order books sampled across the seven cities: in our trading range (mid 0.05-0.55) the bid-ask
          is 2.0c at the median, 2.83c on average, 4.0c at the 75th percentile. So the old H=0.01 assumption was
          right at the median and mildly optimistic on average.

  depth   from the same books: 15 shares at the touch, 124 within 1c, 218 within 2c (medians). Separately, the
          executed tapes (data/tapes_w) say the median market-night sees just 2 trades and 32 shares in the
          21:00-22:00 local hour we trade in -- so the resting depth is real but rarely hit, and an order of ours
          would be the dominant print of the hour.

Orders walk the book: shares up to d0 pay the ask, the next tranche to d1 pays ask+1c, the next to d2 pays
ask+2c, and anything past d2 is treated as unfillable. The maker leg is kept as the bot has it -- full size, no
depth cap, filling 12.5% of the time within the rest window.

  python3 liquidity_sizing.py      (needs out/live_books.csv from the book sampler)
"""

import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import search as S, model_variants_backtest as MV
from bot import evening_config as C
bk=pd.read_csv("out/live_books.csv"); bk=bk[(bk.mid>=0.05)&(bk.mid<=0.55)]
D0,D1,D2=bk.d0.values, bk.d1.values, bk.d2.values          # cumulative shares at ask, ask+1c, ask+2c
SPREAD=0.02                                                 # measured median
P_MAKER=0.125
P=S.augment(MV.P); F=S.walk(P,dict())
def build(base,cap):
    ob,oc=C.STAKE,C.MAX_PER_CITY_DAY_USD
    C.STAKE,C.MAX_PER_CITY_DAY_USD=base,base*3.5
    try: T=S.run2(F, modes=C.MODES, stake_mode="edge", edge_mult=6.0, stake_cap=cap, no_leg=C.NO_LEG, no_mode="both")
    finally: C.STAKE,C.MAX_PER_CITY_DAY_USD=ob,oc
    return T
def walk_book(need, a, d0,d1,d2, cap):
    """Shares obtained and total cost, walking ask -> ask+1c -> ask+2c, limited to cap*depth at each level."""
    lim=cap*d2
    take=min(need, lim)
    t0=min(take, cap*d0); t1=min(take-t0, cap*(d1-d0)); t2=max(0.0, take-t0-t1)
    cost=t0*a + t1*(a+0.01) + t2*(a+0.02)
    return t0+t1+t2, cost
def evaluate(base, cap_usd, depth_cap, maker=True, ndraw=250, seed=11):
    T=build(base,cap_usd)
    want=T.stake.values; mid=T.price.values; won=T.won.values
    ask=mid+SPREAD/2; sh_want=want/mid
    rng=np.random.default_rng(seed); tot=[];fr=[];eff=[]
    for _ in range(ndraw):
        i=rng.integers(0,len(D1),len(T))
        mk=rng.random(len(T))<P_MAKER if maker else np.zeros(len(T),bool)
        got=np.empty(len(T)); cost=np.empty(len(T))
        for k in range(len(T)):
            if mk[k]:
                got[k]=sh_want[k]; cost[k]=sh_want[k]*(ask[k]-0.01)     # maker fill at ask-1c, full size
            else:
                got[k],cost[k]=walk_book(sh_want[k], ask[k], D0[i[k]],D1[i[k]],D2[i[k]], depth_cap)
        px=np.where(got>0, cost/np.maximum(got,1e-9), mid)
        pnl=np.where(won, got-cost, -cost) - 0.05*px*(1-px)*got
        tot.append(pnl.sum()); fr.append(got.sum()/sh_want.sum()); eff.append((px-mid)[got>0].mean())
    return np.mean(tot), np.std(tot), np.mean(fr), np.mean(eff), T.pnl.sum(), want.sum()
print(f"spread {SPREAD*100:.0f}c | depth from {len(D1)} live books (median within 1c: {np.median(D1):.0f} sh)\n")
print(f"{'base':>6}{'cap':>6}{'dcap':>6}{'intended$':>11}{'frictionless':>13}{'realistic':>11}{'sd':>6}{'fill%':>7}{'slip':>7}")
best=None
for dcap in (0.30,0.75):
    for base in (2.0,3.0,4.0,5.0,6.0,8.5,12.0):
        m,s,f,e,ideal,wa=evaluate(base, base*3.0, dcap)
        print(f"{base:>6.1f}{base*3:>6.0f}{dcap:>6.2f}{wa:>11.0f}{ideal:>13.0f}{m:>11.0f}{s:>6.0f}{f:>7.1%}{e*100:>6.2f}c")
        if best is None or m>best[0]: best=(m,base,dcap)
print(f"\nbest realistic P&L: base ${best[1]:.1f}, DEPTH_CAP {best[2]:.2f} -> ${best[0]:.0f}")
