"""Current strategy with a resting LIMIT SELL above the entry (take-profit), filled when the 5-min mid reaches the target
(maker: no fee, price = target); otherwise hold to resolution. Sweep: target = entry + D (cents) and absolute levels; full or
half exits. Same trades/paths as trail_stop_backtest.py."""
import pandas as pd, numpy as np, json, bisect, datetime as dt
from zoneinfo import ZoneInfo
from sweep_local import TZ
exec(open('trail_stop_backtest.py').read().split("TRAILS=")[0])
REL=[0.05,0.10,0.15,0.20,0.30,0.40]; ABS=[0.60,0.70,0.80,0.90,0.95]; FR=[1.0,0.5]
res={}; hold=[]
for r in T.itertuples():
    if r.market_id is None: continue
    try: h=json.load(open(f"data/prices/{r.market_id}.json"))
    except FileNotFoundError: continue
    hs=h['history']
    if not hs: continue
    ts=np.array([x['t'] for x in hs]); p=np.array([x['p'] for x in hs]); yes=p if h['losing_outcome']=='Yes' else 1-p
    tz=ZoneInfo(TZ[r.city]); t0=int(dt.datetime(r.mday.year,r.mday.month,r.mday.day,21,35,tzinfo=tz).timestamp())-86400; i0=bisect.bisect_right(ts,t0)
    pos=yes[i0:] if r.side=='yes' else 1-yes[i0:]
    if len(pos)<2: continue
    entry=(r.price if r.side=='yes' else 1-r.price)+SLIP; sh=STAKE/entry; fee_in=0.05*entry*(1-entry)*sh; won=r.won if r.side=='yes' else (not r.won); final=1.0 if won else 0.0
    pnl_hold=sh*final-STAKE-fee_in; hold.append(pnl_hold); runmax=np.maximum.accumulate(pos)
    for kind,levels in [('rel',REL),('abs',ABS)]:
        for L in levels:
            tgt=min(entry+L,0.99) if kind=='rel' else L
            if tgt<=entry: continue
            hit=runmax.max()>=tgt
            for fr in FR:
                if hit: pnl=fr*sh*tgt+(1-fr)*sh*final-STAKE-fee_in
                else: pnl=pnl_hold
                res.setdefault((kind,L,fr),[]).append((pnl,hit,r.leg,str(r.mday)[:7],won,r.mday))
H=np.array(hold); hd=pd.Series(H,index=[r.mday for r in T.itertuples() if r.market_id is not None][:len(H)]) if False else None
days=[]; 
for r in T.itertuples():
    if r.market_id is None: continue
    days.append(r.mday)
def mdd(pnl,dates):
    d=pd.Series(pnl).groupby(pd.Series(dates).values).sum().sort_index(); eq=d.cumsum(); return (eq-eq.cummax()).min()
first=next(iter(res.values())); hd=[x[5] for x in first]
print(f"HOLD: ${H.sum():+,.0f} on ${STAKE*len(H):,.0f} ({H.sum()/(STAKE*len(H)):+.1%}), n={len(H)}, win rate {(H>0).mean():.1%}, max drawdown ${mdd(H,hd):,.0f}")
rows=[]
for (kind,L,fr),v in res.items():
    d=pd.DataFrame(v,columns=['pnl','hit','leg','month','won','mday']); m=d.groupby('month').pnl.sum()
    rows.append(dict(target=(f"entry+{L:.2f}" if kind=='rel' else f"abs {L:.2f}"),fraction=fr,n=len(d),hit=d.hit.mean(),win_rate=(d.pnl>0).mean(),pnl=d.pnl.sum(),roi=d.pnl.sum()/(STAKE*len(d)),vs_hold=d.pnl.sum()-H.sum(),max_dd=mdd(d.pnl.values,d.mday.values),neg_months=(m<0).sum(),worst_month=m.min()))
R=pd.DataFrame(rows); print("\n### take-profit limit sweep (maker fill at the target when the mid reaches it; unfilled -> hold)"); print(R.round(2).to_string(index=False))
