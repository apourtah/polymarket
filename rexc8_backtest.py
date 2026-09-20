"""Rexc8's trade (entry only when the bucket is already priced 93-98c = the market agrees the max is locked), backtested on the six-week tapes (7 cities, Aug 7 - Sep 18): after hour H local, buy YES on the bucket that
contains the day's RUNNING METAR max at <= ENTRY_MAX (0.98), then either sell at 0.99 (filled when a later taker buys YES /
sells NO at >= 0.99) or hold to resolution. Fills come from the taker tape: taker entry = next print at <= ENTRY_MAX after H;
limit entry at 0.97 = a later taker sells YES / buys NO at < 0.97. Depth proxy = $ printed at <= ENTRY_MAX after H.
Fee 5%*p*(1-p) on taker buys; maker sells free. $100 per market, capped at the depth proxy."""
import pandas as pd, numpy as np, json
from zoneinfo import ZoneInfo
from decimal import Decimal, ROUND_HALF_UP
from sweep_local import TZ
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
rh=lambda x: int(Decimal(str(x)).quantize(Decimal('1'),rounding=ROUND_HALF_UP))
T=pd.read_parquet('out/wallet_fills.parquet',columns=['cid','city','day','lo','ts','side','outcome','p_yes','size','usd','won']).sort_values('ts')
T['sells_yes']=((T.side=='SELL')&(T.outcome=='Yes'))|((T.side=='BUY')&(T.outcome=='No'))
m=pd.read_parquet('data/metar.parquet'); m=m[m.city.isin(T.city.unique())&(m.local_time>='2026-08-06')].copy(); m['tf']=(m.temp_c*9/5+32).map(rh); m['day']=m.local_time.dt.date.astype(str)
won={(c,d,lo):w for (c,d,lo),w in T.groupby(['city','day','lo']).won.first().items()}
ENTRY_MAX=0.98; ENTRY_MIN=0.93; STAKE=100.0
rows=[]
for (c,d),obs in m.groupby(['city','day']):
    tapes=T[(T.city==c)&(T.day==d)]
    if tapes.empty: continue
    tz=ZoneInfo(TZ[c]); final=obs.tf.max()
    for H in [15,16,17,18,19,20]:
        t0=pd.Timestamp(f"{d} {H:02d}:00"); o=obs[obs.local_time<=t0]
        if o.empty: continue
        run=o.tf.max(); lo=run-(run%2); k=(c,d,lo)
        if k not in won: continue
        ts0=t0.tz_localize(tz).timestamp(); g=tapes[(tapes.lo==lo)&(tapes.ts>ts0)]
        if g.empty: continue
        # taker entry: first print at <= ENTRY_MAX after H (we lift the offer at that price)
        e=g[g.p_yes.between(ENTRY_MIN,ENTRY_MAX)]
        depth=e.usd.sum()
        # limit entry at 0.97: someone sells to us below 0.97
        lim=g[g.sells_yes&(g.p_yes<0.97)&(g.p_yes>=ENTRY_MIN-0.03)]
        for mode,ent in [('taker<=0.98',e),('limit@0.97',lim)]:
            if ent.empty: rows.append(dict(city=c,day=d,H=H,mode=mode,filled=False)); continue
            f=ent.iloc[0]; p=f.p_yes if mode.startswith('taker') else 0.97; t_in=f.ts
            later=g[(g.ts>t_in)&(~g.sells_yes)&(g.p_yes>=0.99)]          # a later taker buys YES at >= 0.99 -> our ask at 0.99 is lifted
            exit99=len(later)>0; w=won[k]; stake=min(STAKE,max(depth,5))
            sh=stake/p; fee=0.05*p*(1-p)*sh if mode.startswith('taker') else 0.0
            pnl_hold=(sh-stake if w else -stake)-fee
            pnl_sell=((0.99*sh-stake) if exit99 else (sh-stake if w else -stake))-fee
            rows.append(dict(city=c,day=d,H=H,mode=mode,filled=True,entry=p,stake=stake,depth=depth,won=w,max_rose=final>run,exit99=exit99,hours_to_exit=((later.ts.iloc[0]-t_in)/3600 if exit99 else np.nan),pnl_hold=pnl_hold,pnl_sell=pnl_sell))
R=pd.DataFrame(rows); R.to_csv('out/rexc8_backtest.csv',index=False)
for c_ in ['won','max_rose','exit99']: R[c_]=R[c_].astype(float)
def summ(x):
    f=x[x.filled]
    if len(f)==0: return pd.Series(dict(city_days=len(x),filled=0))
    d=f.groupby('day').pnl_sell.sum()
    return pd.Series(dict(city_days=len(x),fill_rate=len(f)/len(x),avg_entry=f.entry.mean(),med_depth=f.depth.median(),won=f.won.mean(),max_rose=f.max_rose.mean(),exit99=f.exit99.mean(),med_h_to_exit=f.hours_to_exit.median(),
                          staked=f.stake.sum(),pnl_hold=f.pnl_hold.sum(),roi_hold=f.pnl_hold.sum()/f.stake.sum(),pnl_sell99=f.pnl_sell.sum(),roi_sell99=f.pnl_sell.sum()/f.stake.sum(),per_day=f.pnl_sell.sum()/f.day.nunique(),neg_days=(d<0).sum(),worst_day=d.min()))
print("Rexc8 replay, Aug 7 - Sep 18, 7 cities, $100 per market capped at the $ printed at <=0.98 after H (depth proxy)\n")
for mode in ['taker<=0.98','limit@0.97']:
    print(f"### entry {mode}"); print(R[R['mode']==mode].groupby('H').apply(summ).round(3).to_string()); print()
f=R[R.filled&(R['mode']=='taker<=0.98')&(R.H>=17)]
print("taker, H>=17, by entry price band:"); print(f.groupby(pd.cut(f.entry,[0,.9,.95,.97,.98]),observed=True).apply(lambda x: pd.Series(dict(n=len(x),won=x.won.mean(),max_rose=x.max_rose.mean(),exit99=x.exit99.mean(),roi_hold=x.pnl_hold.sum()/x.stake.sum(),roi_sell=x.pnl_sell.sum()/x.stake.sum(),staked=x.stake.sum()))).round(3).to_string())
print("\nlosses (taker, H>=17): "); print(f[f.won==0][['city','day','H','entry','stake','max_rose','exit99','pnl_sell']].to_string(index=False))
print("\nby city, taker H=18, sell@99:"); x=R[R.filled&(R['mode']=='taker<=0.98')&(R.H==18)]; print(x.groupby('city').apply(lambda g: pd.Series(dict(n=len(g),won=g.won.mean(),avg_entry=g.entry.mean(),med_depth=g.depth.median(),pnl=g.pnl_sell.sum(),roi=g.pnl_sell.sum()/g.stake.sum()))).round(3).to_string())
