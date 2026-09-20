"""Timing forensics on the top wallets: do they trade off a faster observation feed (fills seconds after a METAR print that set
a new daily max), do they lead price jumps, do they run on a timer? Uses out/wallet_fills.parquet + data/metar.parquet."""
import pandas as pd, numpy as np, json
from zoneinfo import ZoneInfo
from decimal import Decimal, ROUND_HALF_UP
from sweep_local import TZ
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
rh=lambda x: int(Decimal(str(x)).quantize(Decimal('1'),rounding=ROUND_HALF_UP))
T=pd.read_parquet('out/wallet_fills.parquet'); W=pd.read_csv('out/wallets_all.csv',index_col=0)
site=json.load(open('out/stations.json')); A=W[(W.markets>=10)&(W.days>=8)]
TOP=list(A.sort_values('pnl',ascending=False).head(30).index)+list(A[A.usd>=5000].sort_values('mk1h',ascending=False).head(10).index); TOP=list(dict.fromkeys(TOP))
# ---- METAR prints that set a new daily max (the informative events), per station-day ----
m=pd.read_parquet('data/metar.parquet'); m=m[m.city.isin(T.city.unique())&(m.local_time>='2026-08-06')].copy(); m['tf']=(m.temp_c*9/5+32).map(rh); m['day']=m.local_time.dt.date
m=m.sort_values(['city','local_time']); m['runmax']=m.groupby(['city','day']).tf.cummax(); m['newmax']=(m.tf==m.runmax)&(m.tf>m.groupby(['city','day']).runmax.shift(1).fillna(-999))
ev=m[m.newmax&m.local_time.dt.hour.between(9,19)].copy(); ev['ts']=[pd.Timestamp(t).tz_localize(ZoneInfo(TZ[c])).timestamp() for t,c in zip(ev.local_time,ev.city)]
print(f"{len(ev)} new-daily-max METAR prints (09-19 local) across the 7 stations, Aug 6 - Sep 19")
# for every market-day fill, seconds since the last new-max print of that city (same day), and whether the print bumped the max
T0=T[T.rel_day==0].copy(); out=[]
for c,g in T0.groupby('city'):
    e=ev[ev.city==c].sort_values('ts'); ets=e.ts.values; idx=np.searchsorted(ets,g.ts.values,side='right')-1
    since=np.where(idx>=0,g.ts.values-ets[np.maximum(idx,0)],np.nan); out.append(pd.Series(since,index=g.index))
T0['since_print']=pd.concat(out)
def timing(g):
    sp=g.since_print.dropna(); mins=(g.ts%3600)//60
    return pd.Series(dict(fills_dayof=len(g),within2min=(sp<=120).mean(),within5min=(sp<=300).mean(),med_since_print_s=sp.median(),
                          share_min53_59=((mins>=53)|(mins<=1)).mean()))
base=timing(T0); print("\nbaseline (all wallets, market-day fills): within 2 min of a new-max print %.1f%%, within 5 min %.1f%%, share of fills in minutes :53-:01 %.1f%% (uniform = 15%%)"%(100*base.within2min,100*base.within5min,100*base.share_min53_59))
rows=[]
for w in TOP:
    g=T0[T0.w==w]; 
    if len(g)<10: continue
    t=timing(g); gg=T[T.w==w].sort_values('ts'); gaps=gg.groupby('day').ts.diff().dropna()/60
    modal=gaps.round().value_counts().head(1); t['n_fills']=len(gg); t['modal_gap_min']=modal.index[0] if len(modal) else np.nan; t['modal_gap_share']=modal.iloc[0]/len(gaps) if len(gaps) else np.nan
    t['pnl']=W.loc[w,'pnl']; t['roi']=W.loc[w,'roi']; t['mk1h']=W.loc[w,'mk1h']; t['uname']=W.loc[w,'uname'] if isinstance(W.loc[w,'uname'],str) else ''; rows.append(pd.Series(t,name=w))
R=pd.DataFrame(rows); print("\n### timing signature of the top wallets (market-day fills)"); print(R.sort_values('within2min',ascending=False).round(3).to_string())
# ---- leading price jumps: fills in the 5 min BEFORE a >=10c move, in the move's direction ----
T=T.sort_values(['cid','ts']); T['dp']=T.groupby('cid').p_yes.diff(-1)*-1; T['dt_next']=T.groupby('cid').ts.diff(-1)*-1
J=T[(T.dp.abs()>=0.10)&(T.dt_next<=600)][['cid','ts','dp']].rename(columns={'ts':'jts'})
lead=[]
for cid,g in T[T.w.isin(TOP)].groupby('cid'):
    j=J[J.cid==cid]
    if j.empty: continue
    for r in g.itertuples():
        d=j[(j.jts>r.ts)&(j.jts<=r.ts+300)]
        if len(d): lead.append(dict(w=r.w,dir_ok=(np.sign(d.dp.iloc[0])==np.sign(r.q))))
L=pd.DataFrame(lead)
if len(L):
    l=L.groupby('w').agg(fills_before_jump=('dir_ok','size'),share_right_direction=('dir_ok','mean')); l['fills_total']=T[T.w.isin(TOP)].groupby('w').size(); l['rate']=l.fills_before_jump/l.fills_total
    l['uname']=[W.loc[w,'uname'] if isinstance(W.loc[w,'uname'],str) else '' for w in l.index]
    allr=len(T[(T.dt_next<=300)])  # crude baseline
    print("\n### fills placed within 5 min BEFORE a >=10c jump (in the jump's direction): who trades just before the market moves?"); print(l.sort_values('rate',ascending=False).head(20).round(3).to_string())
R.to_csv('out/wallets_timing.csv')
