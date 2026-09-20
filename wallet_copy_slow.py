"""Copy the SLOW traders, chosen ex-ante. Slow = in the train window (Aug 7-27): <= 12 fills/day, >= 10 markets, >= 8 days,
<= 10% of fills within 5 min of a new-max METAR print (not a sniper), P&L > 0 and t >= 1. Test Aug 28 - Sep 18 in the 7 cities:
copy their first fill per market (a) at the next print >= 60 s later, (b) as a resting limit at their price. Also: how much of
what they buy is already our bot's pick, and what the non-overlapping copies are worth."""
import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
T=pd.read_parquet('out/wallet_fills.parquet',columns=['w','cid','city','day','lo','ts','side','outcome','p_yes','q','usd','pnl','won','rel_day']).sort_values('ts'); W=pd.read_csv('out/wallets_all.csv',index_col=0)
TM=pd.read_csv('out/wallets_timing.csv',index_col=0) if False else None
TRAIN_END='2026-08-27'; tr=T[T.day<=TRAIN_END]; te=T[T.day>TRAIN_END]
# print-proximity from the timing script's method (recompute quickly): new-max METAR prints
from zoneinfo import ZoneInfo
from decimal import Decimal, ROUND_HALF_UP
from sweep_local import TZ
rh=lambda x: int(Decimal(str(x)).quantize(Decimal('1'),rounding=ROUND_HALF_UP))
m=pd.read_parquet('data/metar.parquet'); m=m[m.city.isin(T.city.unique())&(m.local_time>='2026-08-06')].copy(); m['tf']=(m.temp_c*9/5+32).map(rh); m['d']=m.local_time.dt.date
m=m.sort_values(['city','local_time']); m['runmax']=m.groupby(['city','d']).tf.cummax(); m['newmax']=(m.tf==m.runmax)&(m.tf>m.groupby(['city','d']).runmax.shift(1).fillna(-999))
ev=m[m.newmax].copy(); ev['ts']=[pd.Timestamp(t).tz_localize(ZoneInfo(TZ[c])).timestamp() for t,c in zip(ev.local_time,ev.city)]
def near_print(g):
    out=np.zeros(len(g),bool)
    for c,idx in g.groupby('city').groups.items():
        e=np.sort(ev[ev.city==c].ts.values); t=g.loc[idx].ts.values; i=np.searchsorted(e,t,side='right')-1; out[g.index.get_indexer(idx)]=(i>=0)&((t-e[np.maximum(i,0)])<=300)
    return out
tr=tr.copy(); tr['near']=near_print(tr)
S=tr.groupby('w').agg(pnl=('pnl','sum'),usd=('usd','sum'),mk=('cid','nunique'),days=('day','nunique'),fills=('pnl','size'),near=('near','mean'),before=('rel_day',lambda s:(s<0).mean()),avg_p=('p_yes','mean'))
dd=tr.groupby(['w','day']).pnl.sum().groupby('w').agg(['mean','std','count']); S['t']=dd['mean']/(dd['std']/np.sqrt(dd['count'])); S['fpd']=S.fills/S.days
import os
PROF=os.environ.get('PROF','loose')
S['lhour']=tr.assign(h=((tr.ts%86400)/3600)).groupby('w').h.mean()   # rough
night=tr.copy(); night['loc']=[(t/3600)%24 for t in night.ts]  # placeholder
S['night_share']=tr.assign(n=(tr.rel_day==0)&(tr.ts%86400>=0)).groupby('w').n.mean()
if PROF=='loose': SLOW=S[(S.fpd<=12)&(S.mk>=10)&(S.days>=8)&(S.near<=0.10)&(S.pnl>0)&(S.t>=1.0)]
elif PROF=='forecast': SLOW=S[(S.fpd<=6)&(S.mk>=10)&(S.days>=8)&(S.near<=0.10)&(S.pnl>500)&(S.t>=1.5)&S.avg_p.between(0.25,0.55)]
elif PROF=='strong': SLOW=S[(S.fpd<=8)&(S.mk>=10)&(S.days>=8)&(S.near<=0.10)&(S.pnl>1000)&(S.t>=1.3)]
elif PROF=='daybefore': SLOW=S[(S.fpd<=8)&(S.mk>=10)&(S.days>=8)&(S.near<=0.10)&(S.pnl>300)&(S.t>=1.0)&(S.before>=0.5)]
SLOW=SLOW.sort_values('pnl',ascending=False)
name=lambda w: (W.uname.get(w,'') if isinstance(W.uname.get(w,'')) and not str(W.uname.get(w,'')).startswith('0x') else w[:8]) if False else (str(W.uname.get(w,''))[:16] if isinstance(W.uname.get(w,''),str) and not str(W.uname.get(w,'')).startswith('0x') else w[:8])
print(f"profile {PROF}: {len(SLOW)} wallets:", [name(w) for w in SLOW.index][:25])
# execution
te=te.sort_values(['cid','ts']); nx={}; idx={}
for cid,g in te.groupby('cid'):
    ts=g.ts.values; py=g.p_yes.values; i=np.searchsorted(ts,ts+60,side='left'); ok=i<len(ts); nx.update(dict(zip(g.index,np.where(ok,py[np.minimum(i,len(ts)-1)],np.nan))))
    sy=(((g.side=='SELL')&(g.outcome=='Yes'))|((g.side=='BUY')&(g.outcome=='No'))).values; idx[cid]=(ts,py,sy,g.w.values)
te['p60']=pd.Series(nx)
def limit_fill(r):
    ts,py,sy,ww=idx[r.cid]; msk=(ts>r.ts)&(ww!=r.w); ok=msk&sy&(py<r.p_yes) if r.q>0 else msk&(~sy)&(py>r.p_yes); return bool(ok.any())
STAKE=10.0
def clip(F,pcol):
    F=F.dropna(subset=[pcol])
    if len(F)==0: return dict(n=0)
    long=F.q>0; ask=np.where(long,F[pcol],1-F[pcol]).clip(0.01,0.99); sh=STAKE/ask; fee=0.05*ask*(1-ask)*sh; win=np.where(long,F.won,~F.won); p=np.where(win,sh-STAKE,-STAKE)-fee
    d=pd.Series(p,index=F.index).groupby(F.day.values).sum(); return dict(n=len(F),win=round(win.mean(),2),pnl=round(p.sum()),roi=round(p.sum()/(STAKE*len(F)),3),neg_days=int((d<0).sum()),days=len(d))
F=te[te.w.isin(SLOW.index)].sort_values('ts').drop_duplicates(['w','cid']).copy(); F['lim']=[limit_fill(r) for r in F.itertuples()]
print(f"\n### test Aug 28 - Sep 18, first fill per market, $10:")
print("at their own price (their P&L on these fills):", clip(F,'p_yes')); print("copied at the next print >= 60 s:          ", clip(F,'p60')); print(f"resting limit at their price, filled {F.lim.mean():.0%}:", clip(F[F.lim],'p_yes'))
rows=[]
for w in SLOW.index:
    g=F[F.w==w]
    if len(g)==0: continue
    a=clip(g,'p_yes'); b=clip(g,'p60'); c=clip(g[g.lim],'p_yes'); rows.append(dict(name=name(w),orders=len(g),their_roi=a['roi'],copy_next_print=b['roi'],limit_fill=round(g.lim.mean(),2),limit_roi=c.get('roi'),pnl_next=b['pnl'],pnl_limit=c.get('pnl')))
print(pd.DataFrame(rows).head(25).to_string(index=False))
# overlap with our bot's picks (evening rule dump: our YES per city-night)
B=pd.read_parquet('out/backtest_evening_buckets.parquet',columns=['city','mday','lo','price','agree','be','br','won']); B['mday']=pd.to_datetime(B.mday)
MODES={"Los Angeles":{"agree","ewma"},"Austin":{"agree","ewma"},"Chicago":{"agree","ridge"},"Houston":{"ridge"},"Dallas":{"ridge"},"Seattle":{"agree"},"Miami":{"agree"}}
B['modes']=B.city.map(lambda c: ','.join(sorted(MODES[c])))
ours=pd.concat([B[B.agree&(B.lo==B.be)&B.modes.str.contains('agree')&B.price.between(0.10,0.53)],B[~B.agree&(B.lo==B.br)&B.modes.str.contains('ridge')&B.price.between(0.05,0.45)],B[~B.agree&(B.lo==B.be)&B.modes.str.contains('ewma')&B.price.between(0.05,0.45)]]).drop_duplicates(['city','mday'])
ok=set(zip(ours.city,ours.mday.dt.strftime('%Y-%m-%d'),ours.lo)); on=set(zip(ours.city,ours.mday.dt.strftime('%Y-%m-%d')))
F['same_pick']=[(c,d,l) in ok for c,d,l in zip(F.city,F.day,F.lo)]; F['our_night']=[(c,d) in on for c,d in zip(F.city,F.day)]; F['long']=F.q>0
print(f"\noverlap with our bot (test period): of their {len(F)} first fills, {F.same_pick.mean():.0%} are exactly our YES pick; on {F.our_night.mean():.0%} of them our bot traded that city-night")
print("copies split by overlap (next-print execution):")
for k,g in F.groupby(['long','same_pick']): print(f"  long={k[0]} same_pick={k[1]}: {clip(g,'p60')}")
print("  their fills on city-nights where our bot has NO trade:", clip(F[~F.our_night],'p60'), "| limit:", clip(F[~F.our_night&F.lim],'p_yes'))
