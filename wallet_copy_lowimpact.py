"""Copy-trade only what does not move the market. Two versions:
 (a) wallet-level: alpha wallets (train) whose fills have low price impact in the train period (next print within 1c of theirs)
 (b) fill-level: copy any alpha-wallet fill only if the next print >= 60 s later is within IMPACT_MAX of their price
 plus the same fill filter applied to every wallet / the 'informed flow' model. Train Aug 7-27, test Aug 28-Sep 18, $10 clips."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore'); pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
T=pd.read_parquet('out/wallet_fills.parquet').sort_values('ts'); W=pd.read_csv('out/wallets_all.csv',index_col=0)
Tm=T.sort_values(['cid','ts']); nx={}
for cid,g in Tm.groupby('cid'):
    ts=g.ts.values; py=g.p_yes.values
    for lag,name in [(60,'p60'),(600,'p600'),(3600,'p3600')]:
        idx=np.searchsorted(ts,ts+lag,side='left'); ok=idx<len(ts); nx.setdefault(name,{}).update(dict(zip(g.index,np.where(ok,py[np.minimum(idx,len(ts)-1)],np.nan))))
for k,v in nx.items(): T[k]=pd.Series(v)
T['imp60']=np.sign(T.q)*(T.p60-T.p_yes); T['imp600']=np.sign(T.q)*(T.p600-T.p_yes)      # >0 = price moved their way after the fill (impact + information)
TRAIN_END='2026-08-27'; tr=T[T.day<=TRAIN_END]; te=T[T.day>TRAIN_END]
Wtr=tr.groupby('w').agg(pnl=('pnl','sum'),usd=('usd','sum'),mk=('cid','nunique'),days=('day','nunique'),imp60=('imp60','median'),imp60_mean=('imp60','mean'),moved=('imp60',lambda s:(s.abs()>0.01).mean()))
dd=tr.groupby(['w','day']).pnl.sum().groupby('w').agg(['mean','std','count']); Wtr['t']=dd['mean']/(dd['std']/np.sqrt(dd['count']))
ALPHA=Wtr[(Wtr.mk>=8)&(Wtr.days>=6)&(Wtr.pnl>300)&(Wtr.t>1.0)].sort_values('pnl',ascending=False).head(25)
print("train-period alpha candidates with their price impact (imp60: median move of the next print in their direction; moved: share of fills where the next print differs by >1c):")
print(ALPHA.assign(name=[W.uname.get(w,'') for w in ALPHA.index])[['pnl','usd','mk','days','t','imp60','imp60_mean','moved','name']].round(3).to_string())
def clip(F,pcol='p60',stake=10):
    F=F.sort_values('ts').drop_duplicates(['w','cid']).dropna(subset=[pcol])
    if len(F)==0: return pd.Series(dict(clips=0,pnl=0,roi=np.nan,win=np.nan,neg_days=np.nan,days=0))
    long=F.q>0; ask=np.where(long,F[pcol],1-F[pcol]).clip(0.01,0.99); sh=stake/ask; fee=0.05*ask*(1-ask)*sh; win=np.where(long,F.won,~F.won); p=np.where(win,sh-stake,-stake)-fee
    days=pd.Series(p,index=F.index).groupby(F.day.values).sum(); return pd.Series(dict(clips=len(F),pnl=p.sum(),roi=p.sum()/(stake*len(F)),win=win.mean(),neg_days=(days<0).sum(),days=len(days)))
out={}
alpha_ids=ALPHA.index.tolist(); low=ALPHA[ALPHA.moved<=0.5].index.tolist(); high=ALPHA[ALPHA.moved>0.5].index.tolist()
print(f"\nlow-impact alpha wallets (moved <= 50% of fills): {len(low)}; high-impact: {len(high)}")
out['all alpha ids, copy at +60s']=clip(te[te.w.isin(alpha_ids)])
out['(a) LOW-impact alpha wallets, copy at +60s']=clip(te[te.w.isin(low)])
out['    high-impact alpha wallets, copy at +60s']=clip(te[te.w.isin(high)])
for mx in [0.0,0.01,0.02]:
    out[f'(b) alpha ids, copy only if next print within {mx:.2f} of theirs']=clip(te[te.w.isin(alpha_ids)&(te.imp60.abs()<=mx+1e-9)])
    out[f'(b) LOW-impact wallets AND next print within {mx:.2f}']=clip(te[te.w.isin(low)&(te.imp60.abs()<=mx+1e-9)])
out['(b) all wallets, next print within 0.00 (baseline for the filter)']=clip(te[(te.imp60.abs()<=1e-9)].sample(40000,random_state=0))
# horizon check: for the low-impact set, price 10 min / 1 h later (is there still information after they trade?)
for pcol,lab in [('p600','+10min'),('p3600','+1h')]: out[f'(a) LOW-impact alpha wallets, copy at {lab}']=clip(te[te.w.isin(low)],pcol)
print("\n### test Aug 28 - Sep 18, $10 per wallet-market (first fill), fee included"); print(pd.DataFrame(out).T.round(2).to_string())
# per-wallet in the low set
rows=[]
for w in low:
    F=te[te.w==w]; s=clip(F); s60=clip(F[F.imp60.abs()<=1e-9]); rows.append(dict(w=w[:10],name=W.uname.get(w,''),train_pnl=round(ALPHA.loc[w,'pnl']),moved=round(ALPHA.loc[w,'moved'],2),test_clips=int(s.clips),test_roi=round(s.roi,2) if s.clips else np.nan,unmoved_clips=int(s60.clips),unmoved_roi=round(s60.roi,2) if s60.clips else np.nan))
print("\nlow-impact wallets, one by one (test period):"); print(pd.DataFrame(rows).to_string(index=False))
