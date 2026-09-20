"""Select wallets by (alpha in train) x (low market impact in train), then copy ALL their fills in the test period.
Impact measures (train): mean and median of the next-print move in their direction (imp60), share of fills with |move|>1c,
and $-weighted impact. Sweep thresholds; copy every fill at the +60 s print, $10 per fill or size-proportional (capped)."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore'); pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
T=pd.read_parquet('out/wallet_fills.parquet').sort_values('ts'); W=pd.read_csv('out/wallets_all.csv',index_col=0)
Tm=T.sort_values(['cid','ts']); nx={}
for cid,g in Tm.groupby('cid'):
    ts=g.ts.values; py=g.p_yes.values
    for lag,name in [(60,'p60'),(600,'p600')]:
        idx=np.searchsorted(ts,ts+lag,side='left'); ok=idx<len(ts); nx.setdefault(name,{}).update(dict(zip(g.index,np.where(ok,py[np.minimum(idx,len(ts)-1)],np.nan))))
for k,v in nx.items(): T[k]=pd.Series(v)
T['imp60']=np.sign(T.q)*(T.p60-T.p_yes); T['aimp60']=(T.p60-T.p_yes).abs()
TRAIN_END='2026-08-27'; tr=T[T.day<=TRAIN_END]; te=T[T.day>TRAIN_END]
Wtr=tr.groupby('w').agg(pnl=('pnl','sum'),usd=('usd','sum'),fills=('pnl','size'),mk=('cid','nunique'),days=('day','nunique'),imp_med=('imp60','median'),imp_mean=('imp60','mean'),aimp_mean=('aimp60','mean'),moved=('aimp60',lambda s:(s>0.01).mean()))
Wtr['imp_usd']=tr.assign(x=tr.imp60*tr.usd).groupby('w').x.sum()/Wtr.usd
dd=tr.groupby(['w','day']).pnl.sum().groupby('w').agg(['mean','std','count']); Wtr['t']=dd['mean']/(dd['std']/np.sqrt(dd['count'])); Wtr['roi']=Wtr.pnl/Wtr.usd
CAND=Wtr[(Wtr.mk>=8)&(Wtr.days>=6)&(Wtr.pnl>200)&(Wtr.t>0.8)].copy(); print(f"train alpha candidates: {len(CAND)}")
def copy_all(F,mode='flat',stake=10,cap=50):
    F=F.dropna(subset=['p60'])
    if len(F)==0: return pd.Series(dict(fills=0,usd=0,pnl=0,roi=np.nan,win=np.nan,neg_days=np.nan,days=0,wallets=0))
    long=F.q>0; ask=np.where(long,F.p60,1-F.p60).clip(0.01,0.99); st=np.full(len(F),float(stake)) if mode=='flat' else np.minimum(F.usd.values,cap)
    sh=st/ask; fee=0.05*ask*(1-ask)*sh; win=np.where(long,F.won,~F.won); p=np.where(win,sh-st,-st)-fee
    days=pd.Series(p,index=F.index).groupby(F.day.values).sum(); return pd.Series(dict(fills=len(F),usd=st.sum(),pnl=p.sum(),roi=p.sum()/st.sum(),win=win.mean(),neg_days=(days<0).sum(),days=len(days),wallets=F.w.nunique()))
out={}
out['all candidates, every fill, $10']=copy_all(te[te.w.isin(CAND.index)])
for metric,ths in [('moved',[0.45,0.50,0.55,0.60,0.65]),('imp_mean',[0.0,0.002,0.005,0.01]),('imp_med',[0.0,0.002]),('aimp_mean',[0.01,0.02,0.03]),('imp_usd',[0.0,0.005,0.01])]:
    for th in ths:
        sel=CAND[CAND[metric]<=th].index
        if len(sel)==0: continue
        out[f'{metric} <= {th}: {len(sel)} wallets, every fill $10']=copy_all(te[te.w.isin(sel)])
print("\n### test Aug 28 - Sep 18: copy EVERY fill of the selected wallets at the +60 s print, fee included")
print(pd.DataFrame(out).T.round(3).to_string())
# 2-D: alpha strength x impact
print("\n### grid: train t-stat threshold x impact (moved) threshold -> test ROI (fills)")
rows={}
for tmin in [0.8,1.5,2.5]:
    for th in [0.5,0.6,0.7,1.0]:
        sel=CAND[(CAND.t>=tmin)&(CAND.moved<=th)].index; s=copy_all(te[te.w.isin(sel)]); rows.setdefault(f't>={tmin}',{})[f'moved<={th}']=f"{s.roi:+.2f} ({int(s.fills)}f/{int(s.wallets)}w)" if s.fills else '-'
print(pd.DataFrame(rows).T.to_string())
# size-proportional copying (mirror their $ up to $50 per fill)
sel=CAND[(CAND.t>=1.5)&(CAND.moved<=0.6)].index; print("\nsize-proportional (mirror their $, cap $50/fill), t>=1.5 & moved<=0.6:", copy_all(te[te.w.isin(sel)],'prop').round(3).to_dict())
# per-wallet view of the lowest-impact alpha wallets in the test period
low=CAND.sort_values('moved').head(15)
rows=[]
for w in low.index:
    s=copy_all(te[te.w==w]); s10=copy_all(te[te.w==w].assign(p60=te[te.w==w].p600))
    rows.append(dict(w=w[:10],name=str(W.uname.get(w,''))[:16],train_pnl=round(CAND.loc[w,'pnl']),train_t=round(CAND.loc[w,'t'],2),moved=round(CAND.loc[w,'moved'],2),imp_mean=round(CAND.loc[w,'imp_mean'],3),test_fills=int(s.fills),test_roi=round(s.roi,2) if s.fills else np.nan,test_pnl=round(s.pnl),roi_at_10min=round(s10.roi,2) if s10.fills else np.nan))
print("\nlowest-impact alpha wallets (train), copied fill-by-fill in the test period:"); print(pd.DataFrame(rows).to_string(index=False))
