"""Can we re-identify an alpha trader who rotates wallets, from one day of behaviour alone? And is following them worth it?
1. Daily behavioural fingerprint per (wallet, day): local-hour bands, day-before share, long share, YES-price bins, city mix,
   clip-size signature, activity level.  2. Re-ID test: hide the wallet id of an alpha wallet's day, match its footprint
   (full day / first 5 fills) against references built from the wallet's OTHER days, rank among every wallet active that day.
3. Hand-off search: wallets that stopped -> best-matching wallets that started afterwards.  4. Copy-trade backtest,
   train Aug 7-27 (pick the alpha set) / test Aug 28-Sep 18: S1 static id list, S2 yesterday's top-10 ids, S3 fingerprint-
   matched wallets (id-agnostic), S4 fill-level 'informed flow' model (no identity). Copy at the next print in that market
   >= 60 s after their fill (+ fee); S1/S2 die under daily rotation, S3/S4 do not."""
import pandas as pd, numpy as np, warnings
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore'); pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
T=pd.read_parquet('out/wallet_fills.parquet').sort_values('ts'); W=pd.read_csv('out/wallets_all.csv',index_col=0)
T['band']=pd.cut(T.lhour,[-1,3,6,9,12,15,18,21,24],labels=False); T['pbin']=pd.cut(T.p_yes,[0,.1,.25,.45,.65,1],labels=False)
T['szint']=(T['size']%1==0).astype(float); T['szr5']=(T['size']%5==0).astype(float); T['lsz']=np.log1p(T['size']); T['long']=(T.q>0).astype(float); T['before']=(T.rel_day<0).astype(float)
CITIES=sorted(T.city.unique()); T['ci']=T.city.map({c:i for i,c in enumerate(CITIES)})
def fp(g):
    """fingerprint vector of a set of fills"""
    v=[np.bincount(g.band.astype(int),minlength=8)/len(g), np.bincount(g.pbin.fillna(2).astype(int),minlength=5)/len(g), np.bincount(g.ci,minlength=len(CITIES))/len(g),
       [g.long.mean(), g.before.mean(), g.szint.mean(), g.szr5.mean(), g.lsz.median()/6, np.log1p(len(g))/6, np.log1p(g.cid.nunique())/5]]
    return np.concatenate(v)
WEIGHTS=np.concatenate([np.full(8,1.0),np.full(5,1.0),np.full(len(CITIES),0.7),[1.5,1.5,1.0,1.0,1.5,0.5,0.5]])
def sim(a,b): a=a*WEIGHTS; b=b*WEIGHTS; return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
A=W[(W.markets>=10)&(W.days>=8)&(W.pnl>1000)].index.tolist(); print(f"alpha set (all-window, for the re-ID test): {len(A)} wallets")
days=sorted(T.day.unique())
# ---------- 1-2. re-identification, leave-one-day-out ----------
FP={}   # (w,day) -> full-day fingerprint ; FP5 -> first-5-fills fingerprint
for (w,d),g in T.groupby(['w','day']):
    if len(g)>=3: FP[(w,d)]=fp(g); 
for (w,d),g in T[T.w.isin(A)].groupby(['w','day']):
    FP[(w,d,'e')]=fp(g.head(5))
res=[]
for w in A:
    wd=[d for d in days if (w,d) in FP]
    for d in wd:
        ref=np.mean([FP[(w,x)] for x in wd if x!=d],axis=0)
        cands=[(x,FP[(x,d)]) for x in T[T.day==d].w.unique() if (x,d) in FP]
        for mode,q in [('full',FP[(w,d)]),('first5',FP.get((w,d,'e')))]:
            if q is None: continue
            s={x:sim(ref,v) for x,v in cands}; s[w]=sim(ref,q)   # query replaces its own full-day entry
            order=sorted(s.items(),key=lambda kv:-kv[1]); rank=[x for x,_ in order].index(w)+1
            res.append(dict(w=w,day=d,mode=mode,rank=rank,n_cands=len(order),sim_true=s[w],sim_best_other=max(v for x,v in s.items() if x!=w)))
R=pd.DataFrame(res); print("\n### re-ID of alpha wallets from ONE day of behaviour (identity hidden), ranked among all wallets active that day")
print(R.groupby('mode').agg(days=('rank','size'),top1=('rank',lambda r:(r==1).mean()),top3=('rank',lambda r:(r<=3).mean()),top10=('rank',lambda r:(r<=10).mean()),median_rank=('rank','median'),avg_cands=('n_cands','mean')).round(3).to_string())
per=R[R['mode']=='full'].groupby('w').agg(days=('rank','size'),top1=('rank',lambda r:(r==1).mean()),med_rank=('rank','median'),sim=('sim_true','mean')).join(W[['pnl','uname']]).sort_values('pnl',ascending=False)
print("\nper wallet (full-day footprint):"); print(per.round(2).head(25).to_string())
# threshold-based detection: precision/recall at similarity >= tau against ANY alpha reference (open-set)
refs={w:np.mean([FP[(w,x)] for x in days if (w,x) in FP],axis=0) for w in A}
rows=[]
for tau in [0.90,0.93,0.95,0.97]:
    tp=fp_=fn=0
    for (k,v) in FP.items():
        if len(k)==3: continue
        w,d=k; best=max(refs.items(),key=lambda kv: sim(kv[1],v)); hit=sim(best[1],v)>=tau
        if w in A: tp+=hit and best[0]==w; fn+=not (hit and best[0]==w)
        else: fp_+=hit
    rows.append(dict(tau=tau,recall=tp/(tp+fn),false_alarms=fp_,alpha_days=tp+fn,precision=tp/max(tp+fp_,1)))
print("\nopen-set detection: 'this unknown wallet-day IS alpha wallet X' if similarity >= tau"); print(pd.DataFrame(rows).round(3).to_string(index=False))
# ---------- 3. hand-off search ----------
print("\n### hand-off search: wallets with pnl>500 that stopped before Sep 10 -> best-matching wallets that started after they stopped")
stop=W[(W['last']<'2026-09-10')&(W.pnl>500)&(W.markets>=8)]
for w in stop.index:
    ref=np.mean([FP[(w,x)] for x in days if (w,x) in FP],axis=0); after=W[(W['first']>stop.loc[w,'last'])&(W.fills>=10)].index
    cand=[(x,sim(ref,np.mean([FP[(x,d)] for d in days if (x,d) in FP],axis=0))) for x in after if any((x,d) in FP for d in days)]
    cand=sorted(cand,key=lambda kv:-kv[1])[:3]; print(f"  {w[:10]} ({stop.loc[w,'uname'] if isinstance(stop.loc[w,'uname'],str) else ''}) last {stop.loc[w,'last']} pnl {stop.loc[w,'pnl']:+.0f} ->", [(x[:10],round(s,3),W.loc[x,'first'],round(W.loc[x,'pnl'])) for x,s in cand])
# ---------- 4. copy-trade backtest ----------
TRAIN_END='2026-08-27'; tr=T[T.day<=TRAIN_END]; te=T[T.day>TRAIN_END]
Wtr=tr.groupby('w').agg(pnl=('pnl','sum'),usd=('usd','sum'),mk=('cid','nunique'),days=('day','nunique'))
dd=tr.groupby(['w','day']).pnl.sum().groupby('w').agg(['mean','std','count']); Wtr['t']=dd['mean']/(dd['std']/np.sqrt(dd['count']))
ALPHA=Wtr[(Wtr.mk>=8)&(Wtr.days>=6)&(Wtr.pnl>300)&(Wtr.t>1.0)].sort_values('pnl',ascending=False).head(15).index.tolist()
print(f"\n### copy-trade backtest. train {tr.day.min()}..{TRAIN_END}: alpha set = {len(ALPHA)} wallets:", [ (W.loc[w,'uname'] if isinstance(W.loc[w,'uname'],str) and not str(W.loc[w,'uname']).startswith('0x') else w[:8]) for w in ALPHA])
# execution: copy at the next print in the same market >= 60 s later (same YES-equivalent direction sign of price), fee 5%*p*(1-p)
Tm=te.sort_values(['cid','ts']); nxt={}
for cid,g in Tm.groupby('cid'):
    ts=g.ts.values; py=g.p_yes.values; idx=np.searchsorted(ts,ts+60,side='left'); ok=idx<len(ts)
    nxt.update(dict(zip(g.index,np.where(ok,py[np.minimum(idx,len(ts)-1)],np.nan))))
te=te.assign(p_copy=pd.Series(nxt))
def copy_pnl(F,stake=10):
    F=F.dropna(subset=['p_copy']); long=F.q>0; ask=np.where(long,F.p_copy,1-F.p_copy).clip(0.01,0.99); sh=stake/ask; fee=0.05*ask*(1-ask)*sh
    win=np.where(long,F.won,~F.won); pnl=np.where(win,sh-stake,-stake)-fee
    F=F.assign(cpnl=pnl); per_mkt=F.groupby(['w','cid']).cpnl.mean()   # one $10 clip per wallet-market (their first fill decides)
    first=F.sort_values('ts').drop_duplicates(['w','cid']); ask1=np.where(first.q>0,first.p_copy,1-first.p_copy).clip(0.01,0.99); sh1=stake/ask1; fee1=0.05*ask1*(1-ask1)*sh1; win1=np.where(first.q>0,first.won,~first.won); p1=np.where(win1,sh1-stake,-stake)-fee1
    days=pd.Series(p1,index=first.index).groupby(first.day.values).sum()
    return pd.Series(dict(clips=len(first),usd=stake*len(first),pnl=p1.sum(),roi=p1.sum()/(stake*len(first)) if len(first) else np.nan,win=win1.mean() if len(first) else np.nan,neg_days=(days<0).sum(),days=len(days)))
out={}
out['S1 static alpha ids (Aug 28-Sep 18)']=copy_pnl(te[te.w.isin(ALPHA)])
# S2: yesterday's top-10 ids by P&L
picks=[]
for i,d in enumerate(days):
    if d<=TRAIN_END: continue
    prev=T[T.day==days[i-1]].groupby('w').pnl.sum().nlargest(10).index; picks.append(te[(te.day==d)&te.w.isin(prev)])
out['S2 yesterday top-10 ids']=copy_pnl(pd.concat(picks))
# S3: id-agnostic fingerprint match after the wallet's first 5 fills of the day, tau from the open-set test
refs_tr={w:np.mean([FP[(w,x)] for x in days if x<=TRAIN_END and (w,x) in FP],axis=0) for w in ALPHA if any((w,x) in FP for x in days if x<=TRAIN_END)}
for tau in [0.93,0.95,0.97]:
    picks=[]; matched=set()
    for (w,d),g in te.groupby(['w','day']):
        if len(g)<5: continue
        q=fp(g.head(5)); best=max(sim(r,q) for r in refs_tr.values())
        if best>=tau: picks.append(g.iloc[5:]); matched.add((w,d))
    P=pd.concat(picks) if picks else te.iloc[0:0]; s=copy_pnl(P); s['wallet_days']=len(matched); s['of_which_true_alpha']=sum(w in ALPHA for w,d in matched); out[f'S3 fingerprint match tau={tau} (id-agnostic, after 5 fills)']=s
# S4: fill-level informed-flow model without identity: features hour band, rel_day, p_yes, long, size, fills-so-far-today, cities-so-far
def feats(F):
    F=F.sort_values('ts'); F['k']=F.groupby(['w','day']).cumcount(); X=pd.DataFrame({'p':F.p_yes,'p2':F.p_yes**2,'long':F.long,'before':F.before,'lsz':F.lsz,'k':np.log1p(F.k),'szint':F.szint},index=F.index)
    for b in range(8): X[f'h{b}']=(F.band==b).astype(int)
    X['long*p']=F.long*F.p_yes; X['before*p']=F.before*F.p_yes; return X,F
Xtr,Ftr=feats(tr.copy()); ytr=((Ftr.q>0)==Ftr.won).astype(int)          # did the fill's side win?
edge_tr=ytr-np.where(Ftr.q>0,Ftr.p_yes,1-Ftr.p_yes)                    # realised edge vs price paid
m=LogisticRegression(max_iter=2000,C=0.5).fit(Xtr.values,ytr.values); Xte,Fte=feats(te.copy()); pw=m.predict_proba(Xte.values)[:,1]; ask=np.where(Fte.q>0,Fte.p_yes,1-Fte.p_yes)
for e in [0.05,0.10,0.15]:
    sel=Fte[(pw-ask)>=e]; out[f'S4 informed-flow model, predicted edge>={e:.2f}']=copy_pnl(sel)
out['baseline: copy EVERY fill in the test period']=copy_pnl(te.sample(min(len(te),60000),random_state=0))
print("\n$10 per wallet-market clip, copied at the next print >=60 s after their fill, fee included. Test period Aug 28 - Sep 18 (22 days).")
print(pd.DataFrame(out).T.round(2).to_string())
print("\nunder DAILY wallet rotation: S1 and S2 select nothing (new ids are never in the list); S3/S4 are unchanged because they never use the id.")
