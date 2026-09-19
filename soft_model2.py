"""Soft gates v2: keep the production LEG STRUCTURE (which bucket, which city, YES-then-NO pairing), replace every hard price
band by a trained Gaussian gate. Per leg we fit P(win | price, model P's, favorite price, city) as a logistic regression with
quadratic terms (= product of Gaussian gates in log-odds), walk-forward (monthly refit on all earlier nights), then trade on
expected value: EV = P/ask - 1 >= EV_MIN (YES) or (1-P)/ask_no - 1 >= EV_MIN (NO). So a 34c agree pick with strong P still
trades, a 36c pick with weak P does not, and the bands are learned rather than set.
Legs: agree (agree cities), dis_ridge (ridge cities), dis_ewma (ewma cities); NO_A = favorite warmer than the YES pick,
NO_B = bucket 1 warmer than the ridge pick (ridge cities) — both only on nights with a dis_* YES candidate (as production).
Compared with the hard rule (agree 10-50, model 5-60, NO 35-55) on the same months.  env: TEST_FROM, C_REG, EV_YES, EV_NO"""
import pandas as pd, numpy as np, warnings, os
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore'); pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
STAKE=50.0; SLIP=0.01; TEST_FROM=os.environ.get('TEST_FROM','2026-03'); CREG=float(os.environ.get('C_REG','1.0'))
B=pd.read_parquet('out/backtest_evening_buckets.parquet'); B['mday']=pd.to_datetime(B.mday); B['month']=B.mday.dt.to_period('M').astype(str)
MODES={"Los Angeles":{"agree","ewma"},"Austin":{"agree","ewma"},"Chicago":{"agree","ridge"},"Houston":{"ridge"},"Dallas":{"ridge"},"Seattle":{"agree"},"Miami":{"agree"}}
B=B[(B.lo>-999)&(B.hi<999)].copy(); B['pav']=(B.pe+B.pr)/2; B['is_fav']=(B.lo==B.fav).astype(int)
B['mode_agree']=B.city.map(lambda c: int('agree' in MODES[c])); B['mode_ridge']=B.city.map(lambda c: int('ridge' in MODES[c])); B['mode_ewma']=B.city.map(lambda c: int('ewma' in MODES[c]))
# ---- candidates per leg (structure only, no price bands) ----
C={}
C['agree']=B[B.agree&(B.lo==B.be)&(B.mode_agree==1)].copy()
C['dis_ridge']=B[~B.agree&(B.lo==B.br)&(B.mode_ridge==1)].copy()
C['dis_ewma']=B[~B.agree&(B.lo==B.be)&(B.mode_ewma==1)].copy()
ypick=pd.concat([C['dis_ridge'].assign(leg='dis_ridge'),C['dis_ewma'].assign(leg='dis_ewma')]).drop_duplicates(['city','mday'])
yk=ypick.set_index(['city','mday']); B['ypick_lo']=[yk.lo.get((c,d),np.nan) for c,d in zip(B.city,B.mday)]; B['ypick_px']=[yk.price.get((c,d),np.nan) for c,d in zip(B.city,B.mday)]; B['ypick_leg']=[yk.leg.get((c,d)) for c,d in zip(B.city,B.mday)]
noA=B[B.ypick_lo.notna()&(B.is_fav==1)&(B.lo>B.ypick_lo)]; noB=B[(B.ypick_leg=='dis_ridge')&(B.lo==B.br+2)]
C['no']=pd.concat([noA,noB]).drop_duplicates(['city','mday','lo']).copy()
for k,v in C.items(): v['leg']=k; v['side']='no' if k=='no' else 'yes'
HARD={'agree':(0.10,0.50),'dis_ridge':(0.05,0.60),'dis_ewma':(0.05,0.60),'no':(0.35,0.55)}
def pnl(x):
    ask=np.where(x.side=='yes',x.price+SLIP,1-x.price+SLIP).clip(0.01,0.99); sh=STAKE/ask; fee=0.05*ask*(1-ask)*sh
    win=np.where(x.side=='yes',x.won,~x.won); return np.where(win,sh-STAKE,-STAKE)-fee
def summ(x):
    if len(x)==0: return pd.Series(dict(n=0,win=np.nan,px=np.nan,pnl=0,roi=np.nan,neg_m=np.nan,worst_m=np.nan))
    p=pd.Series(pnl(x),index=x.index); m=p.groupby(x.month).sum(); win=np.where(x.side=='yes',x.won,~x.won)
    return pd.Series(dict(n=len(x),win=win.mean(),px=np.where(x.side=='yes',x.price,1-x.price).mean(),pnl=p.sum(),roi=p.sum()/(STAKE*len(x)),neg_m=(m<0).sum(),worst_m=m.min()))
# ---- per-leg soft model ----
def design(X,leg):
    F=pd.DataFrame(index=X.index)
    for c in ['price','pav','pe','pr','fav_p']: F[c]=X[c]; F[c+'^2']=X[c]**2
    F['pav*price']=X.pav*X.price; F['is_fav']=X.is_fav
    if leg=='no': F['ypick_px']=X.ypick_px; F['ypick_px^2']=X.ypick_px**2; F['gap']=(X.lo-X.ypick_lo)/2; F['is_ridge_night']=(X.ypick_leg=='dis_ridge').astype(int)
    for c in sorted(MODES): F['city='+c]=(X.city==c).astype(int)
    return F
def fit(tr,leg):
    F=design(tr,leg); mu=F.mean(); sd=F.std().replace(0,1); m=LogisticRegression(C=CREG,max_iter=3000).fit(((F-mu)/sd).values,tr.won.values.astype(int)); return m,mu,sd
months=sorted(B.month.unique()); test_months=[m for m in months if m>=TEST_FROM]
for k,v in C.items():
    v['P']=np.nan
    for m in test_months:
        tr=v[v.month<m]; te=v.month==m
        if len(tr)<60 or tr.won.nunique()<2: continue
        model,mu,sd=fit(tr,k); v.loc[te,'P']=model.predict_proba(((design(v[te],k)-mu)/sd).values)[:,1]
A=pd.concat(C.values()); T=A[A.P.notna()].copy(); T['ev']=np.where(T.side=='yes',T.P/(T.price+SLIP)-1,(1-T.P)/(1-T.price+SLIP)-1)
print(f"test months {T.month.min()}..{T.month.max()}; candidates per leg:", T.leg.value_counts().to_dict())
print("\ncalibration per leg (P bins):")
for k in C:
    t=T[T.leg==k]; t=t.assign(pb=pd.cut(t.P,[0,.2,.3,.4,.5,.6,.7,1])); print(f"-- {k}"); print(t.groupby('pb',observed=True).agg(n=('won','size'),pred=('P','mean'),actual=('won','mean'),price=('price','mean')).round(2).T.to_string())
def hard(T): return pd.concat([T[(T.leg==k)&T.price.between(*HARD[k])] for k in HARD])
def soft(T,ev_yes,ev_no,no_needs_yes=True):
    y=T[(T.side=='yes')&(T.ev>=ev_yes)]; n=T[(T.side=='no')&(T.ev>=ev_no)]
    if no_needs_yes: yn=set(zip(y.city,y.mday)); n=n[[k in yn for k in zip(n.city,n.mday)]]
    n=n.sort_values('ev',ascending=False).drop_duplicates(['city','mday']); return pd.concat([y,n])
H=hard(T); H=pd.concat([H[H.side=='yes'],H[H.side=='no'][[k in set(zip(H[H.side=='yes'].city,H[H.side=='yes'].mday)) for k in zip(H[H.side=='no'].city,H[H.side=='no'].mday)]]])
print("\n### HARD rule on test months:"); print(pd.DataFrame({k:summ(H[H.leg==k]) for k in HARD}|{'ALL':summ(H)}).T.round(2).to_string())
print("\n### SOFT (per-leg gates, EV threshold) sweep:")
rows=[]
for ey in [0.05,0.10,0.15,0.20,0.25,0.30]:
    for en in [0.05,0.10,0.15,0.20]:
        S=soft(T,ey,en); s=summ(S); rows.append(pd.Series({**{f"n_{k}":(S.leg==k).sum() for k in HARD},'roi_yes':summ(S[S.side=='yes']).roi,'roi_no':summ(S[S.side=='no']).roi,'n':s.n,'pnl':s.pnl,'roi':s.roi,'neg_m':s.neg_m,'worst_m':s.worst_m},name=f"EV yes>={ey:.2f} no>={en:.2f}"))
R=pd.DataFrame(rows).round(2); print(R.to_string())
ey=float(os.environ.get('EV_YES', R.iloc[(R.n-len(H)).abs().argsort()[:1]].index[0].split('yes>=')[1].split()[0])); en=float(os.environ.get('EV_NO','0.15'))
S=soft(T,ey,en); print(f"\n### like-for-like: soft EV yes>={ey} no>={en} ({len(S)} trades) vs hard ({len(H)} trades)")
print(pd.DataFrame({'hard':summ(H),'soft':summ(S)}).T.round(2).to_string())
print("per leg:"); print(pd.DataFrame({f"hard {k}":summ(H[H.leg==k]) for k in HARD}|{f"soft {k}":summ(S[S.leg==k]) for k in HARD}).T.round(2).to_string())
bm=lambda x: pd.Series(pnl(x),index=x.index).groupby(x.month).sum().round(0); print("by month:"); print(pd.DataFrame({'hard':bm(H),'soft':bm(S)}).T.to_string())
print("by city:"); print(pd.DataFrame({'hard':H.assign(p=pnl(H)).groupby('city').p.sum(),'soft':S.assign(p=pnl(S)).groupby('city').p.sum()}).round(0).T.to_string())
kh=set(zip(H.city,H.mday,H.lo,H.side)); ks=set(zip(S.city,S.mday,S.lo,S.side)); ho=H[[k not in ks for k in zip(H.city,H.mday,H.lo,H.side)]]; so=S[[k not in kh for k in zip(S.city,S.mday,S.lo,S.side)]]
print(f"\noverlap: both {len(kh&ks)}  hard-only {len(kh-ks)}  soft-only {len(ks-kh)}"); print("hard-only (soft skips them):", summ(ho).round(2).to_dict()); print("soft-only (hard skips them):", summ(so).round(2).to_dict())
print("soft-only by leg x price:"); print(so.groupby(['leg',pd.cut(so.price,[0,.1,.2,.3,.35,.4,.5,.55,.6,.7])],observed=True).apply(summ)[['n','win','pnl','roi']].round(2).to_string())
print("hard-only by leg x price:"); print(ho.groupby(['leg',pd.cut(ho.price,[0,.1,.2,.3,.35,.4,.5,.55,.6,.7])],observed=True).apply(summ)[['n','win','pnl','roi']].round(2).to_string())
# near misses of the hard bands
print("\n### hard-band near-misses (within 3c outside the band): what soft does with them")
for k,(lo,hi) in HARD.items():
    v=T[(T.leg==k)&(T.price.between(lo-0.03,lo-1e-9)|T.price.between(hi+1e-9,hi+0.03))]
    if k=='no': v=v[[c in set(zip(S[S.side=='yes'].city,S[S.side=='yes'].mday)) for c in zip(v.city,v.mday)]]
    ev=en if k=='no' else ey; take=v[v.ev>=ev]; print(f"  {k:<9} near-misses n={len(v):3d} if all bought: roi {summ(v).roi if len(v) else float('nan'):+.2f} | soft takes {len(take):3d}: roi {summ(take).roi if len(take) else float('nan'):+.2f} | soft leaves {len(v)-len(take):3d}: roi {summ(v[v.ev<ev]).roi if len(v)-len(take) else float('nan'):+.2f}")
# trained gates (final fit)
print("\n### trained price gates per leg (final fit, log-odds centre & width on raw price scale; other features at their mean)")
for k in HARD:
    v=C[k]; model,mu,sd=fit(v[v.month<test_months[-1]],k); coef=pd.Series(model.coef_[0],index=design(v.head(1),k).columns); a,b=coef['price']/sd['price'],coef['price^2']/sd['price^2']
    # log-odds as a function of raw price: a*x + b*x^2 (+ const) -> peak at -a/2b if b<0
    if b<0: print(f"  {k:<9} centre {-a/(2*b):.3f}  sigma {np.sqrt(-1/(2*b)):.3f}   hard band {HARD[k]}")
    else: print(f"  {k:<9} no interior optimum in price (b={b:+.2f}); hard band {HARD[k]}")
S.drop(columns=[c for c in S.columns if str(S[c].dtype).startswith('category')]).to_parquet('out/soft2_trades.parquet')
