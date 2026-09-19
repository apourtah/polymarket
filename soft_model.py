"""Soft-gate model vs the hard-threshold rule.

Idea (user's): replace the stack of hard price thresholds by a product of Gaussian 'gates' g_i(x) = exp(-(x-mu_i)^2 / 2 sigma_i^2)
centred on each feature's ideal value, trained rather than hand-set. In log-odds space a Gaussian gate is a quadratic, so a
logistic regression on (x, x^2) per feature IS a product of trained Gaussian gates: mu_i = -a_i/(2 b_i), sigma_i = sqrt(-1/(2 b_i)).
We fit P(bucket wins | features) that way, walk-forward (refit at each month start on all earlier nights), and trade on
expected value: YES if P/ask - 1 >= EV_MIN, NO if (1-P)/(1-ask) - 1 >= EV_MIN. One calibrated P per bucket drives both legs,
so a 34c pick with strong model support still trades and a 36c pick with weak support does not.

Compared against the production hard rule on the same test months (agree 10-50c in agree cities, model pick 5-60c in
disagree cities, NO leg H at 35-55c). Data: out/backtest_evening_buckets.parquet (7 cities, 21:35 mids, walk-forward EWMA/ridge P)."""
import pandas as pd, numpy as np, warnings, os
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore'); pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
STAKE=50.0; SLIP=0.01; TEST_FROM=os.environ.get('TEST_FROM','2026-03-01')
B=pd.read_parquet('out/backtest_evening_buckets.parquet'); B['mday']=pd.to_datetime(B.mday); B['month']=B.mday.dt.to_period('M').astype(str)
MODES={"Los Angeles":{"agree","ewma"},"Austin":{"agree","ewma"},"Chicago":{"agree","ridge"},"Houston":{"ridge"},"Dallas":{"ridge"},"Seattle":{"agree"},"Miami":{"agree"}}
B=B[(B.lo>-999)&(B.hi<999)].copy()                       # interior buckets only (tails are priced differently)
B['is_be']=(B.lo==B.be).astype(int); B['is_br']=(B.lo==B.br).astype(int); B['is_fav']=(B.lo==B.fav).astype(int); B['agree']=B.agree.astype(int)
B['pav']=(B.pe+B.pr)/2; B['pmin']=B[['pe','pr']].min(axis=1); B['pmax']=B[['pe','pr']].max(axis=1); B['edge']=B.pav-B.price
B['off_be']=(B.lo-B.be)/2; B['off_br']=(B.lo-B.br)/2; B['off_fav']=(B.lo-B.fav)/2
B['agree_pick']=B.agree*B.is_be; B['ridge_pick']=(1-B.agree)*B.is_br; B['ewma_pick']=(1-B.agree)*B.is_be
B['mode_agree']=B.city.map(lambda c: int('agree' in MODES[c])); B['mode_ridge']=B.city.map(lambda c: int('ridge' in MODES[c])); B['mode_ewma']=B.city.map(lambda c: int('ewma' in MODES[c]))
B['rank_px']=B.groupby(['city','mday']).price.rank(ascending=False,method='first')
# ---------- production hard rule ----------
def hard_rule(X):
    y=pd.concat([X[(X.agree==1)&(X.is_be==1)&(X.mode_agree==1)&X.price.between(0.10,0.50)].assign(leg='agree'),
                 X[(X.agree==0)&(X.is_br==1)&(X.mode_ridge==1)&X.price.between(0.05,0.60)].assign(leg='dis_ridge'),
                 X[(X.agree==0)&(X.is_be==1)&(X.mode_ewma==1)&X.price.between(0.05,0.60)].assign(leg='dis_ewma')]).assign(side='yes')
    yk=y.set_index(['city','mday']); ylo=yk.lo.to_dict(); yleg=yk.leg.to_dict()
    X=X.assign(yes_lo=[ylo.get((c,d),np.nan) for c,d in zip(X.city,X.mday)], yleg=[yleg.get((c,d)) for c,d in zip(X.city,X.mday)])
    dis=X[X.yleg.isin(['dis_ridge','dis_ewma'])]
    a=dis[(dis.is_fav==1)&(dis.lo>dis.yes_lo)]; b=dis[(dis.yleg=='dis_ridge')&(dis.lo==dis.br+2)]
    n=pd.concat([a,b]).drop_duplicates(['city','mday','lo']); n=n[n.price.between(0.35,0.55)].assign(side='no',leg='no_H')
    return pd.concat([y,n])
def pnl(x):
    ask=np.where(x.side=='yes',x.price+SLIP,1-x.price+SLIP).clip(0.01,0.99); sh=STAKE/ask; fee=0.05*ask*(1-ask)*sh
    win=np.where(x.side=='yes',x.won,~x.won); return np.where(win,sh-STAKE,-STAKE)-fee
def summ(x):
    if len(x)==0: return pd.Series(dict(n=0,win=np.nan,pnl=0,roi=np.nan,neg_m=np.nan,worst_m=np.nan))
    p=pd.Series(pnl(x),index=x.index); m=p.groupby(x.month).sum(); win=np.where(x.side=='yes',x.won,~x.won)
    return pd.Series(dict(n=len(x),win=win.mean(),pnl=p.sum(),roi=p.sum()/(STAKE*len(x)),neg_m=(m<0).sum(),worst_m=m.min()))
# ---------- soft model: logistic on Gaussian-gate (quadratic) features ----------
CONT=['price','pav','pmin','pmax','edge','fav_p']; DISC=['is_be','is_br','is_fav','agree','agree_pick','ridge_pick','ewma_pick','mode_agree','mode_ridge','mode_ewma']
def design(X):
    F=pd.DataFrame(index=X.index)
    for c in CONT: F[c]=X[c]; F[c+'^2']=X[c]**2
    for c in DISC: F[c]=X[c]
    # interactions that carry the per-city-mode logic: pick-type x mode, pick-type x price, favorite x pick
    F['ridge_pick*mode_ridge']=X.ridge_pick*X.mode_ridge; F['ewma_pick*mode_ewma']=X.ewma_pick*X.mode_ewma; F['agree_pick*mode_agree']=X.agree_pick*X.mode_agree
    F['agree_pick*price']=X.agree_pick*X.price; F['ridge_pick*price']=X.ridge_pick*X.price; F['ewma_pick*price']=X.ewma_pick*X.price
    F['is_fav*price']=X.is_fav*X.price; F['is_fav*pav']=X.is_fav*X.pav; F['off_fav']=X.off_fav; F['off_fav^2']=X.off_fav**2
    F['|off_be|']=X.off_be.abs(); F['|off_br|']=X.off_br.abs()
    for c in sorted(MODES): F['city='+c]=(X.city==c).astype(int)
    return F
def fit(train):
    F=design(train); mu=F.mean(); sd=F.std().replace(0,1); Z=(F-mu)/sd
    m=LogisticRegression(C=float(os.environ.get('C_REG','0.3')),max_iter=2000).fit(Z.values,train.won.values.astype(int)); return m,mu,sd
def predict(m,mu,sd,X): return m.predict_proba(((design(X)-mu)/sd).values)[:,1]
months=sorted(B.month.unique()); test_months=[m for m in months if m>=TEST_FROM[:7]]
B['P']=np.nan
for m in test_months:
    tr=B[B.month<m]; te=B.month==m
    if tr.mday.nunique()<40: continue
    model,mu,sd=fit(tr); B.loc[te,'P']=predict(model,mu,sd,B[te])
T=B[B.P.notna()].copy(); print(f"walk-forward test period {T.month.min()}..{T.month.max()}: {T.groupby(['city','mday']).ngroups} city-nights, {len(T)} priced buckets")
# calibration
T['pbin']=pd.cut(T.P,[0,.1,.2,.3,.4,.5,.6,.7,1]); print("\ncalibration of the soft P (bucket wins):"); print(T.groupby('pbin',observed=True).agg(n=('won','size'),predicted=('P','mean'),actual=('won','mean'),market=('price','mean')).round(3).to_string())
# decisions by expected value
def soft_rule(X,ev_yes,ev_no,yes_max=0.65,no_lo=0.25,no_hi=0.70,one_yes=True):
    y=X[(X.price.between(0.03,yes_max))&(X.P/(X.price+SLIP)-1>=ev_yes)].assign(side='yes',leg='soft_yes',ev=lambda d:d.P/(d.price+SLIP)-1)
    if one_yes: y=y.sort_values('ev',ascending=False).drop_duplicates(['city','mday'])          # at most one YES per city-night (like production)
    n=X[X.price.between(no_lo,no_hi)&((1-X.P)/(1-X.price+SLIP)-1>=ev_no)].assign(side='no',leg='soft_no',ev=lambda d:(1-d.P)/(1-d.price+SLIP)-1)
    n=n.sort_values('ev',ascending=False).drop_duplicates(['city','mday'])
    return pd.concat([y,n])
H=hard_rule(T); print("\n### hard rule (production) on the test period"); print(pd.DataFrame({sd_:summ(H[H.side==sd_]) for sd_ in ['yes','no']}).T.round(2).to_string()); print("total:", summ(H).round(2).to_dict())
print("\n### soft rule: EV threshold sweep (same test period)")
rows=[]
for ev_y in [0.10,0.20,0.30,0.40,0.50]:
    for ev_n in [0.05,0.10,0.15,0.20]:
        S=soft_rule(T,ev_y,ev_n); s=summ(S); sy=summ(S[S.side=='yes']); sn=summ(S[S.side=='no'])
        rows.append(pd.Series(dict(n_yes=sy.n,roi_yes=sy.roi,pnl_yes=sy.pnl,n_no=sn.n,roi_no=sn.roi,pnl_no=sn.pnl,n=s.n,pnl=s.pnl,roi=s.roi,neg_m=s.neg_m,worst_m=s.worst_m),name=f"EV yes>={ev_y:.2f} no>={ev_n:.2f}"))
R=pd.DataFrame(rows).round(2); print(R.to_string())
# pick the setting closest in capital to the hard rule for a like-for-like comparison, and the best ROI
tgt=len(H); best=R.iloc[(R.n-tgt).abs().argsort()[:1]].index[0]; print(f"\nlike-for-like (closest trade count to hard rule's {tgt}): {best}")
ev_y=float(best.split('yes>=')[1].split()[0]); ev_n=float(best.split('no>=')[1]); S=soft_rule(T,ev_y,ev_n)
def bymonth(x): p=pd.Series(pnl(x),index=x.index); return p.groupby(x.month).sum().round(0)
print(pd.DataFrame({'hard':bymonth(H),'soft':bymonth(S)}).T.to_string())
print("\nby city (total P&L):"); print(pd.DataFrame({'hard':H.assign(p=pnl(H)).groupby('city').p.sum(),'soft':S.assign(p=pnl(S)).groupby('city').p.sum()}).round(0).T.to_string())
kh=set(zip(H.city,H.mday,H.lo,H.side)); ks=set(zip(S.city,S.mday,S.lo,S.side))
print(f"\noverlap: hard {len(kh)}  soft {len(ks)}  both {len(kh&ks)}  hard-only {len(kh-ks)}  soft-only {len(ks-kh)}")
ho=H[[k not in ks for k in zip(H.city,H.mday,H.lo,H.side)]]; so=S[[k not in kh for k in zip(S.city,S.mday,S.lo,S.side)]]
print("hard-only trades:", summ(ho).round(2).to_dict()); print("soft-only trades:", summ(so).round(2).to_dict())
print("\nsoft-only YES trades by price / by what they are:"); so_y=so[so.side=='yes']
print(so_y.groupby(pd.cut(so_y.price,[0,.1,.2,.3,.4,.5,.65]),observed=True).apply(summ)[['n','win','pnl','roi']].round(2).to_string())
print(so_y.assign(kind=np.select([so_y.agree_pick==1,so_y.ridge_pick==1,so_y.ewma_pick==1],['agree pick','ridge pick','ewma pick'],'other bucket')).groupby('kind').apply(summ)[['n','win','pnl','roi']].round(2).to_string())
# near-miss test: hard-rule near misses (pick priced within 3c outside its band) -> what does soft do?
nm=pd.concat([T[(T.agree==1)&(T.is_be==1)&(T.mode_agree==1)&(T.price.between(0.50,0.53)|T.price.between(0.07,0.10))],T[(T.agree==0)&((T.is_br==1)&(T.mode_ridge==1)|(T.is_be==1)&(T.mode_ewma==1))&T.price.between(0.60,0.63)]])
nm=nm.assign(side='yes'); print(f"\nhard-rule YES near-misses (pick within 3c outside its band): n={len(nm)}, if bought: {summ(nm).round(2).to_dict()}"); print("  soft model would buy:", int((nm.P/(nm.price+SLIP)-1>=ev_y).sum()), "of them;", summ(nm[nm.P/(nm.price+SLIP)-1>=ev_y]).round(2).to_dict())
# the trained gates: mu/sigma per continuous feature from the final fit (last training window)
model,mu,sd=fit(B[B.month<test_months[-1]]); coef=pd.Series(model.coef_[0],index=design(B.head(1)).columns)
print("\n### trained Gaussian gates (final fit; in standardised units, centre converted back to raw scale)")
for c in CONT:
    a,b=coef[c],coef[c+'^2']
    if b<0: z0=-a/(2*b); print(f"  {c:<6} centre {z0*sd[c]+mu[c]:.3f}  width sigma {np.sqrt(-1/(2*b))*sd[c]:.3f}   (log-odds peak, then fades)")
    else: print(f"  {c:<6} monotone / U-shaped (b={b:+.2f}), no interior optimum")
print("\nlargest coefficients:"); print(coef.abs().sort_values(ascending=False).head(15).round(2).to_string())
S.drop(columns=[c for c in S.columns if str(S[c].dtype).startswith('category')]).to_parquet('out/soft_trades.parquet'); H.drop(columns=[c for c in H.columns if str(H[c].dtype).startswith('category')]).to_parquet('out/hard_trades.parquet')
