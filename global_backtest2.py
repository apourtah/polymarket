"""Global cities v2: primary model per city = the regional/global model with the lowest raw MAE on May 25 - Jun 14 (before the
test window), EWMA-corrected; ridge pooled on primary + differences to every other available model + ECMWF fields.
Prices from the precomputed 21:35-local mids (data/global/mids2135.parquet) -> light on memory. Same legs as production."""
import pandas as pd, numpy as np, re, warnings, os
from scipy.stats import norm
from sklearn.linear_model import Ridge
warnings.filterwarnings('ignore'); pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
START=pd.Timestamp(os.environ.get('START','2026-06-15')); END=pd.Timestamp(os.environ.get('END','2026-09-18')); STAKE=50; SLIP=0.01; MIN_PX=0.05; CAL_END=pd.Timestamp('2026-06-14')
P=pd.read_parquet('data/global/panel.parquet'); P['day']=pd.to_datetime(P.day); R=pd.read_parquet('data/global/regional.parquet'); R['day']=pd.to_datetime(R.day)
P=P.merge(R,on=['city','day'],how='left'); MODELS=['ecmwf','gfs','icon']+[c for c in R.columns if c not in ('city','day')]
# primary per city: lowest MAE in the calibration window among models with >=15 days there
prim={}
for c,g in P[P.day<=CAL_END].groupby('city'):
    best=None
    for m in MODELS:
        v=(g.actual-g[m]).dropna()
        if len(v)>=15 and (best is None or v.abs().mean()<best[1]): best=(m,v.abs().mean())
    prim[c]=best[0] if best else 'ecmwf'
print("primary model per city:", prim)
P['prim']=[r[prim.get(c,'ecmwf')] if pd.notna(r[prim.get(c,'ecmwf')]) else r['ecmwf'] for c,(_,r) in zip(P.city,P.iterrows())]; P['hrrr']=P.prim; P['err']=P.actual-P.prim
for m in MODELS: P['d_'+m]=(P[m]-P.prim).fillna(0)
P=P.sort_values(['city','day']).reset_index(drop=True); P['doy']=P.day.dt.dayofyear; P['dpd']=P.prim-P.dew
P['yday_max']=P.groupby('city').actual.shift(1); P['yday_err']=P.groupby('city').err.shift(1); P['e2']=P.groupby('city').err.shift(2)
P['r5']=P.groupby('city').err.transform(lambda s: s.shift(1).rolling(5,min_periods=3).mean()); P['r14']=P.groupby('city').err.transform(lambda s: s.shift(1).rolling(14,min_periods=5).mean())
FEATS=['hrrr','dew','rhum','wind','cloud','rad','pres','precip','dpd','t_morn','cloud_morn','yday_max','yday_err','e2','r5','r14','doy']+['d_'+m for m in MODELS]
GAINS=[0.05,0.1,0.2,0.3,0.5,0.7]
class Models:
    def __init__(s,hist):
        s.hist=hist.dropna(subset=['err']).copy(); s.cities=sorted(s.hist.city.unique()); s.ewma={}; s.ewma_sd={}; s.ridge_sd={}
        for c in s.cities:
            e=s.hist[s.hist.city==c].sort_values('day').err.values; best=None
            for k in GAINS:
                b=0.0; se=[]
                for x in e: se.append(x-b); b+=k*(x-b)
                v=float(np.sum(np.square(se)))
                if best is None or v<best[0]: best=(v,k,b,np.std(se[-60:]))
            s.ewma[c]=(best[1],best[2]); s.ewma_sd[c]=max(best[3],0.6)
        X=s.hist[FEATS].copy(); s.med=X.median().fillna(0); X=X.fillna(s.med); s.mu=X.mean(); s.sd=X.std().replace(0,1)
        Z=np.c_[((X-s.mu)/s.sd).values,pd.get_dummies(s.hist.city).reindex(columns=s.cities,fill_value=0).values]; s.ridge=Ridge(alpha=20.0).fit(Z,s.hist.err.values)
        res=s.hist.err.values-s.ridge.predict(Z)
        for c in s.cities: s.ridge_sd[c]=max(float(np.std(res[(s.hist.city==c).values][-60:])),0.6)
    def predict(s,city,f):
        x=pd.Series({k:f.get(k,np.nan) for k in FEATS}).fillna(s.med); z=np.r_[((x-s.mu)/s.sd).values,[1.0 if c==city else 0.0 for c in s.cities]]
        return dict(ewma=s.ewma.get(city,(0,0.0))[1],ridge=float(s.ridge.predict(z[None,:])[0]),ewma_sd=s.ewma_sd.get(city,1.5),ridge_sd=s.ridge_sd.get(city,1.5))
def probs(mu,sd,buckets): return {b: norm.cdf((min(b[1],200)+0.5-mu)/sd)-norm.cdf((max(b[0],-200)-0.5-mu)/sd) for b in buckets}
def bucket(q):
    mm=re.search(r'be (-?\d+)°C or below|be (-?\d+)°C or higher|be (-?\d+)°C',q)
    if mm.group(1): return (-999,int(mm.group(1)))
    if mm.group(2): return (int(mm.group(2)),999)
    return (int(mm.group(3)),int(mm.group(3)))
M2=pd.read_parquet('data/global/mids2135.parquet'); M2=M2[M2.closed&M2.yes_res.isin([0.0,1.0])]; M2['b']=M2.question.map(bucket)
mk={k:{r.b:(r.p2135,r.yes_res==1.0) for r in g.itertuples()} for k,g in M2.groupby(['city','day'])}
rows=[]; days=sorted(P[(P.day>=START)&(P.day<=END)].day.unique())
for d in days:
    H=P[P.day<d]
    if H.err.notna().sum()<200: continue
    M=Models(H)
    for c in sorted(P.city.unique()):
        r=P[(P.city==c)&(P.day==d)]
        if r.empty or (c,d) not in mk or pd.isna(r.iloc[0].actual) or pd.isna(r.iloc[0].prim): continue
        r=r.iloc[0]; p=M.predict(c,r.to_dict()); mu_e,mu_r=r.prim+p['ewma'],r.prim+p['ridge']; pr={b:v[0] for b,v in mk[(c,d)].items()}
        if len(pr)<3: continue
        Pe=probs(mu_e,p['ewma_sd'],list(pr)); Pr=probs(mu_r,p['ridge_sd'],list(pr)); be=max(Pe,key=Pe.get); br=max(Pr,key=Pr.get); fav=max(pr,key=pr.get)
        for b in pr: rows.append(dict(city=c,day=d,lo=b[0],hi=b[1],price=pr[b],pe=Pe[b],pr=Pr[b],won=mk[(c,d)][b][1],agree=be==br,be=be[0],br=br[0],fav=fav[0],fav_p=pr[fav],actual=r.actual))
B=pd.DataFrame(rows); B.to_parquet('out/global_buckets_v2.parquet'); n=B.drop_duplicates(['city','day'])
print(f"\n{len(n)} city-nights, {n.city.nunique()} cities; agreement {n.agree.mean():.0%}; EWMA bucket hit {(n.be==n.actual).mean():.0%}, ridge {(n.br==n.actual).mean():.0%}, market favorite {(n.fav==n.actual).mean():.0%} (fav price {n.fav_p.mean():.2f})")
print("hit rates by city (ewma / ridge / favorite):"); hc=n.assign(e=n.be==n.actual,r=n.br==n.actual,f=n.fav==n.actual).groupby('city')[['e','r','f']].mean().round(2); print(hc.T.to_string())
B['month']=B.day.dt.to_period('M').astype(str); B['is_be']=B.lo==B.be; B['is_br']=B.lo==B.br; B['is_fav']=B.lo==B.fav
def pnl(x,side):
    ask=np.where(side=='yes',x.price+SLIP,1-x.price+SLIP).clip(0.01,0.99); sh=STAKE/ask; fee=0.05*ask*(1-ask)*sh; win=np.where(side=='yes',x.won,~x.won); return np.where(win,sh-STAKE,-STAKE)-fee
def legs(X,modes=None):
    ok=lambda c,l: True if modes is None else l in modes.get(c,set())
    Y=pd.concat([X[X.agree&X.is_be&X.price.between(0.10,0.53)&X.city.map(lambda c: ok(c,'agree'))].assign(leg='agree'),X[~X.agree&X.is_br&X.price.between(MIN_PX,0.45)&X.city.map(lambda c: ok(c,'ridge'))].assign(leg='ridge'),X[~X.agree&X.is_be&X.price.between(MIN_PX,0.45)&X.city.map(lambda c: ok(c,'ewma'))].assign(leg='ewma')]).assign(side='yes').sort_values('price').drop_duplicates(['city','day'])
    yk=Y.set_index(['city','day']); X=X.assign(yes_lo=[yk.lo.get((c,d),np.nan) for c,d in zip(X.city,X.day)],yleg=[yk.leg.get((c,d)) for c,d in zip(X.city,X.day)])
    dis=X[X.yleg.isin(['ridge','ewma'])]; N=pd.concat([dis[dis.is_fav&(dis.lo>dis.yes_lo)],dis[(dis.yleg=='ridge')&(dis.lo==dis.br+1)]]).drop_duplicates(['city','day','lo']); N=N[N.price.between(0.35,0.55)].assign(leg='no_H',side='no')
    T=pd.concat([Y,N]); T['pnl']=np.where(T.side=='yes',pnl(T,'yes'),pnl(T,'no')); return T
def summ(T):
    if len(T)==0: return pd.Series(dict(n=0,win=np.nan,pnl=0,roi=np.nan,neg_m=np.nan,worst_m=np.nan))
    m=T.groupby('month').pnl.sum(); return pd.Series(dict(n=len(T),win=(T.pnl>0).mean(),pnl=T.pnl.sum(),roi=T.pnl.sum()/(STAKE*len(T)),neg_m=(m<0).sum(),worst_m=m.min()))
T=legs(B); print("\n### every leg everywhere:"); print(pd.DataFrame({k:summ(T[T.leg==k]) for k in ['agree','ridge','ewma','no_H']}|{'ALL':summ(T)}).T.round(2).to_string()); print("by month:", T.groupby('month').pnl.sum().round(0).to_dict())
tr=legs(B[B.day<='2026-08-01']); sel=tr.groupby(['city','leg']).pnl.agg(['size','sum']); modes={}
for (c,l),r in sel.iterrows():
    if l!='no_H' and r['size']>=8 and r['sum']>0: modes.setdefault(c,set()).add(l)
te=legs(B[B.day>'2026-08-01'],modes); print("\n### out-of-sample (legs selected Jun 15-Aug 1, tested Aug 2-Sep 18):"); print(pd.DataFrame({k:summ(te[te.leg==k]) for k in ['agree','ridge','ewma','no_H']}|{'ALL':summ(te)}).T.round(2).to_string())
print("test by city (top/bottom):"); tc=te.groupby('city').pnl.agg(['size','sum']).assign(roi=lambda d:d['sum']/(STAKE*d['size'])).round(2).sort_values('sum',ascending=False); print(pd.concat([tc.head(8),tc.tail(6)]).to_string())
# YES legs by entry price band and by whether our pick is the favorite (as in the US analysis)
Y=T[T.side=='yes']; print("\nYES legs by price band:"); print(Y.groupby(pd.cut(Y.price,[0,.1,.2,.3,.4,.53]),observed=True).pnl.agg(['size','sum']).assign(roi=lambda d:d['sum']/(STAKE*d['size'])).round(2).to_string())
print("YES legs, pick = favorite vs not:"); print(Y.groupby(Y.is_fav).pnl.agg(['size','sum']).assign(roi=lambda d:d['sum']/(STAKE*d['size'])).round(2).to_string())
