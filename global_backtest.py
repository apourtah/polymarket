"""Production rule on the 37 non-US (°C, 1-degree bucket) cities, walk-forward, day-before global models (ECMWF/GFS/ICON via
Open-Meteo previous-runs) with the bot's EWMA + pooled-ridge correction, entry = YES mid at 21:35 local the evening before.
Legs: agree (10-53c), ridge pick / ewma pick on disagreement nights (5-45c), NO leg H (35-55c). Per-city modes are unknown here,
so: (1) every leg in every city, (2) agree-only, (3) modes selected on the first half and tested on the second half.
Data: data/global/panel.parquet, data/global/markets.json, data/global/prices/. env START (2026-06-15) END (2026-09-18)"""
import pandas as pd, numpy as np, json, re, bisect, datetime as dt, os, warnings
from zoneinfo import ZoneInfo
from scipy.stats import norm
from sklearn.linear_model import Ridge
from decimal import Decimal, ROUND_HALF_UP
from sweep_local import TZ
warnings.filterwarnings('ignore'); pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
rh=lambda x: int(Decimal(str(x)).quantize(Decimal('1'),rounding=ROUND_HALF_UP))
START=pd.Timestamp(os.environ.get('START','2026-06-15')); END=pd.Timestamp(os.environ.get('END','2026-09-18')); STAKE=50; SLIP=0.01; MIN_PX=0.05
P=pd.read_parquet('data/global/panel.parquet'); P['day']=pd.to_datetime(P.day); P=P.sort_values(['city','day']).reset_index(drop=True)
P['hrrr']=P.ecmwf; P['err']=P.actual-P.ecmwf; P['doy']=P.day.dt.dayofyear; P['gfs_minus']=P.gfs-P.ecmwf; P['icon_minus']=P.icon-P.ecmwf; P['dpd']=P.ecmwf-P.dew
P['yday_max']=P.groupby('city').actual.shift(1); P['yday_err']=P.groupby('city').err.shift(1); P['e2']=P.groupby('city').err.shift(2)
P['r5']=P.groupby('city').err.transform(lambda s: s.shift(1).rolling(5,min_periods=3).mean()); P['r14']=P.groupby('city').err.transform(lambda s: s.shift(1).rolling(14,min_periods=5).mean())
FEATS=['hrrr','dew','rhum','wind','cloud','rad','pres','precip','dpd','t_morn','cloud_morn','yday_max','yday_err','e2','r5','r14','gfs_minus','icon_minus','doy']
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
ms=json.load(open('data/global/markets.json')); mk={}
for m in ms:
    if not m.get('closed') or not m.get('outcome_prices') or float(m['outcome_prices'][0]) not in (0.0,1.0): continue
    d=pd.Timestamp(m['end_date'][:10]); mk.setdefault((m['city'],d),{})[bucket(m['question'])]=m
cache={}
def yes_price(m,ts):
    mid=m['market_id']
    if mid not in cache:
        try: h=json.load(open(f"data/global/prices/{mid}.json"))['yes']; cache[mid]=([x['t'] for x in h],[x['p'] for x in h])
        except Exception: cache[mid]=None
    s=cache[mid]
    if not s or not s[0]: return None
    i=bisect.bisect_right(s[0],ts)-1
    return s[1][i] if i>=0 and ts-s[0][i]<=3*3600 else None
rows=[]; days=sorted(P[(P.day>=START)&(P.day<=END)].day.unique())
for d in days:
    H=P[(P.day<d)]
    if H.err.notna().sum()<200: continue
    M=Models(H)
    for c in sorted(P.city.unique()):
        r=P[(P.city==c)&(P.day==d)]
        if r.empty or (c,d) not in mk or pd.isna(r.iloc[0].actual): continue
        r=r.iloc[0]; f=r.to_dict(); p=M.predict(c,f); mu_e,mu_r=r.ecmwf+p['ewma'],r.ecmwf+p['ridge']
        bk=mk[(c,d)]; tz=ZoneInfo(TZ[c]); ts=int(dt.datetime(d.year,d.month,d.day,21,35,tzinfo=tz).timestamp())-86400
        pr={b:yes_price(m,ts) for b,m in bk.items()}; pr={b:v for b,v in pr.items() if v is not None}
        if len(pr)<3: continue
        Pe=probs(mu_e,p['ewma_sd'],list(pr)); Pr=probs(mu_r,p['ridge_sd'],list(pr)); be=max(Pe,key=Pe.get); br=max(Pr,key=Pr.get); fav=max(pr,key=pr.get)
        for b in pr:
            rows.append(dict(city=c,day=d,lo=b[0],hi=b[1],price=pr[b],pe=Pe[b],pr=Pr[b],won=[float(x) for x in bk[b]['outcome_prices']]==[1.0,0.0],agree=be==br,be=be[0],br=br[0],fav=fav[0],fav_p=pr[fav],actual=r.actual,mu_e=mu_e,mu_r=mu_r))
B=pd.DataFrame(rows); B.to_parquet('out/global_buckets.parquet'); n=B.drop_duplicates(['city','day'])
print(f"{len(n)} city-nights, {n.city.nunique()} cities, {n.day.min().date()}..{n.day.max().date()}; agreement {n.agree.mean():.0%}; EWMA bucket hit {(n.be==n.actual).mean():.0%}, ridge hit {(n.br==n.actual).mean():.0%}, market favorite hit {(n.fav==n.actual).mean():.0%} (favorite avg price {n.fav_p.mean():.2f})")
B['month']=B.day.dt.to_period('M').astype(str); B['is_be']=B.lo==B.be; B['is_br']=B.lo==B.br; B['is_fav']=B.lo==B.fav
def pnl(x,side):
    ask=np.where(side=='yes',x.price+SLIP,1-x.price+SLIP).clip(0.01,0.99); sh=STAKE/ask; fee=0.05*ask*(1-ask)*sh; win=np.where(side=='yes',x.won,~x.won); return np.where(win,sh-STAKE,-STAKE)-fee
def legs(X,modes=None):
    def ok(c,leg): return True if modes is None else leg in modes.get(c,set())
    ag=X[X.agree&X.is_be&X.price.between(0.10,0.53)&X.city.map(lambda c: ok(c,'agree'))].assign(leg='agree')
    dr=X[~X.agree&X.is_br&X.price.between(MIN_PX,0.45)&X.city.map(lambda c: ok(c,'ridge'))].assign(leg='ridge')
    de=X[~X.agree&X.is_be&X.price.between(MIN_PX,0.45)&X.city.map(lambda c: ok(c,'ewma'))].assign(leg='ewma')
    Y=pd.concat([ag,dr,de]).assign(side='yes'); Y=Y.sort_values('price').drop_duplicates(['city','day'])   # one YES per city-night (cheapest if both model legs qualify)
    yk=Y.set_index(['city','day']); X=X.assign(yes_lo=[yk.lo.get((c,d),np.nan) for c,d in zip(X.city,X.day)],yleg=[yk.leg.get((c,d)) for c,d in zip(X.city,X.day)])
    dis=X[X.yleg.isin(['ridge','ewma'])]; N=pd.concat([dis[dis.is_fav&(dis.lo>dis.yes_lo)],dis[(dis.yleg=='ridge')&(dis.lo==dis.br+1)]]).drop_duplicates(['city','day','lo'])
    N=N[N.price.between(0.35,0.55)].assign(leg='no_H',side='no'); T=pd.concat([Y,N]); T['pnl']=np.where(T.side=='yes',pnl(T,'yes'),pnl(T,'no')); return T
def summ(T):
    if len(T)==0: return pd.Series(dict(n=0,win=np.nan,pnl=0,roi=np.nan,neg_m=np.nan,worst_m=np.nan))
    m=T.groupby('month').pnl.sum(); return pd.Series(dict(n=len(T),win=(T.pnl>0).mean(),pnl=T.pnl.sum(),roi=T.pnl.sum()/(STAKE*len(T)),neg_m=(m<0).sum(),worst_m=m.min()))
T=legs(B); print("\n### every leg in every city (no per-city modes), $50 clips, mids +1c, fee:"); print(pd.DataFrame({k:summ(T[T.leg==k]) for k in ['agree','ridge','ewma','no_H']}|{'ALL':summ(T)}).T.round(2).to_string())
print("\nby month:", T.groupby('month').pnl.sum().round(0).to_dict())
pc=T.groupby(['city','leg']).pnl.agg(['size','sum']); pc['roi']=pc['sum']/(STAKE*pc['size']); print("\nper city x leg (n, P&L, ROI):"); print(pc.round(2).unstack('leg').to_string())
print("\n### honest out-of-sample: choose each city's legs on Jun 15 - Aug 1 (keep legs with ROI>0 and n>=8), test Aug 2 - Sep 18")
tr=legs(B[B.day<='2026-08-01']); sel=tr.groupby(['city','leg']).pnl.agg(['size','sum']); modes={}
for (c,l),r in sel.iterrows():
    if l!='no_H' and r['size']>=8 and r['sum']>0: modes.setdefault(c,set()).add(l)
print("selected:", {c:sorted(v) for c,v in modes.items()})
te=legs(B[B.day>'2026-08-01'],modes); print(pd.DataFrame({k:summ(te[te.leg==k]) for k in ['agree','ridge','ewma','no_H']}|{'ALL':summ(te)}).T.round(2).to_string())
print("test by city:"); print(te.groupby('city').pnl.agg(['size','sum']).assign(roi=lambda d:d['sum']/(STAKE*d['size'])).round(2).sort_values('sum',ascending=False).to_string())
print("\nfor reference, same test window with all legs everywhere:", summ(legs(B[B.day>'2026-08-01'])).round(2).to_dict())
print("agree-only everywhere, whole period:", summ(T[T.leg=='agree']).round(2).to_dict(), "| test window:", summ(legs(B[B.day>'2026-08-01'])[lambda t:t.leg=='agree']).round(2).to_dict())
