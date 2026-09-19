"""Day-before forecast backtest: forecasts that exist during the DAY BEFORE (12Z / 18Z NBM and GFS-MOS from IEM; 12Z/18Z HRRR
once data/hrrr_t2m_daybefore.parquet is filled), EWMA bias-corrected walk-forward per city, priced against the bucket mids at
15:00 and 20:00 local the day before. Sweeps 'super cheap' (bucket price <= c) and 'super convinced' (P >= p) YES buys, plus
NO on the favorite / runner-up at those hours.  env: CITIES, START, END"""
import pandas as pd, numpy as np, json, re, bisect, datetime as dt, os
from zoneinfo import ZoneInfo
from scipy.stats import norm
from sweep_local import TZ
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
CITIES=os.environ.get('CITIES','Los Angeles,Austin,Chicago,Houston,Dallas,Seattle,Miami').split(',')
START=pd.Timestamp(os.environ.get('START','2026-01-18')); END=pd.Timestamp(os.environ.get('END','2026-09-15')); STAKE=50.0; SLIP=0.01
P=pd.read_parquet('out/backtest_panel.parquet'); P['mday']=pd.to_datetime(P.mday); act=P.set_index(['city','mday']).actual.to_dict()
# ---- day-before model maxes ----
mos=pd.read_parquet('data/mos_nbm.parquet'); mos['runtime']=pd.to_datetime(mos.runtime,utc=True); mos['ftime']=pd.to_datetime(mos.ftime,utc=True); mos=mos[mos.city.isin(CITIES)]
def mos_max(model, run_hours):
    d=mos[(mos.model==model)&(mos.runtime.dt.hour.isin(run_hours))].copy(); out=[]
    for c,g in d.groupby('city'):
        tz=ZoneInfo(TZ[c]); lt=g.ftime.dt.tz_convert(tz); g=g.assign(md=lt.dt.date,lh=lt.dt.hour); g=g[g.lh.between(11,18)]
        g['rday']=g.runtime.dt.tz_convert(tz).dt.date
        g=g[g.md==g.rday+pd.Timedelta(days=1)]           # run made the day before the target
        for md,gg in g.groupby('md'):
            last=gg[gg.runtime==gg.runtime.max()]; out.append(dict(city=c,mday=pd.Timestamp(md),fmax=pd.to_numeric(last.tmp,errors='coerce').max()))
    return pd.DataFrame(out).set_index(['city','mday']).fmax
src={'nbm12':mos_max('NBS',[12,13]),'nbm18':mos_max('NBS',[18,19]),'gfs12':mos_max('GFS',[12]),'gfs18':mos_max('GFS',[18])}
if os.path.exists('data/hrrr_t2m_daybefore.parquet'):
    h=pd.read_parquet('data/hrrr_t2m_daybefore.parquet'); h=h[h.city!='ERR'].drop_duplicates(['date','run','fxx','city']); h['valid_utc']=pd.to_datetime(h.valid_utc,utc=True)
    h['local']=[t.tz_convert(TZ[c]) for t,c in zip(h.valid_utc,h.city)]; h['mday']=pd.to_datetime([x.date() for x in h.local]); h['lh']=[x.hour for x in h.local]; h=h[h.lh.between(11,18)]
    for run in (12,18):
        g=h[h.run==run].groupby(['city','mday']); cnt=g.lh.nunique(); mx=g.t2m_f.max(); src[f'hrrr{run}']=mx[cnt>=6]
    print("HRRR day-before coverage:", {k:len(v) for k,v in src.items() if k.startswith('hrrr')})
D=pd.DataFrame(src); D=D.reset_index(); D['actual']=[act.get((c,d)) for c,d in zip(D.city,D.mday)]; D=D.dropna(subset=['actual']).sort_values(['city','mday'])
print("day-before forecasts:", {k:int(D[k].notna().sum()) for k in src}, "\nraw MAE:", {k:round((D[k]-D.actual).abs().mean(),2) for k in src})
# ---- walk-forward EWMA bias per city per model (same gains as the bot) ----
GAINS=[0.05,0.1,0.2,0.3,0.5,0.7]
for k in src:
    D[k+'_c']=np.nan; D[k+'_sd']=np.nan
    for c,g in D.groupby('city'):
        g=g.dropna(subset=[k]); e=(g.actual-g[k]).values; idx=g.index
        for i in range(len(e)):
            if i<30: continue
            best=None
            for gain in GAINS:
                b=0.0; se=[]
                for x in e[:i]: se.append(x-b); b+=gain*(x-b)
                s=float(np.sum(np.square(se)))
                if best is None or s<best[0]: best=(s,b,np.std(se[-60:]))
            D.loc[idx[i],k+'_c']=g[k].iloc[i]+best[1]; D.loc[idx[i],k+'_sd']=max(best[2],1.0)
print("corrected MAE:", {k:round((D[k+'_c']-D.actual).abs().mean(),2) for k in src})
# ---- markets & prices ----
def bucket(q):
    mm=re.search(r'between (-?\d+)-(-?\d+)|(-?\d+)°[FC] or below|(-?\d+)°[FC] or (?:above|higher)|(-?\d+)°[FC] on',q)
    if mm.group(1): return (int(mm.group(1)),int(mm.group(2)))
    if mm.group(3): return (-999,int(mm.group(3)))
    if mm.group(4): return (int(mm.group(4)),999)
    return (int(mm.group(5)),int(mm.group(5)))
ms=[m for m in json.load(open('data/markets.json')) if re.search(rf'highest temperature in ({"|".join(CITIES)}) be ', m['question'] or '') and m.get('closed')]
mk={}
for m in ms:
    city=re.search(r'temperature in (.+?) be ',m['question']).group(1); d=pd.Timestamp(m['end_date'][:10]); mk.setdefault((city,d),{})[bucket(m['question'])]=m
cache={}
def yes_price(m, ts, stale=3*3600):
    mid=m['market_id']
    if mid not in cache:
        try: hh=json.load(open(f"data/prices/{mid}.json")); cache[mid]=([x['t'] for x in hh['history']],[x['p'] if hh['losing_outcome']=='Yes' else 1-x['p'] for x in hh['history']])
        except FileNotFoundError: cache[mid]=None
    s=cache[mid]
    if not s or not s[0]: return None
    i=bisect.bisect_right(s[0],ts)-1
    if i<0 or ts-s[0][i]>stale: return None
    return s[1][i]
def probs(mu,sd,buckets): return {b: norm.cdf((min(b[1],200)+0.5-mu)/sd)-norm.cdf((max(b[0],-200)-0.5-mu)/sd) for b in buckets}
rows=[]
for r in D[(D.mday>=START)&(D.mday<=END)].itertuples():
    bk=mk.get((r.city,r.mday))
    if not bk: continue
    tz=ZoneInfo(TZ[r.city]); d=r.mday.date()
    for hour,models in [(15,['nbm12','gfs12','hrrr12']),(20,['nbm18','gfs18','hrrr18']),(21.6,['nbm18','gfs18','hrrr18'])]:
        ts=int(dt.datetime(d.year,d.month,d.day,int(hour),int((hour%1)*60),tzinfo=tz).timestamp())-86400
        pr={b:yes_price(m,ts) for b,m in bk.items()}; pr={b:v for b,v in pr.items() if v is not None}
        if len(pr)<3: continue
        fav=max(pr,key=pr.get); ranked=sorted(pr,key=pr.get,reverse=True)
        for k in models:
            mu=getattr(r,k+'_c',np.nan); sd=getattr(r,k+'_sd',np.nan)
            if pd.isna(mu): continue
            Pb=probs(mu,sd,list(pr)); best=max(Pb,key=Pb.get)
            for b in pr:
                rows.append(dict(city=r.city,mday=r.mday,hour=hour,model=k,lo=b[0],price=pr[b],P=Pb[b],is_best=(b==best),rank=ranked.index(b)+1,fav_p=pr[fav],
                                 won=[float(x) for x in bk[b]['outcome_prices']]==[1.0,0.0],mu=mu,sd=sd))
X=pd.DataFrame(rows); X['month']=X.mday.dt.to_period('M').astype(str); X.to_parquet('out/daybefore_buckets.parquet')
def pnl_yes(x): ask=(x.price+SLIP).clip(0.01,0.99); sh=STAKE/ask; fee=0.05*ask*(1-ask)*sh; return np.where(x.won,sh-STAKE,-STAKE)-fee
def pnl_no(x): ask=(1-x.price+SLIP).clip(0.01,0.99); sh=STAKE/ask; fee=0.05*ask*(1-ask)*sh; return np.where(~x.won,sh-STAKE,-STAKE)-fee
def summ(x,side='yes'):
    if len(x)==0: return pd.Series(dict(n=0,win=np.nan,px=np.nan,pnl=0,roi=np.nan,neg_m=np.nan,worst_m=np.nan,top3_share=np.nan))
    p=pd.Series(pnl_yes(x) if side=='yes' else pnl_no(x),index=x.index); m=p.groupby(x.month).sum(); w=x.won if side=='yes' else ~x.won
    return pd.Series(dict(n=len(x),win=w.mean(),px=(x.price if side=='yes' else 1-x.price).mean(),pnl=p.sum(),roi=p.sum()/(STAKE*len(x)),neg_m=(m<0).sum(),worst_m=m.min(),top3_share=p.nlargest(3).sum()/p.sum() if p.sum()>0 else np.nan))
print(f"\n{X.groupby(['city','mday']).ngroups} city-days with day-before prices")
print("\n### model bucket hit rate (best bucket = winner) by model x hour, and the market favorite at that hour")
bb=X[X.is_best]; print(bb.groupby(['hour','model']).won.agg(['mean','size']).round(2).T.to_string()); print("favorite wins:", X[X['rank']==1].drop_duplicates(['city','mday','hour']).groupby('hour').won.mean().round(2).to_dict())
print("\n### A. 'super cheap': buy the model's best bucket if price <= c   (YES, $50, +1c, fee)")
for hour in [15,20]:
    for k in sorted(X.model.unique()):
        Y=X[(X.hour==hour)&(X.model==k)&X.is_best]
        if len(Y)==0: continue
        print(f"\n-- {hour}:00 local day-before, {k}: best-bucket price distribution:", Y.price.describe()[['25%','50%','75%']].round(2).to_dict())
        print(pd.DataFrame({f"px<={c:.2f}":summ(Y[Y.price.between(0.05,c)]) for c in [0.10,0.15,0.20,0.25,0.30,0.40,0.60]}).T.round(2).to_string())
print("\n### B. 'super convinced': buy the best bucket if P >= p (price 0.05-0.60), and P - price >= e")
for hour in [15,20]:
    for k in sorted(X.model.unique()):
        Y=X[(X.hour==hour)&(X.model==k)&X.is_best&X.price.between(0.05,0.60)]
        if len(Y)==0: continue
        print(f"\n-- {hour}:00, {k}:"); print(pd.DataFrame({**{f"P>={p:.2f}":summ(Y[Y.P>=p]) for p in [0.35,0.40,0.45,0.50,0.55]},**{f"P-px>={e:.2f}":summ(Y[Y.P-Y.price>=e]) for e in [0.10,0.15,0.20,0.25]}}).T.round(2).to_string())
print("\n### C. any bucket (not only the best) with P - price >= e, price 0.05-0.40")
for hour in [15,20]:
    for k in sorted(X.model.unique()):
        Y=X[(X.hour==hour)&(X.model==k)&X.price.between(0.05,0.40)]
        if len(Y)==0: continue
        print(f"-- {hour}:00, {k}: "+"  ".join(f"e>={e:.2f}: n={len(Y[Y.P-Y.price>=e])} roi={summ(Y[Y.P-Y.price>=e]).roi:+.2f} negm={summ(Y[Y.P-Y.price>=e]).neg_m:.0f}" for e in [0.10,0.15,0.20,0.30]))
print("\n### D. NO legs at day-before hours (model = nbm at that hour): favorite / runner-up")
for hour,k in [(15,'nbm12'),(20,'nbm18')]:
    Z=X[(X.hour==hour)&(X.model==k)]; fav=Z[Z['rank']==1]; ru=Z[Z['rank']==2]
    print(f"-- {hour}:00 NO on favorite by fav price:"); print(fav.groupby(pd.cut(fav.price,[0,.3,.4,.5,.6,.7,1]),observed=True).apply(lambda x: summ(x,'no')).round(2).to_string())
    print(f"   NO on favorite when it is not the model's bucket, fav 0.30-0.60:", summ(fav[~fav.is_best&fav.price.between(.3,.6)],'no').round(2).to_dict())
    print(f"-- {hour}:00 NO on runner-up by runner-up price:"); print(ru.groupby(pd.cut(ru.price,[0,.15,.2,.25,.3,.35,.4,.5]),observed=True).apply(lambda x: summ(x,'no')).round(2).to_string())
    print(f"   runner-up not the model's bucket, 0.20-0.35:", summ(ru[~ru.is_best&ru.price.between(.2,.35)],'no').round(2).to_dict())
