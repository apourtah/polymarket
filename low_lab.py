"""Walk-forward correction lab for 'lowest temperature' markets (8 US cities): 00Z HRRR min over the local day,
EWMA per city + pooled ridge (night fields, lags, evening obs error, NBM/GFS-MOS min), bucket probabilities,
agree/disagree rule, $50 YES clips at 21:00 local the evening before. Mirrors correction_lab.py / bot rule."""
import pandas as pd, numpy as np, json, re, datetime as dt, warnings, sys
from zoneinfo import ZoneInfo
from decimal import Decimal, ROUND_HALF_UP
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sweep_local import TZ
warnings.filterwarnings('ignore'); pd.set_option('display.width',240)
rh=lambda x: int(Decimal(str(x)).quantize(Decimal('1'),rounding=ROUND_HALF_UP)); b2=lambda v: rh(v)-rh(v)%2
CITIES=['New York City','Miami','Atlanta','Austin','Houston','Los Angeles','Seattle','San Francisco']; site=json.load(open('out/stations.json')); STAKE=50
TEST_START=dt.date(2026,7,15); EDGE=float(sys.argv[1]) if len(sys.argv)>1 else 0.10
# ---------- HRRR 00Z run, full local day ----------
def load_hrrr():
    a=pd.read_parquet('data/hrrr_t2m_night.parquet'); b=pd.read_parquet('data/hrrr_t2m.parquet'); x=pd.concat([a,b]); x=x[(x.city!='ERR')&(x.run==0)]
    return x.drop_duplicates(['date','run','fxx','city']).copy()
h=load_hrrr(); h['valid_utc']=pd.to_datetime(h.valid_utc,utc=True)
h['local']=[t.tz_convert(TZ[ct]) for t,ct in zip(h.valid_utc,h.city)]; h['mday']=[x.date() for x in h.local]; h['lh']=[x.hour for x in h.local]; h['rday']=pd.to_datetime(h.date).dt.date
h=h[h.mday==h.rday]                                                     # the run date's local calendar day
cnt=h.groupby(['city','mday']).lh.nunique(); full=cnt[cnt>=14].index      # need (almost) all local hours (afternoon hours may be missing: irrelevant for the min)
hmin=h.groupby(['city','mday']).t2m_f.min(); hmin=hmin[hmin.index.isin(full)].rename('hrrr')
h_eve=h[h.lh==21].groupby(['city','mday']).t2m_f.first().rename('hrrr_eve')       # run's 21:00-local value the evening before? no: 21:00 of the target day
# evening-before HRRR value: f01/f04 (21:00 local of run date - 1)... 00Z run is issued 20:00 EDT / 17:00 PDT: value at 21:00 local previous evening = f01 (EDT) .. f04 (PDT)
hp=load_hrrr(); hp['valid_utc']=pd.to_datetime(hp.valid_utc,utc=True)
hp['local']=[t.tz_convert(TZ[ct]) for t,ct in zip(hp.valid_utc,hp.city)]; hp['rday']=pd.to_datetime(hp.date).dt.date
hp['lh']=[x.hour for x in hp.local]; hp['ld']=[x.date() for x in hp.local]
eve=hp[(hp.lh==21)&(hp.ld==hp.rday-dt.timedelta(days=1))].groupby(['city','rday']).t2m_f.first()      # NOTE: for EDT cities 21:00 local prev day = 01Z of the run date = f01 -> ld==rday-1 wrong? 01Z run-date == 21:00 EDT of (rday-1). ok.
eve.index.names=['city','mday']; eve=eve.rename('hrrr_eve')
# ---------- METAR ----------
met=pd.read_parquet('data/metar.parquet'); met['day']=met.local_time.dt.date; met['tf']=(met.temp_c*9/5+32).map(rh); met['lh']=met.local_time.dt.hour
dmax=met.groupby(['station','day']).tf.max(); dmin=met.groupby(['station','day']).tf.min()
obs20=met[met.lh==20].groupby(['station','day']).tf.last()          # ~20:53 local obs, known at 21:00
# ---------- Open-Meteo HRRR d1 night fields ----------
w=pd.read_parquet('data/hrrr_wx_d1.parquet'); w['mday']=w.time.dt.date; w['lh']=w.time.dt.hour; night=w[w.lh.between(0,8)]
wx=night.groupby(['city','mday']).agg(dew=('dew_point_2m','mean'),rhum=('relative_humidity_2m','mean'),wind=('wind_speed_10m','mean'),cloud=('cloud_cover','mean'),pres=('surface_pressure','mean'),precip=('precipitation','sum'),
    wdir=('wind_direction_10m',lambda s: np.degrees(np.arctan2(np.sin(np.radians(s)).mean(),np.cos(np.radians(s)).mean()))%360),om_min=('temperature_2m','min'))
aft=w[w.lh.between(11,18)].groupby(['city','mday']).agg(cloud_aft=('cloud_cover','mean'),rad=('shortwave_radiation','mean'))
# ---------- MOS night minima (NBS txn / GFS n_x at the 12Z-ish ftime), latest run before 21:00 local ----------
mos=pd.read_parquet('data/mos_nbm.parquet'); mos['runtime']=pd.to_datetime(mos.runtime,utc=True); mos['ftime']=pd.to_datetime(mos.ftime,utc=True)
def mos_min(model):
    d=mos[(mos.model==model)&(mos.runtime.dt.hour==0)].copy(); out=[]
    col='txn' if model=='NBS' else 'n_x'
    for c,g in d.groupby('city'):
        tz=ZoneInfo(TZ[c]); g=g.copy(); lt=g.ftime.dt.tz_convert(tz); g['md']=lt.dt.date; g['lh']=lt.dt.hour
        g=g[g.lh.between(0,12)]                      # the night-min value is attached to the morning ftimes
        for md,gg in g.groupby('md'):
            cutoff=pd.Timestamp(dt.datetime(md.year,md.month,md.day,21,tzinfo=tz)-dt.timedelta(days=1)); gg=gg[gg.runtime<=cutoff]
            if gg.empty: continue
            gg=gg[pd.to_numeric(gg[col],errors='coerce').notna()]
            if gg.empty: continue
            last=gg[gg.runtime==gg.runtime.max()]; v=pd.to_numeric(last[col],errors='coerce').min()
            rec={'city':c,'mday':md,f'{model.lower()}_min':v}
            if model=='NBS': rec['nbm_sd']=pd.to_numeric(last.tsd,errors='coerce').mean()
            out.append(rec)
    return pd.DataFrame(out).set_index(['city','mday'])
nbs=mos_min('NBS'); gfs=mos_min('GFS')
# ---------- panel ----------
rows=[]
for c in CITIES:
    st=site[c][0]
    for (cc,d) in hmin.index:
        if cc!=c: continue
        act=dmin.get((st,d),np.nan)
        if pd.isna(act): continue
        y=d-dt.timedelta(days=1)
        r=dict(city=c,mday=d,hrrr=hmin[(c,d)],actual=act,err=act-hmin[(c,d)],yday_max=dmax.get((st,y),np.nan),yday_min=dmin.get((st,y),np.nan),doy=d.timetuple().tm_yday,
               obs20=obs20.get((st,y),np.nan),hrrr_eve=eve.get((c,d),np.nan))
        for src in (wx,aft,nbs,gfs):
            if (c,d) in src.index: r.update(src.loc[(c,d)].to_dict())
        rows.append(r)
P=pd.DataFrame(rows).sort_values(['city','mday']).reset_index(drop=True)
P['cur_err']=P.obs20-P.hrrr_eve                    # how far off the run already is at 21:00 local
P['drop_fc']=P.hrrr_eve-P.hrrr                     # forecast cooling from 21:00 to the min
P['yday_err']=P.groupby('city').err.shift(1); P['e2']=P.groupby('city').err.shift(2); P['e3']=P.groupby('city').err.shift(3)
P['r5']=P.groupby('city').err.transform(lambda s: s.shift(1).rolling(5,min_periods=3).mean()); P['r14']=P.groupby('city').err.transform(lambda s: s.shift(1).rolling(14,min_periods=5).mean())
P['dpd']=P.hrrr-P.dew; P['wdir_s']=np.sin(np.radians(P.wdir)); P['wdir_c']=np.cos(np.radians(P.wdir)); P['nbm_minus_hrrr']=P.nbs_min-P.hrrr; P['gfs_minus_hrrr']=P.gfs_min-P.hrrr; P['om_minus_hrrr']=P.om_min-P.hrrr
FEATS=['hrrr','dew','rhum','wind','cloud','pres','precip','wdir_s','wdir_c','dpd','cloud_aft','rad','yday_max','yday_min','yday_err','e2','r5','r14','nbm_minus_hrrr','gfs_minus_hrrr','om_minus_hrrr','cur_err','drop_fc','doy']
P.to_parquet('out/low_panel.parquet')
print(f"panel: {len(P)} city-days {P.mday.min()}..{P.mday.max()}; NBM cov {P.nbs_min.notna().mean():.0%}, GFS-MOS {P.gfs_min.notna().mean():.0%}, obs20 {P.obs20.notna().mean():.0%}")
print("raw 00Z HRRR min bias (actual-hrrr) / MAE by city:"); print(P.groupby('city').err.agg(n='size',bias='mean',mae=lambda e:e.abs().mean()).round(2).to_string())
# ---------- walk-forward ----------
citycode={c:i for i,c in enumerate(CITIES)}; preds=[]
def ewma_pred(e):
    if len(e)<10: return 0.0
    best=None
    for k in [0.05,0.1,0.2,0.3,0.5,0.7]:
        b=0.0; se=0.0
        for x in e: se+=(x-b)**2; b=b+k*(x-b)
        if best is None or se<best[0]: best=(se,k,b)
    return best[2]
for d in sorted(P[P.mday>=TEST_START].mday.unique()):
    tr=P[P.mday<d].dropna(subset=['err']); te=P[P.mday==d]
    if len(tr)<60: continue
    ew={c:ewma_pred(tr[tr.city==c].err.values) for c in CITIES}
    Xtr=tr[FEATS].copy(); med=Xtr.median(); Xtr=Xtr.fillna(med); Xte=te[FEATS].fillna(med)
    Dtr=pd.get_dummies(tr.city.map(citycode)).reindex(columns=range(len(CITIES)),fill_value=0); Dte=pd.get_dummies(te.city.map(citycode)).reindex(columns=range(len(CITIES)),fill_value=0)
    mu=Xtr.mean(); sd=Xtr.std().replace(0,1)
    ridge=Ridge(alpha=20.0).fit(np.c_[((Xtr-mu)/sd).values,Dtr.values],tr.err.values); pr=ridge.predict(np.c_[((Xte-mu)/sd).values,Dte.values])
    for i,(idx,r) in enumerate(te.iterrows()):
        trc=tr[tr.city==r.city]; res_e=(trc.err-trc.err.ewm(alpha=0.2).mean().shift(1)).dropna()
        preds.append(dict(city=r.city,mday=d,hrrr=r.hrrr,actual=r.actual,err=r.err,raw=0.0,ewma=ew[r.city],ridge=pr[i],sd_ewma=max(res_e.std(),1.0) if len(res_e)>5 else 2.5,sd_ridge=max((tr.err.values-ridge.predict(np.c_[((Xtr-mu)/sd).values,Dtr.values])).std(),1.0)))
Q=pd.DataFrame(preds); Q['win_lo']=Q.actual-Q.actual%2
METHODS=['raw','ewma','ridge']; Q['avg']=(Q.ewma+Q.ridge)/2; METHODS.append('avg')
print(f"\n=== walk-forward from {TEST_START}: MAE of corrected min (°F)")
mae=pd.DataFrame({m:(Q.err-Q[m]).abs().groupby(Q.city).mean() for m in METHODS}); mae.loc['ALL']=[(Q.err-Q[m]).abs().mean() for m in METHODS]; print(mae.round(2).to_string())
print("\nhit rate (corrected 2°F bucket == winner):")
hit=pd.DataFrame({m:((Q.hrrr+Q[m]).map(b2)==Q.win_lo).groupby(Q.city).mean() for m in METHODS}); hit.loc['ALL']=[((Q.hrrr+Q[m]).map(b2)==Q.win_lo).mean() for m in METHODS]; print(hit.round(3).to_string())
Q.to_parquet('out/low_preds.parquet')
# ---------- trading vs 21:00-evening-before prices (NYC/Miami books are formed; others reported separately) ----------
M=pd.read_parquet('out/lowest_market_panel.parquet'); M=M.dropna(subset=['price'])
mk={}
for r in M.itertuples(): mk.setdefault((r.city,r.day),{})[(r.lo,r.hi)]=(r.price,r.won)
trades=[]
for r in Q.itertuples():
    if (r.city,r.mday) not in mk: continue
    bk=mk[(r.city,r.mday)]; ssum=sum(p for p,_ in bk.values())
    pe={b:norm.cdf((min(b[1],200)+0.5-(r.hrrr+r.ewma))/r.sd_ewma)-norm.cdf((max(b[0],-200)-0.5-(r.hrrr+r.ewma))/r.sd_ewma) for b in bk}
    pr_={b:norm.cdf((min(b[1],200)+0.5-(r.hrrr+r.ridge))/r.sd_ridge)-norm.cdf((max(b[0],-200)-0.5-(r.hrrr+r.ridge))/r.sd_ridge) for b in bk}
    be=max(pe,key=pe.get); br=max(pr_,key=pr_.get); agree=be==br
    for b,(price,won) in bk.items():
        P_=(pe[b]+pr_[b])/2; edge=P_-price
        sel=None
        if agree and b==be and price<=0.60: sel='agree'
        elif not agree and edge>=EDGE and price<=0.60: sel='disagree'
        if sel:
            sh=STAKE/price; fee=0.05*price*(1-price)*sh
            trades.append(dict(city=r.city,mday=r.mday,bucket=b,why=sel,price=price,P=P_,edge=edge,won=won,pnl=(sh-STAKE if won else -STAKE)-fee,book_sum=ssum))
T=pd.DataFrame(trades); T.to_parquet('out/low_trades.parquet')
def summ(t): return pd.Series(dict(n=len(t),win=t.won.mean(),avg_px=t.price.mean(),pnl=t.pnl.sum(),roi=t.pnl.sum()/(STAKE*len(t)) if len(t) else np.nan))
print(f"\n=== $50 YES clips at 21:00 local evening before (bot rule: agree->agreed bucket <=60c; disagree->avg-P edge>={EDGE:.2f}), mid prices, taker fee")
for tag,sub in [("NYC+Miami (formed books, sum<=1.15)",T[T.city.isin(['New York City','Miami'])&(T.book_sum<=1.15)]),("other 6 cities (thin, since Aug 21, sum<=1.15 only)",T[~T.city.isin(['New York City','Miami'])&(T.book_sum<=1.15)])]:
    print(f"\n-- {tag}"); 
    if len(sub): print(sub.groupby('why').apply(summ).round(3).to_string()); print(sub.groupby('city').apply(summ).round(3).to_string()); print("ALL:", summ(sub).round(3).to_dict())
