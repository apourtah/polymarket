"""Walk-forward comparison of next-day HRRR error correction per station, plus bucket probabilities.
Methods: raw | rolling5 | EWMA (gain tuned on prior data) | pooled MOS ridge (HRRR fields + lags + city) |
pooled shallow GBM | analogs (k nearest past days in feature space) | NBM station TMX | GFS MOS n_x |
blend (NBM + corrected HRRR). Then P(bucket) = corrected max + empirical residual dist -> trade where P - price >= margin.
Evaluation from Jul 15, six cities (SF excluded), $50 clips at 21:00 evening before."""
import pandas as pd, numpy as np, json, re, bisect, datetime as dt, warnings
from zoneinfo import ZoneInfo
from decimal import Decimal, ROUND_HALF_UP
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sweep_local import TZ, CITY_RE
warnings.filterwarnings('ignore')
rh=lambda x: int(Decimal(str(x)).quantize(Decimal('1'),rounding=ROUND_HALF_UP)); b2=lambda v: rh(v)-rh(v)%2
CITIES=['Atlanta','Austin','Houston','Los Angeles','Miami','Seattle']; site=json.load(open('out/stations.json')); STAKE=50
# ---------- data ----------
h=pd.read_parquet('data/hrrr_t2m.parquet'); h=h[h.city!='ERR'].copy(); h['valid_utc']=pd.to_datetime(h.valid_utc,utc=True)
h['local']=[t.tz_convert(TZ[ct]) for t,ct in zip(h.valid_utc,h.city)]; h['mday']=[x.date() for x in h.local]; h['lh']=[x.hour for x in h.local]
hm=h[h.lh.between(11,18)].groupby(['city','mday','run']).t2m_f.max().unstack('run').rename(columns={0:'hrrr00',12:'hrrr12'})
met=pd.read_parquet('data/metar.parquet'); met['day']=met.local_time.dt.date; met['tf']=(met.temp_c*9/5+32).map(rh)
dmax=met.groupby(['station','day']).tf.max(); dmin=met.groupby(['station','day']).tf.min()
w=pd.read_parquet('data/hrrr_wx_d1.parquet'); w['mday']=w.time.dt.date; w['lh']=w.time.dt.hour; aft=w[w.lh.between(11,18)]
wx=aft.groupby(['city','mday']).agg(dew=('dew_point_2m','mean'),rhum=('relative_humidity_2m','mean'),wind=('wind_speed_10m','mean'),cloud=('cloud_cover','mean'),rad=('shortwave_radiation','mean'),pres=('surface_pressure','mean'),precip=('precipitation','sum'),
    wdir=('wind_direction_10m',lambda s: np.degrees(np.arctan2(np.sin(np.radians(s)).mean(),np.cos(np.radians(s)).mean()))%360))
morn=w[w.lh.between(4,8)].groupby(['city','mday']).agg(t_morn=('temperature_2m','mean'),cloud_morn=('cloud_cover','mean'))
mos=pd.read_parquet('data/mos_nbm.parquet'); mos['runtime']=pd.to_datetime(mos.runtime,utc=True); mos['ftime']=pd.to_datetime(mos.ftime,utc=True)
def mos_daily(model,run_hour):
    """Daily afternoon max from the latest run of `model` issued before 21:00 local the evening before target day."""
    d=mos[(mos.model==model)&(mos.runtime.dt.hour==run_hour)].copy(); out=[]
    for c,g in d.groupby('city'):
        tz=ZoneInfo(TZ[c]); g=g.copy(); lt=g.ftime.dt.tz_convert(tz); g['md']=lt.dt.date; g['lh']=lt.dt.hour
        g=g[g.lh.between(11,18)]
        for md,gg in g.groupby('md'):
            cutoff=pd.Timestamp(dt.datetime(md.year,md.month,md.day,21,tzinfo=tz)-dt.timedelta(days=1))
            gg=gg[gg.runtime<=cutoff]
            if gg.empty: continue
            last=gg[gg.runtime==gg.runtime.max()]
            rec=dict(city=c,mday=md,tmax=pd.to_numeric(last.tmp,errors='coerce').max())
            if model=='NBS': rec['tsd']=pd.to_numeric(last.tsd,errors='coerce').mean(); rec['txn']=pd.to_numeric(last.txn,errors='coerce').max()
            if model in ('GFS','NAM'): rec['n_x']=pd.to_numeric(last.n_x,errors='coerce').max()
            out.append(rec)
    return pd.DataFrame(out).set_index(['city','mday'])
nbs=mos_daily('NBS',0).rename(columns={'tmax':'nbm_max','tsd':'nbm_sd','txn':'nbm_txn'}); gfs=mos_daily('GFS',0).rename(columns={'tmax':'gfsmos_max','n_x':'gfsmos_nx'})
# ---------- panel ----------
days=sorted({d for (_,d) in hm.index}); rows=[]
for c in CITIES:
    st=site[c][0]
    for d in days:
        if (c,d) not in hm.index or pd.isna(hm.loc[(c,d),'hrrr00']): continue
        act=dmax.get((st,d),np.nan)
        if pd.isna(act): continue
        r=dict(city=c,mday=d,hrrr=hm.loc[(c,d),'hrrr00'],actual=act,err=act-hm.loc[(c,d),'hrrr00'],yday_max=dmax.get((st,d-dt.timedelta(days=1)),np.nan),yday_min=dmin.get((st,d-dt.timedelta(days=1)),np.nan),doy=d.timetuple().tm_yday)
        for src in (wx,morn,nbs,gfs):
            if (c,d) in src.index: r.update(src.loc[(c,d)].to_dict())
        rows.append(r)
P=pd.DataFrame(rows).sort_values(['city','mday']).reset_index(drop=True)
P['yday_err']=P.groupby('city').err.shift(1); P['e2']=P.groupby('city').err.shift(2); P['e3']=P.groupby('city').err.shift(3)
P['r5']=P.groupby('city').err.transform(lambda s: s.shift(1).rolling(5,min_periods=3).mean()); P['r14']=P.groupby('city').err.transform(lambda s: s.shift(1).rolling(14,min_periods=5).mean())
P['dpd']=P.hrrr-P.dew; P['wdir_s']=np.sin(np.radians(P.wdir)); P['wdir_c']=np.cos(np.radians(P.wdir)); P['nbm_minus_hrrr']=P.nbm_max-P.hrrr; P['gfs_minus_hrrr']=P.gfsmos_max-P.hrrr
FEATS=['hrrr','dew','rhum','wind','cloud','rad','pres','precip','wdir_s','wdir_c','dpd','t_morn','cloud_morn','yday_max','yday_min','yday_err','e2','r5','r14','nbm_minus_hrrr','gfs_minus_hrrr','doy']
P.to_parquet('out/correction_panel.parquet'); print(f"panel: {len(P)} city-days, NBM coverage {P.nbm_max.notna().mean():.0%}, GFS-MOS coverage {P.gfsmos_max.notna().mean():.0%}")
# ---------- walk-forward ----------
TEST_START=dt.date(2026,7,15); preds=[]
citycode={c:i for i,c in enumerate(CITIES)}
for d in sorted(P[P.mday>=TEST_START].mday.unique()):
    tr=P[P.mday<d].dropna(subset=['err']); te=P[P.mday==d]
    if len(tr)<60: continue
    # EWMA gain tuned per city on training errors
    def ewma_pred(c):
        e=tr[tr.city==c].err.values
        if len(e)<10: return 0.0
        best=None
        for k in [0.05,0.1,0.2,0.3,0.5,0.7]:
            b=0.0; se=0.0
            for i in range(len(e)):
                se+=(e[i]-b)**2; b=b+k*(e[i]-b)
            if best is None or se<best[0]: best=(se,k,b)
        return best[2]
    ew={c:ewma_pred(c) for c in CITIES}
    # pooled ridge / GBM / analogs on features (impute with training medians)
    Xtr=tr[FEATS].copy(); med=Xtr.median(); Xtr=Xtr.fillna(med); Xtr['city']=tr.city.map(citycode)
    Xte=te[FEATS].fillna(med); Xte['city']=te.city.map(citycode)
    Xd_tr=pd.get_dummies(Xtr.city,prefix='c').reindex(columns=[f'c_{i}' for i in range(6)],fill_value=0); Xd_te=pd.get_dummies(Xte.city,prefix='c').reindex(columns=[f'c_{i}' for i in range(6)],fill_value=0)
    mu=Xtr[FEATS].mean(); sd=Xtr[FEATS].std().replace(0,1)
    Ztr=np.c_[((Xtr[FEATS]-mu)/sd).values,Xd_tr.values]; Zte=np.c_[((Xte[FEATS]-mu)/sd).values,Xd_te.values]
    ridge=Ridge(alpha=20.0).fit(Ztr,tr.err.values); p_ridge=ridge.predict(Zte)
    gbm=HistGradientBoostingRegressor(max_depth=2,max_iter=60,learning_rate=0.05,min_samples_leaf=15,l2_regularization=1.0,random_state=0).fit(np.c_[Xtr[FEATS].values,Xtr.city.values],tr.err.values); p_gbm=gbm.predict(np.c_[Xte[FEATS].values,Xte.city.values])
    AF=['hrrr','dew','wind','cloud','rad','wdir_s','wdir_c','yday_max','t_morn']
    for i,(idx,r) in enumerate(te.iterrows()):
        c=r.city; trc=tr[tr.city==c]
        # analogs: 12 nearest training days at this station
        A=((trc[AF].fillna(med[AF])-mu[AF])/sd[AF]).values; q=((Xte.loc[idx,AF]-mu[AF])/sd[AF]).values.astype(float)
        dist=np.sqrt(((A-q)**2).sum(1)); nn=np.argsort(dist)[:12]; p_an=trc.err.values[nn].mean() if len(nn) else 0.0
        resid_sd=trc.err.std()
        preds.append(dict(city=c,mday=d,hrrr=r.hrrr,actual=r.actual,win_lo=r.actual-r.actual%2,err=r.err,
            raw=0.0,r5=r.r5 if pd.notna(r.r5) else 0.0,ewma=ew[c],ridge=p_ridge[i],gbm=p_gbm[i],analog=p_an,
            nbm=(r.nbm_max-r.hrrr) if pd.notna(r.nbm_max) else np.nan,nbm_txn=(r.nbm_txn-r.hrrr) if pd.notna(r.nbm_txn) else np.nan,gfsmos=(r.gfsmos_max-r.hrrr) if pd.notna(r.gfsmos_max) else np.nan,
            resid_sd=resid_sd,nbm_sd=r.nbm_sd if pd.notna(r.nbm_sd) else np.nan))
Q=pd.DataFrame(preds); Q['blend']=Q[['ewma','ridge','nbm']].mean(axis=1); Q['blend2']=Q[['ewma','nbm']].mean(axis=1)
METHODS=['raw','r5','ewma','ridge','gbm','analog','nbm','nbm_txn','gfsmos','blend2','blend']
pd.set_option('display.width',240)
print("\n=== point accuracy of the corrected max (MAE °F), walk-forward Jul 15–Sep 16")
mae=pd.DataFrame({m:(Q.err-Q[m]).abs().groupby(Q.city).mean() for m in METHODS}).round(2); mae.loc['ALL']=[(Q.err-Q[m]).abs().mean() for m in METHODS]; print(mae.round(2).to_string())
print("\nhit rate (corrected bucket == winner):"); hit=pd.DataFrame({m:((Q.hrrr+Q[m]-0.5).map(lambda v: b2(v) if pd.notna(v) else np.nan)==Q.win_lo).groupby(Q.city).mean() for m in METHODS}); hit.loc['ALL']=[((Q.hrrr+Q[m]-0.5).map(lambda v: b2(v) if pd.notna(v) else np.nan)==Q.win_lo).mean() for m in METHODS]; print(hit.round(3).to_string())
Q.to_parquet('out/correction_preds.parquet')
